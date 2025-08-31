# experiments/10_scperturb.py
"""
Train on the training split of VCC and on scPerturb data.


"""

import dataclasses
import logging
import os
import pathlib
import pprint
import tomllib
import typing as tp

import anndata as ad
import beartype
import chex
import equinox as eqx
import grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import polars as pl
import tyro
from jaxtyping import Array, Float, Int, jaxtyped

import vcell.nn.optim
import wandb
from vcell import helpers, metrics
from vcell.data import harmonize

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("06")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    h5ad_fpath: pathlib.Path
    """Path to h5ad file."""
    pert_col: str = "target_gene"
    ctrl_label: str = "non-targeting"
    group_by: tuple[str, ...] = ("batch",)
    gene_id_col: str = "ensembl_id"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    seed: int = 42
    """Random seed."""

    vcc: pathlib.Path = pathlib.Path("data/inputs/vcc")
    """Path to vcc challenge data."""

    datasets: list[DatasetConfig] = dataclasses.field(default_factory=list)

    optim: vcell.nn.optim.Config = vcell.nn.optim.Config()
    """Optimizer settings."""

    batch_size: int = 32
    """Batch size."""
    n_train: int = 10_000

    set_size: int = 32

    n_workers: int = 8

    # Logging
    log_every: int = 20
    """how often to log metrics."""
    ckpt_dir: str = os.path.join(".", "checkpoints")
    """where to store temporary/intermediate outputs like the memmap."""
    out_path: pathlib.Path = pathlib.Path("pred_raw.h5ad")
    """final submission-ready H5AD (run cell-eval prep afterwards)."""


@beartype.beartype
class PertVocab:
    """Canonical mapping from target gene (Ensembl) -> integer id, plus OOV."""

    ens2id: dict[str, int]
    id2ens: list[str]
    oov_id: int

    def __init__(self, vocab: harmonize.GeneVocab, cfgs: list[DatasetConfig]):
        symbols = set()
        for cfg in cfgs:
            adata = ad.read_h5ad(str(cfg.h5ad_fpath), backed="r")
            col = adata.obs[cfg.pert_col].astype(str)
            for s in np.unique(col[col != cfg.ctrl_label]):
                if s.startswith("ENSG"):
                    s = harmonize.strip_ens_version(s)
                else:
                    s = harmonize.parse_symbol(s)
                symbols.add(s)

        ens2id = {e: i for i, e in enumerate(ens_list)}
        id2ens = ens_list
        oov_id = len(id2ens)
        return PertVocab(ens2id=ens2id, id2ens=id2ens, oov_id=oov_id)

    def lookup(self, s: str) -> int:
        e = harmonize.strip_ens_version(s) if s.startswith("ENSG") else s
        return self.ens2id.get(e, self.oov_id)


@jaxtyped(typechecker=beartype.beartype)
class Model(eqx.Module):
    """
    Minimal permutation-invariant 'token-mixing' mapper:
      controls (set of transcriptomes) + perturbation ID -> predicted perturbed set.
    """

    in_proj: eqx.nn.Linear
    h_proj: eqx.nn.Linear
    c_proj: eqx.nn.Linear
    p_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    pert_table: Float[Array, "n_perts d"]

    g: int
    d: int
    n_perts: int

    def __init__(self, n_perts: int, g: int, d: int, key: chex.PRNGKey):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        self.in_proj = eqx.nn.Linear(in_features=g, out_features=d, key=k1)
        self.h_proj = eqx.nn.Linear(in_features=d, out_features=d, key=k2)
        self.c_proj = eqx.nn.Linear(in_features=d, out_features=d, key=k3)
        self.p_proj = eqx.nn.Linear(in_features=d, out_features=d, key=k4)
        self.out_proj = eqx.nn.Linear(in_features=d, out_features=g, key=k5)
        self.pert_table = 0.02 * jax.random.normal(k6, (n_perts, d))
        self.g = g
        self.d = d
        self.n_perts = n_perts

    def __call__(
        self, x_sg: Float[Array, "s g"], pert_id: Int[Array, ""]
    ) -> Float[Array, "s g"]:
        x_sd = jax.vmap(self.in_proj)(x_sg)
        h_sd = jax.nn.gelu(x_sd)
        ctx_d = jnp.mean(x_sd, axis=0)
        ctx_d = self.c_proj(ctx_d)
        pert_emb_d = self.pert_table[jnp.asarray(pert_id, dtype=jnp.int32)]
        pert_emb_d = self.p_proj(pert_emb_d)
        s, _ = x_sd.shape
        ctx_sd = jnp.broadcast_to(ctx_d, (s, self.d))
        pert_emb_sd = jnp.broadcast_to(pert_emb_d, (s, self.d))
        m_sd = jax.nn.gelu(jax.vmap(self.h_proj)(h_sd) + ctx_sd + pert_emb_sd)
        delta_sg = jax.vmap(self.out_proj)(m_sd)
        return x_sg + delta_sg


@jaxtyped(typechecker=beartype.beartype)
def loss_and_aux(
    model: eqx.Module,
    ctrls_bsg: Float[Array, "batch set n_genes"],
    perts_b: Int[Array, " batch"],
    tgts_bsg: Float[Array, "batch set n_genes"],
    mask_bg: Int[Array, "batch n_genes"],
) -> tuple[Float[Array, ""], dict]:
    preds_bsg = jax.vmap(model)(ctrls_bsg, perts_b)
    mu_ctrls_bg = ctrls_bsg.mean(axis=1)
    mu_preds_bg = preds_bsg.mean(axis=1)
    mu_tgts_bg = tgts_bsg.mean(axis=1)

    mu_mse = jnp.mean(((mu_preds_bg - mu_tgts_bg) * mask_bg) ** 2)

    effect_pds = metrics.compute_pds(
        mu_preds_bg - mu_ctrls_bg, mu_tgts_bg - mu_ctrls_bg
    )
    pds = metrics.compute_pds(mu_preds_bg, mu_tgts_bg)
    l1 = jnp.mean(jnp.abs(mu_preds_bg - mu_tgts_bg))

    aux = {
        "mu-mse": mu_mse,
        **{f"pds/{k}": v for k, v in pds.items()},
        **{f"effect-pds/{k}": v for k, v in effect_pds.items()},
        "l1": l1,
    }
    return mu_mse, aux


@jaxtyped(typechecker=beartype.beartype)
@eqx.filter_jit(donate="all")
def step_model(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    state: tp.Any,
    ctrls_bsg: Float[Array, "batch set n_genes"],
    perts_b: Int[Array, " batch"],
    tgts_bsg: Float[Array, "batch set n_genes"],
    mask_bg: Int[Array, "batch n_genes"],
) -> tuple[eqx.Module, tp.Any, Float[Array, ""], dict]:
    (loss, metrics), grads = eqx.filter_value_and_grad(loss_and_aux, has_aux=True)(
        model, ctrls_bsg, perts_b, tgts_bsg, mask_bg
    )

    updates, new_state = optim.update(grads, state, model)

    metrics["optim/grad-norm"] = optax.global_norm(grads)
    metrics["optim/update-norm"] = optax.global_norm(updates)

    model = eqx.apply_updates(model, updates)

    return model, new_state, loss, metrics


@beartype.beartype
def make_ctrl_pool(adata, key: chex.PRNGKey, max_controls: int = 20_000):
    """Create a pool of control cell indices for sampling."""
    ctrl_mask = adata.obs["target_gene"].to_numpy() == "non-targeting"
    ctrl_idx_all = np.nonzero(ctrl_mask)[0]
    key, ctrl_key = jax.random.split(key)
    ctrl_pool = jax.random.choice(
        ctrl_key,
        jnp.asarray(ctrl_idx_all),
        shape=(min(len(ctrl_idx_all), max_controls),),
        replace=False,
    )
    return ctrl_pool


@jaxtyped(typechecker=beartype.beartype)
class Sample(tp.TypedDict, total=False):
    cfg: DatasetConfig
    all_rows: Int[np.ndarray, " n"]
    sampled_rows: Int[np.ndarray, " set_size"]
    pert_id: int


@beartype.beartype
class MultiGroupSource(grain.sources.RandomAccessDataSource):
    def __init__(self, cfgs: list[DatasetConfig], set_size: int):
        self.cfgs = cfgs
        self.set_size = set_size

        self._samples: list[Sample] = []
        self._pert2id: dict[str, int] = {}

        for i, cfg in enumerate(cfgs):
            adata = ad.read_h5ad(cfg.h5ad_fpath, backed="r")

            obs = adata.obs

            control_groups = (
                obs[obs[cfg.pert_col] == cfg.ctrl_label]
                .groupby(list(cfg.group_by), observed=True)
                .indices
            )
            target_groups = (
                obs[obs[cfg.pert_col] != cfg.ctrl_label]
                .groupby([cfg.pert_col, *cfg.group_by], observed=True)
                .indices
            )

            for pert, *gb in sorted(target_groups):
                key = tuple(gb) if len(gb) > 1 else gb[0]
                if key not in control_groups:
                    msg = "No observed cells for %s with %s='%s' (control)."
                    logger.info(msg, key, cfg.pert_col, cfg.ctrl_label)
                    continue

                ctrl_rows = control_groups[key]
                pert_rows = target_groups[(pert, *gb)]

                if ctrl_rows.size < set_size:
                    logger.debug(
                        "Skipping %s from %s because only %d control cells (need %d).",
                        ", ".join(f"{k}={v}" for k, v in zip(cfg.group_by, gb)),
                        cfg.h5ad_fpath.stem,
                        ctrl_rows.size,
                        set_size,
                    )
                    continue

                if pert_rows.size < set_size:
                    logger.debug(
                        "Skipping %s from %s because only %d pert cells (need %d).",
                        ", ".join(f"{k}={v}" for k, v in zip(cfg.group_by, gb)),
                        cfg.h5ad_fpath.stem,
                        pert_rows.size,
                        set_size,
                    )
                    continue

                # TODO: Normalize pert_key

                if pert not in self._pert2id:
                    self._pert2id[pert] = len(self._pert2id)
                pert_id = self._pert2id[pert]

                self._samples.append(
                    Sample(
                        cfg=cfg,
                        all_ctrl_rows=ctrl_rows,
                        all_pert_rows=pert_rows,
                        pert_id=pert_id,
                    )
                )

        logger.info("Loaded %s from %d files.", self, len(cfgs))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, k: int) -> Sample:
        return self._samples[k]

    def __repr__(self):
        return f"MultiGroupSource(n={len(self)}, cfgs={len(self.cfgs)}, set={self.set_size})"


@beartype.beartype
class SampleSet(grain.transforms.RandomMap):
    def __init__(self, set_size: int):
        self.set_size = set_size

    def random_map(self, sample: Sample, rng) -> Sample:
        ctrl_i = rng.choice(
            len(sample["all_ctrl_rows"]), size=self.set_size, replace=False
        )
        sample["sampled_ctrl_rows"] = sample["all_ctrl_rows"][ctrl_i]
        pert_i = rng.choice(
            len(sample["all_pert_rows"]), size=self.set_size, replace=False
        )
        sample["sampled_pert_rows"] = sample["all_pert_rows"][pert_i]
        return sample


@beartype.beartype
class LoadAndLift(grain.transforms.Map):
    def __init__(self, vcc_h5ad: str | pathlib.Path):
        self._vcc_h5ad = str(vcc_h5ad)
        self._adatas: dict[DatasetConfig, ad.AnnData] = {}
        self._gmaps: dict[DatasetConfig, harmonize.GeneMap] = {}

        # Lazily initialized objects
        self._logger = None
        self._gene_vocab = None

    @property
    def logger(self):
        if self._logger is None:
            logging.basicConfig(level=logging.DEBUG, format=log_format)
            self._logger = logging.getLogger(f"load-{os.getpid()}")
        return self._logger

    @property
    def gene_vocab(self):
        if self._gene_vocab is None:
            self._gene_vocab = harmonize.GeneVocab(self._vcc_h5ad)
        return self._gene_vocab

    def get_adata(self, cfg: DatasetConfig) -> ad.AnnData:
        if cfg not in self._adatas:
            fpath = str(cfg.h5ad_fpath)
            self._adatas[cfg] = ad.read_h5ad(fpath, backed="r")
            self.logger.info("Opened %s.", fpath)
        return self._adatas[cfg]

    def get_gene_map(self, cfg: DatasetConfig) -> harmonize.GeneMap:
        if cfg not in self._gmaps:
            adata = self.get_adata(cfg)
            self._gmaps[cfg] = self.gene_vocab.make_map(
                adata, gene_id_col=cfg.gene_id_col
            )
        return self._gmaps[cfg]

    def map(self, sample: Sample):
        cfg = sample["cfg"]
        adata = self.get_adata(cfg)
        gmap = self.get_gene_map(cfg)

        # .X can only be indexed in increasing order, so we have to sort. Then we can unsort to preserve the random order again.
        ctrl_sort = np.argsort(sample["sampled_ctrl_rows"])
        pert_sort = np.argsort(sample["sampled_pert_rows"])
        ctrl_unsort = np.argsort(ctrl_sort)
        pert_unsort = np.argsort(pert_sort)

        ctrl_i = sample["sampled_ctrl_rows"][ctrl_sort]
        pert_i = sample["sampled_pert_rows"][pert_sort]

        x_ctrl = adata.X[ctrl_i]
        x_pert = adata.X[pert_i]

        x_ctrl = self.array(x_ctrl)[ctrl_unsort]
        x_pert = self.array(x_pert)[pert_unsort]

        x_ctrl = gmap.lift_to_vcc(x_ctrl)
        x_pert = gmap.lift_to_vcc(x_pert)

        return {
            "control": x_ctrl,
            "target": x_pert,
            "mask": gmap.present_mask,
            "pert_id": sample["pert_id"],
        }

    def array(self, arr):
        if hasattr(arr, "todense"):
            arr = arr.todense()
        return np.asarray(arr)


@beartype.beartype
def make_dataloader(cfg: Config):
    ops = [
        SampleSet(set_size=cfg.set_size),
        LoadAndLift(cfg.vcc / "adata_Training.h5ad"),
        grain.transforms.Batch(batch_size=cfg.batch_size),
    ]
    helpers.check_grain_ops(ops)

    source = MultiGroupSource(cfg.datasets, set_size=cfg.set_size)

    sampler = grain.samplers.IndexSampler(
        num_records=len(source),
        seed=cfg.seed,
        shuffle=True,
        num_epochs=None,  # stream forever
        shard_options=grain.sharding.ShardOptions(shard_index=0, shard_count=1),
    )

    dl = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=ops,
        worker_count=cfg.n_workers,
        worker_buffer_size=2,
        read_options=grain.ReadOptions(num_threads=8, prefetch_buffer_size=500),
    )

    return dl


@beartype.beartype
def main(
    cfg: pathlib.Path | None = None,
    override: tp.Annotated[Config, tyro.conf.arg(name="")] = Config(),
):
    """Run the experiment.

    Args:
        cfg: Path to config file.
        override: CLI options to modify the config file.
    """

    if cfg is None:
        # Use override directly as the config
        cfg = override
    else:
        # Load config from file
        if not cfg.exists():
            raise FileNotFoundError(f"Config file not found: {cfg}")

        with open(cfg, "rb") as fd:
            cfg_dict = tomllib.load(fd)

        # Convert TOML dict to Config dataclass
        loaded_cfg = helpers.dict_to_dataclass(cfg_dict, Config)

        # Find non-default override values
        default_cfg = Config()
        non_default_overrides = helpers.get_non_default_values(override, default_cfg)

        # Merge non-default overrides into loaded config
        cfg = helpers.merge_configs(loaded_cfg, non_default_overrides)

    pprint.pprint(dataclasses.asdict(cfg))

    key = jax.random.key(seed=cfg.seed)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # Load training data (backed) and validation counts
    adata = ad.read_h5ad(cfg.vcc / "adata_Training.h5ad", backed="r")
    genes = np.array(adata.var_names)
    g = genes.size

    ctrl_pool = make_ctrl_pool(adata, key)

    # Init tiny model (OOV=0 row set to 0)
    train_gene_ids = [
        gname for gname in adata.obs["target_gene"].unique() if gname != "non-targeting"
    ]
    key, model_key = jax.random.split(key)
    model = Model(n_perts=1 + len(train_gene_ids), g=g, d=64, key=model_key)
    model = eqx.tree_at(lambda m: m.pert_table, model, model.pert_table.at[0].set(0.0))
    optim = vcell.nn.optim.make(cfg.optim)

    state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    logger.info("Initialized optimizer.")

    dataloader = make_dataloader(cfg)

    # Train
    global_step = 0
    run = wandb.init(
        entity="samuelstevens", project="vcell", config=dataclasses.asdict(cfg)
    )
    for batch in dataloader:
        model, state, loss, metrics = step_model(
            model,
            optim,
            state,
            jnp.array(batch["control"]),
            jnp.array(batch["pert_id"]),
            jnp.array(batch["target"]),
            jnp.array(batch["mask"], dtype=int),
        )
        global_step += 1

        if global_step % cfg.log_every == 0:
            metrics = {k: v.item() for k, v in metrics.items()}
            logger.info("step: %d, loss: %.5f %s", global_step, loss.item(), metrics)
            run.log(
                {"step": global_step, "train/loss": loss.item(), **metrics},
                step=global_step,
            )

        if global_step > cfg.n_train:
            break

    val = pl.read_csv(cfg.vcc / "pert_counts_Validation.csv")
    S = int(val.select(pl.col("n_cells").sum()).item())
    B = 100_000
    C = max(0, B - S)
    N = S + C
    logger.info(
        f"Validation requested cells: {S:,}. Controls to add: {C:,}. Total: {N:,}."
    )

    # Prepare on-disk memmap
    mm_path = os.path.join(cfg.ckpt_dir, "pred_X.float32.mm")
    if os.path.exists(mm_path):
        os.remove(mm_path)
    X_mm = np.memmap(mm_path, mode="w+", dtype=np.float32, shape=(N, g))
    obs_target = np.empty(N, dtype=object)

    # Stream predictions
    off = 0
    logger.info("Generating predictions (streaming to memmap)...")
    for tg, n in helpers.progress(val.select(["target_gene", "n_cells"]).iter_rows()):
        n = int(n)
        key, sample_key = jax.random.split(key)
        take = jax.random.choice(sample_key, ctrl_pool, shape=(n,), replace=True)
        take = np.asarray(take, dtype=np.int64)

        x = adata.X[take].toarray().astype(np.float32)
        x = np.log1p(x)
        y = np.asarray(model(x, pert_id=jnp.array(0, dtype=jnp.int32)))

        X_mm[off : off + n] = y
        obs_target[off : off + n] = tg
        off += n

        del x, y

    logger.info("Finished perturbed predictions.")

    # Add controls to meet the 100k cap
    if C > 0:
        key, ctrl_sample_key = jax.random.split(key)
        ctrl_take = jax.random.choice(
            ctrl_sample_key, jnp.asarray(ctrl_pool), shape=(C,), replace=True
        )
        ctrl_take = np.asarray(ctrl_take, dtype=np.int64)
        ctrl_cells = adata.X[ctrl_take].toarray().astype(np.float32)
        ctrl_cells = np.log1p(ctrl_cells)
        X_mm[off : off + C] = ctrl_cells
        obs_target[off : off + C] = "non-targeting"
        off += C
        del ctrl_cells

    assert off == N, f"Row count mismatch: wrote {off}, expected {N}"

    # Flush memmap and wrap as ndarray view (no copy) so anndata writer accepts it
    X_mm.flush()
    X_arr = np.ndarray(X_mm.shape, dtype=X_mm.dtype, buffer=X_mm)  # plain ndarray view
    assert X_arr.flags["C_CONTIGUOUS"]

    # Build AnnData and write H5AD
    obs_df = pd.DataFrame({"target_gene": pd.Categorical(obs_target)})
    var_df = pd.DataFrame(index=pd.Index(genes, dtype=str))
    pred = ad.AnnData(X=X_arr, obs=obs_df, var=var_df)
    pred.write_h5ad(cfg.out_path, compression="gzip")
    logger.info(f"Wrote predictions to {cfg.out_path} (backed file at {mm_path}).")


if __name__ == "__main__":
    tyro.cli(main)
