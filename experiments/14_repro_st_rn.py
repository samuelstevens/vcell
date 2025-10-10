# experiments/14_repro_st_rn.py
"""
Reproduce the Replogle-Nadig ST model from the STATE paper.
"""

import dataclasses
import itertools
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
import jax.random as jr
import numpy as np
import optax
import pandas as pd
import polars as pl
import tyro
from jaxtyping import Array, Float, Int, jaxtyped

import vcell.nn.optim
import wandb
from vcell import helpers, metrics
from vcell.data import harmonize, vcc

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("14")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    h5ad_fpath: pathlib.Path
    """Path to h5ad file."""
    hvgs_csv: pathlib.Path
    """Path the hvgs.csv file."""
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

    batch_size: int = 256
    """Batch size."""
    n_train: int = 1_000_000

    # For Replogle-Nadig, these are the default hparams
    n_layers: int = 4
    h: int = 128
    n_heads: int = 8
    set_size: int = 32
    n_hvgs: int = 2_000

    n_workers: int = 8

    # Logging
    log_every: int = 20
    """how often to log metrics."""
    ckpt_dir: str = os.path.join(".", "checkpoints")
    """where to store temporary/intermediate outputs like the memmap."""
    out_path: pathlib.Path = pathlib.Path("pred_raw.h5ad")
    """final submission-ready H5AD (run cell-eval prep afterwards)."""


@jaxtyped(typechecker=beartype.beartype)
class Mlp(eqx.Module):
    linears: list[eqx.nn.Linear]
    norms: list[eqx.nn.Linear]
    out: eqx.nn.Linear
    dropout: eqx.nn.Dropout | None
    act: tp.Callable

    def __init__(
        self,
        dims: tp.Sequence[int],
        *,
        dropout_p: float = 0.0,
        ln_eps: float = 1e-5,
        key: chex.PRNGKey,
    ):
        self.linears = []
        self.norms = []

        *dims, d_final = dims

        *keys, key_final = jr.split(key, len(dims))

        for key, (d_in, d_out) in zip(keys, itertools.pairwise(dims)):
            self.linears.append(eqx.nn.Linear(d_in, d_out, key=key))
            self.norms.append(eqx.nn.LayerNorm(d_out))

        self.out = eqx.nn.Linear(d_out, d_final, key=key_final)
        self.dropout = eqx.nn.Dropout(dropout_p)
        self.act = jax.nn.gelu

    def __call__(
        self, x: Float[Array, " d_in"], *, key: chex.PRNGKey
    ) -> Float[Array, " d_out"]:
        keys = jr.split(key, len(self.linears))

        for linear, ln, key in zip(self.linears, self.norms, keys):
            x = linear(x)
            x = ln(x)
            x = self.act(x)
            x = self.dropout(x, key=key)
        return self.out(x)


@jaxtyped(typechecker=beartype.beartype)
class Block(eqx.Module):
    """
    Minimal permutation-invariant 'token-mixing' mapper: controls (set of transcriptomes) + perturbation ID -> predicted perturbed set.
    """

    attn: eqx.nn.MultiheadAttention
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    mlp: eqx.nn.MLP

    def __init__(self, d: int, n_heads: int, ratio: float | int, key: chex.PRNGKey):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=n_heads,
            query_size=d,
            key_size=d,
            value_size=d,
            output_size=d,
            key=k1,
        )
        self.ln1 = eqx.nn.LayerNorm(d)
        self.ln2 = eqx.nn.LayerNorm(d)
        self.mlp = eqx.nn.MLP(
            in_size=d,
            out_size=d,
            width_size=int(ratio * d),
            depth=2,
            activation=jax.nn.gelu,
            key=k2,
        )

    def __call__(self, x_sh: Float[Array, "set h"], *, key: chex.PRNGKey):
        set_size, h = x_sh.shape
        key, *keys = jax.random.split(key, set_size + 1)

        q_sd = k_sd = v_sd = jax.vmap(self.ln1)(x_sh)
        x_sh = x_sh + self.attn(q_sd, k_sd, v_sd, key=key)

        x_sh = x_sh + jax.vmap(self.mlp)(jax.vmap(self.ln2)(x_sh), key=jnp.array(keys))
        return x_sh


@jaxtyped(typechecker=beartype.beartype)
class Transformer(eqx.Module):
    f_cell: Mlp
    emb_pert: eqx.nn.Embedding
    f_pert: Mlp
    blocks: list[Block]
    f_recon: eqx.nn.Linear

    def __init__(
        self,
        *,
        n_genes: int,
        n_perts: int,
        h: int,
        n_layers: int,
        n_heads: int,
        mlp_mult: float | int,
        key: chex.PRNGKey,
    ):
        k1, k2, k3, k4, k5 = jr.split(key, 5)

        # "Perturbation labels are encoded into the same embedding dimension d_h" (State, p22)
        d_pert = h

        self.f_cell = Mlp([n_genes, h, h, h, h], key=k1)
        self.emb_pert = eqx.nn.Embedding(
            num_embeddings=n_perts, embedding_size=d_pert, key=k2
        )
        self.f_pert = Mlp([d_pert, h, h, h, h], key=k3)
        self.blocks = [
            Block(h, n_heads, mlp_mult, key=jr.fold_in(k4, i)) for i in range(n_layers)
        ]
        self.f_recon = eqx.nn.Linear(h, n_genes, key=k5)

    def __call__(
        self,
        x_sg: Float[Array, "set n_genes"],
        pert_id: Int[Array, ""],
        *,
        key: chex.PRNGKey,
    ):
        set_size, n_genes = x_sg.shape
        key, *f_cell_keys = jr.split(key, set_size + 1)
        x_sh = jax.vmap(self.f_cell)(x_sg, key=jnp.array(f_cell_keys))

        key, f_pert_key = jr.split(key)
        p_d = self.f_pert(self.emb_pert(pert_id), key=f_pert_key)
        x_sh = x_sh + jnp.broadcast_to(p_d, x_sh.shape)

        for i, block in enumerate(self.blocks):
            x_sh = block(x_sh, key=jr.fold_in(key, i))

        delta_sg = jax.vmap(self.f_recon)(x_sh)
        return x_sg + delta_sg


@jaxtyped(typechecker=beartype.beartype)
def loss_and_aux(
    model: eqx.Module,
    ctrls_bsg: Float[Array, "batch set n_genes"],
    perts_b: Int[Array, " batch"],
    tgts_bsg: Float[Array, "batch set n_genes"],
    key: chex.PRNGKey,
) -> tuple[Float[Array, ""], dict]:
    keys_b = jr.split(key, len(perts_b))
    preds_bsg = jax.vmap(model)(ctrls_bsg, perts_b, key=keys_b)
    mu_ctrls_bg = ctrls_bsg.mean(axis=1)
    mu_preds_bg = preds_bsg.mean(axis=1)
    mu_tgts_bg = tgts_bsg.mean(axis=1)

    mu_mse = jnp.mean((mu_preds_bg - mu_tgts_bg) ** 2)

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


@eqx.filter_jit(donate="all")
@jaxtyped(typechecker=beartype.beartype)
def step_model(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    state: tp.Any,
    ctrls_bsg: Float[Array, "batch set n_genes"],
    perts_b: Int[Array, " batch"],
    tgts_bsg: Float[Array, "batch set n_genes"],
    key: chex.PRNGKey,
) -> tuple[eqx.Module, tp.Any, Float[Array, ""], dict]:
    (loss, metrics), grads = eqx.filter_value_and_grad(loss_and_aux, has_aux=True)(
        model, ctrls_bsg, perts_b, tgts_bsg, key
    )

    updates, new_state = optim.update(grads, state, model)

    metrics["optim/grad-norm"] = optax.global_norm(grads)
    metrics["optim/update-norm"] = optax.global_norm(updates)

    model = eqx.apply_updates(model, updates)

    return model, new_state, loss, metrics


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


@jaxtyped(typechecker=beartype.beartype)
class StuffForLoading(tp.NamedTuple):
    adata: ad.AnnData
    gmap: harmonize.GeneMap


@beartype.beartype
class LoadAndLift(grain.transforms.Map):
    def __init__(self, vcc_h5ad: str | pathlib.Path, hvgs: list[str]):
        self._vcc_h5ad = str(vcc_h5ad)
        self._hvgs = hvgs
        self._stuff_for_loading: dict[DatasetConfig, StuffForLoading] = {}

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
            self._gene_vocab = harmonize.GeneVocab(self._hvgs)
        return self._gene_vocab

    def get_stuff_for_loading(self, cfg: DatasetConfig) -> StuffForLoading:
        if cfg not in self._stuff_for_loading:
            fpath = str(cfg.h5ad_fpath)

            # anndata
            adata = ad.read_h5ad(fpath, backed="r")
            self.logger.info("Opened %s.", fpath)

            gmap = self.gene_vocab.make_map(adata)

            self._stuff_for_loading[cfg] = StuffForLoading(adata, gmap)

        return self._stuff_for_loading[cfg]

    def map(self, sample: Sample):
        cfg = sample["cfg"]
        adata, gmap = self.get_stuff_for_loading(cfg)

        # .X can only be indexed in increasing order, so we have to sort. Then we can unsort to preserve the random order again.
        ctrl_sort = np.argsort(sample["sampled_ctrl_rows"])
        pert_sort = np.argsort(sample["sampled_pert_rows"])
        ctrl_unsort = np.argsort(ctrl_sort)
        pert_unsort = np.argsort(pert_sort)

        ctrl_i = sample["sampled_ctrl_rows"][ctrl_sort]
        pert_i = sample["sampled_pert_rows"][pert_sort]

        x_ctrl = adata.X[ctrl_i]
        x_pert = adata.X[pert_i]

        x_ctrl = self.to_array(x_ctrl)[ctrl_unsort]
        x_pert = self.to_array(x_pert)[pert_unsort]

        x_ctrl = gmap.lift(x_ctrl)
        x_pert = gmap.lift(x_pert)

        return {
            "control": x_ctrl,
            "target": x_pert,
            "pert_id": sample["pert_id"],
        }

    def to_array(self, arr):
        if hasattr(arr, "todense"):
            arr = arr.todense()
        return np.asarray(arr)


@beartype.beartype
def make_dataloader(cfg: Config):
    hvgs = harmonize.agg_hvgs([
        pl.read_csv(dataset.hvgs_csv) for dataset in cfg.datasets
    ])

    ops = [
        SampleSet(set_size=cfg.set_size),
        LoadAndLift(cfg.vcc / "adata_Training.h5ad", hvgs),
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

    jax.distributed.initialize()

    pprint.pprint(dataclasses.asdict(cfg))

    key = jr.key(seed=cfg.seed)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # Load training data (backed) and validation counts
    vcc_data = vcc.VccData(cfg.vcc)

    # Init tiny model (OOV=0 row set to 0)
    key, model_key = jr.split(key)
    model = Transformer(
        n_genes=cfg.n_hvgs,
        n_perts=1 + len(vcc_data.train_pert_ids),
        h=cfg.h,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        mlp_mult=4,
        key=model_key,
    )

    params, static = eqx.partition(model, eqx.is_array)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    logger.info("Initialized model with %d params.", n_params)

    optim = vcell.nn.optim.make(cfg.optim)

    state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    logger.info("Initialized optimizer.")

    dataloader = make_dataloader(cfg)
    logger.info("Initialized dataloader.")

    # Train
    global_step = 0
    run = wandb.init(
        entity="samuelstevens", project="vcell", config=dataclasses.asdict(cfg)
    )
    for batch in dataloader:
        key, step_key = jr.split(key)
        model, state, loss, metrics = step_model(
            model,
            optim,
            state,
            jnp.array(batch["control"]),
            jnp.array(batch["pert_id"]),
            jnp.array(batch["target"]),
            step_key,
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

    S = int(vcc_data.val.select(pl.col("n_cells").sum()).item())
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
    X_mm = np.memmap(
        mm_path, mode="w+", dtype=np.float32, shape=(N, len(vcc_data.genes))
    )
    obs_target = np.empty(N, dtype=object)

    ctrl_pool = vcc_data.make_ctrl_pool(key)

    # Stream predictions
    off = 0
    logger.info("Generating predictions (streaming to memmap)...")
    for tg, n in helpers.progress(
        vcc_data.val.select(["target_gene", "n_cells"]).iter_rows()
    ):
        n = int(n)
        key, sample_key = jr.split(key)
        take = jr.choice(sample_key, ctrl_pool, shape=(n,), replace=True)
        take = np.asarray(take, dtype=np.int64)

        x = vcc_data.adata.X[take].toarray().astype(np.float32)
        x = np.log1p(x)
        y = np.asarray(model(x, pert_id=jnp.array(0, dtype=jnp.int32)))

        X_mm[off : off + n] = y
        obs_target[off : off + n] = tg
        off += n

        del x, y

    logger.info("Finished perturbed predictions.")

    # Add controls to meet the 100k cap
    if C > 0:
        key, ctrl_sample_key = jr.split(key)
        ctrl_take = jr.choice(
            ctrl_sample_key, jnp.asarray(ctrl_pool), shape=(C,), replace=True
        )
        ctrl_take = np.asarray(ctrl_take, dtype=np.int64)
        ctrl_cells = vcc_data.adata.X[ctrl_take].toarray().astype(np.float32)
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
    var_df = pd.DataFrame(index=pd.Index(vcc_data.genes, dtype=str))
    pred = ad.AnnData(X=X_arr, obs=obs_df, var=var_df)
    pred.write_h5ad(cfg.out_path, compression="gzip")
    logger.info(f"Wrote predictions to {cfg.out_path} (backed file at {mm_path}).")


if __name__ == "__main__":
    tyro.cli(main)
