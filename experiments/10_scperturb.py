# experiments/10_scperturb.py
"""
Train on the training split of VCC and on scPerturb data.


"""

import collections
import dataclasses
import logging
import os
import pathlib
import re
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
import scanpy as sc
import tyro
from jaxtyping import Array, Bool, Float, Int, jaxtyped

import vcell.nn.optim
import wandb
from vcell import helpers, metrics

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("06")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    h5ad_fpath: str
    """Path to h5ad file."""
    pert_col: str = "target_gene"
    ctrl_label: str = "non-targeting"
    group_by: tuple[str, ...] = ("batch",)  # can be empty tuple for "all"


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
) -> tuple[Float[Array, ""], dict]:
    preds_bsg = jax.vmap(model)(ctrls_bsg, perts_b)
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


@jaxtyped(typechecker=beartype.beartype)
@eqx.filter_jit(donate="all")
def step_model(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    state: tp.Any,
    ctrls: Float[Array, "batch set n_genes"],
    perts: Int[Array, " batch"],
    tgts: Float[Array, "batch set n_genes"],
) -> tuple[eqx.Module, tp.Any, Float[Array, ""], dict]:
    (loss, metrics), grads = eqx.filter_value_and_grad(loss_and_aux, has_aux=True)(
        model, ctrls, perts, tgts
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
@dataclasses.dataclass(frozen=True)
class GeneMap:
    """Mapping from a dataset's columns to the canonical VCC gene space."""

    n_genes: int
    """Number of VCC genes"""
    present_mask: Bool[np.ndarray, " G"]
    """which VCC genes exist in this dataset"""
    ds_cols: Int[np.ndarray, " K"]
    """dataset column indices to take"""
    vcc_cols: Int[np.ndarray, " K"]
    """destination VCC columns"""
    stats: dict[str, int]
    """counts for sanity reporting"""

    def lift_to_vcc(self, x_ds: Int[np.ndarray, "..."]) -> Int[np.ndarray, "..."]:
        """
        Project dataset matrix slice (n, n_vars_ds) into VCC order (n, G), filling missing with zeros.
        """
        out = np.zeros((x_ds.shape[0], self.n_genes), dtype=np.float32)
        out[:, self.vcc_cols] = x_ds[:, self.ds_cols]
        return out


@beartype.beartype
class GeneVocab:
    """
    Canonical VCC gene space built from the VCC .h5ad.
    - Prefers Ensembl IDs (stable).
    - Keeps symbols for unique-only fallback.
    """

    def __init__(self, vcc_h5ad: str):
        vcc = sc.read(vcc_h5ad, backed="r")
        if "gene_id" not in vcc.var.columns:
            raise ValueError(
                "Expected VCC .var to contain a 'gene_id' column (Ensembl)."
            )

        self.n_genes = vcc.n_vars

        self.vcc_ens: list[str] = [_strip_ens_version(s) for s in vcc.var["gene_id"]]
        self.vcc_sym: list[str] = vcc.var.index.astype(str).tolist()

        # Ensembl -> VCC index (unique by construction)
        self._ens_to_idx: dict[str, int] = {e: i for i, e in enumerate(self.vcc_ens)}
        # Symbol -> list of indices (can be non-unique)
        self._sym_to_idxs: dict[str, list[int]] = collections.defaultdict(list)
        for i, s in enumerate(self.vcc_sym):
            self._sym_to_idxs[s].append(i)

    def make_map(
        self, ds: ad.AnnData, dup_mode: tp.Literal["sum", "keep", None] = None
    ) -> GeneMap:
        """
        Create a GeneMap from a dataset.
        """

        if dup_mode is None:
            # Try to figure out whether we have raw counts (integers) or log-normalized counts (floats, smaller).
            row = ds.X[0]
            row = row.toarray() if hasattr(row, "toarray") else np.asarray(row)
            if row.max() > 100:
                # Probably raw counts
                dup_mode = "sum"
            elif row[row > 1].min() < 2.0:
                dup_mode = "keep"
            else:
                if ds.isbacked:
                    self.logger.warning(
                        "Not sure whether ds '%s' is raw counts or log normalized.",
                        ds.filename,
                    )
                else:
                    self.logger.warning(
                        "Not sure whether ds is raw counts or log normalized."
                    )

        ds_sym = list(ds.var_names)
        ds_ens = [_strip_ens_version(s) for s in ds.var["ensembl_id"].tolist()]

        assert len(ds_sym) == len(ds_ens)

        present_mask = np.zeros(self.n_genes, dtype=bool)
        ds_cols: list[int] = []
        vcc_cols: list[int] = []

        n_ens_match = 0
        n_sym_match = 0
        n_sym_ambig = 0

        for j, (ens, sym) in enumerate(zip(ds_ens, ds_sym)):
            if ens and ens in self._ens_to_idx:
                i = self._ens_to_idx[ens]
                ds_cols.append(j)
                vcc_cols.append(i)
                present_mask[i] = True
                n_ens_match += 1
            else:
                cand = self._sym_to_idxs.get(sym, [])
                if len(cand) == 1:
                    i = cand[0]
                    ds_cols.append(j)
                    vcc_cols.append(i)
                    present_mask[i] = True
                    n_sym_match += 1
                elif len(cand) > 1:
                    n_sym_ambig += 1
                    # skip ambiguous symbols

        ds_cols = np.asarray(ds_cols, dtype=int)
        vcc_cols = np.asarray(vcc_cols, dtype=int)
        stats = dict(
            vcc_genes=self.n_genes,
            ds_vars=len(ds_sym),
            matched_by_ensembl=int(n_ens_match),
            matched_by_symbol=int(n_sym_match),
            skipped_ambiguous_symbol=int(n_sym_ambig),
            total_matched=int(len(ds_cols)),
            coverage=int(present_mask.sum()),
        )
        return GeneMap(
            n_genes=self.n_genes,
            present_mask=present_mask,
            ds_cols=ds_cols,
            vcc_cols=vcc_cols,
            stats=stats,
        )


@beartype.beartype
def _strip_ens_version(s: str) -> str:
    """ENSG00000187634.5 -> ENSG00000187634"""
    return re.sub(r"\.\d+$", "", s)


@jaxtyped(typechecker=beartype.beartype)
class Sample(tp.TypedDict, total=False):
    filepath: str
    all_rows: Int[np.ndarray, " n"]
    sampled_rows: Int[np.ndarray, " set_size"]
    pert_id: int


@beartype.beartype
class GroupSource(grain.sources.RandomAccessDataSource):
    def __init__(
        self,
        a5hd_fpath,
        group_by: list[str],
        set_size: int,
        pert_col: str = "target_gene",
        ctrl_label: str = "non-targeting",
    ):
        adata = ad.read_h5ad(a5hd_fpath, backed="r")
        control_groups = (
            adata.obs[adata.obs[pert_col] == ctrl_label]
            .groupby(group_by, observed=True)
            .indices
        )
        target_groups = (
            adata.obs[adata.obs[pert_col] != ctrl_label]
            .groupby([pert_col] + group_by, observed=True)
            .indices
        )
        self._samples = []
        self._pert2id = {}
        for pert, *others in sorted(target_groups):
            key = tuple(others) if len(others) > 1 else others[0]
            if key not in control_groups:
                msg = "No observed cells for %s with %s='%s' (control)."
                logger.info(msg, key, pert_col, ctrl_label)
                continue

            ctrl_rows = control_groups[key]
            pert_rows = target_groups[(pert, *others)]

            if ctrl_rows.size < set_size:
                msg = "Skipping %s because only %d control cells (need %d)."
                logger.info(msg, key, ctrl_rows.size, set_size)
                continue

            if pert_rows.size < set_size:
                msg = "Skipping %s because only %d pert cells (need %d)."
                logger.info(msg, key, pert_rows.size, set_size)
                continue

            if pert not in self._pert2id:
                self._pert2id[pert] = len(self._pert2id)

            pert_id = self._pert2id[pert]

            self._samples.append(
                Sample(
                    filepath=a5hd_fpath,
                    all_ctrl_rows=ctrl_rows,
                    all_pert_rows=pert_rows,
                    pert_id=pert_id,
                )
            )

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, k: int):
        return self._samples[k]

    def __repr__(self):
        return f"GroupSource(n={len(self)})"


@beartype.beartype
class SampleSet(grain.transforms.RandomMap):
    def __init__(self, set_size: int):
        self.set_size = set_size

    def random_map(self, sample: Sample, rng) -> Sample:
        ctrl_idx = rng.choice(
            len(sample["all_ctrl_rows"]), size=self.set_size, replace=False
        )
        sample["sampled_ctrl_rows"] = sample["all_ctrl_rows"][ctrl_idx]
        pert_idx = rng.choice(
            len(sample["all_pert_rows"]), size=self.set_size, replace=False
        )
        sample["sampled_pert_rows"] = sample["all_pert_rows"][pert_idx]
        return sample


@beartype.beartype
class LoadH5AD(grain.transforms.Map):
    def __init__(self):
        self._adata = None

    def map(self, sample: Sample):
        if self._adata is None:
            self._adata = ad.read_h5ad(sample["filepath"], backed="r")

        control = self._adata.X[sample["sampled_ctrl_rows"]]
        target = self._adata.X[sample["sampled_pert_rows"]]

        # return numpy arrays (Grain will device-put later)

        return {
            "target": self.array(target),
            "control": self.array(control),
            "pert_id": sample["pert_id"],
        }

    def array(self, arr):
        if hasattr(arr, "todense"):
            arr = arr.todense()
        return np.asarray(arr)


@beartype.beartype
def make_dataloader(cfg: Config):
    source = GroupSource(
        cfg.vcc / "adata_Training.h5ad",
        pert_col="target_gene",
        group_by=["batch"],
        set_size=cfg.set_size,
    )
    ops = [
        SampleSet(set_size=cfg.set_size),
        LoadH5AD(),
        grain.transforms.Batch(batch_size=cfg.batch_size),
    ]

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
    cfg: str = "", override: tp.Annotated[Config, tyro.conf.arg(name="")] = Config()
):
    """Run the experiment.

    Args:
        cfg: Path to config file.
        override: CLI options to modify the config file.
    """

    # Load the config from the cfg path. If it doesn't exist, complain. Then overwrite values in the loaded-from-disk cfg object with any non-default values in override.

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
