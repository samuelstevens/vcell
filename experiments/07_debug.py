# experiments/07_debug.py
"""
Train on the training split of VCC.

Assume no change in predictions.

Adds the PDS/MRR metric and wandb.
"""

import dataclasses
import logging
import os
import pathlib
import typing as tp

import anndata as ad
import beartype
import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import polars as pl
import tyro
from jaxtyping import Array, Float, Int, jaxtyped

import vcell.data
import wandb
from vcell import helpers, metrics

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("06")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    seed: int = 42
    """Random seed."""

    vcc: pathlib.Path = pathlib.Path("data/inputs/vcc")
    """Path to vcc challenge data."""

    data: vcell.data.PerturbationConfig = vcell.data.PerturbationConfig(
        pathlib.Path("data/inputs/vcc/adata_Training.h5ad"), cell_line_col="guide_id"
    )

    learning_rate: float = 1e-4
    """Learning rate."""
    grad_clip: float = 1.0
    """Maximum gradient norm."""
    batch_size: int = 128
    """Batch size."""
    n_train: int = 1_000

    # Logging
    log_every: int = 10
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

    aux = {
        "mu-mse": mu_mse,
        **{f"pds/{k}": v for k, v in pds.items()},
        **{f"effect-pds/{k}": v for k, v in effect_pds.items()},
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
class Batcher:
    def __init__(self, dataloader, batch_size: int):
        self.dataloader = dataloader
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            batch = []
            it = iter(self.dataloader)
            while len(batch) < self.batch_size:
                batch.append(next(it))

            # Transform list of dicts into dict of arrays
            batch_dict = {}
            for key in batch[0].keys():
                batch_dict[key] = jnp.array([item[key] for item in batch])

            yield batch_dict


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


@beartype.beartype
def main(cfg: Config):
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
    optim = optax.adam(learning_rate=cfg.learning_rate)

    if cfg.grad_clip > 0:
        optim = optax.chain(optax.clip_by_global_norm(cfg.grad_clip), optim)

    state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    logger.info("Initialized optimizer.")

    dataloader = vcell.data.PerturbationDataloader(cfg.data)
    dataloader = Batcher(dataloader, cfg.batch_size)

    # Train
    global_step = 0
    run = wandb.init(
        entity="samuelstevens", project="vcell", config=dataclasses.asdict(cfg)
    )
    batch = next(iter(dataloader))
    for _ in range(cfg.n_train):
        model, state, loss, metrics = step_model(
            model, optim, state, batch["control"], batch["pert"], batch["target"]
        )
        global_step += 1

        if global_step % cfg.log_every == 0:
            metrics = {k: v.item() for k, v in metrics.items()}
            logger.info("step: %d, loss: %.5f %s", global_step, loss.item(), metrics)
            run.log({"step": global_step, "train/loss": loss.item(), **metrics})

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
    main(tyro.cli(Config))
