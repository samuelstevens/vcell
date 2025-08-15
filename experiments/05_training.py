# experiments/05_train.py
"""
Train on the training split of VCC.

Assume no change in predictions.
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

from vcell import helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("05")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    seed: int = 42
    """Random seed."""

    vcc: pathlib.Path = pathlib.Path("data/inputs/vcc")
    """Path to vcc challenge data."""

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
def compute_loss(
    model: eqx.Module,
    ctrls: Float[Array, "batch set n_genes"],
    perts: Int[Array, " batch"],
    tgts: Float[Array, "batch set n_genes"],
) -> Float[Array, ""]:
    logits = jax.vmap(model)(ctrls, perts)
    loss = jnp.mean((logits - tgts) ** 2)

    return loss


@jaxtyped(typechecker=beartype.beartype)
@eqx.filter_jit(donate="all")
def step_model(
    model: eqx.Module,
    optim: optax.GradientTransformation,
    state: tp.Any,
    ctrls: Float[Array, "batch set n_genes"],
    perts: Int[Array, " batch"],
    tgts: Float[Array, "batch set n_genes"],
) -> tuple[eqx.Module, tp.Any, Float[Array, ""]]:
    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, ctrls, perts, tgts)
    (updates,), new_state = optim.update([grads], state, [model])

    model = eqx.apply_updates(model, updates)

    return model, new_state, loss


@beartype.beartype
def main(cfg: Config):
    key = jax.random.key(seed=cfg.seed)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # Load training data (backed) and validation counts
    adata = ad.read_h5ad(cfg.vcc / "adata_Training.h5ad", backed="r")
    genes = np.array(adata.var_names)
    g = genes.size

    val = pl.read_csv(cfg.vcc / "pert_counts_Validation.csv")
    S = int(val.select(pl.col("n_cells").sum()).item())
    B = 100_000
    C = max(0, B - S)
    N = S + C
    logger.info(
        f"Validation requested cells: {S:,}. Controls to add: {C:,}. Total: {N:,}."
    )

    # Fixed control pool (deterministic)
    ctrl_mask = adata.obs["target_gene"].to_numpy() == "non-targeting"
    ctrl_idx_all = np.nonzero(ctrl_mask)[0]
    key, ctrl_key = jax.random.split(key)
    ctrl_pool_jax = jax.random.choice(
        ctrl_key,
        jnp.asarray(ctrl_idx_all),
        shape=(min(len(ctrl_idx_all), 20_000),),
        replace=False,
    )
    ctrl_pool = np.asarray(ctrl_pool_jax, dtype=np.int64)

    # Init tiny model (OOV=0 row set to 0)
    train_gene_ids = [
        gname for gname in adata.obs["target_gene"].unique() if gname != "non-targeting"
    ]
    key, model_key = jax.random.split(key)
    model = Model(n_perts=1 + len(train_gene_ids), g=g, d=64, key=model_key)
    model = eqx.tree_at(lambda m: m.pert_table, model, model.pert_table.at[0].set(0.0))
    optim = optax.adamw(
        learning_rate=cfg.learning_rate,
        b1=cfg.beta1,
        b2=cfg.beta2,
        weight_decay=cfg.weight_decay,
    )

    if cfg.grad_clip > 0:
        optim = optax.chain(optim, optax.clip_by_global_norm(cfg.grad_clip))

    state = optim.init(eqx.filter([model], eqx.is_inexact_array))
    logger.info("Initialized optimizer.")

    # Train
    breakpoint()
    global_step = 0

    for epoch in range(cfg.n_epochs):
        for b, (ctrls, perts, tgts) in enumerate(dataloader):
            key, *subkeys = jax.random.split(key, num=cfg.batch_size + 1)

            model, state, loss = step_model(model, optim, state, ctrls, perts, tgts)
            global_step += 1

            if global_step % cfg.log_every == 0:
                logger.info(
                    "epoch: %d, step: %d, loss: %.5f", epoch, global_step, loss.item()
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
        take = jax.random.choice(
            sample_key, jnp.asarray(ctrl_pool), shape=(n,), replace=True
        )
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
