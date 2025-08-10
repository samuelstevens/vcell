# experiments/04_validation.py
"""
An experiment to get validation predictions from an arbitrary neural network.
"""

import dataclasses
import logging
import os
import pathlib

import anndata as ad
import beartype
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl
import tyro
from jaxtyping import Array, Float, Int, jaxtyped

from vcell import helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("04")


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
    """where to store model checkpoints."""


@jaxtyped(typechecker=beartype.beartype)
class Model(eqx.Module):
    """
    Minimal permutation-invariant 'token-mixing' mapper:
      controls (set of transcriptomes) + perturbation ID -> predicted perturbed set.

    Design:
      - Gene mixing: project per-transcriptome from g -> d (shared across tokens).
      - Set interaction: mean-pool features across the set to get context c in R^d.
      - Perturbation embedding: table lookup e_p in R^d.
      - Combine: m_i = GELU( W_h h_i + W_c c + W_p e_p ).
      - Output: delta_i = W_out m_i; y_i = x_i + delta_i (residual to preserve scale).
      - Deterministic: no dropout, no noise; fixed behavior for fixed inputs.

    Shapes:
      x: [s, g]  (s = set size, g = genes)
      pert_id: int in [0, n_perts) (or Array[] scalar)
      y: [s, g]
    """

    # parameters
    in_proj: eqx.nn.Linear  # g -> d
    h_proj: eqx.nn.Linear  # d -> d
    c_proj: eqx.nn.Linear  # d -> d
    p_proj: eqx.nn.Linear  # d -> d
    out_proj: eqx.nn.Linear  # d -> g
    pert_table: Float[Array, "n_perts d"]  # learned embeddings

    g: int
    d: int
    n_perts: int

    def __init__(self, n_perts: int, g: int, d: int, key: Array):
        """Uses small-norm init so the residual starts near identity."""
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
        # Encode per-transcriptome features
        x_sd = jax.vmap(self.in_proj)(x_sg)
        h_sd = jax.nn.gelu(x_sd)
        # Pooled context (permutation-invariant, variable s)
        ctx_d = jnp.mean(x_sd, axis=0)
        ctx_d = self.c_proj(ctx_d)

        # Perturbation embedding
        # If pert_id is out of range, caller should map it to a valid OOV id beforehand.
        pert_emb_d = self.pert_table[jnp.asarray(pert_id, dtype=jnp.int32)]
        pert_emb_d = self.p_proj(pert_emb_d)

        # Broadcast context/perturbation to each token and mix
        s, _ = x_sd.shape
        ctx_sd = jnp.broadcast_to(ctx_d, (s, self.d))
        pert_emb_sd = jnp.broadcast_to(pert_emb_d, (s, self.d))
        m_sd = jax.nn.gelu(jax.vmap(self.h_proj)(h_sd) + ctx_sd + pert_emb_sd)

        # Map back to gene space with residual
        delta_sg = jax.vmap(self.out_proj)(m_sd)
        y = x_sg + delta_sg
        return y


@beartype.beartype
def main(cfg: Config):
    key = jax.random.key(seed=cfg.seed)
    key, model_key = jax.random.split(key)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # 1) Load training data (backed) and validation counts
    adata = ad.read_h5ad(cfg.vcc / "adata_Training.h5ad", backed="r")
    genes = np.array(adata.var_names)
    val = pl.read_csv(cfg.vcc / "pert_counts_Validation.csv")

    # 2) Fixed control pool
    ctrl_mask = adata.obs["target_gene"].to_numpy() == "non-targeting"
    ctrl_idx_all = np.nonzero(ctrl_mask)[0]
    key, ctrl_key = jax.random.split(key)
    ctrl_pool = jax.random.choice(
        ctrl_key, ctrl_idx_all, shape=(min(len(ctrl_idx_all), 20000),), replace=False
    )

    # 3) Init tiny model (with OOV=0 row in pert_table set to 0)
    # Get unique training gene IDs (excluding non-targeting)
    train_gene_ids = [
        g for g in adata.obs["target_gene"].unique() if g != "non-targeting"
    ]
    model = Model(n_perts=1 + len(train_gene_ids), g=genes.size, d=64, key=model_key)

    rows = []
    obs_target = []

    # 4) Loop over validation genes
    for row in helpers.progress(val.select(["target_gene", "n_cells"]).iter_rows()):
        tg, n = row
        # deterministic sample of control inputs for this perturbation
        key, sample_key = jax.random.split(key)
        take = jax.random.choice(sample_key, ctrl_pool, shape=(int(n),), replace=True)
        x = adata.X[take].toarray().astype(np.float32)
        x = np.log1p(x)

        y = np.asarray(model(x, pert_id=0))
        rows.append(y)
        obs_target.extend([tg] * y.shape[0])

    logger.info("Finished predictions.")
    breakpoint()

    # 5) Add controls to fill 100k budget
    S = sum(val["n_cells"])
    B = 100_000
    C = max(0, B - int(S))
    key, ctrl_sample_key = jax.random.split(key)
    ctrl_take = jax.random.choice(ctrl_sample_key, ctrl_pool, shape=(C,), replace=True)
    ctrl_cells = np.log1p(adata.X[ctrl_take].toarray().astype(np.float32))
    rows.append(ctrl_cells)
    obs_target.extend(["non-targeting"] * ctrl_cells.shape[0])

    # 6) Build AnnData and save
    X = np.vstack(rows).astype(np.float32)
    obs = pl.DataFrame({"target_gene": obs_target}).to_pandas()
    pred = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    pred.write_h5ad("pred_raw.h5ad", compression="gzip")


if __name__ == "__main__":
    main(tyro.cli(Config))
