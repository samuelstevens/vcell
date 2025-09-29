import os.path
import pathlib

import anndata as ad
import beartype
import chex
import jax.numpy as jnp
import jax.random as jr
import polars as pl


@beartype.beartype
class VccData:
    def __init__(self, root: pathlib.Path):
        self.adata = ad.read_h5ad(
            os.path.expandvars(root / "adata_Training.h5ad"), backed="r"
        )
        self.val = pl.read_csv(os.path.expandvars(root / "pert_counts_Validation.csv"))

    def make_ctrl_pool(self, key: chex.PRNGKey, max_controls: int = 20_000):
        """Create a pool of control cell indices for sampling."""
        ctrl_mask = self.adata.obs["target_gene"].to_numpy() == "non-targeting"
        ctrl_idx_all = jnp.nonzero(ctrl_mask)[0]
        key, ctrl_key = jr.split(key)
        ctrl_pool = jr.choice(
            ctrl_key,
            jnp.asarray(ctrl_idx_all),
            shape=(min(len(ctrl_idx_all), max_controls),),
            replace=False,
        )
        return ctrl_pool

    @property
    def train_pert_ids(self) -> list[str]:
        return [
            gname
            for gname in self.adata.obs["target_gene"].unique()
            if gname != "non-targeting"
        ]

    def genes(self) -> list[str]:
        return self.adata.var_names
