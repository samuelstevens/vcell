# experiments/00_mean_baseline.py
"""
Generate the trivial "cell-mean" submission for the Virtual Cell Challenge.

On my laptop:

uv run experiments/00_mean_baseline.py --cells-per-pert 1 --controls 1

Then:

uv run cell-eval prep -i outputs/00/preds.h5ad --g data/vcc_data/gene_names.csv -o outputs/00/preds.vcc

Explanation
===========

Let

    X in R^{N x G}   be the training matrix of N cells and G = 18_080 genes
    y in {non-targeting} U {gene symbols}  be the perturbation label for each row of X

1. Compute a single global mean transcriptome

       mu_g = (1 / N) * sum_{i=1..N} X_{i,g}      for g = 1..G

   producing the vector mu in R^{1 x G}.  We iterate over X in small chunks so only O(G) RAM is required.

2. Decide the number of synthetic rows to emit

       total_rows = n_controls + cells_per_pert * n_pert

   where `n_pert` is the number of distinct perturbation genes in the training file (the control label "non-targeting" is excluded).

3. Build the output matrix

       synthetic = repeat(mu, total_rows, axis=0)      # shape: (total_rows,G)

   Every row is identical; there is *no* intra- or inter-perturbation
   variance.
"""

import dataclasses
import logging
import pathlib

import anndata as ad
import beartype
import numpy as np
import polars as pl
import tyro

import vcell.helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("00")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Generate the cell-mean baseline `.h5ad` file."""

    train_path: pathlib.Path = pathlib.Path("data/vcc_data/adata_Training.h5ad")
    gene_list: pathlib.Path = pathlib.Path("data/vcc_data/gene_names.csv")
    out_path: pathlib.Path = pathlib.Path("outputs/00/preds.h5ad")
    cells_per_pert: int = 512
    controls: int = 2048
    chunk_size: int = 1_000


@beartype.beartype
def main(cfg: Config):
    # gene_names.csv has one symbol per line.
    genes = cfg.gene_list.read_text().strip().split("\n")
    adata = ad.read_h5ad(cfg.train_path, backed="r")

    # Calculate mean for each gene across all cells.
    # Because the dataset is quite large, we stream the computation.
    n_cells, n_genes = adata.shape
    accum = np.zeros(n_genes, dtype=np.float32)
    seen = 0

    for sub, start, end in vcell.helpers.progress(
        adata.chunked_X(cfg.chunk_size), total=n_cells // cfg.chunk_size, desc="mean"
    ):
        accum += np.asarray(sub.sum(axis=0)).reshape(-1)
        seen += end - start

    mean_vec = accum / seen

    # 3) Build synthetic cells
    synthetic = np.repeat(
        mean_vec.reshape(1, -1),
        cfg.controls + cfg.cells_per_pert * len(adata.obs["target_gene"].unique()),
        axis=0,
    )

    # 4) Assemble AnnData
    obs = pl.DataFrame({
        "target_gene": (["non-targeting"] * cfg.controls)
        + [
            g
            for g in adata.obs["target_gene"].unique()
            for _ in range(cfg.cells_per_pert)
        ]
    })
    out = ad.AnnData(
        synthetic,
        obs=obs.to_pandas(),
        var=pl.DataFrame({"gene_symbol": genes}).to_pandas().set_index("gene_symbol"),
    )
    # Create parent directories for output path
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(cfg.out_path.with_suffix(".h5ad"))


if __name__ == "__main__":
    main(tyro.cli(Config))
