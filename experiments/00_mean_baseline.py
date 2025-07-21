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
    """Generate the cell-mean baseline `.vcc` submission."""

    train_path: pathlib.Path = pathlib.Path("data/vcc_data/adata_Training.h5ad")
    gene_list: pathlib.Path = pathlib.Path("data/vcc_data/gene_names.csv")
    out_path: pathlib.Path = pathlib.Path("outputs/mean_baseline.vcc")
    cells_per_pert: int = 512
    controls: int = 2048
    chunk_size: int = 1_000


@beartype.beartype
def main(cfg: Config):
    # gene_names.csv has one symbol per line.
    genes = cfg.gene_list.read_text().strip().split("\n")
    adata = ad.read_h5ad(cfg.train_path, backed="r")

    # Calculate mean for each gene across all cells.
    n_cells, n_genes = adata.shape
    accum = np.zeros(n_genes, dtype=np.float64)
    seen = 0

    for sub, start, end in vcell.helpers.progress(
        adata.chunked_X(cfg.chunk_size), total=n_cells // cfg.chunk_size, desc="mean"
    ):
        accum += np.asarray(sub.sum(axis=0)).reshape(-1)
        seen += end - start

    mean_vec = accum / seen

    # breakpoint()

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
    breakpoint()
    out = ad.AnnData(
        synthetic,
        obs=obs.to_pandas(),
        var=pl.DataFrame({"gene_symbol": genes}).to_pandas().set_index("gene_symbol"),
    )
    # make the parent directory of cfg.out_path with pathlib.Path.parent.makedirs or such. It's okay if it exists already. Also create all parent directories. AI!
    out.write_h5ad(cfg.out_path.with_suffix(".h5ad"))


if __name__ == "__main__":
    main(tyro.cli(Config))
