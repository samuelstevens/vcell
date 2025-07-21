import dataclasses
import pathlib

import anndata as ad
import beartype
import numpy as np
import polars as pl
import tyro


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Generate the cell-mean baseline `.vcc` submission."""

    train_path: pathlib.Path = pathlib.Path("data/vcc_data/adata_Training.h5ad")
    gene_list: pathlib.Path = pathlib.Path("data/vcc_data/gene_names.csv")
    out_path: pathlib.Path = pathlib.Path("outputs/mean_baseline.vcc")
    cells_per_pert: int = 512
    controls: int = 2048


@beartype.beartype
def main(cfg: Config):
    # cfg.gene_list is just a gene per line separate by new lines. Use pathlib.Path's read file utility to read this to a list instead of polars. AI!
    genes = pl.read_csv(cfg.gene_list, header=None)[0].tolist()
    adata = ad.read_h5ad(cfg.train_path)

    # 1) Re-index to guarantee correct gene order
    adata = adata[:, genes]

    # 2) Compute mean (assume raw counts or log1p counts)
    mean_vec = np.asarray(adata.X.mean(axis=0)).reshape(1, -1)

    # 3) Build synthetic cells
    synthetic = np.repeat(
        mean_vec,
        cfg.controls + cfg.cells_per_pert * len(adata.obs.pert.unique()),
        axis=0,
    )

    # 4) Assemble AnnData
    obs = pl.DataFrame({
        "target_gene": (["non-targeting"] * cfg.controls)
        + [g for g in adata.obs.pert.unique() for _ in range(cfg.cells_per_pert)]
    })
    out = ad.AnnData(synthetic, obs=obs, var=pl.DataFrame(index=genes))
    out.write_h5ad(cfg.out_path.with_suffix(".h5ad"))


if __name__ == "__main__":
    main(tyro.cli(Config))
