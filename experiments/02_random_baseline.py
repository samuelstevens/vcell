"""
https://github.com/ArcInstitute/cell-eval/blob/main/tutorials/vcc/vcc.ipynb
"""

import dataclasses
import logging
import pathlib

import anndata as ad
import beartype
import numpy as np
import pandas as pd
import polars as pl
import tyro
from jaxtyping import Int64, Shaped, jaxtyped

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("01")


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    seed: int = 42

    pert_counts_path: pathlib.Path = pathlib.Path(
        "data/inputs/vcc/pert_counts_Validation.csv"
    )

    genes_path: pathlib.Path = pathlib.Path("data/inputs/vcc/gene_names.csv")

    tr_adata_path: pathlib.Path = pathlib.Path("data/inputs/vcc/adata_Training.h5ad")

    out_path: pathlib.Path = pathlib.Path("data/outputs/02/preds.h5ad")


@jaxtyped(typechecker=beartype.beartype)
def random_predictor(
    pert_names: Shaped[np.ndarray, "..."],
    cell_counts: Int64[np.ndarray, "..."],
    gene_names: Shaped[np.ndarray, "..."],
    max_count: int | float = 1e4,
    log1p: bool = True,
) -> ad.AnnData:
    """Generate a random AnnData with the expected number of cells / perturbation.

    This is a dummy function that is meant to stand-in for a perturbation model.
    """
    matrix = np.random.randint(
        0, int(max_count), size=(cell_counts.sum(), gene_names.size)
    )
    if log1p:
        matrix = np.log1p(matrix)

    return ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {"target_gene": np.repeat(pert_names, cell_counts)},
            index=np.arange(cell_counts.sum()).astype(str),
        ),
        var=pd.DataFrame(index=gene_names),
    )


@beartype.beartype
def main(cfg: Config):
    pert_counts = pl.read_csv(cfg.pert_counts_path)
    gene_names = pl.read_csv(cfg.genes_path, has_header=False).to_numpy().flatten()
    logger.info("Read pertubations and genes.")

    adata = random_predictor(
        pert_names=pert_counts.get_column("target_gene").to_numpy(),
        cell_counts=pert_counts.get_column("n_cells").to_numpy(),
        gene_names=gene_names,
    )
    logger.info("Made random predictions.")

    tr_adata = ad.read_h5ad(cfg.tr_adata_path, backed="r")
    logger.info("Read training data.")

    # Filter for non-targeting
    ntc_adata = tr_adata[tr_adata.obs["target_gene"] == "non-targeting"]
    logger.info("Filtered for non-targeting.")

    # Append the non-targeting controls to the example anndata if they're missing
    if "non-targeting" not in adata.obs["target_gene"].unique():
        msg = "Gene-Names are out of order or unequal"
        assert np.all(adata.var_names.values == ntc_adata.var_names.values), msg
        breakpoint()
        # Randomly subsample rows of ntc_adata using numpy indices.
        # Since we cannot apply two views to tr_adata since it is a read-only file-backed matrix, we need to first come up with all the possible indices where target_gene == 'non-targeting', then select a random subset of 1000 of these indices. AI!
        adata = ad.concat([adata, ntc_adata])
        logger.info("Appended data.")

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(cfg.out_path)
    logger.info("Wrote data.")


if __name__ == "__main__":
    main(tyro.cli(Config))
