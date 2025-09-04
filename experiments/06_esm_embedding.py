# experiments/04_validation.py
"""
An experiment to get validation predictions from an arbitrary neural network.

Key change: stream predictions into a float32 on-disk memmap to avoid holding ~7GB in RAM. Peak RAM is ~one perturbation's chunk (e.g., ~140MB) + overhead.
"""

import dataclasses
import logging
import os
import pathlib

import anndata as ad
import beartype
import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl
import tyro
from jaxtyping import Array, Float, Int, jaxtyped
import requests
from collections import defaultdict

from vcell import helpers

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

@dataclasses.dataclass(frozen=True)
class ProteinIsoform():
    translation_id: str
    amino_acid_sequence: str


@beartype.beartype
def main(cfg: Config):
    adata = ad.read_h5ad(cfg.vcc / "adata_Training.h5ad", backed="r")
    protein_isoforms = defaultdict(list)
    for gene_id in adata.var["gene_id"]:
        # Look up gene id in Ensembl REST API to get protein isoforms
        ens_rsp = requests.get(f"https://rest.ensembl.org/lookup/id/{gene_id}?expand=1;content-type=application/json")
        if ens_rsp.status_code != 200:
            logger.warning(f"Failed to fetch protein isoforms for {gene_id}")
            continue
        
        for transcript in ens_rsp.json().get("Transcript", []):
            if "Translation" in transcript:
                # Lookup translation id to get amino acid sequence
                translation_id = transcript.get("Translation").get("id")
                amino_rsp = requests.get(f"https://rest.ensembl.org/sequence/id/{translation_id}?content-type=text/plain;type=protein")
                
                if amino_rsp.status_code != 200:
                    logger.warning(f"Failed to fetch amino acid sequence for {translation_id} for gene {gene_id}")  
                    continue
                
                protein_isoforms[gene_id].append(ProteinIsoform(translation_id, amino_rsp.text.strip()))
        # TODO: probably too many requests, need to batch or cache
        #logger.info(f"Found {len(protein_isoforms[gene_id])} isoforms for gene {gene_id}")

# TODO: ESM2 with amino acid sequences


if __name__ == "__main__":
    main(tyro.cli(Config))
