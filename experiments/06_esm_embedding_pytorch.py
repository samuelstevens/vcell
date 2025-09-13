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

import torch
import esm

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

# My PyTorch Implementation of ESM2 inference for protein embeddings
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
        logger.info(f"Found {len(protein_isoforms[gene_id])} isoforms for gene {gene_id}")

        # Pretrained ESM2 inference for getting protein embeddings from amino acid sequences
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()  # disable dropout for deterministic results

        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        model = model.to(device)

        # data of the form (label, sequence)
        data = [(f"{gene_id}_{isoform.translation_id}", isoform.amino_acid_sequence) for isoform in protein_isoforms[gene_id]]
        if len(data) == 0:
            logger.warning(f"No isoforms found for gene {gene_id}, skipping")
            continue

        # Convert to ESM2 tokens and move to device
        # batch tokens convert each sequence to a sequence of numbers 0 to 20 (20 amino acids + padding)
        # the alphabet is integers mapped to amino acids + <cls>, <pad>, <eos>, <unk>, <mask>
        # batch tokens is a tensor of shape (batch_size, max_seq_len)
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        with torch.no_grad():
            # repr_layers specifies which layers to return representations from
            # here we only want the final layer (layer 6 for esm2_t6_8M_UR50D)
            # return_contacts specifies whether to return contact predictions - cant do this locally
            results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)

        # for each sequence in the batch, (batch_size), we have a 1 row for each amino acid in the sequence (seq_len), and each amino acid is represented by a vector of length hidden_dim (hidden_dim)
        token_representations = results["representations"][model.num_layers] # (batch_size, seq_len, hidden_dim)

        # Generate per-sequence representations via averaging
        # Note that the <cls> token (first token) and <eos> token (last token) are not included in the averaging
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            # Remove <cls>, padding, and <eos> tokens, take the averager of the remaining tokens to get a embedding for the entire sequence dimensions = (hidden_dim,)
            seq_rep = token_representations[i, 1 : tokens_len - 1].mean(0)
            sequence_representations.append(seq_rep.cpu().numpy())
        
        sequence_representations = np.stack(sequence_representations)  # (num_isoforms, hidden_dim)
        gene_embedding = sequence_representations.mean(0)  # (hidden_dim,)
        logger.info(f"Gene {gene_id} embedding shape: {gene_embedding.shape}")

if __name__ == "__main__":
    main(tyro.cli(Config))
