# src/vcell/data/harmonize.py
import collections
import dataclasses
import pathlib
import re

import anndata as ad
import beartype
import numpy as np
import polars as pl
import scanpy as sc
from jaxtyping import Bool, Int, jaxtyped


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class GeneMap:
    """Mapping from a dataset's columns to a canonical gene space."""

    n_genes: int
    """Number of canonical genes"""
    present_mask: Bool[np.ndarray, " G"]
    """which canonical genes exist in this dataset"""
    src_cols: Int[np.ndarray, " K"]
    """dataset column indices to take"""
    dst_cols: Int[np.ndarray, " K"]
    """destination columns"""
    stats: dict[str, int]
    """counts for sanity reporting"""

    def lift(self, x_ds: Int[np.ndarray, "..."]) -> Int[np.ndarray, "..."]:
        """
        Project dataset matrix slice (n, n_vars_ds) into canonical order (n, G), filling missing with zeros.
        """
        out = np.zeros((x_ds.shape[0], self.n_genes), dtype=np.float32)
        out[:, self.dst_cols] = x_ds[:, self.src_cols]
        return out


@beartype.beartype
class GeneVocab:
    """
    Generic gene vocabulary for mapping datasets to a canonical gene space.
    Uses only gene symbols for matching (no Ensembl IDs).
    """

    def __init__(self, genes: list[str]):
        """
        Initialize with a list of canonical gene symbols.

        Args:
            genes: List of gene symbols defining the canonical space
        """
        self.genes = genes
        self.n_genes = len(genes)

        # Symbol -> index mapping
        self._sym_to_idx: dict[str, int] = {g: i for i, g in enumerate(genes)}

    def make_map(self, ds: ad.AnnData) -> GeneMap:
        """
        Create a GeneMap from a dataset to this vocabulary.

        Args:
            ds: AnnData dataset to map

        Returns:
            GeneMap object defining the mapping
        """
        ds_sym = list(ds.var_names)

        present_mask = np.zeros(self.n_genes, dtype=bool)
        src_cols: list[int] = []
        dst_cols: list[int] = []

        # Track which vocab genes have been mapped to avoid duplicates
        mapped_vocab_genes = set()

        for j, sym in enumerate(ds_sym):
            if sym in self._sym_to_idx and sym not in mapped_vocab_genes:
                i = self._sym_to_idx[sym]
                src_cols.append(j)
                dst_cols.append(i)
                present_mask[i] = True
                mapped_vocab_genes.add(sym)

        src_cols_arr = np.asarray(src_cols, dtype=int)
        dst_cols_arr = np.asarray(dst_cols, dtype=int)

        stats = dict(
            vocab_genes=self.n_genes,
            ds_vars=len(ds_sym),
            total_matched=int(len(src_cols)),
            coverage=int(present_mask.sum()),
        )

        return GeneMap(
            n_genes=self.n_genes,
            present_mask=present_mask,
            src_cols=src_cols_arr,
            dst_cols=dst_cols_arr,
            stats=stats,
        )


@beartype.beartype
class VccGeneVocab:
    """
    Canonical VCC gene space built from the VCC .h5ad.
    - Prefers Ensembl IDs (stable).
    - Keeps symbols for unique-only fallback.
    """

    def __init__(self, vcc_h5ad: str | pathlib.Path):
        vcc = sc.read(str(vcc_h5ad), backed="r")
        if "gene_id" not in vcc.var.columns:
            raise ValueError(
                "Expected VCC .var to contain a 'gene_id' column (Ensembl)."
            )

        self.n_genes = vcc.n_vars

        self.vcc_ens: list[str] = [strip_ens_version(s) for s in vcc.var["gene_id"]]
        self.vcc_sym: list[str] = vcc.var.index.astype(str).tolist()

        # Ensembl -> VCC index (unique by construction)
        self._ens_to_idx: dict[str, int] = {e: i for i, e in enumerate(self.vcc_ens)}
        # Symbol -> list of indices (can be non-unique)
        self._sym_to_idxs: dict[str, list[int]] = collections.defaultdict(list)
        for i, s in enumerate(self.vcc_sym):
            self._sym_to_idxs[s].append(i)

    def make_map(self, ds: ad.AnnData, *, gene_id_col: str = "ensembl_id") -> GeneMap:
        """
        Create a GeneMap from a dataset.
        """

        ds_sym = list(ds.var_names)
        ds_ens = [strip_ens_version(s) for s in ds.var[gene_id_col].tolist()]

        assert len(ds_sym) == len(ds_ens)

        present_mask = np.zeros(self.n_genes, dtype=bool)
        ds_cols: list[int] = []
        vcc_cols: list[int] = []

        n_ens_match = 0
        n_sym_match = 0
        n_sym_ambig = 0

        for j, (ens, sym) in enumerate(zip(ds_ens, ds_sym)):
            if ens and ens in self._ens_to_idx:
                i = self._ens_to_idx[ens]
                ds_cols.append(j)
                vcc_cols.append(i)
                present_mask[i] = True
                n_ens_match += 1
            else:
                cand = self._sym_to_idxs.get(sym, [])
                if len(cand) == 1:
                    i = cand[0]
                    ds_cols.append(j)
                    vcc_cols.append(i)
                    present_mask[i] = True
                    n_sym_match += 1
                elif len(cand) > 1:
                    n_sym_ambig += 1
                    # skip ambiguous symbols

        ds_cols = np.asarray(ds_cols, dtype=int)
        vcc_cols = np.asarray(vcc_cols, dtype=int)
        stats = dict(
            vcc_genes=self.n_genes,
            ds_vars=len(ds_sym),
            matched_by_ensembl=int(n_ens_match),
            matched_by_symbol=int(n_sym_match),
            skipped_ambiguous_symbol=int(n_sym_ambig),
            total_matched=int(len(ds_cols)),
            coverage=int(present_mask.sum()),
        )
        return GeneMap(
            n_genes=self.n_genes,
            present_mask=present_mask,
            src_cols=ds_cols,
            dst_cols=vcc_cols,
            stats=stats,
        )


@beartype.beartype
def strip_ens_version(s: str) -> str:
    """ENSG00000187634.5 -> ENSG00000187634"""
    return re.sub(r"\.\d+$", "", s)


@beartype.beartype
def agg_hvgs(hvgs: list[pl.DataFrame], n_top=2_000) -> list[str]:
    """
    Aggregate highly variable genes across one or more datasets.

    1. For each dataset, rank genes by variances_normalized (descending); unseen genes get the worst rank.
    2. Aggregate per-gene rank across datasets (mean of ranks).
    3. Pick the top-n by aggregated rank.

    Args:
        hvg_csvs: list of dataframes, typically loaded from the .csv files from `scripts/compute_hvgs solo-hvgs`.
        n_top: Number of HVGs to pick.

    Returns:
        Ordered list of HVG names, with best HVG first.
    """
    if not hvgs:
        return []

    schema = {
        "gene_name": pl.String,
        "means": pl.Float64,
        "variances": pl.Float64,
        "variances_normalized": pl.Float64,
        "highly_variable": pl.Boolean,
    }
    dfs = [df.match_to_schema(schema) for df in hvgs]

    # Collect all unique genes across datasets
    all_genes = set()
    for df in dfs:
        all_genes.update(df["gene_name"].to_list())

    # Dictionary to store ranks for each gene across datasets
    gene_ranks = collections.defaultdict(list)

    # For each dataset, rank genes by variances_normalized
    for df in dfs:
        # Get genes and their variances_normalized
        genes = df["gene_name"].to_list()
        variances_norm = df["variances_normalized"].to_list()

        # Create a list of (gene, variance) pairs and sort by variance descending
        gene_var_pairs = list(zip(genes, variances_norm))
        gene_var_pairs.sort(key=lambda x: x[1], reverse=True)

        # Assign ranks (1 is best)
        gene_to_rank = {}
        for rank, (gene, _) in enumerate(gene_var_pairs, 1):
            gene_to_rank[gene] = rank

        # For genes not in this dataset, assign worst rank
        worst_rank = len(all_genes) + 1

        # Store ranks for all genes
        for gene in all_genes:
            if gene in gene_to_rank:
                gene_ranks[gene].append(gene_to_rank[gene])
            else:
                gene_ranks[gene].append(worst_rank)

    # Calculate mean rank for each gene
    gene_mean_ranks = {}
    for gene, ranks in gene_ranks.items():
        gene_mean_ranks[gene] = sum(ranks) / len(ranks)

    # Sort genes by mean rank (ascending, since lower rank is better)
    sorted_genes = sorted(gene_mean_ranks.items(), key=lambda x: x[1])

    # Take top n_top genes and return as ordered list
    top_genes = []
    for gene, _ in sorted_genes[:n_top]:
        top_genes.append(gene)

    return top_genes
