# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "beartype",
#     "mudata",
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "scipy",
#     "tyro",
# ]
# ///

"""Assign perturbations to cells based on CRISPR guide counts.

This script processes a MuData h5mu file containing CRISPR guide data and assigns perturbations to cells based on UMI counts. Cells are classified as singlets (one dominant guide), multiplets (multiple guides), unassigned (ambiguous), or no_guides (no CRISPR data).
"""

import dataclasses
import pathlib
import re
import sys

import beartype
import mudata as md
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
import tyro


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    h5mu_fpath: pathlib.Path
    """Path to the input .h5mu file containing CRISPR data."""

    output_fpath: pathlib.Path
    """Path to output .parquet or .csv file with assignments."""

    chunk_size: int = 200_000
    """Number of cells to process at once (for memory efficiency)."""

    min_top_umis: int = 3
    """Minimum UMIs for top guide to be considered (default: 3)."""

    ratio_threshold: float = 3.0
    """Ratio of top to second guide UMIs for singlet assignment (default: 3.0)."""


@beartype.beartype
def get_row_stats(
    X_csr: sp.csr_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute statistics for each row of a sparse matrix.

    Args:
        X_csr: Sparse matrix in CSR format

    Returns:
        Tuple of (top_values, second_values, top_indices, nnz_per_row, total_per_row)
    """
    n_rows = X_csr.shape[0]
    top_values = np.zeros(n_rows, dtype=np.int32)
    second_values = np.zeros(n_rows, dtype=np.int32)
    top_indices = np.full(n_rows, -1, dtype=np.int32)
    nnz_per_row = np.zeros(n_rows, dtype=np.int32)
    total_per_row = np.zeros(n_rows, dtype=np.int64)

    for i in range(n_rows):
        start, end = X_csr.indptr[i], X_csr.indptr[i + 1]
        nnz_per_row[i] = end - start

        if end <= start:
            continue

        row_data = X_csr.data[start:end].astype(np.int32)
        row_indices = X_csr.indices[start:end]

        # Find top and second values
        if len(row_data) == 1:
            top_values[i] = row_data[0]
            top_indices[i] = row_indices[0]
            total_per_row[i] = row_data[0]
        else:
            sorted_idx = np.argsort(row_data)[::-1]
            top_values[i] = row_data[sorted_idx[0]]
            top_indices[i] = row_indices[sorted_idx[0]]
            second_values[i] = row_data[sorted_idx[1]]
            total_per_row[i] = np.sum(row_data)

    return top_values, second_values, top_indices, nnz_per_row, total_per_row


@beartype.beartype
def make_assignments(
    top_values: np.ndarray,
    second_values: np.ndarray,
    nnz_per_row: np.ndarray,
    min_top_umis: int,
    ratio_threshold: float,
) -> np.ndarray:
    """Determine cell status based on guide counts.

    Args:
        top_values: Highest UMI count per cell
        second_values: Second highest UMI count per cell
        nnz_per_row: Number of non-zero guides per cell
        min_top_umis: Minimum UMIs for assignment
        ratio_threshold: Ratio threshold for singlet assignment

    Returns:
        Array of status strings
    """
    n_cells = len(top_values)
    status = np.full(n_cells, "unassigned", dtype=object)

    # Identify different categories
    has_guides = nnz_per_row > 0
    no_guides = ~has_guides

    # Calculate singlets: top guide dominates
    singlet_mask = (
        has_guides
        & (top_values >= min_top_umis)
        & (top_values >= ratio_threshold * np.maximum(second_values, 1))
    )

    # Calculate multiplets: multiple strong guides
    multiplet_mask = (
        has_guides & (~singlet_mask) & (nnz_per_row >= 2) & (top_values >= min_top_umis)
    )

    # Assign statuses
    status[no_guides] = "no_guides"
    status[singlet_mask] = "singlet"
    status[multiplet_mask] = "multiplet"

    return status


@beartype.beartype
def process_chunk(
    crispr_data,
    start_idx: int,
    end_idx: int,
    guide_names: np.ndarray,
    target_genes: np.ndarray,
    cfg: Config,
) -> pd.DataFrame:
    """Process a chunk of cells.

    Args:
        crispr_data: CRISPR AnnData object
        start_idx: Start index for chunk
        end_idx: End index for chunk
        guide_names: Array of guide IDs
        target_genes: Array of target gene names
        cfg: Configuration

    Returns:
        DataFrame with assignment results
    """
    # Get data slice
    X_chunk = crispr_data.X[start_idx:end_idx, :]

    # Ensure CSR format
    if not sp.isspmatrix_csr(X_chunk):
        X_chunk = X_chunk.tocsr()

    # Compute statistics
    top_vals, second_vals, top_idx, nnz, total = get_row_stats(X_chunk)

    # Make assignments
    status = make_assignments(
        top_vals, second_vals, nnz, cfg.min_top_umis, cfg.ratio_threshold
    )

    # Get guide and gene assignments for singlets
    is_singlet = status == "singlet"
    guide_id = np.where(is_singlet, guide_names[top_idx], None)
    target_gene = np.where(is_singlet, target_genes[top_idx], None)

    # Build dataframe
    df = pd.DataFrame({
        "cell": crispr_data.obs_names[start_idx:end_idx],
        "status": status,
        "guide_id": guide_id,
        "target_gene": target_gene,
        "dose_umis": top_vals.astype(np.int32),
        "dose_frac": np.divide(top_vals, np.maximum(total, 1), dtype=np.float32),
        "n_guides_nonzero": nnz.astype(np.int32),
        "total_guide_umis": total.astype(np.int64),
    })

    # Mark control guides if present
    if df["guide_id"].notna().any():
        is_control = (
            df["guide_id"]
            .fillna("")
            .str.contains(r"(NTC|CTRL|CONTROL)", case=False, regex=True)
        )
        df.loc[is_control, "status"] = "control_guide"

    return df


@beartype.beartype
def main(cfg: Config):
    """Main function to assign perturbations."""
    print(f"Reading {cfg.h5mu_fpath}", file=sys.stderr)

    # Load data in backed mode for memory efficiency
    mdata = md.read_h5mu(cfg.h5mu_fpath, backed="r")

    if "crispr" not in mdata.mod:
        raise ValueError("No 'crispr' modality found in h5mu file")

    crispr = mdata["crispr"]
    n_cells = crispr.n_obs
    n_guides = crispr.n_vars

    print(f"Found {n_cells:,} cells and {n_guides:,} guides", file=sys.stderr)

    # Extract guide names and target genes
    guide_names = np.array(crispr.var_names, dtype=object)
    target_genes = np.array(
        [re.sub(r"_\d+$", "", g) for g in guide_names], dtype=object
    )

    # Prepare output writer
    use_parquet = cfg.output_fpath.suffix.lower() == ".parquet"

    if use_parquet:
        schema = pa.schema([
            ("cell", pa.string()),
            ("status", pa.string()),
            ("guide_id", pa.string()),
            ("target_gene", pa.string()),
            ("dose_umis", pa.int32()),
            ("dose_frac", pa.float32()),
            ("n_guides_nonzero", pa.int32()),
            ("total_guide_umis", pa.int64()),
        ])
        writer = pq.ParquetWriter(cfg.output_fpath, schema=schema)
    else:
        fd = open(cfg.output_fpath, "w", buffering=1)
        fd.write(
            "cell,status,guide_id,target_gene,dose_umis,dose_frac,n_guides_nonzero,total_guide_umis\n"
        )

    # Process in chunks
    n_chunks = (n_cells + cfg.chunk_size - 1) // cfg.chunk_size

    for chunk_idx in range(n_chunks):
        start = chunk_idx * cfg.chunk_size
        end = min(n_cells, start + cfg.chunk_size)

        # Process chunk
        df = process_chunk(crispr, start, end, guide_names, target_genes, cfg)

        # Write results
        if use_parquet:
            writer.write_table(pa.Table.from_pandas(df, preserve_index=False))
        else:
            df.to_csv(fd, header=False, index=False)

        print(
            f"[{chunk_idx + 1}/{n_chunks}] Processed cells {start:,}-{end:,}",
            file=sys.stderr,
        )

    # Cleanup
    if use_parquet:
        writer.close()
    else:
        fd.close()

    print(f"Done! Assignments written to {cfg.output_fpath}", file=sys.stderr)


if __name__ == "__main__":
    main(tyro.cli(Config))
