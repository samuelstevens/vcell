# src/vcell/data/harmonize.py
import collections
import dataclasses
import pathlib
import re
import typing as tp

import anndata as ad
import beartype
import numpy as np
import scanpy as sc
from jaxtyping import Bool, Int, jaxtyped


@jaxtyped(typechecker=beartype.beartype)
@dataclasses.dataclass(frozen=True)
class GeneMap:
    """Mapping from a dataset's columns to the canonical VCC gene space."""

    n_genes: int
    """Number of VCC genes"""
    present_mask: Bool[np.ndarray, " G"]
    """which VCC genes exist in this dataset"""
    ds_cols: Int[np.ndarray, " K"]
    """dataset column indices to take"""
    vcc_cols: Int[np.ndarray, " K"]
    """destination VCC columns"""
    stats: dict[str, int]
    """counts for sanity reporting"""

    def lift_to_vcc(self, x_ds: Int[np.ndarray, "..."]) -> Int[np.ndarray, "..."]:
        """
        Project dataset matrix slice (n, n_vars_ds) into VCC order (n, G), filling missing with zeros.
        """
        out = np.zeros((x_ds.shape[0], self.n_genes), dtype=np.float32)
        out[:, self.vcc_cols] = x_ds[:, self.ds_cols]
        return out


@beartype.beartype
class GeneVocab:
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

        self.vcc_ens: list[str] = [_strip_ens_version(s) for s in vcc.var["gene_id"]]
        self.vcc_sym: list[str] = vcc.var.index.astype(str).tolist()

        # Ensembl -> VCC index (unique by construction)
        self._ens_to_idx: dict[str, int] = {e: i for i, e in enumerate(self.vcc_ens)}
        # Symbol -> list of indices (can be non-unique)
        self._sym_to_idxs: dict[str, list[int]] = collections.defaultdict(list)
        for i, s in enumerate(self.vcc_sym):
            self._sym_to_idxs[s].append(i)

    def make_map(
        self, ds: ad.AnnData, dup_mode: tp.Literal["sum", "keep", None] = None
    ) -> GeneMap:
        """
        Create a GeneMap from a dataset.
        """

        if dup_mode is None:
            # Try to figure out whether we have raw counts (integers) or log-normalized counts (floats, smaller).
            row = ds.X[0]
            row = row.toarray() if hasattr(row, "toarray") else np.asarray(row)
            if row.max() > 100:
                # Probably raw counts
                dup_mode = "sum"
            elif row[row > 1].min() < 2.0:
                dup_mode = "keep"
            else:
                if ds.isbacked:
                    self.logger.warning(
                        "Not sure whether ds '%s' is raw counts or log normalized.",
                        ds.filename,
                    )
                else:
                    self.logger.warning(
                        "Not sure whether ds is raw counts or log normalized."
                    )

        ds_sym = list(ds.var_names)
        ds_ens = [_strip_ens_version(s) for s in ds.var["ensembl_id"].tolist()]

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
            ds_cols=ds_cols,
            vcc_cols=vcc_cols,
            stats=stats,
        )


@beartype.beartype
def _strip_ens_version(s: str) -> str:
    """ENSG00000187634.5 -> ENSG00000187634"""
    return re.sub(r"\.\d+$", "", s)
