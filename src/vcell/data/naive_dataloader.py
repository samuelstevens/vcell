# src/vcell/data/naive_dataloader.py
import collections
import dataclasses
import difflib
import logging
import pathlib
import warnings

import beartype
import numpy as np
import scanpy as sc


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    h5ad_fpath: pathlib.Path
    """Path to the h5ad file containing perturbation data."""

    set_size: int = 256
    """Number of cells to include in each example set."""

    genes: list[str] = dataclasses.field(default_factory=list)
    """List of gene names to select from the dataset. If empty, all genes are used."""

    pert_col: str = "target_gene"
    """Column name in the AnnData object that contains perturbation information."""

    cell_line_col: str = "cell_line"
    """Column name in the AnnData object that contains cell line information."""

    control_code: str = "non-targeting"
    """Value that identifies control cells in the perturbation column."""


@beartype.beartype
class PerturbationDataloader:
    """
    At each step, randomly samples one unique combination of cell line and perturbation. If there are no observations for the combination, it samples another.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rng = np.random.default_rng()

        # Read the h5ad file
        self.adata = sc.read(self.cfg.h5ad_fpath, backed="r")

        # Apply genes if provided
        if self.cfg.genes:
            # Try to find matching genes in the data
            available_genes = self.adata.var_names.tolist()
            selected_indices = []
            for gene in self.cfg.genes:
                if gene in available_genes:
                    selected_indices.append(available_genes.index(gene))
            if selected_indices:
                self.adata = self.adata[:, selected_indices]

        # Check if data needs normalization
        self._ensure_normalized()

        # Validate column names exist
        self._validate_column(self.cfg.pert_col)
        self._validate_column(self.cfg.cell_line_col)

        # Create ID mappings first
        unique_perts = self.adata.obs[self.cfg.pert_col].unique()
        self.pert2id = {pert: i for i, pert in enumerate(sorted(unique_perts))}

        unique_clines = self.adata.obs[self.cfg.cell_line_col].unique()
        self.cline2id = {cline: i for i, cline in enumerate(sorted(unique_clines))}

    def __iter__(self) -> collections.abc.Iterator[dict]:
        pool = []
        # Randomly sample from the pool
        while True:
            if not pool:
                # If pool is empty, regenerate it
                pool = list(
                    self.adata.obs.groupby(
                        [self.cfg.cell_line_col, self.cfg.pert_col], observed=False
                    ).groups.keys()
                )

            # Randomly select a combination
            idx = self.rng.integers(0, len(pool))
            cell_line, pert = pool.pop(idx)

            # Get perturbed cells for this combination
            pert_mask = (self.adata.obs[self.cfg.cell_line_col] == cell_line) & (
                self.adata.obs[self.cfg.pert_col] == pert
            )
            pert_indices = np.where(pert_mask)[0]

            # Skip if not enough perturbed cells
            if len(pert_indices) < self.cfg.set_size:
                self.logger.debug(
                    "Skipping (%s, %s) with only %d indices.",
                    cell_line,
                    pert,
                    len(pert_indices),
                )
                continue

            # Just match cell line for controls
            control_mask = (self.adata.obs[self.cfg.cell_line_col] == cell_line) & (
                self.adata.obs[self.cfg.pert_col] == self.cfg.control_code
            )
            control_indices = np.where(control_mask)[0]

            # Skip if no control cells available
            if len(control_indices) == 0:
                self.logger.debug(
                    "Skipping (%s, %s) (no control cells).", cell_line, pert
                )
                continue

            # Sample perturbed cells
            sampled_pert_indices = self.rng.choice(
                pert_indices, size=self.cfg.set_size, replace=False
            )

            # Sample control cells (with replacement if needed)
            sampled_control_indices = self.rng.choice(
                control_indices, size=self.cfg.set_size, replace=True
            )

            # Get the expression data
            control_data = self._get_cell_data(sampled_control_indices.tolist())
            target_data = self._get_cell_data(sampled_pert_indices.tolist())

            yield {
                "control": control_data,
                "target": target_data,
                "pert": self.pert2id[pert],
                "cell_line": self.cline2id[cell_line],
            }

    def _ensure_normalized(self):
        """Ensure data is depth-normalized and log1p transformed."""
        # For backed mode, we assume data is already normalized
        # This is because we can't modify backed data
        # In production, this should be checked/documented

        self.logger.warning("Data normalization check not implemented for backed mode")
        warnings.warn(
            "Normalization check not implemented for backed AnnData objects",
            UserWarning,
        )

    def _validate_column(self, col_name: str) -> None:
        """Validate that a column exists, suggesting alternatives if not."""
        if col_name not in self.adata.obs.columns:
            available_cols = list(self.adata.obs.columns)
            # Find similar column names
            suggestions = difflib.get_close_matches(
                col_name, available_cols, n=5, cutoff=0.4
            )

            # Combine suggestions
            all_suggestions = list(dict.fromkeys(suggestions))[:5]

            error_msg = f"Column '{col_name}' not found in AnnData.obs."
            if all_suggestions:
                error_msg += f" Did you mean one of these? {all_suggestions}"
            error_msg += f" Available columns: {available_cols[:10]}{'...' if len(available_cols) > 10 else ''}"

            raise KeyError(error_msg)

    def _get_cell_data(self, indices: list[int]) -> np.ndarray:
        """Get expression data for given cell indices."""
        # h5py requires indices to be sorted and unique
        # Create mapping to handle duplicates
        unique_indices = sorted(set(indices))

        # Get data for unique indices
        data = self.adata[unique_indices].X
        if hasattr(data, "toarray"):
            data = data.toarray()

        # Create lookup for unique indices
        idx_to_data = {idx: data[i] for i, idx in enumerate(unique_indices)}

        # Build result with proper duplicates
        result = np.stack([idx_to_data[idx] for idx in indices])

        return result.astype(np.float32)
