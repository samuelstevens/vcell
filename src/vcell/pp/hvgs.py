"""
Highly variable gene selection methods.
"""

import anndata as ad
import beartype
import numpy as np
import pandas as pd
import scipy.interpolate
import statsmodels.nonparametric.smoothers_lowess

from .. import helpers


def _compute_seurat_v3_variances(
    means_raw: np.ndarray,
    variances_cp10k: np.ndarray,
    n_cells: int,
    span: float = 0.3,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Seurat v3 standardized variances using lowess fitting.

    This implements the variance stabilizing transformation (VST) from Seurat v3.
    Instead of using loess like scanpy, we use lowess which is simpler but similar.

    Returns:
        variances: Standardized variances (residuals from the trend)
        variances_normalized: Normalized variances used for ranking
    """
    # Filter to genes with non-zero variance to avoid log(0)
    mask = variances_cp10k > 0
    if not mask.any() or mask.sum() < 3:
        # All genes have zero variance or too few genes for fitting
        # Return simple standardized variances without trend fitting
        return variances_cp10k, np.zeros_like(means_raw)

    # Work in log space
    log_means = np.log10(means_raw[mask] + eps)
    log_vars = np.log10(variances_cp10k[mask] + eps)

    # Sort for lowess fitting
    order = np.argsort(log_means)
    x_sorted = log_means[order]
    y_sorted = log_vars[order]

    # Fit lowess with robustifying iterations
    try:
        smooth = statsmodels.nonparametric.smoothers_lowess.lowess(
            endog=y_sorted,
            exog=x_sorted,
            frac=span,
            it=3,  # Robust iterations to downweight outliers
            return_sorted=True,
        )
        xs_smooth, ys_smooth = smooth[:, 0], smooth[:, 1]

        # Remove near-duplicate x values for interpolation
        keep = np.concatenate(([True], np.diff(xs_smooth) > 1e-12))
        xs_smooth, ys_smooth = xs_smooth[keep], ys_smooth[keep]

        # Check if we have enough points after deduplication
        if len(xs_smooth) < 2:
            # Not enough unique points for interpolation
            return variances_cp10k, np.zeros_like(means_raw)
    except (ValueError, np.linalg.LinAlgError):
        # Lowess fitting failed, return simple variances
        return variances_cp10k, np.zeros_like(means_raw)

    # Create interpolator for prediction
    interpolator = scipy.interpolate.PchipInterpolator(
        xs_smooth, ys_smooth, extrapolate=True
    )

    # Predict expected log variance for all genes
    expected_log_vars = interpolator(log_means)

    # Compute standardized variances (in original scale)
    # This is simplified from scanpy's full implementation
    # Scanpy uses clipping and additional normalization
    std_factor = 10**expected_log_vars
    std_factor = np.sqrt(std_factor)

    # Compute standardized variance
    # This is a simplified version - the actual Seurat v3 does more complex clipping
    variances_standardized = np.zeros_like(means_raw)
    variances_standardized[mask] = variances_cp10k[mask] / (std_factor**2)

    # For normalized variances used in ranking, compute residuals in log space
    variances_normalized = np.zeros_like(means_raw)
    variances_normalized[mask] = log_vars - expected_log_vars

    return variances_standardized, variances_normalized


@beartype.beartype
def highly_variable_genes_seurat_v3_rows(
    adata: ad.AnnData,
    *,
    n_top_genes: int | None = None,
    layer: str | None = None,
    batch_size: int = 2048,
    target_sum: float = 10_000,
    span: float = 0.3,
) -> pd.DataFrame:
    """
    Compute Seurat v3 HVGs by streaming rows (cells). Efficient for CSR/row-major data.

    Returns DataFrame with 'means', 'variances', 'variances_normalized', 'highly_variable'.
    """
    n_cells, n_genes = adata.shape

    # Get data matrix
    X = adata.layers[layer] if layer else adata.X

    # Initialize accumulators
    weighted_g = np.zeros((n_genes,), dtype=np.float64)
    squared_g = np.zeros((n_genes,), dtype=np.float64)
    det_g = np.zeros((n_genes,), dtype=np.int64)
    means_raw_g = np.zeros(n_genes, dtype=np.float64)

    # Also need raw squared sums for raw variance
    raw_squared_g = np.zeros(n_genes, dtype=np.float64)

    # Stream through rows (cells)
    for start, end in helpers.progress(
        helpers.batched_idx(n_cells, batch_size), desc="rows"
    ):
        x_bg = X[start:end]
        if hasattr(x_bg, "toarray"):
            x_bg = x_bg.toarray()
        sum_b = x_bg.sum(axis=1)
        weight_b = target_sum / np.maximum(sum_b, 1.0)
        weighted_bg = weight_b[:, None] * x_bg
        weighted_g += weighted_bg.sum(axis=0)
        squared_g += (weighted_bg * weighted_bg).sum(axis=0)
        det_g += (x_bg > 0).sum(axis=0)
        means_raw_g += x_bg.sum(axis=0)
        raw_squared_g += (x_bg * x_bg).sum(axis=0)  # Raw squared values
    means_raw_g /= n_cells

    # Compute CP10K variance
    means_norm = weighted_g / n_cells
    variances_cp10k = (squared_g / n_cells) - (means_norm**2)
    variances_cp10k = np.maximum(variances_cp10k, 0)  # Numerical safety
    # Convert to unbiased estimator (if we have more than 1 cell)
    if n_cells > 1:
        variances_cp10k = variances_cp10k * n_cells / (n_cells - 1)

    # Compute Seurat v3 standardized variances
    variances, variances_normalized = _compute_seurat_v3_variances(
        means_raw_g, variances_cp10k, n_cells, span
    )

    # Select top genes
    highly_variable = np.zeros(n_genes, dtype=bool)
    if n_top_genes is not None:
        top_indices = np.argsort(variances_normalized)[-n_top_genes:]
        highly_variable[top_indices] = True

    # Create DataFrame
    # Note: For plotting we need raw variances, not CP10K normalized
    # Compute raw variances from the raw data
    variances_raw = (raw_squared_g / n_cells) - (means_raw_g**2)
    variances_raw = np.maximum(variances_raw, 0)
    if n_cells > 1:
        variances_raw = variances_raw * n_cells / (n_cells - 1)

    result = pd.DataFrame(
        {
            "means": means_raw_g,
            "variances": variances_raw,  # Raw variances for plotting
            "variances_normalized": variances_normalized,
            "highly_variable": highly_variable,
        },
        index=adata.var_names,
    )
    result.index.name = "gene_name"

    return result


@beartype.beartype
def highly_variable_genes_seurat_v3_cols(
    adata: ad.AnnData,
    *,
    n_top_genes: int | None = None,
    layer: str | None = None,
    batch_size: int = 2048,
    target_sum: float = 10_000,
    span: float = 0.3,
) -> pd.DataFrame:
    """
    Compute Seurat v3 HVGs by streaming columns (genes). Efficient for CSC/col-major data.
    Makes two passes: first to compute library sizes, then to compute statistics.

    Returns DataFrame with 'means', 'variances', 'variances_normalized', 'highly_variable'.
    """
    n_cells, n_genes = adata.shape

    # Get data matrix
    X = adata.layers[layer] if layer else adata.X

    # Initialize accumulators
    weighted_g = np.zeros((n_genes,), dtype=np.float64)
    squared_g = np.zeros((n_genes,), dtype=np.float64)
    det_g = np.zeros((n_genes,), dtype=np.int64)

    # First pass: compute library sizes
    sizes_n = np.zeros(n_cells, dtype=np.float64)
    for start, end in helpers.batched_idx(n_genes, batch_size):
        sizes_n += np.asarray(X[:, start:end].sum(axis=1)).squeeze()

    sizes_n = np.maximum(sizes_n, 1.0)  # guard against empty cells
    weights_n = (target_sum / sizes_n).astype(np.float64)

    # Second pass: compute weighted statistics and raw means
    means_raw = np.zeros(n_genes, dtype=np.float64)
    raw_squared_g = np.zeros(n_genes, dtype=np.float64)
    for start, end in helpers.batched_idx(n_genes, batch_size):
        x_ng = X[:, start:end].copy()

        # Handle both sparse and dense
        if hasattr(x_ng, "toarray"):
            # Sparse matrix
            # Compute raw means and raw squared sums before normalization
            raw_sum = np.asarray(x_ng.sum(axis=0)).squeeze()
            means_raw[start:end] = raw_sum / n_cells
            # Raw squared sum (before normalization)
            x_ng_raw = X[:, start:end].copy()
            x_ng_raw.data **= 2
            raw_squared_g[start:end] = np.asarray(x_ng_raw.sum(axis=0)).squeeze()
            # Now apply normalization for variance computation
            x_ng.data *= weights_n[x_ng.indices]
            weighted_g[start:end] += np.asarray(x_ng.sum(axis=0)).squeeze()
            x_ng.data **= 2
            squared_g[start:end] += np.asarray(x_ng.sum(axis=0)).squeeze()
            # Detection counts (only works for CSC format)
            if hasattr(X[:, start:end], "indptr"):
                det_g[start:end] += np.diff(X[:, start:end].indptr)
            else:
                det_g[start:end] += (X[:, start:end] > 0).sum(axis=0)
        else:
            # Dense matrix
            means_raw[start:end] = x_ng.sum(axis=0) / n_cells
            raw_squared_g[start:end] = (x_ng**2).sum(axis=0)
            x_ng_weighted = x_ng * weights_n[:, None]
            weighted_g[start:end] += x_ng_weighted.sum(axis=0)
            squared_g[start:end] += (x_ng_weighted**2).sum(axis=0)
            det_g[start:end] += (x_ng > 0).sum(axis=0)

    # Compute CP10K variance
    means_norm = weighted_g / n_cells
    variances_cp10k = (squared_g / n_cells) - (means_norm**2)
    variances_cp10k = np.maximum(variances_cp10k, 0)  # Numerical safety
    # Convert to unbiased estimator (if we have more than 1 cell)
    if n_cells > 1:
        variances_cp10k = variances_cp10k * n_cells / (n_cells - 1)

    # Compute Seurat v3 standardized variances
    variances, variances_normalized = _compute_seurat_v3_variances(
        means_raw, variances_cp10k, n_cells, span
    )

    # Select top genes
    highly_variable = np.zeros(n_genes, dtype=bool)
    if n_top_genes is not None:
        top_indices = np.argsort(variances_normalized)[-n_top_genes:]
        highly_variable[top_indices] = True

    # Create DataFrame
    # Compute raw variances for plotting
    variances_raw = (raw_squared_g / n_cells) - (means_raw**2)
    variances_raw = np.maximum(variances_raw, 0)
    if n_cells > 1:
        variances_raw = variances_raw * n_cells / (n_cells - 1)

    result = pd.DataFrame(
        {
            "means": means_raw,
            "variances": variances_raw,  # Raw variances for plotting
            "variances_normalized": variances_normalized,
            "highly_variable": highly_variable,
        },
        index=adata.var_names,
    )
    result.index.name = "gene_name"

    return result
