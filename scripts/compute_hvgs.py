"""
Compute HVGs.
"""

import logging
import pathlib
import typing as tp

import anndata as ad
import beartype
import matplotlib.pyplot as plt
import mudata as md
import numpy as np
import scipy.sparse as sp
import tyro

import vcell.helpers
import vcell.pp
import vcell.pp.ensembl

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


@beartype.beartype
def plot_hvgs(
    file: pathlib.Path,
    dump_to: pathlib.Path | None = None,
    n_top_genes: int = 2_000,
):
    """
    Create a scatter plot showing HVGs vs other genes on log(mean) vs log(variance).

    Args:
        file: Path to CSV file with HVG results
        dump_to: Directory to save the plot (defaults to file's parent directory)
        n_top_genes: Number of HVGs for the title (defaults to count in the JSON)
    """
    import polars as pl

    # Load HVG results from JSON
    hvg_df = pl.read_csv(file)

    # Set defaults
    if dump_to is None:
        dump_to = file.parent

    means = hvg_df.get_column("means").to_numpy()
    variances = hvg_df.get_column("variances").to_numpy()
    variances_normalized = hvg_df.get_column("variances_normalized").to_numpy()

    # Re-select top N genes based on variances_norm
    # Use variance residuals for selection
    top_indices = np.argsort(variances_normalized)[-n_top_genes:]
    hvg_mask = np.zeros(hvg_df.height, dtype=bool)
    hvg_mask[top_indices] = True

    # Compute log values with small epsilon for numerical stability
    eps = 1e-12
    log_means = np.log(means + eps)
    log_vars = np.log(variances + eps)

    # Fit lowess trend for visualization (simplified - just for the trend line)
    # We'll use the variance stabilization from our implementation
    from scipy.interpolate import PchipInterpolator
    from statsmodels.nonparametric.smoothers_lowess import lowess

    # Filter to genes with non-zero variance for fitting
    mask = variances > 0
    if mask.sum() > 3:
        # Sort for lowess fitting
        order = np.argsort(log_means[mask])
        x_sorted = log_means[mask][order]
        y_sorted = log_vars[mask][order]

        # Fit lowess
        try:
            smooth = lowess(
                endog=y_sorted,
                exog=x_sorted,
                frac=0.3,
                it=3,
                return_sorted=True,
            )
            xs_smooth, ys_smooth = smooth[:, 0], smooth[:, 1]

            # Remove near-duplicate x values
            keep = np.concatenate(([True], np.diff(xs_smooth) > 1e-12))
            xs_smooth, ys_smooth = xs_smooth[keep], ys_smooth[keep]

            if len(xs_smooth) >= 2:
                # Create interpolator for smooth line
                interpolator = PchipInterpolator(xs_smooth, ys_smooth, extrapolate=True)
                # Generate smooth curve for plotting
                x_plot = np.linspace(log_means[mask].min(), log_means[mask].max(), 100)
                y_plot = interpolator(x_plot)
            else:
                x_plot, y_plot = None, None
        except (ValueError, np.linalg.LinAlgError):
            x_plot, y_plot = None, None
    else:
        x_plot, y_plot = None, None

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300, layout="constrained")

    # Plot non-HVGs (rest)
    ax.scatter(
        log_means[~hvg_mask],
        log_vars[~hvg_mask],
        color="tab:blue",
        alpha=0.4,
        marker=".",
        label="Rest",
        # rasterized=True,  # Rasterize for smaller file size
    )

    # Plot HVGs
    ax.scatter(
        log_means[hvg_mask],
        log_vars[hvg_mask],
        color="tab:green",
        alpha=0.4,
        marker="+",
        label="HVGs",
        # rasterized=True,
    )

    # Plot trend line if available
    if x_plot is not None:
        ax.plot(
            x_plot,
            y_plot,
            color="tab:orange",
            alpha=0.9,
            linewidth=2,
            label=r"$f(\log(\mu))$",
        )

    ax.set_xlabel("log(gene mean)")
    ax.set_ylabel("log(gene variance)")
    ax.set_title(f"Top {n_top_genes} Highly Variable Genes")
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend()

    # Save figure
    save_path = dump_to / f"{file.stem}.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved HVG plot to {save_path}")


@beartype.beartype
def solo_hvgs(
    h5: pathlib.Path,
    dump_to: pathlib.Path,
    order: tp.Literal["rows", "cols", None] = None,
    batch_size: int = 2048,
    target_sum=10_000,
    mod: str = "",
):
    if h5.suffix == ".h5ad":
        adata = ad.read_h5ad(h5, backed="r")
    elif h5.suffix == ".h5mu":
        mdata = md.read_h5mu(h5, backed="r")

        if mdata.n_mod == 1 and not mod:
            mod = mdata.mod_names[0]
            print(f"Assigning mod='{mod}'.")

        if not mod:
            print(f"Need to pass --mod. Available: {mdata.mod_names}")
            return

        if mod not in mdata.mod:
            print(f"Unknown modality --mod '{mod}'. Available: {mdata.mod_names}")
            return

        adata = mdata.mod[mod]
    else:
        print(f"Unknown file type '{h5.suffix}'")
        return

    n_cells, n_genes = adata.shape

    info = describe_layout(adata.X)
    if order is None:
        if info["recommended_stream"] is None:
            print(
                "Cannot figure out what streaming order to use. Pass --order. Exiting."
            )
            return
        order = info["recommended_stream"]
        summary = ", ".join(f"{key}='{value}'" for key, value in info.items() if value)
        print(
            f"Based on this info: {summary}, this script will iterate over '{order}'."
        )

    elif info["recommended_stream"] != order:
        print(
            f"You passed --order {order}. This script believes you should have used --order {info['recommended_stream']}. Good luck!"
        )

    assert order is not None

    if order == "rows":
        hvgs = vcell.pp.highly_variable_genes_seurat_v3_rows(adata, n_top_genes=2_000)
    elif order == "cols":
        hvgs = vcell.pp.highly_variable_genes_seurat_v3_cols(adata, n_top_genes=2_000)
    else:
        tp.assert_never(order)

    dump_to.mkdir(parents=True, exist_ok=True)

    # Save HVG results as JSON
    csv_path = dump_to / f"{h5.stem}.csv"
    hvgs.to_csv(csv_path)
    print(f"Saved HVG results to {csv_path}")

    # Create and save visualization
    plot_hvgs(csv_path, dump_to, n_top_genes=2_000)


def describe_layout(x):
    """
    Return a dict describing storage + a recommended streaming axis: 'rows' or 'cols'.
    """
    info = {
        "kind": "unknown",
        "backed": None,
        "format": None,
        "library": None,
        "recommended_stream": None,
        "notes": "",
    }

    # --- In-memory dense (incl. numpy.memmap) ---
    if isinstance(x, np.ndarray):
        info.update(
            kind="dense",
            library="numpy",
            backed=isinstance(x, np.memmap),
            format="ndarray",
        )
        # Prefer the contiguous axis
        if x.flags["F_CONTIGUOUS"] and not x.flags["C_CONTIGUOUS"]:
            info["recommended_stream"] = "cols"
            info["notes"] = "Fortran-order array; columns contiguous."
        else:
            info["recommended_stream"] = "rows"
            info["notes"] = "C-order array; rows contiguous (common case)."
        return info

    # --- In-memory SciPy sparse ---
    if sp.isspmatrix_csr(x):
        info.update(
            kind="sparse",
            library="scipy",
            backed=False,
            format="csr",
            recommended_stream="rows",
            notes="CSR favors row streaming.",
        )
        return info
    if sp.isspmatrix_csc(x):
        info.update(
            kind="sparse",
            library="scipy",
            backed=False,
            format="csc",
            recommended_stream="cols",
            notes="CSC favors column streaming.",
        )
        return info
    if sp.issparse(x):
        info.update(
            kind="sparse",
            library="scipy",
            backed=False,
            format=x.getformat(),
            recommended_stream=("rows" if x.getformat() == "csr" else "cols"),
        )
        return info

    # --- AnnData's backed sparse wrapper (HDF5/Zarr): SparseDataset ---
    try:
        # present in recent anndata
        from anndata._core.sparse_dataset import SparseDataset

        if isinstance(x, SparseDataset):
            fmt = getattr(x, "format", None)  # 'csr' or 'csc'
            info.update(
                kind="sparse",
                library="anndata",
                backed=True,
                format=fmt,
                recommended_stream=("rows" if fmt == "csr" else "cols"),
                notes="Backed SparseDataset wrapper.",
            )
            return info
    except Exception:
        pass

    # --- HDF5 dense dataset ---
    try:
        import h5py

        if isinstance(x, h5py.Dataset):
            info.update(kind="dense", library="h5py", backed=True, format="hdf5")
            # Use chunking to pick a direction
            chunks = x.chunks  # None = contiguous
            if chunks is None:
                info["recommended_stream"] = "rows"
                info["notes"] = "Contiguous HDF5; row slabs are typically fastest."
            else:
                r_chunk, c_chunk = chunks
                info["recommended_stream"] = "rows" if r_chunk >= c_chunk else "cols"
                info["notes"] = f"HDF5 chunks={chunks}; pick the larger-chunk axis."
            return info
    except Exception:
        pass

    # --- Zarr dense dataset ---
    try:
        import zarr

        if isinstance(x, zarr.Array):
            r_chunk, c_chunk = x.chunks
            info.update(
                kind="dense",
                library="zarr",
                backed=True,
                format="zarr",
                recommended_stream=("rows" if r_chunk >= c_chunk else "cols"),
                notes=f"Zarr chunks={x.chunks}; pick the larger-chunk axis.",
            )
            return info
    except Exception:
        pass

    # --- Fallback for unknown backed sparse types with a .format attribute ---
    fmt = getattr(x, "format", None)
    if fmt in {"csr", "csc"}:
        info.update(
            kind="sparse",
            library="unknown",
            backed=True,
            format=fmt,
            recommended_stream=("rows" if fmt == "csr" else "cols"),
        )
        return info

    return info


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "solo-hvgs": solo_hvgs,
        "plot-hvgs": plot_hvgs,
    })
