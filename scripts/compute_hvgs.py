"""
Compute HVGs.
"""

import logging
import pathlib
import typing as tp

import anndata as ad
import beartype
import mudata as md
import numpy as np
import scipy.sparse as sp
import tyro

import vcell.helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)


@beartype.beartype
def save_cell_weights(
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

    weighted_g = np.zeros((n_genes,), dtype=np.float64)
    squared_g = np.zeros((n_genes,), dtype=np.float64)
    det_g = np.zeros((n_genes,), dtype=np.int64)

    info = describe_layout(adata.X)
    if order is None:
        if info["recommended_stream"] is None:
            print(
                "Cannot figure out what streaming order to use. Pass --order. Exiting."
            )
            return
        order = info["recommended_stream"]
        summary = ", ".join(f"{key}='{value}'" for key, value in info.items() if value)
        print(f"Based on this info: {summary},this script will iterate over '{order}'.")

    elif info["recommended_stream"] != order:
        print(
            f"You passed --order {order}. This script believes you should have used --order {info['recommended_stream']}. Good luck!"
        )

    assert order is not None

    if order == "rows":
        for start, end in vcell.helpers.progress(
            vcell.helpers.batched_idx(n_cells, batch_size)
        ):
            x_bg = adata.X[start:end]
            if hasattr(x_bg, "toarray"):
                x_bg = x_bg.toarray()
            sum_b = x_bg.sum(axis=1)
            weight_b = target_sum / np.maximum(sum_b, 1.0)
            weighted_bg = weight_b[:, None] * x_bg
            weighted_g += weighted_bg.sum(axis=0)
            squared_g += (weighted_bg * weighted_bg).sum(axis=0)
            det_g += (x_bg > 0).sum(axis=0)

    elif order == "cols":
        # Compute the per-gene statistics needed for Seurat v3–style HVG selection on huge, file-backed matrices by streaming columns. Given raw counts X, it makes two passes: (1) compute per-cell library sizes to get CP10K weights w_i; (2) accumulate, for each gene g, \sum_i w_i X_ig, sum_i (w_i X_ig)^2 and detection counts. These O(G)-memory aggregates yield mu_g and var_g to fit the log mean–variance trend and rank HVGs, without ever loading all of X into RAM. We prefer CSC.
        sizes_n = np.zeros(n_cells, dtype=np.float64)
        for start, end in vcell.helpers.progress(
            vcell.helpers.batched_idx(n_genes, batch_size), desc="sizes"
        ):
            sizes_n += np.asarray(adata.X[:, start:end].sum(axis=1)).squeeze()

        sizes_n = np.maximum(sizes_n, 1.0)  # guard against empty cells
        weights_n = (target_sum / sizes_n).astype(np.float64)

        for start, end in vcell.helpers.progress(
            vcell.helpers.batched_idx(n_genes, batch_size), desc="moments"
        ):
            x_ng = adata.X[:, start:end].copy()
            x_ng.data *= weights_n[x_ng.indices]
            weighted_g[start:end] += np.asarray(x_ng.sum(axis=0)).squeeze()
            x_ng.data **= 2
            squared_g[start:end] += np.asarray(x_ng.sum(axis=0)).squeeze()
            det_g[start:end] += np.diff(adata.X[:, start:end].indptr)

    else:
        tp.assert_never(order)

    arrs = dict(
        weighted_g=weighted_g,
        squared_g=squared_g,
        det_g=det_g,
        n_cells=n_cells,
        n_genes=n_genes,
    )
    dump_to.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dump_to / f"{h5.stem}.npz", allow_pickle=False, **arrs)


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
        "save-cell-weights": save_cell_weights,
        "nothing": lambda: None,
    })
