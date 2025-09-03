Module vcell.pp.hvgs
====================
Highly variable gene selection methods.

Functions
---------

`highly_variable_genes_seurat_v3_cols(adata: anndata._core.anndata.AnnData, *, n_top_genes: int | None = None, layer: str | None = None, batch_size: int = 2048, target_sum: float = 10000, span: float = 0.3) ‑> pandas.core.frame.DataFrame`
:   Compute Seurat v3 HVGs by streaming columns (genes). Efficient for CSC/col-major data.
    Makes two passes: first to compute library sizes, then to compute statistics.
    
    Returns DataFrame with 'means', 'variances', 'variances_norm', 'highly_variable'.

`highly_variable_genes_seurat_v3_rows(adata: anndata._core.anndata.AnnData, *, n_top_genes: int | None = None, layer: str | None = None, batch_size: int = 2048, target_sum: float = 10000, span: float = 0.3) ‑> pandas.core.frame.DataFrame`
:   Compute Seurat v3 HVGs by streaming rows (cells). Efficient for CSR/row-major data.
    
    Returns DataFrame with 'means', 'variances', 'variances_norm', 'highly_variable'.