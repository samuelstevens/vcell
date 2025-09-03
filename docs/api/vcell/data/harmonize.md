Module vcell.data.harmonize
===========================

Functions
---------

`strip_ens_version(s: str) ‑> str`
:   ENSG00000187634.5 -> ENSG00000187634

Classes
-------

`GeneMap(n_genes: int, present_mask: jaxtyping.Bool[ndarray, 'G'], ds_cols: jaxtyping.Int[ndarray, 'K'], vcc_cols: jaxtyping.Int[ndarray, 'K'], stats: dict[str, int])`
:   Mapping from a dataset's columns to the canonical VCC gene space.

    ### Instance variables

    `ds_cols: jaxtyping.Int[ndarray, 'K']`
    :   dataset column indices to take

    `n_genes: int`
    :   Number of VCC genes

    `present_mask: jaxtyping.Bool[ndarray, 'G']`
    :   which VCC genes exist in this dataset

    `stats: dict[str, int]`
    :   counts for sanity reporting

    `vcc_cols: jaxtyping.Int[ndarray, 'K']`
    :   destination VCC columns

    ### Methods

    `lift_to_vcc(self, x_ds: jaxtyping.Int[ndarray, '...']) ‑> jaxtyping.Int[ndarray, '...']`
    :   Project dataset matrix slice (n, n_vars_ds) into VCC order (n, G), filling missing with zeros.

`GeneVocab(vcc_h5ad: str | pathlib.Path)`
:   Canonical VCC gene space built from the VCC .h5ad.
    - Prefers Ensembl IDs (stable).
    - Keeps symbols for unique-only fallback.

    ### Methods

    `make_map(self, ds: anndata._core.anndata.AnnData, *, gene_id_col: str = 'ensembl_id') ‑> vcell.data.harmonize.GeneMap`
    :   Create a GeneMap from a dataset.