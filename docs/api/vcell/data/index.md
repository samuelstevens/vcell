Module vcell.data
=================

Sub-modules
-----------
* vcell.data.naive_dataloader

Classes
-------

`PerturbationConfig(h5ad_fpath: pathlib.Path, set_size: int = 256, genes: list[str] = <factory>, pert_col: str = 'target_gene', cell_line_col: str = 'cell_line', control_code: str = 'non-targeting', seed: int = 0)`
:   Config(h5ad_fpath: pathlib.Path, set_size: int = 256, genes: list[str] = <factory>, pert_col: str = 'target_gene', cell_line_col: str = 'cell_line', control_code: str = 'non-targeting', seed: int = 0)

    ### Instance variables

    `cell_line_col: str`
    :   Column name in the AnnData object that contains cell line information.

    `control_code: str`
    :   Value that identifies control cells in the perturbation column.

    `genes: list[str]`
    :   List of gene names to select from the dataset. If empty, all genes are used.

    `h5ad_fpath: pathlib.Path`
    :   Path to the h5ad file containing perturbation data.

    `pert_col: str`
    :   Column name in the AnnData object that contains perturbation information.

    `seed: int`
    :   Random seed.

    `set_size: int`
    :   Number of cells to include in each example set.

`PerturbationDataloader(cfg: vcell.data.naive_dataloader.Config)`
:   At each step, randomly samples one unique combination of cell line and perturbation. If there are no observations for the combination, it samples another.