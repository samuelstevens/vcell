import pathlib

import numpy as np
import pytest
import scanpy as sc

from vcell.data.harmonize import GeneMap, GeneVocab


@pytest.fixture(scope="session")
def vcc_a5hd_path(pytestconfig):
    shards = pytestconfig.getoption("--vcc")
    if shards is None:
        pytest.skip("--vcc not supplied")
    return shards


@pytest.fixture(scope="session")
def scperturb_root(pytestconfig):
    shards = pytestconfig.getoption("--scperturb")
    if shards is None:
        pytest.skip("--scperturb not supplied")
    return shards


@pytest.fixture(scope="session")
def scperturb_a5hd_path(scperturb_root):
    paths = list(pathlib.Path(scperturb_root).glob("**/*.h5ad"))
    return paths[0]


def test_init(vcc_a5hd_path):
    GeneVocab(vcc_a5hd_path)


def test_make_map(vcc_a5hd_path, scperturb_a5hd_path):
    gene_vocab = GeneVocab(vcc_a5hd_path)
    sc_adata = sc.read(scperturb_a5hd_path, backed="r")
    gene_map = gene_vocab.make_map(sc_adata)
    assert isinstance(gene_map, GeneMap)


def test_gene_map_no_dups(vcc_a5hd_path, scperturb_a5hd_path):
    """
    gene_map.vcc_cols is "for each dataset column we kept, which VCC column does it map to?"

    If two (or more) dataset columns point to the same VCC gene, there will be duplicates in vcc_cols.

    Why this happens
    - Duplicate Ensembl IDs in the dataset (same gene shown twice in .var).
    - Same symbol repeated (aliases, disambiguation suffixes like GENE-1, GENE-2, or different rows that share a symbol but only one has Ensembl).
    - Transcript/feature-level rows that both collapse to the same gene (you’re mapping to gene-level VCC).

    How to fix:
    """
    gene_vocab = GeneVocab(vcc_a5hd_path)
    sc_adata = sc.read(scperturb_a5hd_path, backed="r")
    gene_map = gene_vocab.make_map(sc_adata)

    assert np.unique(gene_map.vcc_cols).size == gene_map.vcc_cols.size
