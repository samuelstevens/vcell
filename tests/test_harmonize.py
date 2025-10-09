import pathlib

import numpy as np
import polars as pl
import pytest
import scanpy as sc

from vcell.data.harmonize import GeneMap, GeneVocab, VccGeneVocab, agg_hvgs


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
    VccGeneVocab(vcc_a5hd_path)


def test_make_map(vcc_a5hd_path, scperturb_a5hd_path):
    gene_vocab = VccGeneVocab(vcc_a5hd_path)
    sc_adata = sc.read(scperturb_a5hd_path, backed="r")
    gene_map = gene_vocab.make_map(sc_adata)
    assert isinstance(gene_map, GeneMap)


def test_gene_map_no_dups(vcc_a5hd_path, scperturb_a5hd_path):
    """
    gene_map.dst_cols is "for each dataset column we kept, which canonical column does it map to?"

    If two (or more) dataset columns point to the same VCC gene, there will be duplicates in dst_cols.

    Why this happens
    - Duplicate Ensembl IDs in the dataset (same gene shown twice in .var).
    - Same symbol repeated (aliases, disambiguation suffixes like GENE-1, GENE-2, or different rows that share a symbol but only one has Ensembl).
    - Transcript/feature-level rows that both collapse to the same gene (you're mapping to gene-level VCC).

    How to fix:
    """
    gene_vocab = VccGeneVocab(vcc_a5hd_path)
    sc_adata = sc.read(scperturb_a5hd_path, backed="r")
    gene_map = gene_vocab.make_map(sc_adata)

    assert np.unique(gene_map.dst_cols).size == gene_map.dst_cols.size


# Tests for generic GeneVocab class
def test_gene_vocab_init():
    """Test GeneVocab initialization with a list of gene symbols."""
    genes = ["BRCA1", "TP53", "EGFR", "MYC", "KRAS"]
    vocab = GeneVocab(genes)
    assert vocab.n_genes == 5
    assert vocab.genes == genes


def test_gene_vocab_make_map_exact_match():
    """Test GeneVocab.make_map with exact matching genes."""
    # Create a mock AnnData with matching genes
    import anndata as ad

    X = np.random.randn(100, 4)
    adata = ad.AnnData(X=X)
    adata.var_names = ["TP53", "EGFR", "MYC", "BRCA2"]

    vocab = GeneVocab(["BRCA1", "TP53", "EGFR", "MYC", "KRAS"])
    gene_map = vocab.make_map(adata)

    assert isinstance(gene_map, GeneMap)
    assert gene_map.n_genes == 5
    # Should match TP53 (index 1), EGFR (index 2), MYC (index 3)
    assert len(gene_map.src_cols) == 3
    assert list(gene_map.src_cols) == [0, 1, 2]  # dataset indices for TP53, EGFR, MYC
    assert list(gene_map.dst_cols) == [1, 2, 3]  # vocab indices for TP53, EGFR, MYC
    assert gene_map.stats["total_matched"] == 3
    assert gene_map.stats["coverage"] == 3


def test_gene_vocab_make_map_no_match():
    """Test GeneVocab.make_map with no matching genes."""
    import anndata as ad

    X = np.random.randn(100, 3)
    adata = ad.AnnData(X=X)
    adata.var_names = ["GENE1", "GENE2", "GENE3"]

    vocab = GeneVocab(["BRCA1", "TP53", "EGFR"])
    gene_map = vocab.make_map(adata)

    assert gene_map.n_genes == 3
    assert len(gene_map.src_cols) == 0
    assert len(gene_map.dst_cols) == 0
    assert gene_map.stats["total_matched"] == 0
    assert gene_map.stats["coverage"] == 0


def test_gene_vocab_make_map_partial_match():
    """Test GeneVocab.make_map with partial matching."""
    import anndata as ad

    X = np.random.randn(100, 6)
    adata = ad.AnnData(X=X)
    adata.var_names = ["BRCA1", "UNKNOWN1", "TP53", "UNKNOWN2", "MYC", "UNKNOWN3"]

    vocab = GeneVocab(["BRCA1", "TP53", "EGFR", "MYC", "KRAS"])
    gene_map = vocab.make_map(adata)

    assert gene_map.n_genes == 5
    assert len(gene_map.src_cols) == 3  # BRCA1, TP53, MYC
    assert list(gene_map.src_cols) == [0, 2, 4]  # indices in dataset
    assert list(gene_map.dst_cols) == [0, 1, 3]  # indices in vocab
    assert gene_map.stats["total_matched"] == 3
    assert gene_map.stats["coverage"] == 3


def test_gene_vocab_make_map_duplicate_handling():
    """Test that GeneVocab handles duplicate genes in the dataset correctly."""
    import anndata as ad

    X = np.random.randn(100, 5)
    adata = ad.AnnData(X=X)
    # Dataset has TP53 twice - should only map the first occurrence
    adata.var_names = ["TP53", "BRCA1", "TP53", "EGFR", "MYC"]

    vocab = GeneVocab(["BRCA1", "TP53", "EGFR", "MYC"])
    gene_map = vocab.make_map(adata)

    # Should skip the duplicate TP53
    assert len(gene_map.src_cols) == 4  # TP53(first), BRCA1, EGFR, MYC
    assert list(gene_map.src_cols) == [0, 1, 3, 4]
    assert list(gene_map.dst_cols) == [1, 0, 2, 3]  # mapping to vocab indices
    # Check no duplicates in dst_cols
    assert np.unique(gene_map.dst_cols).size == gene_map.dst_cols.size


def test_gene_vocab_empty_list():
    """Test GeneVocab with empty gene list."""
    vocab = GeneVocab([])
    assert vocab.n_genes == 0

    import anndata as ad

    X = np.random.randn(100, 3)
    adata = ad.AnnData(X=X)
    adata.var_names = ["GENE1", "GENE2", "GENE3"]

    gene_map = vocab.make_map(adata)
    assert gene_map.n_genes == 0
    assert len(gene_map.src_cols) == 0


def test_gene_vocab_single_gene():
    """Test GeneVocab with a single gene."""
    vocab = GeneVocab(["TP53"])
    assert vocab.n_genes == 1

    import anndata as ad

    X = np.random.randn(100, 3)
    adata = ad.AnnData(X=X)
    adata.var_names = ["BRCA1", "TP53", "EGFR"]

    gene_map = vocab.make_map(adata)
    assert gene_map.n_genes == 1
    assert len(gene_map.src_cols) == 1
    assert gene_map.src_cols[0] == 1  # TP53 is at index 1 in dataset
    assert gene_map.dst_cols[0] == 0  # TP53 is at index 0 in vocab


# Tests for GeneMap.lift method
def test_gene_map_lift_basic():
    """Test basic lifting of dataset matrix to canonical space."""
    # Create a simple gene map
    # Mapping: dataset cols [0, 2, 3] -> canonical cols [1, 4, 2]
    gene_map = GeneMap(
        n_genes=6,
        present_mask=np.array([False, True, True, False, True, False]),
        src_cols=np.array([0, 2, 3]),
        dst_cols=np.array([1, 4, 2]),
        stats={"total_matched": 3},
    )

    # Create dataset matrix with 2 samples and 5 genes
    x_ds = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32
    )

    # Lift to canonical space
    result = gene_map.lift(x_ds)

    # Check shape
    assert result.shape == (2, 6)

    # Check values are correctly mapped
    # Column 0 of dataset -> column 1 of result
    assert result[0, 1] == 1.0
    assert result[1, 1] == 6.0

    # Column 2 of dataset -> column 4 of result
    assert result[0, 4] == 3.0
    assert result[1, 4] == 8.0

    # Column 3 of dataset -> column 2 of result
    assert result[0, 2] == 4.0
    assert result[1, 2] == 9.0

    # Missing genes should be zero
    assert result[0, 0] == 0.0
    assert result[0, 3] == 0.0
    assert result[0, 5] == 0.0


def test_gene_map_lift_missing_genes():
    """Test that missing genes are filled with zeros."""
    # Map only 2 genes from dataset to a canonical space of 5 genes
    gene_map = GeneMap(
        n_genes=5,
        present_mask=np.array([False, True, False, True, False]),
        src_cols=np.array([1, 3]),
        dst_cols=np.array([1, 3]),
        stats={"total_matched": 2},
    )

    # Dataset with 3 samples and 4 genes
    x_ds = np.array(
        [[10, 20, 30, 40], [11, 21, 31, 41], [12, 22, 32, 42]], dtype=np.float32
    )

    result = gene_map.lift(x_ds)

    # Check shape
    assert result.shape == (3, 5)

    # Check mapped values
    assert result[0, 1] == 20  # col 1 -> pos 1
    assert result[0, 3] == 40  # col 3 -> pos 3
    assert result[1, 1] == 21
    assert result[1, 3] == 41
    assert result[2, 1] == 22
    assert result[2, 3] == 42

    # Check zeros for missing genes
    assert result[0, 0] == 0
    assert result[0, 2] == 0
    assert result[0, 4] == 0
    assert result[1, 0] == 0
    assert result[2, 0] == 0


def test_gene_map_lift_empty_mapping():
    """Test lifting when no genes match (empty mapping)."""
    # No genes match between dataset and canonical space
    gene_map = GeneMap(
        n_genes=4,
        present_mask=np.array([False, False, False, False]),
        src_cols=np.array([], dtype=int),
        dst_cols=np.array([], dtype=int),
        stats={"total_matched": 0},
    )

    # Dataset matrix
    x_ds = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    result = gene_map.lift(x_ds)

    # Should return all zeros
    assert result.shape == (2, 4)
    assert np.all(result == 0)


def test_gene_map_lift_single_sample():
    """Test lifting with a single sample."""
    gene_map = GeneMap(
        n_genes=3,
        present_mask=np.array([True, False, True]),
        src_cols=np.array([0, 2]),
        dst_cols=np.array([0, 2]),
        stats={"total_matched": 2},
    )

    # Single sample with 3 genes
    x_ds = np.array([[100, 200, 300]], dtype=np.float32)

    result = gene_map.lift(x_ds)

    assert result.shape == (1, 3)
    assert result[0, 0] == 100
    assert result[0, 1] == 0  # missing
    assert result[0, 2] == 300


def test_agg_hvgs_single_dataset_order():
    """Test that agg_hvgs returns genes in order from best to worst."""
    df = pl.DataFrame({
        "gene_name": ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"],
        "means": [1.0, 2.0, 3.0, 4.0, 5.0],
        "variances": [0.1, 0.2, 0.3, 0.4, 0.5],
        # C > B > A > D > E
        "variances_normalized": [0.5, 0.8, 1.2, 0.3, -0.1],
        "highly_variable": [False, True, True, False, False],
    })

    result = agg_hvgs([df], n_top=5)
    assert result == ["GENE_C", "GENE_B", "GENE_A", "GENE_D", "GENE_E"]


def test_agg_hvgs_single_dataset():
    """Test agg_hvgs with a single dataset."""
    # Create a simple dataset with 5 genes
    df = pl.DataFrame({
        "gene_name": ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"],
        "means": [1.0, 2.0, 3.0, 4.0, 5.0],
        "variances": [0.1, 0.2, 0.3, 0.4, 0.5],
        "variances_normalized": [
            0.5,
            0.8,
            1.2,
            0.3,
            0.1,
        ],  # GENE_C has highest, then B, A, D, E
        "highly_variable": [False, True, True, False, False],
    })

    # Test with n_top=2
    result = agg_hvgs([df], n_top=2)
    assert result == ["GENE_C", "GENE_B"]

    # Test with n_top=3
    result = agg_hvgs([df], n_top=3)
    assert result == ["GENE_C", "GENE_B", "GENE_A"]

    # Test with n_top=1
    result = agg_hvgs([df], n_top=1)
    assert result == ["GENE_C"]


def test_agg_hvgs_two_datasets_no_overlap():
    """Test agg_hvgs with two datasets that have no overlapping genes."""
    # Dataset 1
    df1 = pl.DataFrame({
        "gene_name": ["GENE_A", "GENE_B", "GENE_C"],
        "means": [1.0, 2.0, 3.0],
        "variances": [0.1, 0.2, 0.3],
        "variances_normalized": [1.0, 0.8, 0.6],  # A > B > C
        "highly_variable": [True, True, False],
    })

    # Dataset 2 with completely different genes
    df2 = pl.DataFrame({
        "gene_name": ["GENE_D", "GENE_E", "GENE_F"],
        "means": [4.0, 5.0, 6.0],
        "variances": [0.4, 0.5, 0.6],
        "variances_normalized": [0.9, 0.7, 0.5],  # D > E > F
        "highly_variable": [True, True, False],
    })

    # With n_top=3, should get top genes based on mean rank
    # GENE_A: rank 1 in df1, worst (7) in df2 -> mean rank 4.0
    # GENE_B: rank 2 in df1, worst (7) in df2 -> mean rank 4.5
    # GENE_C: rank 3 in df1, worst (7) in df2 -> mean rank 5.0
    # GENE_D: worst (7) in df1, rank 1 in df2 -> mean rank 4.0
    # GENE_E: worst (7) in df1, rank 2 in df2 -> mean rank 4.5
    # GENE_F: worst (7) in df1, rank 3 in df2 -> mean rank 5.0
    # So top 3 should be A and D (tied at 4.0), then B and E (tied at 4.5)
    result = agg_hvgs([df1, df2], n_top=3)
    # Should include A and D first (tied), then one of B or E
    assert "GENE_A" in result[:2]
    assert "GENE_D" in result[:2]
    assert len(result) == 3


def test_agg_hvgs_two_datasets_with_overlap():
    """Test agg_hvgs with two datasets that have overlapping genes."""
    # Dataset 1
    df1 = pl.DataFrame({
        "gene_name": ["GENE_A", "GENE_B", "GENE_C", "GENE_D"],
        "means": [1.0, 2.0, 3.0, 4.0],
        "variances": [0.1, 0.2, 0.3, 0.4],
        "variances_normalized": [1.5, 1.0, 0.5, 0.2],  # A > B > C > D
        "highly_variable": [True, True, False, False],
    })

    # Dataset 2 with some overlapping genes
    df2 = pl.DataFrame({
        "gene_name": ["GENE_B", "GENE_C", "GENE_E", "GENE_F"],
        "means": [2.5, 3.5, 4.5, 5.5],
        "variances": [0.25, 0.35, 0.45, 0.55],
        "variances_normalized": [1.2, 0.8, 0.6, 0.3],  # B > C > E > F
        "highly_variable": [True, True, True, False],
    })

    # Rankings:
    # GENE_A: rank 1 in df1, worst (7) in df2 -> mean rank 4.0
    # GENE_B: rank 2 in df1, rank 1 in df2 -> mean rank 1.5 (best)
    # GENE_C: rank 3 in df1, rank 2 in df2 -> mean rank 2.5
    # GENE_D: rank 4 in df1, worst (7) in df2 -> mean rank 5.5
    # GENE_E: worst (7) in df1, rank 3 in df2 -> mean rank 5.0
    # GENE_F: worst (7) in df1, rank 4 in df2 -> mean rank 5.5

    result = agg_hvgs([df1, df2], n_top=2)
    assert result == ["GENE_B", "GENE_C"]

    result = agg_hvgs([df1, df2], n_top=3)
    assert result == ["GENE_B", "GENE_C", "GENE_A"]


def test_agg_hvgs_three_datasets():
    """Test agg_hvgs with three datasets."""
    df1 = pl.DataFrame({
        "gene_name": ["GENE_A", "GENE_B"],
        "means": [1.0, 2.0],
        "variances": [0.1, 0.2],
        "variances_normalized": [2.0, 1.0],  # A > B
        "highly_variable": [True, True],
    })

    df2 = pl.DataFrame({
        "gene_name": ["GENE_B", "GENE_C"],
        "means": [2.5, 3.5],
        "variances": [0.25, 0.35],
        "variances_normalized": [1.5, 0.5],  # B > C
        "highly_variable": [True, False],
    })

    df3 = pl.DataFrame({
        "gene_name": ["GENE_A", "GENE_C"],
        "means": [1.5, 3.0],
        "variances": [0.15, 0.3],
        "variances_normalized": [0.8, 1.2],  # C > A
        "highly_variable": [False, True],
    })

    # Rankings:
    # Dataset 1: A=1, B=2, C=worst(4)
    # Dataset 2: B=1, C=2, A=worst(4)
    # Dataset 3: C=1, A=2, B=worst(4)
    # Mean ranks:
    # GENE_A: (1 + 4 + 2) / 3 = 2.33
    # GENE_B: (2 + 1 + 4) / 3 = 2.33
    # GENE_C: (4 + 2 + 1) / 3 = 2.33
    # All tied! Should return all 3 for n_top >= 3

    result = agg_hvgs([df1, df2, df3], n_top=3)
    assert set(result) == {"GENE_A", "GENE_B", "GENE_C"}
    assert len(result) == 3

    # For n_top < 3, the selection might depend on tie-breaking
    result = agg_hvgs([df1, df2, df3], n_top=2)
    assert len(result) == 2
    assert set(result).issubset({"GENE_A", "GENE_B", "GENE_C"})


def test_agg_hvgs_empty_list():
    """Test agg_hvgs with empty list of datasets."""
    result = agg_hvgs([], n_top=10)
    assert result == []


def test_agg_hvgs_single_gene():
    """Test agg_hvgs when there's only one gene."""
    df = pl.DataFrame({
        "gene_name": ["GENE_ONLY"],
        "means": [1.0],
        "variances": [0.1],
        "variances_normalized": [1.0],
        "highly_variable": [True],
    })

    result = agg_hvgs([df], n_top=1)
    assert result == ["GENE_ONLY"]

    result = agg_hvgs([df], n_top=10)
    assert result == ["GENE_ONLY"]


def test_agg_hvgs_n_top_larger_than_genes():
    """Test agg_hvgs when n_top is larger than total number of unique genes."""
    df = pl.DataFrame({
        "gene_name": ["GENE_A", "GENE_B"],
        "means": [1.0, 2.0],
        "variances": [0.1, 0.2],
        "variances_normalized": [0.8, 0.5],
        "highly_variable": [True, False],
    })

    result = agg_hvgs([df], n_top=10)
    assert result == ["GENE_A", "GENE_B"]


def test_agg_hvgs_negative_variances_normalized():
    """Test that agg_hvgs handles negative variances_normalized correctly."""
    df = pl.DataFrame({
        "gene_name": ["GENE_A", "GENE_B", "GENE_C"],
        "means": [1.0, 2.0, 3.0],
        "variances": [0.1, 0.2, 0.3],
        "variances_normalized": [-0.5, 1.0, 0.0],  # B > C > A
        "highly_variable": [False, True, False],
    })

    result = agg_hvgs([df], n_top=2)
    assert result == ["GENE_B", "GENE_C"]


def test_agg_hvgs_tied_variances():
    """Test agg_hvgs when multiple genes have the same variance_normalized."""
    df = pl.DataFrame({
        "gene_name": ["GENE_A", "GENE_B", "GENE_C", "GENE_D"],
        "means": [1.0, 2.0, 3.0, 4.0],
        "variances": [0.1, 0.2, 0.3, 0.4],
        "variances_normalized": [1.0, 1.0, 0.5, 0.5],  # A and B tied, C and D tied
        "highly_variable": [True, True, False, False],
    })

    # Top 2 should include both A and B (order may vary due to tie)
    result = agg_hvgs([df], n_top=2)
    assert set(result[:2]) == {"GENE_A", "GENE_B"}
    assert len(result) == 2


def test_highly_variable_genes_index_name():
    """Test that both HVG functions return DataFrames with index name 'gene_name'."""
    import anndata as ad
    import pandas as pd

    from vcell.pp import (
        highly_variable_genes_seurat_v3_cols,
        highly_variable_genes_seurat_v3_rows,
    )

    X = np.random.randn(100, 50)
    adata = ad.AnnData(X=X)
    adata.var_names = [f"GENE_{i}" for i in range(50)]

    result_rows = highly_variable_genes_seurat_v3_rows(adata, n_top_genes=10)
    assert result_rows.index.name == "gene_name"
    assert isinstance(result_rows, pd.DataFrame)

    result_cols = highly_variable_genes_seurat_v3_cols(adata, n_top_genes=10)
    assert result_cols.index.name == "gene_name"
    assert isinstance(result_cols, pd.DataFrame)
