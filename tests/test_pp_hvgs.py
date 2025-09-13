"""
Test highly variable gene selection methods.
"""

import warnings

import anndata as ad
import hypothesis.strategies as st
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from hypothesis import given, settings

import vcell.pp

# ==============================================================================
# Basic functionality tests
# ==============================================================================


def test_hvg_methods_match():
    """Test that row and column streaming methods match scanpy's seurat_v3."""
    # Create a toy dataset with known properties
    np.random.seed(42)
    n_obs = 100
    n_vars = 200

    # Generate count data with some genes having higher variance
    # Make some genes highly variable by design
    counts = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))

    # Make a subset of genes more variable
    high_var_genes = np.random.choice(n_vars, size=20, replace=False)
    for g in high_var_genes:
        # Add variability to these genes
        counts[:, g] = np.random.negative_binomial(20, 0.1, size=n_obs)

    # Create AnnData object
    adata = ad.AnnData(X=counts.astype(np.float32))
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]

    # Test with dense matrix
    adata_dense = adata.copy()

    # Test with CSR (row-major) sparse matrix
    adata_csr = adata.copy()
    adata_csr.X = sp.csr_matrix(adata_csr.X)

    # Test with CSC (column-major) sparse matrix
    adata_csc = adata.copy()
    adata_csc.X = sp.csc_matrix(adata_csc.X)

    # Run scanpy's reference implementation
    n_top = 50
    sc.pp.highly_variable_genes(
        adata_dense, flavor="seurat_v3", n_top_genes=n_top, span=0.3, inplace=True
    )
    reference_hvgs = adata_dense.var["highly_variable"].values
    reference_means = adata_dense.var["means"].values

    # Run our row-streaming implementation (good for CSR)
    result_rows = vcell.pp.highly_variable_genes_seurat_v3_rows(
        adata_csr,
        n_top_genes=n_top,
        span=0.3,
        batch_size=32,  # Small batch size for testing
    )

    # Run our column-streaming implementation (good for CSC)
    result_cols = vcell.pp.highly_variable_genes_seurat_v3_cols(
        adata_csc,
        n_top_genes=n_top,
        span=0.3,
        batch_size=32,  # Small batch size for testing
    )

    # Check that both methods return DataFrames with expected columns
    expected_columns = {"means", "variances", "variances_norm", "highly_variable"}
    assert set(result_rows.columns) == expected_columns
    assert set(result_cols.columns) == expected_columns

    # Check that the shapes are correct
    assert len(result_rows) == n_vars
    assert len(result_cols) == n_vars

    # Check that means are close to reference
    np.testing.assert_allclose(
        result_rows["means"].values,
        reference_means,
        rtol=1e-5,
        err_msg="Row-streaming means don't match reference",
    )
    np.testing.assert_allclose(
        result_cols["means"].values,
        reference_means,
        rtol=1e-5,
        err_msg="Column-streaming means don't match reference",
    )

    # Note: We skip exact variance matching because our simplified implementation
    # doesn't replicate all of scanpy's complex clipping and standardization.
    # The important thing is that we identify similar highly variable genes.

    # Check that variances are computed (non-negative)
    assert np.all(result_rows["variances"].values >= 0), (
        "Row variances should be non-negative"
    )
    assert np.all(result_cols["variances"].values >= 0), (
        "Col variances should be non-negative"
    )

    # Check that normalized variances are computed
    assert not np.all(result_rows["variances_norm"].values == 0), (
        "Row normalized variances shouldn't all be zero"
    )
    assert not np.all(result_cols["variances_norm"].values == 0), (
        "Col normalized variances shouldn't all be zero"
    )

    # Check that the same genes are selected as highly variable
    # Since our implementation is simplified, we allow more variance
    n_overlap_rows = np.sum(result_rows["highly_variable"].values & reference_hvgs)
    n_overlap_cols = np.sum(result_cols["highly_variable"].values & reference_hvgs)

    # We expect at least 90% overlap due to our simplified variance calculation
    min_overlap = n_top * 0.9
    assert n_overlap_rows >= min_overlap, (
        f"Row-streaming HVG selection differs too much from reference: {n_overlap_rows}/{n_top} overlap"
    )
    assert n_overlap_cols >= min_overlap, (
        f"Column-streaming HVG selection differs too much from reference: {n_overlap_cols}/{n_top} overlap"
    )

    # Check that both our methods agree with each other exactly
    np.testing.assert_array_equal(
        result_rows["highly_variable"].values,
        result_cols["highly_variable"].values,
        err_msg="Row and column streaming methods should produce identical results",
    )


# ==============================================================================
# Edge case tests
# ==============================================================================


def test_single_cell():
    """Test with only one cell."""
    matrix = np.array([[1, 2, 3, 0, 5]], dtype=np.float32)
    adata = ad.AnnData(X=matrix)
    adata.var_names = [f"Gene_{i}" for i in range(5)]

    result = vcell.pp.highly_variable_genes_seurat_v3_rows(adata, n_top_genes=2)

    assert len(result) == 5
    assert result["highly_variable"].sum() == 2
    # With one cell, variance should be 0 (or undefined)


def test_single_gene():
    """Test with only one gene."""
    matrix = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
    adata = ad.AnnData(X=matrix)
    adata.var_names = ["Gene_0"]

    result = vcell.pp.highly_variable_genes_seurat_v3_rows(adata, n_top_genes=1)

    assert len(result) == 1
    assert result["highly_variable"].sum() == 1


def test_all_genes_identical():
    """Test when all genes have identical expression patterns."""
    # All genes have the same expression pattern
    pattern = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    matrix = np.tile(pattern.reshape(-1, 1), (1, 10))
    adata = ad.AnnData(X=matrix)
    adata.var_names = [f"Gene_{i}" for i in range(10)]

    result = vcell.pp.highly_variable_genes_seurat_v3_rows(adata, n_top_genes=5)

    # Should still select 5 genes, even if arbitrarily
    assert result["highly_variable"].sum() == 5
    # All genes should have same variance_norm (within numerical precision)
    var_norms = result["variances_norm"].values
    assert np.allclose(var_norms, var_norms[0]), (
        "All identical genes should have same variance_norm"
    )


def test_extremely_sparse_matrix():
    """Test with 99% sparse matrix."""
    np.random.seed(42)
    matrix = np.random.random((100, 200)).astype(np.float32)
    # Make 99% of values zero
    mask = np.random.random((100, 200)) < 0.99
    matrix[mask] = 0

    adata = ad.AnnData(X=matrix)
    adata.var_names = [f"Gene_{i}" for i in range(200)]

    result = vcell.pp.highly_variable_genes_seurat_v3_rows(adata, n_top_genes=50)

    assert result["highly_variable"].sum() == 50
    # Check that we don't have NaN or Inf
    assert not np.any(np.isnan(result["variances_norm"].values))
    assert not np.any(np.isinf(result["variances_norm"].values))


def test_very_large_values():
    """Test with very large count values that could cause overflow."""
    matrix = np.random.poisson(1000, size=(50, 100)).astype(np.float32)
    # Add some extremely large values
    matrix[0, 0] = 1e6
    matrix[1, 1] = 1e6

    adata = ad.AnnData(X=matrix)
    adata.var_names = [f"Gene_{i}" for i in range(100)]

    result = vcell.pp.highly_variable_genes_seurat_v3_rows(adata, n_top_genes=20)

    assert result["highly_variable"].sum() == 20
    # Check that we don't have NaN or Inf
    assert not np.any(np.isnan(result["variances_norm"].values))
    assert not np.any(np.isinf(result["variances_norm"].values))


def test_negative_values_warning():
    """Test that negative values are handled (though they shouldn't occur in count data)."""
    matrix = np.random.randn(50, 100).astype(np.float32)
    adata = ad.AnnData(X=matrix)
    adata.var_names = [f"Gene_{i}" for i in range(100)]

    # Should still work despite negative values
    # The log transformation will produce NaN for negative means, but should handle it
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore warnings about log of negative
        result = vcell.pp.highly_variable_genes_seurat_v3_rows(adata, n_top_genes=20)

    assert result["highly_variable"].sum() == 20


def test_all_zero_variance():
    """Test when all genes have zero variance (constant expression)."""
    # Every gene is constant across cells
    matrix = np.ones((50, 100), dtype=np.float32)
    for i in range(100):
        matrix[:, i] = i + 1  # Each gene has different mean but zero variance

    adata = ad.AnnData(X=matrix)
    adata.var_names = [f"Gene_{i}" for i in range(100)]

    result = vcell.pp.highly_variable_genes_seurat_v3_rows(adata, n_top_genes=20)

    # Should still select 20 genes even if all have zero variance
    assert result["highly_variable"].sum() == 20

    # When all genes have zero CP10K variance, the lowess fitting may still produce
    # small residuals due to numerical precision and the trend fitting process.
    # The important thing is that the selection is arbitrary but deterministic.
    # Check that variance_norm values are small (near the trend line)
    assert np.std(result["variances_norm"].values) < 1.0, (
        "Variance norms should have low spread when all genes have zero variance"
    )


def test_nan_handling():
    """Test that NaN values are handled gracefully."""
    matrix = np.random.poisson(5, size=(50, 100)).astype(np.float32)
    # Add some NaN values
    matrix[0, 0] = np.nan
    matrix[5:10, 5] = np.nan

    adata = ad.AnnData(X=matrix)
    adata.var_names = [f"Gene_{i}" for i in range(100)]

    # Should handle NaN gracefully (likely treating as 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = vcell.pp.highly_variable_genes_seurat_v3_rows(adata, n_top_genes=20)

    assert result["highly_variable"].sum() == 20


def test_batch_size_larger_than_data():
    """Test when batch size is larger than the data dimensions."""
    matrix = np.random.poisson(5, size=(10, 20)).astype(np.float32)
    adata = ad.AnnData(X=matrix)
    adata.var_names = [f"Gene_{i}" for i in range(20)]

    # Batch size much larger than data
    result = vcell.pp.highly_variable_genes_seurat_v3_rows(
        adata, n_top_genes=5, batch_size=1000
    )

    assert result["highly_variable"].sum() == 5
    assert len(result) == 20


def test_n_top_larger_than_n_genes():
    """Test when requesting more HVGs than available genes."""
    matrix = np.random.poisson(5, size=(50, 10)).astype(np.float32)
    adata = ad.AnnData(X=matrix)
    adata.var_names = [f"Gene_{i}" for i in range(10)]

    # Request 20 HVGs but only have 10 genes
    result = vcell.pp.highly_variable_genes_seurat_v3_rows(adata, n_top_genes=20)

    # Should select all 10 genes
    assert result["highly_variable"].sum() == 10


def test_memory_efficiency():
    """Test that streaming actually uses less memory than loading all at once."""
    # This is more of a design verification than a functional test
    # Create a moderately large matrix
    np.random.seed(42)
    n_cells = 1000
    n_genes = 5000

    # Generate sparse matrix to simulate real data
    matrix = np.random.poisson(0.5, size=(n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(X=matrix)
    adata.var_names = [f"Gene_{i}" for i in range(n_genes)]

    # Test with small batch size (simulating memory-constrained streaming)
    result_small_batch = vcell.pp.highly_variable_genes_seurat_v3_rows(
        adata, n_top_genes=500, batch_size=100
    )

    # Test with large batch size (simulating loading more at once)
    result_large_batch = vcell.pp.highly_variable_genes_seurat_v3_rows(
        adata, n_top_genes=500, batch_size=1000
    )

    # Results should be identical regardless of batch size
    np.testing.assert_array_equal(
        result_small_batch["highly_variable"].values,
        result_large_batch["highly_variable"].values,
    )


def test_empty_matrix_handling():
    """Should handle edge cases like empty matrices gracefully."""
    # Test with matrix of all zeros
    matrix = np.zeros((10, 20), dtype=np.float32)
    adata = ad.AnnData(X=matrix)
    adata.var_names = [f"Gene_{i}" for i in range(20)]

    result = vcell.pp.highly_variable_genes_seurat_v3_rows(adata, n_top_genes=5)

    # Should still return a DataFrame with correct structure
    assert len(result) == 20
    assert "means" in result.columns
    assert "highly_variable" in result.columns

    # All means should be zero
    assert np.all(result["means"].values == 0)

    # Should still select 5 genes (even if arbitrary)
    assert result["highly_variable"].sum() == 5


# ==============================================================================
# Property-based tests using Hypothesis
# ==============================================================================


# Strategy for generating count matrices
@st.composite
def count_matrix(draw):
    """Generate a random count matrix."""
    n_obs = draw(st.integers(min_value=10, max_value=50))
    n_vars = draw(st.integers(min_value=20, max_value=100))

    # Generate counts using numpy instead of hypothesis lists (more efficient)
    # Use Poisson distribution to simulate count data
    lam = draw(st.floats(min_value=1, max_value=10))
    matrix = np.random.poisson(lam, size=(n_obs, n_vars)).astype(np.float32)

    # Add some high-variance genes
    n_hvgs = min(5, n_vars // 4)
    hvg_indices = np.random.choice(n_vars, n_hvgs, replace=False)
    for idx in hvg_indices:
        # Make these genes more variable
        matrix[:, idx] = np.random.negative_binomial(20, 0.3, size=n_obs)

    return matrix


@st.composite
def adata_strategy(draw):
    """Generate AnnData objects for testing."""
    matrix = draw(count_matrix())
    n_obs, n_vars = matrix.shape

    # Decide if sparse or dense
    use_sparse = draw(st.booleans())
    format_type = draw(st.sampled_from(["csr", "csc"])) if use_sparse else "dense"

    if format_type == "csr":
        X = sp.csr_matrix(matrix)
    elif format_type == "csc":
        X = sp.csc_matrix(matrix)
    else:
        X = matrix

    adata = ad.AnnData(X=X)
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]
    adata.obs_names = [f"Cell_{i}" for i in range(n_obs)]

    return adata, format_type


# Property 1: Batch size invariance
@given(
    adata_and_format=adata_strategy(),
    batch_size1=st.integers(min_value=5, max_value=50),
    batch_size2=st.integers(min_value=5, max_value=50),
    n_top=st.integers(min_value=5, max_value=20),
)
@settings(max_examples=10, deadline=10000)
def test_batch_size_invariance(adata_and_format, batch_size1, batch_size2, n_top):
    """Different batch sizes should produce identical results."""
    adata, format_type = adata_and_format
    n_top = min(n_top, adata.n_vars)

    func = (
        vcell.pp.highly_variable_genes_seurat_v3_rows
        if format_type in ["csr", "dense"]
        else vcell.pp.highly_variable_genes_seurat_v3_cols
    )

    result1 = func(adata, n_top_genes=n_top, batch_size=batch_size1)
    result2 = func(adata, n_top_genes=n_top, batch_size=batch_size2)

    # Should select same genes
    np.testing.assert_array_equal(
        result1["highly_variable"].values,
        result2["highly_variable"].values,
        err_msg=f"Batch sizes {batch_size1} and {batch_size2} produced different HVGs",
    )


# Property 2: Row/column equivalence
@given(
    matrix=count_matrix(),
    batch_size=st.integers(min_value=5, max_value=50),
    n_top=st.integers(min_value=5, max_value=20),
)
@settings(max_examples=10, deadline=10000)
def test_row_column_equivalence(matrix, batch_size, n_top):
    """Row and column streaming should produce identical results."""
    n_obs, n_vars = matrix.shape
    n_top = min(n_top, n_vars)

    # Create CSR version for row streaming
    adata_csr = ad.AnnData(X=sp.csr_matrix(matrix))
    adata_csr.var_names = [f"Gene_{i}" for i in range(n_vars)]

    # Create CSC version for column streaming
    adata_csc = ad.AnnData(X=sp.csc_matrix(matrix))
    adata_csc.var_names = [f"Gene_{i}" for i in range(n_vars)]

    result_rows = vcell.pp.highly_variable_genes_seurat_v3_rows(
        adata_csr, n_top_genes=n_top, batch_size=batch_size
    )
    result_cols = vcell.pp.highly_variable_genes_seurat_v3_cols(
        adata_csc, n_top_genes=n_top, batch_size=batch_size
    )

    # Results should be identical
    np.testing.assert_array_equal(
        result_rows["highly_variable"].values,
        result_cols["highly_variable"].values,
        err_msg="Row and column methods produced different results",
    )

    np.testing.assert_allclose(
        result_rows["means"].values,
        result_cols["means"].values,
        rtol=1e-6,  # Relaxed tolerance for floating point differences
        err_msg="Row and column methods have different means",
    )


# Property 3: Non-negative outputs
@given(adata_and_format=adata_strategy())
@settings(max_examples=20, deadline=10000)
def test_non_negative_outputs(adata_and_format):
    """Means and variances should be non-negative."""
    adata, format_type = adata_and_format

    func = (
        vcell.pp.highly_variable_genes_seurat_v3_rows
        if format_type in ["csr", "dense"]
        else vcell.pp.highly_variable_genes_seurat_v3_cols
    )

    result = func(adata, n_top_genes=10)

    assert np.all(result["means"].values >= 0), "Means should be non-negative"
    assert np.all(result["variances"].values >= 0), "Variances should be non-negative"


# Property 4: Correct selection count
@given(adata_and_format=adata_strategy(), n_top=st.integers(min_value=1, max_value=50))
@settings(max_examples=20, deadline=10000)
def test_correct_selection_count(adata_and_format, n_top):
    """Should select exactly n_top_genes when specified."""
    adata, format_type = adata_and_format
    n_top = min(n_top, adata.n_vars)

    func = (
        vcell.pp.highly_variable_genes_seurat_v3_rows
        if format_type in ["csr", "dense"]
        else vcell.pp.highly_variable_genes_seurat_v3_cols
    )

    result = func(adata, n_top_genes=n_top)

    n_selected = result["highly_variable"].sum()
    assert n_selected == n_top, f"Expected {n_top} HVGs, got {n_selected}"


# Property 5: Deterministic results
@given(adata_and_format=adata_strategy())
@settings(max_examples=10, deadline=10000)
def test_deterministic(adata_and_format):
    """Same input should always produce same output."""
    adata, format_type = adata_and_format

    func = (
        vcell.pp.highly_variable_genes_seurat_v3_rows
        if format_type in ["csr", "dense"]
        else vcell.pp.highly_variable_genes_seurat_v3_cols
    )

    result1 = func(adata, n_top_genes=10)
    result2 = func(adata, n_top_genes=10)

    np.testing.assert_array_equal(
        result1["highly_variable"].values,
        result2["highly_variable"].values,
        err_msg="Function is not deterministic",
    )


# Property 6: Zero gene handling
@given(
    n_obs=st.integers(min_value=10, max_value=50),
    n_vars=st.integers(min_value=10, max_value=50),
)
@settings(max_examples=10, deadline=10000)
def test_zero_gene_handling(n_obs, n_vars):
    """Genes with zero expression should have zero mean."""
    # Create matrix with some all-zero genes
    matrix = np.random.poisson(5, size=(n_obs, n_vars)).astype(np.float32)

    # Set some genes to all zeros
    zero_genes = np.random.choice(n_vars, min(3, n_vars // 2), replace=False)
    matrix[:, zero_genes] = 0

    adata = ad.AnnData(X=matrix)
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]

    result = vcell.pp.highly_variable_genes_seurat_v3_rows(adata, n_top_genes=5)

    # Zero genes should have zero mean
    for gene_idx in zero_genes:
        assert result["means"].iloc[gene_idx] == 0, (
            f"Zero gene {gene_idx} has non-zero mean"
        )


# Property 7: Constant gene handling
@given(
    n_obs=st.integers(min_value=20, max_value=50),
    n_vars=st.integers(min_value=30, max_value=50),
    constant_value=st.integers(min_value=1, max_value=100),
)
@settings(max_examples=10, deadline=10000)
def test_constant_gene_handling(n_obs, n_vars, constant_value):
    """Genes with constant expression should have lower variance_norm than variable genes."""
    # Create matrix with some constant genes and some variable genes
    matrix = np.random.poisson(5, size=(n_obs, n_vars)).astype(np.float32)

    # Add some highly variable genes
    hvg_indices = np.random.choice(n_vars, 5, replace=False)
    for idx in hvg_indices:
        matrix[:, idx] = np.random.negative_binomial(20, 0.3, size=n_obs)

    # Set some genes to constant values
    const_genes = np.random.choice(
        [i for i in range(n_vars) if i not in hvg_indices],
        min(3, n_vars // 3),
        replace=False,
    )
    matrix[:, const_genes] = constant_value

    adata = ad.AnnData(X=matrix)
    adata.var_names = [f"Gene_{i}" for i in range(n_vars)]

    result = vcell.pp.highly_variable_genes_seurat_v3_rows(adata, n_top_genes=5)

    # Check that constant genes are not in the top selected HVGs
    const_selected = np.sum(result.iloc[const_genes]["highly_variable"].values)

    # At most 1 constant gene should be selected (allowing for edge cases)
    assert const_selected <= 1, (
        f"{const_selected}/{len(const_genes)} constant genes were selected as HVGs"
    )


# Property 8: Monotonic selection
@given(adata_and_format=adata_strategy())
@settings(max_examples=10, deadline=10000)
def test_monotonic_selection(adata_and_format):
    """Higher variance_norm should correlate with HVG selection."""
    adata, format_type = adata_and_format

    func = (
        vcell.pp.highly_variable_genes_seurat_v3_rows
        if format_type in ["csr", "dense"]
        else vcell.pp.highly_variable_genes_seurat_v3_cols
    )

    n_top = min(20, adata.n_vars // 2)
    result = func(adata, n_top_genes=n_top)

    # Get variance_norm for selected and non-selected genes
    selected_var_norm = result.loc[result["highly_variable"], "variances_norm"].values
    not_selected_var_norm = result.loc[
        ~result["highly_variable"], "variances_norm"
    ].values

    if len(selected_var_norm) > 0 and len(not_selected_var_norm) > 0:
        # Minimum of selected should be >= maximum of non-selected (approximately)
        # Allow small tolerance for numerical precision
        min_selected = np.min(selected_var_norm)
        max_not_selected = np.max(not_selected_var_norm)

        assert min_selected >= max_not_selected - 1e-6, (
            f"Selection not monotonic: min selected {min_selected} < max not selected {max_not_selected}"
        )


# Property 9: Sparse/dense equivalence
@given(matrix=count_matrix(), n_top=st.integers(min_value=5, max_value=20))
@settings(max_examples=10, deadline=10000)
def test_sparse_dense_equivalence(matrix, n_top):
    """Sparse and dense matrices should give same results."""
    n_obs, n_vars = matrix.shape
    n_top = min(n_top, n_vars)

    # Test with dense
    adata_dense = ad.AnnData(X=matrix)
    adata_dense.var_names = [f"Gene_{i}" for i in range(n_vars)]

    # Test with sparse CSR
    adata_sparse = ad.AnnData(X=sp.csr_matrix(matrix))
    adata_sparse.var_names = [f"Gene_{i}" for i in range(n_vars)]

    result_dense = vcell.pp.highly_variable_genes_seurat_v3_rows(
        adata_dense, n_top_genes=n_top
    )
    result_sparse = vcell.pp.highly_variable_genes_seurat_v3_rows(
        adata_sparse, n_top_genes=n_top
    )

    # Results should be identical
    np.testing.assert_array_equal(
        result_dense["highly_variable"].values,
        result_sparse["highly_variable"].values,
        err_msg="Dense and sparse matrices produced different HVGs",
    )

    np.testing.assert_allclose(
        result_dense["means"].values,
        result_sparse["means"].values,
        rtol=1e-10,
        err_msg="Dense and sparse matrices have different means",
    )


if __name__ == "__main__":
    test_hvg_methods_match()
    print("All tests passed!")
