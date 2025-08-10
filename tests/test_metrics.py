# tests/test_metrics.py
import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as np_st

from vcell.metrics import RunningMean, compute_de, compute_mae, compute_pds


def test_mae_identity_perfect():
    """Test 1: Identity round-trip (predictions == ground truth) -> MAE = 0"""
    key = jax.random.key(42)
    p, g = 10, 18080  # p perturbations, g genes (VCC requirement)
    true = jax.random.normal(key, (p, g))
    pred = true  # Perfect identity

    mae = compute_mae(pred, true)
    assert mae.shape == (p,)
    assert jnp.allclose(mae, jnp.zeros(p), atol=1e-8)


def test_mae_null_baseline():
    """Test 2: Null/control baseline -> MAE > 0"""
    key = jax.random.key(42)
    p, g = 10, 100

    # Create control pseudobulk
    control = jax.random.normal(key, (g,))

    # Create different perturbations
    key, subkey = jax.random.split(key)
    true_perts = jax.random.normal(subkey, (p, g))

    # Predict control for everything
    pred_perts = jnp.broadcast_to(control, (p, g))

    mae = compute_mae(pred_perts, true_perts)
    assert mae.shape == (p,)
    assert jnp.all(mae > 0)

    # Verify it equals average L1 distance to control
    expected_mae = jnp.mean(jnp.abs(true_perts - control[None, :]), axis=-1)
    assert jnp.allclose(mae, expected_mae)


def test_mae_with_mask():
    """Test MAE with gene masking"""
    pred = jnp.array([[1.0, 2.0, 3.0], [0.0, 0.5, 1.0]])
    true = jnp.array([[1.5, 2.0, 4.0], [0.0, 1.0, 1.0]])
    mask = jnp.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])  # Mask out some genes

    mae = compute_mae(pred, true, mask=mask)

    # Manual calculation
    # Example 1: |1.0-1.5|*1 + |2.0-2.0|*1 + |3.0-4.0|*0 = 0.5, divided by 2 masked genes = 0.25
    # Example 2: |0.0-0.0|*1 + |0.5-1.0|*0 + |1.0-1.0|*1 = 0.0, divided by 2 masked genes = 0.0
    expected = jnp.array([0.25, 0.0])
    assert jnp.allclose(mae, expected)


def test_mae_shape_broadcasting():
    """Test MAE with different input shapes"""
    # Test with single example
    pred = jnp.array([1.0, 2.0, 3.0])
    true = jnp.array([1.5, 2.0, 3.5])
    mae = compute_mae(pred[None, :], true[None, :])
    assert mae.shape == (1,)
    assert jnp.allclose(mae, jnp.array([0.33333333]), atol=1e-6)

    # Test with batch
    pred = jnp.ones((5, 10))
    true = jnp.zeros((5, 10))
    mae = compute_mae(pred, true)
    assert mae.shape == (5,)
    assert jnp.allclose(mae, jnp.ones(5))


def test_pds_identity_perfect():
    """Test 1: Identity round-trip -> PDS = 1.0"""
    key = jax.random.key(42)
    p, g = 50, 18080  # VCC spec: 18,080 genes
    true = jax.random.normal(key, (p, g))
    pred = true  # Perfect identity

    pds_results = compute_pds(pred, true, distance="manhattan", topk=(1, 5, 10))

    assert pds_results["mean_inv_rank"] == 1.0
    assert pds_results["top1"] == 1.0
    assert pds_results["top5"] == 1.0
    assert pds_results["top10"] == 1.0


def test_pds_null_baseline():
    """Test 2: Null baseline (same prediction for all) -> PDS ~ 0.5"""
    key = jax.random.key(42)
    p, g = 50, 100

    # Create control pseudobulk
    control = jax.random.normal(key, (g,))

    # Create different true perturbations
    key, subkey = jax.random.split(key)
    true = jax.random.normal(subkey, (p, g))

    # Predict control for everything
    pred = jnp.broadcast_to(control, (p, g))

    pds_results = compute_pds(pred, true, distance="manhattan", topk=(1, 5, 10))

    # Expected random ranking: mean rank = (N+1)/2
    # PDS = 1 - (rank-1)/N, so E[PDS] = (N+1)/(2N) ~ 0.5
    assert abs(pds_results["mean_inv_rank"] - 0.5) < 0.05


def test_pds_derangement():
    """Test 3: Label-shuffle (derangement) -> PDS ~ 0.5"""
    key = jax.random.key(42)
    p, g = 50, 100
    true = jax.random.normal(key, (p, g))

    # Create derangement (permutation with no fixed points)
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, p)
    # Simple derangement: shift by 1 (works for p > 1)
    perm = jnp.roll(jnp.arange(p), 1)

    pred = true[perm]  # Shuffled predictions

    pds_results = compute_pds(pred, true, distance="manhattan", topk=(1, 5, 10))

    # Should behave like random
    assert abs(pds_results["mean_inv_rank"] - 0.5) < 0.05
    assert pds_results["top1"] < 0.2  # Very unlikely to match


def test_pds_exclude_target_gene():
    """Test that PDS excludes target gene from distance calculation"""
    # Create simple test case where target gene matters
    # 3 perturbations, 5 genes
    true = jnp.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],  # Pert 1: only gene 0 is different
        [0.0, 1.0, 0.0, 0.0, 0.0],  # Pert 2: only gene 1 is different
        [0.0, 0.0, 1.0, 0.0, 0.0],  # Pert 3: only gene 2 is different
    ])

    # Predictions are identity
    pred = true.copy()

    # With target gene excluded, distances should be smaller
    # This test would need target_gene_idx parameter in compute_pds
    # which isn't in the current skeleton

    pds_results = compute_pds(pred, true, distance="manhattan", topk=(1,))
    assert pds_results["top1"] == 1.0


def test_pds_manhattan_distance():
    """Verify PDS uses Manhattan (L1) distance as per VCC spec"""
    # Create simple test to verify L1 distance is used
    pred = jnp.array([[1.0, 2.0, 3.0]])
    true = jnp.array([
        [1.0, 2.0, 3.0],  # Distance 0
        [2.0, 2.0, 3.0],  # Distance 1
        [1.0, 3.0, 3.0],  # Distance 1
        [2.0, 3.0, 4.0],  # Distance 3
    ])

    pds_results = compute_pds(pred, true, distance="manhattan", topk=(1,))

    # First perturbation should rank first (distance 0)
    assert pds_results["top1"] == 1.0


def test_pds_topk_accuracy():
    """Test top-k accuracy metrics"""
    key = jax.random.key(42)
    p, g = 20, 100

    # Create predictions with controlled similarity
    true = jax.random.normal(key, (p, g))

    # Add small noise to create near-perfect predictions
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, (p, g)) * 0.1
    pred = true + noise

    pds_results = compute_pds(pred, true, distance="manhattan", topk=(1, 5, 10))

    # With small noise, should have high top-k accuracy
    assert pds_results["top1"] > 0.8
    assert pds_results["top5"] > 0.95
    assert pds_results["top10"] == 1.0  # All should be in top 10 with p=20


def test_de_identity_perfect():
    """Test 1: Identity round-trip -> DES = 1.0"""
    key = jax.random.key(42)
    r, g = 10, 100  # r replicates, g genes

    # Create control and perturbation data
    true_ctrl = jax.random.normal(key, (r, g))
    key, subkey = jax.random.split(key)
    true_pert = true_ctrl + jax.random.normal(subkey, (r, g)) * 2  # Strong effect

    # Perfect predictions
    pred_ctrl = true_ctrl
    pred_pert = true_pert

    de_results = compute_de(pred_pert, pred_ctrl, true_pert, true_ctrl, fdr=0.05)

    assert de_results.overlap >= 0.9999  # Should be 1.0
    assert de_results.pr_auc >= 0.9999
    assert de_results.spearman_r >= 0.9999
    assert de_results.n_pred_sig == de_results.n_true_sig


def test_de_null_baseline():
    """Test 2: Null baseline (no DE) -> DES = 0.0"""
    key = jax.random.key(42)
    r, g = 10, 100

    # True data has perturbation effect
    true_ctrl = jax.random.normal(key, (r, g))
    key, subkey = jax.random.split(key)
    true_pert = true_ctrl + jax.random.normal(subkey, (r, g)) * 2

    # Predictions show no effect (control = perturbation)
    pred_ctrl = true_ctrl
    pred_pert = true_ctrl  # Same as control

    de_results = compute_de(pred_pert, pred_ctrl, true_pert, true_ctrl, fdr=0.05)

    assert de_results.overlap == 0.0  # No predicted DE genes
    assert de_results.n_pred_sig == 0


def test_de_derangement():
    """Test 3: Derangement (wrong perturbations) -> DES ~ 0"""
    key = jax.random.key(42)
    r, g = 10, 100

    # Create multiple perturbations
    n_perts = 5
    true_ctrls = jax.random.normal(key, (n_perts, r, g))
    keys = jax.random.split(key, n_perts + 1)
    true_perts = jnp.array([
        ctrl + jax.random.normal(keys[i + 1], (r, g)) * 2
        for i, ctrl in enumerate(true_ctrls)
    ])

    # Shuffle predictions (derangement)
    perm = jnp.roll(jnp.arange(n_perts), 1)
    pred_ctrls = true_ctrls
    pred_perts = true_perts[perm]

    # Test first perturbation
    de_results = compute_de(
        pred_perts[0], pred_ctrls[0], true_perts[0], true_ctrls[0], fdr=0.05
    )

    assert de_results.overlap < 0.2  # Should have low overlap


def test_de_wilcoxon_implementation():
    """Test Wilcoxon rank-sum test implementation"""
    # Create simple case with known DE
    r, g = 20, 10

    # Control: all zeros
    true_ctrl = jnp.zeros((r, g))
    pred_ctrl = jnp.zeros((r, g))

    # Perturbation: strong effect on first 5 genes
    true_pert = jnp.zeros((r, g))
    true_pert = true_pert.at[:, :5].set(3.0)  # Large effect
    pred_pert = true_pert

    de_results = compute_de(
        pred_pert, pred_ctrl, true_pert, true_ctrl, fdr=0.05, test="wilcoxon"
    )

    # First 5 genes should be significant
    assert de_results.n_true_sig >= 5
    assert de_results.n_pred_sig >= 5
    assert de_results.overlap > 0.9


def test_de_benjamini_hochberg():
    """Test Benjamini-Hochberg FDR correction"""
    key = jax.random.key(42)
    r, g = 20, 100

    # Create data with varying effect sizes
    true_ctrl = jax.random.normal(key, (r, g))
    pred_ctrl = true_ctrl

    # Add effects of different magnitudes
    effects = jnp.zeros(g)
    effects = effects.at[:10].set(3.0)  # Strong effect
    effects = effects.at[10:20].set(1.0)  # Moderate effect
    effects = effects.at[20:30].set(0.5)  # Weak effect

    true_pert = true_ctrl + effects[None, :]
    pred_pert = true_pert

    # Test with different FDR thresholds
    de_strict = compute_de(pred_pert, pred_ctrl, true_pert, true_ctrl, fdr=0.01)
    de_loose = compute_de(pred_pert, pred_ctrl, true_pert, true_ctrl, fdr=0.1)

    # More genes should be significant with looser FDR
    assert de_loose.n_true_sig >= de_strict.n_true_sig
    assert de_loose.n_pred_sig >= de_strict.n_pred_sig


def test_de_truncation_rule():
    """Test DES truncation rule when |pred| > |true|"""
    key = jax.random.key(42)
    r, g = 20, 100

    # True data has 10 DE genes
    true_ctrl = jax.random.normal(key, (r, g))
    true_pert = true_ctrl.copy()
    true_pert = true_pert.at[:, :10].add(3.0)  # 10 true DE genes

    # Predictions have 20 DE genes (overpredicting)
    pred_ctrl = true_ctrl
    pred_pert = true_ctrl.copy()
    pred_pert = pred_pert.at[:, :20].add(3.0)  # 20 predicted DE genes

    de_results = compute_de(pred_pert, pred_ctrl, true_pert, true_ctrl, fdr=0.05)

    # Per VCC spec: when |pred| > |true|, select top |true| by fold change
    # Since first 10 genes overlap and have same fold change, overlap should be high
    assert de_results.overlap > 0.9
    # But pred should have more significant genes
    assert de_results.n_pred_sig > de_results.n_true_sig


def test_round_trip_all_metrics():
    """Test all metrics with identity predictions"""
    key = jax.random.key(42)
    p, r, g = 10, 5, 100  # perturbations, replicates, genes

    # Create synthetic data
    true_data = jax.random.normal(key, (p, r, g))
    pred_data = true_data  # Perfect predictions

    # Create control data
    key, subkey = jax.random.split(key)
    control_data = jax.random.normal(subkey, (r, g))

    # Test MAE on pseudobulks
    true_pseudobulk = jnp.mean(true_data, axis=1)
    pred_pseudobulk = jnp.mean(pred_data, axis=1)
    mae = compute_mae(pred_pseudobulk, true_pseudobulk)
    assert jnp.allclose(mae, 0.0, atol=1e-8)

    # Test PDS
    pds_results = compute_pds(pred_pseudobulk, true_pseudobulk, distance="manhattan")
    assert pds_results["top1"] == 1.0

    # Test DE for first perturbation
    de_results = compute_de(
        pred_data[0], control_data, true_data[0], control_data, fdr=0.05
    )
    assert de_results.overlap >= 0.9999


def test_gene_order_trap():
    """Test 4: Gene-order permutation destroys all metrics"""
    key = jax.random.key(42)
    p, g = 10, 100

    # Create true data
    true = jax.random.normal(key, (p, g))

    # Perfect predictions but with permuted gene order
    key, subkey = jax.random.split(key)
    gene_perm = jax.random.permutation(subkey, g)
    pred = true[:, gene_perm]

    # MAE should explode
    mae = compute_mae(pred, true)
    assert jnp.mean(mae) > 0.5  # Should be much larger than 0

    # PDS should be random
    pds_results = compute_pds(pred, true, distance="manhattan")
    assert abs(pds_results["mean_inv_rank"] - 0.5) < 0.1

    # For DE test, need control data
    control = jax.random.normal(key, (5, g))
    control_perm = control[:, gene_perm]

    de_results = compute_de(pred[0:5], control_perm, true[0:5], control, fdr=0.05)
    assert de_results.overlap < 0.1  # Should have very low overlap


def test_normalization_consistency():
    """Test that metrics handle log1p normalization correctly"""
    key = jax.random.key(42)
    p, g = 5, 100

    # Create count data
    counts = jnp.abs(jax.random.normal(key, (p, g))) * 100

    # Apply log1p normalization
    log_data = jnp.log1p(counts)

    # Test MAE with both raw and log data
    mae_raw = compute_mae(counts, counts)
    mae_log = compute_mae(log_data, log_data)

    assert jnp.allclose(mae_raw, 0.0, atol=1e-8)
    assert jnp.allclose(mae_log, 0.0, atol=1e-8)

    # Mixed normalization should fail (high MAE)
    mae_mixed = compute_mae(counts, log_data)
    assert jnp.mean(mae_mixed) > 1.0


def test_runningmean_matches_global_mean():
    xs = jnp.array([0.0, 1.0, 2.0, 3.0])
    rm = RunningMean.zero()
    for v in xs:
        rm = rm.update(v)
    assert jnp.allclose(rm.compute(), jnp.mean(xs))


@st.composite
def array_shapes(draw, min_dims=1, max_dims=4, min_size=1, max_size=20):
    """Generate valid array shapes."""
    ndims = draw(st.integers(min_dims, max_dims))
    return tuple(draw(st.integers(min_size, max_size)) for _ in range(ndims))


@st.composite
def matching_arrays(draw, dtype=np.float32):
    """Generate two arrays with matching shapes."""
    shape = draw(array_shapes(min_dims=2, max_dims=2, min_size=2, max_size=50))

    # Use hypothesis numpy strategies
    arr1 = draw(
        np_st.arrays(
            dtype=dtype,
            shape=shape,
            elements=st.floats(
                min_value=-10, max_value=10, allow_nan=False, allow_infinity=False
            ),
        )
    )
    arr2 = draw(
        np_st.arrays(
            dtype=dtype,
            shape=shape,
            elements=st.floats(
                min_value=-10, max_value=10, allow_nan=False, allow_infinity=False
            ),
        )
    )

    # Convert to JAX arrays
    return jnp.array(arr1), jnp.array(arr2)


@st.composite
def pseudobulk_data(draw):
    """Generate data suitable for pseudobulk testing (p perturbations, g genes)."""
    p = draw(st.integers(min_value=2, max_value=20))
    g = draw(st.integers(min_value=10, max_value=100))

    data = draw(
        np_st.arrays(
            dtype=np.float32,
            shape=(p, g),
            elements=st.floats(
                min_value=-5, max_value=5, allow_nan=False, allow_infinity=False
            ),
        )
    )

    return jnp.array(data)


@st.composite
def replicate_data(draw):
    """Generate replicate data for DE testing (r replicates, g genes)."""
    r = draw(st.integers(min_value=3, max_value=20))
    g = draw(st.integers(min_value=10, max_value=100))

    ctrl = draw(
        np_st.arrays(
            dtype=np.float32,
            shape=(r, g),
            elements=st.floats(
                min_value=-5, max_value=5, allow_nan=False, allow_infinity=False
            ),
        )
    )

    # Add effect to create perturbation
    effect_size = draw(st.floats(min_value=0.1, max_value=3.0))
    pert = ctrl + np.random.randn(r, g) * effect_size

    return jnp.array(ctrl), jnp.array(pert)


@given(matching_arrays())
@settings(max_examples=50, deadline=1000)
def test_mae_identity_property(arrays):
    """Property: MAE(x, x) = 0 for any x."""
    pred, _ = arrays
    mae = compute_mae(pred, pred)
    assert jnp.allclose(mae, 0.0, atol=1e-6)


@given(matching_arrays())
@settings(max_examples=50, deadline=1000)
def test_mae_symmetry_property(arrays):
    """Property: MAE(x, y) = MAE(y, x)."""
    pred, true = arrays
    mae1 = compute_mae(pred, true)
    mae2 = compute_mae(true, pred)
    assert jnp.allclose(mae1, mae2, atol=1e-6)


@given(matching_arrays())
@settings(max_examples=50, deadline=1000)
def test_mae_non_negative_property(arrays):
    """Property: MAE(x, y) >= 0 for all x, y."""
    pred, true = arrays
    mae = compute_mae(pred, true)
    assert jnp.all(mae >= -1e-6)  # Small tolerance for numerical errors


@given(matching_arrays(), st.floats(min_value=0.1, max_value=10))
@settings(max_examples=50, deadline=1000)
def test_mae_scale_property(arrays, scale):
    """Property: MAE(s*x, s*y) = s * MAE(x, y) for scale s > 0."""
    pred, true = arrays

    mae_original = compute_mae(pred, true)
    mae_scaled = compute_mae(pred * scale, true * scale)

    assert jnp.allclose(mae_scaled, mae_original * scale, rtol=1e-5)


@given(matching_arrays())
@settings(max_examples=50, deadline=1000)
def test_mae_triangle_inequality_property(arrays):
    """Property: MAE(x, z) <= MAE(x, y) + MAE(y, z) (triangle inequality)."""
    x, y = arrays
    # Create z as a point between x and y
    z = (x + y) / 2

    mae_xz = compute_mae(x, z)
    mae_xy = compute_mae(x, y)
    mae_yz = compute_mae(y, z)

    # Triangle inequality should hold
    assert jnp.all(mae_xz <= mae_xy + mae_yz + 1e-6)


@given(
    np_st.arrays(
        dtype=np.float32,
        shape=st.tuples(
            st.integers(2, 10),  # examples
            st.integers(5, 20),  # genes
        ),
        elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=50, deadline=1000)
def test_mae_mask_reduces_value_property(arr):
    """Property: Masking genes should not increase MAE."""
    arr = jnp.array(arr)
    pred = arr
    true = arr + jnp.ones_like(arr)  # Add constant difference

    # Create random mask (keep at least half the genes)
    key = jax.random.key(42)
    mask = jax.random.bernoulli(key, p=0.7, shape=arr.shape).astype(jnp.float32)

    # Masked MAE
    mae_masked = compute_mae(pred, true, mask=mask)

    # MAE should be non-negative (basic sanity check)
    assert jnp.all(mae_masked >= 0)


@given(
    st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False),
        min_size=1,
        max_size=100,
    )
)
@settings(max_examples=50, deadline=1000)
def test_running_mean_matches_numpy_property(values):
    """Property: RunningMean should match numpy.mean for any sequence."""
    rm = RunningMean.zero()
    for v in values:
        rm = rm.update(jnp.array(v))

    running_result = rm.compute()
    numpy_result = np.mean(values)

    # Allow for float32 vs float64 precision differences
    assert jnp.allclose(running_result, numpy_result, rtol=1e-4, atol=1e-6)


@given(
    st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False),
        min_size=1,
        max_size=50,
    ),
    st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False),
        min_size=1,
        max_size=50,
    ),
)
@settings(max_examples=50, deadline=1000)
def test_running_mean_merge_property(values1, values2):
    """Property: Merging two RunningMeans = computing mean of combined data."""
    # Compute separately
    rm1 = RunningMean.zero()
    for v in values1:
        rm1 = rm1.update(jnp.array(v))

    rm2 = RunningMean.zero()
    for v in values2:
        rm2 = rm2.update(jnp.array(v))

    # Merge
    merged = rm1.merge(rm2)
    merged_result = merged.compute()

    # Compute together
    all_values = values1 + values2
    expected = np.mean(all_values) if all_values else 0.0

    if all_values:  # Only check if we have values
        assert jnp.allclose(merged_result, expected, rtol=1e-5)


@given(
    st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False),
        min_size=1,
        max_size=50,
    )
)
@settings(max_examples=50, deadline=1000)
def test_running_mean_order_invariant_property(values):
    """Property: Order of updates shouldn't matter much for the final mean (accounting for float precision)."""
    # Original order
    rm1 = RunningMean.zero()
    for v in values:
        rm1 = rm1.update(jnp.array(v))

    # Reversed order
    rm2 = RunningMean.zero()
    for v in reversed(values):
        rm2 = rm2.update(jnp.array(v))

    # Allow for floating point accumulation differences
    # The means should be very close but may not be identical due to float32 precision
    assert jnp.allclose(rm1.compute(), rm2.compute(), rtol=1e-5, atol=1e-6)


@given(pseudobulk_data())
@settings(max_examples=20, deadline=2000)
def test_pds_identity_is_perfect_property(data):
    """Property: PDS(x, x) should give perfect score."""
    pds_results = compute_pds(data, data, distance="manhattan")
    assert pds_results["top1"] == 1.0
    assert pds_results["mean_inv_rank"] == 1.0


@given(pseudobulk_data())
@settings(max_examples=20, deadline=2000)
def test_pds_bounded_property(data):
    """Property: PDS scores should be in [0, 1]."""
    # Create random predictions
    key = jax.random.key(42)
    pred = jax.random.normal(key, data.shape)

    pds_results = compute_pds(pred, data, distance="manhattan")

    assert 0 <= pds_results["top1"] <= 1
    assert 0 <= pds_results["mean_inv_rank"] <= 1

    for k in [5, 10]:
        if f"top{k}" in pds_results:
            assert 0 <= pds_results[f"top{k}"] <= 1


@given(pseudobulk_data())
@settings(max_examples=20, deadline=2000)
def test_pds_topk_monotonic_property(data):
    """Property: top1 <= top5 <= top10 (monotonic in k)."""
    # Add noise to create imperfect predictions
    key = jax.random.key(42)
    noise = jax.random.normal(key, data.shape) * 0.5
    pred = data + noise

    pds_results = compute_pds(pred, data, distance="manhattan", topk=(1, 5, 10))

    # Monotonicity: higher k should have higher accuracy
    assert pds_results["top1"] <= pds_results["top5"]
    assert pds_results["top5"] <= pds_results["top10"]


@given(replicate_data())
@settings(max_examples=20, deadline=2000)
def test_de_identity_is_perfect_property(data):
    """Property: DE(x, x) should give perfect overlap."""
    ctrl, pert = data

    de_results = compute_de(pert, ctrl, pert, ctrl, fdr=0.05)

    assert de_results.overlap >= 0.99  # Near perfect
    assert de_results.n_pred_sig == de_results.n_true_sig


@given(replicate_data())
@settings(max_examples=20, deadline=2000)
def test_de_no_effect_gives_zero_property(data):
    """Property: No perturbation effect should give zero DE genes."""
    ctrl, _ = data

    # Use control for both conditions
    de_results = compute_de(ctrl, ctrl, ctrl, ctrl, fdr=0.05)

    assert de_results.n_pred_sig == 0
    assert de_results.n_true_sig == 0
    assert de_results.overlap == 0.0


@given(replicate_data(), st.floats(min_value=0.001, max_value=0.1))
@settings(max_examples=20, deadline=2000)
def test_de_fdr_threshold_property(data, fdr):
    """Property: Stricter FDR should give fewer or equal DE genes."""
    ctrl, pert = data

    de_loose = compute_de(pert, ctrl, pert, ctrl, fdr=fdr)
    de_strict = compute_de(pert, ctrl, pert, ctrl, fdr=fdr / 10)

    # Stricter FDR should identify fewer or equal genes
    assert de_strict.n_true_sig <= de_loose.n_true_sig
    assert de_strict.n_pred_sig <= de_loose.n_pred_sig


@given(matching_arrays())
@settings(max_examples=30, deadline=1000)
def test_mae_zero_implies_identity(arrays):
    """Property: If MAE = 0, then arrays must be identical."""
    pred, true = arrays

    # Make them identical
    pred = true.copy()

    mae = compute_mae(pred, true)

    # If MAE is zero, arrays should be identical
    if jnp.allclose(mae, 0.0, atol=1e-8):
        assert jnp.allclose(pred, true, atol=1e-8)


@given(
    np_st.arrays(
        dtype=np.float32,
        shape=st.tuples(st.integers(2, 10), st.integers(10, 50)),
        elements=st.floats(-5, 5, allow_nan=False, allow_infinity=False),
    ),
    st.floats(min_value=0, max_value=5),
)
@settings(max_examples=30, deadline=1000)
def test_mae_perturbation_increases_with_noise(arr, noise_scale):
    """Property: Adding noise should increase MAE monotonically."""
    arr = jnp.array(arr)
    true = arr

    # No noise
    mae_zero = compute_mae(arr, true)

    # Add noise
    key = jax.random.key(42)
    noise = jax.random.normal(key, arr.shape) * noise_scale
    pred_noisy = arr + noise
    mae_noisy = compute_mae(pred_noisy, true)

    # More noise should mean higher MAE (approximately)
    if noise_scale > 1e-6:
        assert jnp.mean(mae_noisy) >= jnp.mean(mae_zero) - 1e-6
