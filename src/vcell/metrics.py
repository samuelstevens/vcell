# src/vcell/metrics.py
import typing as tp

import beartype
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped

__all__ = ["DEDetails", "RunningMean", "compute_mae", "compute_pds", "compute_de"]


@jaxtyped(typechecker=beartype.beartype)
class DEDetails(eqx.Module):
    """Immutable-ish result packet for DE agreement metrics."""

    overlap: float  # e.g., Jaccard/F1 over true-significant genes
    pr_auc: float  # PR-AUC of predicted DE vs true DE
    spearman_r: float  # rank corr of (signed) logFC
    n_true_sig: int
    n_pred_sig: int


@jaxtyped(typechecker=beartype.beartype)
class RunningMean(eqx.Module):
    """Numerically stable running mean of scalar values.

    Note: prefer returning a *new* instance rather than in-place mutation
    to keep things functional/JAX-friendly.
    """

    total: Float[Array, ""]
    count: Float[Array, ""]

    @staticmethod
    @jaxtyped(typechecker=beartype.beartype)
    def zero() -> "RunningMean":
        return RunningMean(total=jnp.array(0.0), count=jnp.array(0.0))

    @jaxtyped(typechecker=beartype.beartype)
    def update(
        self, value: Float[Array, ""], weight: Float[Array, ""] | int = 1
    ) -> "RunningMean":
        total = self.total + value * jnp.asarray(weight, dtype=self.total.dtype)
        count = self.count + jnp.asarray(weight, dtype=self.count.dtype)
        return RunningMean(total=total, count=count)

    @jaxtyped(typechecker=beartype.beartype)
    def merge(self, other: "RunningMean") -> "RunningMean":
        return RunningMean(
            total=self.total + other.total, count=self.count + other.count
        )

    @jaxtyped(typechecker=beartype.beartype)
    def compute(self) -> Float[Array, ""]:
        # Returns NaN if count==0; caller should guard or accept NaN to signal "empty".
        return self.total / self.count


@jaxtyped(typechecker=beartype.beartype)
def compute_mae(
    pred: Float[Array, "... g"],
    true: Float[Array, "... g"],
    *,
    mask: Float[Array, "... g"] | None = None,
) -> Float[Array, "..."]:
    """Per-example MAE across genes. Reduces over the last (gene) axis only.

    Shapes:
      pred, true: [..., g]
      mask (optional): same shape; 1.0 keeps a gene, 0.0 drops it.
    Returns:
      mae: [...]  (one scalar per leading example/perturbation index)
    """
    diff = jnp.abs(pred - true)
    if mask is not None:
        # Avoid divide-by-zero: normalise by sum(mask) along gene axis.
        masked = diff * mask
        denom = jnp.clip(jnp.sum(mask, axis=-1), a_min=1e-12)
        return jnp.sum(masked, axis=-1) / denom
    return jnp.mean(diff, axis=-1)


@jaxtyped(typechecker=beartype.beartype)
def compute_pds(
    pred_by_pert: Float[Array, "p g"],
    true_by_pert: Float[Array, "p g"],
    *,
    distance: tp.Literal["cosine", "euclidean"] = "cosine",
    topk: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Perturbation Discrimination Score (skeleton).

    Intent:
      For each perturbation i, rank true profiles by distance to pred[i].
      Report mean inverse rank and top-k accuracy.

    Returns:
      {
        "mean_inv_rank": float,
        "top1": float,
        "top5": float,
        ...
      }
    """
    raise NotImplementedError(
        "compute_pds is a skeleton; implement ranking + reductions."
    )


@jaxtyped(typechecker=beartype.beartype)
def compute_de(
    pred_pert: Float[Array, "r g"],
    pred_ctrl: Float[Array, "r g"],
    true_pert: Float[Array, "r g"],
    true_ctrl: Float[Array, "r g"],
    *,
    fdr: float = 0.05,
    test: tp.Literal["wilcoxon"] = "wilcoxon",
    two_sided: bool = True,
) -> DEDetails:
    """Differential-expression agreement (skeleton).

    Intent:
      1) Make DE calls (pred vs ctrl) and (true vs ctrl) with the same test+FDR.
      2) Compare sets/ranks: overlap (e.g., Jaccard/F1), PR-AUC, Spearman on |logFC| or signed logFC.

    Notes:
      - Callers should pass consistent normalisation/gene order.
      - r = number of 'replicates' per condition (can be 1 if using pseudobulks).

    Returns:
      DEDetails(overlap=..., pr_auc=..., spearman_r=..., n_true_sig=..., n_pred_sig=...)
    """
    raise NotImplementedError(
        "compute_de is a skeleton; implement DE test, BH-FDR, and comparisons."
    )
