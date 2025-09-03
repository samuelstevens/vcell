Module vcell.metrics
====================

Functions
---------

`compute_de(pred_pert: jaxtyping.Float[Array, 'r g'], pred_ctrl: jaxtyping.Float[Array, 'r g'], true_pert: jaxtyping.Float[Array, 'r g'], true_ctrl: jaxtyping.Float[Array, 'r g'], *, fdr: float = 0.05, test: Literal['wilcoxon'] = 'wilcoxon', two_sided: bool = True) ‑> vcell.metrics.DEDetails`
:   Differential-expression agreement (skeleton).
    
    Intent:
      1) Make DE calls (pred vs ctrl) and (true vs ctrl) with the same test+FDR.
      2) Compare sets/ranks: overlap (e.g., Jaccard/F1), PR-AUC, Spearman on |logFC| or signed logFC.
    
    Notes:
      - Callers should pass consistent normalisation/gene order.
      - r = number of 'replicates' per condition (can be 1 if using pseudobulks).
    
    Returns:
      DEDetails(overlap=..., pr_auc=..., spearman_r=..., n_true_sig=..., n_pred_sig=...)

`compute_mae(pred: jaxtyping.Float[Array, '... g'], true: jaxtyping.Float[Array, '... g'], *, mask: jaxtyping.Float[Array, '... g'] | None = None) ‑> jaxtyping.Float[Array, '...']`
:   Per-example MAE across genes. Reduces over the last (gene) axis only.
    
    Shapes:
      pred, true: [..., g]
      mask (optional): same shape; 1.0 keeps a gene, 0.0 drops it.
    Returns:
      mae: [...]  (one scalar per leading example/perturbation index)

`compute_pds(pred: jaxtyping.Float[Array, 'p g'], true: jaxtyping.Float[Array, 'p g'], *, topk: tuple[int, ...] = (1, 5, 10)) ‑> dict[str, jaxtyping.Float[Array, '']]`
:   Perturbation Discrimination Score.
    
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

Classes
-------

`DEDetails(overlap: float, pr_auc: float, spearman_r: float, n_true_sig: int, n_pred_sig: int)`
:   Immutable result packet for DE agreement metrics.

    ### Ancestors (in MRO)

    * equinox._module._module.Module
    * collections.abc.Hashable

    ### Instance variables

    `n_pred_sig: int`
    :

    `n_true_sig: int`
    :

    `overlap: float`
    :   e.g., Jaccard/F1 over true-significant genes.

    `pr_auc: float`
    :   PR-AUC of predicted DE vs true DE.

    `spearman_r: float`
    :   rank corr of (signed) logFC.

`RunningMean(total: jaxtyping.Float[Array, ''], count: jaxtyping.Float[Array, ''])`
:   Numerically stable running mean of scalar values.
    
    Note: prefer returning a *new* instance rather than in-place mutation
    to keep things functional/JAX-friendly.

    ### Ancestors (in MRO)

    * equinox._module._module.Module
    * collections.abc.Hashable

    ### Static methods

    `zero() ‑> vcell.metrics.RunningMean`
    :

    ### Instance variables

    `count: jaxtyping.Float[Array, '']`
    :

    `total: jaxtyping.Float[Array, '']`
    :

    ### Methods

    `compute(self) ‑> jaxtyping.Float[Array, '']`
    :

    `merge(self, other: RunningMean) ‑> vcell.metrics.RunningMean`
    :

    `update(self, value: jaxtyping.Float[Array, ''], weight: jaxtyping.Float[Array, ''] | int = 1) ‑> vcell.metrics.RunningMean`
    :