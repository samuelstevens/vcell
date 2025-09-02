import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.interpolate
    import statsmodels.nonparametric.smoothers_lowess
    from jaxtyping import Float

    return Float, np, plt, scipy, statsmodels


@app.cell
def _(np):
    cached = np.load(
        "/Volumes/samuel-stevens-2TB/datasets/nourreddine2025/cached/KOLF_Pan_Genome_Aggregate.npz",
    )
    weighted_g = cached["weighted_g"]
    squared_g = cached["squared_g"]
    det_g = cached["det_g"]
    n_cells = cached["n_cells"]
    return det_g, n_cells, squared_g, weighted_g


@app.cell
def _(n_cells, squared_g, weighted_g):
    mu_g = weighted_g / n_cells
    var_g = (squared_g - n_cells * mu_g * mu_g) / (n_cells - 1)
    return mu_g, var_g


@app.cell
def _(f, keep, keep_idx, mu_g, np, plt, var_g, xs, ys):
    log_residuals = np.log(var_g + 1e-12) - f(np.log(mu_g + 1e-12))
    rank = np.argsort(log_residuals[keep])
    n_hvgs = 200
    hvgs = keep_idx[rank[-n_hvgs:]]
    rest = np.setdiff1d(np.arange(mu_g.size), hvgs, assume_unique=False)

    fig, ax = plt.subplots(dpi=600, layout="constrained")
    ax.scatter(
        np.log(mu_g[hvgs]),
        np.log(var_g[hvgs]),
        color="tab:green",
        alpha=0.5,
        marker="+",
        # linewidth=0,
        label="HVGs",
    )
    ax.scatter(
        np.log(mu_g[rest]),
        np.log(var_g[rest]),
        color="tab:blue",
        alpha=0.5,
        marker=".",
        linewidth=0,
        label="Rest",
    )
    ax.plot(xs, ys, color="tab:orange", alpha=0.9, label=r"$f(\log(\mu))$")
    ax.set_xlabel("log(gene mean)")
    ax.set_ylabel("log(gene variance)")
    ax.set_title(f"Top {n_hvgs} Highly Variable Genes")
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend()
    fig
    return


@app.cell
def _(keep, mu_g, np, scipy, statsmodels, var_g):
    def fit_trend_lowess(mu, var, frac=0.3, it=2, eps=1e-12):
        """
        Fit f : log_var ~= f(log_mu) using robust LOWESS, then return a callable f(x).
        """
        x = np.log(mu + eps)
        y = np.log(var + eps)

        # Sort by x for stability
        order = np.argsort(x)
        xs, ys = x[order], y[order]

        # Robust lowess (it>0 does robust reweighting to downweight outliers/HVGs)
        smooth = statsmodels.nonparametric.smoothers_lowess.lowess(
            endog=ys, exog=xs, frac=frac, it=it, return_sorted=True
        )
        xs_s, ys_s = smooth[:, 0], smooth[:, 1]

        # Remove near-duplicate x to keep interpolator happy
        keep = np.concatenate(([True], np.diff(xs_s) > 1e-12))
        xs_s, ys_s = xs_s[keep], ys_s[keep]

        # Shape-preserving monotone interpolator (good extrapolation behavior too)
        interpolator = scipy.interpolate.PchipInterpolator(xs_s, ys_s, extrapolate=True)

        def f(x_new):
            return interpolator(np.asarray(x_new, float))

        return f, xs_s, ys_s

    # Example usage:
    # f, xs, ys = fit_trend_lowess(mu_g, var_g, weights=det_g)  # if you have detection counts
    # log_residuals = np.log(var_g + 1e-12) - f(np.log(mu_g + 1e-12))
    f, xs, ys = fit_trend_lowess(mu_g[keep], var_g[keep])
    return f, xs, ys


@app.cell
def _(Float, det_g, n_cells, np):
    def detection_mask(
        det_g: Float[np.ndarray, "n_genes"],
        n_cells: int,
        min_detect_frac: float = 0.001,  # 0.1%
        min_detect_count: int = 5,
    ) -> np.ndarray:
        thr = max(min_detect_count, int(np.ceil(min_detect_frac * n_cells)))
        return det_g >= thr

    keep = detection_mask(det_g, n_cells, min_detect_frac=0.0001, min_detect_count=3)
    keep_idx = np.flatnonzero(keep)
    len(keep_idx), len(keep)
    return keep, keep_idx


@app.cell
def _(np):
    np.e**-10
    return


@app.cell
def _():
    4.5 / 100_000
    return


if __name__ == "__main__":
    app.run()
