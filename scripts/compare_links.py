#!/usr/bin/env python3
"""
Compare Bayesian ECI fits that differ only in their link function.

Each fit must have been saved with ``fit_eci.py --link <link> --save-trace``.
Everything here is computed from the saved traces by PSIS-LOO, with no
refitting: the expected log predictive density (ELPD) of each fit and its
paired difference from a reference fit, and leave-one-out calibration plots.

Usage:
    python scripts/compare_links.py logit=outputs/trace \\
        probit=outputs/links/probit/trace stukel=outputs/links/stukel/trace \\
        --output-dir outputs/link_comparison

The first fit listed is the reference the others are compared against.
"""

import argparse
from pathlib import Path

import arviz as az
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
from scipy import optimize, stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eci.evaluation import _flatten, _posterior_mu, loo_diagnostics, sampler_diagnostics  # noqa: E402
from eci.links import STUKEL_PARAM, link_mean  # noqa: E402

# Pareto k above this marks an observation whose PSIS-LOO estimate is unreliable.
K_THRESHOLD = 0.7
# NUTS's tree depth limit (PyMC's default); a move that reaches it was cut short.
MAX_TREE_DEPTH = 10
# The sampling checks a fit must pass before its other metrics mean anything:
# column, whether a value passes, and the rule as text.
SAMPLING_CHECKS = {
    "divergences": (lambda v: v == 0, "0"),
    "max_rhat": (lambda v: v <= 1.01, "<= 1.01"),
    "min_ess_bulk": (lambda v: v >= 400, ">= 400"),
    "min_ess_tail": (lambda v: v >= 400, ">= 400"),
    "min_bfmi": (lambda v: v > 0.3, "> 0.3"),
    "share_at_max_tree_depth": (lambda v: v == 0, "0"),
}
MU_EPS = 1e-6

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"
REFERENCE = "#8a8983"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY, "text.color": INK,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load_fit(trace_dir: Path) -> dict:
    """Everything the comparison needs from one saved fit."""
    trace = xr.open_datatree(trace_dir / "trace.nc")
    df = pd.read_csv(trace_dir / "fit_data.csv")
    summary, per_obs, weights = loo_diagnostics(trace, df)
    per_obs = per_obs.sort_index()

    y = trace["observed_data"]["obs"].to_numpy()
    mu = np.clip(_posterior_mu(trace, df), MU_EPS, 1 - MU_EPS)
    bench = list(trace["posterior"].coords["benchmark"].values)
    kappa = _flatten(trace["posterior"]["kappa"])[:, df["benchmark"].map(bench.index).to_numpy()]
    # Leave-one-out predictive mean and PIT: the posterior expectation of the
    # Beta's mean and CDF, reweighted as if the observation had been left out.
    loo_mean = (weights * mu.T).sum(axis=1)
    loo_pit = (weights * stats.beta.cdf(y, mu * kappa, (1 - mu) * kappa).T).sum(axis=1)

    capability = _flatten(trace["posterior"]["capability"])
    models = list(trace["posterior"].coords["model"].values)
    sampler = sampler_diagnostics(trace)[0].iloc[0]
    tree_depth = trace["sample_stats"]["tree_depth"].to_numpy()
    fit = {
        "df": df, "y": y, "elpd_i": per_obs["elpd_i"].to_numpy(),
        "pareto_k": per_obs["pareto_k"].to_numpy(), "loo_mean": loo_mean, "loo_pit": loo_pit,
        "summary": summary.iloc[0], "models": models, "capability": capability,
        "link": trace["posterior"].attrs.get("link", "logit"),
        "draws_per_chain": int(trace["posterior"].sizes["draw"]),
        "divergences": int(sampler["divergences"]),
        "max_rhat": float(sampler["max_r_hat"]),
        "min_ess_bulk": float(sampler["min_ess_bulk"]),
        "min_ess_tail": float(sampler["min_ess_tail"]),
        "min_bfmi": float(sampler["min_bfmi"]),
        "max_tree_depth": int(tree_depth.max()),
        "share_at_max_tree_depth": float((tree_depth >= MAX_TREE_DEPTH).mean()),
        "max_abs_capability": float(np.abs(capability).max()),
    }
    if STUKEL_PARAM in trace["posterior"]:
        fit["alpha"] = _flatten(trace["posterior"][STUKEL_PARAM])
    return fit


def eci_means(fit: dict, low: str, high: str) -> pd.Series:
    """Posterior mean ECI per model, each draw anchored at 130 / 150."""
    cap = fit["capability"]
    lo, hi = cap[:, [fit["models"].index(low)]], cap[:, [fit["models"].index(high)]]
    return pd.Series((130 + 20 * (cap - lo) / (hi - lo)).mean(axis=0), index=fit["models"])


def clipped_at_bounds(y: np.ndarray, eps: float = 0.01) -> bool:
    """Whether the scores pile up at a clip just inside 0 and 1."""
    return bool(y.min() < eps and y.max() > 1 - eps and (y == y.min()).sum() + (y == y.max()).sum() > 2)


def comparison_table(fits: dict, anchors: tuple[str, str]) -> pd.DataFrame:
    reference = next(iter(fits.values()))
    n = len(reference["y"])
    reliable = np.all([f["pareto_k"] <= K_THRESHOLD for f in fits.values()], axis=0)
    # Scores the fit clipped off 0 or 1: the Beta density is steepest there,
    # so a few of them can carry much of an ELPD difference.
    y = reference["y"]
    interior = (y > y.min() + 1e-12) & (y < y.max() - 1e-12) if clipped_at_bounds(y) else np.ones(n, bool)
    ref_eci = eci_means(reference, *anchors)
    rows = []
    for name, fit in fits.items():
        diff = fit["elpd_i"] - reference["elpd_i"]
        eci = eci_means(fit, *anchors)
        resid = fit["y"] - fit["loo_mean"]
        row = {
            "link": name,
            "elpd_loo": fit["summary"]["elpd_loo"],
            "se": fit["summary"]["se"],
            "p_loo": fit["summary"]["p_loo"],
            "elpd_diff": diff.sum(),
            "diff_se": np.sqrt(n * diff.var(ddof=1)),
            "elpd_diff_reliable_k": diff[reliable].sum(),
            "diff_se_reliable_k": np.sqrt(reliable.sum() * diff[reliable].var(ddof=1)),
            "elpd_diff_interior": diff[interior].sum(),
            "diff_se_interior": np.sqrt(interior.sum() * diff[interior].var(ddof=1)),
            "n_k_above_0.7": int((fit["pareto_k"] > K_THRESHOLD).sum()),
            "max_k": fit["pareto_k"].max(),
            "loo_rmse": np.sqrt((resid**2).mean()),
            "loo_pit_ks": stats.kstest(fit["loo_pit"], "uniform").statistic,
            "draws_per_chain": fit["draws_per_chain"],
            **{check: fit[check] for check in SAMPLING_CHECKS},
            "max_tree_depth": fit["max_tree_depth"],
            "sampling_ok": all(passes(fit[check]) for check, (passes, _) in SAMPLING_CHECKS.items()),
            "max_abs_capability": fit["max_abs_capability"],
            "eci_rmsd_vs_ref": np.sqrt(((eci - ref_eci) ** 2).mean()),
            "eci_max_abs_diff_vs_ref": (eci - ref_eci).abs().max(),
            "eci_spearman_vs_ref": eci.corr(ref_eci, method="spearman"),
        }
        if "alpha" in fit:
            for i, tail in enumerate(("upper", "lower")):
                lo, mid, hi = np.quantile(fit["alpha"][:, i], [0.05, 0.5, 0.95])
                row[f"alpha_{tail}"] = f"{mid:.2f} ({lo:.2f}, {hi:.2f})"
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("elpd_loo", ascending=False)
    table.attrs["n_reliable"] = int(reliable.sum())
    table.attrs["n_interior"] = int(interior.sum())
    return table


def _grid(n: int, width: float = 3.3, height: float = 2.7):
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows),
                             sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    return fig, axes.ravel()[:n]


def plot_elpd(table: pd.DataFrame, reference: str, path: Path) -> None:
    """ELPD difference from the reference, with +/- 1 and 2 SE of the paired difference."""
    t = table[table["link"] != reference].sort_values("elpd_diff")
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(t) + 1.5))
    pos = np.arange(len(t))
    ax.axvline(0, color=REFERENCE, lw=1)
    ax.hlines(pos, t["elpd_diff"] - 2 * t["diff_se"], t["elpd_diff"] + 2 * t["diff_se"], color=SERIES, lw=1.2)
    ax.hlines(pos, t["elpd_diff"] - t["diff_se"], t["elpd_diff"] + t["diff_se"], color=SERIES, lw=3.5)
    ax.plot(t["elpd_diff"], pos, "o", color=SERIES, ms=8, mec=SURFACE, mew=1.5)
    for y, (d, se) in enumerate(zip(t["elpd_diff"], t["diff_se"])):
        ax.annotate(f"{d:+.1f} (SE {se:.1f})", (d, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(pos, t["link"])
    ax.set_ylim(-0.6, len(t) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"ELPD (PSIS-LOO) minus the {reference} fit's; right of 0 predicts better")
    ax.set_title(f"Predictive accuracy of each link against {reference}")
    note = "Thick bar: ±1 SE of the paired difference; thin bar: ±2 SE."
    if t["link"].str.endswith("*").any():
        note += "\n* failed the sampling checks: indicative only."
    ax.text(0, -0.36 if len(t) < 4 else -0.3, note, va="top",
            transform=ax.transAxes, fontsize=8, color=INK_SECONDARY)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_elpd_by_score(fits: dict, reference: str, path: Path, bins: int = 10) -> None:
    """Where on the score scale each link gains or loses ELPD against the reference."""
    ref = fits[reference]
    others = [n for n in fits if n != reference]
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(ref["y"], edges) - 1, 0, bins - 1)
    fig, axes = _grid(len(others))
    for ax, name in zip(axes, others):
        diff = fits[name]["elpd_i"] - ref["elpd_i"]
        totals = np.array([diff[which == b].sum() for b in range(bins)])
        ax.bar(edges[:-1] + 0.05, totals, width=0.085, color=SERIES)
        ax.axhline(0, color=REFERENCE, lw=1)
        ax.set_title(f"{name}  ({diff.sum():+.1f} in total)")
        ax.grid(axis="x", visible=False)
    fig.supxlabel("Observed score (bins of 0.1)", fontsize=9, color=INK_SECONDARY)
    fig.supylabel(f"ELPD minus {reference}, summed over the bin", fontsize=9, color=INK_SECONDARY)
    fig.suptitle(f"Where each link predicts better (above 0) or worse than {reference}", fontsize=11, fontweight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loo_pit(fits: dict, path: Path) -> None:
    """LOO-PIT ECDF minus the uniform's: a flat line at 0 is perfect calibration."""
    fig, axes = _grid(len(fits))
    for ax, (name, fit) in zip(axes, fits.items()):
        pit = np.sort(fit["loo_pit"])
        n = len(pit)
        band = 1.358 / np.sqrt(n)  # 95% Kolmogorov-Smirnov band
        ax.fill_between([0, 1], -band, band, color=GRID, lw=0)
        ax.axhline(0, color=REFERENCE, lw=1)
        ax.plot(pit, np.arange(1, n + 1) / n - pit, color=SERIES, lw=1.6)
        ax.set_title(f"{name}  (KS {stats.kstest(pit, 'uniform').statistic:.3f})")
    fig.supxlabel("LOO-PIT: where each score falls in its leave-one-out predictive distribution",
                  fontsize=9, color=INK_SECONDARY)
    fig.supylabel("ECDF minus uniform", fontsize=9, color=INK_SECONDARY)
    fig.suptitle("Leave-one-out calibration: the line should stay inside the grey 95% band", fontsize=11, fontweight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loo_residuals(fits: dict, path: Path, bins: int = 10) -> None:
    """Mean LOO residual against the LOO predictive mean: misfit of the curve's shape."""
    fig, axes = _grid(len(fits))
    edges = np.linspace(0, 1, bins + 1)
    for ax, (name, fit) in zip(axes, fits.items()):
        which = np.clip(np.digitize(fit["loo_mean"], edges) - 1, 0, bins - 1)
        resid = fit["y"] - fit["loo_mean"]
        x, mean, se = [], [], []
        for b in range(bins):
            r = resid[which == b]
            if len(r) >= 5:
                x.append(fit["loo_mean"][which == b].mean())
                mean.append(r.mean())
                se.append(r.std(ddof=1) / np.sqrt(len(r)))
        x, mean, se = map(np.array, (x, mean, se))
        ax.axhline(0, color=REFERENCE, lw=1)
        ax.vlines(x, mean - 2 * se, mean + 2 * se, color=SERIES, lw=1.2)
        ax.plot(x, mean, "o-", color=SERIES, lw=1.6, ms=5, mec=SURFACE, mew=1)
        ax.set_title(f"{name}  (LOO RMSE {np.sqrt((resid**2).mean()):.4f})")
    fig.supxlabel("Leave-one-out predicted score (bins of 0.1)", fontsize=9, color=INK_SECONDARY)
    fig.supylabel("Observed minus predicted, bin mean ±2 SE", fontsize=9, color=INK_SECONDARY)
    fig.suptitle("Leave-one-out bias along the curve: points should sit on 0", fontsize=11, fontweight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _standardised(link: str, x: np.ndarray, alpha=(None, None)) -> np.ndarray:
    """The link's curve, shifted and scaled to cross 0.5 at x = 0 with the
    logistic's slope there, so that only its shape is left to compare."""
    f = lambda eta: link_mean(link, np.asarray(eta, dtype=float), *alpha)  # noqa: E731
    centre = optimize.brentq(lambda e: f(e) - 0.5, -20, 20)
    slope = (f(centre + 1e-5) - f(centre - 1e-5)) / 2e-5
    return f(centre + 0.25 / slope * x)


def plot_link_shapes(fits: dict, reference: str, path: Path) -> None:
    others = [n for n in fits if n != reference]
    x = np.linspace(-8, 8, 321)
    ref_curve = _standardised(fits[reference]["link"], x)
    fig, axes = _grid(len(others))
    for ax, name in zip(axes, others):
        fit = fits[name]
        ax.plot(x, ref_curve, color=REFERENCE, lw=1.4, label=reference)
        if "alpha" in fit:
            keep = fit["alpha"][np.linspace(0, len(fit["alpha"]) - 1, 400).astype(int)]
            curves = np.array([_standardised(fit["link"], x, tuple(a)) for a in keep])
            lo, hi = np.quantile(curves, [0.05, 0.95], axis=0)
            ax.fill_between(x, lo, hi, color=SERIES, alpha=0.25, lw=0)
            curve = _standardised(fit["link"], x, tuple(np.median(fit["alpha"], axis=0)))
        else:
            curve = _standardised(fit["link"], x)
        ax.plot(x, curve, color=SERIES, lw=2, label=name)
        ax.set_title(name)
        ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.supxlabel("Capability minus difficulty, rescaled so every curve has the same slope at 0.5",
                  fontsize=9, color=INK_SECONDARY)
    fig.supylabel("Expected score", fontsize=9, color=INK_SECONDARY)
    fig.suptitle(f"Shape of each link's curve against {reference} (stukel: posterior median, 90% band)", fontsize=11, fontweight="bold")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fits", nargs="+", metavar="NAME=TRACE_DIR",
                        help="A label and the directory holding its trace.nc and fit_data.csv; "
                             "the first is the reference")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/link_comparison"))
    parser.add_argument("--anchors", nargs=2, default=["Claude 3.5 Sonnet", "GPT-5"],
                        metavar=("LOW", "HIGH"), help="Models pinned at ECI 130 and 150")
    args = parser.parse_args()

    fits = {}
    for spec in args.fits:
        name, _, directory = spec.partition("=")
        print(f"Loading {name} from {directory}...")
        fits[name] = load_fit(Path(directory))
    # Mark fits that fail a sampling check: their numbers are indicative only.
    fits = {
        name if all(passes(fit[c]) for c, (passes, _) in SAMPLING_CHECKS.items()) else f"{name}*": fit
        for name, fit in fits.items()
    }
    if any(name.endswith("*") for name in fits):
        print("* failed a sampling check; its draws do not describe the posterior reliably")
    reference = next(iter(fits))
    for name, fit in fits.items():
        same = fit["df"][["Model", "benchmark"]].equals(fits[reference]["df"][["Model", "benchmark"]])
        if not (same and np.allclose(fit["y"], fits[reference]["y"])):
            raise ValueError(f"{name} was not fitted to the same observations as {reference}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = comparison_table(fits, tuple(args.anchors))
    table.to_csv(args.output_dir / "link_comparison.csv", index=False)
    pd.DataFrame({
        "Model": fits[reference]["df"]["Model"], "benchmark": fits[reference]["df"]["benchmark"],
        "y": fits[reference]["y"],
        **{f"{stat}_{name}": fit[stat] for name, fit in fits.items()
           for stat in ("elpd_i", "pareto_k", "loo_mean", "loo_pit")},
    }).to_csv(args.output_dir / "pointwise.csv", index=False)

    plot_elpd(table, reference, args.output_dir / "elpd_difference.png")
    plot_elpd_by_score(fits, reference, args.output_dir / "elpd_by_score.png")
    plot_loo_pit(fits, args.output_dir / "loo_pit.png")
    plot_loo_residuals(fits, args.output_dir / "loo_residuals.png")
    plot_link_shapes(fits, reference, args.output_dir / "link_shapes.png")

    print("\nSampling checks (a fit that fails one should be fixed before its other metrics are read):")
    checks = table.set_index("link")[["draws_per_chain", *SAMPLING_CHECKS, "max_tree_depth", "sampling_ok"]]
    print(checks.rename(columns={c: f"{c} ({rule})" for c, (_, rule) in SAMPLING_CHECKS.items()}).T.to_string())
    print()
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
    print(f"\n{table.attrs['n_reliable']} of {len(fits[reference]['y'])} observations have "
          f"Pareto k <= {K_THRESHOLD} in every fit (the *_reliable_k columns); "
          f"{table.attrs['n_interior']} were not clipped at 0 or 1 (the *_interior columns)")
    print(f"Wrote the table, pointwise values and five figures to {args.output_dir}")


if __name__ == "__main__":
    main()
