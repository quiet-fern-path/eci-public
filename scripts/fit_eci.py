#!/usr/bin/env python3
"""
Fit ECI model and save scores to outputs directory. 

Defaults to the Bayesian fit using pymc, but can also run the 
least-squares fit (deprecated) for reproducing old results.

Usage:
    python scripts/fit_eci.py
    python scripts/fit_eci.py --input path/to/benchmark_data.zip
    python scripts/fit_eci.py --evaluate
    python scripts/fit_eci.py --save-trace
    python scripts/fit_eci.py --method least-squares
"""

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from eci import (
    DEFAULT_ANCHOR_ECI_HIGH,
    DEFAULT_ANCHOR_ECI_LOW,
    DEFAULT_ANCHOR_MODEL_HIGH,
    DEFAULT_ANCHOR_MODEL_LOW,
    fit_bayesian_eci,
    fit_eci_model,
    prepare_benchmark_data,
    save_evaluation,
    summarise_evaluation,
)
from eci.dataloader import BENCHMARK_DATA_URL
from eci.links import LINKS


OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Input columns carried over to the ECI output when present (the fit itself
# returns bare statistical results)
MODEL_METADATA_COLS = ["date", "Organization", "model_version", "source"]

# Per-benchmark columns written out, in order, when the fit produces them
EDI_VALUE_COLS = ["edi", "discriminability_scaled", "sigma"]

# Packages whose versions are recorded next to a saved trace
RECORDED_PACKAGES = ["eci", "pymc", "pytensor", "arviz", "numpy", "scipy", "pandas", "xarray"]


def save_trace(draws: dict, df: pd.DataFrame, args, output_dir: Path) -> list[Path]:
    """Write everything predictive checks and PSIS-LOO need to ``output_dir``.

    - trace.nc: the full trace, every draw of every chain. Its groups are
      posterior, sample_stats, log_likelihood (the pointwise log likelihood
      PSIS-LOO reweights), posterior_predictive (one replicated dataset per
      draw) and observed_data (the clipped scores the likelihood saw).
    - fit_data.csv: the fitted rows in the trace's observation order, so row
      i is observation i of the pointwise arrays.
    - run_info.json: the sampler settings, each run's diagnostics, the anchor
      models and package versions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "trace.nc"
    draws["trace"].to_netcdf(trace_path)

    data_path = output_dir / "fit_data.csv"
    df.reset_index(drop=True).rename_axis("obs").to_csv(data_path)

    posterior = draws["trace"].posterior
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": str(args.input),
        "n_observations": len(df),
        "n_models": len(draws["model_names"]),
        "n_benchmarks": len(draws["benchmark_names"]),
        "chains": int(posterior.sizes["chain"]),
        "draws_per_chain": int(posterior.sizes["draw"]),
        "target_accept": args.target_accept,
        "seed": args.seed,
        "link": draws["link"],
        "capability_bound": draws["capability_bound"],
        "started_from": draws["started_from"],
        "attempts": draws["attempts"],
        "anchors": {
            DEFAULT_ANCHOR_MODEL_LOW: DEFAULT_ANCHOR_ECI_LOW,
            DEFAULT_ANCHOR_MODEL_HIGH: DEFAULT_ANCHOR_ECI_HIGH,
        },
        "python": platform.python_version(),
        "packages": {name: version(name) for name in RECORDED_PACKAGES},
    }
    info_path = output_dir / "run_info.json"
    info_path.write_text(json.dumps(info, indent=2) + "\n")
    return [trace_path, data_path, info_path]


def join_model_metadata(eci_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Attach per-model metadata columns from the input data, first value wins."""
    cols = [c for c in MODEL_METADATA_COLS if c in df.columns]
    if not cols:
        return eci_df
    metadata = df.drop_duplicates("model_id").set_index("model_id")[cols]
    return eci_df.join(metadata, on="model_id")


def main():
    parser = argparse.ArgumentParser(description="Fit ECI model and save scores")
    parser.add_argument(
        "--input",
        default=BENCHMARK_DATA_URL,
        help=f"Path or URL of benchmark_data.zip (default: {BENCHMARK_DATA_URL})",
    )
    parser.add_argument(
        "--method",
        choices=["bayesian", "least-squares"],
        default="bayesian",
        help="Fitting method: Bayesian MCMC with posterior credible intervals "
             "(default), or least squares with bootstrap CIs (deprecated/"
             "historical - kept for reproducing old results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for scores (default: {OUTPUT_DIR})",
    )

    bayesian = parser.add_argument_group("bayesian options")
    bayesian.add_argument(
        "--draws", type=int, default=None,
        help="Posterior draws kept per chain, fixed for every run (default: "
             "start small and add draws until the fit passes the convergence "
             "checks)",
    )
    bayesian.add_argument(
        "--tune", type=int, default=None,
        help="Tuning steps per chain, discarded (default: 500 with the dense "
             "mass matrix, 1000 without)",
    )
    bayesian.add_argument(
        "--chains", type=int, default=4, help="Number of chains (default: 4)",
    )
    bayesian.add_argument(
        "--target-accept", type=float, default=0.95,
        help="NUTS target acceptance rate; (default: 0.95)",
    )
    bayesian.add_argument(
        "--seed", type=int, default=12345, help="Sampler seed (default: 12345)",
    )
    bayesian.add_argument(
        "--link", choices=LINKS, default="logit",
        help="Curve mapping discriminability * (capability - difficulty) to "
             "the mean score; stukel learns two tail-shape parameters "
             "(default: logit)",
    )
    bayesian.add_argument(
        "--capability-bound", type=float, default=50.0,
        help="Capabilities and difficulties are uniform on [-bound, bound] "
             "on the raw scale; heavier-tailed links may need more (default: 50)",
    )
    bayesian.add_argument(
        "--evaluate",
        action="store_true",
        help="Run the diagnostic suite (convergence, accuracy, LOO, "
             "calibration, posterior predictive checks) alongside the fit"
             "and write it as a PDF report to <output-dir>/evaluation/",
    )
    bayesian.add_argument(
        "--save-trace",
        action="store_true",
        help="Save the full trace, with the pointwise log likelihood and "
             "posterior predictive draws, to <output-dir>/trace/ so predictive "
             "checks and PSIS-LOO can be rerun later. Implies --evaluate",
    )

    least_squares = parser.add_argument_group("least-squares options")
    least_squares.add_argument(
        "--bootstrap-samples",
        type=int,
        default=500,
        help="Number of bootstrap samples for confidence intervals (default: 500)",
    )
    least_squares.add_argument(
        "--numeric-jacobian",
        action="store_true",
        help="Use numerical Jacobian instead of analytical (slower)",
    )

    args = parser.parse_args()
    # The pointwise log likelihood and the replicated datasets are only
    # computed for the evaluation
    args.evaluate = args.evaluate or args.save_trace

    if args.evaluate and args.method != "bayesian":
        parser.error(
            "--evaluate scores a posterior, so it needs --method bayesian; "
            "the least-squares fit has no diagnostic suite"
        )

    print(f"Preparing benchmark data from {args.input}...")
    df = prepare_benchmark_data(args.input)
    print(f"  Loaded {len(df)} performance records")
    print(f"  {df['model_id'].nunique()} models, {df['benchmark_id'].nunique()} benchmarks")

    if args.method == "bayesian":
        print(f"\nFitting Bayesian IRT model ({args.link} link, {args.chains} chains, "
              f"{args.draws or 'adaptive'} draws, {args.tune or 'default'} tuning steps)...")
        eci_df, edi_df, draws = fit_bayesian_eci(
            df,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
            random_seed=args.seed,
            link=args.link,
            capability_bound=args.capability_bound,
            evaluate=args.evaluate,
        )
    else:
        jacobian_type = "numerical" if args.numeric_jacobian else "analytical"
        print(f"\nFitting IRT model ({jacobian_type} Jacobian, "
              f"{args.bootstrap_samples} bootstrap samples)...")
        eci_df, edi_df, _ = fit_eci_model(
            df,
            bootstrap_samples=args.bootstrap_samples,
            use_analytical_jacobian=not args.numeric_jacobian,
        )

    eci_df = join_model_metadata(eci_df, df)

    # Prepare output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Save ECI scores
    eci_output = args.output_dir / "eci_scores.csv"
    eci_cols = ["Model", "eci"]
    if "eci_ci_low" in eci_df.columns:
        eci_cols += ["eci_ci_low", "eci_ci_high"]
    eci_cols += [col for col in MODEL_METADATA_COLS if col in eci_df.columns]
    eci_df[eci_cols].to_csv(eci_output, index=False)
    print(f"\nSaved ECI scores to {eci_output}")

    # Save EDI scores
    if "benchmark_release_date" in df.columns:
        release_dates = df.drop_duplicates("benchmark_id").set_index("benchmark_id")["benchmark_release_date"]
        edi_df = edi_df.join(release_dates, on="benchmark_id")
    edi_output = args.output_dir / "edi_scores.csv"
    edi_cols = ["benchmark", "edi", "discriminability_scaled"]
    if "benchmark_release_date" in edi_df.columns:
        edi_cols.append("benchmark_release_date")
    # is_anchor is least-squares only (the Bayesian fit pins no benchmark);
    # sigma is Bayesian only
    edi_cols += [c for c in ("is_anchor", "sigma") if c in edi_df.columns]
    edi_cols += [
        f"{name}_{bound}"
        for name in EDI_VALUE_COLS
        for bound in ("ci_low", "ci_high")
        if f"{name}_ci_low" in edi_df.columns
    ]
    edi_df[edi_cols].to_csv(edi_output, index=False)
    print(f"Saved EDI scores to {edi_output}")

    if args.method == "bayesian" and args.evaluate:
        evaluation = draws["evaluation"]
        written = save_evaluation(evaluation, args.output_dir / "evaluation")
        for path in written:
            print(f"\nSaved evaluation report to {path}")
        print("\n=== Evaluation ===")
        print(summarise_evaluation(evaluation))

    if args.method == "bayesian" and args.save_trace:
        for path in save_trace(draws, df, args, args.output_dir / "trace"):
            print(f"Saved {path}")

    # Print summary
    print("\n=== Top 10 Models by ECI ===")
    print(eci_df[["Model", "eci"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
