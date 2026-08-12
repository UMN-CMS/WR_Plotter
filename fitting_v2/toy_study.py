#!/usr/bin/env python3
"""
Toy pseudodata study: generate N toys from MC background, fit S+B, extract
signal strength distribution.

Each toy histogram is generated using a Poisson-Gamma compound that accounts
for both MC statistical uncertainty (per-bin sumw2) and Poisson counting
statistics, producing integer event counts like real data.

Usage:
    # Quick test (10 toys, ee only):
    python fitting_v2/toy_study.py \\
        --era RunIII2024Summer24 \\
        --dir 20260223_nlo_dy \\
        --signal WR2000_N1100 \\
        --channel ee --n-toys 10

    # Full run (100 toys, both channels):
    python fitting_v2/toy_study.py \\
        --era RunIII2024Summer24 \\
        --dir 20260223_nlo_dy \\
        --signal WR2000_N1100 \\
        --n-toys 100
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from scipy.stats import norm

# Add repo root to path so wrplotter and fitting_v2 imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import ROOT
except ImportError:
    sys.exit(
        "ERROR: PyROOT is not available. Please set up a ROOT-enabled environment.\n"
        "  Options: cmsenv (CMSSW), conda install root, or source LCG views."
    )

from wrplotter.cli_utils import setup_logging
from wrplotter.config import load_lumi
from wrplotter.paths import input_dirs_for_era, repo_root

from fitting_v2.fit_utils import (
    add_common_args,
    build_region_name,
    build_hist_key,
    load_and_sum_backgrounds,
    load_signal,
    create_roodatahist,
    run_fit,
    extract_fit_results,
)
from fitting_v2.fit_background import (
    build_sb_model,
    discover_window_json,
    load_window_json,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Toy generation
# ---------------------------------------------------------------------------

def generate_toy_histogram(
    mc_hist: ROOT.TH1D,
    rng: np.random.Generator,
    name: str = "h_toy",
) -> ROOT.TH1D:
    """Generate one toy pseudodata histogram using Poisson-Gamma compound.

    For each bin:
      1. Extract MC predicted rate mu_i and uncertainty sigma_i
      2. Compute effective counts k_eff = mu_i^2 / sigma_i^2 and scale
         theta = sigma_i^2 / mu_i
      3. Sample true rate: lambda_i ~ Gamma(shape=k_eff, scale=theta)
      4. Sample observed count: n_i ~ Poisson(lambda_i)

    Returns a TH1D with integer bin contents and sqrt(n) errors.
    """
    toy = mc_hist.Clone(name)
    toy.SetDirectory(0)
    toy.Reset()

    n_bins = mc_hist.GetNbinsX()
    for i in range(1, n_bins + 1):
        content = mc_hist.GetBinContent(i)
        error = mc_hist.GetBinError(i)

        if content <= 0 or error <= 0:
            toy.SetBinContent(i, 0)
            toy.SetBinError(i, 0)
            continue

        k_eff = content**2 / error**2
        theta = error**2 / content

        lam = rng.gamma(shape=k_eff, scale=theta)
        n = rng.poisson(lam)

        toy.SetBinContent(i, float(n))
        toy.SetBinError(i, np.sqrt(float(n)))

    # Zero out under/overflow
    toy.SetBinContent(0, 0)
    toy.SetBinError(0, 0)
    toy.SetBinContent(n_bins + 1, 0)
    toy.SetBinError(n_bins + 1, 0)

    return toy


# ---------------------------------------------------------------------------
# Single toy fit
# ---------------------------------------------------------------------------

def run_single_toy(
    toy_hist: ROOT.TH1D,
    observable: ROOT.RooRealVar,
    bkg_model: str,
    sig_hist: ROOT.TH1D,
    toy_index: int,
) -> dict:
    """Fit one toy pseudodata with the S+B model.

    Returns dict with fit status, parameter values, and convergence flag.
    """
    data_hist = create_roodatahist(toy_hist, observable, name=f"toy_{toy_index}")
    n_total = data_hist.sumEntries()

    if n_total < 1:
        return {
            "toy_index": toy_index,
            "fit_status": -1,
            "cov_quality": -1,
            "edm": -1,
            "n_sig": 0.0,
            "n_sig_error": 0.0,
            "n_bkg": 0.0,
            "c": 0.0,
            "converged": False,
            "n_events": n_total,
        }

    # Build fresh S+B model — n_sig initialized at 0 (bkg-only hypothesis)
    sb_pdf, params, n_sig_expected = build_sb_model(
        observable, bkg_model, n_total, sig_hist, inject_mu=0.0,
    )

    fit_result = run_fit(sb_pdf, data_hist, params, print_level=-1)
    fit_results = extract_fit_results(fit_result, params)

    status = fit_results["fit_status"]["status"]
    cov_qual = fit_results["fit_status"]["cov_quality"]

    result = {
        "toy_index": toy_index,
        "fit_status": status,
        "cov_quality": cov_qual,
        "edm": fit_results["fit_status"]["edm"],
        "n_sig": params["n_sig"].getVal(),
        "n_sig_error": params["n_sig"].getError(),
        "n_bkg": params["n_bkg"].getVal() if "n_bkg" in params else 0.0,
        "c": params["c"].getVal() if "c" in params else 0.0,
        "converged": (status == 0 and cov_qual == 3),
        "n_events": n_total,
        "n_sig_expected": n_sig_expected,
    }
    return result


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary_stats(values: np.ndarray) -> dict:
    """Compute summary statistics for an array of fitted values."""
    if len(values) == 0:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0,
                "q16": 0, "q84": 0, "q2_5": 0, "q97_5": 0, "n": 0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "q16": float(np.percentile(values, 16)),
        "q84": float(np.percentile(values, 84)),
        "q2_5": float(np.percentile(values, 2.5)),
        "q97_5": float(np.percentile(values, 97.5)),
        "n": len(values),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_distribution(
    values: np.ndarray,
    xlabel: str,
    title: str,
    output_path: Path,
    *,
    n_bins: int = 50,
    vline: float | None = None,
    vline_label: str | None = None,
) -> None:
    """Plot a 1D histogram of fitted parameter values with summary stats."""
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(values, bins=n_bins, histtype="stepfilled", alpha=0.6,
            edgecolor="black", linewidth=0.8)

    if vline is not None:
        ax.axvline(vline, color="red", linestyle="--", linewidth=1.5,
                   label=vline_label or f"x = {vline}")
        ax.legend(fontsize=14)

    # Stats box
    stats_text = (
        f"Entries: {len(values)}\n"
        f"Mean: {np.mean(values):.3f}\n"
        f"Std: {np.std(values):.3f}\n"
        f"Median: {np.median(values):.3f}"
    )
    ax.text(
        0.97, 0.97, stats_text, transform=ax.transAxes,
        fontsize=12, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Toys", fontsize=16)
    ax.set_title(title, fontsize=16)

    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    logger.info("Saved distribution plot: %s", output_path.stem)


def plot_pull_distribution(
    pulls: np.ndarray,
    output_path: Path,
    *,
    n_bins: int = 50,
) -> None:
    """Plot pull distribution with Gaussian overlay."""
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(8, 6))

    counts, bin_edges, _ = ax.hist(
        pulls, bins=n_bins, histtype="stepfilled", alpha=0.6,
        edgecolor="black", linewidth=0.8, density=True,
    )

    # Standard normal overlay
    x = np.linspace(pulls.min() - 0.5, pulls.max() + 0.5, 200)
    ax.plot(x, norm.pdf(x), "r-", linewidth=2, label=r"$\mathcal{N}(0, 1)$")

    # Fit Gaussian to pulls
    pull_mean = np.mean(pulls)
    pull_std = np.std(pulls)

    stats_text = (
        f"Entries: {len(pulls)}\n"
        f"Mean: {pull_mean:.3f}\n"
        f"Std: {pull_std:.3f}"
    )
    ax.text(
        0.97, 0.97, stats_text, transform=ax.transAxes,
        fontsize=12, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    ax.set_xlabel("Pull: " + r"$n_{\mathrm{sig}} / \sigma_{n_{\mathrm{sig}}}$",
                   fontsize=16)
    ax.set_ylabel("Normalized", fontsize=16)
    ax.set_title("Pull distribution (bkg-only hypothesis)", fontsize=16)
    ax.legend(fontsize=14)

    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(output_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    logger.info("Saved pull plot: %s", output_path.stem)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Toy pseudodata study: generate N toys from MC background, "
            "fit S+B model, extract signal strength distribution."
        )
    )
    add_common_args(parser)

    parser.add_argument(
        "--n-toys", type=int, default=100,
        help="Number of toy datasets (default: 100)",
    )
    parser.add_argument(
        "--model",
        choices=["single-exp", "double-exp"],
        default="single-exp",
        help="Background model (default: single-exp)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if not args.signal:
        sys.exit(
            "ERROR: --signal is required.\n"
            "  Example: --signal WR2000_N1100"
        )

    ROOT.gROOT.SetBatch(True)
    # Suppress all RooFit messages below ERROR for toy loop
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)

    # --- Resolve input path ---
    info = load_lumi(args.era)
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    input_dir = input_dirs[0]
    logger.info("Input directory: %s", input_dir)

    # --- Output directory ---
    signal_tag = args.signal
    base_output_dir = Path(
        args.output_dir
        or str(
            repo_root() / "fitting_v2" / "outputs" / args.era
            / "toy_study" / signal_tag
        )
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    channels = [args.channel] if args.channel else ["ee", "mumu"]

    for channel in channels:
        logger.info("=" * 60)
        logger.info(
            "Toy study: %s  channel=%s  model=%s  n_toys=%d  seed=%d",
            signal_tag, channel, args.model, args.n_toys, args.seed,
        )

        # --- Load signal window ---
        win_json_path = discover_window_json(args.era, args.signal, channel)
        win_data = load_window_json(win_json_path)
        sig_stats = win_data["signal_stats"]
        fit_window = win_data["fit_window"]
        fit_lo = fit_window["lo"]
        fit_hi = fit_window["hi"]

        logger.info(
            "Signal: mean=%.1f GeV, RMS=%.1f GeV",
            sig_stats["mean"], sig_stats["rms"],
        )
        logger.info("Fit window: [%.1f, %.1f] GeV", fit_lo, fit_hi)

        # --- Build histogram key ---
        region = build_region_name(channel, args.topology)
        mass_var = (
            "mass_twoobject" if args.topology == "boosted"
            else "mass_fourobject"
        )
        hist_key = build_hist_key(region, mass_var)

        # --- Load MC background ---
        bkg_hist = load_and_sum_backgrounds(input_dir, hist_key)
        rebinned = bkg_hist.Rebin(args.rebin, f"h_bkg_{channel}_rebinned")
        bin_width_gev = rebinned.GetBinWidth(1)
        logger.info(
            "MC background: %d bins (%.0f GeV/bin), %.1f events total",
            rebinned.GetNbinsX(), bin_width_gev, rebinned.Integral(),
        )

        # --- Load signal MC ---
        sig_hist_raw = load_signal(input_dir, hist_key, signal_tag)
        sig_rebinned = sig_hist_raw.Rebin(
            args.rebin, f"h_sig_{channel}_rebinned",
        )

        # --- Define observable ---
        mass = ROOT.RooRealVar("mass", r"m_{lljj}", fit_lo, fit_hi, "GeV")

        # --- Compute n_sig_expected ---
        tmp_sig_dh = create_roodatahist(sig_rebinned, mass, name="tmp_sig")
        n_sig_expected = tmp_sig_dh.sumEntries()
        logger.info("Signal template: %.1f events in window (at mu=1)",
                     n_sig_expected)

        # --- Run toys ---
        rng = np.random.default_rng(args.seed)
        results_list = []
        n_converged = 0

        # Suppress fit_utils logger during toy loop (it logs per-fit)
        fit_utils_logger = logging.getLogger("fitting_v2.fit_utils")
        fit_utils_level = fit_utils_logger.level
        fit_utils_logger.setLevel(logging.WARNING)

        for i in range(args.n_toys):
            if i == 0 or (i + 1) % 10 == 0:
                logger.info("Toy %d / %d", i + 1, args.n_toys)

            toy_hist = generate_toy_histogram(
                rebinned, rng, name=f"h_toy_{i}",
            )
            result = run_single_toy(
                toy_hist, mass, args.model, sig_rebinned, i,
            )

            # Compute mu = n_sig / n_sig_expected
            if n_sig_expected > 0:
                result["mu_fitted"] = result["n_sig"] / n_sig_expected
                result["mu_error"] = result["n_sig_error"] / n_sig_expected
            else:
                result["mu_fitted"] = 0.0
                result["mu_error"] = 0.0

            if result["converged"]:
                n_converged += 1
            results_list.append(result)

        # Restore fit_utils logger
        fit_utils_logger.setLevel(fit_utils_level)

        n_failed = args.n_toys - n_converged
        logger.info(
            "Completed: %d converged, %d failed", n_converged, n_failed,
        )

        # --- Aggregate results ---
        converged_mask = np.array([r["converged"] for r in results_list])
        all_n_sig = np.array([r["n_sig"] for r in results_list])
        all_n_sig_err = np.array([r["n_sig_error"] for r in results_list])
        all_mu = np.array([r["mu_fitted"] for r in results_list])

        conv_n_sig = all_n_sig[converged_mask]
        conv_n_sig_err = all_n_sig_err[converged_mask]
        conv_mu = all_mu[converged_mask]

        # Pulls: (n_sig - 0) / sigma, guard against zero error
        valid_err_mask = conv_n_sig_err > 0
        pulls = conv_n_sig[valid_err_mask] / conv_n_sig_err[valid_err_mask]

        # --- Save JSON ---
        summary = {
            "metadata": {
                "era": args.era,
                "channel": channel,
                "topology": args.topology,
                "signal_tag": signal_tag,
                "model": args.model,
                "n_toys": args.n_toys,
                "seed": args.seed,
                "n_converged": n_converged,
                "n_failed": n_failed,
                "fit_window": [fit_lo, fit_hi],
                "n_sig_expected": n_sig_expected,
                "bin_width_gev": bin_width_gev,
                "timestamp": datetime.now().isoformat(),
            },
            "summary_stats": {
                "n_sig": compute_summary_stats(conv_n_sig),
                "n_sig_error": compute_summary_stats(conv_n_sig_err),
                "mu": compute_summary_stats(conv_mu),
                "pulls": compute_summary_stats(pulls),
            },
            "toys": results_list,
        }

        json_path = base_output_dir / f"toys_{channel}.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved toy results: %s", json_path)

        # --- Summary plots ---
        if n_converged > 0:
            plot_distribution(
                conv_n_sig,
                xlabel=r"$n_{\mathrm{sig}}$",
                title=f"Fitted signal yield ({channel}, {signal_tag})",
                output_path=base_output_dir / f"dist_n_sig_{channel}.pdf",
                vline=0.0, vline_label="Bkg-only truth",
            )
            plot_distribution(
                conv_mu,
                xlabel=r"$\mu = n_{\mathrm{sig}} / n_{\mathrm{sig}}^{\mathrm{exp}}$",
                title=f"Fitted signal strength ({channel}, {signal_tag})",
                output_path=base_output_dir / f"dist_mu_{channel}.pdf",
                vline=0.0, vline_label="Bkg-only truth",
            )
            plot_distribution(
                conv_n_sig_err,
                xlabel=r"$\sigma_{n_{\mathrm{sig}}}$",
                title=f"Signal yield uncertainty ({channel}, {signal_tag})",
                output_path=base_output_dir / f"dist_n_sig_err_{channel}.pdf",
            )

        if len(pulls) > 5:
            plot_pull_distribution(
                pulls,
                output_path=base_output_dir / f"pulls_{channel}.pdf",
            )

        # --- Console summary ---
        logger.info("=" * 60)
        logger.info(
            "Toy study: %s  %s  %s", signal_tag, channel, args.model,
        )
        logger.info("  N toys:      %d (%d converged, %d failed)",
                     args.n_toys, n_converged, n_failed)
        logger.info("  n_sig_exp:   %.1f (at mu=1)", n_sig_expected)
        if n_converged > 0:
            logger.info(
                "  n_sig:       mean = %.1f +/- %.1f,  median = %.1f",
                np.mean(conv_n_sig), np.std(conv_n_sig),
                np.median(conv_n_sig),
            )
            logger.info(
                "  mu:          mean = %.3f +/- %.3f",
                np.mean(conv_mu), np.std(conv_mu),
            )
        if len(pulls) > 5:
            logger.info(
                "  Pull:        mean = %.3f,  sigma = %.3f",
                np.mean(pulls), np.std(pulls),
            )
        logger.info("  Output:    %s", base_output_dir)
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
