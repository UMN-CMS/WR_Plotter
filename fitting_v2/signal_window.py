#!/usr/bin/env python3
"""
Compute signal fitting window from MC histogram mean and RMS.

Loads the signal MC m_lljj histogram, computes the mean and RMS directly
from the histogram (no parametric fit), and defines a symmetric fitting
window: [mean - n_sigma * RMS, mean + n_sigma * RMS].

The output JSON is consumed by fit_background.py to set the fit range.

Usage:
    python fitting_v2/signal_window.py \\
        --era RunIII2024Summer24 \\
        --dir 20260223_nlo_dy \\
        --signal WR2000_N1100

    # Wider window:
    python fitting_v2/signal_window.py \\
        --era RunIII2024Summer24 \\
        --dir 20260223_nlo_dy \\
        --signal WR2000_N1100 \\
        --n-sigma 4.0
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

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
    load_signal,
    save_results_json,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_signal_window(
    hist: ROOT.TH1D,
    mean: float,
    rms: float,
    n_sigma: float,
    window_lo: float,
    window_hi: float,
    output_path: Path,
    *,
    lumi: float,
    com: float,
    channel: str,
    topology: str,
    era: str,
    signal_tag: str,
    bin_width_gev: float,
) -> None:
    """Plot signal histogram with vertical mean line and horizontal RMS bar."""
    # Extract histogram arrays
    n_bins = hist.GetNbinsX()
    edges = np.array([hist.GetBinLowEdge(i) for i in range(1, n_bins + 2)])
    contents = np.array([hist.GetBinContent(i) for i in range(1, n_bins + 1)])
    errors = np.array([hist.GetBinError(i) for i in range(1, n_bins + 1)])
    centers = (edges[:-1] + edges[1:]) / 2

    hep.style.use("CMS")
    fig, ax = plt.subplots()

    # Format signal tag for legend: WR2000_N1100 -> $W_R$ 2000, $N$ 1100
    parts = signal_tag.split("_")
    wr_mass = parts[0].replace("WR", "")
    n_mass = parts[1].replace("N", "")
    sig_label = rf"Signal MC ($W_R$ {wr_mass}, $N$ {n_mass})"

    # Histogram as step plot
    ax.stairs(contents, edges, color="#1f77b4", linewidth=1.5, label=sig_label)
    ax.errorbar(centers, contents, yerr=errors, fmt="none", ecolor="#1f77b4", linewidth=1)

    # Vertical dashed line at mean, stopping at the peak height
    peak = contents.max()
    ax.vlines(mean, 1e-1, peak, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean:.0f} GeV")

    # Horizontal line for RMS (centered at mean, spanning ±RMS)
    # Use geometric mean between y_min and peak for log scale
    y_rms = np.sqrt(1e-1 * peak)
    ax.hlines(y_rms, mean - rms, mean + rms, color="orange", linewidth=2.5, label=f"RMS = {rms:.0f} GeV")
    # Small vertical caps (multiplicative in log space)
    cap_factor = 1.5
    for x_cap in [mean - rms, mean + rms]:
        ax.vlines(x_cap, y_rms / cap_factor, y_rms * cap_factor, color="orange", linewidth=2)

    # Shade the fitting window
    ax.axvspan(window_lo, window_hi, color="green", alpha=0.10, zorder=0,
               label=f"Fit window ({n_sigma:.0f}$\\sigma$)")

    ax.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
    ax.set_ylabel(f"Events / {bin_width_gev:.0f} GeV")
    ax.set_xlim(edges[0], 6000)
    ax.set_yscale("log")
    ax.set_ylim(1e-1, 10 ** (np.ceil(np.log10(contents.max())) + 1))
    ax.legend(loc="upper right", fontsize=16)

    ch_labels = {"ee": "ee", "mumu": r"$\mu\mu$"}
    ch_label = ch_labels.get(channel, channel)
    ax.text(
        0.05, 0.96, f"{ch_label}\n{topology.capitalize()} SR\n{era}",
        transform=ax.transAxes, fontsize=18, verticalalignment="top",
    )

    hep.cms.label(
        loc=0, ax=ax, data=False,
        label="Work in Progress",
        lumi=f"{lumi:.2f}", com=com, fontsize=18,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

    logger.info("Saved signal window plot: %s (.pdf and .png)", output_path.stem)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute signal fitting window from MC histogram mean and RMS."
        )
    )
    add_common_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    if not args.signal:
        sys.exit("ERROR: --signal is required. Example: --signal WR2000_N1100")

    ROOT.gROOT.SetBatch(True)

    # --- Resolve input path ---
    info = load_lumi(args.era)
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    input_dir = input_dirs[0]
    logger.info("Input directory: %s", input_dir)

    # --- Output directory ---
    base_output_dir = Path(
        args.output_dir
        or str(repo_root() / "fitting_v2" / "outputs" / args.era / "signal_window" / args.signal)
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    channels = [args.channel] if args.channel else ["ee", "mumu"]

    for channel in channels:
        logger.info("=" * 60)
        logger.info("Channel: %s", channel)

        # --- Build histogram key ---
        region = build_region_name(channel, args.topology)
        mass_var = "mass_twoobject" if args.topology == "boosted" else "mass_fourobject"
        hist_key = build_hist_key(region, mass_var)
        logger.info("Histogram key: %s", hist_key)

        # --- Load signal histogram ---
        sig_hist = load_signal(input_dir, hist_key, args.signal)

        # --- Rebin (optional, for smoother stats) ---
        if args.rebin > 1:
            sig_hist = sig_hist.Rebin(args.rebin, f"h_sig_{channel}_rebinned")

        # --- Compute statistics from histogram ---
        mean = sig_hist.GetMean()
        rms = sig_hist.GetRMS()
        n_events = sig_hist.Integral()
        skewness = sig_hist.GetSkewness()
        kurtosis = sig_hist.GetKurtosis()

        # --- Compute fitting window ---
        n_sigma = args.n_sigma
        window_lo = mean - n_sigma * rms
        window_hi = mean + n_sigma * rms

        # Clamp to mass range if specified
        mass_lo, mass_hi = args.mass_range
        window_lo = max(window_lo, mass_lo)
        window_hi = min(window_hi, mass_hi)

        # --- Assemble results ---
        results = {
            "metadata": {
                "era": args.era,
                "channel": channel,
                "topology": args.topology,
                "region": region,
                "signal": args.signal,
                "variable": "mass_fourobject",
                "rebin": args.rebin,
                "mass_range": [mass_lo, mass_hi],
                "timestamp": datetime.now().isoformat(),
            },
            "signal_stats": {
                "mean": mean,
                "rms": rms,
                "n_events": n_events,
                "skewness": skewness,
                "kurtosis": kurtosis,
            },
            "fit_window": {
                "lo": window_lo,
                "hi": window_hi,
                "n_sigma": n_sigma,
            },
        }

        json_path = base_output_dir / f"window_{channel}.json"
        save_results_json(results, json_path)

        # --- Plot signal histogram with mean and RMS ---
        plot_path = base_output_dir / f"sig_win_{args.signal}_{channel}.pdf"
        plot_signal_window(
            sig_hist, mean, rms, n_sigma,
            window_lo, window_hi,
            plot_path,
            lumi=info["lumi"],
            com=info.get("com", 13.6),
            channel=channel,
            topology=args.topology,
            era=args.era,
            signal_tag=args.signal,
            bin_width_gev=sig_hist.GetBinWidth(1),
        )

        # --- Summary ---
        logger.info("-" * 40)
        logger.info("Signal: %s  Channel: %s", args.signal, channel)
        logger.info("  Mean:      %.1f GeV", mean)
        logger.info("  RMS:       %.1f GeV", rms)
        logger.info("  N events:  %.1f", n_events)
        logger.info("  Skewness:  %.3f", skewness)
        logger.info("  Kurtosis:  %.3f", kurtosis)
        logger.info(
            "  Window (%.1f sigma): [%.1f, %.1f] GeV  (width: %.1f GeV)",
            n_sigma, window_lo, window_hi, window_hi - window_lo,
        )
        logger.info("  Output: %s", json_path)
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
