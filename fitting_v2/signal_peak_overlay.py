#!/usr/bin/env python3
"""
Plot signal peaks for multiple WR2000 mass points, combining 2022 eras.

Loads the signal MC m_lljj histograms for each mass point from both
Run3Summer22 and Run3Summer22EE, sums them, and produces one plot per
signal point showing the peak with mean and RMS annotations.

Usage:
    python fitting_v2/signal_peak_overlay.py \
        --dir 20260223_nlo_dy

    # Single channel:
    python fitting_v2/signal_peak_overlay.py \
        --dir 20260223_nlo_dy --channel ee
"""
from __future__ import annotations

import argparse
import logging
import sys
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
    build_region_name,
    build_hist_key,
    load_signal,
)

logger = logging.getLogger(__name__)

ERA = "2022"
SIGNAL_POINTS = [
    "WR2000_N100",
    "WR2000_N700",
    "WR2000_N1300",
    "WR2000_N1900",
]
COLOR = "#1f77b4"

# Signal points that use boosted topology (all others use resolved)
BOOSTED_SIGNALS = {"WR2000_N100"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_and_combine_signal(
    input_dirs: list[Path],
    hist_key: str,
    signal_tag: str,
) -> ROOT.TH1D:
    """Load signal histogram from each sub-era directory and sum them."""
    combined = None
    for d in input_dirs:
        h = load_signal(d, hist_key, signal_tag)
        if combined is None:
            combined = h.Clone(f"h_sig_{signal_tag}_combined")
            combined.SetDirectory(0)
        else:
            combined.Add(h)
    if combined is None:
        raise RuntimeError(f"No signal histograms found for {signal_tag}")
    return combined


def format_signal_label(signal_tag: str) -> str:
    """WR2000_N1100 -> '$W_R$ 2000, $N$ 1100'"""
    parts = signal_tag.split("_")
    wr_mass = parts[0].replace("WR", "")
    n_mass = parts[1].replace("N", "")
    return rf"$W_R$ {wr_mass}, $N$ {n_mass}"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_signal_peak(
    hist: ROOT.TH1D,
    signal_tag: str,
    output_path: Path,
    *,
    lumi: float,
    com: float,
    channel: str,
    topology: str,
) -> None:
    """Plot a single signal peak with mean and RMS annotations."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()

    # Extract histogram arrays
    n_bins = hist.GetNbinsX()
    edges = np.array([hist.GetBinLowEdge(j) for j in range(1, n_bins + 2)])
    contents = np.array([hist.GetBinContent(j) for j in range(1, n_bins + 1)])
    errors = np.array([hist.GetBinError(j) for j in range(1, n_bins + 1)])
    centers = (edges[:-1] + edges[1:]) / 2

    mean = hist.GetMean()
    rms = hist.GetRMS()
    peak = contents.max()

    sig_label = format_signal_label(signal_tag)

    # Histogram as step plot
    ax.stairs(contents, edges, color=COLOR, linewidth=1.5, label=sig_label)
    ax.errorbar(centers, contents, yerr=errors, fmt="none", ecolor=COLOR, linewidth=1)

    # Vertical dashed line at mean, stopping at peak
    ax.vlines(mean, 0, peak, color="red", linestyle="--", linewidth=2,
              label=f"Mean = {mean:.0f} GeV")

    # Horizontal RMS bar at half the peak height
    y_rms = 0.5 * peak
    ax.hlines(y_rms, mean - rms, mean + rms, color="orange", linewidth=2.5,
              label=f"RMS = {rms:.0f} GeV")
    # Small vertical caps
    cap_height = 0.05 * peak
    for x_cap in [mean - rms, mean + rms]:
        ax.vlines(x_cap, y_rms - cap_height, y_rms + cap_height, color="orange", linewidth=2)

    bin_width = hist.GetBinWidth(1)
    ax.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
    ax.set_ylabel(f"Events / {bin_width:.0f} GeV")
    ax.set_xlim(edges[0], 6000)

    # Dynamic linear y-axis scaling
    ax.set_ylim(0, peak * 1.4)

    ax.legend(loc="upper right", fontsize=16)

    ch_labels = {"ee": "ee", "mumu": r"$\mu\mu$"}
    ch_label = ch_labels.get(channel, channel)
    ax.text(
        0.05, 0.96, f"{ch_label}\n{topology.capitalize()} SR\n{ERA} combined",
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

    logger.info("Saved signal peak plot: %s (.pdf and .png)", output_path.stem)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot signal peaks for multiple WR2000 mass points (2022 combined)."
    )
    parser.add_argument(
        "--dir", type=str, default="",
        help="Subdirectory under rootfiles/<run>/<year>/<era>/ (e.g. 20260223_nlo_dy)",
    )
    parser.add_argument(
        "--channel", choices=["ee", "mumu"], default=None,
        help="Lepton channel. If omitted, runs both ee and mumu.",
    )
    parser.add_argument(
        "--rebin", type=int, default=2,
        help="Rebin factor (default: 2 -> 20 GeV/bin)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: fitting_v2/outputs/2022/signal_peak_overlay/)",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity (-v or -vv)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    ROOT.gROOT.SetBatch(True)

    # Resolve input paths for combined 2022
    info = load_lumi(ERA)
    input_dirs, _ = input_dirs_for_era(ERA, repo_root(), args.dir)
    logger.info("Input directories: %s", [str(d) for d in input_dirs])

    # Output directory
    base_output_dir = Path(
        args.output_dir
        or str(repo_root() / "fitting_v2" / "outputs" / ERA / "signal_peak_overlay")
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    channels = [args.channel] if args.channel else ["ee", "mumu"]

    for channel in channels:
        logger.info("=" * 60)
        logger.info("Channel: %s", channel)

        # Load, combine, and plot each signal point separately
        for sig in SIGNAL_POINTS:
            topology = "boosted" if sig in BOOSTED_SIGNALS else "resolved"
            region = build_region_name(channel, topology)
            mass_var = "mass_twoobject" if topology == "boosted" else "mass_fourobject"
            hist_key = build_hist_key(region, mass_var)
            logger.info("  %s: topology=%s, hist_key=%s", sig, topology, hist_key)

            try:
                h = load_and_combine_signal(input_dirs, hist_key, sig)
                if args.rebin > 1:
                    h = h.Rebin(args.rebin, f"h_sig_{sig}_{channel}_rebinned")
                logger.info("  %s: mean=%.1f, rms=%.1f, integral=%.1f",
                            sig, h.GetMean(), h.GetRMS(), h.Integral())
            except Exception as e:
                logger.warning("  Skipping %s: %s", sig, e)
                continue

            plot_path = base_output_dir / f"signal_peak_{sig}_{channel}.pdf"
            plot_signal_peak(
                h, sig, plot_path,
                lumi=info["lumi"],
                com=info.get("com", 13.6),
                channel=channel,
                topology=topology,
            )

        logger.info("=" * 60)


if __name__ == "__main__":
    main()
