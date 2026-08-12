#!/usr/bin/env python3
"""Overlay normalized signal peaks for multiple M_N at fixed M_WR.

For each chosen M_WR value, produce one plot showing all available M_N
points overlaid, each normalized to unit area. Color-coded by M_N.

The point: visualize how the on-shell peak shape and position change with
the neutrino mass at a fixed W_R mass, motivating the [0.7, 1.3]*M_WR
on-shell window used downstream.

Output:
    0_signal_samples/peak_overlays/overlay_WR<m>_<topology>_<channel>.{pdf,png}

Usage:
    python signal_fitting/0_signal_samples/signal_peak_overlay.py
    python signal_fitting/0_signal_samples/signal_peak_overlay.py \
        --wr-list 2400 4000 5600
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))

from wrplotter.cli_utils import setup_logging
from wrplotter.config import load_lumi
from wrplotter.paths import input_dirs_for_era, repo_root

from measure_fwhm import (
    build_hist_key,
    build_region_name,
    load_and_combine_signal,
    parse_masses,
    rebin_histogram,
)
from shape_estimators import load_master_masses

logger = logging.getLogger(__name__)

DEFAULT_ERA = "RunIISummer20UL18"
# Resolved is the four-object mass (ℓℓjj); boosted is the two-object mass —
# one lepton + one large-R jet (ℓJ).
MASS_LABEL = {
    "resolved": r"$m_{\ell\ell jj}$ [GeV]",
    "boosted": r"$m_{\ell J}$ [GeV]",
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_n_overlay(
    histograms: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path, *,
    wr_mass: int, channel: str, topology: str, era: str, com: float,
    x_range: tuple[float, float] = (0, 10000),
) -> None:
    """Multi-M_N overlay at fixed M_WR, normalized to unit area."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()

    # Sort by M_N for legend ordering and color ramp.
    items = sorted(histograms.items(), key=lambda kv: parse_masses(kv[0])[1])
    n_masses = [parse_masses(t)[1] for t, _ in items]
    cmap = mcm.viridis
    # Guard the degenerate single-M_N case (common for boosted, where a W_R may
    # have only one low-x point) so Normalize doesn't collapse to NaN.
    n_lo, n_hi = min(n_masses), max(n_masses)
    norm = mcolors.Normalize(vmin=n_lo, vmax=n_hi if n_hi > n_lo else n_lo + 1)

    for sig_tag, (edges, values) in items:
        bin_widths = np.diff(edges)
        total = float(values.sum())
        if total <= 0:
            continue
        density = values / (total * bin_widths)
        _, m_n = parse_masses(sig_tag)
        color = cmap(norm(m_n))

        ax.stairs(density, edges, color=color, linewidth=1.5)

    ax.set_xlabel(MASS_LABEL[topology])
    ax.set_ylabel("a.u.")
    ax.set_xlim(*x_range)
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(0, y_hi * 1.30)

    hep.cms.label(loc=0, ax=ax, data=False,
                  label="Work in Progress", com=com, fontsize=18)

    ch = {"ee": "ee", "mumu": r"$\mu\mu$"}.get(channel, channel)
    ax.text(
        0.04, 0.96,
        f"{ch}\n{topology.capitalize()} SR\n{era}\n"
        rf"$M_{{W_R}}={wr_mass}$ GeV",
        transform=ax.transAxes, fontsize=14, verticalalignment="top",
    )

    # Colorbar for M_N — replaces the per-curve legend.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$M_N$ [GeV]", fontsize=24)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", output_path.stem)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era", default=DEFAULT_ERA)
    p.add_argument("--dir", default="20260624_signals")
    p.add_argument("--channel", choices=["ee", "mumu"], default=None,
                   help="If omitted, runs both ee and mumu.")
    p.add_argument("--wr-list", nargs="+", type=int, default=None,
                   help="M_WR values (GeV) to make overlay plots for. "
                        "Default: every W_R mass in the CSV for this topology.")
    p.add_argument("--mass-csv", type=Path,
                   default=Path(__file__).resolve().parents[1] / "master_masses.csv")
    p.add_argument("--topology", choices=["resolved", "boosted"], default="resolved")
    p.add_argument("--rebin", type=int, default=8,
                   help="Rebin factor (native = 10 GeV bins; default 8 -> 80 GeV bins).")
    p.add_argument("--output-dir", default=None)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    info = load_lumi(args.era)
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)

    out_base = Path(args.output_dir or
                    str(Path(__file__).resolve().parent / "peak_overlays"))
    out_base.mkdir(parents=True, exist_ok=True)

    channels = [args.channel] if args.channel else ["ee", "mumu"]
    mass_var = "mass_twoobject" if args.topology == "boosted" else "mass_fourobject"

    # Signal grid from the master CSV: group this topology's tags by M_WR.
    masses = load_master_masses(args.mass_csv, topology=args.topology)
    grid: dict[int, list[str]] = {}
    for tag in masses:
        m_wr, _ = parse_masses(tag)
        grid.setdefault(int(m_wr), []).append(tag)

    wr_list = args.wr_list if args.wr_list else sorted(grid)
    n_points = sum(len(grid.get(w, [])) for w in wr_list)
    logger.info("Topology %s: %d W_R masses, %d signal points",
                args.topology, len(wr_list), n_points)

    for channel in channels:
        logger.info("=" * 60)
        logger.info("Channel: %s", channel)
        region = build_region_name(channel, args.topology)
        hist_key = build_hist_key(region, mass_var)

        for wr_mass in wr_list:
            wr_label = f"WR{wr_mass}"
            sig_tags = grid.get(wr_mass, [])
            if not sig_tags:
                logger.warning("No signals found for %s", wr_label)
                continue

            histograms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for sig in sig_tags:
                try:
                    edges, values, variances = load_and_combine_signal(
                        input_dirs, hist_key, sig,
                    )
                    if args.rebin > 1:
                        edges, values, _ = rebin_histogram(
                            edges, values, variances, args.rebin,
                        )
                    histograms[sig] = (edges, values)
                except Exception as e:
                    logger.warning("  Skipping %s: %s", sig, e)
                    continue
            logger.info("%s: %d signal points loaded", wr_label, len(histograms))
            if not histograms:
                continue

            plot_n_overlay(
                histograms,
                out_base / f"overlay_{wr_label}_{args.topology}_{channel}.pdf",
                wr_mass=wr_mass, channel=channel, topology=args.topology,
                era=args.era, com=info.get("com", 13.0),
            )


if __name__ == "__main__":
    main()
