#!/usr/bin/env python3
"""Signal shape study: on-shell FWHM and peak per (M_WR, M_N, channel) point.

Per signal point, measure within the on-shell window [0.7, 1.3]*M_WR:
  - FWHM (half-max linear interpolation)
  - peak position (bin-center of the highest-count bin)

Outputs:
  - fwhm_<channel>.{pdf,png}      trend plot per channel
  - fwhm_comparison.{pdf,png}     ee vs mumu overlay
  - (optional) per-point fit diagnostic and per-WR shape overlay plots

Feeds fit_fwhm_parameterization.py, which fits FWHM(M_N/M_WR, M_WR), and
fit_signal_toy.py, which uses peak_onshell as the bifur/gauss mu central
value when --peak-mode fix-pred is selected.

Usage:
    python signal_fitting/measure_fwhm.py --era RunIISummer20UL18 --dir 20260406_signals -v
    python signal_fitting/measure_fwhm.py --era RunIISummer20UL18 --dir 20260406_signals --fit-plots --shape-overlays
"""
from __future__ import annotations

import argparse
import dataclasses
import json
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
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wrplotter.cli_utils import setup_logging
from wrplotter.config import load_lumi
from wrplotter.paths import input_dirs_for_era, repo_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Minimum M_N/M_WR ratio (below this is the boosted-N regime).
MIN_RATIO = 0.10

# Exclude the most-resolved grid point, which sits at N = M_WR - EXCLUDE_MASS_GAP_GEV.
# The kinematic phase space collapses there and the m_lljj shape distorts.
EXCLUDE_MASS_GAP_GEV = 100

# On-shell window for all width measurements: [LO*M_WR, HI*M_WR].
ONSHELL_WINDOW_LO_FRAC: float = 0.7
ONSHELL_WINDOW_HI_FRAC: float = 1.3


# ---------------------------------------------------------------------------
# Grid discovery and helpers
# ---------------------------------------------------------------------------

def discover_signal_grid(
    input_dirs: list[Path],
    min_ratio: float = MIN_RATIO,
) -> dict[str, list[str]]:
    """Auto-discover signal points from WRAnalyzer_signal_*.root files.

    Returns dict keyed by 'WR<mass>' with sorted signal-tag lists,
    filtered to M_N/M_WR >= min_ratio.
    """
    import re
    pattern = re.compile(r"WRAnalyzer_signal_(WR\d+_N\d+)\.root$")

    found: dict[str, list[tuple[int, str]]] = {}
    for d in input_dirs:
        if not d.exists():
            continue
        for fpath in d.glob("WRAnalyzer_signal_*.root"):
            m = pattern.search(fpath.name)
            if not m:
                continue
            tag = m.group(1)
            try:
                wr, n = parse_masses(tag)
            except (ValueError, IndexError):
                continue
            if n / wr < min_ratio:
                continue
            if wr - n == EXCLUDE_MASS_GAP_GEV:
                continue
            found.setdefault(f"WR{wr}", []).append((n, tag))

    grid: dict[str, list[str]] = {}
    for wr_label in sorted(found.keys(), key=lambda s: int(s[2:])):
        sorted_tags = sorted(set(found[wr_label]), key=lambda x: x[0])
        grid[wr_label] = [t for _, t in sorted_tags]
    return grid


def run_label(era: str) -> str:
    """'run2' / 'run3' output-subdirectory label for an era (lumi.yaml's run)."""
    return {"RunII": "run2", "Run3": "run3"}[str(load_lumi(era)["run"])]


def build_region_name(channel: str, topology: str) -> str:
    return f"wr_{channel}_{topology}_sr"


def build_hist_key(region: str, variable: str = "mass_fourobject") -> str:
    return f"{region}/{variable}_{region}"


def format_signal_label(signal_tag: str) -> str:
    """WR2000_N1100 -> '$W_R$ 2000, $N$ 1100'"""
    parts = signal_tag.split("_")
    wr_mass = parts[0].replace("WR", "")
    n_mass = parts[1].replace("N", "")
    return rf"$W_R$ {wr_mass}, $N$ {n_mass}"


def parse_masses(signal_tag: str) -> tuple[int, int]:
    """Extract (WR mass, N mass) from e.g. 'WR2000_N1100'."""
    parts = signal_tag.split("_")
    return int(parts[0].replace("WR", "")), int(parts[1].replace("N", ""))


# ---------------------------------------------------------------------------
# Histogram I/O
# ---------------------------------------------------------------------------

def load_signal_uproot(
    input_dir: Path, hist_key: str, signal_tag: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fpath = input_dir / f"WRAnalyzer_signal_{signal_tag}.root"
    f = uproot.open(fpath)
    h = f[hist_key]
    return h.axes[0].edges(), h.values(), h.variances()


def load_and_combine_signal(
    input_dirs: list[Path], hist_key: str, signal_tag: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum the histogram across sub-era input directories."""
    edges = values = variances = None
    for d in input_dirs:
        e, v, var = load_signal_uproot(d, hist_key, signal_tag)
        if edges is None:
            edges, values, variances = e, v.copy(), var.copy()
        else:
            values += v
            variances += var
    if edges is None:
        raise RuntimeError(f"No signal histograms found for {signal_tag}")
    return edges, values, variances


def rebin_histogram(
    edges: np.ndarray, values: np.ndarray, variances: np.ndarray, factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(values)
    n_new = n // factor
    values_new = values[: n_new * factor].reshape(n_new, factor).sum(axis=1)
    variances_new = variances[: n_new * factor].reshape(n_new, factor).sum(axis=1)
    edges_new = edges[:: factor][: n_new + 1]
    return edges_new, values_new, variances_new


# ---------------------------------------------------------------------------
# Shape parameters
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ShapeParams:
    signal_tag: str
    channel: str
    topology: str
    integral: float
    fwhm_onshell: float
    peak_onshell: float    # bin-center of the highest-count bin (bin-quantized)


def _compute_fwhm(
    centers: np.ndarray, values: np.ndarray,
) -> tuple[float, float, float, float]:
    """FWHM via linear interpolation at half-maximum, plus the peak position.

    Returns (fwhm, lo_edge, hi_edge, peak); peak is the bin-center of the
    highest-count bin (bin-quantized, no sub-bin interpolation). Returns
    (nan, nan, nan, nan) on failure.
    """
    if values.max() <= 0:
        return np.nan, np.nan, np.nan, np.nan

    half_max = values.max() / 2.0
    peak_idx = int(np.argmax(values))
    peak = float(centers[peak_idx])

    lo = np.nan
    for i in range(peak_idx, 0, -1):
        if values[i - 1] <= half_max:
            frac = (half_max - values[i - 1]) / (values[i] - values[i - 1])
            lo = centers[i - 1] + frac * (centers[i] - centers[i - 1])
            break

    hi = np.nan
    for i in range(peak_idx, len(values) - 1):
        if values[i + 1] <= half_max:
            frac = (half_max - values[i + 1]) / (values[i] - values[i + 1])
            hi = centers[i] + frac * (centers[i + 1] - centers[i])
            break

    fwhm = hi - lo if np.isfinite(lo) and np.isfinite(hi) else np.nan
    return fwhm, lo, hi, peak


def compute_shape_params(
    edges: np.ndarray, values: np.ndarray,
    signal_tag: str, channel: str, topology: str,
) -> ShapeParams:
    """On-shell mean / RMS / FWHM in [0.8, 1.2]*M_WR for one signal point."""
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = values.sum()
    wr_mass, _ = parse_masses(signal_tag)
    win_lo = ONSHELL_WINDOW_LO_FRAC * float(wr_mass)
    win_hi = ONSHELL_WINDOW_HI_FRAC * float(wr_mass)

    if total <= 0:
        nan = np.nan
        return ShapeParams(signal_tag, channel, topology, 0.0, nan, nan)

    fwhm_onshell = peak_onshell = np.nan
    pmask = (centers >= win_lo) & (centers <= win_hi)
    if pmask.sum() >= 3:
        pvals = values[pmask]
        pcent = centers[pmask]
        if pvals.sum() > 0:
            f, _, _, peak = _compute_fwhm(pcent, pvals)
            fwhm_onshell = float(f)
            peak_onshell = float(peak)

    return ShapeParams(
        signal_tag=signal_tag, channel=channel, topology=topology,
        integral=float(total),
        fwhm_onshell=fwhm_onshell, peak_onshell=peak_onshell,
    )


# ---------------------------------------------------------------------------
# Optional per-point diagnostic plot (--fit-plots)
# ---------------------------------------------------------------------------

def plot_single_fit(
    edges: np.ndarray, values: np.ndarray, sp: ShapeParams,
    output_path: Path, *, com: float, channel: str, era: str,
) -> None:
    """Histogram + on-shell window shade for one (M_WR, M_N, channel) point."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()

    centers = (edges[:-1] + edges[1:]) / 2.0
    ax.stairs(values, edges, color="#3f90da", linewidth=2, label="MC")

    wr_mass, _ = parse_masses(sp.signal_tag)
    win_lo = ONSHELL_WINDOW_LO_FRAC * wr_mass
    win_hi = ONSHELL_WINDOW_HI_FRAC * wr_mass
    mask = (centers >= win_lo) & (centers <= win_hi)
    if mask.any():
        ax.fill_between(
            centers[mask], 0, values[mask],
            color="#3f90da", alpha=0.25, zorder=0,
            label=rf"[{ONSHELL_WINDOW_LO_FRAC}, {ONSHELL_WINDOW_HI_FRAC}]$\,\times\,M_{{W_R}}$",
        )

    ax.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
    ax.set_ylabel("Events / bin")
    ax.set_xlim(0, min(wr_mass * 2, 10000))

    ch = {"ee": "ee", "mumu": r"$\mu\mu$"}.get(channel, channel)
    hep.cms.label(loc=0, ax=ax, data=False,
                  label="Work in Progress", com=com, fontsize=18)
    ax.text(
        0.05, 0.96,
        f"{ch}\nResolved SR\n{era}\n{format_signal_label(sp.signal_tag)}\n"
        rf"Shaded: $[0.7\,M_{{W_R}},\ 1.3\,M_{{W_R}}]$",
        transform=ax.transAxes, fontsize=16, verticalalignment="top",
    )
    ax.text(
        0.95, 0.95,
        rf"Shaded peak bin center $= {sp.peak_onshell:.0f}$ GeV"
        "\n"
        rf"Shaded FWHM $= {sp.fwhm_onshell:.0f}$ GeV",
        transform=ax.transAxes, fontsize=13,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="gray", alpha=0.8),
    )
    _, y_hi = ax.get_ylim()
    ax.set_ylim(0, y_hi * 1.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Optional shape overlay per WR group (--shape-overlays)
# ---------------------------------------------------------------------------

def plot_shape_overlay(
    histograms: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path, *,
    com: float, channel: str, topology: str, era: str,
    x_range: tuple[float, float] = (0, 10000),
) -> None:
    """Overlay all M_N normalized shapes for one WR group, colored by M_N."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()

    n_masses = [parse_masses(t)[1] for t in histograms]
    cmap = mcm.viridis
    norm = mcolors.Normalize(vmin=min(n_masses), vmax=max(n_masses))

    for sig_tag, (edges, values) in histograms.items():
        bin_widths = np.diff(edges)
        total = values.sum()
        if total <= 0:
            continue
        density = values / (total * bin_widths)
        _, m_n = parse_masses(sig_tag)
        color = cmap(norm(m_n))
        ax.stairs(density, edges, color=color, linewidth=2)

        wr, _ = parse_masses(sig_tag)
        win_lo = ONSHELL_WINDOW_LO_FRAC * wr
        win_hi = ONSHELL_WINDOW_HI_FRAC * wr
        centers = (edges[:-1] + edges[1:]) / 2.0
        mask = (centers >= win_lo) & (centers <= win_hi)
        if mask.any():
            ax.fill_between(centers[mask], 0, density[mask],
                            color=color, alpha=0.20, zorder=0)

    ax.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
    ax.set_ylabel("a.u.")
    ax.set_xlim(*x_range)

    ch = {"ee": "ee", "mumu": r"$\mu\mu$"}.get(channel, channel)
    hep.cms.label(loc=0, ax=ax, data=False,
                  label="Work in Progress", com=com, fontsize=20)
    wr_mass = parse_masses(next(iter(histograms)))[0]
    ax.text(
        0.05, 0.96,
        f"{ch}\n{topology.capitalize()} SR\n{era}\n"
        rf"$M_{{W_R}}$ = {wr_mass} GeV",
        transform=ax.transAxes, fontsize=18, verticalalignment="top",
    )
    _, y_hi = ax.get_ylim()
    ax.set_ylim(0, y_hi * 1.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$M_N$ [GeV]", fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved overlay: %s", output_path.stem)


# ---------------------------------------------------------------------------
# Trend plots: quantity vs M_N/M_WR, colored by M_WR
# ---------------------------------------------------------------------------

TREND_QUANTITIES: list[tuple[str, str]] = [
    ("fwhm_onshell", "FWHM [GeV]"),
]


def attr_stem(attr: str) -> str:
    """File-name stem for a trend attribute (drop '_onshell' suffix)."""
    return attr[:-len("_onshell")] if attr.endswith("_onshell") else attr


def _plot_trend(
    params_by_wr: dict[str, list[ShapeParams]],
    output_path: Path, *,
    value_attr: str, ylabel: str,
    com: float, channel: str, topology: str, era: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    hep.style.use("CMS")
    fig, ax = plt.subplots()

    wr_values = [int(k[2:]) for k in params_by_wr]
    use_colorbar = len(wr_values) > 4
    if use_colorbar:
        cmap = mcm.viridis
        norm = mcolors.Normalize(vmin=min(wr_values), vmax=max(wr_values))
    else:
        color_cycle = ["#3f90da", "#bd1f01", "#832db6", "#e76300"]

    for idx, (wr_label, params_list) in enumerate(params_by_wr.items()):
        sorted_params = sorted(
            params_list, key=lambda p: parse_masses(p.signal_tag)[1]
        )
        m_wr = parse_masses(sorted_params[0].signal_tag)[0]
        ratios = [parse_masses(p.signal_tag)[1] / m_wr for p in sorted_params]
        values = [getattr(p, value_attr) for p in sorted_params]
        color = cmap(norm(m_wr)) if use_colorbar else color_cycle[idx % len(color_cycle)]

        ax.plot(
            ratios, values,
            marker="o", markersize=5, linewidth=2, color=color,
            label=None if use_colorbar else wr_label,
            markeredgecolor="black", markeredgewidth=0.3,
        )

    ax.set_xlabel(r"$M_N / M_{W_R}$")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)

    ch = {"ee": "ee", "mumu": r"$\mu\mu$"}.get(channel, channel)
    hep.cms.label(loc=0, ax=ax, data=False,
                  label="Work in Progress", com=com, fontsize=20)
    ax.text(
        0.05, 0.96, f"{ch}\n{topology.capitalize()} SR\n{era}",
        transform=ax.transAxes, fontsize=20, verticalalignment="top",
    )

    if use_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=14)
    else:
        ax.legend(fontsize=16, loc="upper right")

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(y_lo, y_hi + 0.30 * (y_hi - y_lo))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved trend: %s", output_path.stem)


def compute_shared_trend_ylims(
    params_by_channel: dict[str, dict[str, list[ShapeParams]]],
) -> dict[str, tuple[float, float]]:
    """Union y-range per trend quantity across channels."""
    out: dict[str, tuple[float, float]] = {}
    for attr, _ in TREND_QUANTITIES:
        vals: list[float] = []
        for pbw in params_by_channel.values():
            for params_list in pbw.values():
                vals.extend(
                    getattr(p, attr) for p in params_list
                    if np.isfinite(getattr(p, attr))
                )
        if not vals:
            continue
        y_max = max(vals)
        out[attr] = (0.0, y_max * 1.30)
    return out


def plot_shape_parameters_vs_mass(
    params_by_wr: dict[str, list[ShapeParams]],
    output_dir: Path, *,
    com: float, channel: str, topology: str, era: str,
    shared_ylims: dict[str, tuple[float, float]] | None = None,
) -> None:
    for attr, ylabel in TREND_QUANTITIES:
        _plot_trend(
            params_by_wr,
            output_dir / f"{attr_stem(attr)}_{channel}.pdf",
            value_attr=attr, ylabel=ylabel,
            com=com, channel=channel, topology=topology, era=era,
            ylim=shared_ylims.get(attr) if shared_ylims else None,
        )


def plot_channel_comparison(
    params_by_wr_ee: dict[str, list[ShapeParams]],
    params_by_wr_mumu: dict[str, list[ShapeParams]],
    output_dir: Path, *,
    com: float, topology: str, era: str,
) -> None:
    """ee (solid) vs mumu (dashed) overlay for each trend quantity."""
    from matplotlib.lines import Line2D

    common_wr = sorted(
        set(params_by_wr_ee.keys()) & set(params_by_wr_mumu.keys()),
        key=lambda s: int(s[2:]),
    )
    if not common_wr:
        return

    wr_values = [int(k[2:]) for k in common_wr]
    use_colorbar = len(common_wr) > 4
    if use_colorbar:
        cmap = mcm.viridis
        norm = mcolors.Normalize(vmin=min(wr_values), vmax=max(wr_values))
    else:
        color_cycle = ["#3f90da", "#bd1f01", "#832db6", "#e76300"]

    for attr, ylabel in TREND_QUANTITIES:
        hep.style.use("CMS")
        fig, ax = plt.subplots()

        for idx, wr_label in enumerate(common_wr):
            m_wr = int(wr_label[2:])
            color = cmap(norm(m_wr)) if use_colorbar else color_cycle[idx % len(color_cycle)]
            channel_styles = [
                # (label, params, linestyle, marker, face_color, edge_color, edge_width)
                ("ee",   params_by_wr_ee,   "-",            "o", color,   "black", 0.3),
                ("mumu", params_by_wr_mumu, (0, (4, 2, 1, 2)), "s", "white", color,   1.4),
            ]
            for ch, pbw, ls, marker, mfc, mec, mew in channel_styles:
                params_list = pbw.get(wr_label, [])
                if not params_list:
                    continue
                sorted_params = sorted(
                    params_list, key=lambda p: parse_masses(p.signal_tag)[1]
                )
                ratios = [parse_masses(p.signal_tag)[1] / m_wr for p in sorted_params]
                values = [getattr(p, attr) for p in sorted_params]
                label = None
                if not use_colorbar and ls == "-":
                    label = f"{wr_label} {ch}"
                ax.plot(
                    ratios, values,
                    marker=marker, markersize=5, linewidth=1.8,
                    linestyle=ls, color=color, label=label,
                    markerfacecolor=mfc, markeredgecolor=mec, markeredgewidth=mew,
                )

        ax.set_xlabel(r"$M_N / M_{W_R}$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        hep.cms.label(loc=0, ax=ax, data=False,
                      label="Work in Progress", com=com, fontsize=20)
        ax.text(
            0.05, 0.96,
            f"ee vs $\\mu\\mu$\n{topology.capitalize()} SR\n{era}",
            transform=ax.transAxes, fontsize=18, verticalalignment="top",
        )

        style_handles = [
            Line2D([0], [0], color="gray", linestyle="-", linewidth=2,
                   marker="o", markersize=6,
                   markerfacecolor="gray", markeredgecolor="black",
                   markeredgewidth=0.3, label="ee"),
            Line2D([0], [0], color="gray", linestyle=(0, (4, 2, 1, 2)), linewidth=2,
                   marker="s", markersize=6,
                   markerfacecolor="white", markeredgecolor="gray",
                   markeredgewidth=1.4, label=r"$\mu\mu$"),
        ]
        if use_colorbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=14)
            ax.legend(handles=style_handles, fontsize=14, loc="upper right")
        else:
            handles, labels = ax.get_legend_handles_labels()
            handles.extend(style_handles)
            labels.extend(["ee (solid)", r"$\mu\mu$ (dashed)"])
            ax.legend(handles, labels, fontsize=12, loc="upper right")

        _, y_hi = ax.get_ylim()
        ax.set_ylim(0, y_hi * 1.30)

        out = output_dir / f"{attr_stem(attr)}_comparison.pdf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info("Saved channel comparison: %s", out.stem)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era", default="RunIISummer20UL18")
    parser.add_argument(
        "--dir", default="",
        help="Subdir under rootfiles/<run>/<year>/<era>/ (e.g. 20260406_signals)",
    )
    parser.add_argument(
        "--channel", choices=["ee", "mumu"], default=None,
        help="If omitted, runs both channels.",
    )
    parser.add_argument("--topology", choices=["resolved", "boosted"], default="resolved")
    parser.add_argument("--rebin", type=int, default=6,
                        help="Rebin factor (default 6 -> 60 GeV/bin)")
    parser.add_argument(
        "--output-dir", default=None,
        help="Default: signal_fitting/outputs/<era>/fwhm/",
    )
    parser.add_argument(
        "--fit-plots", action="store_true",
        help="Also write per-(M_WR, M_N) on-shell-window diagnostic plots.",
    )
    parser.add_argument(
        "--shape-overlays", action="store_true",
        help="Also write per-WR normalized shape overlay plots.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    era = args.era
    topology = args.topology
    rebin = args.rebin
    channels = [args.channel] if args.channel else ["ee", "mumu"]

    info = load_lumi(era)
    com = info.get("com", 13.6)

    input_dirs, _ = input_dirs_for_era(era, repo_root(), args.dir)
    logger.info("Input directories: %s", [str(d) for d in input_dirs])

    base_output = Path(args.output_dir or str(
        repo_root() / "signal_fitting" / "outputs" / era / "fwhm"
    ))

    mass_var = "mass_twoobject" if topology == "boosted" else "mass_fourobject"

    signal_grid = discover_signal_grid(input_dirs)
    logger.info("Discovered %d WR mass groups", len(signal_grid))
    if not signal_grid:
        logger.error("No signal files found in input directories")
        return

    all_shape_params: list[ShapeParams] = []
    params_by_channel: dict[str, dict[str, list[ShapeParams]]] = {}

    for channel in channels:
        logger.info("=" * 60)
        logger.info("Channel: %s", channel)

        region = build_region_name(channel, topology)
        hist_key = build_hist_key(region, mass_var)

        params_by_wr: dict[str, list[ShapeParams]] = {}

        for wr_label, signal_tags in signal_grid.items():
            wr_histograms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            wr_params: list[ShapeParams] = []

            for sig in signal_tags:
                try:
                    edges, values, variances = load_and_combine_signal(
                        input_dirs, hist_key, sig,
                    )
                    if rebin > 1:
                        edges, values, variances = rebin_histogram(
                            edges, values, variances, rebin,
                        )
                    sp = compute_shape_params(edges, values, sig, channel, topology)
                    logger.info(
                        "  %s: FWHM_on=%.1f, peak_on=%.1f, integral=%.1f",
                        sig, sp.fwhm_onshell, sp.peak_onshell, sp.integral,
                    )

                    wr_histograms[sig] = (edges, values)
                    wr_params.append(sp)
                    all_shape_params.append(sp)

                    if args.fit_plots:
                        plot_single_fit(
                            edges, values, sp,
                            base_output / "diagnostics" / f"{sig}_{channel}.pdf",
                            com=com, channel=channel, era=era,
                        )
                except Exception as e:
                    logger.warning("  Skipping %s: %s", sig, e)
                    continue

            params_by_wr[wr_label] = wr_params

            if args.shape_overlays and wr_histograms:
                plot_shape_overlay(
                    wr_histograms,
                    base_output / f"overlay_{wr_label}_{channel}.pdf",
                    com=com, channel=channel, topology=topology, era=era,
                )

        params_by_channel[channel] = params_by_wr
        logger.info("=" * 60)

    # Trend plots (shared y-limits across channels so ee and mumu are directly comparable)
    shared_ylims = compute_shared_trend_ylims(params_by_channel)
    for channel, pbw in params_by_channel.items():
        if any(len(v) >= 2 for v in pbw.values()):
            plot_shape_parameters_vs_mass(
                pbw, base_output,
                com=com, channel=channel, topology=topology, era=era,
                shared_ylims=shared_ylims,
            )

    # Channel comparison (ee vs mumu)
    if len(channels) == 2 and all(
        any(len(v) >= 2 for v in params_by_channel[ch].values()) for ch in channels
    ):
        plot_channel_comparison(
            params_by_channel["ee"], params_by_channel["mumu"],
            base_output, com=com, topology=topology, era=era,
        )

    logger.info("Done. Outputs in %s", base_output)


if __name__ == "__main__":
    main()
