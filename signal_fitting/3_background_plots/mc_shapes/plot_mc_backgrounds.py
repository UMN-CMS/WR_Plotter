#!/usr/bin/env python3
"""Stage 3 — inspect the MC background mass shapes before choosing a fit function.

The background model targets the *summed* background

    B = DY+jets + tt/tW + nonprompt + other,

so before picking a smooth analytic form we look at the MC that makes it up, per
signal region. Three views are produced (choose with --plots):

  overlay     four step curves overlaid on log y, per region:
                total / DY+jets / tt+tW / nonprompt+other
  stack       the classic CMS stacked, shaded plot: the four MC processes
                stacked (filled) with a stat-uncertainty band (no data — SR)
  individual  one filled plot per single process (just DYJets, just tt+tW, ...)

separately for the four signal regions:

  ee resolved, mumu resolved, ee boosted, mumu boosted.

Mass observable per topology (the reconstructed W_R mass proxy, matching the
signal study and the analysis plot configs):
  resolved -> mass_fourobject  (m_lljj)   — the four-object mass
  boosted  -> mass_twoobject   (m_lJ)     — lepton + large-R jet
The boosted SR has no four-object mass on disk, so it is m_lJ there.

The MC histograms in these files are already scaled to the era luminosity at
production time (the stack pipeline applies no lumi factor and all k-factors are
1.0 for this era), so they are loaded and summed directly. Process colors, labels,
and stack order follow the analysis sample groups (data/sample_groups.yaml); the
stacking/shading recipe mirrors wrplotter.plotting_helpers.plot_stack.

Outputs (co-located with this script):
  component_overlay/{channel}_{topology}.{png,pdf}   per-region overlay
  component_overlay/all_regions.{png,pdf}            2x2 overlay grid
  stack/{channel}_{topology}.{png,pdf}               per-region stacked plot
  stack/all_regions.{png,pdf}                         2x2 stack grid
  individual/{process}/{channel}_{topology}.{png,pdf} single-process plots
  component_yields.csv                               integrals per component

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

Usage:
  python plot_mc_backgrounds.py -v
  python plot_mc_backgrounds.py --plots stack --era RunIII2024Summer24 --dir 20260317_lo_dy
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from wrplotter.cli_utils import setup_logging
from wrplotter.config import load_lumi
from wrplotter.paths import input_dirs_for_era, repo_root
from wrplotter.plotting_helpers import custom_log_formatter
from wrplotter.sample_groups import load_sample_groups

logger = logging.getLogger(__name__)

# A histogram is carried around as the tuple (edges, values, variances).
Hvv = "tuple[np.ndarray, np.ndarray, np.ndarray]"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# (channel, topology) — the four signal regions, in plot order.
REGIONS: list[tuple[str, str]] = [
    ("ee", "resolved"),
    ("mumu", "resolved"),
    ("ee", "boosted"),
    ("mumu", "boosted"),
]

# Mass observable / axis label per topology (see module docstring).
MASS_VAR = {"resolved": "mass_fourobject", "boosted": "mass_twoobject"}
MASS_LABEL = {
    "resolved": r"$m_{\ell\ell jj}$ [GeV]",
    "boosted": r"$m_{\ell J}$ [GeV]",
}

CH_LAB = {"ee": "ee", "mumu": r"$\mu\mu$"}
TOPO_LAB = {"resolved": "Resolved SR", "boosted": "Boosted SR"}

# Default x-range per topology. Lower edge matches the SR mass threshold / the
# analysis plot configs (data/plot_settings); below it the SR is empty.
TOPO_XMIN = 800.0
TOPO_XMAX = {"resolved": 5000.0, "boosted": 3500.0}

# Legend labels per sample-group key (colors come from sample_groups.yaml).
PROCESS_LABEL = {
    "dy": "DY+jets",
    "ttbar": r"$t\bar{t}$+tW",
    "nonprompt": "Nonprompt",
    "other": "Other",
}

TOTAL_COLOR = "#000000"

# Stat-uncertainty band style (same hatched band as plot_stack).
ERRPS = {"hatch": "////", "facecolor": "none", "lw": 0, "edgecolor": "k", "alpha": 0.5}


# ---------------------------------------------------------------------------
# Histogram I/O
# ---------------------------------------------------------------------------

def build_region_name(channel: str, topology: str) -> str:
    return f"wr_{channel}_{topology}_sr"


def load_component(input_dirs, samples, region, variable):
    """Sum the histograms for `samples` across all sub-era input dirs.

    Returns (edges, values, variances), or None if nothing was found.
    """
    hist_key = f"{region}/{variable}_{region}"
    edges = values = variances = None
    for d in input_dirs:
        for sample in samples:
            fpath = d / f"WRAnalyzer_{sample}.root"
            if not fpath.exists():
                logger.warning("Missing file: %s", fpath)
                continue
            try:
                h = uproot.open(fpath)[hist_key]
            except (KeyError, OSError) as exc:
                logger.warning("Missing '%s' in %s: %s", hist_key, fpath.name, exc)
                continue
            e, v, var = h.axes[0].edges(), h.values(), h.variances()
            if edges is None:
                edges, values, variances = e, v.copy(), var.copy()
            else:
                values += v
                variances += var
    if edges is None:
        return None
    return edges, values, variances


def rebin(edges, values, variances, factor):
    """Merge `factor` adjacent bins (drops a partial trailing group)."""
    if factor <= 1:
        return edges, values, variances
    n_new = len(values) // factor
    v = values[: n_new * factor].reshape(n_new, factor).sum(axis=1)
    var = variances[: n_new * factor].reshape(n_new, factor).sum(axis=1)
    e = edges[:: factor][: n_new + 1]
    return e, v, var


def sum_hvv(items):
    """Sum a list of (edges, values, variances) sharing the same edges."""
    items = [x for x in items if x is not None]
    if not items:
        return None
    edges = items[0][0]
    values = items[0][1].copy()
    variances = items[0][2].copy()
    for _, v, var in items[1:]:
        values += v
        variances += var
    return edges, values, variances


# ---------------------------------------------------------------------------
# Shared axis cosmetics
# ---------------------------------------------------------------------------

def compute_ylim(hvvs, xlim):
    """Log-scale y-range from a list of histograms within the visible x-range.

    Top is set off the largest curve; floor is the smallest positive bin across
    all inputs, clamped to at most ~6 decades below the peak.
    """
    peak = 0.0
    smallest = np.inf
    for edges, values, _ in hvvs:
        centers = 0.5 * (edges[:-1] + edges[1:])
        mask = (centers >= xlim[0]) & (centers <= xlim[1])
        vis = values[mask]
        pos = vis[vis > 0]
        if pos.size:
            peak = max(peak, float(vis.max()))
            smallest = min(smallest, float(pos.min()))
    if peak <= 0:
        return 1e-3, 1.0
    ymin = max(smallest / 3.0, peak * 1e-6)
    ymax = peak * 30.0
    return ymin, ymax


def style_axes(ax, channel, topology, *, bin_width, xlim, ylim, era, info_fontsize):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(MASS_LABEL[topology])
    ax.set_ylabel(f"Events / {bin_width:.0f} GeV")
    ax.text(
        0.05, 0.95,
        f"{CH_LAB[channel]}\n{TOPO_LAB[topology]}\n{era}",
        transform=ax.transAxes, fontsize=info_fontsize, va="top",
    )


def add_cms_label(ax, lumi, com, fontsize):
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=fontsize)


def draw_stat_band(ax, edges, values, variances, *, label=None):
    """Hatched ±stat band around `values` (mplhep here lacks histtype='band').

    Same look as plot_stack's uncertainty band; empty bins are left blank and
    the lower edge is clamped positive so it renders on a log axis.
    """
    err = np.sqrt(variances)
    pos = values[values > 0]
    floor = (float(pos.min()) * 1e-3) if pos.size else 1e-9
    empty = values <= 0
    hi = np.where(empty, np.nan, values + err)
    lo = np.where(empty, np.nan, np.maximum(values - err, floor))
    ax.stairs(hi, edges, baseline=lo, label=label, **ERRPS)


def save(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Overlay (step curves)
# ---------------------------------------------------------------------------

def draw_overlay(ax, channel, topology, comps, *, bin_width, xlim, ylim, era,
                 info_fontsize, legend_fontsize):
    """comps = list of (label, hvv, color); first entry is the total."""
    for label, (edges, values, _), color in comps:
        is_total = label.startswith("Total")
        ax.stairs(values, edges, color=color,
                  linewidth=2.6 if is_total else 1.7,
                  zorder=5 if is_total else 3, label=label)
    style_axes(ax, channel, topology, bin_width=bin_width, xlim=xlim, ylim=ylim,
               era=era, info_fontsize=info_fontsize)
    ax.legend(loc="upper right", fontsize=legend_fontsize, framealpha=0.85)


# ---------------------------------------------------------------------------
# Stack (filled, stacked processes)
# ---------------------------------------------------------------------------

def draw_stack(ax, channel, topology, stacked, total, *, bin_width, xlim, ylim,
               era, info_fontsize, legend_fontsize):
    """stacked = list of (label, hvv, color) bottom->top; total is the sum hvv."""
    edges = stacked[0][1][0]
    values_list = [s[1][1] for s in stacked]
    colors = [s[2] for s in stacked]
    labels = [s[0] for s in stacked]
    hep.histplot(values_list, bins=edges, stack=True, histtype="fill",
                 alpha=0.7, color=colors, label=labels, ax=ax)

    t_edges, t_vals, t_var = total
    draw_stat_band(ax, t_edges, t_vals, t_var, label="MC stat. unc.")

    style_axes(ax, channel, topology, bin_width=bin_width, xlim=xlim, ylim=ylim,
               era=era, info_fontsize=info_fontsize)

    # Legend: stat band first, then processes top-of-stack first (reverse draw order).
    handles, lbls = ax.get_legend_handles_labels()
    n = len(stacked)
    order = [n] + list(range(n - 1, -1, -1))  # band, then reversed processes
    order = [i for i in order if i < len(handles)]
    ax.legend([handles[i] for i in order], [lbls[i] for i in order],
              loc="upper right", fontsize=legend_fontsize, framealpha=0.85)


# ---------------------------------------------------------------------------
# Individual (single process, filled)
# ---------------------------------------------------------------------------

def draw_individual(ax, channel, topology, label, hvv, color, *, bin_width, xlim,
                    ylim, era, info_fontsize, legend_fontsize):
    edges, values, variances = hvv
    hep.histplot(values, bins=edges, histtype="fill", alpha=0.7, color=color,
                 label=label, ax=ax)
    hep.histplot(values, bins=edges, histtype="step", color=color, lw=1.5, ax=ax)
    draw_stat_band(ax, edges, values, variances, label="MC stat. unc.")
    style_axes(ax, channel, topology, bin_width=bin_width, xlim=xlim, ylim=ylim,
               era=era, info_fontsize=info_fontsize)
    ax.legend(loc="upper right", fontsize=legend_fontsize, framealpha=0.85)


# ---------------------------------------------------------------------------
# Per-region figure wrappers + grids
# ---------------------------------------------------------------------------

def plot_single(draw_fn, channel, topology, out_path, *, ylim_inputs, xlim,
                lumi, com, era, bin_width, draw_kwargs):
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    ylim = compute_ylim(ylim_inputs, xlim)
    draw_fn(ax, channel, topology, **draw_kwargs, bin_width=bin_width, xlim=xlim,
            ylim=ylim, era=era, info_fontsize=16, legend_fontsize=14)
    add_cms_label(ax, lumi, com, fontsize=18)
    save(fig, out_path)
    logger.info("  wrote %s", out_path.with_suffix(".png"))


def plot_grid(draw_fn, per_region, out_path, *, xlims, lumi, com, era, bin_width,
              ylim_input_fn, draw_kwargs_fn):
    """2x2: resolved (top row) / boosted (bottom), ee (left) / mumu (right)."""
    hep.style.use("CMS")
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    layout = [
        [("ee", "resolved"), ("mumu", "resolved")],
        [("ee", "boosted"), ("mumu", "boosted")],
    ]
    for r, row in enumerate(layout):
        for c, key in enumerate(row):
            ax = axes[r][c]
            data = per_region.get(key)
            if not data:
                ax.set_visible(False)
                continue
            xlim = xlims[key[1]]
            ylim = compute_ylim(ylim_input_fn(data), xlim)
            draw_fn(ax, key[0], key[1], **draw_kwargs_fn(data), bin_width=bin_width,
                    xlim=xlim, ylim=ylim, era=era, info_fontsize=15,
                    legend_fontsize=13)
            add_cms_label(ax, lumi, com, fontsize=15)
    fig.tight_layout()
    save(fig, out_path)
    logger.info("  wrote %s", out_path.with_suffix(".png"))


# ---------------------------------------------------------------------------
# Yields table
# ---------------------------------------------------------------------------

def integral_in_range(edges, values, xlim):
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (centers >= xlim[0]) & (centers <= xlim[1])
    return float(values[mask].sum())


def write_yields_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["channel", "topology", "component",
                                          "yield_full", "yield_in_range"])
        w.writeheader()
        w.writerows(rows)
    logger.info("  wrote %s", out_path)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIII2024Summer24")
    p.add_argument("--dir", default="20260317_lo_dy",
                   help="Subdir under rootfiles/<run>/<year>/<era>/.")
    p.add_argument("--plots", nargs="+", default=["overlay", "stack", "individual"],
                   choices=["overlay", "stack", "individual"],
                   help="Which plot types to make. Default: all three.")
    p.add_argument("--bin-width", type=float, default=100.0,
                   help="Display bin width in GeV (rebinned from native). Default 100.")
    p.add_argument("--xmin", type=float, default=TOPO_XMIN,
                   help=f"Lower x-limit (GeV) for all plots. Default {TOPO_XMIN:.0f} "
                        "(the SR mass threshold).")
    p.add_argument("--xmax", type=float, default=None,
                   help="Upper x-limit (GeV). Default: per-topology "
                        f"({TOPO_XMAX['resolved']:.0f} resolved, "
                        f"{TOPO_XMAX['boosted']:.0f} boosted).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: <script dir>/<run2|run3>, chosen by --era.")
    p.add_argument("--no-grid", action="store_true",
                   help="Skip the 2x2 summary grids (overlay/stack).")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def mc_process_keys(groups, order):
    """MC group keys in bottom->top stack order (stack_position='top' last)."""
    keys = [k for k in order if groups[k].kind == "mc"]
    top = [k for k in keys if groups[k].stack_position == "top"]
    return [k for k in keys if k not in top] + top


def main():
    args = parse_args()
    setup_logging(args.verbose)
    # Keep -v focused on this script; mute matplotlib's font/locator debug spam.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    era = args.era
    info = load_lumi(era)
    lumi = info["lumi"]
    com = info.get("com", 13.6)
    if args.output_dir is None:
        args.output_dir = (Path(__file__).resolve().parent
                           / {"RunII": "run2", "Run3": "run3"}[str(info["run"])])

    input_dirs, _ = input_dirs_for_era(era, repo_root(), args.dir)
    logger.info("Era %s — input dirs: %s", era, [str(d) for d in input_dirs])

    groups, order = load_sample_groups()
    mc_keys = mc_process_keys(groups, order)  # bottom->top, e.g. other,nonprompt,ttbar,dy
    logger.info("MC stack order (bottom->top): %s", mc_keys)

    native = 10.0  # native bin width (GeV) of the input histograms
    factor = max(1, round(args.bin_width / native))
    bin_width = native * factor

    xlims = {
        topo: (args.xmin, args.xmax if args.xmax is not None else TOPO_XMAX[topo])
        for topo in ("resolved", "boosted")
    }

    out = args.output_dir
    region_overlay: dict[tuple[str, str], list] = {}
    region_stack: dict[tuple[str, str], tuple] = {}
    yield_rows: list[dict] = []

    for channel, topology in REGIONS:
        region = build_region_name(channel, topology)
        variable = MASS_VAR[topology]
        xlim = xlims[topology]
        logger.info("Region %s  (%s)", region, variable)

        # Load each MC process once, then derive total and grouped components.
        base = {}
        for k in mc_keys:
            loaded = load_component(input_dirs, groups[k].samples, region, variable)
            if loaded is None:
                logger.warning("  No histogram for process '%s' in %s", k, region)
                continue
            base[k] = rebin(*loaded, factor)
        if not base:
            logger.warning("  Region %s has no MC — skipping", region)
            continue

        total = sum_hvv(list(base.values()))

        # Grouped components for the overlay + yields (total, DY, tt+tW, nonprompt+other).
        overlay_comps = [("Total background", total, TOTAL_COLOR)]
        if "dy" in base:
            overlay_comps.append((PROCESS_LABEL["dy"], base["dy"], groups["dy"].color))
        if "ttbar" in base:
            overlay_comps.append((PROCESS_LABEL["ttbar"], base["ttbar"], groups["ttbar"].color))
        np_other = sum_hvv([base.get("nonprompt"), base.get("other")])
        if np_other is not None:
            overlay_comps.append(("Nonprompt + other", np_other, groups["nonprompt"].color))

        for label, (edges, values, _), _ in overlay_comps:
            yield_rows.append({
                "channel": channel, "topology": topology, "component": label,
                "yield_full": round(float(values.sum()), 4),
                "yield_in_range": round(integral_in_range(edges, values, xlim), 4),
            })
            logger.info("  %-18s integral=%.2f (in %.0f-%.0f: %.2f)",
                        label, values.sum(), xlim[0], xlim[1],
                        integral_in_range(edges, values, xlim))

        region_overlay[(channel, topology)] = overlay_comps

        # Stacked processes, bottom->top.
        stacked = [(PROCESS_LABEL[k], base[k], groups[k].color)
                   for k in mc_keys if k in base]
        region_stack[(channel, topology)] = (stacked, total)

        # ---- per-region figures ----
        if "overlay" in args.plots:
            plot_single(
                draw_overlay, channel, topology,
                out / "component_overlay" / f"{channel}_{topology}",
                ylim_inputs=[c[1] for c in overlay_comps], xlim=xlim,
                lumi=lumi, com=com, era=era, bin_width=bin_width,
                draw_kwargs=dict(comps=overlay_comps),
            )
        if "stack" in args.plots:
            plot_single(
                draw_stack, channel, topology,
                out / "stack" / f"{channel}_{topology}",
                ylim_inputs=[total], xlim=xlim,
                lumi=lumi, com=com, era=era, bin_width=bin_width,
                draw_kwargs=dict(stacked=stacked, total=total),
            )
        if "individual" in args.plots:
            for k in mc_keys:
                if k not in base:
                    continue
                plot_single(
                    draw_individual, channel, topology,
                    out / "individual" / k / f"{channel}_{topology}",
                    ylim_inputs=[base[k]], xlim=xlim,
                    lumi=lumi, com=com, era=era, bin_width=bin_width,
                    draw_kwargs=dict(label=PROCESS_LABEL[k], hvv=base[k],
                                     color=groups[k].color),
                )

    if not region_overlay:
        logger.error("No regions produced any histograms. Check --era/--dir.")
        sys.exit(1)

    # ---- summary grids ----
    if not args.no_grid:
        if "overlay" in args.plots:
            plot_grid(
                draw_overlay, region_overlay, out / "component_overlay" / "all_regions",
                xlims=xlims, lumi=lumi, com=com, era=era, bin_width=bin_width,
                ylim_input_fn=lambda comps: [c[1] for c in comps],
                draw_kwargs_fn=lambda comps: dict(comps=comps),
            )
        if "stack" in args.plots:
            plot_grid(
                draw_stack, region_stack, out / "stack" / "all_regions",
                xlims=xlims, lumi=lumi, com=com, era=era, bin_width=bin_width,
                ylim_input_fn=lambda d: [d[1]],
                draw_kwargs_fn=lambda d: dict(stacked=d[0], total=d[1]),
            )

    write_yields_csv(yield_rows, out / "component_yields.csv")
    logger.info("Done. Outputs in %s", out)


if __name__ == "__main__":
    main()
