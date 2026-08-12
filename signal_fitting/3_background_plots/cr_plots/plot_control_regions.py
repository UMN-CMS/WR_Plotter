#!/usr/bin/env python3
"""Stage 3 — data/MC validation plots for the DY and flavor control regions.

The background fit is validated where there is data: the DY control region
(Z-peak enriched) and the flavor (e-mu) control region (tt/tW enriched). For
each CR this draws the standard CMS plot — stacked, shaded MC with data points
overlaid and a Data/Sim. ratio panel — for the reconstructed W_R mass observable
(resolved → m_lljj `mass_fourobject`; boosted → m_lJ `mass_twoobject`).

CRs are unblinded, so data is shown. The flavor CR is an e-mu region recorded in
both the EGamma and Muon datasets; it is plotted once per dataset (the MC is the
same, only the overlaid data differs), matching the analysis.

This reuses the analysis plotting stack directly:
  wrplotter.plotting_helpers.plot_stack   (stack + data + ratio, CMS style)
  wrplotter.histogram_cache.load_and_rebin (per-sample load, k-factors, overflow)
  region / variable / plot-config / sample-group loaders
so binning (data/plot_settings), colors, ordering, and blinding all match
bin/make_stackplots.py. The MC uncertainty band is statistical only (no
systematics collected here).

Regions plotted (DY CR + flavor CR; SR excluded):
  wr_{ee,mumu}_{resolved,boosted}_dy_cr
  wr_resolved_flavor_cr, wr_{emu,mue}_boosted_flavor_cr   (× egamma, muon data)

Outputs (co-located with this script):
  dy_cr/{region}_{dataset}.{png,pdf}
  flavor_cr/{region}_{dataset}.{png,pdf}

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

Usage:
  python plot_control_regions.py -v
  python plot_control_regions.py --era RunIII2024Summer24 --dir 20260317_lo_dy
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

from wrplotter.cli_utils import setup_logging
from wrplotter.config import (load_lumi, load_plot_settings, index_plot_settings,
                              get_var_cfg, load_kfactors, get_kfactor)
from wrplotter.histogram_cache import load_and_rebin
from wrplotter.paths import input_dirs_for_era, repo_root
from wrplotter.plotting_helpers import (custom_log_formatter, set_y_label,
                                        reorder_legend, _safe_sum_hists)
from wrplotter.regions import regions_for_era
from wrplotter.sample_groups import load_sample_groups
from wrplotter.variables import build_variables

logger = logging.getLogger(__name__)

# Mass observable per topology (the reconstructed W_R mass proxy).
MASS_VAR = {True: "mass_twoobject", False: "mass_fourobject"}  # keyed by is_boosted

# Hatched MC stat-uncertainty band style (same as plot_stack).
ERRPS = {"hatch": "////", "facecolor": "none", "lw": 0, "edgecolor": "k", "alpha": 0.5}


def plot_data_mc(region, variable, *, stack_list, stack_colors, stack_labels,
                 data_hist, xlim, ylim, ratio_ylim, lumi, com):
    """Stacked MC + data + Data/Sim. ratio (stat-only band).

    A local, LCG_106-compatible rendering of bin/make_stackplots' plot: the
    analysis plot_stack draws its band with hep.histplot(histtype='band'),
    which this mplhep lacks, so the band is drawn manually via ax.stairs.
    Reuses plot_stack's helpers for everything else (log ticks, y-label,
    legend order).
    """
    tot = _safe_sum_hists(list(stack_list or []))
    data = _safe_sum_hists(list(data_hist or []))
    if tot is None:
        raise ValueError(f"No MC for {region.name}")
    edges = tot.axes[0].edges

    hep.style.use("CMS")
    fig, (ax, rax) = plt.subplots(
        2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), sharex=True)

    # main panel: stacked MC + data
    hep.histplot(list(stack_list), stack=True, label=stack_labels,
                 color=stack_colors, histtype="fill", alpha=0.7, ax=ax, flow="hint")
    if data is not None:
        hep.histplot(data, label="Data", xerr=True, color="k",
                     histtype="errorbar", ax=ax, flow="hint")

    # MC stat band (manual; empty bins blank, lower edge clamped for log axis)
    tot_vals = tot.values()
    tot_errs = np.sqrt(tot.variances())
    pos = tot_vals[tot_vals > 0]
    floor = (float(pos.min()) * 1e-3) if pos.size else 1e-9
    empty = tot_vals <= 0
    hi = np.where(empty, np.nan, tot_vals + tot_errs)
    lo = np.where(empty, np.nan, np.maximum(tot_vals - tot_errs, floor))
    ax.stairs(hi, edges, baseline=lo, label="MC stat. unc.", **ERRPS)

    # ratio panel
    rel = np.divide(tot_errs, tot_vals, out=np.zeros_like(tot_vals, dtype=float),
                    where=tot_vals > 0)
    if data is not None:
        data_vals = data.values()
        ratio = np.divide(data_vals, tot_vals,
                          out=np.zeros_like(data_vals, dtype=float), where=tot_vals > 0)
        ratio_err = np.divide(np.sqrt(data_vals), tot_vals,
                              out=np.zeros_like(data_vals, dtype=float), where=tot_vals > 0)
        hep.histplot(ratio, edges, yerr=ratio_err, xerr=False, ax=rax,
                     histtype="errorbar", color="k", capsize=4)
    band_low, band_high = 1 - rel, 1 + rel
    bad = tot_vals <= 0
    band_low[bad] = np.nan
    band_high[bad] = np.nan
    rax.stairs(band_low, edges, baseline=band_high, **ERRPS)
    rax.axhline(1.0, ls="--", color="k")

    # cosmetics
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.text(0.05, 0.96, region.tlatex_alias.replace("\\n", "\n"),
            transform=ax.transAxes, fontsize=20, va="top")
    hep.cms.label(loc=0, ax=ax, data=region.unblind_data, label="Work in Progress",
                  lumi=f"{lumi:.2f}", com=com, fontsize=20)
    xlabel = (f"{variable.tlatex_alias} [{variable.unit}]"
              if getattr(variable, "unit", "") else variable.tlatex_alias)
    rax.set_xlabel(xlabel)
    ax.set_xlabel("")
    rax.set_ylabel("Data/Sim.")
    rax.set_ylim(*ratio_ylim)
    set_y_label(ax, tot, variable)
    reorder_legend(ax, fontsize=18)
    return fig


def is_control_region(region) -> bool:
    return ("_dy_cr" in region.name) or ("_flavor_cr" in region.name)


def cr_subdir(region) -> str:
    return "flavor_cr" if "_flavor_cr" in region.name else "dy_cr"


def build_stack(region, variable, *, rebin, xmax, input_dirs, input_lumis, era,
                scales, sample_groups, density):
    """Build (stack_list, colors, labels, data_hist) for one region/variable.

    Mirrors the per-(region, variable) loop of bin/make_stackplots.py: data
    groups are restricted to the region's primary dataset, MC groups stacked,
    and stack_position='top' groups moved to the top of the visual stack (bottom
    in flavor CRs).
    """
    hist_key = f"{region.name}/{variable.name}_{region.name}"

    stack_list, stack_colors, stack_labels, stack_sgs = [], [], [], []
    data_hist = []

    for sg in sample_groups:
        is_data = getattr(sg, "kind", "mc") == "data"
        if is_data and sg.key != region.primary_dataset:
            continue

        combined = None
        for sample in sg.samples:
            hist_obj = load_and_rebin(
                input_dirs=input_dirs, sample=sample, hist_key=hist_key,
                n_rebin=rebin, sublumis=input_lumis,
                era_for_scale=era, get_kfactor_fn=get_kfactor, scales=scales,
                density=density, xmax=xmax,
            )
            if hist_obj is None:
                continue
            has_ov = hist_obj.view(flow=True)["value"][-1] > 0
            combined = hist_obj if combined is None else (combined + hist_obj)
            if has_ov:
                combined.view(flow=True)["value"][-1] = 1.0
        if combined is None:
            continue

        if is_data:
            data_hist.append(combined)
        else:
            stack_list.append(combined)
            stack_colors.append(getattr(sg, "color", "#000000"))
            stack_labels.append(getattr(sg, "tlatex_alias", sg.key))
            stack_sgs.append(sg)

    # Reorder by stack_position (top -> top of stack; bottom of stack in flavor CRs).
    if stack_sgs:
        items = list(zip(stack_list, stack_colors, stack_labels, stack_sgs))
        top_idxs = [i for i, it in enumerate(items) if it[3].stack_position == "top"]
        for i in sorted(top_idxs, reverse=True):
            top_item = items.pop(i)
            if "flavor_cr" in region.name:
                items.insert(0, top_item)
            else:
                items.append(top_item)
        if items:
            stack_list, stack_colors, stack_labels, stack_sgs = (
                [list(x) for x in zip(*items)]
            )

    return stack_list, stack_colors, stack_labels, data_hist


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIII2024Summer24")
    p.add_argument("--dir", default="20260317_lo_dy",
                   help="Subdir under rootfiles/<run>/<year>/<era>/.")
    p.add_argument("--plot-config", default=None,
                   help="Plot-settings YAML or era key (default: --era).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Default: <script dir>/<run2|run3>, chosen by --era.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    era = args.era
    info = load_lumi(era)
    lumi, com = info["lumi"], info.get("com", 13.6)
    if args.output_dir is None:
        args.output_dir = (Path(__file__).resolve().parent
                           / {"RunII": "run2", "Run3": "run3"}[str(info["run"])])

    input_dirs, input_lumis = input_dirs_for_era(era, repo_root(), args.dir)
    logger.info("Era %s — input dirs: %s", era, [str(d) for d in input_dirs])

    groups, order = load_sample_groups()
    sample_groups = [groups[k] for k in order if k in groups]
    scales = load_kfactors()
    name_to_var = {v.name: v for v in build_variables()}

    region_cfgs, common_vars = index_plot_settings(
        load_plot_settings(args.plot_config or era))

    regions = [r for r in regions_for_era(era) if is_control_region(r)]
    logger.info("Plotting %d CR region/dataset variants", len(regions))

    n_saved = 0
    for region in regions:
        var_name = MASS_VAR[region.is_boosted]
        variable = name_to_var.get(var_name)
        if variable is None:
            logger.warning("  Variable %s not defined — skipping %s", var_name, region.name)
            continue

        vcfg = get_var_cfg(region_cfgs, common_vars, region.name, var_name) or {}
        rebin = vcfg.get("rebin", 1)
        xmin, xmax = map(float, vcfg.get("xlim", (0.0, 8000.0)))
        ymin, ymax = map(float, vcfg.get("ylim", (1.0, 1e6)))
        ratio_ylim = tuple(map(float, vcfg.get("ratio_ylim", (0.5, 2.0))))

        stack_list, stack_colors, stack_labels, data_hist = build_stack(
            region, variable, rebin=rebin, xmax=xmax,
            input_dirs=input_dirs, input_lumis=input_lumis,
            era=era, scales=scales, sample_groups=sample_groups, density=True,
        )
        if not stack_list and not data_hist:
            logger.warning("  %s [%s] — no histograms, skipping",
                           region.name, region.primary_dataset)
            continue

        fig = plot_data_mc(
            region, variable,
            stack_list=stack_list, stack_colors=stack_colors,
            stack_labels=stack_labels, data_hist=data_hist,
            xlim=(xmin, xmax), ylim=(ymin, ymax),
            ratio_ylim=ratio_ylim, lumi=lumi, com=com,
        )

        out_path = (args.output_dir / cr_subdir(region) /
                    f"{region.name}_{region.primary_dataset}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        n_saved += 1
        logger.info("  wrote %s  (%s)", out_path.with_suffix(".png").name, var_name)

    if n_saved == 0:
        logger.error("No CR plots produced. Check --era/--dir.")
        sys.exit(1)
    logger.info("Done — %d CR plot(s). Output: %s", n_saved, args.output_dir)


if __name__ == "__main__":
    main()
