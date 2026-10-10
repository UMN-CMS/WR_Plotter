#!/usr/bin/env python3
"""Background stack in one analysis window: SR (ee or mumu) and the e-mu
flavor CR, house style (Stage-3 build_stack + plot_data_mc, card binning),
k5 window shaded, per-background window yields annotated, and the
per-window transfer factor R = SR tt / CR tt.

  python plot_window_stack.py --channel mumu --mass 3000
      -> explain_fcr/window_stack_mumu_m3000_{sr,fcr}.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = Path(__file__).resolve().parent          # 8_combine_limits/optimization
SIGFIT = HERE.parents[1]
sys.path.insert(0, str(SIGFIT.parent))          # repo root
sys.path.insert(0, str(SIGFIT / "3_background_plots" / "cr_plots"))
sys.path.insert(0, str(HERE))

from wrplotter.cli_utils import setup_logging                           # noqa: E402
from wrplotter.config import (load_lumi, load_plot_settings,            # noqa: E402
                              index_plot_settings, get_var_cfg)
from wrplotter.paths import input_dirs_for_era, repo_root               # noqa: E402
from wrplotter.regions import regions_for_era                           # noqa: E402
from wrplotter.sample_groups import load_sample_groups                  # noqa: E402
from wrplotter.variables import build_variables                         # noqa: E402

import uproot                                                           # noqa: E402
from plot_control_regions import build_stack, plot_data_mc              # noqa: E402
from plot_transfer_ratio import k5_window                               # noqa: E402

logger = logging.getLogger("plot_window_stack")

ERA, BKG_DIR = "RunIISummer20UL18", "20260714_run2_bkgs"
FCR_REGION = "wr_resolved_flavor_cr"
SR_REGIONS = {"ee": "wr_ee_resolved_sr", "mumu": "wr_mumu_resolved_sr"}
# window-yield rows: (label, samples)
YIELD_ROWS = [(r"$t\bar{t}$+tW", ["tt_tW"]), ("DY", ["DYJets"]),
              ("nonprompt", ["Nonprompt"]), ("other", ["Other"])]


def window_yield(bkg_dirs, region, samples, lo, hi):
    """Summed MC yield of `samples` in `region` with bin centers in
    [lo, hi] (native 10 GeV bins)."""
    tot = 0.0
    for smp in samples:
        for d in bkg_dirs:
            fp = d / f"WRAnalyzer_{smp}.root"
            if not fp.exists():
                continue
            h = uproot.open(fp)[f"{region}/mass_fourobject_{region}"]
            e, v = h.axes[0].edges(), h.values()
            c = 0.5 * (e[:-1] + e[1:])
            m = (c >= lo) & (c <= hi) & np.isfinite(v)
            tot += float(v[m].sum())
    return tot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--channel", choices=list(SR_REGIONS), default="mumu")
    p.add_argument("--mass", type=int, default=3000)
    p.add_argument("--k", type=float, default=5.0)
    p.add_argument("--bw", type=float, default=50.0, help="card bin width")
    p.add_argument("--xmax", type=float, default=4500.0)
    p.add_argument("--output-dir", type=Path, default=HERE / "explain_fcr")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    info = load_lumi(ERA)
    lumi, com = info["lumi"], info.get("com", 13.6)
    bkg_dirs, input_lumis = input_dirs_for_era(ERA, repo_root(), BKG_DIR)
    sr_region = SR_REGIONS[args.channel]

    # the k5 window from the Stage-2 parameterization, snapped to the card grid
    with open(SIGFIT / "2_width_parameterization" / "wr"
              / "window_params.json") as fh:
        wpar = json.load(fh)[args.channel]["resolved"]
    h0 = uproot.open(bkg_dirs[0] / "WRAnalyzer_tt_tW.root")[
        f"{sr_region}/mass_fourobject_{sr_region}"]
    slo, shi = k5_window(float(args.mass), wpar, h0.axes[0].edges(),
                         bw=args.bw, k=args.k)
    logger.info("%s m=%d: k%.0f window [%.0f, %.0f]", args.channel,
                args.mass, args.k, slo, shi)

    # window yields per background, both regions
    yields = {}
    for rname in (sr_region, FCR_REGION):
        yields[rname] = {lab: window_yield(bkg_dirs, rname, smps, slo, shi)
                         for lab, smps in YIELD_ROWS}
        tot = sum(yields[rname].values())
        logger.info("%s window yields: %s  total %.1f", rname,
                    {k: round(v, 1) for k, v in yields[rname].items()}, tot)
    r_win = (yields[sr_region][YIELD_ROWS[0][0]]
             / yields[FCR_REGION][YIELD_ROWS[0][0]])
    logger.info("per-window R = %.3f", r_win)

    # house-style stacks
    groups, order = load_sample_groups()
    all_groups = [groups[k] for k in order if k in groups]
    mc_groups = [g for g in all_groups if getattr(g, "kind", "mc") != "data"]
    variable = {vv.name: vv for vv in build_variables()}["mass_fourobject"]
    region_cfgs, common_vars = index_plot_settings(load_plot_settings(ERA))
    rebin = max(1, round(args.bw / 10.0))
    regions = {}
    for r in regions_for_era(ERA):
        if r.name == sr_region and sr_region not in regions:
            regions[sr_region] = r
        if (r.name == FCR_REGION and (FCR_REGION not in regions
                or "egamma" in str(r.primary_dataset).lower())):
            regions[FCR_REGION] = r

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, rname in (("sr", sr_region), ("fcr", FCR_REGION)):
        region = regions[rname]
        vcfg = get_var_cfg(region_cfgs, common_vars,
                           region.name, variable.name) or {}
        ylim = tuple(map(float, vcfg.get("ylim", (1e-2, 1e6))))
        ratio_ylim = tuple(map(float, vcfg.get("ratio_ylim", (0.5, 2.0))))
        sgs = mc_groups if rname == sr_region else all_groups  # SR blind
        stack_list, stack_colors, stack_labels, data_hist = build_stack(
            region, variable, rebin=rebin, xmax=args.xmax,
            input_dirs=bkg_dirs, input_lumis=input_lumis,
            era=ERA, sample_groups=sgs, density=True)
        fig = plot_data_mc(
            region, variable, stack_list=stack_list,
            stack_colors=stack_colors, stack_labels=stack_labels,
            data_hist=data_hist, xlim=(800.0, args.xmax), ylim=ylim,
            ratio_ylim=ratio_ylim, lumi=lumi, com=com)
        ax = fig.axes[0]
        for a in fig.axes[:2]:
            a.axvspan(slo, shi, color="gold", alpha=0.15, zorder=0)
        y = yields[rname]
        tot = sum(y.values())
        lines = [fr"$m_{{W_R}} = {args.mass}$, window [{slo:.0f}, {shi:.0f}]:"]
        for lab, _ in YIELD_ROWS:
            lines.append(f"    {lab}  {y[lab]:.1f}  ({y[lab] / tot:.0%})")
        lines.append(f"    total  {tot:.1f}")
        if rname == FCR_REGION:
            lines.append(fr"$R$ = SR $t\bar{{t}}$ / CR $t\bar{{t}}$ = "
                         fr"{yields[sr_region][YIELD_ROWS[0][0]]:.1f} / "
                         fr"{y[YIELD_ROWS[0][0]]:.1f} = {r_win:.3f}")
        ax.text(0.60, 0.58, "\n".join(lines), transform=ax.transAxes,
                va="top", ha="left", fontsize=13)
        out = args.output_dir / f"window_stack_{args.channel}_m{args.mass}_{suffix}"
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
        logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
