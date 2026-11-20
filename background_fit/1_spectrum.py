#!/usr/bin/env python3
"""Step 1: fix the spectrum on the raw MC.

For each channel and DY variant: yields, effective MC counts, last event, negative bins
and the yields above 3 / 4 / 5 TeV per component (the spread of the high-mass prediction
between DY samples); the threshold scan (the Pearson chi2 of the exponential and of the
power x exponential on the summed spectrum for fit starts 800-1200 GeV); and the
spectrum plots. Writes results/<channel>/1_spectrum/{spectrum.csv, threshold_scan.csv,
summary.md, plots/}.

  cd WR_Plotter && python background_fit/1_spectrum.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))
import config as C                                                       # noqa: E402
import plots as P                                                        # noqa: E402
from fits import bin_weights, fit_all, pearson_at                        # noqa: E402
from spectrum import (COMPONENT_COLOR, COMPONENT_LABEL, centers,         # noqa: E402
                      last_populated, load_components, rebin, summary_rows)
from util import write_csv                                               # noqa: E402
from wrplotter.cli_utils import setup_logging                            # noqa: E402

logger = logging.getLogger("1_spectrum")


def plot_variants(channel, spectra, out):
    """The summed spectrum and the DY component for every variant, on one figure each."""
    for what in ["sum", "DYJets"]:
        fig, (ax, axr) = P.spectrum_figure()
        ref = None
        colors = ["#000000", "#5790fc", "#e42536", "#f89c20", "#964a8b", "#2ca02c"]
        for i, (variant, comps) in enumerate(spectra.items()):
            e, v, var = rebin(*comps[what])
            c = centers(e)
            sel = c >= C.THRESHOLD
            lab = C.DY_SHORT[variant]                 # the region text names the sum or the DY component
            ax.step(e[:-1][sel], np.maximum(v[sel], P.YMIN), where="post", color=colors[i % 6], lw=1.5, label=lab)
            if ref is None:
                ref = v
                rel = np.where(v > 0, np.sqrt(np.maximum(var, 0)) / np.where(v > 0, v, 1), 0)
                axr.fill_between(c[sel], 1 - rel[sel], 1 + rel[sel], step="mid", hatch="////",
                                 facecolor="none", edgecolor="k", lw=0, alpha=0.5)
            else:
                ratio = np.where(ref > 0, v / np.where(ref > 0, ref, np.nan), np.nan)
                axr.stairs(ratio[sel], np.append(e[:-1][sel], e[1:][sel][-1]), color=colors[i % 6], lw=1.4)   # a histogram ratio: steps
        top = float(max(np.max(rebin(*comps[what])[1]) for comps in spectra.values()))
        P.finish(ax, axr, channel=channel, xmax=8000.0, top=top,
                 extra=("DY variants" if what == "DYJets" else "total per DY variant"),
                 ratio_label="variant / LO HT")
        P.save(fig, out / f"variants_{what}")


def plot_components(channel, variant, comps, out):
    fig, (ax, axr) = P.spectrum_figure()
    e, vs, vars_ = rebin(*comps["sum"])
    c = centers(e)
    sel = c >= C.THRESHOLD
    for comp in ["sum", "DYJets", "tt_tW", "Nonprompt", "Other"]:
        ee, v, var = rebin(*comps[comp])
        P.step_hist(ax, ee, v, var, COMPONENT_COLOR[comp], COMPONENT_LABEL[comp], errors=(comp == "sum"))
        if comp != "sum":
            frac = np.where(vs > 0, v / np.where(vs > 0, vs, np.nan), np.nan)
            axr.stairs(frac[sel], np.append(ee[:-1][sel], ee[1:][sel][-1]), color=COMPONENT_COLOR[comp], lw=1.4)
    P.finish(ax, axr, channel=channel, xmax=8000.0, top=float(vs.max()), extra=C.DY_SHORT[variant],
             ratio_label="fraction", ratio_lim=(0.0, 1.05))
    P.save(fig, out / f"components_{variant}")


STACK_ORDER = ["Other", "Nonprompt", "tt_tW", "DYJets"]      # bottom to top, the plotter's SR order (data/sample_groups.yaml)
STAT_BAND = {"hatch": "////", "facecolor": "none", "lw": 0, "edgecolor": "k", "alpha": 0.5}   # the plotter's band


def plot_components_stack(channel, variant, comps, out):
    """The composition in the plotter's stack style: the groups as filled, stacked histograms in
    the house colours and order, the hatched statistical band on the total; below, each group's fraction."""
    import mplhep as hep
    fig, (ax, axr) = P.spectrum_figure()
    e, vs, vars_ = rebin(*comps["sum"])
    c = centers(e)
    sel = c >= C.THRESHOLD
    es = np.append(e[:-1][sel], e[1:][sel][-1])
    stack = [np.clip(rebin(*comps[comp])[1][sel], 0.0, None) for comp in STACK_ORDER]
    hep.histplot(stack, es, stack=True, histtype="fill", color=[COMPONENT_COLOR[k] for k in STACK_ORDER],
                 label=[COMPONENT_LABEL[k] for k in STACK_ORDER], ax=ax, flow="none")
    err = np.sqrt(np.maximum(vars_[sel], 0.0))
    ax.stairs(vs[sel] + err, es, baseline=np.clip(vs[sel] - err, P.YMIN_FULL, None), label="Sim. stat. unc.", **STAT_BAND)
    for comp in STACK_ORDER:
        v = rebin(*comps[comp])[1]
        frac = np.where(vs > 0, v / np.where(vs > 0, vs, np.nan), np.nan)
        axr.stairs(frac[sel], es, color=COMPONENT_COLOR[comp], lw=1.4)
    P.finish(ax, axr, channel=channel, xmax=P.XMAX_FULL, ymin=P.YMIN_FULL, top=float(vs.max()), extra=C.DY_SHORT[variant],
             ratio_label="fraction", ratio_lim=(0.0, 1.05))
    # the legend as on the stack plots: the band first, then the groups from the top of the stack down
    by_label = dict(zip(*reversed(ax.get_legend_handles_labels())))
    order = ["Sim. stat. unc."] + [COMPONENT_LABEL[k] for k in reversed(STACK_ORDER)]
    ax.legend([by_label[lab] for lab in order], order, loc="upper right", fontsize=P.FONT_SIZE_LEGEND, handlelength=1.6)
    P.fit_header(ax)
    P.save(fig, out / f"components_stack_{variant}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--variants", nargs="+", default=list(C.DY_VARIANTS))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)
    for channel in args.channels:
        out = C.RESULTS / channel / "1_spectrum"
        (out / "plots").mkdir(parents=True, exist_ok=True)
        spectra, rows, scan = {}, [], []
        for variant in args.variants:
            comps = load_components(channel, variant)
            spectra[variant] = comps
            rows += summary_rows(channel, variant, comps)
            plot_components(channel, variant, comps, out / "plots")
            plot_components_stack(channel, variant, comps, out / "plots")
            # threshold scan on the summed spectrum: the full menu is fitted (the same seeds as step 2,
            # so the 1000 GeV row equals the step 2 table); expo, powexp and wprime7 are reported
            e, v, var = rebin(*comps["sum"])
            c = centers(e)
            hi = last_populated(e, v)
            w = bin_weights(e, v, var)
            for lo in C.LO_VARIANTS:
                _, P_, _ = fit_all(e, v, var, lo, hi, C.PIVOT, menu=C.MENU, w_arr=w)
                row = {"channel": channel, "variant": variant, "lo": lo, "hi": hi}
                for name, fp in P_.items():
                    row[f"chi2_{name}"] = fp["chi2"] if fp else float("nan")
                    row[f"ndf_{name}"] = fp["ndf"] if fp else 0
                    row[f"passed_{name}"] = fp["passed"] if fp else False
                scan.append(row)
            logger.info("%s %s: sum %.1f, DY %.1f (N_eff %.0f), last %.0f GeV", channel, variant,
                        comps["sum"][1].sum(), comps["DYJets"][1].sum(),
                        next(r["neff"] for r in rows if r["channel"] == channel and r["variant"] == variant
                             and r["component"] == "DYJets"), hi)
        plot_variants(channel, spectra, out / "plots")
        write_csv(out / "spectrum.csv", rows)
        write_csv(out / "threshold_scan.csv", scan)
        # summary
        L = [f"# Step 1, {channel} {C.REGION_NAME}, {C.ERA} {C.DIR}, {C.LUMI} fb-1", "",
             f"Native 10 GeV histograms rebinned to {C.BIN_WIDTH:.0f} GeV; selection threshold {C.THRESHOLD:.0f} GeV; "
             f"fit start {C.FIT_LO:.0f} GeV; pivot {C.PIVOT:.0f} GeV.", "",
             "## Components per DY variant (yield above the threshold, effective MC count, last MC event, "
             "negative bins, yields above 3 / 4 / 5 TeV with the MC statistical error)", "",
             "| variant | component | yield | N_eff | N_eff > 3 TeV | last event | neg. bins | > 3 TeV | > 4 TeV | > 5 TeV |",
             "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['variant']} | {r['component']} | {r['yield']:.1f} | {r['neff']:.0f} | {r['neff_gt3000']:.0f} | "
                     f"{r['last_event']:.0f} | {r['neg_bins']} | {r['gt3000']:.2f} +- {r['gt3000_err']:.2f} | "
                     f"{r['gt4000']:.2f} +- {r['gt4000_err']:.2f} | {r['gt5000']:.3f} +- {r['gt5000_err']:.3f} |")
        L += ["", "## Threshold scan on the summed spectrum (Pearson chi2/ndf at MC precision, "
              "fit start varied; the end is the last MC event)", "",
              "| variant | start | expo | powexp | wprime7 |", "|---|---|---|---|---|"]
        for r in scan:
            cells = []
            for name in ["expo", "powexp", "wprime7"]:
                cells.append(f"{r[f'chi2_{name}']:.1f}/{r[f'ndf_{name}']}" + ("" if r[f"passed_{name}"] else " !"))
            L.append(f"| {r['variant']} | {r['lo']:.0f} | " + " | ".join(cells) + " |")
        (out / "summary.md").write_text("\n".join(L) + "\n")
        logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
