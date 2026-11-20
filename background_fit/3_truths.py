#!/usr/bin/env python3
"""Step 4: smooth, one truth set per DY sample.

For each channel and DY variant, the truths of config.TRUTH_NAMES (raw, kde_expo, kde_powexp,
kernel_expo, kernel_powexp) are built per component and summed (the four groups, or with
--components split9 the nine processes of config.COMPONENT_SETS), checked against the MC
(tails above 3 / 4 / 5 TeV, bulk pull trend) and saved as
results/<channel>/3_truths/<variant>/truth_<name>.{npz,json}. Plots: the smoothed sum over the
raw MC with a ratio panel (plots/truth_<variant>_<name>), and each component for the
kde_expo truth (plots/components_<variant>_kde_expo_<component>). Summary: summary.md.

  cd WR_Plotter && python background_fit/3_truths.py
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
from spectrum import (COMPONENT_COLOR, COMPONENT_LABEL, COMPONENT_LS, centers,   # noqa: E402
                      components_of, load_components, rebin)
from truth_sets import build_truth_set, save_truth                       # noqa: E402
from util import fmt                                                     # noqa: E402
from wrplotter.cli_utils import setup_logging                            # noqa: E402

logger = logging.getLogger("3_truths")
TRUTH_LABEL = {"raw": "raw MC", "kde": "RooKeysPdf alone (no tail fit)", "kde_expo": "RooKeysPdf bulk + expo tail",
               "kde_powexp": "RooKeysPdf bulk + powexp tail", "kernel_expo": "kernel bulk + expo tail",
               "kernel_powexp": "kernel bulk + powexp tail"}


def plot_truth_sum(channel, variant, name, comps10, parts, diags, out, extra_line=""):
    comps = components_of(parts)
    e, v, var = comps10["sum"]
    e1, v1, var1 = rebin(e, v, var)
    _, s1, _ = rebin(e, parts["sum"], var)
    c = centers(e1)
    sel = c >= C.THRESHOLD
    fig, (ax, axr) = P.spectrum_figure()
    err = np.sqrt(np.maximum(var1, 0.0))
    # short entries when the legend has two columns (the split sets), so it fits beside the label block
    ax.errorbar(c[sel], v1[sel], yerr=err[sel], fmt="o", ms=4, color="#717581",
                label="MC (sum)" if len(comps) > 5 else "Simulation (sum)")
    ax.plot(c[sel], np.maximum(s1[sel], P.YMIN_FULL), color="k", lw=1.8, label="truth (sum)")
    for comp in comps:
        _, sc, _ = rebin(e, parts[comp], var)
        ax.plot(c[sel], np.maximum(sc[sel], P.YMIN_FULL), color=COMPONENT_COLOR[comp], lw=1.4,
                ls=COMPONENT_LS.get(comp, "--"),
                label=f"{COMPONENT_LABEL[comp]}" + (f", $m_T$ {diags[comp]['m_T'] / 1000:.2g} TeV"
                                                   if diags[comp].get("m_T") and len(comps) <= 5 else ""))
    ok = sel & (v1 > 0) & (s1 > 0)                      # MC / truth per bin with the MC's error: points, no line
    axr.errorbar(c[ok], v1[ok] / s1[ok], yerr=err[ok] / s1[ok], fmt="o", ms=4, color="#717581", lw=1.2)
    # where each process hands over from its bulk smoother to its fitted tail: a dotted
    # vertical line at the process's switch mass, in the process's colour, in both panels
    switches = [(comp, diags[comp]["m_T"]) for comp in comps
                if diags[comp].get("m_T") and not diags[comp].get("no_tail") and "_" in name]
    sum_switch = diags.get("sum", {}).get("m_T")        # one tail fitted to the summed curve (22_deck_models.py sumtail)
    if switches or sum_switch:
        ax.plot([], [], color="0.3", ls=":", lw=1.3, label="tail starts")   # short: a wider entry pushes the
        # two-column legend into the label block, and fit_header then drops it onto the curves
    if len(comps) > 5:
        # many curves (the split sets): the smoother and the process count become the title of a two-column
        # legend, and the label block keeps three lines, so the two stay side by side at the top of the frame
        # the title names the background model, not the process count
        P.finish(ax, axr, channel=channel, xmax=P.XMAX_FULL, ymin=P.YMIN_FULL, top=float(v1[sel].max()), ncol=2,
                 legend_title={"kde": "RooKeysPdf Bkg"}.get(name, P.TRUTH_SHORT.get(name, name)),
                 extra=C.DY_SHORT[variant], ratio_label="MC / truth", ratio_lim=(0.0, 2.5))
    else:
        P.finish(ax, axr, channel=channel, xmax=P.XMAX_FULL, ymin=P.YMIN_FULL, top=float(v1[sel].max()),
                 extra=f"{C.DY_SHORT[variant]}\n{P.TRUTH_SHORT.get(name, name)}" + extra_line,
                 ratio_label="MC / truth", ratio_lim=(0.0, 2.5))
    # drawn after the layout: a full-height line before it would count as content under the legend
    for comp, m_T in switches:
        ax.axvline(m_T, ymax=0.5, color=COMPONENT_COLOR[comp], ls=":", lw=1.3, zorder=1)   # below the label block and legend
        axr.axvline(m_T, color=COMPONENT_COLOR[comp], ls=":", lw=1.3, zorder=1)
    if sum_switch:
        ax.axvline(sum_switch, ymax=0.5, color="k", ls=":", lw=1.6, zorder=1)
        axr.axvline(sum_switch, color="k", ls=":", lw=1.6, zorder=1)
    P.save(fig, out)


def plot_truth_components(channel, variant, name, comps10, parts, diags, out):
    """One figure per component: the simulation with the truth over it, the trust mass, a ratio panel."""
    for comp in components_of(parts):
        e, v, var = comps10[comp]
        e1, v1, var1 = rebin(e, v, var)
        _, s1, _ = rebin(e, parts[comp], var)
        c = centers(e1)
        sel = c >= C.THRESHOLD
        err = np.sqrt(np.maximum(var1, 0.0))
        fig, (ax, axr) = P.spectrum_figure()
        ax.errorbar(c[sel], v1[sel], yerr=err[sel], fmt="o", ms=4, color=COMPONENT_COLOR[comp], label="Simulation")
        ax.plot(c[sel], np.maximum(s1[sel], P.YMIN), color="k", lw=1.8, label=P.TRUTH_SHORT.get(name, name))
        m_T = diags[comp].get("m_T")
        if m_T:
            for a_ in (ax, axr):
                a_.axvline(m_T, color="0.45", ls=":", lw=1.5)
            ax.plot([], [], color="0.45", ls=":", lw=1.5, label=rf"$m_T$ = {m_T:.0f} GeV")
        ok = sel & (v1 > 0) & (s1 > 0)
        axr.errorbar(c[ok], v1[ok] / s1[ok], yerr=err[ok] / s1[ok], fmt="o", ms=4, color=COMPONENT_COLOR[comp], lw=1.2)
        P.finish(ax, axr, channel=channel, xmax=8000.0, top=max(float(np.nanmax(v1[sel])), 1e-3),
                 extra=f"{C.DY_SHORT[variant]}\n{COMPONENT_LABEL[comp]}", ratio_label="MC / truth", ratio_lim=(0.0, 2.5))
        P.save(fig, Path(str(out) + f"_{comp}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--variants", nargs="+", default=list(C.DY_VARIANTS))
    ap.add_argument("--truths", nargs="+", default=C.TRUTH_NAMES)
    ap.add_argument("--trust-neff", type=float, default=C.TRUST_NEFF,
                    help="effective MC events per 500 GeV below which the tail fit takes over (the trust-mass rule)")
    ap.add_argument("--out", default="3_truths", help="results/<channel>/<out>/ (another name for another threshold)")
    ap.add_argument("--components", default="groups4", choices=sorted(C.COMPONENT_SETS),
                    help="the processes smoothed one by one and summed: the four groups, or Nonprompt and Other "
                         "split into their seven processes (split9, from config.SPLIT_DIR)")
    ap.add_argument("--component-plots", nargs="*", default=["kde_expo"],
                    help="truth names that also get one figure per component")
    ap.add_argument("--plots-only", action="store_true",
                    help="redraw the truth-sum figures from the saved truths (no rebuild, no summary)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)
    for channel in args.channels:
        out = C.RESULTS / channel / args.out
        C.TRUST_NEFF = args.trust_neff                   # read by truth_sets.smooth_component
        (out / "plots").mkdir(parents=True, exist_ok=True)
        L = [f"# Step 4, {channel} {C.REGION_NAME}: the smoothed truths per DY variant", "",
             "Per component: trust mass m_T (first 50 GeV point where the running N_eff over the next 500 GeV "
             f"drops below {C.TRUST_NEFF:.0f}); tails = truth / MC yield above 3 / 4 / 5 TeV with the pull "
             "(MC - truth)/sigma_MC; bulk pull = mean (MC - truth)/sigma_MC per 100 GeV bin below m_T "
             "(the RMS in brackets; a mean far from 0 is a shape distortion). A truth marked unavailable "
             "had a component whose tail fit failed.", ""]
        extra_line = "" if args.components == "groups4" else f"\n{len(C.COMPONENT_SETS[args.components])} processes"
        L.insert(2, f"Components: {args.components} = {', '.join(C.COMPONENT_SETS[args.components])}.")
        if args.plots_only:
            from truth_sets import load_truth
            for variant in args.variants:
                comps10 = load_components(channel, variant, args.components)
                for name in args.truths:
                    tpath = out / variant / f"truth_{name}"
                    if name == "raw" or not tpath.with_suffix(".npz").exists():
                        continue
                    _, parts, diags = load_truth(tpath)
                    plot_truth_sum(channel, variant, name, comps10, parts, diags, out / "plots" / f"truth_{variant}_{name}",
                                   extra_line=extra_line)
            logger.info("redrew the truth figures in %s", out / "plots")
            continue
        for variant in args.variants:
            comps10 = load_components(channel, variant, args.components)
            L += [f"## {variant} ({C.DY_LABEL[variant]})", "",
                  "| truth | component | m_T | truth/MC > 3 TeV (pull) | > 4 TeV | > 5 TeV | bulk pull mean (RMS, bins) |",
                  "|---|---|---|---|---|---|---|"]
            for name in args.truths:
                parts, diags, ok, reason = build_truth_set(comps10, name)
                if not ok:
                    L.append(f"| {name} | unavailable: {reason} | | | | | |")
                    logger.warning("%s %s %s: unavailable (%s)", channel, variant, name, reason)
                    continue
                save_truth(out / variant / f"truth_{name}", comps10["sum"][0], parts, diags)
                for comp in components_of(parts) + ["sum"]:
                    d = diags[comp]
                    cells = []
                    for thr in C.TAIL_EDGES:
                        t = d[f"gt{int(thr)}"]
                        cells.append(f"{fmt(t['ratio'])} ({fmt(t['pull'], 1)})")
                    L.append(f"| {name} | {comp} | {fmt(d.get('m_T'), 0) if d.get('m_T') else '-'} | " + " | ".join(cells)
                             + f" | {fmt(d.get('bulk_pull_mean'))} ({fmt(d.get('bulk_pull_rms'))}, {d.get('bulk_nbins')}) |")
                if name != "raw":
                    plot_truth_sum(channel, variant, name, comps10, parts, diags, out / "plots" / f"truth_{variant}_{name}",
                                   extra_line=extra_line)
                if name in args.component_plots:
                    plot_truth_components(channel, variant, name, comps10, parts, diags,
                                          out / "plots" / f"components_{variant}_{name}")
                logger.info("%s %s %s: sum truth/MC above 3/4/5 TeV %s/%s/%s", channel, variant, name,
                            fmt(diags["sum"]["gt3000"]["ratio"]), fmt(diags["sum"]["gt4000"]["ratio"]),
                            fmt(diags["sum"]["gt5000"]["ratio"]))
            L.append("")
        (out / "summary.md").write_text("\n".join(L) + "\n")
        logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
