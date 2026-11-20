#!/usr/bin/env python3
"""Steps 2 and 3: screen the function families on the raw MC, full range, per DY variant.

For each channel, DY variant and component (the sum, DY, tt+tW, Nonprompt, Other), every form
of the menu is fitted over [FIT_LO, last MC event]:
  - the Pearson chi2 at MC precision with the F-test ladder (the selection statistic),
  - the Poisson fit with its yields above 3 / 4 / 5 TeV against the MC, the extrapolation
    from a fit below EXTRAP_SPLIT, and the expected data-level F statistic between nested
    forms (2 x the drop of the Poisson deviance: what a data fit of this luminosity would
    see on average).
Writes results/<channel>/2_screen/{fits.csv, ladder.csv, summary.md, plots/}:
  plots/fit_<variant>_<component>.{png,pdf}     spectrum, every form, pull panel (MC - fit)/sigma_MC
  plots/fit_<variant>_sum_<form>.{png,pdf}      one form with its parameters
  plots/tails_<variant>.{png,pdf}               fitted / MC yield above 3, 4, 5 TeV per form

  cd WR_Plotter && python background_fit/2_screen.py
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))
import config as C                                                       # noqa: E402
import plots as P
from wrplotter.plotting_helpers import FONT_SIZE_LEGEND                                                        # noqa: E402
import published_functions as pf                                         # noqa: E402
from fits import (bin_weights, f_test, fit_all, fit_poisson, integral,   # noqa: E402
                  pearson_at, seeds_for, wilks)
from spectrum import (COMPONENT_COLOR, COMPONENT_LABEL, centers,         # noqa: E402
                      last_populated, load_components, rebin, tail)
from util import fmt, write_csv                                          # noqa: E402
from wrplotter.cli_utils import setup_logging                            # noqa: E402

logger = logging.getLogger("2_screen")
COMPONENTS = ["sum", "DYJets", "tt_tW", "Nonprompt", "Other"]
UPPER = 12000.0


def screen_component(channel, variant, comp, hist, lo):
    """Fit every form to one spectrum; returns (rows, ladder rows, fits_L, fits_P)."""
    edges, values, variances = hist
    c = centers(edges)
    hi = last_populated(edges, values)
    w_arr = bin_weights(edges, values, variances)
    L, Pf, _ = fit_all(edges, values, variances, lo, hi, C.PIVOT, w_arr=w_arr)
    mc_tail = {t: tail(values, variances, edges, t) for t in C.TAIL_EDGES}
    split = C.EXTRAP_SPLIT
    mc_split = tail(values, variances, edges, split)
    rows = []
    ext = {}
    for name in C.MENU:
        fl, fp = L.get(name), Pf.get(name)
        row = {"channel": channel, "variant": variant, "component": comp, "function": name,
               "npar": pf.FUNCS[name][0], "lo": lo, "hi": hi}
        if fp is not None:
            row.update({"P_chi2": fp["chi2"], "P_ndf": fp["ndf"], "P_p": fp["p"], "P_passed": fp["passed"],
                        "P_failed": "+".join(k for k, ok in fp["checks"].items() if not ok)})
        if fl is not None:
            row.update({"L_dev": fl["dev"], "L_passed": fl["passed"],
                        "L_failed": "+".join(k for k, ok in fl["checks"].items() if not ok)})
            chi2_at_L, ndf_at_L = pearson_at(fl["model"], values, w_arr, c, lo, hi, fl["npar"])
            row.update({"L_chi2P": chi2_at_L, "L_chi2P_ndf": ndf_at_L})
            for t in C.TAIL_EDGES:
                fy = integral(fl, t, UPPER)
                my, me = mc_tail[t]
                row[f"fit_gt{int(t)}"] = fy
                row[f"mc_gt{int(t)}"] = my
                row[f"mcerr_gt{int(t)}"] = me
                row[f"ratio_gt{int(t)}"] = fy / my if my > 0 else float("nan")
                row[f"pull_gt{int(t)}"] = (my - fy) / me if me > 0 else float("nan")
            # extrapolation: fit on [lo, split], predict above
            seeds = seeds_for(name, c, values, variances, C.PIVOT, ext) + [list(fl["params"])]
            fx = fit_poisson(name, edges, values, variances, lo, split, C.PIVOT, seeds)
            ext[name] = fx
            if fx is not None:
                pred = integral(fx, split, UPPER)
                my, me = mc_split
                row["extrap_ratio"] = pred / my if my > 0 else float("nan")
                row["extrap_pull"] = (my - pred) / me if me > 0 else float("nan")
            for i, p in enumerate(fl["params"]):
                row[f"p{i}"] = p
                row[f"e{i}"] = fl["perr"][i]
        rows.append(row)
    ladder = []
    for par, child in C.LADDER:
        lp, lc = Pf.get(par), Pf.get(child)
        Lp, Lc = L.get(par), L.get(child)
        r = {"channel": channel, "variant": variant, "component": comp, "parent": par, "child": child}
        if lp is not None and lc is not None:
            p = f_test(lp["chi2"], lc["chi2"], lp["npar"], lc["npar"], lc["nbins"])
            r.update({"P_chi2_parent": lp["chi2"], "P_chi2_child": lc["chi2"], "p_F": p,
                      "add_at_alpha": bool(p < C.F_ALPHA) if math.isfinite(p) else None,
                      "child_passed": lc["passed"]})
        if Lp is not None and Lc is not None:
            d, pw = wilks(Lp["dev"], Lc["dev"], pf.FUNCS[child][0] - pf.FUNCS[par][0])
            r.update({"expected_2dNLL": d, "p_expected": pw})
        ladder.append(r)
    return rows, ladder, L, Pf, hi


def plot_fits(channel, variant, comp, hist, L, out, names=None, with_box=False):
    edges, values, variances = hist
    c = centers(edges)
    sel = c >= C.THRESHOLD
    fig, (ax, axr) = P.spectrum_figure()
    color = COMPONENT_COLOR[comp]
    err = np.sqrt(np.maximum(variances, 0.0))
    ax.errorbar(c[sel], values[sel], yerr=err[sel], fmt="o", ms=4, color=color, label="Simulation")
    hi = max((f["hi"] for f in L.values() if f), default=6000.0)
    grid = np.linspace(C.FIT_LO, hi, 600)
    for name in (names or C.MENU):
        f = L.get(name)
        if f is None:
            continue
        y = np.array([f["tf1"].Eval(x) for x in grid])
        tag = "" if f["passed"] else " (!)"
        ax.plot(grid, np.maximum(y, P.YMIN), color=pf.COLOR[name], ls=pf.LS[name], lw=1.6,
                label=f"{name} [{f['npar']}p]{tag}")
        pull = np.where(err > 0, (values - f["model"]) / np.where(err > 0, err, 1.0), np.nan)
        m = sel & (c >= C.FIT_LO) & (c <= hi)
        axr.stairs(pull[m], np.append(edges[:-1][m], edges[1:][m][-1]), color=pf.COLOR[name], ls=pf.LS[name], lw=1.2)
    axr.axhline(0, color="k", lw=0.8)
    for y in (-2, 2):
        axr.axhline(y, color="0.5", lw=0.6, ls="--")
    if with_box and names and len(names) == 1 and L.get(names[0]):
        f = L[names[0]]
        txt = pf.param_text(names[0], f["params"], f["perr"]) + f"\ndev = {f['dev']:.2f}, [{f['lo']:.0f}, {f['hi']:.0f}] GeV"
        ax.text(0.97, 0.55, txt, transform=ax.transAxes, ha="right", va="top", fontsize=FONT_SIZE_LEGEND,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
    P.finish(ax, axr, channel=channel, xmax=min(hi + 500.0, 9000.0), top=float(values[sel].max()),
             extra=f"{C.DY_SHORT[variant]}\n{COMPONENT_LABEL[comp]}", ratio_label=r"(MC $-$ fit) / $\sigma_{\rm MC}$",
             ratio_lim=(-4, 4), ratio_ref=0.0, ncol=2 if len(names or C.MENU) > 4 else 1,
             legend_loc="lower left" if len(names or C.MENU) > 4 else "upper right")   # the empty corner of a falling spectrum
    P.save(fig, out)


def plot_tails(channel, variant, rows, out):
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 8))
    names = [r["function"] for r in rows if r["component"] == "sum" and "ratio_gt3000" in r]
    xs = np.arange(len(names))
    cols = ["#5790fc", "#e42536", "#f89c20"]
    for j, t in enumerate(C.TAIL_EDGES):
        rat = [next(r[f"ratio_gt{int(t)}"] for r in rows if r["component"] == "sum" and r["function"] == n) for n in names]
        ax.plot(xs + 0.12 * (j - 1), rat, "o", ms=7, color=cols[j], label=f"fit / MC above {t/1000:.0f} TeV")
        r0 = next(r for r in rows if r["component"] == "sum" and r["function"] == names[0])
        rel = r0[f"mcerr_gt{int(t)}"] / r0[f"mc_gt{int(t)}"] if r0[f"mc_gt{int(t)}"] > 0 else 0
        ax.axhspan(1 - rel, 1 + rel, color=cols[j], alpha=0.08)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 50.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Fitted yield / MC yield, total background")
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(channel, C.DY_SHORT[variant]), year=C.YEAR)
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    fit_header(ax)
    P.save(fig, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--variants", nargs="+", default=list(C.DY_VARIANTS))
    ap.add_argument("--components", nargs="+", default=COMPONENTS)
    ap.add_argument("--lo", type=float, default=C.FIT_LO)
    ap.add_argument("--out", default="2_screen", help="results/<channel>/<out>/ (another name for another --lo)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)
    for channel in args.channels:
        out = C.RESULTS / channel / args.out
        (out / "plots").mkdir(parents=True, exist_ok=True)
        all_rows, all_ladder = [], []
        for variant in args.variants:
            comps = load_components(channel, variant)
            for comp in args.components:
                hist = rebin(*comps[comp])
                rows, ladder, L, Pf, hi = screen_component(channel, variant, comp, hist, args.lo)
                all_rows += rows
                all_ladder += ladder
                for r in rows:
                    if "P_chi2" in r:
                        logger.info("%s %-8s %-9s %-9s %dp P %8.1f/%d %s  L dev %7.2f  tails %s/%s/%s  extrap %s",
                                    channel, variant, comp, r["function"], r["npar"], r["P_chi2"], r["P_ndf"],
                                    "ok " if r["P_passed"] else "BAD", r.get("L_dev", float("nan")),
                                    fmt(r.get("ratio_gt3000")), fmt(r.get("ratio_gt4000")), fmt(r.get("ratio_gt5000")),
                                    fmt(r.get("extrap_ratio")))
                if comp in ("sum", "DYJets", "tt_tW"):
                    plot_fits(channel, variant, comp, hist, L, out / "plots" / f"fit_{variant}_{comp}")
                if comp == "sum":
                    for name in C.MENU:
                        if L.get(name):
                            plot_fits(channel, variant, comp, hist, L, out / "plots" / f"fit_{variant}_sum_{name}",
                                      names=[name], with_box=True)
                    plot_tails(channel, variant, rows, out / "plots" / f"tails_{variant}")
        write_csv(out / "fits.csv", all_rows)
        write_csv(out / "ladder.csv", all_ladder)
        write_summary(channel, all_rows, all_ladder, out / "summary.md", args.lo)
        logger.info("wrote %s", out)


def write_summary(channel, rows, ladder, path, lo):
    L = [f"# Step 2/3, {channel} {C.REGION_NAME}: the forms on the raw MC, full range from {lo:.0f} GeV", "",
         "P = Pearson chi2/ndf at MC precision (the selection statistic; `!` = a failed check: minimum, "
         "covariance, parameter at a limit, monotonic); dev = the Poisson deviance, the expected data-level "
         "discrimination of the form from the MC at 109.7 fb-1; tails = Poisson-fit yield / MC yield above "
         "3 / 4 / 5 TeV (pull in sigma_MC); extrap = fit on [lo, 3000] predicting the MC above 3 TeV.", ""]
    variants = []
    for r in rows:
        if r["variant"] not in variants:
            variants.append(r["variant"])
    for comp in COMPONENTS:
        L += [f"## {comp}", ""]
        for variant in variants:
            rs = [r for r in rows if r["component"] == comp and r["variant"] == variant]
            if not rs:
                continue
            L += [f"### {variant} ({C.DY_LABEL[variant]}), range [{rs[0]['lo']:.0f}, {rs[0]['hi']:.0f}] GeV", "",
                  "| form | np | P chi2/ndf (p) | dev | fit/MC > 3 TeV (pull) | > 4 TeV | > 5 TeV | extrap (pull) |",
                  "|---|---|---|---|---|---|---|---|"]
            for r in sorted(rs, key=lambda r: (r["npar"], r.get("P_chi2", 1e99))):
                if "P_chi2" not in r:
                    L.append(f"| {r['function']} | {r['npar']} | no fit | | | | | |")
                    continue
                pc = f"{fmt(r['P_chi2'], 1)}/{r['P_ndf']} ({fmt(r['P_p'], 3)})" + ("" if r["P_passed"] else f" !{r['P_failed']}")
                L.append(f"| {r['function']} | {r['npar']} | {pc} | {fmt(r.get('L_dev'), 2)} | "
                         f"{fmt(r.get('ratio_gt3000'))} ({fmt(r.get('pull_gt3000'), 1)}) | "
                         f"{fmt(r.get('ratio_gt4000'))} ({fmt(r.get('pull_gt4000'), 1)}) | "
                         f"{fmt(r.get('ratio_gt5000'))} ({fmt(r.get('pull_gt5000'), 1)}) | "
                         f"{fmt(r.get('extrap_ratio'))} ({fmt(r.get('extrap_pull'), 1)}) |")
            ls = [x for x in ladder if x["component"] == comp and x["variant"] == variant]
            if ls:
                L += ["", "F-test ladder (Pearson, alpha = 0.05) and the expected data-level statistic (2 dNLL, Wilks p):", "",
                      "| parent -> child | chi2 parent | chi2 child | p_F | add? | expected 2 dNLL (p) |", "|---|---|---|---|---|---|"]
                for x in ls:
                    L.append(f"| {x['parent']} -> {x['child']} | {fmt(x.get('P_chi2_parent'), 1)} | {fmt(x.get('P_chi2_child'), 1)} | "
                             f"{fmt(x.get('p_F'), 3)} | {fmt(x.get('add_at_alpha'))} | "
                             f"{fmt(x.get('expected_2dNLL'), 2)} ({fmt(x.get('p_expected'), 3)}) |")
            L.append("")
    # cross-variant verdict on the sum: best 3-parameter form per variant by Pearson chi2
    L += ["## Across DY variants (summed background): best form per parameter count and the ladder verdicts", "",
          "| variant | best 2p | best 3p | best 4p | expo -> powexp | powexp -> dilep8 | powlaw -> atlasemu |",
          "|---|---|---|---|---|---|---|"]
    for variant in variants:
        rs = [r for r in rows if r["component"] == "sum" and r["variant"] == variant and "P_chi2" in r]
        cells = []
        for npar in (2, 3, 4):
            cand = [r for r in rs if r["npar"] == npar and r["P_passed"]] or [r for r in rs if r["npar"] == npar]
            if cand:
                b = min(cand, key=lambda r: r["P_chi2"])
                cells.append(f"{b['function']} ({fmt(b['P_chi2'], 0)}/{b['P_ndf']})")
            else:
                cells.append("-")
        for par, child in [("expo", "powexp"), ("powexp", "dilep8"), ("powlaw", "atlasemu")]:
            x = next((l for l in ladder if l["component"] == "sum" and l["variant"] == variant
                      and l["parent"] == par and l["child"] == child), None)
            cells.append(f"p_F {fmt(x.get('p_F'), 3)}, {'add' if x.get('add_at_alpha') else 'keep'}" if x and x.get("p_F") is not None else "-")
        L.append(f"| {variant} | " + " | ".join(cells) + " |")
    Path(path).write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
