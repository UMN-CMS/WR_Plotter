#!/usr/bin/env python3
"""The simultaneous SR + CR fit (step 8) against the SR-only fit: verdicts and sensitivity side by side.

For each channel and fit start found (results/<channel>/8_flavor_cr[_lo1000]/toys_all.csv against
results/<channel>/4_toys[_lo1200]/toys_all.csv): the step 6 verdict of every configuration, the
Asimov sigma_fit relative to the SR-only powexp fit at the same start, the fitted transfer factor
of the free configuration, and per channel a figure (S_spur band over the primary truths and
sigma_fit per configuration vs m_WR). Writes results/FLAVOR_CR.md and results/<channel>/8_flavor_cr*/
fcr_methods.

  cd WR_Plotter && python background_fit/9_compare_fcr.py
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))
import config as C                                                       # noqa: E402
from util import fmt                                                     # noqa: E402

D = importlib.import_module("5_decide")
F = importlib.import_module("8_flavor_cr")                               # registers colours and labels
STARTS = [(1200.0, "8_flavor_cr", "4_toys_lo1200"), (1000.0, "8_flavor_cr_lo1000", "4_toys")]
SIG_MASSES = [1400, 1600, 2000, 2600, 3200, 4000]
CONFIG_NAMES = [F.cfg_name(*c) for c in F.CONFIGS]


def ref_err(rows, cand="powexp", variant="lo_ht", truth="raw"):
    return {r["mass"]: r["asimov_err"] for r in rows if r["candidate"] == cand and r["variant"] == variant
            and r["truth"] == truth and r["inject"] == 0.0 and np.isfinite(r["asimov_err"])}


def ratio(rows, cand, ref, masses, variant="lo_ht", truth="raw"):
    out = {}
    for m in masses:
        r = next((x for x in rows if x["candidate"] == cand and x["variant"] == variant and x["truth"] == truth
                  and x["inject"] == 0.0 and x["mass"] == m), None)
        out[m] = (r["asimov_err"] / ref[m]) if (r is not None and m in ref and ref[m] > 0 and np.isfinite(r["asimov_err"])) else float("nan")
    return out


def plot_methods(channel, lo, rows_fcr, rows_sr, out):
    import matplotlib.pyplot as plt
    import mplhep as hep
    import plots as P
    import published_functions as pf
    hep.style.use("CMS")
    ref = ref_err(rows_sr)
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    fig, (ax, axb) = P.scan_figure2()
    def short(label):                           # "CR powexp + DY expo, TF fixed" -> "pe / ex, TF fixed"
        t = label.replace("CR ", "").replace(" + DY ", " / ")
        for a, b in (("powexp", "pe"), ("expo", "ex"), ("atlasemu", "ae")):
            t = t.replace(a, b)
        return t
    series = [("powexp", rows_sr, "k", "-", "SR only, pe")] + \
             [(n, rows_fcr, pf.COLOR[n], pf.LS[n], short(C.MENU_LABEL[n])) for n in CONFIG_NAMES
              if any(r["candidate"] == n for r in rows_fcr)]
    for cand, rows, col, ls, label in series:
        rs = [r for r in rows if r["candidate"] == cand and r["variant"] in D.PRIMARY and r["inject"] == 0.0
              and np.isfinite(r["bias_median"])]
        masses = sorted({r["mass"] for r in rs})
        if not masses:
            continue
        lo_ = [min(r["bias_median"] for r in rs if r["mass"] == m) for m in masses]
        hi_ = [max(r["bias_median"] for r in rs if r["mass"] == m) for m in masses]
        med = [float(np.median([r["bias_median"] for r in rs if r["mass"] == m])) for m in masses]
        ax.fill_between(masses, lo_, hi_, color=col, alpha=0.15, lw=0)
        ax.plot(masses, med, color=col, lw=1.8, ls=ls, marker="o", ms=4, label=label)
        if ref:
            rat = ratio(rows, cand, ref, masses)
            xs = [m for m in masses if np.isfinite(rat[m])]
            axb.plot(xs, [rat[m] for m in xs], color=col, lw=1.8, ls=ls, marker="o", ms=4)
    P.bias_lines(ax)
    ax.set_ylim(-3.8, 2.4)                                   # room for the legend under the lines
    ax.set_ylabel(P.SSPUR_LABEL)
    cms_label(ax, C.LUMI, C.COM, data=False,
              text=P.region_text(channel, f"SR + e$\\mu$ CR fits from {lo:.0f} GeV\n"
                                          "CR / DY form: pe powexp, ex expo, ae running power law"),
              year=C.YEAR)
    ax.legend(loc="lower right", fontsize=FONT_SIZE_LEGEND, ncol=2, handlelength=1.4, columnspacing=0.8)
    fit_header(ax)
    axb.axhline(1.0, color="k", lw=0.8)
    axb.set_ylim(0.4, 1.6)
    axb.set_ylabel(r"$\sigma_{\rm fit}$ / SR-only powexp")
    axb.set_xlabel(P.MWR_LABEL)
    P.save(fig, out / "fcr_methods")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    args = ap.parse_args()
    L = ["# background_fit: the simultaneous SR + e-mu CR fit against the SR-only fit (tried and dropped)", "",
         "Step 8 toys: SR = TF x f_tt + f_DY + signal, e-mu CR = f_tt, fitted together over the full range, TF fixed to the "
         "truth ratio or free. Same truths, toys and criteria as step 5; the reference is the SR-only powexp fit at the same "
         "start (results/<channel>/4_toys_lo1200 or 4_toys). sigma_fit ratios on the LO HT raw truth. Verdicts over the "
         f"primary truths ({', '.join(D.PRIMARY)} x {len(C.TRUTH_NAMES)} smoothers), float regime m_WR <= {D.FLOAT_MAX:.0f} GeV.", ""]
    passing = {}
    for lo, step_fcr, step_sr in STARTS:
        any_ = False
        for channel in args.channels:
            rows_fcr = D.load_rows(channel, step_fcr)
            rows_sr = D.load_rows(channel, step_sr)
            if not rows_fcr:
                continue
            any_ = True
            out = C.RESULTS / channel / step_fcr
            ref = ref_err(rows_sr)
            L += [f"## {channel} {C.REGION_NAME}, fits from {lo:.0f} GeV", "",
                  "| configuration | max abs S_spur (where) | max abs bias at 1x / 2x (where) | min passed fraction | "
                  "width range | cells > 0.3 | cells > 0.5 | sigma_fit / SR-only powexp | verdict |",
                  "|---|---|---|---|---|---|---|---|---|"]
            Vsr = D.verdicts(rows_sr, D.PRIMARY, D.FLOAT_MAX) if rows_sr else {}
            if "powexp" in Vsr:
                v = Vsr["powexp"]
                L.append(f"| SR only, powexp (reference) | {fmt(v['max_sspur'])} ({v['where_sspur']}) | {fmt(v['max_bias_inj'])} "
                         f"({v['where_bias_inj']}) | {fmt(v['min_passed'])} | {fmt(v['width_min'])} - {fmt(v['width_max'])} | "
                         f"{v['n_fail03']} / {v['n_cells']} | {v['n_fail05']} | 1.00 | {'pass' if v['pass'] else 'fail'} |")
            V = D.verdicts(rows_fcr, D.PRIMARY, D.FLOAT_MAX)
            passing[(lo, channel)] = [c for c, v in V.items() if v["pass"]]
            for cand in [c for c in CONFIG_NAMES if c in V]:
                v = V[cand]
                rat = ratio(rows_fcr, cand, ref, [m for m in sorted(ref) if m <= D.FLOAT_MAX])
                vals = [x for x in rat.values() if np.isfinite(x)]
                srel = f"{min(vals):.2f} - {max(vals):.2f}" if vals else "-"
                verdict = "target" if v["target"] else ("pass" if v["pass"] else "fail")
                L.append(f"| {C.MENU_LABEL[cand]} | {fmt(v['max_sspur'])} ({v['where_sspur']}) | {fmt(v['max_bias_inj'])} "
                         f"({v['where_bias_inj']}) | {fmt(v['min_passed'])} | {fmt(v['width_min'])} - {fmt(v['width_max'])} | "
                         f"{v['n_fail03']} / {v['n_cells']} | {v['n_fail05']} | {srel} | {verdict} |")
            L += ["", "Sparse regime (m_WR >= 3400 GeV): " + "; ".join(
                f"{C.MENU_LABEL[c]}: max {fmt(v['max_sspur'])}, converged {fmt(v['min_passed'])}"
                for c, v in D.verdicts([r for r in rows_fcr if r["mass"] >= 3400], D.PRIMARY, 1e9).items()), ""]
            L += [f"Asimov sigma_fit relative to the SR-only powexp (LO HT raw truth):", "",
                  "| configuration | " + " | ".join(str(m) for m in SIG_MASSES) + " |", "|---|" + "---|" * len(SIG_MASSES)]
            for cand in [c for c in CONFIG_NAMES if c in V]:
                rat = ratio(rows_fcr, cand, ref, SIG_MASSES)
                L.append(f"| {C.MENU_LABEL[cand]} | " + " | ".join(fmt(rat[m]) for m in SIG_MASSES) + " |")
            free = []
            for r in rows_fcr:
                if r["candidate"].endswith("_free") and r["inject"] == 0.0 and r["variant"] in D.PRIMARY and r["mass"] <= D.FLOAT_MAX:
                    tfm, tft = D.F(r.get("tf_mean")), D.F(r.get("tf_true"))
                    if np.isfinite(tfm) and np.isfinite(tft) and tft > 0:
                        free.append((tfm, tft))
            if free:
                pulls = [(a - b) / b for a, b in free]
                L += ["", f"Free transfer factor: fitted / true - 1 over the float-regime cells, mean {np.mean(pulls):+.2f}, "
                      f"range {min(pulls):+.2f} to {max(pulls):+.2f} (true TF {free[0][1]:.3f})."]
            L.append("")
            plot_methods(channel, lo, rows_fcr, rows_sr, out)
            L += [f"Figure: results/{channel}/{step_fcr}/fcr_methods.", ""]
        if not any_:
            L += [f"## fits from {lo:.0f} GeV: no step 8 results", ""]
    L += ["## Across channels (float regime, primary truths)", ""]
    for lo, _, _ in STARTS:
        chans = [ch for (l_, ch) in passing if l_ == lo]
        if not chans:
            continue
        common = set.intersection(*[set(passing[(lo, ch)]) for ch in chans])
        per = "; ".join(f"{ch}: {', '.join(C.MENU_LABEL[c] for c in passing[(lo, ch)]) or 'none'}" for ch in chans)
        L.append(f"- from {lo:.0f} GeV: passing in every channel {', '.join(C.MENU_LABEL[c] for c in sorted(common)) or 'none'} ({per}).")
    L.append("")
    (C.RESULTS / "FLAVOR_CR.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-5:]))
    print("wrote", C.RESULTS / "FLAVOR_CR.md")


if __name__ == "__main__":
    main()
