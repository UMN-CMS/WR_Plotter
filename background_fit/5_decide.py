#!/usr/bin/env python3
"""Step 6: decide nominal or envelope from the toys.

Reads results/<channel>/4_toys/toys_all.csv and writes results/DECISION.md plus two figures
per channel:
  results/<channel>/5_decision/sspur_by_candidate_<candidate>   S_spur vs m_WR, one line per primary truth
  results/<channel>/5_decision/sigma_fit_vs_mass      sigma_fit of each candidate relative to the exponential

Verdict per candidate and channel over the PRIMARY truths (the DY variants with enough
effective MC in the tail: lo_ht and nlo_mll, all five smoothers each): pass when every
|S_spur| (median N_sig / RMS at zero injection) and every |bias| at 1x and 2x the expected
limit stay below BIAS_PASS and the fit passes its checks in >= 90 % of the toys, for
m_WR <= 3200 GeV (the float regime; the sparse regime above is reported separately).
"target" when the same holds below BIAS_TARGET. The other variants are the STRESS truths
(low N_eff, negative weights) and are reported, not required.

Nominal = the lowest-parameter candidate that passes in both channels on every primary
truth; envelope recommended when two families pass and their median N_sig at zero
injection differ by more than BIAS_TARGET x RMS at any mass.

  cd WR_Plotter && python background_fit/5_decide.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))
import config as C                                                       # noqa: E402
from util import fmt, read_csv                                           # noqa: E402

PRIMARY = ["lo_ht", "nlo_mll"]
FLOAT_MAX = C.FLOAT_MAX


def F(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_rows(channel, step="4_toys"):
    """Typed rows of results/<channel>/<step>/*/toys_*.csv (step = "4_toys" or "6_windows/k3", ...)."""
    rows = []
    for p in sorted((C.RESULTS / channel / step).glob("*/toys_*.csv")):
        rows += read_csv(p)
    out = []
    for r in rows:
        if r.get("status") != "ok":
            continue
        r = dict(r)
        for k in ("mass", "inject", "n_inj", "ul_expected", "asimov_nsig", "asimov_err", "nsig_median", "nsig_mean",
                  "nsig_rms", "bias_median", "bias_mean", "pull_mean", "pull_width", "b_core"):
            r[k] = F(r.get(k))
        r["n_conv"], r["n_passed"] = int(F(r.get("n_conv"))), int(F(r.get("n_passed")))
        out.append(r)
    return out


KERNEL_SET = ["raw", "kernel_expo", "kernel_powexp"]   # the truths without the RooKeysPdf bulk, which
                                                        # distorts the mumu LO HT-binned DY (step 4)


def same_family(cand, truth):
    """A smoothed truth whose fitted tail is the candidate's own family (kde_expo for expo, kernel_powexp
    for powexp): that cell is circular (the candidate reproduces the tail by construction) and is not
    counted in the verdict. The raw truth and the cross-family truths always count."""
    return truth != "raw" and truth.endswith("_" + cand)


def verdicts(rows, variants, mass_max=FLOAT_MAX, truths=None):
    """{candidate: dict of worst-case metrics over truths x masses} (truths = None: every smoother)."""
    cands = sorted({r["candidate"] for r in rows}, key=lambda n: C.CANDIDATES.index(n) if n in C.CANDIDATES else 99)
    out = {}
    for cand in cands:
        rs = [r for r in rows if r["candidate"] == cand and r["variant"] in variants and r["mass"] <= mass_max
              and (truths is None or r["truth"] in truths) and not same_family(cand, r["truth"])]
        r0 = [r for r in rs if r["inject"] == 0.0 and np.isfinite(r["bias_median"])]
        r12 = [r for r in rs if r["inject"] in (1.0, 2.0) and np.isfinite(r["bias_mean"])]
        if not r0:
            continue
        worst0 = max(r0, key=lambda r: abs(r["bias_median"]))
        worst12 = max(r12, key=lambda r: abs(r["bias_mean"])) if r12 else None
        frac = [r["n_passed"] / max(r["n_conv"], 1) for r in r0]
        widths = [r["pull_width"] for r in r0 if np.isfinite(r["pull_width"])]
        fails05 = sorted({(r["variant"], r["truth"], int(r["mass"])) for r in r0 if abs(r["bias_median"]) > C.BIAS_PASS})
        fails03 = sorted({(r["variant"], r["truth"], int(r["mass"])) for r in r0 if abs(r["bias_median"]) > C.BIAS_TARGET})
        out[cand] = {
            "max_sspur": abs(worst0["bias_median"]), "where_sspur": f"{worst0['variant']}/{worst0['truth']} @ {worst0['mass']:.0f}",
            "max_bias_inj": abs(worst12["bias_mean"]) if worst12 else float("nan"),
            "where_bias_inj": (f"{worst12['variant']}/{worst12['truth']} @ {worst12['mass']:.0f}, {worst12['inject']:g}x" if worst12 else "-"),
            "min_passed": min(frac), "width_min": min(widths) if widths else float("nan"),
            "width_max": max(widths) if widths else float("nan"),
            "n_cells": len(r0), "n_fail05": len(fails05), "n_fail03": len(fails03),
            "fail05": fails05[:8], "fail03": fails03[:8],
        }
        v = out[cand]
        v["pass"] = (v["max_sspur"] < C.BIAS_PASS and v["max_bias_inj"] < C.BIAS_PASS and v["min_passed"] >= 0.9)
        v["target"] = (v["max_sspur"] < C.BIAS_TARGET and v["max_bias_inj"] < C.BIAS_TARGET and v["min_passed"] >= 0.9)
    return out


def candidate_spread(rows, variants, cands, mass_max=FLOAT_MAX):
    """Largest difference between two candidates' median N_sig (zero injection) in units of the
    larger RMS, over the primary truths and masses: the case for an envelope."""
    worst = (0.0, "")
    for variant in variants:
        truths = sorted({r["truth"] for r in rows if r["variant"] == variant})
        for truth in truths:
            masses = sorted({r["mass"] for r in rows if r["variant"] == variant and r["truth"] == truth and r["mass"] <= mass_max})
            for m in masses:
                vals = {}
                for cand in cands:
                    r = next((x for x in rows if x["variant"] == variant and x["truth"] == truth and x["mass"] == m
                              and x["candidate"] == cand and x["inject"] == 0.0), None)
                    if r and np.isfinite(r["nsig_median"]) and np.isfinite(r["nsig_rms"]) and r["nsig_rms"] > 0:
                        vals[cand] = (r["nsig_median"], r["nsig_rms"])
                names = list(vals)
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        a, b = vals[names[i]], vals[names[j]]
                        d = abs(a[0] - b[0]) / max(a[1], b[1])
                        if d > worst[0]:
                            worst = (d, f"{names[i]} vs {names[j]}, {variant}/{truth} @ {m:.0f}")
    return worst


def plot_sspur_by_candidate(channel, rows, out):
    """One figure per candidate: S_spur vs m_WR, one line per primary truth (colour = smoother,
    style = DY sample), the 0.3 and 0.5 lines."""
    import matplotlib.pyplot as plt
    import plots as P
    cands = [c for c in C.CANDIDATES if any(r["candidate"] == c for r in rows)]
    cands += sorted({r["candidate"] for r in rows} - set(cands))        # e.g. the step 8 configurations
    for cand in cands:
        fig, ax = P.scan_figure()
        handles = P.truth_lines(ax, rows, cand, PRIMARY)
        if not handles:
            plt.close(fig)
            continue
        P.bias_lines(ax)
        P.finish_scan(ax, channel=channel, extra=C.MENU_LABEL.get(cand, cand), ylabel=P.SSPUR_LABEL,
                      ylim=(-2.0, 3.0), handles=handles, ncol=1)
        P.save(fig, out / f"sspur_by_candidate_{cand}")


def plot_sigma_fit(channel, rows, out):
    import matplotlib.pyplot as plt
    import mplhep as hep
    import plots as P
    import published_functions as pf
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 8))
    ref = {}
    for r in rows:
        if r["candidate"] == "expo" and r["variant"] == "lo_ht" and r["truth"] == "raw" and r["inject"] == 0.0:
            ref[r["mass"]] = r["asimov_err"]
    for cand in C.CANDIDATES:
        rs = sorted([r for r in rows if r["candidate"] == cand and r["variant"] == "lo_ht" and r["truth"] == "raw"
                     and r["inject"] == 0.0 and r["mass"] in ref], key=lambda r: r["mass"])
        if not rs:
            continue
        ax.plot([r["mass"] for r in rs], [r["asimov_err"] / ref[r["mass"]] for r in rs], marker="o", ms=5, lw=1.5,
                color=pf.COLOR[cand], ls=pf.LS[cand], label=C.MENU_LABEL.get(cand, cand))
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_ylim(0.5, 3.0)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"$\sigma_{\rm fit}(N_{\rm sig})$ / $\sigma_{\rm fit}$ of the exponential (Asimov, raw MC)")
    cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(channel, "LO HT-binned, raw MC truth"), year=C.YEAR)
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND, ncol=2, handlelength=1.8)
    fit_header(ax)
    P.save(fig, out / "sigma_fit_vs_mass")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--step", default="4_toys", help="results/<channel>/<step>/ to judge (e.g. 4_toys_lo1200)")
    ap.add_argument("--tag", default="", help="suffix of DECISION_<tag>.md and 5_decision_<tag>/")
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""
    L = ["# background_fit decision: nominal or envelope" + (f" ({args.tag})" if args.tag else ""), "",
         f"Inputs: results/<channel>/{args.step}/toys_all.csv ({C.NTOYS} toys per cell, injections "
         f"{', '.join(f'{k:g}x' for k in C.INJECT)} the expected limit, full-range S+B fits). "
         f"Primary truths: DY variants {', '.join(PRIMARY)} x smoothers {', '.join(C.TRUTH_NAMES)}; a second table "
         f"judges the raw + kernel truths only ({', '.join(KERNEL_SET)}), because the RooKeysPdf bulk distorts the "
         f"mumu LO HT-binned DY (step 4). Float regime: m_WR <= {FLOAT_MAX:.0f} GeV. Pass: |S_spur| and |bias at 1x, 2x| "
         f"< {C.BIAS_PASS} and >= 90 % of the toys pass the fit checks; target {C.BIAS_TARGET}.", ""]
    overall, overall_k = {}, {}
    for channel in args.channels:
        rows = load_rows(channel, args.step)
        if not rows:
            L += [f"## {channel}: no toy results", ""]
            continue
        out = C.RESULTS / channel / f"5_decision{sfx}"
        out.mkdir(parents=True, exist_ok=True)
        plot_sspur_by_candidate(channel, rows, out)
        plot_sigma_fit(channel, rows, out)
        L += [f"## {channel} {C.REGION_NAME}", ""]
        tables = [("Primary truths, all smoothers, float regime (m_WR <= 3200 GeV)", PRIMARY, False, None),
                  ("Primary truths, raw + kernel smoothers only, float regime", PRIMARY, False, KERNEL_SET),
                  ("Primary truths, all smoothers, sparse regime (m_WR >= 3400 GeV)", PRIMARY, True, None),
                  ("Stress truths (nlo_njet, lo_inc, nlo_inc), float regime",
                   [v for v in C.DY_VARIANTS if v not in PRIMARY], False, None)]
        for label, variants, sparse, truths in tables:
            sub = [r for r in rows if r["mass"] >= 3400.0] if sparse else rows
            V = verdicts(sub, variants, 1e9 if sparse else FLOAT_MAX, truths)
            if not V:
                continue
            L += [f"### {label}", "",
                  "| candidate | max abs S_spur (where) | max abs bias at 1x / 2x (where) | min passed fraction | "
                  "width range | cells | cells > 0.3 | cells > 0.5 | verdict |", "|---|---|---|---|---|---|---|---|---|"]
            for cand, v in V.items():
                verdict = "target" if v["target"] else ("pass" if v["pass"] else "fail")
                L.append(f"| {cand} | {fmt(v['max_sspur'])} ({v['where_sspur']}) | {fmt(v['max_bias_inj'])} ({v['where_bias_inj']}) | "
                         f"{fmt(v['min_passed'])} | {fmt(v['width_min'])} - {fmt(v['width_max'])} | {v['n_cells']} | "
                         f"{v['n_fail03']} | {v['n_fail05']} | {verdict} |")
            if not sparse and variants == PRIMARY:
                (overall if truths is None else overall_k)[channel] = V
            L.append("")
        V0 = overall.get(channel, {})
        cands_pass = [c for c, v in V0.items() if v["pass"]]
        cands_pass_k = [c for c, v in overall_k.get(channel, {}).items() if v["pass"]]
        # the envelope question is asked of the candidates within reach of the pass line (2x the
        # bias limit, half the toys converged); the exponential is out of it in both channels
        near = [c for c, v in V0.items() if v["max_sspur"] < 2 * C.BIAS_PASS and v["max_bias_inj"] < 2 * C.BIAS_PASS
                and v["min_passed"] >= 0.5]
        L.append(f"Passing candidates in the float regime: all smoothers {', '.join(cands_pass) or 'none'}; "
                 f"raw + kernel truths {', '.join(cands_pass_k) or 'none'}.")
        if len(near) >= 2:
            d, where = candidate_spread(rows, PRIMARY, near)
            L.append(f"Candidates within 2x the pass line ({', '.join(near)}): the largest difference between two of "
                     f"their median N_sig at zero injection is {d:.2f} RMS ({where}); above {C.BIAS_TARGET} RMS it "
                     f"argues for an envelope over them.")
        else:
            L.append(f"Fewer than two candidates within 2x the pass line ({', '.join(near) or 'none'}): no envelope "
                     f"spread to quote.")
        L.append("")
        for cand, v in V0.items():
            if v["fail05"]:
                L.append(f"- {cand}: cells above {C.BIAS_PASS} (first 8): " + "; ".join(f"{a}/{b} @ {m}" for a, b, m in v["fail05"]))
        L.append("")
    L += ["## Across channels", ""]
    import published_functions as pf
    for name, O in [("all smoothers", overall), ("raw + kernel truths", overall_k)]:
        chans = [c for c in args.channels if c in O]
        if not chans:
            continue
        common = None
        for ch in chans:
            s_ = {c for c, v in O[ch].items() if v["pass"]}
            common = s_ if common is None else (common & s_)
        common = sorted(common or [], key=lambda n: C.CANDIDATES.index(n) if n in C.CANDIDATES else 99)
        per = "; ".join(f"{ch}: {', '.join(c for c, v in O[ch].items() if v['pass']) or 'none'}" for ch in chans)
        L.append(f"- {name}: passing in every channel {', '.join(common) or 'none'} ({per})."
                 + (f" Nominal by the lowest parameter count: **{min(common, key=lambda n: pf.FUNCS.get(n, (99,))[0])}**."
                    if common else ""))
    L += ["", f"Figures: results/<channel>/5_decision{sfx}/sspur_by_candidate (S_spur vs mass per candidate, every primary "
          "truth) and sigma_fit_vs_mass (the cost of the parameters)."]
    (C.RESULTS / f"DECISION{sfx}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-6:]))
    print("wrote", C.RESULTS / f"DECISION{sfx}.md")


if __name__ == "__main__":
    main()
