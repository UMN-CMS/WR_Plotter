#!/usr/bin/env python3
"""Full-range fit (step 5) versus window fits (6_windows.py): verdicts and sensitivity side by side.

Reads results/<channel>/4_toys/toys_all.csv and every results/<channel>/6_windows/k<k>/toys_all.csv
and applies the step 6 verdict to every method (primary truths lo_ht + nlo_mll, float regime
m_WR <= 3200 GeV, pass |S_spur| and |bias at 1x, 2x| < 0.5 with >= 90 % of the toys converged).
Writes results/WINDOWS.md and, per channel, results/<channel>/6_windows/:
  sspur_methods_<candidate>_<range>   S_spur vs m_WR for the full range, +-3 and +-5 sigma, one line per primary truth
  window_width_<candidate>    the width scan (expo, powlaw, powexp, atlasemu): band of S_spur over the truths and sigma_fit per window, vs m_WR
  sigma_fit_methods           Asimov sigma_fit of (method, candidate) relative to the full-range powexp

  cd WR_Plotter && python background_fit/7_compare_windows.py
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))
import config as C                                                       # noqa: E402
from util import fmt                                                     # noqa: E402

D = importlib.import_module("5_decide")
FULL = ("full", "4_toys", "full range [1000, 8000]")
PANEL_METHODS = ["full", "k3", "k5"]          # the three-panel figure
REF = ("full", "powexp")                      # the sensitivity reference: the ee nominal of step 6
SIG_MASSES = [1400, 1600, 2000, 2600, 3200, 4000]
CAND_ORDER = ["expo", "powlaw", "powexp", "atlasemu", "dilep8"]
WIDTH_CANDIDATES = ["expo", "powlaw", "powexp", "atlasemu"]   # the width scan (k = 2, 3, 4, 5): tables and figures
LS = {"full": "-", "k2": (0, (1, 1)), "k3": "--", "k4": "-.", "k5": ":"}


def methods_for(channel):
    """[(key, step, label)]: the full range plus every window size found on disk."""
    out = [FULL]
    ks = []
    for p in (C.RESULTS / channel / "6_windows").glob("k*/toys_all.csv"):
        mm = re.fullmatch(r"k([0-9.]+)", p.parent.name)
        if mm:
            ks.append(float(mm.group(1)))
    for k in sorted(ks):
        out.append((f"k{k:g}", f"6_windows/k{k:g}", f"window +-{k:g} sigma"))
    return out


def mlabel(key):
    return "full range" if key == "full" else rf"window $\pm{key[1:]}\sigma$"


def mls(key):
    return LS.get(key, "--")


def ref_err(rows_by_method, variant="lo_ht", truth="raw"):
    """{mass: Asimov sigma_fit} of the reference (method, candidate) on one truth."""
    out = {}
    for r in rows_by_method.get(REF[0], []):
        if (r["candidate"] == REF[1] and r["variant"] == variant and r["truth"] == truth and r["inject"] == 0.0
                and np.isfinite(r["asimov_err"])):
            out[r["mass"]] = r["asimov_err"]
    return out


def sigma_ratio(rows, cand, ref, masses, variant="lo_ht", truth="raw"):
    """Asimov sigma_fit / reference per mass (nan where either is missing)."""
    out = {}
    for m in masses:
        r = next((x for x in rows if x["candidate"] == cand and x["variant"] == variant and x["truth"] == truth
                  and x["inject"] == 0.0 and x["mass"] == m), None)
        out[m] = (r["asimov_err"] / ref[m]) if (r is not None and m in ref and ref[m] > 0 and np.isfinite(r["asimov_err"])) else float("nan")
    return out


def plot_sspur_methods(channel, cand, rows_by_method, out):
    """S_spur vs m_WR for one candidate and one fit range (full, +-3, +-5 sigma), one figure each,
    one line per primary truth (colour = smoother, style = DY sample)."""
    import matplotlib.pyplot as plt
    import plots as P
    for key in PANEL_METHODS:
        if key not in rows_by_method or not any(r["candidate"] == cand for r in rows_by_method[key]):
            continue
        fig, ax = P.scan_figure()
        handles = P.truth_lines(ax, rows_by_method[key], cand, D.PRIMARY)
        if not handles:
            plt.close(fig)
            continue
        P.bias_lines(ax)
        P.finish_scan(ax, channel=channel, extra=f"{C.MENU_LABEL.get(cand, cand)}\n{mlabel(key)}",
                      ylabel=P.SSPUR_LABEL, ylim=(-2.5, 3.0), handles=handles, ncol=1)
        P.save(fig, out / f"sspur_methods_{cand}_{key}")


def plot_window_width(channel, cand, rows_by_method, methods, out):
    """The width scan of one candidate: top, the band (min to max over the primary truths) and the
    median of S_spur vs m_WR per method; bottom, the Asimov sigma_fit relative to the full-range
    powexp. Full range in black, the windows from narrow (light) to wide (dark)."""
    import matplotlib.pyplot as plt
    import mplhep as hep
    import plots as P
    hep.style.use("CMS")
    keys = [k for k, _, _ in methods if k in rows_by_method and any(r["candidate"] == cand for r in rows_by_method[k])]
    wins = [k for k in keys if k != "full"]
    if not wins:
        return
    cmap = plt.get_cmap("viridis")
    color = {"full": "k"}
    for i, k in enumerate(wins):
        color[k] = cmap(0.15 + 0.7 * i / max(len(wins) - 1, 1))
    ref = ref_err(rows_by_method)
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    fig, (ax, axb) = P.scan_figure2()
    for key in keys:
        rows = [r for r in rows_by_method[key] if r["candidate"] == cand and r["variant"] in D.PRIMARY
                and r["inject"] == 0.0 and np.isfinite(r["bias_median"])]
        masses = sorted({r["mass"] for r in rows})
        lo = [min(r["bias_median"] for r in rows if r["mass"] == m) for m in masses]
        hi = [max(r["bias_median"] for r in rows if r["mass"] == m) for m in masses]
        med = [float(np.median([r["bias_median"] for r in rows if r["mass"] == m])) for m in masses]
        ax.fill_between(masses, lo, hi, color=color[key], alpha=0.18, lw=0)
        ax.plot(masses, med, color=color[key], lw=1.8, ls=mls(key), marker="o", ms=4,
                label="full range" if key == "full" else rf"$\pm{key[1:]}\sigma$")
        if ref:
            rat = sigma_ratio(rows_by_method[key], cand, ref, masses)
            xs = [m for m in masses if np.isfinite(rat[m])]
            axb.plot(xs, [rat[m] for m in xs], color=color[key], lw=1.8, ls=mls(key), marker="o", ms=4)
    P.bias_lines(ax)
    ax.set_ylim(-4.4, 2.4)                                   # room for the legend under the lines
    ax.set_ylabel(P.SSPUR_LABEL)
    cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(channel, C.MENU_LABEL.get(cand, cand)), year=C.YEAR)
    # band = the range over the 10 primary truths, line = their median
    ax.legend(loc="lower right", fontsize=FONT_SIZE_LEGEND, ncol=3, handlelength=1.2, columnspacing=0.8)
    fit_header(ax)
    axb.axhline(1.0, color="k", lw=0.8)
    axb.set_ylim(0.6, 4.0)
    axb.set_ylabel(r"$\sigma_{\rm fit}$ / full-range powexp")
    axb.set_xlabel(P.MWR_LABEL)
    P.save(fig, out / f"window_width_{cand}")


def plot_sigma_methods(channel, rows_by_method, out, cands=("expo", "powexp", "atlasemu"), keys=("full", "k3", "k5")):
    """Asimov sigma_fit of (fit range, candidate) relative to the full-range powexp: colour = candidate,
    style = fit range."""
    import published_functions as pf
    import plots as P
    from matplotlib.lines import Line2D
    ref = ref_err(rows_by_method)
    if not ref:
        return
    masses = sorted(ref)
    fig, ax = P.scan_figure()
    seen_c, seen_k = [], []
    for key in keys:
        if key not in rows_by_method:
            continue
        for cand in cands:
            rat = sigma_ratio(rows_by_method[key], cand, ref, masses)
            xs = [m for m in masses if np.isfinite(rat[m])]
            if not xs:
                continue
            ax.plot(xs, [rat[m] for m in xs], marker="o", ms=4, lw=1.6, color=pf.COLOR[cand], ls=mls(key))
            if cand not in seen_c:
                seen_c.append(cand)
            if key not in seen_k:
                seen_k.append(key)
    ax.axhline(1.0, color="k", lw=0.8)
    handles = [Line2D([], [], color=pf.COLOR[c], lw=1.8, label=C.MENU_LABEL.get(c, c)) for c in seen_c]
    handles += [Line2D([], [], color="k", lw=1.8, ls=mls(k), label=mlabel(k)) for k in seen_k]
    P.finish_scan(ax, channel=channel, extra="raw MC truth",
                  ylabel=r"$\sigma_{\rm fit}(N_{\rm sig})$ / full-range powexp (Asimov)", ylim=(0.5, 4.0),
                  handles=handles, ncol=1)
    P.save(fig, out / "sigma_fit_methods")


def verdict_table(L, rows_by_method, methods, sparse, ref):
    L += ["| method | candidate | max abs S_spur (where) | max abs bias at 1x / 2x (where) | min passed fraction | "
          "width range | cells > 0.3 | cells > 0.5 | sigma_fit / full powexp | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    passing = {}
    for key, _, label in methods:
        if key not in rows_by_method:
            continue
        rows = rows_by_method[key]
        sub = [r for r in rows if r["mass"] >= 3400.0] if sparse else rows
        V = D.verdicts(sub, D.PRIMARY, 1e9 if sparse else D.FLOAT_MAX)
        passing[key] = [c for c, v in V.items() if v["pass"]]
        for cand in sorted(V, key=lambda n: (CAND_ORDER + [n]).index(n)):
            v = V[cand]
            rat = sigma_ratio(rows, cand, ref, [m for m in sorted(ref) if (m >= 3400) == sparse])
            vals = [x for x in rat.values() if np.isfinite(x)]
            srel = f"{min(vals):.2f} - {max(vals):.2f}" if vals else "-"
            verdict = "target" if v["target"] else ("pass" if v["pass"] else "fail")
            L.append(f"| {label} | {cand} | {fmt(v['max_sspur'])} ({v['where_sspur']}) | {fmt(v['max_bias_inj'])} "
                     f"({v['where_bias_inj']}) | {fmt(v['min_passed'])} | {fmt(v['width_min'])} - {fmt(v['width_max'])} | "
                     f"{v['n_fail03']} / {v['n_cells']} | {v['n_fail05']} | {srel} | {verdict} |")
    L.append("")
    return passing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    args = ap.parse_args()
    L = ["# background_fit: full-range fit versus window fits", "",
         f"Same truths, toys ({C.NTOYS} per cell), candidates and injections as step 5; the fit range is the only "
         f"difference. Windows: [max(mu - k sigma, {C.FIT_LO:.0f}), min(mu + k sigma, 8000)] GeV, every background "
         "parameter free; k = 3 is the current method, k = 5 the wider variant (all candidates), k = 2 and 4 the width "
         "scan (exponential and power law). Verdicts over the primary truths "
         f"({', '.join(D.PRIMARY)} x {len(C.TRUTH_NAMES)} smoothers), float regime m_WR <= {D.FLOAT_MAX:.0f} GeV; pass "
         f"|S_spur| and |bias| < {C.BIAS_PASS} with >= 90 % of the toys converged. With {C.NTOYS} toys one S_spur value "
         f"carries a statistical error of about {1.25 / np.sqrt(C.NTOYS):.2f}. sigma_fit / full powexp: the Asimov "
         "sigma_fit of the method and candidate over that of the full-range powexp fit (the ee nominal), LO HT raw "
         "truth, range over the masses of the regime.", ""]
    passing_all, methods_all = {}, {}
    for channel in args.channels:
        methods = methods_for(channel)
        methods_all[channel] = methods
        rows_by_method = {}
        for key, step, _ in methods:
            rows = D.load_rows(channel, step)
            if rows:
                rows_by_method[key] = rows
        if "full" not in rows_by_method:
            L += [f"## {channel}: no step 5 results", ""]
            continue
        out = C.RESULTS / channel / "6_windows"
        out.mkdir(parents=True, exist_ok=True)
        ref = ref_err(rows_by_method)
        L += [f"## {channel} {C.REGION_NAME}", "", "### Primary truths, float regime (m_WR <= 3200 GeV)", ""]
        passing_all[channel] = verdict_table(L, rows_by_method, methods, False, ref)
        L += ["### Primary truths, sparse regime (m_WR >= 3400 GeV)", ""]
        verdict_table(L, rows_by_method, methods, True, ref)
        L += ["### Asimov sigma_fit relative to the full-range powexp (LO HT raw truth)", "",
              "| method | candidate | " + " | ".join(str(m) for m in SIG_MASSES) + " |", "|---|---|" + "---|" * len(SIG_MASSES)]
        for key, _, label in methods:
            if key not in rows_by_method:
                continue
            for cand in CAND_ORDER:
                rat = sigma_ratio(rows_by_method[key], cand, ref, SIG_MASSES)
                if all(not np.isfinite(x) for x in rat.values()):
                    continue
                L.append(f"| {label} | {cand} | " + " | ".join(fmt(rat[m]) for m in SIG_MASSES) + " |")
        L.append("")
        for cand in WIDTH_CANDIDATES:
            L += [f"### {C.MENU_LABEL.get(cand, cand)}: S_spur over the primary truths, median [min, max], float regime", "",
                  "| method | " + " | ".join(str(m) for m in C.MASSES if m <= D.FLOAT_MAX) + " | max abs | cells > 0.5 |",
                  "|---|" + "---|" * (len([m for m in C.MASSES if m <= D.FLOAT_MAX]) + 2)]
            for key, _, label in methods:
                if key not in rows_by_method:
                    continue
                rows = [r for r in rows_by_method[key] if r["candidate"] == cand and r["variant"] in D.PRIMARY
                        and r["inject"] == 0.0 and np.isfinite(r["bias_median"]) and r["mass"] <= D.FLOAT_MAX]
                if not rows:
                    continue
                cells, worst, nbad = [], 0.0, 0
                for m in [m for m in C.MASSES if m <= D.FLOAT_MAX]:
                    v = [r["bias_median"] for r in rows if r["mass"] == m]
                    if not v:
                        cells.append("-")
                        continue
                    cells.append(f"{np.median(v):+.2f} [{min(v):+.2f}, {max(v):+.2f}]")
                    worst = max(worst, max(abs(x) for x in v))
                    nbad += sum(abs(x) > C.BIAS_PASS for x in v)
                L.append(f"| {label} | " + " | ".join(cells) + f" | {worst:.2f} | {nbad} |")
            L.append("")
        cands = sorted({r["candidate"] for rows in rows_by_method.values() for r in rows},
                       key=lambda n: (CAND_ORDER + [n]).index(n))
        for cand in cands:
            plot_sspur_methods(channel, cand, rows_by_method, out)
        for cand in WIDTH_CANDIDATES:
            plot_window_width(channel, cand, rows_by_method, methods, out)
        plot_sigma_methods(channel, rows_by_method, out)
        L += [f"Figures: results/{channel}/6_windows/sspur_methods_<candidate>, window_width_<candidate> "
              f"({', '.join(WIDTH_CANDIDATES)}) and sigma_fit_methods.", ""]
    L += ["## Across channels (float regime, primary truths)", ""]
    keys = []
    for ch in passing_all:
        for key, _, _ in methods_all[ch]:
            if key not in keys:
                keys.append(key)
    for key in keys:
        if any(key not in passing_all[ch] for ch in passing_all):
            L.append(f"- {mlabel(key).replace('$', '').replace('\\\\pm', '+-').replace('\\\\sigma', ' sigma')}: not run in every channel.")
            continue
        common = set.intersection(*[set(passing_all[ch][key]) for ch in passing_all])
        per = "; ".join(f"{ch}: {', '.join(passing_all[ch][key]) or 'none'}" for ch in passing_all)
        name = "full range" if key == "full" else f"window +-{key[1:]} sigma"
        L.append(f"- {name}: passing in both channels: {', '.join(sorted(common)) or 'none'} ({per}).")
    L.append("")
    (C.RESULTS / "WINDOWS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-8:]))
    print("wrote", C.RESULTS / "WINDOWS.md")


if __name__ == "__main__":
    main()
