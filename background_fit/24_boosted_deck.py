#!/usr/bin/env python3
"""The figures and numbers of the boosted study: the stacked backgrounds first, then the toys and the pulls.
The boosted fit: the running power law over the full range.

  models  per channel, results_boosted*/<ch>/24_deck/:
            fit_curves         the summed MC with the two fitted curves that serve as toy backgrounds (the power x
                               exponential and the running power law, 3_truths_fit); below, (MC - fit) / sigma_MC
            cumulative         N(m > m_min) of the MC (band = its statistical error) and of the four curves, from
                               the fit floor up; below, curve / MC
            slices_cumulative  events per 100 GeV slice to the last MC event; below, the running sum of curve - MC
          and results_boosted*/BOOSTED_MODELS.md, deck_numbers_models.json
  grids   the zero-injection pulls against the mass (toys 6_windows/k<k>[_lo<lo>]<tag>/), per channel in 24_deck/:
            bias5_<function>_lo<floor>_{pull_mean,width}<tag>        the five backgrounds, full range from the floor
            bias1_<function>_lo<floor>_raw_{pull_mean,width}<tag>    the raw MC alone, full range from the floor
            ranges_<function>_<background>_{pull_mean,width,pass_fraction}<tag>
                                                                     +-5 sigma, +-7 sigma, the full range (from 800 GeV)
            floors_<function>_<background>_{pull_mean,width}<tag>    the full range from 800 / 1000 / 1200 GeV
            functions_lo<floor>_<background>_{pull_mean,width}<tag>  the power x exponential against the running power law
          and results_boosted*/deck_numbers_grids.json
  signal  the Gaussian of the S+B fit: the straight lines for its mean and width on the
          core fits they were fitted to, 24_deck/signal_{mean,width}; thirteen of those points drawn out together,
          24_deck/signal_overlay (the samples with m_N closest to 0.05 m_WR, each spectrum with its Gaussian); a
          check on Run 3 samples for the notes;
          results_boosted*/deck_numbers_signal.json

  cd WR_Plotter && FIT_TOPOLOGY=boosted FIT_DATASET=run3 python background_fit/24_boosted_deck.py models|grids [--tag _100k]
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))
import config as C                                                       # noqa: E402

logger = logging.getLogger("24_boosted_deck")
VARIANT = "lo_ht"
FLOOR = 800.0                    # the fit floor of the main line: the full range from 800 GeV, no windows
XRUN = 9000.0                    # the running-difference figure ends past the last MC event (8.6 TeV)
MC_COLOR = "#717581"
CUM_YMIN = 1e-3
PASS = 0.9
# key -> (results sub-directory, file, legend label, colour, line style)
MODELS = {
    "kde9": ("3_truths_kde9", "truth_kde", "RooKeysPdf", "#7a21dd", "-"),
    "kde_expo9": ("3_truths_split9", "truth_kde_expo", "RooKeysPdf + expo tail", "#5790fc", "--"),
    "fitpowexp": ("3_truths_fit", "truth_fitpowexp", r"power $\times$ expo fit", "#f89c20", "-."),
    "fitatlasemu": ("3_truths_fit", "truth_fitatlasemu", "running power law fit", "#e42536", ":"),
}
FIT_LABEL = {"fitpowexp": r"power $\times$ exponential", "fitatlasemu": "running power law"}
REF = [1000.0, 2000.0, 3000.0, 4000.0]
XLABEL = r"$m_\mathrm{min}$ [GeV]"
YLABEL = r"Events with $m_{\ell J} > m_\mathrm{min}$"
# the grids: key -> (legend label, colour, line style, marker)
BKG = {"raw": ("Raw MC", MC_COLOR, "-", "o"),
       "kde9": ("RooKeysPdf", "#7a21dd", "--", "s"),
       "kde_expo9": ("RooKeysPdf + expo tail", "#5790fc", "-.", "^"),
       "fitpowexp": (r"Power $\times$ expo fit", "#f89c20", ":", "D"),
       "fitatlasemu": ("Running power law fit", "#e42536", (0, (4, 1, 1, 1)), "v")}
FLOOR_STYLE = {800.0: ("#5790fc", "--", "s"), 1000.0: ("#000000", "-", "o"), 1200.0: ("#e42536", "-.", "^")}
FUNC_STYLE = {"powexp": ("#f89c20", "--", "s"), "atlasemu": ("#e42536", "-", "o"), "expo": ("#5790fc", ":", "^")}
# one-line names of the background for the label block (a two-line name pushes the block onto the 0.5 lines)
BKG_LINE = {"raw": "Raw MC Bkg", "kde9": "RooKeysPdf Bkg", "kde_expo9": "RooKeysPdf + tail Bkg",
            "fitpowexp": r"Power $\times$ expo fit Bkg", "fitatlasemu": "Running power law fit Bkg"}
STAT = {"pull_mean": ("mu", r"Gaussian $\mu$ of the pulls", (-2.0, 3.0)),
        "width": ("sigma", r"Gaussian $\sigma$ of the pulls", (0.0, 2.5)),
        "pass_fraction": ("pass", "Fraction of fits passing", (0.0, 1.45))}     # the four fit checks and |pull| <= 5


def rcum(x):
    """N(>= e_i): the sum of the bins from i on."""
    return np.cumsum(np.asarray(x, float)[::-1])[::-1]


def mc_first(ax, P):
    h, l = ax.get_legend_handles_labels()
    idx = sorted(range(len(l)), key=lambda i: 0 if l[i].startswith(("MC,", "Simulation")) else 1 if l[i].startswith("MC stat") else 2)
    ax.legend([h[i] for i in idx], [l[i] for i in idx], loc="upper right", fontsize=P.FONT_SIZE_LEGEND, handlelength=1.6)


# ---------------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------------
def run_models(args):
    import matplotlib.ticker as mticker
    import plots as P
    from spectrum import centers, load_components, rebin
    from truth_sets import load_truth
    L = [f"# The background models, {C.ERA} {C.REGION_NAME} ({C.LUMI:.1f} fb-1), LO HT-binned DY, nine processes", ""]
    N = {}
    for ch in args.channels:
        out = C.RESULTS / ch / "24_deck"
        out.mkdir(parents=True, exist_ok=True)
        comps10 = load_components(ch, VARIANT, "split9")
        e10, v10, var10 = comps10["sum"]
        c10 = centers(e10)
        curves, diags = {}, {}
        for key, (sub, name, *_r) in MODELS.items():
            p = C.RESULTS / ch / sub / VARIANT / name
            if not p.with_suffix(".npz").exists():
                logger.warning("%s: no %s", ch, p)
                continue
            _, parts, dg = load_truth(p)
            curves[key], diags[key] = np.asarray(parts["sum"], float), dg
        n = N.setdefault(ch, {})
        # --- the two fitted curves on the MC ---------------------------------------------------------------
        e, v, var = rebin(e10, v10, var10)
        c = centers(e)
        sig = np.sqrt(np.maximum(var, 0.0))
        sel = (c >= C.THRESHOLD) & (c <= P.XMAX_FULL)
        fig, (ax, axr) = P.spectrum_figure()
        ax.errorbar(c[sel], v[sel], yerr=sig[sel], fmt="o", ms=4, color="k", label="Simulation")
        fit_lo = None
        for key in ("fitpowexp", "fitatlasemu"):
            if key not in curves:
                continue
            fit_lo = float(diags[key]["sum"]["lo"])
            _, f, _ = rebin(e10, curves[key], np.zeros_like(v10))
            m = sel & (c >= fit_lo)
            ax.plot(c[m], np.maximum(f[m], P.YMIN_FULL), color=MODELS[key][3], lw=1.8, ls=MODELS[key][4], label=FIT_LABEL[key])
            pull = np.where(sig > 0, (v - f) / np.where(sig > 0, sig, 1.0), np.nan)
            axr.stairs(pull[m], np.append(e[:-1][m], e[1:][m][-1]), color=MODELS[key][3], lw=1.4, ls=MODELS[key][4])
            n[key] = {"lo": fit_lo, "dev": float(diags[key]["sum"]["dev"]), "params": [float(x) for x in diags[key]["sum"]["params"]]}
        for y in (-2, 2):
            axr.axhline(y, color="0.5", lw=0.6, ls="--")
        P.finish(ax, axr, channel=ch, xmax=P.XMAX_FULL, top=float(v[sel].max()), ymin=P.YMIN_FULL,
                 extra=f"{C.DY_SHORT[VARIANT]}\nfits from {fit_lo:.0f} GeV", ratio_label=r"(MC $-$ fit) / $\sigma_{\rm MC}$",
                 ratio_lim=(-5, 5), ratio_ref=0.0)
        axr.set_ylabel(r"(MC $-$ fit) / $\sigma_{\rm MC}$", fontsize=20)
        mc_first(ax, P)
        P.save(fig, out / "fit_curves")
        # --- N(m > m_min) -----------------------------------------------------------------------------------
        lo_e = e10[:-1]
        N_mc, E_mc = rcum(v10), np.sqrt(rcum(var10))
        sel = (lo_e >= FLOOR) & (lo_e <= P.XMAX_FULL)
        fig, (ax, axr) = P.spectrum_figure()
        ax.step(lo_e[sel], np.maximum(N_mc[sel], CUM_YMIN), where="post", color=MC_COLOR, lw=1.8, label="MC, 9 processes")
        ax.fill_between(lo_e[sel], np.maximum(N_mc[sel] - E_mc[sel], CUM_YMIN), np.maximum(N_mc[sel] + E_mc[sel], CUM_YMIN),
                        step="post", color=MC_COLOR, alpha=0.3, lw=0, label="MC stat. error")
        ok = sel & (N_mc > 0)
        axr.fill_between(lo_e[ok], 1.0 - E_mc[ok] / N_mc[ok], 1.0 + E_mc[ok] / N_mc[ok], step="post", color=MC_COLOR, alpha=0.3, lw=0)
        N_of = {}
        for key, cur in curves.items():
            _, _, lab, col, ls = MODELS[key]
            N_of[key] = rcum(cur)
            ax.plot(lo_e[sel], np.maximum(N_of[key][sel], CUM_YMIN), color=col, lw=1.8, ls=ls, label=lab)
            axr.plot(lo_e[ok], N_of[key][ok] / N_mc[ok], color=col, lw=1.8, ls=ls)
        P.finish(ax, axr, channel=ch, xmin=FLOOR, xmax=P.XMAX_FULL, top=float(N_mc[sel].max()), ymin=CUM_YMIN, ylabel=YLABEL,
                 extra=C.DY_SHORT[VARIANT], ratio_label="Curve / MC", ratio_lim=(0.2, 20.0))
        axr.set_xlabel(XLABEL)
        axr.set_yscale("log")
        axr.set_ylim(0.2, 20.0)
        axr.yaxis.set_major_locator(mticker.FixedLocator([0.2, 0.5, 1, 2, 5, 10, 20]))
        axr.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"{val:g}"))
        axr.yaxis.set_minor_formatter(mticker.NullFormatter())
        mc_first(ax, P)
        P.save(fig, out / "cumulative")
        # --- events per 100 GeV slice; below, the running sum of curve - MC from the threshold up -----------------
        fe = np.arange(C.THRESHOLD, XRUN + 50.0, 100.0)
        fsum = lambda x: np.array([float(np.asarray(x)[int(np.searchsorted(e10, a_)):int(np.searchsorted(e10, b_))].sum())   # noqa: E731
                                   for a_, b_ in zip(fe[:-1], fe[1:])])
        fmc, ferr = fsum(v10), np.sqrt(np.maximum(fsum(var10), 0.0))
        fc, fw = 0.5 * (fe[:-1] + fe[1:]), fe[1:] - fe[:-1]
        fig, (ax, axr) = P.spectrum_figure()
        ax.errorbar(fc, fmc, xerr=fw / 2, yerr=ferr, fmt="o", ms=5, color=MC_COLOR, lw=1.4, label="MC, 9 processes")
        top, runs = 0.0, {}
        for key, cur in curves.items():
            _, _, lab, col, ls = MODELS[key]
            cs = fsum(cur)
            ax.step(fe, np.append(cs, cs[-1]), where="post", color=col, lw=1.8, ls=ls, label=lab)
            runs[key] = np.cumsum(cs - fmc)
            axr.step(fe, np.append(runs[key], runs[key][-1]), where="post", color=col, lw=2.2, ls=ls)
            top = max(top, float(np.abs(runs[key]).max()))
        lim = 1.3 * top
        P.finish(ax, axr, channel=ch, xmax=float(fe[-1]), top=float(fmc.max()), ymin=1e-4, ylabel="Events in slice",
                 extra=C.DY_SHORT[VARIANT], ratio_label=r"$\sum$ (curve $-$ MC)", ratio_lim=(-lim, lim), ratio_ref=0.0)
        axr.set_ylabel(r"$\sum$ (curve $-$ MC)", fontsize=20)
        mc_first(ax, P)
        P.save(fig, out / "slices_cumulative")
        # --- the numbers ----------------------------------------------------------------------------------------
        heads = [k for k in curves]
        at = lambda arr, m: float(arr[int(np.searchsorted(lo_e, m))])                         # noqa: E731
        n["above"] = {f"{m:.0f}": {"mc": at(N_mc, m), "err": at(E_mc, m), **{k: at(N_of[k], m) for k in heads}}
                      for m in [C.THRESHOLD] + REF}
        n["run"] = {k: {"ext": float(runs[k][int(np.argmax(np.abs(runs[k])))]),
                        "at": float(fe[1:][int(np.argmax(np.abs(runs[k])))]), "end": float(runs[k][-1])} for k in heads}
        n["last_event"] = float(c10[np.nonzero(v10 > 0)[0][-1]])
        # per process: its share of the MC above the threshold, and where its exponential tail starts
        tot = float(v10[c10 >= C.THRESHOLD].sum())
        n["share"] = {k: float(comps10[k][1][c10 >= C.THRESHOLD].sum()) / tot for k in comps10 if k != "sum"}
        tails = {k: d for k, d in diags.get("kde_expo9", {}).items() if isinstance(d, dict) and k != "sum"}
        n["tail_start"] = {k: float(d["m_T"]) for k, d in tails.items() if d.get("tail_family")}
        n["no_tail"] = sorted(k for k, d in tails.items() if not d.get("tail_family"))      # too few events: the RooKeysPdf alone
        rng = lambda arr, a, b: float(np.asarray(arr)[(c10 >= a) & (c10 < b)].sum())           # noqa: E731
        n["mid"] = {"range": [3000.0, 5500.0], "mc": rng(v10, 3000.0, 5500.0), "err": float(np.sqrt(rng(var10, 3000.0, 5500.0))),
                    **{k: rng(curves[k], 3000.0, 5500.0) for k in heads}}
        L += [f"## {ch}", "", f"Last MC event at {n['last_event']:.0f} GeV.", "", "Events above m_min:", "",
              "| m_min [GeV] | MC | " + " | ".join(MODELS[k][2] for k in heads) + " |", "|---|---|" + "---|" * len(heads)]
        for m, row in n["above"].items():
            L.append(f"| {m} | {row['mc']:.4g} +- {row['err']:.2g} | "
                     + " | ".join(f"{row[k]:.4g} ({row[k] / row['mc']:.2f} x, {(row[k] - row['mc']) / row['err']:+.1f})" for k in heads) + " |")
        L += ["", "Running sum of curve - MC from the threshold, 100 GeV slices:", "",
              "| curve | largest excursion [events] | reached at [GeV] | at the end [events] |", "|---|---|---|---|"]
        L += [f"| {MODELS[k][2]} | {n['run'][k]['ext']:+.2f} | {n['run'][k]['at']:.0f} | {n['run'][k]['end']:+.2f} |" for k in heads]
        L += ["", "Events between 3 and 5.5 TeV: MC " + f"{n['mid']['mc']:.3f} +- {n['mid']['err']:.3f}; "
              + "; ".join(f"{MODELS[k][2]} {n['mid'][k]:.3f}" for k in heads), "",
              "Share of the MC above the threshold and the start of the exponential tail, per process:", "",
              "| process | share | tail starts [GeV] |", "|---|---|---|"]
        L += [f"| {k} | {s:.3f} | {'no tail fit' if k in n['no_tail'] else format(n['tail_start'].get(k, float('nan')), '.0f')} |"
              for k, s in sorted(n["share"].items(), key=lambda kv: -kv[1])]
        L.append("")
        logger.info("%s: wrote %s", ch, out)
    (C.RESULTS / "BOOSTED_MODELS.md").write_text("\n".join(L) + "\n")
    (C.RESULTS / "deck_numbers_models.json").write_text(json.dumps(N, indent=1))
    logger.info("wrote %s", C.RESULTS / "BOOSTED_MODELS.md")


# ---------------------------------------------------------------------------------------------
# grids
# ---------------------------------------------------------------------------------------------
PULL_RANGE = 5.0                 # the pull histogram and its Gaussian fit cover |pull| <= 5


def load_usable(B, ch, k, lo, truth, tag):
    """B.load_cells with the fits that count made explicit. With N_sig free, a toy with nothing under the peak is
    fitted where S+B reaches zero: the fit converges and passes the four fit checks, with an error of a few
    hundredths of an event and a pull near -100, outside the +-5 the pull histogram covers. A fit PASSES when it
    passes the four checks and |pull| <= 5 (the fifth criterion).
    Per cell: "pass_checks" = the share passing the four checks, "edge" = the share of those with |pull| > 5,
    "pass" = the share passing all five (what the 90 % rule, the filled markers and the example mass use)."""
    cells = B.load_cells(ch, k, lo, truth, tag)
    if not cells:
        return cells
    npz = C.RESULTS / ch / "6_windows" / (B.kdir(k, lo) + tag) / VARIANT / f"nsig_{truth}.npz"
    if not npz.exists():
        return cells
    z = np.load(npz)
    for (cand, m), cell in cells.items():
        key = f"{cand}_{int(m)}_0"
        cell["pass_checks"], cell["edge"] = cell["pass"], 0.0
        if key in z.files:
            n, e, ok = z[key], z[key + "_err"], z[key + "_ok"].astype(bool)
            if ok.any():
                pull = n[ok] / e[ok]
                cell["edge"] = float((np.abs(pull) > PULL_RANGE).mean())
                cell["pass"] = cell["pass_checks"] * (1.0 - cell["edge"])
    return cells


def of_candidate(cells, cand, max_mass):
    """{mass: cell} of one function, up to max_mass."""
    return {m: cell for (c, m), cell in (cells or {}).items() if c == cand and m <= max_mass}


def summarize(cells):
    """The passing cells (>= 90 % of the fits pass, a Gaussian was fitted): first, last, the largest |mu| with its mass,
    the cells beyond 0.5, the range of the pull sigma."""
    good = sorted(m for m, cell in cells.items() if cell["pass"] >= PASS and np.isfinite(cell["mu"]))
    if not good:
        return None
    worst = max(good, key=lambda m: abs(cells[m]["mu"]))
    sig = [cells[m]["sigma"] for m in good if np.isfinite(cells[m]["sigma"])]
    return {"first": good[0], "last": good[-1], "max": cells[worst]["mu"], "at": worst,
            "gt05": [[m, cells[m]["mu"]] for m in good if abs(cells[m]["mu"]) > C.BIAS_PASS],
            "smin": min(sig) if sig else None, "smax": max(sig) if sig else None,
            "pass": {f"{m:.0f}": cells[m]["pass"] for m in sorted(cells)},
            "pass_checks": {f"{m:.0f}": cells[m].get("pass_checks", cells[m]["pass"]) for m in sorted(cells)},
            "edge": {f"{m:.0f}": cells[m].get("edge", 0.0) for m in sorted(cells)},
            "sfit": {f"{m:.0f}": cells[m]["sfit"] for m in sorted(cells)},
            "mu": {f"{m:.0f}": cells[m]["mu"] for m in sorted(cells)},
            "signal": {f"{m:.0f}": [cells[m]["sig_mu"], cells[m]["sig_sigma"]] for m in sorted(cells)}}   # the Gaussian of the fit


def scan(P, ch, series, extra, stat, path):
    """One quantity against the mass, one line per entry of series = [(label, colour, line style, marker, {mass: cell})];
    filled markers where at least 90 % of the fits pass."""
    from matplotlib.lines import Line2D
    key, ylabel, ylim = STAT[stat]
    fig, ax = P.scan_figure()
    handles = []
    for lab, col, ls, mk, cells in series:
        ms = np.array(sorted(cells))
        if not len(ms):
            continue
        y = np.array([cells[m][key] for m in ms], float)
        ok = np.array([cells[m]["pass"] >= PASS for m in ms]) if stat != "pass_fraction" else np.ones(len(ms), bool)
        if stat == "pull_mean":
            y = np.clip(y, ylim[0] + 0.02, ylim[1] - 0.02)
        ax.plot(ms, y, lw=1.5, color=col, ls=ls)
        ax.plot(ms[ok], y[ok], mk, ms=6, color=col, ls="none")
        ax.plot(ms[~ok], y[~ok], mk, ms=6, mfc="white", mec=col, ls="none")
        handles.append(Line2D([], [], color=col, lw=1.5, ls=ls, marker=mk, ms=6, label=lab))
    if stat == "pull_mean":
        P.bias_lines(ax)
    elif stat == "width":
        ax.axhline(1.0, color="k", lw=0.8)
    else:
        ax.axhline(PASS, color="0.3", lw=0.9, ls="--")
    if stat != "pass_fraction":
        handles.append(Line2D([], [], color="k", lw=0, marker="o", mfc="white", ms=6, label="open: < 90 % pass"))
    P.finish_scan(ax, channel=ch, extra=extra, ylabel=ylabel, ylim=ylim, handles=handles, ncol=1)
    P.save(fig, path)


def run_grids(args):
    import plots as P
    B = importlib.import_module("21_boosted")
    tag, mx = args.tag, args.max_mass
    N = {}
    for ch in args.channels:
        out = C.RESULTS / ch / "24_deck"
        out.mkdir(parents=True, exist_ok=True)
        n = N.setdefault(ch, {"bias": {}, "ranges": {}, "floors": {}, "main": {}})
        load = lambda k, lo, tr, cand: of_candidate(load_usable(B, ch, k, lo, tr, tag), cand, mx)      # noqa: E731
        for cand in args.candidates:
            form = B.FORM_LABEL.get(cand, cand)
            # the five backgrounds, the full range from each floor
            for lo in B.FLOORS:
                series = [(lab, col, ls, mk, load(100, lo, tr, cand)) for tr, (lab, col, ls, mk) in BKG.items()]
                series = [s for s in series if s[4]]
                if not series:
                    continue
                for stat in ("pull_mean", "width"):
                    scan(P, ch, series, f"{form}\nfull range\nfrom {lo:.0f} GeV", stat, out / f"bias5_{cand}_lo{lo:.0f}_{stat}{tag}")
                # the raw MC alone: the main line (toys from the raw MC, full range from 800 GeV)
                # its mass range may reach further than the other grids' (--main-max-mass)
                main = of_candidate(load_usable(B, ch, 100, lo, "raw", tag), cand, args.main_max_mass or mx)
                if main:
                    lab, col, ls, mk = BKG["raw"]
                    for stat in ("pull_mean", "width"):
                        scan(P, ch, [(lab, col, ls, mk, main)], f"{form}\nfull range\nfrom {lo:.0f} GeV", stat,
                             out / f"bias1_{cand}_lo{lo:.0f}_raw_{stat}{tag}")
                    n["main"].setdefault(cand, {})[f"{lo:.0f}"] = summarize(main)
                n["bias"].setdefault(cand, {})[f"{lo:.0f}"] = {tr: summarize(load(100, lo, tr, cand)) for tr in BKG if load(100, lo, tr, cand)}
            for tr in BKG:
                # the window width at the 800 GeV floor
                series = [(B.WIDTH_LABEL[k], B.WIDTH_COLOR[k], "-", "o", load(k, 800.0, tr, cand)) for k in (5, 7, 100)]
                series = [s for s in series if s[4]]
                if len(series) > 1:
                    for stat in STAT:
                        scan(P, ch, series, f"{form}\n{BKG_LINE[tr]}\nfrom 800 GeV", stat, out / f"ranges_{cand}_{tr}_{stat}{tag}")
                    n["ranges"].setdefault(cand, {})[tr] = {f"{k}": summarize(load(k, 800.0, tr, cand)) for k in (5, 7, 100) if load(k, 800.0, tr, cand)}
                # the floor of the full range
                series = [(f"from {lo:.0f} GeV", *FLOOR_STYLE[lo], load(100, lo, tr, cand)) for lo in B.FLOORS]
                series = [s for s in series if s[4]]
                if len(series) > 1:
                    for stat in ("pull_mean", "width"):
                        scan(P, ch, series, f"{form}\n{BKG_LINE[tr]}\nfull range", stat, out / f"floors_{cand}_{tr}_{stat}{tag}")
        # the two functions over the full range
        for lo in B.FLOORS:
            for tr in BKG:
                series = [(B.FORM_LABEL.get(cand, cand), *FUNC_STYLE[cand], load(100, lo, tr, cand)) for cand in args.candidates]
                series = [s for s in series if s[4]]
                if len(series) > 1:
                    for stat in ("pull_mean", "width"):
                        scan(P, ch, series, f"{BKG_LINE[tr]}\nfull range\nfrom {lo:.0f} GeV", stat, out / f"functions_lo{lo:.0f}_{tr}_{stat}{tag}")
        logger.info("%s: wrote %s", ch, out)
    # the default toys (the UL18-based signal map) keep the plain name; another tag (e.g. _100k_sig2024) has its own file
    name = f"deck_numbers_grids{'' if tag == '_100k' else tag}.json"
    (C.RESULTS / name).write_text(json.dumps(N, indent=1))
    logger.info("wrote %s", C.RESULTS / name)


# ---------------------------------------------------------------------------------------------
# signal: the Gaussian of the S+B fit, its mean and width against the mass
# ---------------------------------------------------------------------------------------------
SIGNAL_X = 0.05                            # per mass, the sample with m_N closest to this fraction of m_WR
SIGNAL_MAX = 6000.0                        # the highest mass of the lines
SIGNAL_REBIN = 2                           # at least this many 10 GeV bins per drawn bin
SIGNAL_GRID_LO = 1000.0                    # the lowest mass of the toy grid: the lines are drawn down to it
SIGNAL_SOURCES = {
    # the Run 3 samples: one sample per mass, the one with m_N
    # closest to SIGNAL_X m_WR, core-fitted here; the lines are fitted here too and written to signal_map_2024.json
    "2024": {"label": "2024 signal", "year": "2024", "com": 13.6,
             "files": HERE.parent / "rootfiles" / "Run3" / "2024" / "RunIII2024Summer24" / "20261116_boosted_signals",
             "overlay": [2000, 2400, 2800, 3200, 3600, 4000, 4400, 4800, 5200, 6000]},
    # the map the toys used before, from the UL18 samples (median over m_N of the samples with m_N / m_WR < 0.1,
    # the lines of window_params.json); its table of per-sample Gaussian fits, gauss_fit_table.csv, is not tracked
    "ul18": {"label": "UL18 signal", "year": "2018", "com": 13.0,
             "files": HERE.parent / "rootfiles" / "RunII" / "2018" / "RunIISummer20UL18" / "20260624_signals",
             "table": HERE / "gauss_fit_table.csv",
             "map": HERE / "window_params.json",
             "overlay": list(range(1200, 6001, 400))},
}
SIGNAL_RANGE = "[0.7,1.3]"                 # the seed window of the core fits


def core_fit(c, v, e, m):
    """The stage-1 recipe: a Gaussian fitted in [0.7, 1.3] m, refitted in mean +- 2 sigma until both move by < 2 %."""
    from scipy.optimize import curve_fit
    gauss = lambda x, a, mu, s: a * np.exp(-0.5 * ((x - mu) / s) ** 2)      # noqa: E731
    lo, hi, mu, s = 0.7 * m, 1.3 * m, float(m), 0.06 * m
    for it in range(40):
        sel = (c >= lo) & (c <= hi) & (v > 0)
        par, _ = curve_fit(gauss, c[sel], v[sel], p0=[v[sel].max(), mu, s], sigma=e[sel], absolute_sigma=True)
        mu_new, s_new = float(par[1]), abs(float(par[2]))
        done = it >= 2 and abs(mu_new - mu) < 0.02 * abs(mu) and abs(s_new - s) < 0.02 * s
        mu, s = mu_new, s_new
        lo, hi = mu - 2 * s, mu + 2 * s
        if done:
            break
    return mu, s


def signal_hist(path, ch, topo):
    """(edges, values, variances) of the signal's mass spectrum in the SR, 10 GeV bins."""
    import uproot
    var = "mass_twoobject" if topo == "boosted" else "mass_fourobject"
    h = uproot.open(path)[f"wr_{ch}_{topo}_sr/{var}_wr_{ch}_{topo}_sr"]
    v, edges = h.to_numpy()
    return edges, v, h.variances()


def signal_inputs(source, ch, topo):
    """Per source: the points of the line figures [(m_WR, mean, width)], the lines {"mean": (slope, intercept),
    "width": ...} and the samples drawn in the overlay {m_WR: (m_N, mean, width, path)}."""
    import collections
    import csv
    import re
    cfg = SIGNAL_SOURCES[source]
    files = collections.defaultdict(dict)
    for f in Path(cfg["files"]).glob("WRAnalyzer_signal_WR*_N*.root"):
        mt = re.match(r"WRAnalyzer_signal_WR(\d+)_N(\d+)\.root", f.name)
        files[int(mt.group(1))][int(mt.group(2))] = f
    closest = lambda m, mns: min(sorted(mns), key=lambda x: (abs(x / m - SIGNAL_X), x))      # noqa: E731
    if "table" in cfg:                                   # the per-sample Gaussian fits
        if not Path(cfg["table"]).exists():
            raise SystemExit(f"--signal-source {source} needs {cfg['table']}")
        grid = collections.defaultdict(dict)
        for r in csv.DictReader(open(cfg["table"])):
            if r["fit_range"] == SIGNAL_RANGE and r["category"] == topo and r["channel"] == ch:
                grid[int(float(r["mWR"]))][int(float(r["mN"]))] = (float(r["mu_gaus"]), float(r["sigma_gaus"]))
        ms = sorted(m for m in grid if m <= SIGNAL_MAX)
        pts = [(m, float(np.median([x[0] for x in grid[m].values()])), float(np.median([x[1] for x in grid[m].values()]))) for m in ms]
        mp = json.loads(Path(cfg["map"]).read_text())[ch][topo]
        lines = {"mean": tuple(mp["mu"]), "width": tuple(mp["sigma_median"])}
        samples = {}
        for m in ms:
            mn = closest(m, grid[m])
            if mn in files.get(m, {}):
                samples[m] = (mn, *grid[m][mn], files[m][mn])
        how = "the median over $m_N$"
    else:                                                # fitted here, one sample per mass
        samples = {}
        for m in sorted(x for x in files if x <= SIGNAL_MAX):
            mn = closest(m, files[m])
            edges, v, var = signal_hist(files[m][mn], ch, topo)
            mu, sg = core_fit(0.5 * (edges[:-1] + edges[1:]), v, np.sqrt(var), float(m))
            samples[m] = (mn, mu, sg, files[m][mn])
        pts = [(m, s_[1], s_[2]) for m, s_ in sorted(samples.items())]
        x = np.array([p_[0] for p_ in pts], float)
        lines = {"mean": tuple(float(t) for t in np.polyfit(x, [p_[1] for p_ in pts], 1)),
                 "width": tuple(float(t) for t in np.polyfit(x, [p_[2] for p_ in pts], 1))}
        how = "one sample per mass"
    return pts, lines, samples, how


def run_signal(args):
    """The Gaussian the fit takes its signal from, per channel and source (--signal-source), in 24_deck/:
      signal_overlay_<source>   the samples with m_N closest to 0.05 m_WR, each in its window 0.7-1.3 m_WR and scaled
                                so that its Gaussian core fit peaks at 1 (thin: simulation, thick: the fit)
      signal_mean_<source>, signal_width_<source>
                                the core fits against the mass with the straight line; below, point / line - 1
    For the 2024 source the lines are fitted here and written to background_fit/signal_map_2024.json (the layout of
    window_params.json; FIT_SIGNAL_MAP points the toys at it). Numbers: deck_numbers_signal_<source>.json."""
    import matplotlib.pyplot as plt
    import plots as P
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.lines import Line2D
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    topo, source = C.TOPOLOGY, args.signal_source
    cfg = SIGNAL_SOURCES[source]
    other = next(k for k in SIGNAL_SOURCES if k != source)
    N, out_map = {}, {}
    for ch in args.channels:
        out = C.RESULTS / ch / "24_deck"
        out.mkdir(parents=True, exist_ok=True)
        pts, lines, samples, how = signal_inputs(source, ch, topo)
        try:
            _, lines_other, _, _ = signal_inputs(other, ch, topo)
        except Exception as ex:                                  # the comparison is for the notes only
            logger.warning("%s: no %s lines (%s)", ch, other, ex)
            lines_other = None
        ms = np.array([p_[0] for p_ in pts], float)
        n = N.setdefault(ch, {"source": cfg["label"], "how": how, "n": int(len(ms)), "mmin": float(ms[0]), "mmax": float(ms[-1]),
                              "x_min": float(min(s_[0] / m for m, s_ in samples.items())),
                              "x_max": float(max(s_[0] / m for m, s_ in samples.items()))})
        label_extra = cfg["label"] + (f"\n$m_N \\approx {SIGNAL_X:g}\\,m_{{W_R}}$" if "table" not in cfg else "\n$m_N / m_{W_R} < 0.1$")
        # --- the lines ---------------------------------------------------------------------------------------
        for what, col_i, ylabel in (("mean", 1, "Signal mean [GeV]"), ("width", 2, "Signal width [GeV]")):
            b, a = lines[what]                                   # slope, intercept
            line = lambda x, a=a, b=b: a + b * x                  # noqa: E731
            y = np.array([p_[col_i] for p_ in pts], float)
            resid = 100.0 * (y - line(ms)) / line(ms)
            n[what] = {"a": float(a), "b": float(b), "resid_max": float(np.abs(resid).max()),
                       "resid_at": float(ms[int(np.argmax(np.abs(resid)))]),
                       "at": {f"{m:.0f}": float(line(m)) for m in (1000.0, 2000.0, 2400.0, 4000.0)}}
            if lines_other:
                bo, ao = lines_other[what]
                n[what]["other"] = {"source": SIGNAL_SOURCES[other]["label"],
                                    "at": {f"{m:.0f}": float(ao + bo * m) for m in (1000.0, 2000.0, 2400.0, 4000.0)}}
            fig, (ax, axr) = P.scan_figure2(ratios=(3, 1))
            ax.plot(ms, y, "o", ms=7, color="k")
            xs = np.linspace(ms[0], ms[-1], 200)
            ax.plot(xs, line(xs), color="#e42536", lw=2.0)
            handles = [Line2D([], [], color="k", lw=0, marker="o", ms=7, label="Gaussian core fits"),
                       Line2D([], [], color="#e42536", lw=2.0,
                              label=rf"{b:.4f}$\,m_{{W_R}}$ {'+' if a >= 0 else '$-$'} {abs(a):.0f} GeV")]
            if ms[0] > SIGNAL_GRID_LO:                           # the toy grid starts below the first sample
                xe = np.linspace(SIGNAL_GRID_LO, ms[0], 50)
                ax.plot(xe, line(xe), color="#e42536", lw=2.0, ls=(0, (2, 2)))
                handles.append(Line2D([], [], color="#e42536", lw=2.0, ls=(0, (2, 2)), label=f"extrapolated below {ms[0] / 1000:g} TeV"))
            axr.plot(ms, resid, "o", ms=6, color="k")
            axr.axhline(0.0, color="#e42536", lw=1.5)
            lim = max(2.0, 1.3 * float(np.abs(resid).max()))
            axr.set_ylim(-lim, lim)
            axr.set_ylabel("Point / line $-$ 1 [%]", fontsize=20)
            axr.set_xlabel(P.MWR_LABEL)
            axr.set_xlim(min(SIGNAL_GRID_LO, ms[0]) - 250.0, ms[-1] + 250.0)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0.0, 1.05 * float(max(y.max(), line(ms[-1]))))
            cms_label(ax, None, cfg["com"], data=False, year=cfg["year"], text=P.region_text(ch, label_extra))
            ax.legend(handles=handles, loc="lower right", fontsize=FONT_SIZE_LEGEND, handlelength=1.8)
            fit_header(ax)
            P.save(fig, out / f"signal_{what}_{source}")
        out_map[ch] = {topo: {"mu": [lines["mean"][0], lines["mean"][1]], "sigma_median": [lines["width"][0], lines["width"][1]],
                              "mwr_min": float(ms[0]), "mwr_max": float(ms[-1]), "n_mwr": int(len(ms)),
                              "source": f"{cfg['label']}, {how}, m_N closest to {SIGNAL_X:g} m_WR, core fit seeded in {SIGNAL_RANGE} m_WR",
                              "points": [[float(m), float(samples[int(m)][0]), float(mu), float(sg)] for m, mu, sg in pts] if "table" not in cfg else None}}
        # --- the samples overlaid: each in its window, scaled so that its Gaussian peaks at 1 ----------------------
        ov = [m for m in cfg["overlay"] if m in samples]
        cmap = ListedColormap(plt.cm.viridis(np.linspace(0.0, 0.88, 256)))
        norm = Normalize(ov[0] / 1000.0, ov[-1] / 1000.0)
        fig, ax = P.scan_figure()
        n["overlay"] = []
        for mx in ov:
            mnx, mu, sg, path = samples[mx]
            edges, v, var_ = signal_hist(path, ch, topo)
            # bins of about 0.3 sigma (at least SIGNAL_REBIN native bins): the wide high-mass peaks are not drawn in
            # the narrow bins the low-mass ones need; each curve is scaled to its own fit, so the widths may differ
            k = max(SIGNAL_REBIN, int(round(0.3 * sg / C.NATIVE_BW)))
            nb = (len(v) // k) * k
            v2, e2 = v[:nb].reshape(-1, k).sum(1), np.sqrt(var_[:nb].reshape(-1, k).sum(1))
            ed = edges[:nb + 1:k]
            c2 = 0.5 * (ed[:-1] + ed[1:])
            gauss = lambda x, mu=mu, sg=sg: np.exp(-0.5 * ((x - mu) / sg) ** 2)             # noqa: E731
            core = (c2 >= mu - 2 * sg) & (c2 <= mu + 2 * sg) & (e2 > 0)
            amp = float((v2[core] * gauss(c2[core]) / e2[core] ** 2).sum() / (gauss(c2[core]) ** 2 / e2[core] ** 2).sum())
            c10 = 0.5 * (edges[:-1] + edges[1:])
            total = float(v.sum())
            n["overlay"].append({"mwr": mx, "mn": float(mnx), "x": float(mnx / mx), "mean": float(mu), "width": float(sg),
                                 "in_core": float(v[(c10 >= mu - 2 * sg) & (c10 <= mu + 2 * sg)].sum()) / total,
                                 "below": float(v[c10 < mu - 2 * sg].sum()) / total, "bin": float(k * C.NATIVE_BW),
                                 "n_eff": float(total ** 2 / var_.sum())})
            col = cmap(norm(mx / 1000.0))
            sel = (c2 >= 0.7 * mx) & (c2 <= 1.3 * mx)
            ax.stairs(v2[sel] / amp, np.append(ed[:-1][sel], ed[1:][sel][-1]), color=col, lw=1.3)
            xs_ = np.linspace(mu - 2 * sg, mu + 2 * sg, 120)
            ax.plot(xs_, gauss(xs_), color=col, lw=2.2)
        ax.set_xlim(C.THRESHOLD, 1.3 * ov[-1])
        ax.set_ylim(0.0, 1.3)
        ax.set_xlabel(P.MASS_LABEL)
        ax.set_ylabel("Events (core fit peak = 1)")
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02, aspect=30)
        cb.set_label(r"$m_{W_R}$ [TeV]")
        cb.set_ticks([m / 1000.0 for m in (ov[::2] + ([ov[-1]] if (len(ov) - 1) % 2 else []))])   # the last mass labelled too
        handles = [Line2D([], [], color="0.25", lw=1.3, label="Signal simulation"),
                   Line2D([], [], color="0.25", lw=2.2, label="Gaussian core fit")]
        cms_label(ax, None, cfg["com"], data=False, year=cfg["year"],
                  text=P.region_text(ch, f"{cfg['label']}\n$m_N \\approx {SIGNAL_X:g}\\,m_{{W_R}}$"))
        ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE_LEGEND, handlelength=1.8)
        fit_header(ax)
        P.save(fig, out / f"signal_overlay_{source}")
        logger.info("%s %s: mean = %.4f m %+.0f, width = %.4f m %+.0f (%d masses, %.0f-%.0f GeV)", ch, source,
                    lines["mean"][0], lines["mean"][1], lines["width"][0], lines["width"][1], len(ms), ms[0], ms[-1])
    if "table" not in cfg:
        (HERE / f"signal_map_{source}.json").write_text(json.dumps(out_map, indent=1))
        logger.info("wrote %s", HERE / f"signal_map_{source}.json")
    (C.RESULTS / f"deck_numbers_signal_{source}.json").write_text(json.dumps(N, indent=1))
    logger.info("wrote %s", C.RESULTS / f"deck_numbers_signal_{source}.json")


# ---------------------------------------------------------------------------------------------
# shapes: the power law, the power x exponential and the running power law on the same spectrum
# ---------------------------------------------------------------------------------------------
SHAPES = {"powlaw": ("power law", "#9c9ca1", "--"), "powexp": (r"power $\times$ exponential", "#f89c20", "-."),
          "atlasemu": ("running power law", "#e42536", "-")}


def run_shapes(args):
    """The three forms fitted to the summed raw MC from the floor, on a log y axis:
    24_deck/function_shapes, the MC
    with the three curves on the log axis and, below, how fast each curve falls there, in decades per TeV
    (-d log10 f / dm): a straight line on the log axis is a constant. Numbers: deck_numbers_shapes.json."""
    import plots as P
    from fits import fit_all
    from spectrum import centers, last_populated, load_components, rebin
    N = {}
    for ch in args.channels:
        out = C.RESULTS / ch / "24_deck"
        out.mkdir(parents=True, exist_ok=True)
        e10, v10, var10 = load_components(ch, VARIANT, "split9")["sum"]
        e, v, var = rebin(e10, v10, var10)
        c = centers(e)
        hi = last_populated(e, v)
        fits, _, _ = fit_all(e, v, var, FLOOR, hi, C.PIVOT, menu=list(SHAPES))
        sig = np.sqrt(np.maximum(var, 0.0))
        sel = (c >= C.THRESHOLD) & (c <= P.XMAX_FULL)
        fig, (ax, axr) = P.spectrum_figure()
        ax.errorbar(c[sel], v[sel], yerr=sig[sel], fmt="o", ms=4, color="k", label="Simulation")
        xs = np.linspace(FLOOR, P.XMAX_FULL, 721)
        n = N.setdefault(ch, {"floor": FLOOR, "hi": float(hi)})
        for name, (lab, col, ls) in SHAPES.items():
            f = fits.get(name)
            if f is None:
                logger.warning("%s: no %s fit", ch, name)
                continue
            y = np.array([f["tf1"].Eval(x) for x in xs])
            ok = y > 0
            ax.plot(xs[ok], np.maximum(y[ok], P.YMIN_FULL), color=col, lw=2.0, ls=ls, label=lab)
            fall = -np.gradient(np.log10(y[ok]), xs[ok]) * 1000.0            # decades per TeV
            axr.plot(xs[ok], fall, color=col, lw=2.0, ls=ls)
            at = lambda m: float(np.interp(m, xs[ok], fall))                  # noqa: E731
            n[name] = {"dev": float(f["dev"]), "params": [float(x) for x in f["params"]], "npar": len(f["params"]),
                       "fall": {f"{m:.0f}": at(m) for m in (1000.0, 2000.0, 3000.0, 4000.0, 6000.0)},
                       "events_above": {f"{m:.0f}": float(sum(f["tf1"].Eval(x) for x in c[c >= m])) for m in (3000.0, 4000.0)}}
        n["mc_above"] = {f"{m:.0f}": float(v[c >= m].sum()) for m in (3000.0, 4000.0)}
        P.finish(ax, axr, channel=ch, xmax=P.XMAX_FULL, top=float(v[sel].max()), ymin=P.YMIN_FULL,
                 extra=f"{C.DY_SHORT[VARIANT]}\nfits from {FLOOR:.0f} GeV", ratio_label="Fall [decades / TeV]",
                 ratio_lim=(0.0, 4.0), ratio_ref=0.0)
        axr.set_ylabel("Fall [decades / TeV]", fontsize=20)
        mc_first(ax, P)
        P.save(fig, out / "function_shapes")
        logger.info("%s: %s", ch, "; ".join(f"{k} dev {n[k]['dev']:.1f}, fall at 1/2/4 TeV {n[k]['fall']['1000']:.2f}/{n[k]['fall']['2000']:.2f}/{n[k]['fall']['4000']:.2f}"
                                           for k in SHAPES if k in n))
    (C.RESULTS / "deck_numbers_shapes.json").write_text(json.dumps(N, indent=1))
    logger.info("wrote %s", C.RESULTS / "deck_numbers_shapes.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["models", "grids", "signal", "shapes"])
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--candidates", nargs="+", default=["atlasemu", "powexp"])
    ap.add_argument("--tag", default="_100k", help="suffix of the 6_windows folders")
    ap.add_argument("--max-mass", type=float, default=3200.0, help="grids: the highest mass drawn (the window grids stop at 3200 GeV)")
    ap.add_argument("--main-max-mass", type=float, default=None,
                    help="grids: the highest mass of the raw-MC-only figure and summary (default: --max-mass)")
    ap.add_argument("--signal-source", choices=list(SIGNAL_SOURCES), default="2024", help="signal: the samples the Gaussian is taken from")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if not C.BOOSTED:
        raise SystemExit("run with FIT_TOPOLOGY=boosted")
    from wrplotter.cli_utils import setup_logging
    setup_logging(args.verbose)
    {"models": run_models, "grids": run_grids, "signal": run_signal, "shapes": run_shapes}[args.step](args)


if __name__ == "__main__":
    main()
