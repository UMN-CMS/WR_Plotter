#!/usr/bin/env python3
"""The boosted SR (m_lJ): the function, the fit floor and the window width, on the same chain as the resolved study.

Run with FIT_TOPOLOGY=boosted (results_boosted*/, 50 GeV bins, the boosted window map). Two steps:

  fit     every form of FORMS fitted to the summed nine-process MC from 800, 1000 and 1200 GeV (Poisson deviance =
          what a dataset of this luminosity sees of the shape mismatch); events per mass slice for the MC, the
          smoothed backgrounds and the fits from 800 GeV; the fitted curves of TRUTH_FORMS saved as toy backgrounds
          (3_truths_fit/lo_ht/truth_fit<form>); figures 21_boosted/{fit_sum,slices}; BOOSTED_BACKGROUND.md.
  matrix  the window grids 6_windows/k<k>[_lo<lo>]<tag>/ of every width, floor, background and function on one page
          (BOOSTED_MATRIX<tag>.md): the last mass at which >= 90 % of the fits pass the checks, the largest |mu| of
          the pulls up to there, the cells beyond 0.5, sigma_fit against the full range; figures
          21_boosted/<form>_lo<lo>_<background>_{pull_mean,pass_fraction} (one line per width).

  cd WR_Plotter && FIT_TOPOLOGY=boosted FIT_DATASET=run3 python background_fit/21_boosted.py fit|matrix [--tag _100k]
"""
from __future__ import annotations

import argparse
import csv
import importlib
import logging
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))
import config as C                                                       # noqa: E402

logger = logging.getLogger("21_boosted")
VARIANT = "lo_ht"
COMPONENTS = "split9"
STARTS = [800.0, 1000.0, 1200.0]
TRUTH_LO = 800.0                        # the fitted backgrounds start at the selection threshold
FORMS = ["expo", "powlaw", "powexp", "atlasemu", "wprime7", "expo2", "dilep8"]
TRUTH_FORMS = ["powexp", "atlasemu"]    # two families, so each function is also tested on the other's curve
FORM_LABEL = {"expo": "exponential", "powlaw": "power law", "powexp": r"power $\times$ exponential",
              "atlasemu": "running power law", "wprime7": "W' function", "expo2": "exp. of a quadratic",
              "dilep8": r"power $\times$ exp(quadratic)"}
FORM_COLOR = {"expo": "#5790fc", "powlaw": "#9c9ca1", "powexp": "#f89c20", "atlasemu": "#e42536", "wprime7": "#964a8b",
              "expo2": "#92dadd", "dilep8": "#717581", "kde9": "#7a21dd", "kde_expo9": "#3f90da"}
SLICES = [800.0, 900.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0, 2400.0, 2800.0, 3200.0, 4000.0, 8000.0]
XMAX = 5000.0
# the toy grids
WIDTHS = [3, 5, 7, 10, 100]
WIDTH_LABEL = {3: r"$\pm 3\sigma$", 5: r"$\pm 5\sigma$", 7: r"$\pm 7\sigma$", 10: r"$\pm 10\sigma$", 100: "full range"}
WIDTH_COLOR = {3: "#fde725", 5: "#5ec962", 7: "#21918c", 10: "#3b528b", 100: "#000000"}
FLOORS = [800.0, 1000.0, 1200.0]
TRUTHS = ["raw", "kde9", "kde_expo9", "fitpowexp", "fitatlasemu"]
TRUTH_LABEL = {"raw": "Raw MC Bkg", "kde9": "RooKeysPdf Bkg", "kde_expo9": "RooKeysPdf +\nexpo tail Bkg",
               "fitpowexp": "Bkg: MC fitted with\n" + r"power $\times$ exponential",
               "fitatlasemu": "Bkg: MC fitted with\nrunning power law"}
TRUTH_SHORT = {"raw": "raw MC", "kde9": "RooKeysPdf", "kde_expo9": "RooKeysPdf + expo tail",
               "fitpowexp": "powexp fit", "fitatlasemu": "running power law fit"}
SAME_FAMILY = {("powexp", "fitpowexp"), ("atlasemu", "fitatlasemu")}     # circular cells: a closure, not a test
PASS = 0.9


def kdir(k, lo):
    return f"k{k:g}" + ("" if lo == C.FIT_LO else f"_lo{lo:.0f}")


# ---------------------------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------------------------

def run_fit(args):
    import plots as P
    from fits import fit_all
    from spectrum import centers, last_populated, load_components, rebin
    from truth_sets import load_truth, save_truth
    E2 = importlib.import_module("19_expo2")
    L = [f"# The boosted SR background, {C.ERA} ({C.LUMI:.1f} fb-1)", "",
         f"Summed MC of the nine processes (LO HT-binned DY), {C.BIN_WIDTH:.0f} GeV bins, fit from the start to the last "
         "MC event. dev = the Poisson deviance of the likelihood fit to the MC yields: the -2 ln L a dataset of this "
         "luminosity pays for the shape mismatch (the raw MC's own spikes above 2.5 TeV put a floor of about 4.3 under "
         "every form); `!` = a failed fit check (minimum, covariance, parameter at a limit, monotonic).", ""]
    for ch in args.channels:
        out = C.RESULTS / ch / "21_boosted"
        out.mkdir(parents=True, exist_ok=True)
        comps = load_components(ch, VARIANT, COMPONENTS)
        e10, v10, var10 = comps["sum"]
        e, v, var = rebin(e10, v10, var10)
        hi = last_populated(e, v)
        L += [f"## {ch}", "", "| start [GeV] | " + " | ".join(FORMS) + " |", "|---|" + "---|" * len(FORMS)]
        fits_by_lo = {}
        for lo in STARTS:
            Lf, _, _ = fit_all(e, v, var, lo, hi, C.PIVOT, menu=FORMS)
            fits_by_lo[lo] = Lf
            L.append(f"| {lo:.0f} | " + " | ".join(
                "-" if Lf.get(n) is None else f"{Lf[n]['dev']:.2f}{'' if Lf[n]['passed'] else '!'}" for n in FORMS) + " |")
        Lf = fits_by_lo[TRUTH_LO]
        edges = np.array(SLICES)
        mc, err = E2.slice_sums(e10, v10, edges), np.sqrt(E2.slice_sums(e10, var10, edges))
        curves = {}
        for key, sub, name in (("kde9", "3_truths_kde9", "truth_kde"), ("kde_expo9", "3_truths_split9", "truth_kde_expo")):
            p = C.RESULTS / ch / sub / VARIANT / name
            if p.with_suffix(".npz").exists():
                curves[key] = load_truth(p)[1]["sum"]
        for name in ("expo", "powexp", "atlasemu"):
            if Lf.get(name):
                curves[name] = E2.curve10(Lf[name], e10, v10, TRUTH_LO)
        head = {"kde9": "RooKeysPdf", "kde_expo9": "RooKeysPdf + expo tail", "expo": "expo fit", "powexp": "powexp fit",
                "atlasemu": "running power law fit"}
        L += ["", f"Events per slice, fits from {TRUTH_LO:.0f} GeV (difference to the MC in units of its error):", "",
              "| slice [GeV] | MC | " + " | ".join(head[k] for k in curves) + " |", "|---|---|" + "---|" * len(curves)]
        sums = {k: E2.slice_sums(e10, val, edges) for k, val in curves.items()}
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            L.append(f"| {a:.0f}-{b:.0f} | {mc[i]:.3g} +- {err[i]:.2g} | "
                     + " | ".join(f"{sums[k][i]:.3g} ({(sums[k][i] - mc[i]) / err[i]:+.1f})" for k in curves) + " |")
        L.append("")
        for name in TRUTH_FORMS:
            fl = Lf.get(name)
            if fl:
                save_truth(C.RESULTS / ch / "3_truths_fit" / VARIANT / f"truth_fit{name}", e10,
                           {"sum": np.maximum(curves[name], 0.0)},
                           {"sum": {"function": name, "lo": TRUTH_LO, "hi": hi, "pivot": C.PIVOT,
                                    "params": [float(x) for x in fl["params"]], "dev": fl["dev"]}})
        # the spectrum with the fits from TRUTH_LO; below, (MC - fit) / sigma_MC
        c = centers(e)
        sig = np.sqrt(np.maximum(var, 0.0))
        sel = (c >= C.THRESHOLD) & (c <= XMAX)
        fig, (ax, axr) = P.spectrum_figure()
        ax.errorbar(c[sel], v[sel], yerr=sig[sel], fmt="o", ms=4, color="k", label="Simulation")
        grid = np.linspace(TRUTH_LO, XMAX, 500)
        stair = np.append(e[:-1][sel], e[1:][sel][-1])
        for name in ("expo", "powexp", "atlasemu"):
            f = Lf.get(name)
            if not f:
                continue
            ls = "--" if name == "expo" else "-"
            ax.plot(grid, np.maximum([f["tf1"].Eval(x) for x in grid], P.YMIN), color=FORM_COLOR[name], lw=1.8, ls=ls,
                    label=FORM_LABEL[name])
            pull = np.where(sig > 0, (v - f["model"]) / np.where(sig > 0, sig, 1.0), np.nan)
            axr.stairs(pull[sel], stair, color=FORM_COLOR[name], lw=1.2, ls=ls)
        for y in (-2, 2):
            axr.axhline(y, color="0.5", lw=0.6, ls="--")
        P.finish(ax, axr, channel=ch, xmax=XMAX, top=float(v[sel].max()), ymin=1e-3,
                 extra=f"{C.DY_SHORT[VARIANT]}\nfits from {TRUTH_LO:.0f} GeV", ratio_label=r"(MC $-$ fit) / $\sigma_{\rm MC}$",
                 ratio_lim=(-5, 5), ratio_ref=0.0)
        axr.set_ylabel(r"(MC $-$ fit) / $\sigma_{\rm MC}$", fontsize=20)
        E2.mc_first(ax, "Simulation")
        P.save(fig, out / "fit_sum")
        # events per slice: the MC, the smoothed backgrounds, the fits
        cs_, w = 0.5 * (edges[:-1] + edges[1:]), edges[1:] - edges[:-1]
        fig, (ax, axr) = P.spectrum_figure()
        ax.errorbar(cs_, mc, xerr=w / 2, yerr=err, fmt="o", ms=5, color="k", lw=1.4, label="MC, 9 processes")
        top = 0.0
        for name in ("kde9", "kde_expo9", "powexp", "atlasemu"):
            if name not in sums:
                continue
            ls = ":" if name.startswith("kde") else "-"
            lab = head[name]
            ax.step(edges, np.append(sums[name], sums[name][-1]), where="post", color=FORM_COLOR[name], lw=1.8, ls=ls, label=lab)
            d = sums[name] - mc
            axr.stairs(d, edges, color=FORM_COLOR[name], lw=1.8, ls=ls)
            top = max(top, float(np.abs(d).max()))
        axr.fill_between(edges, np.append(-err, -err[-1]), np.append(err, err[-1]), step="post", color="0.5", alpha=0.3, lw=0)
        ax.fill_between([], [], [], color="0.5", alpha=0.3, lw=0, label="MC stat. error (lower panel)")
        lim = 1.3 * max(top, float(err.max()))
        P.finish(ax, axr, channel=ch, xmin=edges[0], xmax=XMAX, top=float(mc.max()), ymin=1e-2, ylabel="Events in slice",
                 extra=C.DY_SHORT[VARIANT], ratio_label=r"Curve $-$ MC [events]", ratio_lim=(-lim, lim), ratio_ref=0.0)
        axr.set_ylabel(r"Curve $-$ MC [events]", fontsize=20)
        E2.mc_first(ax, "MC,")
        P.save(fig, out / "slices")
        logger.info("%s: wrote %s", ch, out)
    (C.RESULTS / "BOOSTED_BACKGROUND.md").write_text("\n".join(L) + "\n")
    logger.info("wrote %s", C.RESULTS / "BOOSTED_BACKGROUND.md")


# ---------------------------------------------------------------------------------------------
# matrix
# ---------------------------------------------------------------------------------------------

def load_cells(ch, k, lo, truth, tag):
    """{(candidate, mass): row} of the zero-injection cells of one grid, or None when the grid was not run."""
    p = C.RESULTS / ch / "6_windows" / (kdir(k, lo) + tag) / VARIANT / f"toys_{truth}.csv"
    if not p.exists():
        return None
    out = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok" or float(r["inject"]) != 0.0:
                continue
            g = lambda key: float(r[key]) if r.get(key) not in ("", "None", None) else float("nan")   # noqa: E731
            nconv = g("n_conv")
            out[(r["candidate"], float(r["mass"]))] = {
                "mu": g("pull_gauss_mean"), "sigma": g("pull_gauss_sigma"), "pass": g("n_passed") / max(nconv, 1.0),
                "sfit": g("asimov_err"), "lo": g("lo"), "hi": g("hi"), "b_core": g("b_core"), "sig_mu": g("mu"),
                "sig_sigma": g("sigma"), "asimov": g("asimov_pull"), "mean": g("nsig_mean")}
    return out


def summarize(cells, cand):
    """Of one grid and function: the masses in order, the passing range (the cells up to the last one with >= 90 %
    of the fits passing, a lower cell failing on its own left out), max |mu| there and the cells beyond 0.5 / 0.3."""
    ms = sorted(m for (c, m) in cells if c == cand)
    if not ms:
        return None
    good = [m for m in ms if cells[(cand, m)]["pass"] >= PASS and np.isfinite(cells[(cand, m)]["mu"])]
    if not good:
        return {"masses": ms, "first": None, "last": None, "good": [], "max": None, "gt05": [], "gt03": []}
    worst = max(good, key=lambda m: abs(cells[(cand, m)]["mu"]))
    return {"masses": ms, "first": good[0], "last": good[-1], "good": good,
            "max": (abs(cells[(cand, worst)]["mu"]), worst, cells[(cand, worst)]["mu"]),
            "gt05": [m for m in good if abs(cells[(cand, m)]["mu"]) > C.BIAS_PASS],
            "gt03": [m for m in good if abs(cells[(cand, m)]["mu"]) > C.BIAS_TARGET],
            "holes": [m for m in ms if good[0] < m < good[-1] and m not in good]}


def run_matrix(args):
    import plots as P
    from matplotlib.lines import Line2D
    tag = args.tag
    grids = {}
    for ch in args.channels:
        for lo in FLOORS:
            for k in WIDTHS:
                for tr in TRUTHS:
                    cells = load_cells(ch, k, lo, tr, tag)
                    if cells:
                        grids[(ch, lo, k, tr)] = cells
    if not grids:
        raise SystemExit(f"no 6_windows grids with tag {tag!r} under {C.RESULTS}")
    cands = [c for c in args.candidates if any(c == cc for g in grids.values() for (cc, _) in g)]
    L = [f"# The boosted SR: window width, fit floor, function and background, {C.ERA} ({C.LUMI:.1f} fb-1)", "",
         f"Toys of 6_windows/k<k>[_lo<lo>]{tag}/ ({C.BIN_WIDTH:.0f} GeV bins, Gaussian signal from the boosted window map, "
         "zero injection). Per cell: **last** = the highest mass at which >= 90 % of the fits pass the checks (valid, "
         "full covariance, no parameter at a bound, falling background); **max** = the largest |mu| of the Gaussian "
         "fitted to the pulls among the passing masses (the mass in brackets); **> 0.5** = the passing masses beyond "
         "the CMS line. A function on the curve of its own family is a closure and is marked (c).", ""]
    for cand in cands:
        L += [f"## {cand} ({FORM_LABEL.get(cand, cand)})", ""]
        for ch in args.channels:
            for lo in FLOORS:
                have = [(k, tr) for k in WIDTHS for tr in TRUTHS if (ch, lo, k, tr) in grids]
                if not have:
                    continue
                trs = [tr for tr in TRUTHS if any(t == tr for _, t in have)]
                L += [f"### {ch}, floor {lo:.0f} GeV", "", "| width | " + " | ".join(TRUTH_SHORT[t] for t in trs) + " |",
                      "|---|" + "---|" * len(trs)]
                for k in WIDTHS:
                    cellsL = []
                    for tr in trs:
                        g = grids.get((ch, lo, k, tr))
                        s = summarize(g, cand) if g else None
                        if s is None:
                            cellsL.append("-")
                        elif s["last"] is None:
                            cellsL.append("no mass passes")
                        else:
                            txt = (f"last {s['last']:.0f}; max {s['max'][2]:+.2f} ({s['max'][1]:.0f}); > 0.5: "
                                   f"{', '.join(f'{m:.0f}' for m in s['gt05']) or 'none'}")
                            if s["first"] > s["masses"][0] or s.get("holes"):
                                fails = [m for m in s["masses"] if m < s["last"] and m not in s["good"]]
                                txt += f"; fails below: {', '.join(f'{m:.0f}' for m in fails)}"
                            cellsL.append(txt + (" (c)" if (cand, tr) in SAME_FAMILY else ""))
                    L.append(f"| {WIDTH_LABEL[k].replace('$', '').replace(chr(92) + 'pm', '+-').replace(chr(92) + 'sigma', ' sigma')} | "
                             + " | ".join(cellsL) + " |")
                L.append("")
        # sigma_fit of each width over the full range's, same floor, on the RooKeysPdf + expo tail background
        L += [f"sigma_fit of the window over the full range's (same floor, RooKeysPdf + expo tail background, {cand}):", "",
              "| channel | floor | width | " + " | ".join(f"{m:.0f}" for m in args.ref_masses) + " |",
              "|---|---|---|" + "---|" * len(args.ref_masses)]
        for ch in args.channels:
            for lo in FLOORS:
                full = grids.get((ch, lo, 100, "kde_expo9"))
                if not full:
                    continue
                for k in WIDTHS[:-1]:
                    g = grids.get((ch, lo, k, "kde_expo9"))
                    if not g:
                        continue
                    L.append(f"| {ch} | {lo:.0f} | +-{k} sigma | " + " | ".join(
                        f"{g[(cand, m)]['sfit'] / full[(cand, m)]['sfit']:.2f}" if (cand, m) in g and (cand, m) in full
                        and full[(cand, m)]["sfit"] > 0 else "-" for m in args.ref_masses) + " |")
        L.append("")
        # the full range at every mass, the sparse regime included: the S+B fit of the background itself (no
        # fluctuation), N_sig / sigma_fit, and the mean fitted N_sig of the toys in events
        L += [f"The full range without fluctuations ({cand}): the largest |N_sig / sigma_fit| of the S+B fit to the "
              "background itself over all masses (its mass), then the mean fitted N_sig of the toys in events at "
              "3 / 4 / 5.2 TeV:", "", "| channel | floor | " + " | ".join(TRUTH_SHORT[t] for t in TRUTHS) + " |",
              "|---|---|" + "---|" * len(TRUTHS)]
        for ch in args.channels:
            for lo in FLOORS:
                cellsL = []
                for tr in TRUTHS:
                    g = grids.get((ch, lo, 100, tr))
                    ms = sorted(m for (c, m) in g if c == cand) if g else []
                    ms = [m for m in ms if np.isfinite(g[(cand, m)]["asimov"])]
                    if not ms:
                        cellsL.append("-")
                        continue
                    w = max(ms, key=lambda m: abs(g[(cand, m)]["asimov"]))
                    ev = " / ".join(f"{g[(cand, m)]['mean']:+.2f}" if (cand, m) in g else "-" for m in (3000.0, 4000.0, 5200.0))
                    cellsL.append(f"{g[(cand, w)]['asimov']:+.2f} ({w:.0f}); {ev}" + (" (c)" if (cand, tr) in SAME_FAMILY else ""))
                if any(c != "-" for c in cellsL):
                    L.append(f"| {ch} | {lo:.0f} | " + " | ".join(cellsL) + " |")
        L.append("")
        # figures: one line per width, per (channel, floor, background)
        for ch in args.channels:
            out = C.RESULTS / ch / "21_boosted"
            out.mkdir(parents=True, exist_ok=True)
            for lo in FLOORS:
                for tr in TRUTHS:
                    ks = [k for k in WIDTHS if (ch, lo, k, tr) in grids
                          and any(c == cand for (c, _) in grids[(ch, lo, k, tr)])]
                    if not ks:
                        continue
                    for quantity, ylabel, ylim in (("mu", r"Pull mean $\mu$ (no injection)", (-2.0, 2.0)),
                                                   ("pass", "Fraction of fits passing the checks", (0.0, 1.45))):
                        fig, ax = P.scan_figure()
                        for k in ks:
                            g = grids[(ch, lo, k, tr)]
                            ms = sorted(m for (c, m) in g if c == cand)
                            ys = np.array([g[(cand, m)][quantity] for m in ms])
                            ok = np.array([g[(cand, m)]["pass"] >= PASS for m in ms])
                            if quantity == "mu":
                                ys = np.clip(ys, ylim[0] + 0.02, ylim[1] - 0.02)
                            ax.plot(ms, ys, color=WIDTH_COLOR[k], lw=1.6)
                            ax.plot(np.array(ms)[ok], ys[ok], "o", ms=6, color=WIDTH_COLOR[k])
                            ax.plot(np.array(ms)[~ok], ys[~ok], "o", ms=6, mfc="white", color=WIDTH_COLOR[k])
                        if quantity == "mu":
                            P.bias_lines(ax)
                        else:
                            ax.axhline(PASS, color="0.3", lw=0.9, ls="--")
                        handles = [Line2D([], [], color=WIDTH_COLOR[k], lw=1.8, marker="o", label=WIDTH_LABEL[k]) for k in ks]
                        handles.append(Line2D([], [], color="k", lw=0, marker="o", mfc="white", label="open: < 90 % pass"))
                        P.finish_scan(ax, channel=ch, extra=f"{FORM_LABEL.get(cand, cand)}\n{TRUTH_LABEL[tr]}\nfrom {lo:.0f} GeV",
                                      ylabel=ylabel, ylim=ylim, handles=handles, ncol=1)
                        P.save(fig, out / f"{cand}_lo{lo:.0f}_{tr}_{'pull_mean' if quantity == 'mu' else 'pass_fraction'}{tag}")
    (C.RESULTS / f"BOOSTED_MATRIX{tag}.md").write_text("\n".join(L) + "\n")
    logger.info("wrote %s", C.RESULTS / f"BOOSTED_MATRIX{tag}.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fit", "matrix"])
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--candidates", nargs="+", default=["expo", "powlaw", "powexp", "atlasemu", "expo2"])
    ap.add_argument("--tag", default="", help="suffix of the 6_windows folders (e.g. _100k)")
    ap.add_argument("--ref-masses", nargs="+", type=float, default=[1400.0, 2000.0, 2400.0, 3000.0])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if not C.BOOSTED:
        raise SystemExit("21_boosted.py is the boosted study: run it with FIT_TOPOLOGY=boosted")
    from wrplotter.cli_utils import setup_logging
    setup_logging(args.verbose)
    (run_fit if args.step == "fit" else run_matrix)(args)


if __name__ == "__main__":
    main()
