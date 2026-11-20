#!/usr/bin/env python3
"""The exponential of a quadratic, two uses.

The form  [0]*exp(-(([1]/1000.0)*(x-m0))-([2]/1000000)*(x-m0)*(x-m0))  is the registry's `expo2`,
exp(a + b u + c u^2) with u = (x - m0)/1000: N = e^a, [1] = -b, [2] = -c. The pivot m0 moves the
parameters, never the curve. `expo2n` is the form exactly as written, with m0 at the signal mean of
each S+B fit; `expo2` has m0 = 2000 GeV everywhere.

  fit     the whole summed background (nine processes, LO HT-binned DY) fitted with expo2 from 800, 1000
          and 1200 GeV to the last MC event, next to the exponential and the power x exponential:
          Pearson chi2 at MC precision, the Poisson deviance (what a dataset of this luminosity sees),
          where the fitted exponent turns upward, events per mass slice against the MC and the
          RooKeysPdf. Writes the fitted curves from 1200 GeV as toy truths,
          results*/<ch>/3_truths_fit/lo_ht/truth_fit{expo2,powexp} (the MC below 1200 GeV, where no
          window reaches), and the figures results*/<ch>/19_expo2/{fit_sum, slices}.
  matrix  reads the window grids 6_windows/k<k>_lo1200<tag>/lo_ht/toys_<truth>.csv (6_windows.py
          --candidates expo2 --append ...) and writes background_fit/EXPO2<tag>.md: expo2 next to the
          exponential and the power x exponential for every width, truth and dataset, the cost in
          sigma_fit, and expo2 against expo2n; figures results*/<ch>/19_expo2/pull_mean_vs_mass_*,
          width_vs_mass_*, sigma_fit_*.

  cd WR_Plotter && FIT_DATASET=2024 python background_fit/19_expo2.py fit
  cd WR_Plotter && python background_fit/19_expo2.py matrix [--tag _100k]
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))
import config as C                                                       # noqa: E402

logger = logging.getLogger("19_expo2")
VARIANT = "lo_ht"
COMPONENTS = "split9"
STARTS = [800.0, 1000.0, 1200.0]
TRUTH_LO = 1200.0                      # the floor of the window study: the fitted truths start here
FORMS = ["expo", "powexp", "expo2"]
FORM_LABEL = {"expo": "exponential", "powexp": r"power $\times$ exponential", "expo2": "exp. of a quadratic",
              "expo2n": "exp. of a quadratic, pivot at the signal"}
FORM_COLOR = {"expo": "#5790fc", "powexp": "#f89c20", "expo2": "#e42536", "expo2n": "#964a8b", "kde9": "#7a21dd"}
SLICES = [1200.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 5000.0, 6000.0, 8000.0]
WIDTHS = [(3, "+-3 sigma"), (5, "+-5 sigma"), (7, "+-7 sigma"), (10, "+-10 sigma"), (100, "full range")]
# Run 3 = the campaigns summed; the scaled 2024 MC is a cross-check and comes last
DATASETS = ["2024", "run3", "run3_scaled"]
DS_SHORT = {"2024": "2024", "run3_scaled": "2024 MC x 2.58 (cross-check)", "run3": "summed 2022-2025"}
TRUTHS = ["raw", "kde9", "fitpowexp", "fitexpo2"]
TRUTH_LABEL = {"raw": "the raw MC", "kde9": "the nine-process RooKeysPdf", "fitpowexp": "the power x exponential fitted to the MC",
               "fitexpo2": "the exponential of a quadratic fitted to the MC"}
# two short lines for the fitted truths: one long line widens the label block into the legend (fit_header then
# moves the legend under it and raises the axis)
TRUTH_FIG = {"raw": "Raw MC Bkg", "kde9": "RooKeysPdf Bkg", "fitpowexp": "Bkg: MC fitted with\n" + r"power $\times$ exponential",
             "fitexpo2": "Bkg: MC fitted with\nexp. of a quadratic"}
REF_MASSES = (2000.0, 3000.0, 3400.0)
FLOOR = 1200.0


def use_dataset(ds):
    """Point config at a dataset (the module reads FIT_DATASET on import)."""
    import os
    os.environ["FIT_DATASET"] = ds
    importlib.reload(C)
    for name in ("spectrum", "fits", "plots", "sbfit", "truth_sets"):
        if name in sys.modules:
            importlib.reload(sys.modules[name])


# ---------------------------------------------------------------------------------------------
# fit: the whole background
# ---------------------------------------------------------------------------------------------

def vertex(params):
    """Mass at which the expo2 exponent a + b u + c u^2 stops falling (None: it falls everywhere above)."""
    _, b, c = (float(x) for x in params)
    return C.PIVOT - 1000.0 * b / (2.0 * c) if c > 0 else None


def slice_sums(e10, v10, edges):
    c = 0.5 * (e10[:-1] + e10[1:])
    return np.array([v10[(c >= a) & (c < b)].sum() for a, b in zip(edges[:-1], edges[1:])])


def curve10(fit, e10, raw10, lo):
    """The fitted TF1 (counts per BIN_WIDTH at the bin centre) on the 10 GeV grid from lo up, the MC below."""
    c = 0.5 * (e10[:-1] + e10[1:])
    out = np.array(raw10, float)
    m = c >= lo
    out[m] = np.array([fit["tf1"].Eval(x) for x in c[m]]) * (C.NATIVE_BW / C.BIN_WIDTH)
    return out


def run_fit(args):
    import plots as P
    from fits import fit_all, pearson_at
    from spectrum import centers, last_populated, load_components, rebin
    from truth_sets import load_truth, save_truth
    from util import write_csv
    for ds in args.datasets:
        use_dataset(ds)
        import plots as P                                                # noqa: F811 - reloaded with the dataset
        from fits import fit_all, pearson_at                             # noqa: F811
        from spectrum import centers, last_populated, load_components, rebin   # noqa: F811
        from truth_sets import load_truth, save_truth                    # noqa: F811
        L = [f"# The exponential of a quadratic on the whole background, {DS_SHORT[ds]} ({C.LUMI:.1f} fb-1)", "",
             "Summed MC of the nine processes (LO HT-binned DY), 100 GeV bins, fit from the start to the last MC event. "
             "Pearson = chi2/ndf with the MC statistical errors (p-value); dev = the Poisson deviance of the "
             "likelihood fit to the MC yields: the -2 ln L a dataset of this luminosity would pay for the shape "
             "mismatch alone, on top of about one unit per bin of statistical scatter; turn = the mass above which "
             "the fitted curve rises again (Poisson fit).", ""]
        rows = []
        for ch in args.channels:
            out = C.RESULTS / ch / "19_expo2"
            out.mkdir(parents=True, exist_ok=True)
            comps = load_components(ch, VARIANT, COMPONENTS)
            e10, v10, var10 = comps["sum"]
            e, v, var = rebin(e10, v10, var10)
            c = centers(e)
            hi = last_populated(e, v)
            kpath = C.RESULTS / ch / "3_truths_kde9" / VARIANT / "truth_kde"
            kde10 = load_truth(kpath)[1]["sum"] if kpath.with_suffix(".npz").exists() else None
            L += [f"## {ch}", "", "| start [GeV] | function | Pearson chi2/ndf (p) | dev | turn [GeV] |", "|---|---|---|---|---|"]
            fits_by_lo = {}
            for lo in STARTS:
                Lf, Pf, w = fit_all(e, v, var, lo, hi, C.PIVOT, menu=FORMS)
                fits_by_lo[lo] = Lf
                for name in FORMS:
                    fl, fp = Lf.get(name), Pf.get(name)
                    if fl is None or fp is None:
                        L.append(f"| {lo:.0f} | {name} | no fit | | |")
                        continue
                    vt = vertex(fl["params"]) if name == "expo2" else None
                    rows.append({"dataset": ds, "channel": ch, "lo": lo, "hi": hi, "function": name, "P_chi2": fp["chi2"],
                                 "P_ndf": fp["ndf"], "P_p": fp["p"], "L_dev": fl["dev"], "turn": vt,
                                 **{f"p{i}": float(x) for i, x in enumerate(fl["params"])},
                                 **{f"e{i}": float(x) for i, x in enumerate(fl["perr"])}})
                    L.append(f"| {lo:.0f} | {name} | {fp['chi2']:.0f}/{fp['ndf']} ({fp['p']:.2g}) | {fl['dev']:.2f} | "
                             f"{'-' if vt is None else f'{vt:.0f}'} |")
            # events per slice, the fits from TRUTH_LO
            Lf = fits_by_lo[TRUTH_LO]
            edges = np.array(SLICES)
            mc, err = slice_sums(e10, v10, edges), np.sqrt(slice_sums(e10, var10, edges))
            curves = {}
            if kde10 is not None:
                curves["kde9"] = kde10
            for name in ("powexp", "expo2"):
                if Lf.get(name):
                    curves[name] = curve10(Lf[name], e10, v10, TRUTH_LO)
            L += ["", f"Events per slice, fits from {TRUTH_LO:.0f} GeV (difference to the MC in units of its error):", "",
                  "| slice [GeV] | MC | " + " | ".join("RooKeysPdf" if k == "kde9" else k for k in curves) + " |",
                  "|---|---|" + "---|" * len(curves)]
            sums = {k: slice_sums(e10, val, edges) for k, val in curves.items()}
            for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
                L.append(f"| {a:.0f}-{b:.0f} | {mc[i]:.3g} +- {err[i]:.2g} | "
                         + " | ".join(f"{sums[k][i]:.3g} ({(sums[k][i] - mc[i]) / err[i]:+.1f})" for k in curves) + " |")
            L.append("")
            # the fitted curves as toy truths
            for name in ("powexp", "expo2"):
                fl = Lf.get(name)
                if not fl:
                    continue
                tdir = C.RESULTS / ch / "3_truths_fit" / VARIANT
                save_truth(tdir / f"truth_fit{name}", e10, {"sum": np.maximum(curves[name], 0.0)},
                           {"sum": {"function": name, "lo": TRUTH_LO, "hi": hi, "pivot": C.PIVOT,
                                    "params": [float(x) for x in fl["params"]], "dev": fl["dev"],
                                    "below_lo": "the MC histogram"}})
            plot_fit(P, ch, e, v, var, Lf, kde10, e10, out / "fit_sum")
            plot_slices(P, ch, edges, mc, err, sums, out / "slices")
            logger.info("%s %s: wrote %s", ds, ch, out)
        (C.RESULTS / "EXPO2_BACKGROUND.md").write_text("\n".join(L) + "\n")
        write_csv(C.RESULTS / "expo2_background.csv", rows)
        logger.info("wrote %s", C.RESULTS / "EXPO2_BACKGROUND.md")


def mc_first(ax, prefix):
    """Legend with the MC entry first, then its band, then the curves (matplotlib lists lines before errorbars)."""
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND
    h, l = ax.get_legend_handles_labels()
    idx = sorted(range(len(l)), key=lambda i: 0 if l[i].startswith(prefix) else 1 if l[i].startswith("MC stat") else 2)
    ax.legend([h[i] for i in idx], [l[i] for i in idx], loc="upper right", fontsize=FONT_SIZE_LEGEND, handlelength=1.6)


def plot_fit(P, ch, e, v, var, Lf, kde10, e10, out):
    """The summed MC with the fitted curves from TRUTH_LO and the RooKeysPdf; below, (MC - curve) / sigma_MC."""
    from spectrum import centers, rebin
    c = centers(e)
    err = np.sqrt(np.maximum(var, 0.0))
    sel = (c >= C.THRESHOLD) & (c <= 8000.0)
    fig, (ax, axr) = P.spectrum_figure()
    ax.errorbar(c[sel], v[sel], yerr=err[sel], fmt="o", ms=4, color="k", label="Simulation")
    m = sel & (c >= TRUTH_LO)
    stair_edges = np.append(e[:-1][m], e[1:][m][-1])
    if kde10 is not None:
        _, k100, _ = rebin(e10, kde10, np.zeros_like(kde10))
        ax.plot(c[sel], np.maximum(k100[sel], P.YMIN), color=FORM_COLOR["kde9"], lw=1.6, ls=":", label="RooKeysPdf")
        axr.stairs(((v - k100) / np.where(err > 0, err, 1.0))[m], stair_edges, color=FORM_COLOR["kde9"], lw=1.2, ls=":")
    grid = np.linspace(TRUTH_LO, 8000.0, 500)
    for name in FORMS:
        f = Lf.get(name)
        if not f:
            continue
        ax.plot(grid, np.maximum([f["tf1"].Eval(x) for x in grid], P.YMIN), color=FORM_COLOR[name], lw=1.8,
                ls="--" if name == "expo" else "-", label=FORM_LABEL[name])
        pull = np.where(err > 0, (v - f["model"]) / np.where(err > 0, err, 1.0), np.nan)
        axr.stairs(pull[m], stair_edges, color=FORM_COLOR[name], lw=1.2, ls="--" if name == "expo" else "-")
    for y in (-2, 2):
        axr.axhline(y, color="0.5", lw=0.6, ls="--")
    P.finish(ax, axr, channel=ch, xmax=8000.0, top=float(v[sel].max()), ymin=1e-3,
             extra=f"{C.DY_SHORT[VARIANT]}\nfits from {TRUTH_LO:.0f} GeV", ratio_label=r"(MC $-$ fit) / $\sigma_{\rm MC}$",
             ratio_lim=(-5, 5), ratio_ref=0.0)
    axr.set_ylabel(r"(MC $-$ fit) / $\sigma_{\rm MC}$", fontsize=20)
    mc_first(ax, "Simulation")
    P.save(fig, out)


def plot_slices(P, ch, edges, mc, err, sums, out):
    """Events per mass slice: the MC (points) and each curve (steps); below, curve - MC in events per slice with
    the MC error band."""
    c, w = 0.5 * (edges[:-1] + edges[1:]), edges[1:] - edges[:-1]
    fig, (ax, axr) = P.spectrum_figure()
    ax.errorbar(c, mc, xerr=w / 2, yerr=err, fmt="o", ms=5, color="k", lw=1.4, label="MC, 9 processes")
    top = 0.0
    names = {"kde9": "RooKeysPdf", "powexp": FORM_LABEL["powexp"] + " fit", "expo2": FORM_LABEL["expo2"] + " fit"}
    styles = {"kde9": ":", "powexp": "-", "expo2": "-"}
    for name, cs in sums.items():
        ax.step(edges, np.append(cs, cs[-1]), where="post", color=FORM_COLOR[name], lw=1.8, ls=styles[name], label=names[name])
        d = cs - mc
        axr.stairs(d, edges, color=FORM_COLOR[name], lw=1.8, ls=styles[name])
        top = max(top, float(np.abs(d).max()))
    axr.fill_between(edges, np.append(-err, -err[-1]), np.append(err, err[-1]), step="post", color="0.5", alpha=0.3, lw=0)
    ax.fill_between([], [], [], color="0.5", alpha=0.3, lw=0, label="MC stat. error (lower panel)")
    lim = 1.3 * top
    P.finish(ax, axr, channel=ch, xmin=edges[0], xmax=edges[-1], top=float(mc.max()), ymin=1e-2, ylabel="Events in slice",
             extra=C.DY_SHORT[VARIANT], ratio_label=r"Curve $-$ MC [events]", ratio_lim=(-lim, lim), ratio_ref=0.0)
    axr.set_ylabel(r"Curve $-$ MC [events]", fontsize=20)
    mc_first(ax, "MC,")
    P.save(fig, out)


# ---------------------------------------------------------------------------------------------
# matrix: the window grids
# ---------------------------------------------------------------------------------------------

def run_matrix(args):
    T = importlib.import_module("4_toys")
    M = importlib.import_module("16_matrix")
    cands = args.candidates
    cells, missing = {}, []
    rows_of = {}
    for ds in args.datasets:
        for ch in args.channels:
            for k, _ in WIDTHS:
                for tr in TRUTHS:
                    rows = M.load(ds, ch, k, tr, args.tag)
                    for extra in args.extra_tags:                    # a candidate run into its own folders joins the cell
                        have = {r["candidate"] for r in rows}
                        rows = rows + [r for r in M.load(ds, ch, k, tr, extra) if r["candidate"] not in have]
                    rows_of[(ds, ch, k, tr)] = rows
                    for cand in cands + ["expo2n"]:
                        if any(r["candidate"] == cand for r in rows):
                            cells[(ds, ch, k, tr, cand)] = M.Cell([r for r in rows if r["candidate"] == cand], cand)
                        elif cand in cands:
                            missing.append(f"{DS_SHORT[ds]} {ch} {dict(WIDTHS)[k]} {tr} {cand}")
    cols = [(ds, ch) for ds in args.datasets for ch in args.channels]
    head = "| width | function | " + " | ".join(f"{DS_SHORT[ds]} {ch}" for ds, ch in cols) + " |"
    sep = "|---|---|" + "---|" * len(cols)
    ntoys = sorted({int(c.ntoys) for c in cells.values() if c.ntoys})
    L = [f"# The exponential of a quadratic in the S+B fit: the window-width matrix ({' / '.join(f'{n:,}' for n in ntoys)} toys per cell)", "",
         "expo2 = exp(a + b u + c u^2), u = (m - 2000 GeV)/1000: the family of "
         "[0]*exp(-(([1]/1000.0)*(x-m0))-([2]/1000000)*(x-m0)*(x-m0)). Grids 6_windows/k<k>_lo1200" + args.tag
         + ", LO HT-binned DY, floor 1200 GeV. Each cell: last mass of the float regime; cells with |mu| beyond 0.5 / beyond 0.3; "
         "largest |mu| (at mass). Float regime: masses above 1200 GeV with >= 90 % of the zero-injection fits passing the "
         "checks (valid minimum, full covariance, no parameter at a bound, background falling over the fit range), below the first sparse mass. "
         "mu = mean of the Gaussian fitted to the zero-injection pulls"
         + (f" (error {1 / math.sqrt(ntoys[0]):.3f} per cell)." if ntoys else "."), ""]
    if missing:
        L += [f"Missing: {len(missing)} cells ({', '.join(missing[:6])}{', ...' if len(missing) > 6 else ''}).", ""]
    for tr in TRUTHS:
        if not any((ds, ch, k, tr, cand) in cells for ds, ch in cols for k, _ in WIDTHS for cand in cands):
            continue
        L += [f"## Toys from {TRUTH_LABEL[tr]}", "", head, sep]
        for k, lab in WIDTHS:
            for cand in cands:
                L.append(f"| {lab} | {cand} | " + " | ".join(
                    cells[(ds, ch, k, tr, cand)].summary() if (ds, ch, k, tr, cand) in cells else "-" for ds, ch in cols) + " |")
        L += ["", "sigma_fit relative to the exponential in the same window, at "
              + " / ".join(f"{m_ / 1000:g}" for m_ in REF_MASSES) + " TeV:", "", head, sep]
        for k, lab in WIDTHS:
            for cand in [c_ for c_ in cands if c_ != "expo"]:
                row = []
                for ds, ch in cols:
                    c, ref = cells.get((ds, ch, k, tr, cand)), cells.get((ds, ch, k, tr, "expo"))
                    if c is None or ref is None:
                        row.append("-")
                        continue
                    row.append(" / ".join(f"{c.sfit[m_] / ref.sfit[m_]:.2f}" if m_ in c.sfit and m_ in ref.sfit and ref.sfit[m_] > 0
                                          else "-" for m_ in REF_MASSES))
                L.append(f"| {lab} | {cand} | " + " | ".join(row) + " |")
        L += ["", "Fraction of zero-injection fits passing the checks at 2 / 3 / 4 TeV:", "", head, sep]
        for k, lab in WIDTHS:
            for cand in cands:
                row = []
                for ds, ch in cols:
                    rows = rows_of.get((ds, ch, k, tr), [])
                    frac = dict(T.pass_fractions(rows, cand))
                    row.append(" / ".join(f"{frac[m_]:.2f}" if m_ in frac else "-" for m_ in (2000.0, 3000.0, 4000.0)))
                L.append(f"| {lab} | {cand} | " + " | ".join(row) + " |")
        L.append("")
    # the two parameter bases, toy by toy (each candidate ran alone on the cell's seed: the same zero-injection toys)
    L += ["## expo2 against expo2n (the form as written, pivot at the signal mean)", "",
          "The same zero-injection toys fitted in both bases. Each cell: the smallest share of toys with the same "
          "N_sig (|difference| < 0.01 sigma_fit) over the masses where both pass in >= 90 % of the toys; "
          "the two last float-regime masses (expo2 / expo2n); the mean passing fraction over 3400-4800 GeV (expo2 / expo2n).", "",
          "| truth | width | " + " | ".join(f"{DS_SHORT[ds]} {ch}" for ds, ch in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for tr in TRUTHS:
        for k, lab in WIDTHS:
            row, any_ = [], False
            for ds, ch in cols:
                a, b = cells.get((ds, ch, k, tr, "expo2")), cells.get((ds, ch, k, tr, "expo2n"))
                npz = HERE / C.DATASETS[ds]["results"] / ch / "6_windows" / f"k{k}_lo1200{args.tag}" / VARIANT / f"nsig_{tr}.npz"
                if a is None or b is None or not npz.exists():
                    row.append("-")
                    continue
                any_ = True
                z = np.load(npz)
                rows = rows_of[(ds, ch, k, tr)]
                fa, fb = dict(T.pass_fractions(rows, "expo2")), dict(T.pass_fractions(rows, "expo2n"))
                same = []
                for m_ in sorted(fa):
                    ka, kb = f"expo2_{int(m_)}_0", f"expo2n_{int(m_)}_0"
                    if m_ <= FLOOR or m_ not in fb or min(fa[m_], fb[m_]) < 0.9 or ka not in z.files or kb not in z.files:
                        continue
                    na, nb, ea = z[ka], z[kb], z[ka + "_err"]
                    if len(na) != len(nb):
                        continue
                    same.append(float(np.mean(np.abs(na - nb) < 0.01 * np.where(ea > 0, ea, np.inf))))
                rng = [m_ for m_ in fa if 3400.0 <= m_ <= 4800.0 and m_ in fb]
                pa = np.mean([fa[m_] for m_ in rng]) if rng else float("nan")
                pb = np.mean([fb[m_] for m_ in rng]) if rng else float("nan")
                la = "-" if a.last is None else f"{a.last:.0f}"
                lb = "-" if b.last is None else f"{b.last:.0f}"
                row.append((f"{min(same):.3f}" if same else "-") + f"; {la} / {lb}; {pa:.2f} / {pb:.2f}")
            if any_:
                L.append(f"| {tr} | {lab} | " + " | ".join(row) + " |")
    out = HERE / f"EXPO2{args.tag}.md"
    out.write_text("\n".join(L) + "\n")
    logger.info("wrote %s (%d cells)", out, len(cells))
    if args.no_plots:
        return
    for ds in args.datasets:
        use_dataset(ds)
        T = importlib.reload(importlib.import_module("4_toys"))
        T.CAND_COLOR.update(FORM_COLOR)
        T.TRUTH_BKG.update(TRUTH_FIG)
        for ch in args.channels:
            out = C.RESULTS / ch / "19_expo2"
            for k, _ in WIDTHS:
                tag = "\nfull range\nfrom 1200 GeV" if k >= 100 else f"\nwindow $\\pm{k:g}\\sigma$"
                for tr in TRUTHS:
                    rows = rows_of.get((ds, ch, k, tr), [])
                    have = [c_ for c_ in cands if any(r["candidate"] == c_ for r in rows)]
                    if len(have) < 2:
                        continue
                    plot_cands(T, ch, have, tr, rows, out / f"k{k}{args.tag}", tag)
            logger.info("%s %s: figures in %s", ds, ch, out)


def plot_cands(T, channel, cands, truth, rows, out, tag):
    """The candidates on one truth and one width: the Gaussian mu and sigma of the pulls and sigma_fit relative to
    the exponential against m_WR; filled markers where >= 90 % of the cell's fits pass, open ones where not."""
    import matplotlib.pyplot as plt
    import mplhep as hep
    import plots as P
    from matplotlib.lines import Line2D
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    hep.style.use("CMS")
    extra = f"{C.DY_SHORT[VARIANT]}\n{TRUTH_FIG[truth]}{tag}"           # no more lines: the block must stay above the 0.5 line
    ref = dict(zip(*T._series(rows, "expo", 0.0, "asimov_err")))
    for stat, key, ylabel, ylim in [("pull_mean", "pull_gauss_mean", r"Gaussian $\mu$ of the pulls", (-2.0, 3.0)),
                                    ("width", "pull_width", r"Gaussian $\sigma$ of the pulls", (0.0, 3.0)),
                                    ("sigma_fit", "asimov_err", r"$\sigma_{\rm fit}$ / $\sigma_{\rm fit}$(exponential)", (0.8, 2.6))]:
        fig, ax = plt.subplots(figsize=(10, 8))
        for cnd in cands:
            x, y = T._series(rows, cnd, 0.0, key)
            x, y = np.array(x), np.array(y, float)
            if stat == "sigma_fit":
                if cnd == "expo":
                    continue
                keep = np.array([m_ in ref and ref[m_] > 0 for m_ in x], bool)
                x, y = x[keep], np.array([yy / ref[m_] for m_, yy in zip(x[keep], y[keep])])
            frac = dict(T.pass_fractions(rows, cnd))
            good = np.array([frac.get(m_, 0.0) >= T.PASS_FRAC for m_ in x], bool)
            col = FORM_COLOR.get(cnd, "k")
            ax.plot(x, y, lw=1.5, color=col, label=FORM_LABEL.get(cnd, cnd))
            ax.plot(x[good], y[good], "o", ms=6, color=col)
            ax.plot(x[~good], y[~good], "o", ms=6, mfc="white", mec=col)
        if stat == "pull_mean":
            P.bias_lines(ax)
        else:
            ax.axhline(1.0, color="k", lw=0.8)
        ax.set_ylim(*ylim)
        ax.set_xlabel(P.MWR_LABEL)
        ax.set_ylabel(ylabel)
        cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(channel, extra), year=C.YEAR)
        handles, _ = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], marker="o", mfc="white", mec="k", color="none", ms=6, label=r"open: $< 90\,\%$ pass"))
        ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE_LEGEND)
        fit_header(ax, ignore_above=(stat != "pull_mean"))
        P.save(fig, out / f"{stat}_vs_mass_{truth}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fit", "matrix"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=sorted(C.DATASETS))
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--candidates", nargs="+", default=FORMS, help="matrix: the functions in the tables and figures")
    ap.add_argument("--tag", default="", help="matrix: the grids' folder suffix (6_windows.py --tag), e.g. _100k")
    ap.add_argument("--extra-tags", nargs="*", default=[],
                    help="matrix: more folder suffixes whose candidates join the cells, e.g. _100k_expo2 (expo2 alone at 100k toys)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    from wrplotter.cli_utils import setup_logging
    setup_logging(args.verbose)
    (run_fit if args.step == "fit" else run_matrix)(args)


if __name__ == "__main__":
    main()
