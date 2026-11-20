#!/usr/bin/env python3
"""Does more luminosity, and does the smoothed truth, keep the window's fits passing to higher masses?

The fraction of the +-5 sigma window's zero-injection S+B fits that pass the checks (a valid fit with a
full covariance matrix, no parameter at a bound, a falling background; lib/sbfit.py), against m_WR. Every
fit of these grids converges, so this fraction is the "convergence" behind the grid's 90 % rule
(4_toys.pass_fractions, passed / converged). Two datasets on one axis, from their 6_windows/k5_lo1200 toys
(500 per cell, lo_ht, the 1200 GeV floor; --toys-tag _100k reads the 100k-toy grids k5_lo1200_100k and suffixes
the outputs): 109.7 fb-1 (2024, results/) and the Run 3 luminosity as the
2024 MC x 2.58 (results_run3scaled/, the study deck's stand-in; --datasets adds the summed 2022-2025 MC,
results_run3/), each on the raw MC and on the nine-process RooKeysPdf. One function per figure, colour =
dataset, line style and marker = truth (raw solid / circles, RooKeysPdf dashed / squares), filled where the
cell passes the 90 % rule, open where not.

Figures lumi_pass_fraction_<cand>.pdf, and in the same layout lumi_pull_mean_<cand>.pdf (the Gaussian mu of the
pulls at zero injection against m_WR, the +-0.3 / +-0.5 lines) and lumi_width_<cand>.pdf (its sigma), plus
summary.md (per function, dataset and truth: the last mass passing the 90 % rule, the first failing above the
floor, the pass fraction at 3.4 / 4 / 4.4 TeV, the gain in the last passing mass from the luminosity and from
the smoothing; the masses with |mu| > 0.5 among passing cells, the largest |mu| up to 3.2 TeV, mu at a few
masses) under <out-root>/<ch>/14_luminosity/, out-root = results_run3scaled by default.

  cd WR_Plotter && python background_fit/14_luminosity.py
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))
import config as C                                                       # noqa: E402
T = importlib.import_module("4_toys")                                    # noqa: E402
from util import read_csv                                                # noqa: E402

logger = logging.getLogger("14_luminosity")
PASS_FRAC = T.PASS_FRAC
FOLDER = "k5_lo1200"                 # the +-5 sigma window at the 1200 GeV floor (6_windows.py kdir)
VARIANT = "lo_ht"
TRUTHS = ["raw", "kde9"]
CANDS = ["expo", "powexp"]
CAND_LABEL = {"expo": "Expo", "powexp": r"Power $\times$ expo"}
COLOUR = {"2024": "#5790fc", "run3_scaled": "#e42536", "run3": "#964a8b"}   # colour = dataset
REF_MASSES = (3400.0, 4000.0, 4400.0)          # pass fractions quoted here
PULL_MASSES = (1600.0, 1800.0, 2000.0, 3400.0)  # pull means quoted here (the exponential dips at 1600-1800)


def ds_label(key):
    ds = C.DATASETS[key]
    return rf"{ds['lumi']:.1f} $\mathrm{{fb^{{-1}}}}$ ({ds['year']})"


def load(key, channel, truth, folder=FOLDER):
    p = HERE / C.DATASETS[key]["results"] / channel / "6_windows" / folder / VARIANT / f"toys_{truth}.csv"
    if not p.exists():
        logger.warning("no toys: %s", p)
        return []
    return [T._type_row(r) for r in read_csv(p)]


def pass_series(rows, cand):
    """(masses, pass fractions, >= 90 %) of the candidate's zero-injection cells, in mass order."""
    pts = T.pass_fractions(rows, cand)
    x = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    return x, y, y >= PASS_FRAC


def last_passing(x, good, floor):
    """The highest mass above the floor at which the cell passes the 90 % rule (the floor's own cell is a
    window clipped at the floor and does not count, as in 4_toys.first_sparse)."""
    ok = [m_ for m_, g in zip(x, good) if g and m_ > floor]
    return ok[-1] if ok else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--datasets", nargs="+", default=["2024", "run3_scaled"], choices=sorted(C.DATASETS))
    ap.add_argument("--cands", nargs="+", default=CANDS)
    ap.add_argument("--out-root", default=C.DATASETS["run3_scaled"]["results"],
                    help="results folder (under background_fit/) that takes <ch>/14_luminosity/")
    ap.add_argument("--toys-tag", default="", help="folder suffix of a high-statistics rerun (6_windows.py --tag _100k); "
                    "also suffixes the figures and summary<tag>.md")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    tag = args.toys_tag
    folder = FOLDER + tag
    from wrplotter.cli_utils import setup_logging
    setup_logging(args.verbose)
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.lines import Line2D
    import plots as P
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    hep.style.use("CMS")
    floor = float(FOLDER.split("_lo")[1])
    for channel in args.channels:
        out = HERE / args.out_root / channel / "14_luminosity"
        out.mkdir(parents=True, exist_ok=True)
        data = {(key, tr): load(key, channel, tr, folder) for key in args.datasets for tr in TRUTHS}
        ntoys = sorted({int(r["n_toys"]) for rows in data.values() for r in rows[:1] if r.get("n_toys")})
        per_cell = " / ".join(f"{n:,}" for n in ntoys) + " per cell" if ntoys else "toy count in the CSVs"
        sets = ", ".join(f"{C.DATASETS[k]['results']} ({C.DATASETS[k]['lumi']:.1f} fb-1)" for k in args.datasets)
        L = [f"# {channel}: the fraction of fits passing the checks against the luminosity and the truth", "",
             f"Toys 6_windows/{folder} ({VARIANT}, {per_cell}, the +-5 sigma window from {floor:.0f} GeV) of {sets}; "
             f"pass fraction = passed / converged of the zero-injection fits (every fit converges); the 90 % line "
             f"and the last passing mass as in 4_toys (the {floor:.0f} GeV cell, a window clipped at the floor, does not count).", ""]
        for cand in args.cands:
            L += [f"## {cand}", "",
                  "| dataset | truth | last mass passing 90 % | first failing above the floor | "
                  f"pass fraction at {' / '.join(f'{m_ / 1000:g}' for m_ in REF_MASSES)} TeV |",
                  "|---|---|---|---|---|"]
            last = {}
            fig, ax = plt.subplots(figsize=(10, 8))
            for key in args.datasets:
                for tr in TRUTHS:
                    rows = data[(key, tr)]
                    if not rows:
                        continue
                    x, y, good = pass_series(rows, cand)
                    ls, mk = T.TRUTH_LM.get(tr, ("-", "o"))
                    ax.plot(x, y, lw=1.5, color=COLOUR[key], ls=ls)
                    ax.plot(x[good], y[good], mk, ms=6, color=COLOUR[key])
                    ax.plot(x[~good], y[~good], mk, ms=6, mfc="white", mec=COLOUR[key])
                    lp = last_passing(x, good, floor)
                    last[(key, tr)] = lp
                    first_fail = T.first_sparse(rows, cand, PASS_FRAC)
                    at = dict(zip(x, y))
                    cell = lambda v: "-" if v is None else f"{v:.0f}"
                    L.append(f"| {C.DATASETS[key]['results']} ({C.DATASETS[key]['lumi']:.1f} fb-1) | {tr} | "
                             f"{cell(lp)} | {cell(first_fail)} | "
                             + " / ".join(f"{at[m_]:.2f}" if m_ in at else "-" for m_ in REF_MASSES) + " |")
            # the gains in the last passing mass: luminosity (per truth) and smoothing (per dataset)
            L.append("")
            if len(args.datasets) >= 2:
                a, b = args.datasets[0], args.datasets[1]
                for tr in TRUTHS:
                    if last.get((a, tr)) is not None and last.get((b, tr)) is not None:
                        L.append(f"- luminosity, {tr}: {C.DATASETS[a]['results']} {last[(a, tr)]:.0f} -> "
                                 f"{C.DATASETS[b]['results']} {last[(b, tr)]:.0f} GeV ({last[(b, tr)] - last[(a, tr)]:+.0f})")
            for key in args.datasets:
                if last.get((key, "raw")) is not None and last.get((key, "kde9")) is not None:
                    L.append(f"- smoothing, {C.DATASETS[key]['results']}: raw {last[(key, 'raw')]:.0f} -> "
                             f"kde9 {last[(key, 'kde9')]:.0f} GeV ({last[(key, 'kde9')] - last[(key, 'raw')]:+.0f})")
            L.append("")
            ax.axhline(PASS_FRAC, color="0.2", lw=0.9, ls="--")
            ax.set_ylim(0.0, 1.6)
            ax.set_xlabel(r"$m_{W_R}$ [GeV]")
            ax.set_ylabel("fraction of fits passing the checks")
            # the luminosities are the legend's entries: the frame's label carries the energy only. The block
            # and the legend hang side by side from the top: keep both narrow (a wider block or legend makes
            # fit_header move the legend onto the curves)
            extra = f"{CAND_LABEL.get(cand, cand)}\n$\\pm 5\\sigma$ window\nfrom {floor:.0f} GeV\nspurious signal"
            cms_label(ax, None, C.COM, data=False, text=P.region_text(channel, extra))
            handles = [Line2D([], [], color=COLOUR[key], lw=1.5, label=ds_label(key)) for key in args.datasets
                       if any(data[(key, tr)] for tr in TRUTHS)]
            for tr in TRUTHS:
                ls, mk = T.TRUTH_LM.get(tr, ("-", "o"))
                handles.append(Line2D([], [], color="k", lw=1.5, ls=ls, marker=mk, ms=6, label=T.TRUTH_BKG.get(tr, tr)))
            handles.append(Line2D([], [], color="0.2", lw=0.9, ls="--", label=r"$90\,\%$"))
            handles.append(Line2D([], [], marker="o", mfc="white", mec="k", color="none", ms=6, label=r"open: $< 90\,\%$ pass"))
            ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE_LEGEND, ncol=1, handlelength=1.6)
            fit_header(ax, ignore_above=True)
            P.save(fig, out / f"lumi_pass_fraction_{cand}{tag}")
            # the same datasets and truths for the Gaussian mu and sigma of the pulls, beside
            # the pass fraction
            for stat, key_col, ylabel, ylim in [("pull_mean", "pull_gauss_mean", r"Gaussian $\mu$ of the pulls", (-2.0, 3.0)),
                                                ("width", "pull_width", r"Gaussian $\sigma$ of the pulls", (0.0, 3.0))]:
                fig, ax = plt.subplots(figsize=(10, 8))
                rows_md = []
                for key in args.datasets:
                    for tr in TRUTHS:
                        rows = data[(key, tr)]
                        if not rows:
                            continue
                        x, y = T._series(rows, cand, 0.0, key_col)     # cells with fewer than 10 passing toys hold no value
                        x, y = np.array(x, float), np.array(y, float)
                        frac = dict(T.pass_fractions(rows, cand))
                        good = np.array([frac.get(m_, 0.0) >= PASS_FRAC for m_ in x], bool)
                        ls, mk = T.TRUTH_LM.get(tr, ("-", "o"))
                        ax.plot(x, y, lw=1.5, color=COLOUR[key], ls=ls)
                        ax.plot(x[good], y[good], mk, ms=6, color=COLOUR[key])
                        ax.plot(x[~good], y[~good], mk, ms=6, mfc="white", mec=COLOUR[key])
                        if stat == "pull_mean":
                            at = dict(zip(x, y))
                            f5 = [f"{m_:.0f}" for m_, v, g in zip(x, y, good) if g and abs(v) > C.BIAS_PASS]
                            fl = good & (x <= C.FLOAT_MAX) & (x > floor)
                            i = int(np.argmax(np.abs(y[fl]))) if fl.any() else None
                            mx = f"{y[fl][i]:+.2f} ({x[fl][i]:.0f})" if i is not None else "-"
                            rows_md.append(f"| {C.DATASETS[key]['results']} ({C.DATASETS[key]['lumi']:.1f} fb-1) | {tr} | "
                                           f"{', '.join(f5) or '-'} | {mx} | "
                                           + " / ".join(f"{at[m_]:+.2f}" if m_ in at else "-" for m_ in PULL_MASSES) + " |")
                if stat == "pull_mean":
                    L += ["| dataset | truth | masses with abs(mu) > 0.5 (cells with >= 90 % passing) | "
                          f"largest abs(mu) above the floor up to {C.FLOAT_MAX / 1000:g} TeV (mass) | "
                          f"mu at {' / '.join(f'{m_ / 1000:g}' for m_ in PULL_MASSES)} TeV |", "|---|---|---|---|---|"] + rows_md + [""]
                    for yv, c_ in [(C.BIAS_TARGET, "0.5"), (C.BIAS_PASS, "0.2")]:
                        ax.axhline(yv, color=c_, lw=0.9, ls="--")
                        ax.axhline(-yv, color=c_, lw=0.9, ls="--")
                    ax.axhline(0, color="k", lw=0.8)
                else:
                    ax.axhline(1.0, color="k", lw=0.8)
                ax.set_ylim(*ylim)
                ax.set_xlabel(r"$m_{W_R}$ [GeV]")
                ax.set_ylabel(ylabel)
                cms_label(ax, None, C.COM, data=False, text=P.region_text(channel, extra))
                handles = [Line2D([], [], color=COLOUR[key], lw=1.5, label=ds_label(key)) for key in args.datasets
                           if any(data[(key, tr)] for tr in TRUTHS)]
                for tr in TRUTHS:
                    ls, mk = T.TRUTH_LM.get(tr, ("-", "o"))
                    handles.append(Line2D([], [], color="k", lw=1.5, ls=ls, marker=mk, ms=6, label=T.TRUTH_BKG.get(tr, tr)))
                handles.append(Line2D([], [], marker="o", mfc="white", mec="k", color="none", ms=6, label=r"open: $< 90\,\%$ pass"))
                ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE_LEGEND, ncol=1, handlelength=1.6)
                fit_header(ax, ignore_above=(stat != "pull_mean"))
                P.save(fig, out / f"lumi_{stat}_{cand}{tag}")
        (out / f"summary{tag}.md").write_text("\n".join(L) + "\n")
        logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
