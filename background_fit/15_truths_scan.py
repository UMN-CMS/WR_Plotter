#!/usr/bin/env python3
"""The +-5 sigma window on four background estimates: the Gaussian mu and sigma of the
zero-injection pulls against m_WR for the exponential and the power x exponential, one function per figure,
colour = background estimate:

  raw MC, LO HT-binned DY                     6_windows/<folder>/lo_ht/toys_raw.csv
  raw MC, LO HT-binned DY reweighted           6_windows/<folder>/lo_ht_rw/toys_raw.csv   (Z-pT + jet-pT reweight)
  raw MC, NLO m(ll)-binned DY                 6_windows/<folder>/nlo_mll/toys_raw.csv
  nine-process RooKeysPdf, LO HT-binned DY    6_windows/<folder>/lo_ht/toys_kde9.csv

Filled markers where >= 90 % of the cell's fits pass the checks, open where not; the +-0.3 and +-0.5 lines.
Writes results/<ch>/<out>/truths4_<cand>_{pull_mean,width}.{pdf,png} and results/TRUTHS4<tag>.md: per estimate
and function the sparse mass, the cells beyond 0.5 and 0.3, the largest |mu| in the float regime, mu at a few
masses, the sigma range; a per-mass table of mu (sigma) for every estimate; and the spread of mu between the
estimates per mass. The dataset is FIT_DATASET as everywhere (2024, run3, run3_scaled).

  cd WR_Plotter && python background_fit/15_truths_scan.py [--folder k5_lo1200_100k]
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
from util import fmt, read_csv                                           # noqa: E402

T = importlib.import_module("4_toys")                                    # noqa: E402
logger = logging.getLogger("truths_scan")
PASS_FRAC = T.PASS_FRAC

# (key, DY variant, truth name in the files, label, colour, line style, marker)
TRUTHS4 = [("raw_lo", "lo_ht", "raw", "raw MC, LO HT DY", "#717581", "-", "o"),
           ("raw_rw", "lo_ht_rw", "raw", "raw MC, LO HT DY reweighted", "#5790fc", "-", "^"),
           ("raw_nlo", "nlo_mll", "raw", r"raw MC, NLO $m(\ell\ell)$ DY", "#e42536", "-", "D"),
           ("kde9", "lo_ht", "kde9", "RooKeysPdf, LO HT DY", "#7a21dd", "--", "s")]
TRUTH_LONG = {"raw_lo": "raw MC, LO HT-binned DY", "raw_rw": "raw MC, LO HT-binned DY, Z-pT + jet-pT reweighted",
              "raw_nlo": "raw MC, NLO m(ll)-binned DY", "kde9": "nine-process RooKeysPdf, LO HT-binned DY"}   # the tables
CAND_LABEL = {"expo": "Expo", "powexp": r"Power $\times$ expo"}
REF_MASSES = (1600.0, 2000.0, 3000.0, 3400.0)


def load(channel, folder, variant, truth):
    p = C.RESULTS / channel / "6_windows" / folder / variant / f"toys_{truth}.csv"
    return [T._type_row(r) for r in read_csv(p)] if p.exists() else []


def series(rows, cand, key):
    x, y = T._series(rows, cand, 0.0, key)
    frac = dict(T.pass_fractions(rows, cand))
    x, y = np.array(x, float), np.array(y, float)
    return x, y, np.array([frac.get(m_, 0.0) >= PASS_FRAC for m_ in x], bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--candidates", nargs="+", default=["expo", "powexp"])
    ap.add_argument("--folder", default="k5_lo1200_100k", help="the window folder under 6_windows/")
    ap.add_argument("--out", default="15_truths_scan")
    ap.add_argument("--md-tag", default="", help="suffix of results/TRUTHS4<tag>.md")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.lines import Line2D
    import plots as P
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    hep.style.use("CMS")
    ntoys = None
    import re
    kk = re.match(r"k(\d+)", args.folder)                   # the half-width in the titles follows the folder (k7_lo1200_100k: 7)
    kk = kk.group(1) if kk else "5"
    L = [f"# The +-{kk} sigma window on four background estimates ({C.ERA}, 6_windows/{args.folder})", "",
         "Gaussian mu and sigma of the zero-injection pulls; a cell counts while >= 90 % of its fits pass the checks "
         "(passed / converged); the float regime ends at the first mass above the last passing one.", ""]
    for channel in args.channels:
        out = C.RESULTS / channel / args.out
        out.mkdir(parents=True, exist_ok=True)
        data = {key: load(channel, args.folder, variant, truth) for key, variant, truth, *_ in TRUTHS4}
        for key, rows in data.items():
            if rows and ntoys is None:
                ntoys = rows[0].get("n_toys")
            if not rows:
                logger.warning("%s: no toys for %s in %s", channel, key, args.folder)
        for cand in args.candidates:
            L += [f"## {channel}, {cand}", "",
                  "| background estimate | sparse from | masses with abs(mu) > 0.5 (cells >= 90 % passing) | > 0.3 | max abs(mu) (mass) | "
                  + " / ".join(f"mu at {m_:.0f}" for m_ in REF_MASSES) + " | sigma min-max |", "|---|---|---|---|---|---|---|"]
            kept = {}
            for stat, key, ylabel in [("pull_mean", "pull_gauss_mean", r"Gaussian $\mu$ of the pulls"),
                                      ("width", "pull_width", r"Gaussian $\sigma$ of the pulls")]:
                fig, ax = plt.subplots(figsize=(10, 8))
                ys = []
                for tkey, variant, truth, tlabel, col, ls, mk in TRUTHS4:
                    rows = data[tkey]
                    if not rows:
                        continue
                    x, y, good = series(rows, cand, key)
                    if not len(x):
                        continue
                    ax.plot(x, y, lw=1.5, color=col, ls=ls)
                    ax.plot(x[good], y[good], mk, ms=6, color=col)
                    ax.plot(x[~good], y[~good], mk, ms=6, mfc="white", mec=col)
                    ys.append(y[good])
                    if stat == "pull_mean":
                        kept[tkey] = (x, y, good)
                        sparse = T.first_sparse(rows, cand, PASS_FRAC)
                        fl = good & (x < (sparse if sparse else 1e9))
                        f5 = [f"{m_:.0f}" for m_, v, g in zip(x, y, fl) if g and abs(v) > C.BIAS_PASS]
                        f3 = [f"{m_:.0f}" for m_, v, g in zip(x, y, fl) if g and abs(v) > C.BIAS_TARGET]
                        if fl.any():
                            i = int(np.argmax(np.where(fl, np.abs(y), -1.0)))
                            mx = f"{y[i]:+.2f} ({x[i]:.0f})"
                        else:
                            mx = "-"
                        at = {m_: v for m_, v in zip(x, y)}
                        xs_, sig, gs = series(rows, cand, "pull_width")
                        sig_fl = sig[gs & (xs_ < (sparse if sparse else 1e9))]
                        L.append(f"| {TRUTH_LONG[tkey]} | {fmt(sparse, 0) if sparse else '-'} | {', '.join(f5) or '-'} | {', '.join(f3) or '-'} | {mx} | "
                                 + " / ".join(f"{at[m_]:+.2f}" if m_ in at else "-" for m_ in REF_MASSES)
                                 + f" | {np.nanmin(sig_fl):.2f}-{np.nanmax(sig_fl):.2f} |" if len(sig_fl) else " | - |")
                if stat == "pull_mean":
                    for yv, c_ in [(C.BIAS_TARGET, "0.5"), (C.BIAS_PASS, "0.2")]:
                        ax.axhline(yv, color=c_, lw=0.9, ls="--")
                        ax.axhline(-yv, color=c_, lw=0.9, ls="--")
                    ax.axhline(0, color="k", lw=0.8)
                    ymin = float(np.nanmin(np.concatenate(ys))) if ys and len(np.concatenate(ys)) else -2.0
                    ax.set_ylim(min(-2.0, ymin - 0.3), 3.0)
                else:
                    ax.axhline(1.0, color="k", lw=0.8)
                    ax.set_ylim(0.0, 2.5)
                ax.set_xlim(1000.0, 6400.0)                 # the sparse regime beyond is open markers only
                ax.set_xlabel(r"$m_{W_R}$ [GeV]")
                ax.set_ylabel(ylabel)
                cms_label(ax, C.LUMI, C.COM, data=False,
                          text=P.region_text(channel, f"{CAND_LABEL.get(cand, cand)}\n$\\pm {kk}\\sigma$ window"), year=C.YEAR)
                handles = [Line2D([], [], color=col, lw=1.5, ls=ls, marker=mk, ms=6, label=tlabel)
                           for tkey, _, _, tlabel, col, ls, mk in TRUTHS4 if data[tkey]]
                handles.append(Line2D([], [], marker="o", mfc="white", mec="k", color="none", ms=6, label=r"open: $< 90\,\%$ of fits pass"))
                ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE_LEGEND, ncol=1, handlelength=1.6)
                fit_header(ax, ignore_above=(stat != "pull_mean"))
                P.save(fig, out / f"truths4_{cand}_{stat}")
            # the per-mass table and the spread between the estimates
            keys = [k for k, *_ in TRUTHS4 if k in kept]
            if keys:
                masses = sorted(set().union(*[set(kept[k][0].tolist()) for k in keys]))
                sig_of = {k: dict(zip(*series(data[k], cand, "pull_width")[:2])) for k in keys}
                L += ["", f"Per mass, mu (sigma), '*' = fewer than 90 % of the fits pass:", "",
                      "| m_WR | " + " | ".join(TRUTH_LONG[k] for k in keys) + " | spread of mu |",
                      "|---|" + "---|" * (len(keys) + 1)]
                for m_ in masses:
                    cells, vals, allgood = [], [], True
                    for k in keys:
                        x, y, good = kept[k]
                        j = np.nonzero(x == m_)[0]
                        if not len(j):
                            cells.append("-"); allgood = False; continue
                        v, g = float(y[j[0]]), bool(good[j[0]])
                        allgood &= g
                        vals.append(v)
                        cells.append(f"{v:+.2f} ({sig_of[k].get(m_, float('nan')):.2f}){'' if g else '*'}")
                    spread = f"{max(vals) - min(vals):.2f}" + ("" if allgood and len(vals) == len(keys) else "*") if vals else "-"
                    L.append(f"| {m_:.0f} | " + " | ".join(cells) + f" | {spread} |")
            L.append("")
    L.insert(2, f"Toys per cell: {ntoys}." if ntoys else "")
    (C.RESULTS / f"TRUTHS4{args.md_tag}.md").write_text("\n".join(L) + "\n")
    logger.info("wrote %s and results/<ch>/%s/", C.RESULTS / f"TRUTHS4{args.md_tag}.md", args.out)


if __name__ == "__main__":
    main()
