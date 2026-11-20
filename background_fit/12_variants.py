#!/usr/bin/env python3
"""What moves the bias-versus-mass curve: the window width and the function, on one axis.

Reads the window study's toys (results/<channel>/6_windows/k<k>/lo_ht/toys_<truth>.csv; k = 100
is the full range from FIT_LO) and draws, per channel and truth, the variants
  expo in +-3 and +-5 sigma windows and over the full range; powexp in +-5 sigma and the full range
  (the +-10 sigma window behaves as the full range and is left out of the figures):
  pull_mean_vs_mass_<truth>   the bias as CMS quotes it (Gaussian mu of the pulls) vs m_WR
  width_vs_mass_<truth>       the Gaussian sigma of the pulls vs m_WR
  sigfit_vs_mass_<truth>      the cost: the Asimov sigma_fit in events vs m_WR (log)
  sigfit_ratio_vs_mass_<truth> the same over the full-range powexp's sigma_fit
Filled markers at the masses where at least 90 % of the variant's fits pass the checks, open markers
where they do not. summary.md: per variant, the
masses failing 0.5 and 0.3 among the filled points, the largest |mu| there, and sigma_fit over
the full-range powexp at 2 and 3 TeV.

  cd WR_Plotter && python background_fit/12_variants.py [--truths kde9 raw]
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
from util import fmt, read_csv                                           # noqa: E402

logger = logging.getLogger("12_variants")
# (candidate, k, label, colour, line style, marker); k = 100 is the full range from FIT_LO
VARIANTS = [
    ("expo", 3, r"expo, $\pm 3\sigma$", "#5790fc", "-", "o"),
    ("expo", 5, r"expo, $\pm 5\sigma$", "#e42536", "-", "o"),
    ("expo", 100, "expo, full range", "#964a8b", "-", "o"),
    ("powexp", 5, r"powexp, $\pm 5\sigma$", "#9c9ca1", "--", "s"),
    ("powexp", 100, "powexp, full range", "#7a21dd", "--", "s"),
]
PASS_FRAC = 0.9


def load(channel, k, truth):
    p = C.RESULTS / channel / "6_windows" / f"k{k:g}" / "lo_ht" / f"toys_{truth}.csv"
    return [T._type_row(r) for r in read_csv(p)] if p.exists() else []


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--truths", nargs="+", default=["kde9", "raw"])
    ap.add_argument("--out", default="12_variants")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    from wrplotter.cli_utils import setup_logging
    setup_logging(args.verbose)
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.lines import Line2D
    import plots as P
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    hep.style.use("CMS")
    L = ["# What moves the bias curve: window width and function", "",
         f"Toys of the window study (500 per cell, lo_ht), the Gaussian fitted to the pulls at zero injection; a variant's "
         f"points count while >= {PASS_FRAC:.0%} of its fits pass the checks. sigma_fit = the Asimov fit error.", ""]
    for channel in args.channels:
        out = C.RESULTS / channel / args.out
        out.mkdir(parents=True, exist_ok=True)
        for truth in args.truths:
            data = {(cand, k): load(channel, k, truth) for cand, k, *_ in VARIANTS}
            extra = T.TRUTH_BKG.get(truth, truth)          # a short label block leaves room for the legend beside it
            # the full-range powexp as the sigma_fit reference
            ref_rows = data[("powexp", 100)]
            frac_all = {(cand, k): dict(T.pass_fractions(data[(cand, k)], cand)) for cand, k, *_ in VARIANTS if data[(cand, k)]}
            ref_sig = dict(zip(*T._series(ref_rows, "powexp", 0.0, "asimov_err")))
            L += [f"## {channel}, truth {truth}", "",
                  "| variant | sparse from | masses with abs(mu) > 0.5 (cells with >= 90 % passing) | > 0.3 | max abs(mu) | sigma_fit / full powexp at 2, 3 TeV |",
                  "|---|---|---|---|---|---|"]
            for stat, key, ylabel, ylim, log, loc in [
                    ("pull_mean", "pull_gauss_mean", r"Gaussian $\mu$ of the pulls", (-2.0, 3.0), False, "upper right"),
                    ("width", "pull_width", r"Gaussian $\sigma$ of the pulls", (0.0, 3.0), False, "upper right"),
                    ("sigfit", "asimov_err", r"$\sigma_\mathrm{fit}$ [events]", (0.3, 3000.0), True, "upper right"),
                    ("sigfit_ratio", "asimov_err", r"$\sigma_\mathrm{fit}$ / $\sigma_\mathrm{fit}$(powexp, full range)", (0.6, 2.4), False, "upper right")]:
                fig, ax = plt.subplots(figsize=(10, 8))
                for cand, k, label, col, ls, mk in VARIANTS:
                    rows = data[(cand, k)]
                    if not rows:
                        continue
                    x, y = T._series(rows, cand, 0.0, key)
                    x, y = np.array(x), np.array(y, float)
                    if stat == "sigfit_ratio":                       # the cost against the best full-range function,
                        y = np.array([v / ref_sig[m_] if m_ in ref_sig and ref_sig[m_] > 0 else np.nan for m_, v in zip(x, y)])
                        keep = np.array([frac_all.get((cand, k), {}).get(m_, 0.0) >= PASS_FRAC for m_ in x])   # where the fits pass
                        x, y = x[keep], y[keep]
                    frac = dict(T.pass_fractions(rows, cand))
                    good = np.array([frac.get(m_, 0.0) >= PASS_FRAC for m_ in x])     # per point: filled when >= 90 % pass
                    sparse = T.first_sparse(rows, cand, PASS_FRAC)
                    ax.plot(x, y, lw=1.5, color=col, ls=ls, label=label)
                    ax.plot(x[good], y[good], mk, ms=6, color=col)
                    ax.plot(x[~good], y[~good], mk, ms=6, mfc="white", mec=col)
                    if stat == "pull_mean":
                        xs, ys = T._series(rows, cand, 0.0, "pull_gauss_mean")
                        xs, ys = np.array(xs), np.array(ys, float)
                        g = np.array([frac.get(m_, 0.0) >= PASS_FRAC for m_ in xs])
                        f5 = [f"{m_:.0f}" for m_, v in zip(xs[g], ys[g]) if abs(v) > C.BIAS_PASS]
                        f3 = [f"{m_:.0f}" for m_, v in zip(xs[g], ys[g]) if abs(v) > C.BIAS_TARGET]
                        mx = float(np.nanmax(np.abs(ys[g]))) if g.any() else float("nan")
                        sig = dict(zip(*T._series(rows, cand, 0.0, "asimov_err")))
                        ratios = " / ".join(f"{sig[m_] / ref_sig[m_]:.2f}" if m_ in sig and m_ in ref_sig else "-" for m_ in (2000.0, 3000.0))
                        L.append(f"| {label.replace('$', '').replace(chr(92) + 'pm ', '+-')} | {fmt(sparse, 0) if sparse else '-'} | "
                                 f"{', '.join(f5) or '-'} | {', '.join(f3) or '-'} | {mx:.2f} | {ratios} |")
                if stat == "pull_mean":
                    for yv, c_ in [(C.BIAS_TARGET, "0.5"), (C.BIAS_PASS, "0.2")]:
                        ax.axhline(yv, color=c_, lw=0.9, ls="--")
                        ax.axhline(-yv, color=c_, lw=0.9, ls="--")
                    ax.axhline(0, color="k", lw=0.8)
                elif stat in ("width", "sigfit_ratio"):
                    ax.axhline(1.0, color="k", lw=0.8)
                if log:
                    ax.set_yscale("log")
                ax.set_ylim(*ylim)
                ax.set_xlabel(r"$m_{W_R}$ [GeV]")
                ax.set_ylabel(ylabel)
                cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(channel, extra), year=C.YEAR)
                handles, labels = ax.get_legend_handles_labels()
                if stat != "sigfit_ratio":                       # the ratio figure keeps only the passing cells
                    handles.append(Line2D([], [], marker="o", mfc="white", mec="k", color="none", ms=6, label=r"open: $< 90\,\%$ of fits pass"))
                ax.legend(handles=handles, loc=loc, fontsize=FONT_SIZE_LEGEND, ncol=1, handlelength=1.6)
                fit_header(ax, ignore_above=(stat != "pull_mean"))
                P.save(fig, out / f"{stat}_vs_mass_{truth}")
            L.append("")
        (out / "summary.md").write_text("\n".join(L) + "\n")
        logger.info("wrote %s", out)
    (C.RESULTS / "VARIANTS.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
