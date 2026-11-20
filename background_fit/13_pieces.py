#!/usr/bin/env python3
"""The bias figures in pieces: one change at a time against the starting configuration.

Reads the window study's toys (results/<channel>/6_windows/k<k>[_lo<lo>]/lo_ht/toys_<truth>.csv:
k3, k5 and k100_lo1200 = the full range from 1200 GeV) and draws, per channel, both truths on one
axis (raw MC: solid, circles; nine-process RooKeysPdf: dashed, squares), colour = configuration,
filled markers where at least 90 % of the cell's fits pass the checks and open ones where not:
  piece 1, the function is the problem, not the lever arm: the exponential in +-3 sigma, +-5 sigma
          and over the full range from 1200 GeV
      expo_range_pull_mean, expo_range_width, expo_range_sigfit_ratio (over the +-5 sigma window)
  piece 2, with the function fixed, does the window still buy anything: the power x exponential in
          the +-5 sigma window and over the full range from 1200 GeV
      powexp_range_pull_mean, powexp_range_width, powexp_range_sigfit_ratio (window over full range),
      powexp_range_pass_fraction (the fraction of fits passing the checks, with the 90 % line)
  piece 3, the tail: the power x exponential over the full range and in the window on truths whose
          tails differ (raw MC and nine-process RooKeysPdf + fitted expo tail, each on the LO HT-binned
          and the NLO m(ll)-binned DY sample; the RooKeysPdf alone as the stress case)
      tails_full_pull_mean, tails_full_width, tails_window_pull_mean, tails_window_width,
      tails_spread (max - min of the Gaussian mu over the four plausible truths per mass, both ranges)
  window sizes (the talk, 2026-10-27): the window half-width at the 1200 GeV fit floor, +-3, 4, 5, 7,
          10 sigma and the full range, one function and one truth per figure, colour = half-width
          (light to dark as the window widens, the full range black)
      wsize_<cand>_<truth>_pull_mean, wsize_<cand>_<truth>_width, wsize_<cand>_<truth>_sigfit_ratio
      (over the full range), for expo and powexp on raw and kde9; --wsize-lo 1000 draws the same from
      the 1000 GeV floor's k<k> folders as wsize_lo1000_<cand>_<truth>_<stat>; --wsize-k 3 5 100 draws
      only those widths (same colours) as wsize_k3-5-100_<cand>_<truth>_<stat> (the talk's body frames)
summary.md per channel and results/PIECES.md: per configuration and truth, the masses failing 0.5
and 0.3 among the cells with >= 90 % passing, the largest |mu| there, where the sparse regime
begins, and sigma_fit over the reference configuration at 2, 3 and 3.4 TeV.

  cd WR_Plotter && python background_fit/13_pieces.py
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

logger = logging.getLogger("13_pieces")
PASS_FRAC = T.PASS_FRAC
TRUTHS = ["raw", "kde9"]
# the configurations: (key, candidate, folder under 6_windows/, legend label, colour); the +-5 sigma
# window keeps the candidate's colour of the mass-grid frames (expo blue, powexp orange)
PIECES = {
    "expo_range": {
        "block": "Expo",
        "ref": "k5",
        "configs": [("k3", "expo", "k3", r"$\pm 3\sigma$ window", "#e42536"),
                    ("k5", "expo", "k5", r"$\pm 5\sigma$ window", "#5790fc"),
                    ("full", "expo", "k100_lo1200", "full range from 1200 GeV", "#964a8b")],
    },
    "powexp_range": {
        "block": r"Power $\times$ expo",
        "ref": "full",
        "configs": [("k5", "powexp", "k5", r"$\pm 5\sigma$ window", "#f89c20"),
                    ("full", "powexp", "k100_lo1200", "full range from 1200 GeV", "#964a8b")],
    },
}
REF_MASSES = (2000.0, 3000.0, 3400.0)


TOYS_TAG = ""                       # --toys-tag: the folder suffix of a high-statistics rerun (6_windows.py --tag _100k)


def load(channel, folder, truth, variant="lo_ht"):
    p = C.RESULTS / channel / "6_windows" / (folder + TOYS_TAG) / variant / f"toys_{truth}.csv"
    return [T._type_row(r) for r in read_csv(p)] if p.exists() else []


# piece 3: (key, DY variant, truth name in the files, label, colour); colour = smoother, line style
# and marker = DY sample (LO HT-binned solid / circles, NLO m(ll)-binned dashed / squares)
TAIL_TRUTHS = [
    ("raw_lo", "lo_ht", "raw", "Raw MC Bkg", "#717581"),
    ("raw_nlo", "nlo_mll", "raw", "Raw MC Bkg", "#717581"),
    ("kdet_lo", "lo_ht", "kde_expo9", "RooKeysPdf + tail Bkg", "#7a21dd"),
    ("kdet_nlo", "nlo_mll", "kde_expo9", "RooKeysPdf + tail Bkg", "#7a21dd"),
    ("kde_lo", "lo_ht", "kde9", "RooKeysPdf Bkg, no tail", "#e42536"),
    ("kde_nlo", "nlo_mll", "kde9", "RooKeysPdf Bkg, no tail", "#e42536"),
]
PLAUSIBLE = ["raw_lo", "raw_nlo", "kdet_lo", "kdet_nlo"]       # the spread is taken over these
SAMPLE_LM = {"lo_ht": ("-", "o"), "nlo_mll": ("--", "s")}
SAMPLE_LABEL = {"lo_ht": "LO HT-binned DY", "nlo_mll": r"NLO $m(\ell\ell)$-binned DY"}
TAIL_CONFIGS = [("full", "k100_lo1200", "full range from 1200 GeV", "#964a8b"),
                ("window", "k5", r"$\pm 5\sigma$ window", "#f89c20")]


def series(rows, cand, key):
    """(masses, values, >= 90 % passing) of the zero-injection cells, in mass order."""
    x, y = T._series(rows, cand, 0.0, key)
    frac = dict(T.pass_fractions(rows, cand))
    x, y = np.array(x, float), np.array(y, float)
    return x, y, np.array([frac.get(m_, 0.0) >= PASS_FRAC for m_ in x], bool)


def draw(ax, x, y, good, col, ls, mk):
    ax.plot(x, y, lw=1.5, color=col, ls=ls)
    ax.plot(x[good], y[good], mk, ms=6, color=col)
    ax.plot(x[~good], y[~good], mk, ms=6, mfc="white", mec=col)


def legend_handles(configs, truths, skip=None):
    """A compact legend: one colour entry per configuration, one black line-style + marker entry per
    truth (the long "configuration, truth" product would cover the curves)."""
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=col, lw=1.5, label=label) for key, _, _, label, col in configs if key != skip]
    for tr in truths:
        ls, mk = T.TRUTH_LM.get(tr, ("-", "o"))
        hs.append(Line2D([], [], color="k", lw=1.5, ls=ls, marker=mk, ms=6, label=T.TRUTH_BKG.get(tr, tr)))
    return hs


def tails_piece(channel, out, L, cand="powexp"):
    """Piece 3: the candidate over the full range and in the window on truths whose tails differ."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import plots as P
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    data = {(key, cfg): load(channel, folder, truth, variant)
            for key, variant, truth, _, _ in TAIL_TRUTHS for cfg, folder, _, _ in TAIL_CONFIGS}
    cand_label = {"powexp": r"Power $\times$ expo", "expo": "Expo"}.get(cand, cand)
    L += [f"## {channel}, tails ({cand})", "",
          "| range | truth | DY sample | sparse from | masses with abs(mu) > 0.5 (cells with >= 90 % passing) | > 0.3 | max abs(mu) |",
          "|---|---|---|---|---|---|---|"]
    series_by = {}
    for cfg, folder, cfg_label, cfg_col in TAIL_CONFIGS:
        extra = f"{cand_label}\n{cfg_label}\nspurious signal"
        for stat, key, ylabel, ylim in [("pull_mean", "pull_gauss_mean", r"Gaussian $\mu$ of the pulls", (-2.0, 4.0)),
                                        ("width", "pull_width", r"Gaussian $\sigma$ of the pulls", (0.0, 3.5))]:
            fig, ax = plt.subplots(figsize=(10, 8))
            for tkey, variant, truth, tlabel, col in TAIL_TRUTHS:
                rows = data[(tkey, cfg)]
                if not rows:
                    continue
                ls, mk = SAMPLE_LM[variant]
                x, y, good = series(rows, cand, key)
                draw(ax, x, y, good, col, ls, mk)
                if stat == "pull_mean":
                    series_by[(tkey, cfg)] = (x, y, good)
                    sparse = T.first_sparse(rows, cand, PASS_FRAC)
                    f5 = [f"{m_:.0f}" for m_, v, g in zip(x, y, good) if g and abs(v) > C.BIAS_PASS]
                    f3 = [f"{m_:.0f}" for m_, v, g in zip(x, y, good) if g and abs(v) > C.BIAS_TARGET]
                    mx = float(np.nanmax(np.abs(y[good]))) if good.any() else float("nan")
                    L.append(f"| {cfg_label.replace('$', '').replace(chr(92) + 'pm ', '+-')} | {tlabel} | {C.DY_SHORT[variant]} | "
                             f"{fmt(sparse, 0) if sparse else '-'} | {', '.join(f5) or '-'} | {', '.join(f3) or '-'} | {mx:.2f} |")
            if stat == "pull_mean":
                for yv, c_ in [(C.BIAS_TARGET, "0.5"), (C.BIAS_PASS, "0.2")]:
                    ax.axhline(yv, color=c_, lw=0.9, ls="--")
                    ax.axhline(-yv, color=c_, lw=0.9, ls="--")
                ax.axhline(0, color="k", lw=0.8)
            else:
                ax.axhline(1.0, color="k", lw=0.8)
            ax.set_ylim(*ylim)
            ax.set_xlabel(r"$m_{W_R}$ [GeV]")
            ax.set_ylabel(ylabel)
            cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(channel, extra), year=C.YEAR)
            seen, handles = set(), []
            for _, _, _, tlabel, col in TAIL_TRUTHS:
                if tlabel not in seen:
                    seen.add(tlabel)
                    handles.append(Line2D([], [], color=col, lw=1.5, label=tlabel))
            for variant, (ls, mk) in SAMPLE_LM.items():
                handles.append(Line2D([], [], color="k", lw=1.5, ls=ls, marker=mk, ms=6, label=SAMPLE_LABEL[variant]))
            handles.append(Line2D([], [], marker="o", mfc="white", mec="k", color="none", ms=6, label=r"open: $< 90\,\%$ of fits pass"))
            ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE_LEGEND, ncol=1, handlelength=1.6)
            fit_header(ax, ignore_above=(stat != "pull_mean"))
            P.save(fig, out / f"tails_{cfg}_{stat}")
    # the spread: max - min of the Gaussian mu over the plausible truths, per mass and range
    fig, ax = plt.subplots(figsize=(10, 8))
    L += ["", "| range | max spread of mu over the four plausible truths, m_WR <= 3200 (where) | masses <= 3200 with spread > 0.5 |", "|---|---|---|"]
    for cfg, folder, cfg_label, cfg_col in TAIL_CONFIGS:
        per_mass = {}
        for tkey in PLAUSIBLE:
            if (tkey, cfg) not in series_by:
                continue
            x, y, good = series_by[(tkey, cfg)]
            for m_, v, g in zip(x, y, good):
                per_mass.setdefault(m_, []).append((v, g))
        xs = np.array(sorted(m_ for m_, vals in per_mass.items() if len(vals) == len(PLAUSIBLE)), float)
        if not len(xs):
            continue
        ys = np.array([max(v for v, _ in per_mass[m_]) - min(v for v, _ in per_mass[m_]) for m_ in xs])
        good = np.array([all(g for _, g in per_mass[m_]) for m_ in xs], bool)
        draw(ax, xs, ys, good, cfg_col, "-", "o")
        fl = xs <= C.FLOAT_MAX
        if (good & fl).any():
            i = int(np.argmax(np.where(good & fl, ys, -1.0)))
            L.append(f"| {cfg_label.replace('$', '').replace(chr(92) + 'pm ', '+-')} | {ys[i]:.2f} ({xs[i]:.0f}) | "
                     f"{', '.join(f'{m_:.0f}' for m_, v, g in zip(xs, ys, good) if g and v > C.BIAS_PASS and m_ <= C.FLOAT_MAX) or '-'} |")
    ax.axhline(C.BIAS_PASS, color="0.2", lw=0.9, ls="--")
    ax.set_ylim(0.0, 1.6)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"spread of the Gaussian $\mu$ over the truths")
    cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(channel, f"{cand_label}\nspurious signal"), year=C.YEAR)
    handles = [Line2D([], [], color=col, lw=1.5, marker="o", ms=6, label=lab) for _, _, lab, col in TAIL_CONFIGS]
    handles.append(Line2D([], [], marker="o", mfc="white", mec="k", color="none", ms=6, label=r"open: a truth $< 90\,\%$ passing"))
    ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE_LEGEND, ncol=1, handlelength=1.6)
    fit_header(ax, ignore_above=True)
    P.save(fig, out / "tails_spread")
    L.append("")


# window sizes at the 1200 GeV fit floor: (k, folder under 6_windows/, legend label); the colours run
# light to dark as the window widens (viridis 0.75 -> 0.10), the full range is black
WSIZE_K = [(3, r"$\pm 3\sigma$"), (4, r"$\pm 4\sigma$"), (5, r"$\pm 5\sigma$"), (7, r"$\pm 7\sigma$"),
           (10, r"$\pm 10\sigma$"), (100, "full range")]
WSIZE_LO = 1200                     # the talk's floor; --wsize-lo 1000 reads the k<k> folders (FIT_LO) instead
CAND_LABEL = {"powexp": r"Power $\times$ expo", "expo": "Expo"}


def wsize_colours():
    import matplotlib.pyplot as plt
    return [plt.cm.viridis(v) for v in (0.75, 0.58, 0.42, 0.26, 0.10)] + ["k"]


def wsize_folder(k, lo):
    """6_windows/k<k> for the default floor (FIT_LO), k<k>_lo<lo> otherwise (6_windows.py kdir)."""
    return f"k{k:g}" + ("" if lo == C.FIT_LO else f"_lo{lo:.0f}")


def window_sizes_piece(channel, out, L, cands=("expo", "powexp"), truths=TRUTHS, lo=WSIZE_LO, ks=None):
    """The window half-width at the fit floor lo: one function and one truth per figure, one line per
    half-width, the full range as the sigma_fit reference. Files wsize_<cand>_<truth>_<stat> for the
    talk's 1200 GeV floor, wsize_lo<lo>_<cand>_<truth>_<stat> for another; ks restricts the widths drawn
    (each keeps its colour of the full set) and adds _k<k1>-<k2>-... to the stem (the talk's body frames
    since 2026-10-27 afternoon: 3, 5, 100)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import plots as P
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    colour_of = dict(zip([k for k, _ in WSIZE_K], wsize_colours()))
    widths = [(k, lab) for k, lab in WSIZE_K if ks is None or k in ks]
    cols = [colour_of[k] for k, _ in widths]
    folders = {k: wsize_folder(k, lo) for k, _ in widths}
    ref_k = WSIZE_K[-1][0]
    assert ref_k in folders, "the full range (k = 100) is the sigma_fit reference and must be among the widths"
    stem = ("wsize" if lo == WSIZE_LO else f"wsize_lo{lo:.0f}") + ("" if ks is None else "_k" + "-".join(f"{k:g}" for k, _ in widths)) + TOYS_TAG
    for cand in cands:
        for tr in truths:
            data = {k: load(channel, folders[k], tr) for k, _ in widths}
            if not any(data.values()):
                continue
            ref_rows = data[ref_k]
            rx, ry, rgood = series(ref_rows, cand, "asimov_err") if ref_rows else (np.array([]), np.array([]), np.array([], bool))
            rsig, rok = dict(zip(rx, ry)), dict(zip(rx, rgood))
            # function and truth on separate lines: a wide label block overlaps the legend horizontally and
            # fit_header then stacks the two, pushing the legend onto the curves
            extra = f"{CAND_LABEL.get(cand, cand)}\n{T.TRUTH_BKG.get(tr, tr)}\nwindows from {lo:.0f} GeV\nspurious signal"
            L += [f"## {channel}, window sizes at the {lo:.0f} GeV floor: {cand} on {tr}" + ("" if ks is None else f" ({', '.join(lab for _, lab in widths)})"), "",
                  f"| window | sparse from | masses with abs(mu) > 0.5 (cells with >= 90 % passing) | > 0.3 | max abs(mu) | "
                  f"sigma_fit / full range at {', '.join(f'{m_ / 1000:g}' for m_ in REF_MASSES)} TeV |",
                  "|---|---|---|---|---|---|"]
            ytop = 4.5 if cand == "expo" else 4.0
            for stat, key, ylabel, ylim in [("pull_mean", "pull_gauss_mean", r"Gaussian $\mu$ of the pulls", (-2.5, ytop)),
                                            ("width", "pull_width", r"Gaussian $\sigma$ of the pulls", (0.0, 3.5)),
                                            ("sigfit_ratio", "asimov_err", r"$\sigma_\mathrm{fit}$ / $\sigma_\mathrm{fit}$(full range)", (0.4, 2.4))]:
                fig, ax = plt.subplots(figsize=(10, 8))
                ys = []                                           # every drawn value: the lower edge follows the data
                for (k, label), col in zip(widths, cols):
                    rows = data[k]
                    if not rows or (stat == "sigfit_ratio" and k == ref_k):
                        continue
                    x, y, good = series(rows, cand, key)
                    if stat == "sigfit_ratio":
                        y = np.array([v / rsig[m_] if m_ in rsig and rsig[m_] > 0 else np.nan for m_, v in zip(x, y)])
                        good = good & np.array([rok.get(m_, False) for m_ in x])
                    draw(ax, x, y, good, col, "-", "o")
                    ys.append(np.asarray(y, float))
                    if stat == "pull_mean":
                        sparse = T.first_sparse(rows, cand, PASS_FRAC)
                        f5 = [f"{m_:.0f}" for m_, v, g in zip(x, y, good) if g and abs(v) > C.BIAS_PASS]
                        f3 = [f"{m_:.0f}" for m_, v, g in zip(x, y, good) if g and abs(v) > C.BIAS_TARGET]
                        mx = float(np.nanmax(np.abs(y[good]))) if good.any() else float("nan")
                        sx, sy, sgood = series(rows, cand, "asimov_err")
                        sig, sok = dict(zip(sx, sy)), dict(zip(sx, sgood))
                        ratios = " / ".join(f"{sig[m_] / rsig[m_]:.2f}" if m_ in sig and m_ in rsig and sok[m_] and rok.get(m_, False) else "-"
                                            for m_ in REF_MASSES)                     # only where both cells pass
                        L.append(f"| {label.replace('$', '').replace(chr(92) + 'pm ', '+-')} | {fmt(sparse, 0) if sparse else '-'} | "
                                 f"{', '.join(f5) or '-'} | {', '.join(f3) or '-'} | {mx:.2f} | {ratios} |")
                if stat == "pull_mean":
                    for yv, c_ in [(C.BIAS_TARGET, "0.5"), (C.BIAS_PASS, "0.2")]:
                        ax.axhline(yv, color=c_, lw=0.9, ls="--")
                        ax.axhline(-yv, color=c_, lw=0.9, ls="--")
                    ax.axhline(0, color="k", lw=0.8)
                else:
                    ax.axhline(1.0, color="k", lw=0.8)
                ymin = float(np.nanmin(np.concatenate(ys))) if ys and np.isfinite(np.concatenate(ys)).any() else ylim[0]
                # the full-range exponential on a larger dataset falls below the fixed edge (-3.1 on Run 3): follow it
                ax.set_ylim(min(ylim[0], ymin - 0.3) if stat == "pull_mean" else ylim[0], ylim[1])
                ax.set_xlabel(r"$m_{W_R}$ [GeV]")
                ax.set_ylabel(ylabel)
                cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(channel, extra), year=C.YEAR)
                handles = [Line2D([], [], color=col, lw=1.5, marker="o", ms=6, label=label)
                           for (k, label), col in zip(widths, cols) if data[k] and not (stat == "sigfit_ratio" and k == ref_k)]
                handles.append(Line2D([], [], marker="o", mfc="white", mec="k", color="none", ms=6, label=r"open: $< 90\,\%$ of fits pass"))
                # one column: a two-column legend overlaps the label block horizontally, and fit_header then
                # stacks them and blows up the y range
                ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE_LEGEND, ncol=1, handlelength=1.6)
                fit_header(ax, ignore_above=(stat != "pull_mean"))
                P.save(fig, out / f"{stem}_{cand}_{tr}_{stat}")
            L.append("")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--pieces", nargs="+", default=list(PIECES) + ["tails", "window_sizes"])
    ap.add_argument("--truths", nargs="+", default=TRUTHS)
    ap.add_argument("--out", default="13_pieces")
    ap.add_argument("--wsize-lo", type=float, nargs="+", default=[WSIZE_LO],
                    help="window_sizes: the fit floor(s) to draw, 1200 (the talk, k<k>_lo1200) and / or 1000 (k<k>)")
    ap.add_argument("--toys-tag", default="", help="read the window folders with this suffix (6_windows.py --tag, e.g. _100k) and add it to the figure stems and to PIECES<tag>.md")
    ap.add_argument("--wsize-k", type=float, nargs="+", default=None,
                    help="window_sizes: draw only these half-widths (100 = full range, always needed), e.g. 3 5 100 for the talk's body; the files get _k3-5-100")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    from wrplotter.cli_utils import setup_logging
    setup_logging(args.verbose)
    global TOYS_TAG
    TOYS_TAG = args.toys_tag
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.lines import Line2D
    import plots as P
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    hep.style.use("CMS")
    per_cell = "500 per cell" if not TOYS_TAG else f"folders {TOYS_TAG}, the toy count in their CSVs"
    L = ["# The pieces: one change at a time against the +-5 sigma window with the exponential", "",
         f"Toys of the window study ({per_cell}, lo_ht; k100_lo1200 = the full range from 1200 GeV), the Gaussian fitted "
         f"to the pulls at zero injection; a configuration's cells count while >= {PASS_FRAC:.0%} of their fits pass the checks. "
         f"sigma_fit = the Asimov fit error.", ""]
    for channel in args.channels:
        out = C.RESULTS / channel / args.out
        out.mkdir(parents=True, exist_ok=True)
        for piece in args.pieces:
            if piece == "tails":
                tails_piece(channel, out, L)
                continue
            if piece == "window_sizes":
                for lo in args.wsize_lo:
                    window_sizes_piece(channel, out, L, truths=args.truths, lo=lo, ks=args.wsize_k)
                continue
            spec = PIECES[piece]
            data = {(cfg[0], tr): load(channel, cfg[2], tr) for cfg in spec["configs"] for tr in args.truths}
            cand_of = {cfg[0]: cfg[1] for cfg in spec["configs"]}
            label_of = {cfg[0]: cfg[3] for cfg in spec["configs"]}
            ref = spec["ref"]
            extra = f"{spec['block']}\nspurious signal"
            L += [f"## {channel}, {piece}", "",
                  f"| configuration | truth | sparse from | masses with abs(mu) > 0.5 (cells with >= 90 % passing) | > 0.3 | max abs(mu) | "
                  f"sigma_fit / {label_of[ref].replace('$', '').replace(chr(92) + 'pm ', '+-')} at {', '.join(f'{m_ / 1000:g}' for m_ in REF_MASSES)} TeV |",
                  "|---|---|---|---|---|---|---|"]
            ytop = 4.5 if piece == "expo_range" else 4.0           # the full-range exponential reaches +-2
            figs = [("pull_mean", "pull_gauss_mean", r"Gaussian $\mu$ of the pulls", (-2.5, ytop), 1),
                    ("width", "pull_width", r"Gaussian $\sigma$ of the pulls", (0.0, 3.5), 1),
                    ("sigfit_ratio", "asimov_err", rf"$\sigma_\mathrm{{fit}}$ / $\sigma_\mathrm{{fit}}$({label_of[ref]})", (0.4, 2.4), 1)]
            if piece == "powexp_range":
                # the top leaves the label block and the legend clear of the curves at 1.0 (fit_header only raises)
                figs.append(("pass_fraction", None, "fraction of fits passing the checks", (0.0, 2.6), 2))
            for stat, key, ylabel, ylim, ncol in figs:
                fig, ax = plt.subplots(figsize=(10, 8))
                for cfg_key, cand, folder, label, col in spec["configs"]:
                    if stat == "sigfit_ratio" and cfg_key == ref:
                        continue
                    for tr in args.truths:
                        rows = data[(cfg_key, tr)]
                        if not rows:
                            continue
                        ls, mk = T.TRUTH_LM.get(tr, ("-", "o"))
                        if stat == "pass_fraction":
                            pts = T.pass_fractions(rows, cand)
                            x = np.array([p[0] for p in pts], float)
                            y = np.array([p[1] for p in pts], float)
                            good = y >= PASS_FRAC
                        elif stat == "sigfit_ratio":
                            x, y, good = series(rows, cand, key)
                            rx, ry, rgood = series(data[(ref, tr)], cand_of[ref], key)
                            rsig = dict(zip(rx, ry))
                            rok = dict(zip(rx, rgood))
                            y = np.array([v / rsig[m_] if m_ in rsig and rsig[m_] > 0 else np.nan for m_, v in zip(x, y)])
                            good = good & np.array([rok.get(m_, False) for m_ in x])   # filled when both pass
                        else:
                            x, y, good = series(rows, cand, key)
                        draw(ax, x, y, good, col, ls, mk)
                        if stat == "pull_mean":
                            sparse = T.first_sparse(rows, cand, PASS_FRAC)
                            f5 = [f"{m_:.0f}" for m_, v, g in zip(x, y, good) if g and abs(v) > C.BIAS_PASS]
                            f3 = [f"{m_:.0f}" for m_, v, g in zip(x, y, good) if g and abs(v) > C.BIAS_TARGET]
                            mx = float(np.nanmax(np.abs(y[good]))) if good.any() else float("nan")
                            sx, sy, _ = series(rows, cand, "asimov_err")
                            rx, ry, _ = series(data[(ref, tr)], cand_of[ref], "asimov_err")
                            sig, rsig = dict(zip(sx, sy)), dict(zip(rx, ry))
                            ratios = " / ".join(f"{sig[m_] / rsig[m_]:.2f}" if m_ in sig and m_ in rsig else "-" for m_ in REF_MASSES)
                            L.append(f"| {label.replace('$', '').replace(chr(92) + 'pm ', '+-')} | {tr} | {fmt(sparse, 0) if sparse else '-'} | "
                                     f"{', '.join(f5) or '-'} | {', '.join(f3) or '-'} | {mx:.2f} | {ratios} |")
                if stat == "pull_mean":
                    for yv, c_ in [(C.BIAS_TARGET, "0.5"), (C.BIAS_PASS, "0.2")]:
                        ax.axhline(yv, color=c_, lw=0.9, ls="--")
                        ax.axhline(-yv, color=c_, lw=0.9, ls="--")
                    ax.axhline(0, color="k", lw=0.8)
                elif stat in ("width", "sigfit_ratio"):
                    ax.axhline(1.0, color="k", lw=0.8)
                elif stat == "pass_fraction":
                    ax.axhline(PASS_FRAC, color="0.2", lw=0.9, ls="--")
                ax.set_ylim(*ylim)
                ax.set_xlabel(r"$m_{W_R}$ [GeV]")
                ax.set_ylabel(ylabel)
                cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(channel, extra), year=C.YEAR)
                handles = legend_handles(spec["configs"], args.truths, skip=ref if stat == "sigfit_ratio" else None)
                if stat == "pass_fraction":
                    handles.append(Line2D([], [], color="0.2", lw=0.9, ls="--", label=r"$90\,\%$"))
                else:
                    handles.append(Line2D([], [], marker="o", mfc="white", mec="k", color="none", ms=6, label=r"open: $< 90\,\%$ of fits pass"))
                ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE_LEGEND, ncol=ncol, handlelength=1.6)
                fit_header(ax, ignore_above=(stat != "pull_mean"))
                P.save(fig, out / f"{piece}_{stat}")
            L.append("")
        (out / "summary.md").write_text("\n".join(L) + "\n")
        logger.info("wrote %s", out)
    (C.RESULTS / f"PIECES{TOYS_TAG}.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
