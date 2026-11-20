#!/usr/bin/env python3
"""The fit floor: what starting the fit below 1200 GeV does, and which mass points it touches.

Four figures per channel in results/<channel>/17_fit_floor/ (2024, 109.7 fb-1; the 1000 GeV grids exist
only there):

  floor_fit_<cand>          the summed raw MC (LO HT-binned DY) from the 800 GeV selection cut, the candidate
                            fitted over [start, last MC event] for starts 800, 1000 and 1200 GeV; below, the
                            pulls (MC - fit) / sigma_MC of each fit from its own start. The legend carries the
                            Pearson chi2/ndf of the threshold scan (1_spectrum/summary.md).
  floor_windows             the +-5 sigma window and the +-2 sigma core of every grid mass against the three
                            floors: the masses whose window is clipped by a floor are the ones the floor acts on.
  floor_bias_<cand>         the Gaussian mu of the pulls against m_WR for the +-5 sigma window and the full range
                            with the floor at 1000 and at 1200 GeV, raw-MC truth (the +-5 sigma window on the
                            RooKeysPdf too); 500 toys per cell (6_windows/k5, k5_lo1200, 4_toys, 4_toys_lo1200).
                            The band marks the masses whose +-5 sigma window reaches below 1200 GeV.
  floor_sigfit_<cand>       sigma_fit(1200 floor) / sigma_fit(1000 floor) against m_WR, the +-5 sigma window and
                            the full range: the cost of the higher floor, per mass.

  cd WR_Plotter && python background_fit/17_fit_floor.py [--channels ee mumu]
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

logger = logging.getLogger("17_fit_floor")
STARTS = [(800.0, "#e42536"), (1000.0, "#f89c20"), (1200.0, "#5790fc")]   # the selection cut, the study's floor, the talk's
FLOORS = [(1000.0, "#f89c20", "6_windows/k5", "4_toys"), (1200.0, "#5790fc", "6_windows/k5_lo1200", "4_toys_lo1200")]
CAND_LABEL = {"powexp": r"Power $\times$ expo", "expo": "Expo"}
ZOOM = 3000.0                        # the spectrum figure stops here: the floor acts on the first bins
K = 5.0


def load_rows(channel, folder, truth, variant="lo_ht"):
    p = C.RESULTS / channel / folder / variant / f"toys_{truth}.csv"
    return [T._type_row(r) for r in read_csv(p)] if p.exists() else []


def series(rows, cand, key):
    x, y = T._series(rows, cand, 0.0, key)
    frac = dict(T.pass_fractions(rows, cand))
    x, y = np.array(x, float), np.array(y, float)
    return x, y, np.array([frac.get(m_, 0.0) >= T.PASS_FRAC for m_ in x], bool)


def draw(ax, x, y, good, col, ls, mk):
    ax.plot(x, y, lw=1.5, color=col, ls=ls)
    ax.plot(x[good], y[good], mk, ms=6, color=col)
    ax.plot(x[~good], y[~good], mk, ms=6, mfc="white", mec=col)


def clipped_masses(channel, floor, masses):
    from sbfit import signal_pars
    return [m for m in masses if signal_pars(channel, m)[0] - K * signal_pars(channel, m)[1] < floor]


def fit_floor_figure(channel, cand, out, L):
    """The spectrum at the threshold with the candidate fitted from 800, 1000 and 1200 GeV."""
    import plots as P
    import published_functions as pf
    from fits import bin_weights, fit_all
    from spectrum import centers, last_populated, load_components, rebin
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND
    edges, values, variances = rebin(*load_components(channel, "lo_ht")["sum"])
    c = centers(edges)
    hi = last_populated(edges, values)
    w_arr = bin_weights(edges, values, variances)
    err = np.sqrt(np.maximum(variances, 0.0))
    sel = (c >= C.THRESHOLD) & (c <= ZOOM)
    fig, (ax, axr) = P.spectrum_figure()
    ax.errorbar(c[sel], values[sel], yerr=err[sel], fmt="o", ms=4, color="k", label="Simulation", zorder=5)
    L += [f"## {channel}, {cand}: the fit from three starts (Poisson fit to the last MC event, {hi:.0f} GeV)", "",
          "| start | Pearson chi2/ndf | pulls in the first three bins of the fit | pulls at 1200-2000 GeV (max abs) |", "|---|---|---|---|"]
    for start, col in STARTS:
        Lf, Pf, _ = fit_all(edges, values, variances, start, hi, C.PIVOT, menu=[cand], w_arr=w_arr)
        f, p = Lf.get(cand), Pf.get(cand)
        if f is None:
            continue
        chi2 = (f"{p['chi2']:.0f}/{p.get('ndf', p.get('nbins', 0) - f['npar'])}" if p else "-")
        grid = np.linspace(start, ZOOM, 400)
        y = np.array([f["tf1"].Eval(x) for x in grid])
        ax.plot(grid, np.maximum(y, P.YMIN), color=col, lw=1.8, label=f"fit from {start:.0f} GeV, $\\chi^2$/ndf {chi2}")
        pull = np.where(err > 0, (values - f["model"]) / np.where(err > 0, err, 1.0), np.nan)
        m = (c >= start) & (c <= ZOOM)
        axr.stairs(pull[m], np.append(edges[:-1][m], edges[1:][m][-1]), color=col, lw=1.4)
        first = pull[(c >= start)][:3]
        mid = pull[(c >= 1200.0) & (c <= 2000.0)]
        L.append(f"| {start:.0f} | {chi2} | {', '.join(f'{v:+.1f}' for v in first)} | {np.nanmax(np.abs(mid)):.1f} |")
    for start, col in STARTS[1:]:
        ax.axvline(start, color=col, lw=0.9, ls=":")
        axr.axvline(start, color=col, lw=0.9, ls=":")
    for yv in (-2, 2):
        axr.axhline(yv, color="0.5", lw=0.6, ls="--")
    P.finish(ax, axr, channel=channel, xmax=ZOOM, top=float(values[sel].max()),
             extra=f"{C.DY_SHORT['lo_ht']}\nTotal background\n{CAND_LABEL.get(cand, cand)}",
             ratio_label=r"(MC $-$ fit) / $\sigma_{\rm MC}$", ratio_lim=(-4, 4), ratio_ref=0.0, legend_loc="lower left", ymin=1e-2)
    P.save(fig, out / f"floor_fit_{cand}")
    L.append("")


def windows_figure(channel, out, L):
    """Every grid mass's window against the floors."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import plots as P
    from sbfit import signal_pars
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    masses = [1200.0] + [float(m) for m in C.MASSES if m <= 3400]
    fig, ax = plt.subplots(figsize=(10, 8))
    L += [f"## {channel}: the windows against the floors", "",
          "| m_WR | mu | sigma | +-5 sigma window | +-2 sigma core | clipped at 1000 | clipped at 1200 | core below 1200 |", "|---|---|---|---|---|---|---|---|"]
    for m in masses:
        mu, s = signal_pars(channel, m)
        lo5, hi5, lo2, hi2 = mu - K * s, mu + K * s, mu - 2 * s, mu + 2 * s
        col = "#e42536" if lo5 < 1000.0 else ("#f89c20" if lo5 < 1200.0 else "#717581")
        ax.plot([m, m], [lo5, hi5], color=col, lw=2.2, solid_capstyle="butt")
        ax.plot([m, m], [lo2, hi2], color=col, lw=7, alpha=0.45, solid_capstyle="butt")
        ax.plot(m, mu, "o", color=col, ms=5)
        L.append(f"| {m:.0f} | {mu:.0f} | {s:.0f} | [{lo5:.0f}, {hi5:.0f}] | [{lo2:.0f}, {hi2:.0f}] | "
                 f"{'yes' if lo5 < 1000 else '-'} | {'yes' if lo5 < 1200 else '-'} | {'yes' if lo2 < 1200 else '-'} |")
    for yv, col, lab in [(C.THRESHOLD, "k", "selection cut 800 GeV"), (1000.0, "#f89c20", "floor 1000 GeV"), (1200.0, "#5790fc", "floor 1200 GeV")]:
        ax.axhline(yv, color=col, lw=1.1, ls="--" if yv > C.THRESHOLD else "-")
    ax.set_xlim(1050, 3550)
    ax.set_ylim(400, 5000)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"$m_{\ell\ell jj}$ window [GeV]")
    cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(channel, r"$\pm 5\sigma$ window" + "\n" + r"$\pm 2\sigma$ core (thick)"), year=C.YEAR)
    handles = [Line2D([], [], color="k", lw=1.1, label="selection cut 800 GeV"),
               Line2D([], [], color="#f89c20", lw=1.1, ls="--", label="floor 1000 GeV"),
               Line2D([], [], color="#5790fc", lw=1.1, ls="--", label="floor 1200 GeV"),
               Line2D([], [], color="#e42536", lw=2.2, label="window reaches below 1000"),
               Line2D([], [], color="#f89c20", lw=2.2, label="window reaches below 1200"),
               Line2D([], [], color="#717581", lw=2.2, label="window clear of both floors")]
    ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE_LEGEND, handlelength=1.6)
    fit_header(ax, ignore_above=True)
    P.save(fig, out / "floor_windows")
    L.append("")


def bias_figure(channel, cand, out, L):
    """The Gaussian mu of the pulls with the floor at 1000 and at 1200 GeV, window and full range."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import plots as P
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    clipped = clipped_masses(channel, 1200.0, [1200.0] + [float(m) for m in C.MASSES])
    for truth, tlabel in [("raw", "Raw MC Bkg"), ("kde9", "RooKeysPdf Bkg")]:
        fig, ax = plt.subplots(figsize=(10, 8))
        L += [f"## {channel}, {cand} on {tlabel}: the floor at 1000 against 1200 GeV", "",
              "| range | floor | mu at 1200 / 1400 / 1600 / 1800 / 2000 / 2200 GeV | cells beyond 0.5 up to 3200 | largest abs(mu) up to 3200 (mass) |", "|---|---|---|---|---|"]
        for floor, col, wfolder, ffolder in FLOORS:
            for rng, folder, ls, mk in [("window", wfolder, "-", "o"), ("full", ffolder, "--", "s")]:
                if rng == "full" and truth != "raw":
                    continue                                        # the full-range grids hold the raw and the tailed truths only
                rows = load_rows(channel, folder, truth)
                if not rows:
                    continue
                x, y, good = series(rows, cand, "pull_gauss_mean")
                keep = x <= 3400
                draw(ax, x[keep], y[keep], good[keep], col, ls, mk)
                at = {m_: v for m_, v in zip(x, y)}
                fl = (x <= C.FLOAT_MAX) & good
                bad = [f"{m_:.0f}" for m_, v, g in zip(x, y, fl) if g and abs(v) > C.BIAS_PASS]
                mx = int(np.argmax(np.where(fl, np.abs(y), -1.0))) if fl.any() else None
                L.append(f"| {rng} | {floor:.0f} | " + " / ".join(fmt(at.get(m_, float('nan'))) for m_ in (1200, 1400, 1600, 1800, 2000, 2200))
                         + f" | {', '.join(bad) or '-'} | {fmt(abs(y[mx]))} ({x[mx]:.0f}) |" if mx is not None else f"| {rng} | {floor:.0f} | - | - | - |")
        if clipped:
            ax.axvspan(min(clipped) - 100, max(clipped) + 100, ymin=0.0, ymax=0.05, color="#5790fc", alpha=0.35, lw=0)
        for yv, c_ in [(C.BIAS_TARGET, "0.5"), (C.BIAS_PASS, "0.2")]:
            ax.axhline(yv, color=c_, lw=0.9, ls="--")
            ax.axhline(-yv, color=c_, lw=0.9, ls="--")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlim(900, 3500)
        ax.set_ylim(-2.5, 2.5)
        ax.set_xlabel(r"$m_{W_R}$ [GeV]")
        ax.set_ylabel(r"Gaussian $\mu$ of the pulls")
        cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(channel, f"{CAND_LABEL.get(cand, cand)}\n{tlabel}\nspurious signal"), year=C.YEAR)
        handles = [Line2D([], [], color=col, lw=1.5, label=f"floor {floor:.0f} GeV") for floor, col, _, _ in FLOORS]
        handles.append(Line2D([], [], color="k", lw=1.5, ls="-", marker="o", ms=6, label=r"$\pm 5\sigma$ window"))
        if truth == "raw":
            handles.append(Line2D([], [], color="k", lw=1.5, ls="--", marker="s", ms=6, label="full range from the floor"))
        handles.append(Line2D([], [], marker="o", mfc="white", mec="k", color="none", ms=6, label=r"open: $< 90\,\%$ of fits pass"))
        if clipped:
            handles.append(plt.Rectangle((0, 0), 1, 1, fc="#5790fc", alpha=0.35, ec="none", label=r"strip: $\pm 5\sigma$ window reaches below 1200 GeV"))
        ax.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE_LEGEND, ncol=2, handlelength=1.6, columnspacing=0.8)
        fit_header(ax, ignore_above=False)
        P.save(fig, out / f"floor_bias_{cand}_{truth}")
        L.append("")


def sigfit_figure(channel, cand, out, L):
    """sigma_fit with the 1200 floor over the 1000 floor, per mass."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import plots as P
    from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
    fig, ax = plt.subplots(figsize=(10, 8))
    L += [f"## {channel}, {cand}: sigma_fit(1200 floor) / sigma_fit(1000 floor), raw MC", "",
          "| range | at 1400 / 1600 / 1800 / 2000 / 2200 / 2600 GeV |", "|---|---|"]
    for rng, idx, ls, mk, col in [("window", 2, "-", "o", "#964a8b"), ("full", 3, "--", "s", "#717581")]:
        a = {r["mass"]: r["asimov_err"] for r in load_rows(channel, FLOORS[0][idx], "raw") if r["candidate"] == cand and r["inject"] == 0.0 and r.get("asimov_err")}
        b = {r["mass"]: r["asimov_err"] for r in load_rows(channel, FLOORS[1][idx], "raw") if r["candidate"] == cand and r["inject"] == 0.0 and r.get("asimov_err")}
        ms = sorted(m_ for m_ in a if m_ in b and 1400 <= m_ <= 3400)
        ratio = np.array([b[m_] / a[m_] for m_ in ms])
        ax.plot(ms, ratio, lw=1.5, ls=ls, marker=mk, ms=6, color=col, label=(r"$\pm 5\sigma$ window" if rng == "window" else "full range from the floor"))
        at = dict(zip(ms, ratio))
        L.append(f"| {rng} | " + " / ".join(fmt(at.get(m_, float('nan'))) for m_ in (1400, 1600, 1800, 2000, 2200, 2600)) + " |")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlim(1300, 3500)
    ax.set_ylim(0.8, 2.0)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel(r"$\sigma_{\rm fit}$(floor 1200) / $\sigma_{\rm fit}$(floor 1000)")
    cms_label(ax, C.LUMI, C.COM, data=False, text=P.region_text(channel, f"{CAND_LABEL.get(cand, cand)}\nRaw MC Bkg"), year=C.YEAR)
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND, handlelength=1.6)
    fit_header(ax, ignore_above=True)
    P.save(fig, out / f"floor_sigfit_{cand}")
    L.append("")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--candidates", nargs="+", default=["powexp", "expo"])
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    from wrplotter.cli_utils import setup_logging
    setup_logging(args.verbose)
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    L = ["# The fit floor: starting below 1200 GeV, and the mass points it touches", "",
         f"Dataset {C.YEAR}, {C.LUMI} fb-1, LO HT-binned DY. Toys: 500 per cell (6_windows/k5 and 4_toys at the 1000 GeV floor, "
         "6_windows/k5_lo1200 and 4_toys_lo1200 at 1200). The spectrum fits are Poisson fits to the summed raw MC up to the last MC event; "
         "the chi2/ndf are the Pearson values of the threshold scan.", ""]
    for channel in args.channels:
        out = C.RESULTS / channel / "17_fit_floor"
        out.mkdir(parents=True, exist_ok=True)
        windows_figure(channel, out, L)
        for cand in args.candidates:
            fit_floor_figure(channel, cand, out, L)
            bias_figure(channel, cand, out, L)
            sigfit_figure(channel, cand, out, L)
        logger.info("%s -> %s", channel, out)
    (C.RESULTS / "FIT_FLOOR.md").write_text("\n".join(L) + "\n")
    print(C.RESULTS / "FIT_FLOOR.md")


if __name__ == "__main__":
    main()
