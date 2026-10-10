#!/usr/bin/env python3
"""Sum of the per-process RooKeysPdf curves vs ONE RooKeysPdf of the summed MC.

Reads smoothed_kde.root / .json (smooth_background.py, any component set),
sums the raw per-process histograms, runs RooKeysPdf once on that sum with
exactly the settings recorded in the json, and compares three things at the
card binning: the binned total MC, the sum of the per-process curves, and the
single curve.

  <output-dir>/sum_vs_single.{png,pdf}   MC + both curves; MC / curve for both
  <output-dir>/sum_vs_single.csv         yields per mass band
  <input dir>/single_kde.root            the single curve as a TH1D (bkg_single)

  PYTHONPATH=/opt/homebrew/lib/root ../../../.venv-fitting/bin/python \\
      compare_sum_vs_single.py --input run3/ee_resolved_split8/smoothed_kde.root
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ROOT                                                              # noqa: E402
import smooth_lib as sl                                                  # noqa: E402
from smooth_lib import centers_of, rebin                                 # noqa: E402
from smooth_background import XMIN, _cms, _save                          # noqa: E402
from plot_step1_kde import _figure, kde_param_text                        # noqa: E402
from wrplotter.cli_utils import setup_logging                            # noqa: E402
from wrplotter.config import load_lumi                                   # noqa: E402
from wrplotter.plotting_helpers import custom_log_formatter              # noqa: E402

logger = logging.getLogger("compare_sum_vs_single")

SUM_COLOR, SINGLE_COLOR = "black", "#2ca02c"
BANDS = [(800, 1200), (1200, 2000), (2000, 3000), (3000, 3500),
         (3500, 4500), (4500, 6000), (800, 8000)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path,
                   default=HERE / "run2" / "ee_resolved_split8" / "smoothed_kde.root")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="default: <input dir>/plots/step1")
    p.add_argument("--bw", type=float, default=50.0)
    p.add_argument("--xmax", type=float, default=6000.0)
    p.add_argument("--ylim", type=float, nargs=2, default=(2e-5, 2e4))
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    out = args.output_dir or args.input.parent / "plots" / "step1"
    out.mkdir(parents=True, exist_ok=True)

    with open(args.input.with_suffix(".json")) as fh:
        meta = json.load(fh)
    era = meta.get("era", sl.ERA)
    info = load_lumi(era)
    lumi, com = meta.get("lumi", info["lumi"]), info.get("com", 13.0)
    kw = meta.get("kwargs", {})
    kw = {k: kw[k] for k in ("var_kind", "mirror", "rho", "resample") if k in kw}

    f = uproot.open(args.input)
    edges = f["bkg_raw"].axes[0].edges()
    raw, var = f["bkg_raw"].values(), f["bkg_raw"].variances()
    sum_curve = f["bkg_smooth"].values()
    comps = [k.split(";")[0] for k in f.keys()
             if not k.split(";")[0].endswith("_raw") and not k.startswith("bkg_")]
    logger.info("%d per-process curves in %s; single RooKeysPdf on their sum with %s",
                len(comps), args.input.name, kw)

    if meta.get("kwargs", {}).get("entries") == "events":
        # one RooKeysPdf of ALL the MC events of all components
        from wrplotter.paths import input_dirs_for_era, repo_root
        dirs, _ = input_dirs_for_era(era, repo_root(), meta["bkg_dir"])
        ms, ws = zip(*(sl.load_component_events(dirs, sl.COMPONENTS_SPLIT.get(c, [c]))
                       for c in comps))
        kw_ev = {k: v for k, v in kw.items() if k != "resample"}
        single, diag = sl.smooth_kde_events(edges, np.concatenate(ms), np.concatenate(ws), **kw_ev)
    else:
        single, diag = sl.smooth_kde(edges, raw, var, **kw)
    logger.info("single KDE: %d entries, %d negative bins dropped, total %.2f",
                diag["n_entries"], diag["n_negative_dropped"], diag["total_raw"])

    # the single curve as a histogram next to the input
    fout = ROOT.TFile(str(args.input.parent / "single_kde.root"), "RECREATE")
    e = np.ascontiguousarray(edges, np.float64)
    for name, vals, errs in (("bkg_single", single, var), ("bkg_raw", raw, var)):
        h = ROOT.TH1D(name, ";m [GeV];events", len(vals), e)
        for i, (x, s2) in enumerate(zip(vals, errs), start=1):
            h.SetBinContent(i, float(x))
            h.SetBinError(i, float(np.sqrt(max(s2, 0.0))))
        h.Write()
    fout.Close()

    # yields per band
    c = centers_of(edges)
    rows = []
    print(f"{'band [GeV]':>12s} {'MC':>9s} {'± stat':>7s} {'sum of N':>9s} "
          f"{'single':>9s} {'sumN/MC':>8s} {'single/MC':>10s}")
    for lo, hi in BANDS:
        m = (c >= lo) & (c < hi)
        r, er = float(raw[m].sum()), float(np.sqrt(var[m].sum()))
        a, b = float(sum_curve[m].sum()), float(single[m].sum())
        rows.append({"lo": lo, "hi": hi, "mc": round(r, 4), "mc_err": round(er, 4),
                     "sum_of_curves": round(a, 4), "single_curve": round(b, 4),
                     "sum_over_mc": round(a / r, 3) if r > 0 else "",
                     "single_over_mc": round(b / r, 3) if r > 0 else ""})
        print(f"{lo:>5d}-{hi:<6d} {r:9.3f} {er:7.3f} {a:9.3f} {b:9.3f} "
              f"{a / r if r > 0 else float('nan'):8.2f} {b / r if r > 0 else float('nan'):10.2f}")
    with open(out / "sum_vs_single.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # the plot, at the card binning
    fac = int(round(args.bw / sl.NATIVE_BW))
    e5, v5, var5 = rebin(edges, raw, var, fac)
    _, a5, _ = rebin(edges, sum_curve, np.zeros_like(raw), fac)
    _, b5, _ = rebin(edges, single, np.zeros_like(raw), fac)
    c5 = centers_of(e5)
    vis = (c5 >= XMIN) & (c5 <= args.xmax)
    pos = vis & (v5 > 0)
    fig, (ax, axr) = _figure()
    ax.errorbar(c5[pos], v5[pos], yerr=np.sqrt(var5[pos]), fmt="o", color="black",
                markersize=4, linewidth=1.0, zorder=6, label="MC (binned) $\\pm$ MC stat")
    ok = vis & (a5 > 0)
    ax.plot(c5[ok], a5[ok], color=SUM_COLOR, linewidth=2.8, zorder=5,
            label=f"sum of the {len(comps)} per-process curves")
    ok = vis & (b5 > 0)
    ax.plot(c5[ok], b5[ok], color=SINGLE_COLOR, linewidth=2.6, linestyle="--",
            zorder=5, label="one RooKeysPdf of the summed MC")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylim(*args.ylim)
    ax.set_xlim(XMIN, args.xmax)
    ax.set_ylabel(f"Events / {args.bw:.0f} GeV")
    ax.text(0.04, 0.95, f"Total background\nee resolved SR, {era}",
            transform=ax.transAxes, fontsize=14, va="top")
    ax.text(0.04, 0.04, kde_param_text(diag, XMIN, float(edges[-1]), diag["total_raw"])
            .replace("RooKeysPdf:", "single RooKeysPdf:"),
            transform=ax.transAxes, fontsize=10.5, va="bottom", ha="left",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.6",
                      alpha=1.0), zorder=10)
    ax.legend(loc="upper right", fontsize=12)
    for curve, color, mk, lab in ((a5, SUM_COLOR, "o", "MC / sum of curves"),
                                  (b5, SINGLE_COLOR, "s", "MC / single curve")):
        okr = vis & (v5 > 0) & (curve > 0)
        axr.errorbar(c5[okr], v5[okr] / curve[okr], yerr=np.sqrt(var5[okr]) / curve[okr],
                     fmt=mk, color=color, markersize=3, linewidth=0.8,
                     markerfacecolor="white" if mk == "s" else color, label=lab)
    axr.axhline(1.0, color="0.4", linewidth=1.2)
    axr.set_ylim(0.0, 2.5)
    axr.set_xlim(XMIN, args.xmax)
    axr.set_ylabel("MC / curve")
    axr.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
    axr.legend(loc="upper left", fontsize=10, ncol=2)
    _cms(ax, lumi, com)
    _save(fig, out / "sum_vs_single")


if __name__ == "__main__":
    main()
