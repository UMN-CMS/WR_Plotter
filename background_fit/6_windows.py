#!/usr/bin/env python3
"""Window comparison: the step 5 toys refitted in a window around the signal.

Same truths, candidates, injections and statistics as 4_toys.py (its run_mass worker does the
work), but the S+B fit runs over [max(mu - k sigma, lo), min(mu + k sigma, 8000)] GeV with
every background parameter free: k = 3 is the current window method, k = 5 the wider variant,
k = 100 the full range from lo. lo is FIT_LO (1000 GeV) unless --lo says otherwise; another lo
gets its own folders, k<k>_lo<lo> (k100_lo1200: the full range from 1200 GeV).
The exponential is the form of interest; powlaw, powexp, atlasemu and dilep8 run beside it.
Default truths: the primary DY variants (lo_ht, nlo_mll), all five smoothers.

Outputs results/<channel>/6_windows/k<k>[_lo<lo>]/<variant>/toys_<truth>.csv, nsig_<truth>.npz, plots/,
toys_all.csv and summary.md in the step 5 format; 7_compare_windows.py reads the k<k> ones,
12_variants.py and 13_pieces.py read both.

  cd WR_Plotter && python background_fit/6_windows.py --workers 8
      [--k 2 4 --candidates powexp atlasemu --append]   # more candidates into existing window folders
"""
from __future__ import annotations

import argparse
import importlib
import logging
import multiprocessing as mp
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))
import config as C                                                       # noqa: E402

T = importlib.import_module("4_toys")                                    # the worker and the writers
logger = logging.getLogger("6_windows")
PRIMARY = ["lo_ht", "nlo_mll"]
WINDOW_K = [3.0, 5.0]
WINDOW_CANDIDATES = ["expo", "powlaw", "powexp", "atlasemu", "dilep8"]
FIT_HI = 8000.0


def window(channel, m, k, lo=C.FIT_LO):
    """The fit window of mass m: +-k sigma around the signal mean, clipped to [lo, FIT_HI]."""
    from sbfit import signal_pars
    mu, sigma = signal_pars(channel, m)
    return max(mu - k * sigma, lo), min(mu + k * sigma, FIT_HI)


def kdir(k, lo=C.FIT_LO):
    """k5 for the default fit floor, k5_lo1200 for another one."""
    return f"k{k:g}" + ("" if lo == C.FIT_LO else f"_lo{lo:.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--variants", nargs="+", default=PRIMARY)
    ap.add_argument("--truths", nargs="+", default=C.TRUTH_NAMES)
    ap.add_argument("--candidates", nargs="+", default=WINDOW_CANDIDATES)
    ap.add_argument("--k", nargs="+", type=float, default=WINDOW_K, help="window half-widths in units of sigma (100: the full range)")
    ap.add_argument("--lo", type=float, default=C.FIT_LO,
                    help="the fit floor the windows are clipped at, GeV (default FIT_LO; another value writes k<k>_lo<lo> folders)")
    ap.add_argument("--masses", nargs="+", type=float, default=C.MASSES)
    ap.add_argument("--ntoys", type=int, default=C.NTOYS)
    ap.add_argument("--inject", nargs="+", type=float, default=C.INJECT)
    ap.add_argument("--workers", type=int, default=max(1, min(8, mp.cpu_count() - 1)))
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--summary-only", action="store_true", help="rebuild toys_all.csv and summary.md from the per-truth CSVs")
    ap.add_argument("--plots-only", action="store_true", help="rebuild the plots from the per-truth CSV and npz files")
    ap.add_argument("--plot-candidates", nargs="+", default=None, help="summary figures of these candidates only (4_toys.py)")
    ap.add_argument("--compare-truths", nargs="+", default=None, help="plots-only: overlay these truths per candidate (4_toys.py)")
    ap.add_argument("--truth-dir", default="3_truths", help="results/<channel>/<truth-dir>/ to take the truths from (3_truths_kde9: the nine-process RooKeysPdf)")
    ap.add_argument("--truth-suffix", default="", help="appended to the truth name in the rows and files (kde + 9 = kde9), so truths of the same name from two directories coexist")
    ap.add_argument("--tag", default="", help="suffix of the window folder, e.g. _100k: 6_windows/k5_lo1200_100k, so a high-statistics rerun sits next to the 500-toy grid instead of replacing its cells")
    ap.add_argument("--restat", action="store_true", help="recompute the pull statistics from the npz arrays (4_toys.py --restat), then rebuild")
    ap.add_argument("--append", action="store_true",
                    help="keep the other candidates' rows and N_sig arrays already in results/<ch>/6_windows/k<k>/<variant>/ "
                         "(a new candidate joins the files instead of replacing them)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    from wrplotter.cli_utils import setup_logging
    setup_logging(args.verbose)
    from spectrum import rebin
    from truth_sets import load_truth
    from util import write_csv
    ctx = mp.get_context("spawn")
    for channel in args.channels:
        for k in args.k:
            out = C.RESULTS / channel / "6_windows" / (kdir(k, args.lo) + args.tag)
            (out / "plots").mkdir(parents=True, exist_ok=True)
            full = k >= 100                                     # the window covers the whole range
            tag = (f"\nfull range\nfrom {args.lo:.0f} GeV" if full          # two short lines: one long line widens the label
                   else f"\nwindow $\\pm{k:g}\\sigma$")      # block into the legend, which then drops onto the curves
            title = (f"# Window comparison, {channel} {C.REGION_NAME}: toys from every truth, S+B fit "
                     + (f"over the full range from {args.lo:.0f} GeV" if full else f"in a +-{k:g} sigma window")
                     + " with every candidate")
            range_text = (f"Fit range [{args.lo:.0f}, 8000] GeV" if full else
                          f"Fit range [max(mu - {k:g} sigma, {args.lo:.0f}), min(mu + {k:g} sigma, 8000)] GeV per mass")
            if args.restat:
                T.restat(out)
                args.plots_only = True
            if args.summary_only or args.plots_only:
                all_rows = T.collect_rows(out)
                if args.plots_only:
                    for p in sorted(out.glob("*/toys_*.csv")):
                        variant, truth_name = p.parent.name, p.stem[len("toys_"):]
                        rows = [r for r in all_rows if r["variant"] == variant and r["truth"] == truth_name]
                        z = np.load(p.parent / f"nsig_{truth_name}.npz")
                        T.plot_truth_summary(channel, variant, truth_name, rows, {kk: z[kk] for kk in z.files},
                                             out / "plots" / variant, tag=tag, only=args.plot_candidates)
                    T.compare_truths(all_rows, args, out, tag=tag)
                write_csv(out / "toys_all.csv", all_rows)
                T.write_summary(channel, all_rows, out / "summary.md", title=title, range_text=range_text)
                continue
            for variant in args.variants:
                vdir = C.RESULTS / channel / args.truth_dir / variant
                for truth_file in args.truths:
                    tpath = vdir / f"truth_{truth_file}"
                    truth_name = truth_file + args.truth_suffix          # the name in the rows, files and figures
                    if not tpath.with_suffix(".npz").exists():
                        logger.warning("%s %s %s: no truth file, skipped", channel, variant, truth_file)
                        continue
                    edges10, parts, _ = load_truth(tpath)
                    e100, t100, var100 = rebin(edges10, parts["sum"], np.maximum(parts["sum"], 0.0))
                    jobs = []
                    for m in args.masses:
                        lo, hi = window(channel, m, k, args.lo)
                        jobs.append({"channel": channel, "variant": variant, "truth_name": truth_name, "edges": e100,
                                     "truth": t100, "variances": var100, "mass": m, "lo": lo, "hi": hi,
                                     "candidates": args.candidates, "ntoys": args.ntoys, "inject": args.inject,
                                     "seed": C.SEED + 1000 * int(m)
                                     + zlib.crc32(f"{channel}/{variant}/{truth_name}".encode()) % 997})
                    with ctx.Pool(args.workers) as pool:
                        results = pool.map(T.run_mass, jobs)
                    rows, arrays = [], {}
                    for r, a in results:
                        rows += r
                        arrays.update(a)
                    (out / variant).mkdir(parents=True, exist_ok=True)
                    if args.append:
                        rows, arrays = T.merge_previous(out / variant, truth_name, args.candidates, rows, arrays)
                    write_csv(out / variant / f"toys_{truth_name}.csv", rows)
                    np.savez_compressed(out / variant / f"nsig_{truth_name}.npz", **arrays)
                    if not args.no_plots:
                        T.plot_truth_summary(channel, variant, truth_name, rows, arrays, out / "plots" / variant, tag=tag, only=args.plot_candidates)
                    ok = [r for r in rows if r.get("status") == "ok" and r["inject"] == 0.0 and r.get("bias_median") is not None]
                    for cand in args.candidates:
                        cs = [r for r in ok if r["candidate"] == cand]
                        if cs:
                            worst = max(cs, key=lambda r: abs(r["bias_median"]))
                            logger.info("%s k=%g %s %-14s %-9s max|S_spur| %.2f at %4.0f; >0.3 at %s", channel, k, variant,
                                        truth_name, cand, abs(worst["bias_median"]), worst["mass"],
                                        ",".join(f"{r['mass']:.0f}" for r in cs if abs(r["bias_median"]) > C.BIAS_TARGET) or "-")
            all_rows = T.collect_rows(out)
            write_csv(out / "toys_all.csv", all_rows)
            T.write_summary(channel, all_rows, out / "summary.md", title=title, range_text=range_text)
            logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
