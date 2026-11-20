#!/usr/bin/env python3
"""The fitted background parameters of every window, as a table.

Per channel, background and signal mass m_WR: the window [max(mu - k sigma, lo), min(mu + k sigma, 8000)] GeV of
6_windows.py, and each function fitted in it to the expected spectrum (the background itself, no fluctuation; a
Poisson likelihood, so the errors are those of a dataset of this luminosity). One row per (background, function,
mass):

  mu, sigma, lo, hi      the signal mean and width at this mass, and the window
  n_window, n_core       expected events in the window and within +-2 sigma of the signal mean
  formula, pivot         the function in events per bin of bin_width GeV (ROOT TF1 syntax, x in GeV)
  p<i>, p<i>_err         its parameters from the background-only fit, their errors
  cov<i><j>              the covariance of p<i> and p<j>
  outcome                pass, or the first check the fit fails
  ..._sb                 the same parameters from the signal + background fit, with N_sig and its error

The backgrounds are the stored ones of 3_truths.py: `raw` (3_truths/) and `kde9`, the nine-process RooKeysPdf
(3_truths_kde9/truth_kde); one that is not on disk is skipped.

Writes results*/<channel>/26_window_params/window_params_k<k>_lo<lo>.csv.

  cd WR_Plotter && FIT_DATASET=run3 python background_fit/26_window_params.py [--k 5 --lo 1200 --candidates expo powexp]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))
import config as C                                                       # noqa: E402
import published_functions as pf                                         # noqa: E402
from sbfit import FIT_HI, SBModel, bkg_only, nsig_limits, pivot, signal_pars   # noqa: E402
from spectrum import centers, rebin                                      # noqa: E402
from truth_sets import load_truth                                        # noqa: E402
from util import write_csv                                               # noqa: E402

logger = logging.getLogger("window_params")
# background name -> (folder under results*/<channel>/, truth file name)
BACKGROUNDS = {"raw": ("3_truths", "raw"), "kde9": ("3_truths_kde9", "kde")}
MASSES = list(range(1200, 4001, 200)) + list(range(4400, 8001, 400))
# the checks of a fit, in the order they are judged, with the name of failing each
CHECKS = [("valid", "no valid minimum"), ("cov", "covariance incomplete"), ("limit", "parameter at a bound"),
          ("mono", "not falling")]


def outcome(fit):
    if fit is None:
        return "no fit"
    return next((text for key, text in CHECKS if not fit["checks"][key]), "pass")


def rows_of(channel, variant, background, edges, truth, variances, masses, candidates, k, lo):
    c = centers(edges)
    rows = []
    for m in masses:
        mu, sigma = signal_pars(channel, m)
        w_lo, w_hi = max(mu - k * sigma, lo), min(mu + k * sigma, FIT_HI)
        inwin = (c >= w_lo) & (c <= w_hi)
        core = (c >= mu - C.SIG_CORE_K * sigma) & (c <= mu + C.SIG_CORE_K * sigma)
        n_core = float(truth[core].sum())
        parents = {}
        for name in sorted(candidates, key=lambda n: pf.FUNCS[n][0]):
            m0 = pivot(name, mu)
            row = {"channel": channel, "dataset": C.DATASET, "lumi": C.LUMI, "variant": variant,
                   "background": background, "function": name, "k": k, "mass": m, "mu": mu, "sigma": sigma,
                   "lo": w_lo, "hi": w_hi, "n_window": float(truth[inwin].sum()), "n_core": n_core,
                   "formula": pf.formula(name, m0, C.SQRTS), "pivot": m0, "bin_width": C.BIN_WIDTH,
                   "par_names": " ".join(pf.FUNCS[name][2])}
            fb = bkg_only(name, edges, truth, variances, w_lo, w_hi, parents, m0=m0)
            parents[name] = fb
            row["outcome"] = outcome(fb)
            if fb is not None:
                npar = fb["npar"]
                for i in range(npar):
                    row[f"p{i}"], row[f"p{i}_err"] = float(fb["params"][i]), float(fb["perr"][i])
                for i in range(npar):
                    for j in range(i + 1, npar):
                        row[f"cov{i}{j}"] = float(fb["cov"][i][j])
                sb = SBModel(name, mu, sigma, w_lo, w_hi).fit(edges, truth, fb["params"], 0.0, nsig_limits(n_core))
                row["outcome_sb"] = outcome(sb)
                if sb is not None:
                    row["nsig_sb"], row["nsig_sb_err"] = sb["nsig"], sb["nsig_err"]
                    for i in range(npar):
                        row[f"p{i}_sb"] = float(sb["bkg"][i])
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--variant", default="lo_ht", help="the DY sample of the background")
    ap.add_argument("--backgrounds", nargs="+", default=list(BACKGROUNDS), choices=list(BACKGROUNDS))
    ap.add_argument("--candidates", nargs="+", default=["expo", "powexp"], help="the functions")
    ap.add_argument("--k", type=float, default=5.0, help="window half-width in units of sigma")
    ap.add_argument("--lo", type=float, default=1200.0, help="the floor the windows are clipped at, GeV")
    ap.add_argument("--masses", nargs="+", type=float, default=MASSES)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    from wrplotter.cli_utils import setup_logging
    setup_logging(args.verbose)
    for channel in args.channels:
        rows = []
        for background in args.backgrounds:
            folder, truth_name = BACKGROUNDS[background]
            path = C.RESULTS / channel / folder / args.variant / f"truth_{truth_name}"
            if not path.with_suffix(".npz").exists():
                logger.warning("%s %s: no %s, skipped", channel, background, path.with_suffix(".npz"))
                continue
            edges10, parts, _ = load_truth(path)
            edges, truth, variances = rebin(edges10, parts["sum"], np.maximum(parts["sum"], 0.0))
            rows += rows_of(channel, args.variant, background, edges, truth, variances, args.masses,
                            args.candidates, args.k, args.lo)
        if not rows:
            logger.warning("%s: nothing to write", channel)
            continue
        out = C.RESULTS / channel / "26_window_params"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"window_params_k{args.k:g}_lo{args.lo:.0f}.csv"
        write_csv(path, rows)
        n_pass = sum(r["outcome"] == "pass" for r in rows)
        logger.info("%s: %d rows, %d pass -> %s", channel, len(rows), n_pass, path)


if __name__ == "__main__":
    main()
