#!/usr/bin/env python3
"""Legacy (converted HNWRAnalyzer) vs new-framework (WrCoffea) 2018 background
histograms, per component, in the ee resolved SR.

The production run2 limit uses the legacy set (20260714_run2_bkgs) as its
observation; any per-process smoothing / anchor work has to come from a
WrCoffea rerun. This script measures how far apart the two are: SR total,
effective MC entries, the Poisson-ML expo slope on [1000, 3500], and the
yield in coarse mass bands.

  PYTHONPATH=/opt/homebrew/lib/root ../../../.venv-fitting/bin/python compare_frameworks.py
  -> prints the table and writes run2/ee_resolved/framework_comparison.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import smooth_lib as sl                                                  # noqa: E402
from wrplotter.cli_utils import setup_logging                            # noqa: E402
from wrplotter.paths import repo_root                                    # noqa: E402

logger = logging.getLogger("compare_frameworks")

# new_oct1 = the per-process rerun with the committed analyzer (DY Z-pT
# reweight + reshape wired in on Sep 7); new_sep (files of Sep 6) predates it.
DEFAULT_SETS = ["legacy=20260714_run2_bkgs", "new_oct1=20261001_split_bkgs",
                "new_sep=dyreshape_20260907",
                "new_mar_nlo=20260313_nlo_dy", "new_mar_loinc=20260313_lo_inc_dy"]
BANDS = [(800, 1200), (1200, 2000), (2000, 3000), (3000, 3500),
         (3500, 4500), (4500, 6000)]
FIT_RANGE = (1000.0, 3500.0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sets", nargs="+", default=DEFAULT_SETS,
                   help="label=<subdir of rootfiles/RunII/2018/RunIISummer20UL18>")
    p.add_argument("--components", nargs="+",
                   default=["tt_tW", "DYJets", "Nonprompt", "Other"])
    p.add_argument("--output", type=Path,
                   default=HERE / "run2" / "ee_resolved" / "framework_comparison.csv")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    base = repo_root() / "rootfiles" / "RunII" / "2018" / "RunIISummer20UL18"
    key = f"{sl.REGION}/{sl.VARIABLE}_{sl.REGION}"
    sets = dict(s.split("=", 1) for s in args.sets)

    rows = []
    for comp in args.components:
        print(f"\n== {comp}")
        print(f"{'set':14s} {'total':>8s} {'Neff':>6s} {'slope':>15s}  "
              + " ".join(f"{lo}-{hi}".rjust(9) for lo, hi in BANDS))
        for label, sub in sets.items():
            f = base / sub / f"WRAnalyzer_{comp}.root"
            try:
                h = uproot.open(f)[key]
            except Exception:
                print(f"{label:14s}  (no SR histogram)")
                continue
            e, v, var = h.axes[0].edges(), h.values(), h.variances()
            c = sl.centers_of(e)
            m = c >= sl.THRESHOLD
            tot = float(v[m].sum())
            neff = tot * tot / float(var[m].sum()) if var[m].sum() > 0 else 0.0
            e5, v5, var5 = sl.rebin(e, v, var, 5)
            fit = sl.fit_component("expo", e5, v5, var5, *FIT_RANGE, 2000.0, "poisson")
            b, berr = (fit["params"][1], fit["perr"][1]) if fit else (np.nan, np.nan)
            bands = [float(v[(c >= lo) & (c < hi)].sum()) for lo, hi in BANDS]
            print(f"{label:14s} {tot:8.2f} {neff:6.0f} {b:+7.3f}±{berr:.3f}  "
                  + " ".join(f"{x:9.3f}" for x in bands))
            rows.append({"component": comp, "set": label, "dir": sub,
                         "total": round(tot, 3), "neff": round(neff, 1),
                         "slope": round(b, 4), "slope_err": round(berr, 4),
                         **{f"y_{lo}_{hi}": round(x, 4)
                            for (lo, hi), x in zip(BANDS, bands)}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
