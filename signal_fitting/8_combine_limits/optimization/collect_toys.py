#!/usr/bin/env python3
"""Stage 10.9, step 6 -- the toy validation table for the winner config.

Reads the FitDiagnostics toy trees (tree_fit_sb) and reports, per (config,
mass, injection):

  n_ok        toys with fit_status >= 0 (out of thrown)
  mean r-hat  the spurious signal (null) / recovered signal (injection), fb
  RMS r-hat   the toy spread -- compare to the asymptotic likelihood sigma
  pull mean   <(r-hat - r_inj)/sigma_r>   (sigma_r = symmetrized MINOS err)
  pull RMS    coverage: ~1 means the quoted error is honest

Pass criteria used in the printout: |null pull mean| < 0.2, pull RMS in
[0.85, 1.15], injection recovery within ~10% of r_inj.

  python collect_toys.py            (LCG_106)
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
RINJ = {("inj", 2000): 0.78, ("inj", 2600): 0.37, ("inj", 3200): 0.19}


def read_toys(path):
    with uproot.open(path) as f:
        t = f["tree_fit_sb"]
        arr = t.arrays(["r", "rLoErr", "rHiErr", "fit_status"], library="np")
    ok = arr["fit_status"] >= 0
    r = arr["r"][ok]
    err = 0.5 * (np.abs(arr["rLoErr"][ok]) + np.abs(arr["rHiErr"][ok]))
    good_err = err > 1e-6
    return r, err, int(ok.sum()), len(arr["r"]), good_err


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path,
                   default=HERE / "results" / "ee_resolved")
    args = p.parse_args()

    rows = []
    for f in sorted(args.results_dir.glob("fitDiagnostics_null_*_m*.root")) + \
            sorted(args.results_dir.glob("fitDiagnostics_inj_*_m*.root")):
        name = f.name.replace("fitDiagnostics_", "").replace(".root", "")
        kind, rest = name.split("_", 1)
        config, mass_s = rest.rsplit("_m", 1)
        rows.append((kind, config, int(mass_s), f))

    out = []
    hdr = (f"{'kind':<6}{'config':<22}{'m':>6}{'n_ok':>7}{'r_inj':>7}"
           f"{'mean r':>9}{'RMS r':>8}{'pull mu':>9}{'pull RMS':>9}")
    print(hdr + "\n" + "-" * len(hdr))
    for kind, config, mass, f in sorted(rows, key=lambda x: (x[0], x[1], x[2])):
        r, err, n_ok, n_tot, good = read_toys(f)
        rinj = 0.0 if kind == "null" else RINJ.get((kind, mass), float("nan"))
        pull = (r[good] - rinj) / err[good]
        out.append({
            "kind": kind, "config": config, "mWR": mass, "r_inj": rinj,
            "n_ok": n_ok, "n_tot": n_tot,
            "mean_r": round(float(r.mean()), 4),
            "rms_r": round(float(r.std()), 4),
            "pull_mean": round(float(pull.mean()), 3),
            "pull_rms": round(float(pull.std()), 3),
        })
        o = out[-1]
        print(f"{kind:<6}{config:<22}{mass:>6}{n_ok:>5}/{n_tot:<4}"
              f"{rinj:>6.2f}{o['mean_r']:>9.3f}{o['rms_r']:>8.3f}"
              f"{o['pull_mean']:>9.3f}{o['pull_rms']:>9.3f}")

    with open(HERE / "toy_validation_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote toy_validation_table.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
