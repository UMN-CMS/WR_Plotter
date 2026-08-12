#!/usr/bin/env python3
"""Stage 9, step 4 -- collect combine results into a CSV and compare with the
homemade (Stage-7) machinery.

Reads results/{tag}/higgsCombine_{fn}.AsymptoticLimits.mH{m}.root, joins the
inputs JSON (rate, xsec, Stage-6 toy moments) and writes
combine_limit_table_{tag}.csv with the limits both as cross sections (fb, the
POI unit) and as event yields (r x rate).

For each point it also evaluates the homemade closed-form CLs band from the
Stage-6 moments -- `--center zero`, since combine's expected band is likewise
centered on the background-only expectation -- and prints the side-by-side
comparison. Expected agreement: exact in the formula, differing by the sigma
convention (combine: Asimov / profile-likelihood error; homemade: toy RMS,
which the Stage-6 pulls show is ~10-25% wider).

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import ROOT

ROOT.gROOT.SetBatch(True)
HERE = Path(__file__).resolve().parent

QUANTS = {0.025: "m2s", 0.16: "m1s", 0.5: "med", 0.84: "p1s", 0.975: "p2s",
          -1.0: "obs"}
BANDS = (-2, -1, 0, 1, 2)
LBL = {-2: "m2s", -1: "m1s", 0: "med", 1: "p1s", 2: "p2s"}


def cls_ul(n_hat, sigma, alpha=0.05):
    """Homemade asymptotic CLs upper limit (same as expected_limit.py)."""
    if not (sigma > 0.0 and math.isfinite(sigma) and math.isfinite(n_hat)):
        return float("nan")
    p = 1.0 - alpha * ROOT.TMath.Freq(n_hat / sigma)
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return n_hat + sigma * ROOT.TMath.NormQuantile(p)


def read_limit_tree(path):
    """{quantile label: r} from one higgsCombine file."""
    f = ROOT.TFile(str(path))
    t = f.Get("limit")
    out = {}
    for e in t:
        key = QUANTS.get(round(float(e.quantileExpected), 3))
        if key:
            out[key] = float(e.limit)
    f.Close()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--channel", default="ee")
    p.add_argument("--topology", default="resolved")
    p.add_argument("--run", default="run3", choices=["run2", "run3"],
                   help="run-label subdir for the default in/out dirs")
    p.add_argument("--input-dir", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args()
    if args.input_dir is None:
        args.input_dir = HERE / args.run / "inputs"
    if args.results_dir is None:
        args.results_dir = HERE / args.run / "results"
    if args.output_dir is None:
        args.output_dir = HERE / args.run
    tag = f"{args.channel}_{args.topology}"

    with open(args.input_dir / f"{tag}.json") as fh:
        meta = json.load(fh)

    rows = []
    for path in sorted((args.results_dir / tag).glob(
            "higgsCombine_*.AsymptoticLimits.mH*.root")):
        fn = path.name.split("_")[1].split(".")[0]
        mass = path.name.split(".mH")[1].split(".")[0]
        m = meta["masses"].get(mass)
        if m is None:
            continue
        lims = read_limit_tree(path)          # r == sigma in fb
        rate = m["rate_per_fb"]
        ref = m["ref"].get(fn, {})
        rms, asi = ref.get("rms_Nsp"), ref.get("nsp_asimov")
        home_ev = {N: cls_ul(rms * N if rms else float("nan"), rms)
                   for N in BANDS}            # --center zero band, events
        home_obs_ev = cls_ul(asi if asi is not None else float("nan"), rms)
        row = {
            "channel": args.channel, "topology": args.topology,
            "function": fn, "mWR": float(mass), "m_N": m["m_N"],
            "signal_tag": m["signal_tag"], "xsec_pb": m["xsec_pb"],
            "eff": round(m["eff"], 6), "rate_per_fb": round(rate, 4),
            **{f"comb_fb_{k}": round(lims.get(k, float("nan")), 5)
               for k in ("m2s", "m1s", "med", "p1s", "p2s", "obs")},
            **{f"comb_ev_{k}": round(lims[k] * rate, 3)
               for k in lims},
            **{f"home_ev_{LBL[N]}": round(home_ev[N], 3)
               if math.isfinite(home_ev[N]) else "" for N in BANDS},
            "home_ev_obs": round(home_obs_ev, 3)
            if math.isfinite(home_obs_ev) else "",
        }
        rows.append(row)

    if not rows:
        raise SystemExit(f"No combine results found in {args.results_dir / tag}")
    out = args.output_dir / f"combine_limit_table_{tag}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)\n")

    hdr = f"{'m':>6} {'fn':<7} {'':<9}" + "".join(f"{q:>9}" for q in
        ("-2s", "-1s", "med", "+1s", "+2s", "obs"))
    print(hdr + "\n" + "-" * len(hdr))
    for r in rows:
        ce = [r.get(f"comb_ev_{k}", float("nan")) for k in
              ("m2s", "m1s", "med", "p1s", "p2s", "obs")]
        he = [r.get(f"home_ev_{k}") or float("nan") for k in
              ("m2s", "m1s", "med", "p1s", "p2s", "obs")]
        print(f"{r['mWR']:>6.0f} {r['function']:<7} {'combine':<9}"
              + "".join(f"{v:>9.1f}" for v in ce))
        print(f"{'':>6} {'':<7} {'homemade':<9}"
              + "".join(f"{v:>9.1f}" for v in he)
              + "   [events; homemade = center-zero, sigma = toy RMS]")


if __name__ == "__main__":
    main()
