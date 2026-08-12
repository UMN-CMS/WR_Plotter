#!/usr/bin/env python3
"""Stage 10.8, step 3 -- collect the refined run2 limits, compare to baseline.

Stitches the regime-split combine outputs into one expected band:

  float (1400-3200)     AsymptoticLimits on the floating-background card
  anch_low (1000-1200)  AsymptoticLimits on the anchored card (B ~ 900)
  anch_sparse (>= 3400) HybridNew expected quantiles on the anchored card
                        (asymptotic kept as a diagnostic column)

Model spread at anchored masses = max |median shift| across the anchor
members (tail / expo2 / powexp vs central), from their asymptotic medians.

Writes refined_limit_table_{tag}.csv and two plots:
  plots/{tag}/refined.*          official-style expected band (sigma x BR, fb)
  plots/{tag}/refined_vs_baseline.*  refined vs Stage-9 baseline + ratio panel

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import uproot

HERE = Path(__file__).resolve().parent
STAGE9 = HERE.parent / "baseline"
sys.path.insert(0, str(HERE.parents[2]))                        # repo root
sys.path.insert(0, str(HERE.parents[1] / "4_background_fits"))  # bkg_fit_lib
sys.path.insert(0, str(HERE.parents[1] / "7_limit_plots"))      # plot_band

from wrplotter.cli_utils import setup_logging                   # noqa: E402
from xsec_limit import plot_band, BANDS                         # noqa: E402

logger = logging.getLogger("collect_refined")

QKEY = {0.025: -2, 0.16: -1, 0.5: 0, 0.84: 1, 0.975: 2}
LBL = {-2: "m2s", -1: "m1s", 0: "med", 1: "p1s", 2: "p2s"}


def read_asymptotic(path):
    """{N: r} from an AsymptoticLimits tree (N = -2..2, 'obs')."""
    with uproot.open(path) as f:
        t = f["limit"]
        lim = t["limit"].array(library="np")
        q = t["quantileExpected"].array(library="np")
    out = {}
    for li, qi in zip(lim, q):
        if qi < 0:
            out["obs"] = float(li)
        else:
            key = QKEY.get(round(float(qi), 3))
            if key is not None:
                out[key] = float(li)
    return out


def read_hybrid(res_dir, variant, mass):
    """{N: r} from the HybridNew quantile files (missing quantiles skipped)."""
    out = {}
    for q, key in QKEY.items():
        pat = f"{res_dir}/higgsCombine_{variant}.HybridNew.mH{mass}.quant{q:.3f}.root"
        hits = glob.glob(pat)
        if not hits:
            continue
        with uproot.open(hits[0]) as f:
            arr = f["limit"]["limit"].array(library="np")
        if len(arr):
            out[key] = float(arr[-1])
    return out


def read_baseline(table, fn="expo"):
    """Stage-9 run2 baseline: {mass: {N: fb, 'obs': fb}}."""
    out = {}
    with open(table, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("function") != fn:
                continue
            try:
                m = int(float(r["mWR"]))
                out[m] = {N: float(r[f"comb_fb_{LBL[N]}"]) for N in BANDS}
                out[m]["obs"] = float(r["comb_fb_obs"])
            except (TypeError, ValueError, KeyError):
                pass
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--channel", default="ee")
    p.add_argument("--topology", default="resolved")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=HERE)
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    tag = f"{args.channel}_{args.topology}"
    res_dir = args.results_dir or (HERE / "results" / tag)

    with open(STAGE9 / "run2" / "inputs" / f"{tag}.json") as fh:
        meta = json.load(fh)
    lumi, com = meta["lumi"], meta.get("com", 13)
    masses = meta["masses"]

    manifest = {}
    with open(HERE / "cards" / tag / "manifest.txt") as fh:
        for line in fh:
            card, mass, variant, regime, rmax = line.strip().split("\t")
            manifest.setdefault(int(mass), {})[variant] = regime

    rows, pts = [], []
    for mass in sorted(manifest):
        regime = next(iter(manifest[mass].values()))
        m = masses[str(mass)]
        asym = {v: {} for v in manifest[mass]}
        for v in manifest[mass]:
            f = res_dir / f"higgsCombine_{v}.AsymptoticLimits.mH{mass}.root"
            if f.exists():
                asym[v] = read_asymptotic(f)
        central_v = "float" if regime == "float" else "anchored"
        band = dict(asym.get(central_v, {}))
        method = "asymptotic"
        if regime == "anch_sparse":
            hyb = read_hybrid(res_dir, "anchored", mass)
            if 0 in hyb:
                band, method = {**band, **hyb}, "hybridnew"
        if 0 not in band:
            logger.warning("m=%d: no median, skipping", mass)
            continue
        spread = 0.0
        if regime != "float":
            meds = [asym[v].get(0) for v in manifest[mass]
                    if v != central_v and asym[v].get(0) is not None]
            c = asym.get(central_v, {}).get(0)
            if c and meds:
                spread = max(abs(x - c) for x in meds)
        rows.append({
            "channel": args.channel, "topology": args.topology, "mWR": mass,
            "regime": regime, "method": method,
            "eff": round(float(m["eff"]), 6),
            "xsec_pb": float(m["xsec_pb"]),
            **{f"fb_{LBL[N]}": round(band.get(N, float("nan")), 5)
               for N in BANDS},
            "fb_obs": round(band.get("obs", float("nan")), 5),
            "asym_med_fb": round(asym.get(central_v, {}).get(0, float("nan")), 5),
            "model_spread_fb": round(spread, 5),
        })
        pts.append({
            "mWR": float(mass),
            "sigma": {N: band.get(N, float("nan")) / 1000.0 for N in BANDS},
            "sigma_obs": float("nan"),          # expected-only study
            "xsec_pb": float(m["xsec_pb"]),
        })

    out_csv = args.output_dir / f"refined_limit_table_{tag}.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s (%d rows)", out_csv, len(rows))

    # official-style expected band
    plot_dir = args.output_dir / "plots" / tag
    plot_band("refined (anchored+HybridNew)", pts, plot_dir / "refined",
              ykey="sigma", obskey="sigma_obs", theory=True, scale=1000.0,
              ylabel=(r"$\sigma(pp \to W_R)\,\mathcal{B}"
                      r"(W_R \to eeq\bar{q}\,')$ (fb)"),
              channel=args.channel, topology=args.topology, com=com,
              lumi=lumi, cl=0.95, trust_max=None, center="zero (combine)")

    # refined vs Stage-9 baseline + ratio panel
    base = read_baseline(STAGE9 / "run2" / f"combine_limit_table_{tag}.csv")
    hep.style.use("CMS")
    fig, (ax, axr) = plt.subplots(2, 1, sharex=True, height_ratios=[3, 1],
                                  gridspec_kw={"hspace": 0.06},
                                  figsize=(10, 11))
    mm = [r["mWR"] for r in rows]
    ax.fill_between(mm, [r["fb_m2s"] for r in rows],
                    [r["fb_p2s"] for r in rows], color="#f5d800",
                    label="refined 95% expected")
    ax.fill_between(mm, [r["fb_m1s"] for r in rows],
                    [r["fb_p1s"] for r in rows], color="#00cc00",
                    label="refined 68% expected")
    ax.plot(mm, [r["fb_med"] for r in rows], "k:", lw=2,
            label="refined expected (anchored + HybridNew)")
    bm = [m for m in mm if m in base]
    ax.plot(bm, [base[m][0] for m in bm], color="#5790fc", ls="--", lw=2,
            label="Stage-9 baseline expected (float + asymptotic)")
    for m_, r in zip(mm, rows):
        if r["model_spread_fb"] > 0:
            ax.vlines(m_, r["fb_med"] - r["model_spread_fb"],
                      r["fb_med"] + r["model_spread_fb"],
                      color="grey", lw=3, alpha=0.5)
    ax.set_yscale("log")
    ax.set_ylabel(r"95% CL UL on $\sigma\,\mathcal{B}(eeq\bar{q}\,')$ (fb)")
    ax.legend(loc="lower left", fontsize=13, frameon=False)
    ax.text(0.95, 0.93, "run2 2018, ee resolved\ngrey bars: anchor-model spread",
            transform=ax.transAxes, ha="right", va="top", fontsize=13)
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=15)
    ratio = [(m_, r["fb_med"] / base[m_][0]) for m_, r in zip(mm, rows)
             if m_ in base and base[m_][0] > 0]
    axr.plot([x for x, _ in ratio], [y for _, y in ratio], "o-",
             color="#5790fc", ms=4, lw=1.5)
    axr.axhline(1.0, color="grey", lw=0.8, ls=":")
    axr.set_ylabel("refined / baseline", fontsize=13)
    axr.set_xlabel(r"$m_{W_R}$ (GeV)")
    plot_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(plot_dir / f"refined_vs_baseline{ext}", bbox_inches="tight",
                    **({"dpi": 150} if ext == ".png" else {}))
    plt.close(fig)
    logger.info("plots in %s", plot_dir)

    hdr = f"{'m':>6} {'regime':<12} {'method':<11}" + "".join(
        f"{LBL[N]:>9}" for N in BANDS) + f"{'baseline':>10}{'ratio':>7}"
    print(hdr + "\n" + "-" * len(hdr))
    for r in rows:
        b = base.get(r["mWR"], {}).get(0, float("nan"))
        ratio = r["fb_med"] / b if b and b > 0 else float("nan")
        print(f"{r['mWR']:>6} {r['regime']:<12} {r['method']:<11}"
              + "".join(f"{r[f'fb_{LBL[N]}']:>9.3f}" for N in BANDS)
              + f"{b:>10.3f}{ratio:>7.2f}")


if __name__ == "__main__":
    main()
