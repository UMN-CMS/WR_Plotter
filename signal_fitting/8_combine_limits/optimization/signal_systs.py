#!/usr/bin/env python3
"""Signal systematics from the analyzer's variations: for each source, the
change of the in-window signal yield (-> log-normal on the signal rate) and,
for the sources that move the peak, the change of the Gaussian-core mean and
width (-> param constraints on mu_sig / sigma_sig).

  python signal_systs.py --era RunIII2024Summer24 --dir 20261004_full_systs \\
      --tags WR2000_N1100 WR4000_N2100 WR6000_N3100 --inputs inputs_2024 \\
      --out inputs_2024/signal_systs.json

The window for each tag is k5 around the nominal Gaussian-core fit of that
sample (same recipe as prepare_opt --geo-from-signal); yields are ratios, so
the sample's normalization bookkeeping drops out.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
SIGFIT = HERE.parents[1]
sys.path.insert(0, str(SIGFIT.parent))
sys.path.insert(0, str(SIGFIT / "shared"))
from wrplotter.cli_utils import setup_logging                           # noqa: E402
from wrplotter.config import load_systematics                           # noqa: E402
from wrplotter.paths import input_dirs_for_era, repo_root               # noqa: E402
from shape_estimators import gaussian_core_fit                          # noqa: E402
from syst_band import variation_keys                                    # noqa: E402

logger = logging.getLogger("signal_systs")
REGION = "wr_ee_resolved_sr"
SHAPE_SOURCES = ("elescale", "elesmear", "jes", "jer")
PLACEHOLDER = {"muscale", "musmear"}
# groups for the card: several analyzer sources -> one nuisance each
GROUPS = {
    "lumi": ["lumi"],
    "pileup": ["pileup"],
    "ele_sf": ["electronrecosf", "electronidsf", "electrontrigsf", "looseelectronrecosf"],
    "mu_sf": ["muonrecosf", "muonidsf", "muonisosf", "muontrigsf", "loosemuonrecosf", "loosemuonidsf"],
    "jes": ["jes"], "jer": ["jer"], "elescale": ["elescale"], "elesmear": ["elesmear"],
    "scale": ["scale"],            # envelope of muR / muF / muRF (syst_band.ENVELOPES)
    "ps": ["psisr", "psfsr"],
}


def hist(dirs, tag, key):
    for d in dirs:
        fp = Path(d) / f"WRAnalyzer_signal_{tag}.root"
        if fp.exists():
            f = uproot.open(fp)
            if key in f:
                h = f[key]
                v = h.values()
                return h.axes[0].edges(), np.where(np.isfinite(v) & (v > 0), v, 0.0)
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era", default="RunIII2024Summer24")
    p.add_argument("--dir", default="20261004_full_systs")
    p.add_argument("--tags", nargs="+", default=["WR2000_N1100", "WR4000_N2100", "WR6000_N3100"])
    p.add_argument("--k", type=float, default=5.0)
    p.add_argument("--exclude", nargs="*", default=[],
                   help="sources to leave out of the groups/totals (still "
                        "evaluated and written under 'sources'), e.g. pileup "
                        "while the 2024 signal is the 23BPix placeholder")
    p.add_argument("--out", type=Path, default=HERE / "inputs_2024" / "signal_systs.json")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    sources = [s for s in load_systematics() if s not in PLACEHOLDER]
    for extra in ("electronidsf", "electrontrigsf", "looseelectronrecosf"):
        if extra not in sources:
            sources.append(extra)
    out = {"era": args.era, "dir": args.dir, "region": REGION, "k": args.k, "masses": {}}
    for tag in args.tags:
        mwr = int(tag[2:].split("_N")[0])
        nom = hist(dirs, tag, f"{REGION}/mass_fourobject_{REGION}")
        if nom is None:
            logger.warning("%s: no nominal histogram in %s", tag, args.dir)
            continue
        e, v = nom
        c = 0.5 * (e[:-1] + e[1:])
        g0 = gaussian_core_fit(e, v, float(mwr), 0.7, 1.3)
        mu0, s0 = g0["mu"], g0["sigma"]
        lo, hi = mu0 - args.k * s0, mu0 + args.k * s0
        sel = (c >= lo) & (c <= hi)
        y0 = float(v[sel].sum())
        entry = {"m_c": mu0, "sigma": s0, "lo": lo, "hi": hi, "yield": y0, "sources": {}}
        logger.info("%s: mu=%.1f sigma=%.1f window [%.0f, %.0f] yield %.1f", tag, mu0, s0, lo, hi, y0)
        for src in sources:
            row = {}
            for dirn, key in variation_keys(src, REGION, "mass_fourobject"):
                r = hist(dirs, tag, key)
                if r is None:
                    continue
                ev, vv = r
                cv = 0.5 * (ev[:-1] + ev[1:])
                yv = float(vv[(cv >= lo) & (cv <= hi)].sum())
                row[dirn] = {"yield_ratio": yv / y0 if y0 > 0 else float("nan")}
                if src in SHAPE_SOURCES:
                    g = gaussian_core_fit(ev, vv, float(mwr), 0.7, 1.3)
                    row[dirn]["mu_ratio"] = g["mu"] / mu0
                    row[dirn]["sigma_ratio"] = g["sigma"] / s0
            if not row:
                continue
            dirs_present = [d for d, _ in variation_keys(src, REGION, "mass_fourobject") if d in row]
            rate = max(abs(row[d]["yield_ratio"] - 1.0) for d in dirs_present)
            row["rate_rel"] = rate
            if src in SHAPE_SOURCES:
                row["mu_rel"] = max(abs(row[d]["mu_ratio"] - 1.0) for d in dirs_present)
                row["sigma_rel"] = max(abs(row[d]["sigma_ratio"] - 1.0) for d in dirs_present)
            entry["sources"][src] = row
            extra = (f"  mu {100 * row['mu_rel']:.2f}%  sigma {100 * row['sigma_rel']:.2f}%"
                     if src in SHAPE_SOURCES else "")
            logger.info("  %-22s rate %s  -> %.2f%%%s", src,
                        " / ".join(f"{100 * (row[d]['yield_ratio'] - 1):+6.2f}%" for d in dirs_present),
                        100 * rate, extra)
        # grouped numbers for the card (quadrature within a group)
        groups = {}
        entry["excluded"] = list(args.exclude)
        for gname, members in GROUPS.items():
            rr = [entry["sources"][m]["rate_rel"] for m in members
                  if m in entry["sources"] and m not in args.exclude]
            if rr:
                groups[gname] = {"rate_rel": float(np.sqrt(np.sum(np.square(rr))))}
                if gname in SHAPE_SOURCES and gname not in args.exclude:
                    groups[gname]["mu_rel"] = entry["sources"][gname]["mu_rel"]
                    groups[gname]["sigma_rel"] = entry["sources"][gname]["sigma_rel"]
        entry["groups"] = groups
        tot = float(np.sqrt(np.sum(np.square([g["rate_rel"] for g in groups.values()]))))
        entry["rate_total_rel"] = tot
        logger.info("  %s total on the rate: %.2f%%  (groups %s)", tag, 100 * tot,
                    {k: round(100 * g["rate_rel"], 2) for k, g in groups.items()})
        out["masses"][str(mwr)] = entry
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
