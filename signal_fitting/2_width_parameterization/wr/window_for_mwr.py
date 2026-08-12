#!/usr/bin/env python3
"""Stage 2 (wr) -- given ONLY a W_R mass, return the window (mu, sigma, edges).

Evaluates the linear mu(m_WR)/sigma(m_WR) parameterization fit by
parameterize_window.py (window_params.json). Works for any m_WR, on- or off-grid:

    python window_for_mwr.py --mwr 2341 --channel ee --topology resolved --k 3

prints the window centre mu, width sigma, and the [mu - k*sigma, mu + k*sigma]
edges. Importable too:

    from window_for_mwr import window_for_mwr
    w = window_for_mwr(2341, "ee", "resolved", k=3)   # -> dict(mu, sigma, lo, hi, ...)

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

HERE = Path(__file__).resolve().parent
logger = logging.getLogger("window_for_mwr")

DEFAULT_PARAMS = HERE / "window_params.json"


def load_window_params(path=DEFAULT_PARAMS):
    with open(path) as fh:
        return json.load(fh)


def window_for_mwr(mwr, channel, topology, k=3.0, sigma="median",
                   params=None, params_path=DEFAULT_PARAMS):
    """Window (mu, sigma, lo, hi) for a single W_R mass from the linear fit.

    sigma: "median" (central) or "conservative" (trimmed-max coverage). Raises
    KeyError on an unknown channel/topology; warns (logger) if mwr is outside the
    fitted m_WR range -- the linear fit still extrapolates, but flag it.
    """
    p = (params or load_window_params(params_path))[channel][topology]
    key = {"median": "sigma_median",
           "conservative": "sigma_conservative"}[sigma]
    b_mu, a_mu = p["mu"]
    b_s, a_s = p[key]
    mu = a_mu + b_mu * mwr
    sig = a_s + b_s * mwr
    extrap = not (p["mwr_min"] <= mwr <= p["mwr_max"])
    if extrap:
        logger.warning("m_WR=%.0f is outside the fit range [%.0f, %.0f] "
                       "-- extrapolating", mwr, p["mwr_min"], p["mwr_max"])
    return {"mwr": float(mwr), "mu": float(mu), "sigma": float(sig),
            "lo": float(mu - k * sig), "hi": float(mu + k * sig),
            "k": float(k), "sigma_kind": sigma, "extrapolated": extrap}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mwr", type=float, required=True, help="W_R mass [GeV]")
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved",
                   choices=["resolved", "boosted"])
    p.add_argument("--k", type=float, default=3.0, help="window half-width in sigma")
    p.add_argument("--sigma", default="median",
                   choices=["median", "conservative"])
    p.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    w = window_for_mwr(args.mwr, args.channel, args.topology, k=args.k,
                       sigma=args.sigma, params_path=args.params)
    print(f"  channel/topology : {args.channel} {args.topology}")
    print(f"  m_WR             : {w['mwr']:.0f} GeV"
          + ("   [EXTRAPOLATED]" if w["extrapolated"] else ""))
    print(f"  mu  (centre)     : {w['mu']:.1f} GeV")
    print(f"  sigma ({w['sigma_kind']:>12}) : {w['sigma']:.1f} GeV")
    print(f"  window (mu +/- {args.k:g} sigma) : [{w['lo']:.1f}, {w['hi']:.1f}] GeV")


if __name__ == "__main__":
    main()
