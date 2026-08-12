#!/usr/bin/env python3
"""Stage 10 shared toy driver -- input loading, toy loops, raw-toy CSV I/O.

Everything the Stage-10 compute scripts (1_nsp_diagnostics, 2_prior_scan,
3_injection_validation, 4_bkg_envelope) share and Stage 6/8 kept inline:
loading the summed background and Stage-2 windows, picking/loading signal MC
templates, the accept/reject bookkeeping of a toy loop, and -- new in Stage 10 --
persisting the RAW per-toy fit results so downstream analyses (Gaussianity,
empirical-quantile limits) never again depend on mean+RMS summaries alone.

Source LCG_106 before importing (PyROOT)."""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent                      # 10_limit_refinement
_SIGF = _HERE.parent                                          # signal_fitting
sys.path.insert(0, str(_SIGF.parent))                         # repo root
sys.path.insert(0, str(_SIGF / "shared"))
sys.path.insert(0, str(_SIGF / "4_background_fits"))

import ROOT                                                   # noqa: E402

from wrplotter.paths import input_dirs_for_era, repo_root     # noqa: E402
from bkg_fit_lib import (                                     # noqa: E402
    MASS_VAR, load_summed_background, grid_widths_from_params,
)
from measure_fwhm import (                                    # noqa: E402
    build_region_name, build_hist_key, parse_masses,
)
from shape_estimators import load_master_masses               # noqa: E402
from sb_fit import pick_signal_tag, load_signal_shape         # noqa: E402

# Stage-10 canonical inputs (Stage 6/8 defaults, signal dir bumped to the
# canonical 20260624 grid which includes the low-mass WR1000-1800 points).
DEFAULT_ERA = "RunIII2024Summer24"
DEFAULT_BKG_DIR = "20260317_lo_dy"
DEFAULT_SIGNAL_ERA = "RunIISummer20UL18"
DEFAULT_SIGNAL_DIR = "20260624_signals"
DEFAULT_WINDOW_PARAMS = _SIGF / "2_width_parameterization" / "wr" / "window_params.json"
DEFAULT_MASS_CSV = _SIGF / "master_masses.csv"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

class Inputs:
    """Bundles the background spectrum, windows, and signal-template access for
    one (era, channel, topology)."""

    def __init__(self, *, era=DEFAULT_ERA, bkg_dir=DEFAULT_BKG_DIR,
                 signal_era=DEFAULT_SIGNAL_ERA, signal_dir=DEFAULT_SIGNAL_DIR,
                 channel="ee", topology="resolved", bin_width=100.0,
                 window_params=DEFAULT_WINDOW_PARAMS, sigma_kind="median",
                 mass_csv=DEFAULT_MASS_CSV):
        self.channel, self.topology = channel, topology
        self.era = era
        root = repo_root()
        self.bkg_dirs, _ = input_dirs_for_era(era, root, bkg_dir)
        self.sig_dirs, _ = input_dirs_for_era(signal_era, root, signal_dir)
        region = f"wr_{channel}_{topology}_sr"
        factor = max(1, round(bin_width / 10.0))
        self.edges, self.values, self.variances = load_summed_background(
            self.bkg_dirs, region, MASS_VAR[topology], factor)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.binwidth = float(self.edges[1] - self.edges[0])
        # background-only Poisson mean; negative/non-finite MC bins -> 0
        # (sparse tail outside any fit window; same clip as Stage 6)
        self.mu_bkg = np.array([float(v) if (math.isfinite(v) and v > 0.0)
                                else 0.0 for v in self.values])
        self.window_params = Path(window_params)
        self.sigma_kind = sigma_kind
        self.sig_tags = load_master_masses(Path(mass_csv), topology)
        self.hist_key = build_hist_key(build_region_name(channel, topology),
                                       MASS_VAR[topology])

    def window(self, mWR):
        """(m_c, sigma_win) from the Stage-2 linear parameterization."""
        grid = grid_widths_from_params(
            self.window_params, self.channel, self.topology, [mWR],
            self.sigma_kind)
        return grid[float(mWR)]

    def fit_range(self, mWR, k=3.0, fit_min=800.0, fit_max=6000.0):
        m_c, sigma_win = self.window(mWR)
        lo = max(m_c - k * sigma_win, fit_min)
        hi = min(m_c + k * sigma_win, fit_max)
        return m_c, sigma_win, lo, hi

    def signal_tag(self, mWR, mn_frac=0.5):
        return pick_signal_tag(self.sig_tags, mWR, mn_frac)

    def signal_shape(self, tag):
        """Unit-area MC signal template on the background binning (or None)."""
        return load_signal_shape(self.sig_dirs, self.hist_key, tag, self.edges)

    def b_window(self, mWR, k=3.0):
        m_c, sigma_win = self.window(mWR)
        win = ((self.centers >= m_c - k * sigma_win)
               & (self.centers <= m_c + k * sigma_win))
        return float(self.values[win].sum())


# ---------------------------------------------------------------------------
# Toy loop
# ---------------------------------------------------------------------------

def accepted(res):
    """Stage-6 acceptance: converged, positive finite error, finite yield."""
    return (res is not None and res["status"] == 0
            and res["nsig_err"] > 0 and math.isfinite(res["nsig_err"])
            and math.isfinite(res["nsig"]))


def run_toys(mu_expect, fitter, ntoys, rng):
    """Draw `ntoys` bin-wise Poisson toys of `mu_expect` (ROOT TRandom3) and fit
    each with `fitter(data) -> res dict or None`.

    Returns (records, counts): one record per toy -- {"itoy", "acc", and the
    fit fields when the fit returned} -- and the acceptance bookkeeping
    {"ntoys", "n_ok", "n_none", "n_badstatus", "n_baderr"}.  ALL toys are
    recorded (acc=0 rows keep their status), so downstream selection effects
    can be studied instead of silently baked in.
    """
    records, counts = [], {"ntoys": ntoys, "n_ok": 0, "n_none": 0,
                           "n_badstatus": 0, "n_baderr": 0}
    for itoy in range(ntoys):
        data = np.array([rng.Poisson(m) for m in mu_expect], dtype=float)
        res = fitter(data)
        if res is None:
            counts["n_none"] += 1
            records.append({"itoy": itoy, "acc": 0, "status": -1})
            continue
        acc = accepted(res)
        if acc:
            counts["n_ok"] += 1
        elif res["status"] != 0:
            counts["n_badstatus"] += 1
        else:
            counts["n_baderr"] += 1
        records.append({
            "itoy": itoy, "acc": int(acc),
            "nsig": res["nsig"], "nsig_err": res["nsig_err"],
            "mu": res.get("mu", float("nan")),
            "sigma": res.get("sigma", float("nan")),
            "status": res["status"], "cov": res.get("cov", -1),
            "passed": int(res.get("passed", False)),
            "mu_railed": int(res.get("mu_railed", False)),
            "sigma_railed": int(res.get("sigma_railed", False)),
            "nsig_railed": int(res.get("nsig_railed", False))})
    return records, counts


# ---------------------------------------------------------------------------
# Raw-toy CSV I/O
# ---------------------------------------------------------------------------

RAW_FIELDS = ["itoy", "acc", "nsig", "nsig_err", "mu", "sigma",
              "status", "cov", "passed",
              "mu_railed", "sigma_railed", "nsig_railed"]


def write_raw(path, meta, records):
    """One raw-toy CSV per (point, config): `meta` (constant columns) merged
    into every record row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(meta.keys()) + RAW_FIELDS
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        for r in records:
            w.writerow({**meta, **{k: _fmt(r.get(k)) for k in RAW_FIELDS}})


def read_raw(path, accepted_only=True):
    """Read a raw-toy CSV -> list of dicts with floats parsed."""
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if accepted_only and r.get("acc") != "1":
                continue
            for k in ("nsig", "nsig_err", "mu", "sigma"):
                r[k] = float(r[k]) if r.get(k) not in (None, "") else float("nan")
            out.append(r)
    return out


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return v


def make_rng(base_seed, mWR, tagint):
    """Deterministic ROOT RNG per (point, config) -- Stage-6 seed scheme."""
    return ROOT.TRandom3(base_seed * 1_000_003 + int(mWR) * 1009 + int(tagint))
