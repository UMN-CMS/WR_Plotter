#!/usr/bin/env python3
"""Shared S+B fit core for the spurious-signal (Stage 5) and signal-injection
(Stage 6) studies.

The background TF1 (Stage-4 recentered) plus a fixed-shape Gaussian signal, fit
to a given spectrum inside the window; plus the per-point width and signal-MC
template loaders. Both stages import from here so the fit is identical.

Source LCG_106 before importing (PyROOT)."""
from __future__ import annotations

import csv
import ctypes
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent                      # signal_fitting/shared
sys.path.insert(0, str(_HERE))                               # measure_fwhm
sys.path.insert(0, str(_HERE.parent / "4_background_fits"))  # bkg_fit_lib

import ROOT                                                  # noqa: E402
ROOT.TH1.AddDirectory(False)          # don't register TH1Ds in gDirectory -- they
                                      # would pile up over the toy loop's many fits
from bkg_fit_lib import (                                    # noqa: E402
    FUNCS, TF1_FORMULA, PAR_LIMITS, BASELINE_FIT_RANGE, _seed, predict,
)
from measure_fwhm import parse_masses, load_and_combine_signal  # noqa: E402

_UID = 0

# Same four Minuit fit-quality checks as the Stage-4 background-only fit
# (bkg_fit_lib.fit_model). Applied here to the BACKGROUND component of the S+B
# fit; N_sig is unbounded so it never trips the limit check, and monotonicity is
# tested on the background curve alone (the Gaussian bump is expected). Keeps the
# spurious-signal fits held to the same standard as the background fits.
CHECK_KEYS = ["valid_minimum", "cov_ok", "no_param_at_limit", "monotonic"]
CHECK_SHORT = {"valid_minimum": "min", "cov_ok": "cov",
               "no_param_at_limit": "limit", "monotonic": "mono"}


# ---------------------------------------------------------------------------
# Signal templates & widths
# ---------------------------------------------------------------------------

def load_point_widths(width_csv, channel, topology):
    """{(mWR, mN): (mu, sigma)} per signal point (baseline fit range), NOT
    aggregated -- the Gaussian model uses the point's own width."""
    out = {}
    with open(width_csv) as fh:
        for r in csv.DictReader(fh):
            if (r["channel"] == channel and r["category"] == topology
                    and r["fit_range"] == BASELINE_FIT_RANGE):
                out[(int(float(r["mWR"])), int(float(r["mN"])))] = (
                    float(r["mu_gaus"]), float(r["sigma_gaus"]))
    return out


def pick_signal_tag(tags, mWR, mn_frac):
    """master-grid signal tag for this m_WR with m_N closest to mn_frac*m_WR.

    Same signal samples as 0_signal_samples (master_masses.csv + the signal era).
    Ties -> lower m_N.
    """
    target = mn_frac * mWR
    cand = [(t, parse_masses(t)[1]) for t in tags if parse_masses(t)[0] == int(mWR)]
    if not cand:
        return None
    return min(cand, key=lambda tm: (abs(tm[1] - target), tm[1]))[0]


def load_signal_shape(sig_dirs, hist_key, tag, bkg_edges):
    """Unit-area signal MC shape rebinned onto the background bin edges."""
    edges_n, vals_n, _ = load_and_combine_signal(sig_dirs, hist_key, tag)
    centers_n = 0.5 * (edges_n[:-1] + edges_n[1:])
    on_bkg, _ = np.histogram(centers_n, bins=bkg_edges, weights=vals_n)
    tot = on_bkg.sum()
    return on_bkg / tot if tot > 0 else None


def gauss_per_bin(centers, mu, sigma, binwidth):
    """Per-bin events for a unit-yield Gaussian -- the fit's signal model."""
    amp = binwidth / (sigma * math.sqrt(2.0 * math.pi))
    return amp * np.exp(-0.5 * ((centers - mu) / sigma) ** 2)


# ---------------------------------------------------------------------------
# S+B fit (background TF1 + fixed-shape Gaussian) to a given spectrum
# ---------------------------------------------------------------------------

def fit_splusb(name, edges, data, lo, hi, m_c, sigma_win, mu_sig, sigma_sig,
               binwidth, k):
    """Fit background TF1 + fixed Gaussian to `data`, inside [lo, hi] only.

    Window/sideband and the background recentering use the median (m_c,
    sigma_win); the fixed-shape Gaussian uses the signal point's own
    (mu_sig, sigma_sig). Bin errors are expected Poisson sqrt(content).
    Returns dict(nsig, nsig_err, params, perr, status, cov, chi2, ndf, nbins)
    or None when the window has too few populated bins or no seed.
    """
    global _UID
    centers = 0.5 * (edges[:-1] + edges[1:])
    npar = FUNCS[name][2]
    infit = (centers >= lo) & (centers <= hi) & (data > 0)
    if infit.sum() <= npar + 1:
        return None
    win = (centers >= m_c - k * sigma_win) & (centers <= m_c + k * sigma_win)
    sb = infit & ~win
    seed_src = sb if sb.sum() > npar else infit
    bseed = _seed(name, centers[seed_src], data[seed_src], data[seed_src], m_c)
    if bseed is None:
        return None

    _UID += 1
    h = ROOT.TH1D(f"_sb_h{_UID}", "", len(data),
                  np.ascontiguousarray(edges, np.float64))
    for i, v in enumerate(data, start=1):
        h.SetBinContent(i, v)
        h.SetBinError(i, math.sqrt(v) if v > 0 else 0.0)       # expected Poisson

    amp = binwidth / (sigma_sig * math.sqrt(2.0 * math.pi))
    bkg = TF1_FORMULA[name].format(m0=f"{m_c:.10g}")
    sig = (f"[{npar}]*{amp:.10g}*exp(-0.5*pow("
           f"(x-{mu_sig:.10g})/{sigma_sig:.10g},2))")
    f = ROOT.TF1(f"_sb_f{_UID}", f"({bkg})+({sig})", lo, hi)
    lims = PAR_LIMITS.get(name, [])
    for i, s in enumerate(bseed):
        if i < len(lims) and lims[i] is not None:   # clamp the seed into the box
            s = min(max(s, lims[i][0]), lims[i][1])
        f.SetParameter(i, s)
    f.SetParameter(npar, 0.0)
    for i, lim in enumerate(lims):
        if lim is not None:
            f.SetParLimits(i, lim[0], lim[1])

    r = h.Fit(f, "RNSQ").Get()
    ROOT.gROOT.GetListOfFunctions().Remove(f)    # TF1s register globally -> drop it
    if not r:
        return None
    np1 = npar + 1
    corr = np.array([[float(r.Correlation(i, j)) for j in range(np1)]
                     for i in range(np1)])         # full param correlation matrix

    # Same four quality checks as the Stage-4 background fit, on the BACKGROUND
    # component (params[:npar]); N_sig is unbounded so it can't rail a limit.
    bkg_params = np.array([float(r.Parameter(i)) for i in range(npar)])
    at_limit = False
    for i in range(npar):
        plo, phi = ctypes.c_double(), ctypes.c_double()
        f.GetParLimits(i, plo, phi)
        if phi.value > plo.value:                  # limits were set
            rng = phi.value - plo.value
            if (bkg_params[i] - plo.value) < 1e-3 * rng \
                    or (phi.value - bkg_params[i]) < 1e-3 * rng:
                at_limit = True
    mgrid = np.linspace(lo, hi, 200)               # background must fall (no rise)
    fmono = predict(name, bkg_params, mgrid, m_c)
    monotonic = bool(np.all(fmono[1:] <= fmono[:-1] * (1.0 + 1e-6)))
    checks = {"valid_minimum": bool(r.IsValid()),
              "cov_ok": int(r.CovMatrixStatus()) == 3,
              "no_param_at_limit": not at_limit,
              "monotonic": monotonic}

    return dict(nsig=float(r.Parameter(npar)), nsig_err=float(r.ParError(npar)),
                params=np.array([float(r.Parameter(i)) for i in range(np1)]),
                perr=np.array([float(r.ParError(i)) for i in range(np1)]),
                corr=corr,                          # corr[npar, i] = corr(N_sig, par_i)
                checks=checks, passed=all(checks.values()),
                status=int(r.Status()), cov=int(r.CovMatrixStatus()),
                chi2=float(r.Chi2()), ndf=int(r.Ndf()), nbins=int(infit.sum()))
