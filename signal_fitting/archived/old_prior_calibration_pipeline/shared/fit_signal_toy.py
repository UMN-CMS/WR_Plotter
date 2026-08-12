#!/usr/bin/env python3
"""Compare {single Gaussian, bifurcated Gaussian} x {free, FWHM-constrained}.

Draws one ~20-event toy from the MC histogram at a chosen (M_WR, M_N) and
fits the same events with all four configurations:

  gauss/free       : RooGaussian, sigma free
  gauss/constrain  : RooGaussian, sigma ~ N(sigma_pred, sigma_sigma)
  bifur/free       : RooBifurGauss, Sigma free, Delta free
  bifur/constrain  : RooBifurGauss, Sigma ~ N(Sigma_pred, sigma_sigma), Delta free

mu is fixed to M_WR in every config. The width prior central value
(sigma_pred for gauss, Sigma_pred for bifur) and its uncertainty
sigma_sigma both come from the FWHM parameterization (results.json,
model a_linear), via predict_sigma().

Output:
  signal_toy_compare_<sig_tag>_<channel>_seed<seed>_n<n>.pdf  (2x2 grid)
  signal_toy_compare_<sig_tag>_<channel>_seed<seed>_n<n>.json (all 4 fits)

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

Usage:
  python signal_fitting/fit_signal_toy.py --wr 4000 --n 2000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

try:
    import ROOT
except ImportError:
    sys.exit(
        "ERROR: PyROOT not available.\n"
        "  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh"
    )

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wrplotter.cli_utils import setup_logging
from wrplotter.config import load_lumi
from wrplotter.paths import input_dirs_for_era, repo_root

from measure_fwhm import (
    ONSHELL_WINDOW_LO_FRAC,
    ONSHELL_WINDOW_HI_FRAC,
    build_hist_key,
    build_region_name,
    load_and_combine_signal,
    compute_shape_params,
    rebin_histogram,
)

logger = logging.getLogger(__name__)

# FWHM-to-width conversions for the two PDFs:
#   single Gaussian:    FWHM = 2*sqrt(2 ln 2) * sigma            ~ 2.3548 * sigma
#   bifurcated Gauss:   FWHM = sqrt(2 ln 2) * (sigma_L + sigma_R) ~ 1.1774 * Sigma
# (The bifur factor is half the gauss factor because the bifur "Sigma" = sigma_L+sigma_R
#  is the sum of two HWHM-like widths, not 2x a single-half-Gaussian sigma.)
FWHM_TO_GAUSS_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))   # 2.3548
FWHM_TO_BIFUR_SIGMA = np.sqrt(2.0 * np.log(2.0))         # 1.1774

SYSTEMATIC_FWHM_FLOOR_GEV = 25.0  # default: covers Stage 1 residual scatter (chi2/ndf ~ 1.3) above the parameterization stat error.
                                  # Pass --systematic-floor-gev 0 on the CLI to disable.

# Width of the µ prior, expressed as a fraction of FWHM_pred:
#   sigma_mu_prior = MU_PRIOR_ALPHA[channel] * FWHM_pred(M_WR, x)
# Channel-dependent (May 2026) calibrated empirically by scanning bulk M_WR ≥
# 3000 cells in both channels to drive the per-cell µ-pull spread toward 1.
# μμ has wider intrinsic peaks → larger natural data-error scale → wants a
# slightly larger α. Pre-May-2026 setup used sigma_peak_boot (the bootstrap
# uncertainty on the MC peak position) which produced spreads ~0.3–0.6.
MU_PRIOR_ALPHA = {"ee": 0.22, "mumu": 0.25}

# Width prior:
#   FWHM_prior_central     = FWHM_boot       (per-cell bootstrap-mean MC FWHM)
#   sigma_FWHM_prior       = WIDTH_PRIOR_ALPHA[channel] * FWHM_param
# The central uses FWHM_boot — the per-cell MC truth FWHM — to put the prior
# *on* truth (eliminating the −5σ to +14σ Σ-pull bias range that resulted
# when the parameterization was ~100 GeV off the actual MC FWHM at individual
# cells). The σ uses FWHM_param (the global parameterization), matching the
# µ-prior σ convention so both priors share the same "size" reference.
WIDTH_PRIOR_ALPHA = {"ee": 0.06, "mumu": 0.08}

# Legacy 4-config list (model x width-mode) — used only by the --only-config
# single-panel path. The 2x2 comparison plots use COMPARE_CONFIGS below.
CONFIGS = [
    ("gauss", "free"),
    ("gauss", "constrain"),
    ("bifur", "free"),
    ("bifur", "constrain"),
]

# Prior-grid configs for the new 2x2 comparison plot, one grid per PDF.
# Each entry: (name, mu_mode, width_mode, label).
#   mu_mode    : "free" (mu floats no prior) | "constrained" (Gaussian prior on mu)
#   width_mode : "free" (no prior on sigma/Sigma) | "constrained" (Gaussian prior)
# In bifur, Delta is always free regardless.
COMPARE_CONFIGS = [
    ("no_priors",  "free",        "free",        "No Priors"),
    ("mu_only",    "constrained", "free",        r"$\mu$ constrained"),
    ("width_only", "free",        "constrained", "Width constrained"),
    ("both",       "constrained", "constrained", "Both constrained"),
]


# ---------------------------------------------------------------------------
# FWHM-parameterization width prior
# ---------------------------------------------------------------------------

def predict_fwhm(model: dict, x: float, M_WR: float) -> tuple[float, float]:
    """Return (FWHM_pred, FWHM_err) in GeV from the a_linear results entry.

    FWHM = a*x + b*M_WR. Uncertainty combines the parameterization stat error
    (J^T cov J) with the systematic residual-scatter floor in quadrature.

    Each downstream PDF converts FWHM to its own width parameter using the
    appropriate factor (FWHM_TO_GAUSS_SIGMA or FWHM_TO_BIFUR_SIGMA).
    """
    pnames = model["param_order"]
    assert pnames == ["a", "b"], f"unexpected order {pnames}"
    a = model["params"]["a"]
    b = model["params"]["b"]
    cov = np.asarray(model["covariance"])

    fwhm_pred = a * x + b * M_WR
    J = np.array([x, M_WR])
    var_fwhm_stat = float(J @ cov @ J)
    var_fwhm_sys = SYSTEMATIC_FWHM_FLOOR_GEV ** 2
    fwhm_err = np.sqrt(var_fwhm_stat + var_fwhm_sys)

    return fwhm_pred, fwhm_err


# ---------------------------------------------------------------------------
# Toy generation
# ---------------------------------------------------------------------------

def bootstrap_fwhm_estimate(
    edges: np.ndarray, vals: np.ndarray,
    sig_tag: str, channel: str, topology: str,
    n_toys: int = 100, seed: int = 12345,
) -> tuple[float, float]:
    """Bootstrap-derived (central, uncertainty) for fwhm_onshell.

    Each toy: Poisson-resample bin counts, recompute fwhm_onshell.
    Returns (bootstrap_mean, sqrt(sigma_bootstrap^2 + (Δm/√6)^2)) — bootstrap
    mean as central; bootstrap stdev ⊕ FWHM binning floor as uncertainty.
    Uncertainty is computed but not currently used (we use the parameterization
    σ_FWHM downstream); kept for diagnostic logging.
    """
    rng = np.random.default_rng(seed)
    lam = np.maximum(vals, 0.0)
    fwhms: list[float] = []
    for _ in range(n_toys):
        toy = rng.poisson(lam).astype(float)
        if toy.max() <= 0:
            continue
        sp_toy = compute_shape_params(edges, toy, sig_tag, channel, topology)
        if np.isfinite(sp_toy.fwhm_onshell):
            fwhms.append(float(sp_toy.fwhm_onshell))
    if not fwhms:
        return float("nan"), float("nan")
    mean_fwhm = float(np.mean(fwhms))
    sigma_bootstrap = float(np.std(fwhms))
    bin_width = float(edges[1] - edges[0])
    sigma_binning = bin_width * np.sqrt(2.0) / np.sqrt(12.0)  # Δm/√6
    sigma_total = float(np.sqrt(sigma_bootstrap ** 2 + sigma_binning ** 2))
    return mean_fwhm, sigma_total


def bootstrap_peak_estimate(
    edges: np.ndarray, vals: np.ndarray,
    sig_tag: str, channel: str, topology: str,
    n_toys: int = 100, seed: int = 12345,
) -> tuple[float, float]:
    """Bootstrap-derived (central, uncertainty) for peak_onshell.

    Each toy: Poisson-resample bin counts, recompute peak_onshell.
    Returns (bootstrap_mean, sqrt(sigma_bootstrap^2 + (bin_width/sqrt(12))^2)) —
    bootstrap mean as central (averages over Poisson realizations of the MC
    histogram), bootstrap stdev combined with the in-bin position-resolution
    floor for uncertainty.
    """
    rng = np.random.default_rng(seed)
    lam = np.maximum(vals, 0.0)
    peaks: list[float] = []
    for _ in range(n_toys):
        toy = rng.poisson(lam).astype(float)
        if toy.max() <= 0:
            continue
        sp_toy = compute_shape_params(edges, toy, sig_tag, channel, topology)
        if np.isfinite(sp_toy.peak_onshell):
            peaks.append(float(sp_toy.peak_onshell))
    if not peaks:
        return float("nan"), float("nan")
    mean_peak = float(np.mean(peaks))
    sigma_bootstrap = float(np.std(peaks))
    bin_width = float(edges[1] - edges[0])
    sigma_binning = bin_width / np.sqrt(12.0)
    sigma_total = float(np.sqrt(sigma_bootstrap ** 2 + sigma_binning ** 2))
    return mean_peak, sigma_total


def sample_from_hist(edges, vals, n_events, rng) -> np.ndarray:
    """Sample n_events from a histogram treated as a PDF (uniform within bin)."""
    probs = np.maximum(vals, 0.0)
    probs = probs / probs.sum()
    idx = rng.choice(len(vals), size=n_events, p=probs)
    return rng.uniform(edges[idx], edges[idx + 1])


def sample_from_hist_root(edges, vals, n_events, seed) -> np.ndarray:
    """ROOT-based sampler: TH1::GetRandom on a fresh TH1 built from the bins.

    Equivalent to sample_from_hist (CDF-binned + uniform-within-bin), but uses
    ROOT.gRandom for the random numbers so the gauss-only study runs entirely
    through ROOT. Takes an integer `seed` instead of a numpy Generator.

    --- RNG choice: TRandom3 (gRandom) with Knuth-hashed seed, no warm-up ---

    ROOT.gRandom is TRandom3 (MT19937). It has a known weakness: SetSeed(s)
    for small consecutive integers s = 1, 2, 3, ... produces correlated
    internal states, so neighbouring seeds give partially-correlated initial
    streams. With a fixed pull-study seed grid (100 seeds per cell), this
    biases toy-to-toy variance and shifts the pull-spread calibration away
    from numpy's PCG64 reference.

    We mitigate by hashing the seed with Knuth's multiplicative constant
    2654435761 (= 2^32 / φ) before SetSeed. This spreads close integers
    uniformly across the 32-bit seed space.

    We DO NOT do a warm-up (discarding the first N draws of the stream).
    Empirically it made the calibration worse, not better: the warm-up
    matches numpy's per-toy sample-RMS on average, but introduces additional
    correlations between consecutive (warmed-up) seeds that show up as worse
    pull-spread calibration. Hash-only is the better compromise.

    Single-cell calibration vs numpy PCG64 at N=20, WR4000_N2000 ee
    (100 toys, α_µ=2.0, α_σ=0.15):

      numpy PCG64           µ-spread=0.99  width-spread=1.00  [reference]
      TRandom3 + Knuth hash µ-spread=0.96  width-spread=0.84  [this fn]
      TRandom3 + hash + 1000-step warmup   width-spread=0.79 — worse

    Full-grid calibration ends up needing slightly different α values than
    the numpy reference (see signal_fitting/README.md decisions log).
    """
    n_bins = len(vals)
    edges_arr = np.ascontiguousarray(edges, dtype=np.float64)
    h = ROOT.TH1D("", "", n_bins, edges_arr)
    h.SetDirectory(0)
    for i in range(n_bins):
        h.SetBinContent(i + 1, max(float(vals[i]), 0.0))
    # Knuth multiplicative hash spreads close integer seeds across the
    # 32-bit seed space — see notes above.
    seed_hashed = (int(seed) * 2654435761) & 0xFFFFFFFF
    ROOT.gRandom.SetSeed(seed_hashed)
    return np.fromiter((h.GetRandom() for _ in range(int(n_events))),
                       dtype=np.float64, count=int(n_events))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_dataset(events, fit_lo, fit_hi, suffix):
    m = ROOT.RooRealVar(f"m_{suffix}", "m_{lljj}", fit_lo, fit_hi, "GeV")
    data = ROOT.RooDataSet(f"data_{suffix}", "toy", ROOT.RooArgSet(m))
    n_used = 0
    for ev in events:
        if fit_lo <= ev <= fit_hi:
            m.setVal(float(ev))
            data.add(ROOT.RooArgSet(m))
            n_used += 1
    return m, data, n_used


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------

def fit_single_gaussian(
    events, M_WR, fwhm_pred, fwhm_err, fit_lo, fit_hi, *,
    width_mode: str = "constrained",
    mu_mode: str = "fixed",
    mu_central: float | None = None,
    mu_sigma: float | None = None,
    suffix_extra: str = "",
):
    sigma_pred = fwhm_pred / FWHM_TO_GAUSS_SIGMA
    sigma_sigma = fwhm_err / FWHM_TO_GAUSS_SIGMA
    constrain = (width_mode == "constrained")
    mu_constrained = (mu_mode == "constrained")
    mu_floating = (mu_mode in ("free", "constrained"))
    suffix = f"g_{'c' if constrain else 'f'}{suffix_extra}"
    m, data, n_used = make_dataset(events, fit_lo, fit_hi, suffix)

    mu_val = float(mu_central if mu_central is not None else M_WR)
    if mu_floating:
        mu = ROOT.RooRealVar(f"mu_{suffix}", "mu", mu_val,
                             float(fit_lo), float(fit_hi))
    else:
        mu = ROOT.RooRealVar(f"mu_{suffix}", "mu", mu_val)
        mu.setConstant(True)
    sigma = ROOT.RooRealVar(
        f"sigma_{suffix}", "sigma",
        float(sigma_pred), 1.0, 10.0 * float(sigma_pred),
    )
    shape = ROOT.RooGaussian(f"gauss_shape_{suffix}", "gauss shape", m, mu, sigma)

    # Extended: yield decoupled from shape. Pure Gaussian; for one-component
    # fits, n_sig_fit = n_used by construction with sqrt(n_used) error.
    n_sig = ROOT.RooRealVar(
        f"n_sig_{suffix}", "signal yield",
        float(n_used), 0.1, 10.0 * float(max(n_used, 1)),
    )
    pdf = ROOT.RooExtendPdf(f"gauss_{suffix}", "ext gauss", shape, n_sig)

    fit_kwargs = [ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-1),
                  ROOT.RooFit.Extended(True)]
    keepalive = [shape]
    constraints = ROOT.RooArgSet()
    if constrain:
        s_obs = ROOT.RooRealVar(f"sigma_obs_{suffix}", "obs", float(sigma_pred))
        s_obs.setConstant(True)
        s_err = ROOT.RooRealVar(f"sigma_err_{suffix}", "err", float(sigma_sigma))
        s_err.setConstant(True)
        prior = ROOT.RooGaussian(
            f"sigma_prior_{suffix}", "sigma prior", s_obs, sigma, s_err,
        )
        keepalive.extend([s_obs, s_err, prior])
        constraints.add(prior)
    if mu_constrained:
        mu_obs = ROOT.RooRealVar(f"mu_obs_{suffix}", "mu obs", mu_val)
        mu_obs.setConstant(True)
        mu_err = ROOT.RooRealVar(f"mu_err_{suffix}", "mu err", float(mu_sigma))
        mu_err.setConstant(True)
        mu_prior = ROOT.RooGaussian(
            f"mu_prior_{suffix}", "mu prior", mu_obs, mu, mu_err,
        )
        keepalive.extend([mu_obs, mu_err, mu_prior])
        constraints.add(mu_prior)
    if constraints.getSize() > 0:
        fit_kwargs.append(ROOT.RooFit.ExternalConstraints(constraints))

    fit_result = pdf.fitTo(data, *fit_kwargs)

    return {
        "model": "gauss",
        "constrain": constrain,
        "width_mode": width_mode,
        "mu_mode": mu_mode,
        "params": {
            "mu": float(mu.getVal()),
            "sigma": float(sigma.getVal()),
            "n_sig": float(n_sig.getVal()),
        },
        "errors": {
            "mu": float(mu.getError()) if mu_floating else 0.0,
            "sigma": float(sigma.getError()),
            "n_sig": float(n_sig.getError()),
        },
        "width_prior": {
            "central": float(sigma_pred),
            "uncertainty": float(sigma_sigma),
            "label": "sigma",
        },
        "mu_prior": {
            "central": mu_val,
            "uncertainty": float(mu_sigma) if mu_constrained else 0.0,
            "constrained": mu_constrained,
        },
        "n_injected": int(n_used),
        "n_used": n_used,
        "minuit_status": int(fit_result.status()),
        "covqual": int(fit_result.covQual()),
        "min_nll": float(fit_result.minNll()),
        "ndof_free": 2 + (1 if mu_constrained else 0),
    }


def fit_bifurcated_gaussian(
    events, M_WR, fwhm_pred, fwhm_err, fit_lo, fit_hi, *,
    width_mode: str = "constrained",
    mu_mode: str = "fixed",
    mu_central: float | None = None,
    mu_sigma: float | None = None,
    suffix_extra: str = "",
):
    sigma_pred = fwhm_pred / FWHM_TO_BIFUR_SIGMA   # = sigma_L + sigma_R for the bifur
    sigma_sigma = fwhm_err / FWHM_TO_BIFUR_SIGMA
    constrain = (width_mode == "constrained")
    mu_constrained = (mu_mode == "constrained")
    mu_floating = (mu_mode in ("free", "constrained"))
    suffix = f"b_{'c' if constrain else 'f'}{suffix_extra}"
    m, data, n_used = make_dataset(events, fit_lo, fit_hi, suffix)

    mu_val = float(mu_central if mu_central is not None else M_WR)
    if mu_floating:
        mu = ROOT.RooRealVar(f"mu_{suffix}", "mu", mu_val,
                             float(fit_lo), float(fit_hi))
    else:
        mu = ROOT.RooRealVar(f"mu_{suffix}", "mu", mu_val)
        mu.setConstant(True)

    Sigma = ROOT.RooRealVar(
        f"Sigma_{suffix}", "sigma_L + sigma_R",
        float(sigma_pred), 1.0, 10.0 * float(sigma_pred),
    )
    Delta = ROOT.RooRealVar(
        f"Delta_{suffix}", "sigma_R - sigma_L",
        0.0, -float(sigma_pred), float(sigma_pred),
    )
    sigmaL = ROOT.RooFormulaVar(
        f"sigmaL_{suffix}", "max(1.0, 0.5*(@0 - @1))",
        ROOT.RooArgList(Sigma, Delta),
    )
    sigmaR = ROOT.RooFormulaVar(
        f"sigmaR_{suffix}", "max(1.0, 0.5*(@0 + @1))",
        ROOT.RooArgList(Sigma, Delta),
    )
    shape = ROOT.RooBifurGauss(f"bifur_shape_{suffix}", "bifur shape", m, mu, sigmaL, sigmaR)

    n_sig = ROOT.RooRealVar(
        f"n_sig_{suffix}", "signal yield",
        float(n_used), 0.1, 10.0 * float(max(n_used, 1)),
    )
    pdf = ROOT.RooExtendPdf(f"bifur_{suffix}", "ext bifur", shape, n_sig)

    fit_kwargs = [ROOT.RooFit.Save(True), ROOT.RooFit.PrintLevel(-1),
                  ROOT.RooFit.Extended(True)]
    keepalive = [shape]
    constraints = ROOT.RooArgSet()
    if constrain:
        s_obs = ROOT.RooRealVar(f"Sigma_obs_{suffix}", "obs", float(sigma_pred))
        s_obs.setConstant(True)
        s_err = ROOT.RooRealVar(f"Sigma_err_{suffix}", "err", float(sigma_sigma))
        s_err.setConstant(True)
        prior = ROOT.RooGaussian(
            f"Sigma_prior_{suffix}", "Sigma prior", s_obs, Sigma, s_err,
        )
        keepalive.extend([s_obs, s_err, prior])
        constraints.add(prior)
    if mu_constrained:
        mu_obs = ROOT.RooRealVar(f"mu_obs_{suffix}", "mu obs", mu_val)
        mu_obs.setConstant(True)
        mu_err = ROOT.RooRealVar(f"mu_err_{suffix}", "mu err", float(mu_sigma))
        mu_err.setConstant(True)
        mu_prior = ROOT.RooGaussian(
            f"mu_prior_{suffix}", "mu prior", mu_obs, mu, mu_err,
        )
        keepalive.extend([mu_obs, mu_err, mu_prior])
        constraints.add(mu_prior)
    if constraints.getSize() > 0:
        fit_kwargs.append(ROOT.RooFit.ExternalConstraints(constraints))

    fit_result = pdf.fitTo(data, *fit_kwargs)

    return {
        "model": "bifur",
        "constrain": constrain,
        "width_mode": width_mode,
        "mu_mode": mu_mode,
        "params": {
            "mu": float(mu.getVal()),
            "Sigma": float(Sigma.getVal()),
            "Delta": float(Delta.getVal()),
            "sigmaL": float(sigmaL.getVal()),
            "sigmaR": float(sigmaR.getVal()),
            "n_sig": float(n_sig.getVal()),
        },
        "errors": {
            "mu": float(mu.getError()) if mu_floating else 0.0,
            "Sigma": float(Sigma.getError()),
            "Delta": float(Delta.getError()),
            "n_sig": float(n_sig.getError()),
        },
        "width_prior": {
            "central": float(sigma_pred),
            "uncertainty": float(sigma_sigma),
            "label": "Sigma",
        },
        "mu_prior": {
            "central": mu_val,
            "uncertainty": float(mu_sigma) if mu_constrained else 0.0,
            "constrained": mu_constrained,
        },
        "n_injected": int(n_used),
        "n_used": n_used,
        "minuit_status": int(fit_result.status()),
        "covqual": int(fit_result.covQual()),
        "min_nll": float(fit_result.minNll()),
        "ndof_free": 3 + (1 if mu_constrained else 0),
    }


def run_fit(model, width_mode, events, M_WR, fwhm_pred, fwhm_err, fit_lo, fit_hi,
            mu_mode="fixed", mu_central=None, mu_sigma=None, suffix_extra=""):
    """Dispatch to gauss/bifur fit functions, forwarding both width and mu modes."""
    if model == "gauss":
        return fit_single_gaussian(
            events, M_WR, fwhm_pred, fwhm_err, fit_lo, fit_hi,
            width_mode=width_mode, mu_mode=mu_mode,
            mu_central=mu_central, mu_sigma=mu_sigma, suffix_extra=suffix_extra,
        )
    if model == "bifur":
        return fit_bifurcated_gaussian(
            events, M_WR, fwhm_pred, fwhm_err, fit_lo, fit_hi,
            width_mode=width_mode, mu_mode=mu_mode,
            mu_central=mu_central, mu_sigma=mu_sigma, suffix_extra=suffix_extra,
        )
    raise ValueError(f"unknown model {model!r}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def evaluate_fit_curve(fit, xs, M_WR):
    """Return the fitted PDF (normalized to unit integral over xs).

    Uses the fitted/fixed mu from the result dict (which may differ from M_WR
    when --peak-mode fix-pred is in effect).
    """
    mu = fit["params"]["mu"]
    if fit["model"] == "gauss":
        sigma = fit["params"]["sigma"]
        y = np.exp(-0.5 * ((xs - mu) / sigma) ** 2)
    elif fit["model"] == "bifur":
        sL = fit["params"]["sigmaL"]
        sR = fit["params"]["sigmaR"]
        left = xs < mu
        y = np.empty_like(xs)
        y[left]  = np.exp(-0.5 * ((xs[left]  - mu) / sL) ** 2)
        y[~left] = np.exp(-0.5 * ((xs[~left] - mu) / sR) ** 2)
    else:
        raise ValueError(fit["model"])
    return y / np.trapz(y, xs)


def plot_one_config(ax, fit, events, edges, vals, *,
                    M_WR, M_N, fit_lo, fit_hi,
                    bin_width, n_events,
                    channel, era, com,
                    panel_label: str | None = None):
    """CMS-styled per-panel plot for one (model, mode) of the 2x2 comparison.

    `panel_label` overrides the default header in the parameter box; used
    by the prior-grid 2x2 plots ("No Priors", "µ constrained", etc.).
    """
    sigma_pred = fit["width_prior"]["central"]
    sigma_sigma = fit["width_prior"]["uncertainty"]
    n_bins = int(round((fit_hi - fit_lo) / bin_width))
    plot_edges = np.linspace(fit_lo, fit_hi, n_bins + 1)
    plot_centers = 0.5 * (plot_edges[:-1] + plot_edges[1:])

    # MC truth
    centers = 0.5 * (edges[:-1] + edges[1:])
    mc_per_plot_bin = np.zeros(n_bins)
    for i in range(n_bins):
        in_b = (centers >= plot_edges[i]) & (centers < plot_edges[i + 1])
        mc_per_plot_bin[i] = float(vals[in_b].sum())
    mc_total = mc_per_plot_bin.sum()
    if mc_total > 0:
        mc_scaled = mc_per_plot_bin / mc_total * n_events
        ax.stairs(mc_scaled, plot_edges, color="black", alpha=0.35,
                  linewidth=1.5, label=f"MC shape (scaled to {n_events} events)")

    # Toy data
    h_toy, _ = np.histogram(events, bins=plot_edges)
    h_toy_err = np.sqrt(np.maximum(h_toy, 0))
    ax.errorbar(
        plot_centers, h_toy, yerr=h_toy_err,
        marker="o", linestyle="", color="black", markersize=6,
        label=f"toy ({n_events} events)",
    )

    # Fit curve
    xs = np.linspace(fit_lo, fit_hi, 400)
    pdf = evaluate_fit_curve(fit, xs, M_WR)
    if panel_label is not None:
        # Panel-mode legend label: PDF identity, used as the only legend entry.
        fit_label = "Gaussian Fit" if fit["model"] == "gauss" else "Bifurcated Gaussian Fit"
    else:
        fit_label = "fit (Gauss)" if fit["model"] == "gauss" else "fit (BifurGauss)"
    fit_line, = ax.plot(xs, pdf * n_events * bin_width, color="red",
                        linewidth=2, label=fit_label)

    ax.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
    ax.set_ylabel(f"Events / {bin_width:.0f} GeV")
    ax.set_xlim(fit_lo, fit_hi)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)

    hep.cms.label(loc=0, ax=ax, data=False,
                  label="Work in Progress", com=com, fontsize=18)

    # Top-left: channel / region / era / masses. (mu line shown in stat box for
    # panel mode; legacy single-panel mode still includes a mu_line below.)
    ch = {"ee": "ee", "mumu": r"$\mu\mu$"}[channel]
    if panel_label is None:
        mu_fit = fit["params"]["mu"]
        mu_err = fit["errors"]["mu"]
        mu_mode_fit = fit.get("mu_mode", "fixed")
        mu_prior = fit.get("mu_prior", {})
        if mu_mode_fit == "constrained":
            mu_pred = mu_prior["central"]; mu_pred_err = mu_prior["uncertainty"]
            mu_line = (
                rf"$\mu_{{\mathrm{{pred}}}}={mu_pred:.0f}\pm{mu_pred_err:.1f}$, "
                rf"$\mu_{{\mathrm{{fit}}}}={mu_fit:.0f}\pm{mu_err:.1f}$"
            )
        elif mu_mode_fit == "free":
            mu_line = rf"$\mu_{{\mathrm{{fit}}}}={mu_fit:.0f}\pm{mu_err:.1f}$ (free)"
        elif abs(mu_fit - M_WR) < 0.5:
            mu_line = rf"$\mu = M_{{W_R}}$ (fixed)"
        else:
            mu_line = rf"$\mu = {mu_fit:.0f}$ GeV (fixed, peak from MC)"
        top_left = (
            f"{ch}\nResolved SR\n{era}\n"
            rf"$M_{{W_R}}={M_WR:.0f}$ GeV, $M_N={M_N:.0f}$ GeV"
            f"\n{mu_line}"
        )
    else:
        top_left = (
            f"{ch}\nResolved SR\n{era}\n"
            rf"$M_{{W_R}}={M_WR:.0f}$ GeV, $M_N={M_N:.0f}$ GeV"
        )
    ax.text(
        0.04, 0.96, top_left,
        transform=ax.transAxes, fontsize=12, verticalalignment="top",
    )

    # Top-right: stat box.
    pdf_name = "Single Gaussian" if fit["model"] == "gauss" else "Bifurcated Gaussian"
    n_obs = fit.get("n_used", fit.get("n_injected", n_events))
    n_fit_line = (
        rf"$N_{{\mathrm{{fit}}}} = {fit['params']['n_sig']:.1f} \pm {fit['errors']['n_sig']:.1f}$"
        rf" ($N_{{\mathrm{{obs}}}}={n_obs}$)"
    )
    if panel_label is None:
        # Legacy detailed stat box for the single-panel --only-config plots.
        mode_name = "constrained" if fit["constrain"] else "free width"
        header = f"{pdf_name} ({mode_name})"
        if fit["model"] == "gauss":
            param_text = (
                f"{header}"
                "\n"
                rf"$\sigma_{{\mathrm{{pred}}}} = {sigma_pred:.1f} \pm {sigma_sigma:.1f}$ GeV"
                "\n"
                rf"$\sigma_{{\mathrm{{fit}}}} = {fit['params']['sigma']:.1f} \pm {fit['errors']['sigma']:.1f}$"
                "\n"
                f"{n_fit_line}"
            )
        else:
            param_text = (
                f"{header}"
                "\n"
                rf"$\Sigma_{{\mathrm{{pred}}}} = {sigma_pred:.1f} \pm {sigma_sigma:.1f}$ GeV"
                "\n"
                rf"$\Sigma_{{\mathrm{{fit}}}} = {fit['params']['Sigma']:.1f} \pm {fit['errors']['Sigma']:.1f}$"
                "\n"
                rf"$\Delta_{{\mathrm{{fit}}}} = {fit['params']['Delta']:.1f} \pm {fit['errors']['Delta']:.1f}$"
                "\n"
                rf"$\sigma_L = {fit['params']['sigmaL']:.1f}$, $\sigma_R = {fit['params']['sigmaR']:.1f}$"
                "\n"
                f"{n_fit_line}"
            )
        ax.text(
            0.96, 0.96, param_text,
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.85),
        )
    else:
        # Prior-grid 2x2: top-right column from top to bottom is
        #   1) single-entry legend (PDF identity + fit line),
        #   2) compact stat box: functional form + parameter values,
        #   3) HUGE bold black panel label.
        ax.legend(handles=[fit_line], loc="upper right",
                  bbox_to_anchor=(0.96, 0.96), fontsize=14, framealpha=0.85)
        if fit["model"] == "gauss":
            lines = [
                r"$f(m) \propto \exp(-(m-\mu)^2 / (2\sigma^2))$",
                rf"$\mu = {fit['params']['mu']:.1f} \pm {fit['errors']['mu']:.1f}$ GeV",
                rf"$\sigma = {fit['params']['sigma']:.1f} \pm {fit['errors']['sigma']:.1f}$ GeV",
            ]
        else:
            lines = [
                r"$f(m) \propto \exp(-(m-\mu)^2 / (2\sigma_{L/R}^2))$",
                r"$\sigma_L = (\Sigma - \Delta)/2,\ \sigma_R = (\Sigma + \Delta)/2$",
                rf"$\mu = {fit['params']['mu']:.1f} \pm {fit['errors']['mu']:.1f}$ GeV",
                rf"$\Sigma = {fit['params']['Sigma']:.1f} \pm {fit['errors']['Sigma']:.1f}$ GeV",
                rf"$\Delta = {fit['params']['Delta']:.1f} \pm {fit['errors']['Delta']:.1f}$ GeV",
            ]
        if fit["minuit_status"] != 0 or fit["covqual"] < 3:
            lines.append(
                rf"[status={fit['minuit_status']}, cov={fit['covqual']}]"
            )
        param_text = "\n".join(lines)
        ax.text(
            0.96, 0.85, param_text,
            transform=ax.transAxes, fontsize=14, verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.85),
        )
        # HUGE black label below the stat box.
        ax.text(
            0.96, 0.55, panel_label,
            transform=ax.transAxes, fontsize=30, fontweight="bold",
            color="black",
            verticalalignment="top", horizontalalignment="right",
        )
        return  # skip the legacy bottom-center legend below
    ax.legend(loc="lower center", fontsize=11)


def plot_single_panel(
    fit, events, edges, vals, *,
    output_path, channel, era, com,
    M_WR, M_N, n_events, bin_width=200.0,
):
    """Original fit_bifur_toy.py-style single-panel plot, generalized for either PDF."""
    fit_lo = ONSHELL_WINDOW_LO_FRAC * M_WR
    fit_hi = ONSHELL_WINDOW_HI_FRAC * M_WR

    n_bins = int(round((fit_hi - fit_lo) / bin_width))
    plot_edges = np.linspace(fit_lo, fit_hi, n_bins + 1)
    plot_centers = 0.5 * (plot_edges[:-1] + plot_edges[1:])

    hep.style.use("CMS")
    fig, ax = plt.subplots()

    centers = 0.5 * (edges[:-1] + edges[1:])
    mc_per_plot_bin = np.zeros(n_bins)
    for i in range(n_bins):
        in_b = (centers >= plot_edges[i]) & (centers < plot_edges[i + 1])
        mc_per_plot_bin[i] = float(vals[in_b].sum())
    mc_total = mc_per_plot_bin.sum()
    if mc_total > 0:
        mc_scaled = mc_per_plot_bin / mc_total * n_events
        ax.stairs(mc_scaled, plot_edges, color="black", alpha=0.35,
                  linewidth=1.5, label=f"MC shape (scaled to {n_events} events)")

    h_toy, _ = np.histogram(events, bins=plot_edges)
    h_toy_err = np.sqrt(np.maximum(h_toy, 0))
    ax.errorbar(
        plot_centers, h_toy, yerr=h_toy_err,
        marker="o", linestyle="", color="black", markersize=6,
        label=f"toy ({n_events} events)",
    )

    xs = np.linspace(fit_lo, fit_hi, 400)
    pdf = evaluate_fit_curve(fit, xs, M_WR)
    fit_label = "fit (Gauss)" if fit["model"] == "gauss" else "fit (BifurGauss)"
    ax.plot(xs, pdf * n_events * bin_width, color="red", linewidth=2, label=fit_label)

    ax.set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
    ax.set_ylabel(f"Events / {bin_width:.0f} GeV")
    ax.set_xlim(fit_lo, fit_hi)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    hep.cms.label(loc=0, ax=ax, data=False,
                  label="Work in Progress", com=com, fontsize=20)

    ch = {"ee": "ee", "mumu": r"$\mu\mu$"}[channel]
    mu_fit = fit["params"]["mu"]
    mu_err = fit["errors"]["mu"]
    mu_mode_fit = fit.get("mu_mode", "fixed")
    mu_prior = fit.get("mu_prior", {})
    if mu_mode_fit == "constrained":
        mu_pred = mu_prior["central"]; mu_pred_err = mu_prior["uncertainty"]
        mu_line = (
            rf"$\mu_{{\mathrm{{pred}}}}={mu_pred:.0f}\pm{mu_pred_err:.1f}$, "
            rf"$\mu_{{\mathrm{{fit}}}}={mu_fit:.0f}\pm{mu_err:.1f}$"
        )
    elif mu_mode_fit == "free":
        mu_line = rf"$\mu_{{\mathrm{{fit}}}}={mu_fit:.0f}\pm{mu_err:.1f}$ (free)"
    elif abs(mu_fit - M_WR) < 0.5:
        mu_line = rf"$\mu = M_{{W_R}}$ (fixed)"
    else:
        mu_line = rf"$\mu = {mu_fit:.0f}$ GeV (fixed, peak from MC)"
    ax.text(
        0.04, 0.96,
        f"{ch}\nResolved SR\n{era}\n"
        rf"$M_{{W_R}}={M_WR:.0f}$ GeV, $M_N={M_N:.0f}$ GeV"
        f"\n{mu_line}",
        transform=ax.transAxes, fontsize=14, verticalalignment="top",
    )

    sigma_pred = fit["width_prior"]["central"]
    sigma_sigma = fit["width_prior"]["uncertainty"]
    n_obs = fit.get("n_used", fit.get("n_injected", n_events))
    n_fit_line = (
        rf"$N_{{\mathrm{{fit}}}} = {fit['params']['n_sig']:.1f} \pm {fit['errors']['n_sig']:.1f}$"
        rf" ($N_{{\mathrm{{obs}}}}={n_obs}$)"
    )
    if fit["model"] == "gauss":
        param_text = (
            rf"$\sigma_{{\mathrm{{pred}}}} = {sigma_pred:.1f} \pm {sigma_sigma:.1f}$ GeV"
            "\n"
            rf"$\sigma_{{\mathrm{{fit}}}} = {fit['params']['sigma']:.1f} \pm {fit['errors']['sigma']:.1f}$"
            "\n"
            f"{n_fit_line}"
        )
    else:
        param_text = (
            rf"$\Sigma_{{\mathrm{{pred}}}} = {sigma_pred:.1f} \pm {sigma_sigma:.1f}$ GeV"
            "\n"
            rf"$\Sigma_{{\mathrm{{fit}}}} = {fit['params']['Sigma']:.1f} \pm {fit['errors']['Sigma']:.1f}$"
            "\n"
            rf"$\Delta_{{\mathrm{{fit}}}} = {fit['params']['Delta']:.1f} \pm {fit['errors']['Delta']:.1f}$"
            "\n"
            rf"$\sigma_L = {fit['params']['sigmaL']:.1f}$, $\sigma_R = {fit['params']['sigmaR']:.1f}$"
            "\n"
            f"{n_fit_line}"
        )
    ax.text(
        0.96, 0.96, param_text,
        transform=ax.transAxes, fontsize=12, verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="gray", alpha=0.85),
    )
    ax.legend(loc="lower center", fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def make_compare_plot(
    fits, events, edges, vals, *,
    output_path, channel, era, com,
    M_WR, M_N, n_events, bin_width=200.0,
    model: str | None = None,
):
    """Single-panel for --only-config; otherwise a 2x2 prior-grid for one model.

    Prior grid layout (columns = mu treatment, rows = width treatment):
        ┌──────────────────────┬──────────────────────┐
        │ no_priors            │ mu_only              │
        │ (mu free, w free)    │ (mu prior, w free)   │
        ├──────────────────────┼──────────────────────┤
        │ width_only           │ both                 │
        │ (mu free, w prior)   │ (mu prior, w prior)  │
        └──────────────────────┴──────────────────────┘
    """
    fit_lo = ONSHELL_WINDOW_LO_FRAC * M_WR
    fit_hi = ONSHELL_WINDOW_HI_FRAC * M_WR
    hep.style.use("CMS")

    if len(fits) == 1:
        # Legacy single-panel path (called via --only-config).
        first = next(iter(fits.values()))
        plot_single_panel(
            first, events, edges, vals,
            output_path=output_path,
            channel=channel, era=era, com=com,
            M_WR=M_WR, M_N=M_N, n_events=n_events, bin_width=bin_width,
        )
        return

    # 2x2 prior grid for one model (gauss or bifur).
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), sharex=True, sharey=True)
    # COMPARE_CONFIGS is in order: no_priors, mu_only, width_only, both
    flat_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    for ax, (cfg_name, _mu_mode, _wm, label) in zip(flat_axes, COMPARE_CONFIGS):
        fit = fits[cfg_name]
        plot_one_config(
            ax, fit, events, edges, vals,
            M_WR=M_WR, M_N=M_N,
            fit_lo=fit_lo, fit_hi=fit_hi,
            bin_width=bin_width, n_events=n_events,
            channel=channel, era=era, com=com,
            panel_label=label,
        )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--dir", default="20260406_signals")
    p.add_argument("--channel", choices=["ee", "mumu"], default="ee")
    p.add_argument("--wr", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--n-events", type=int, default=20)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--results-json", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--topology", choices=["resolved", "boosted"], default="resolved")
    p.add_argument(
        "--only-config",
        choices=[f"{m}_{mode}" for m, mode in CONFIGS], default=None,
        help="If set, run/plot only this single config (single-panel PDF).",
    )
    p.add_argument(
        "--systematic-floor-gev", type=float, default=None,
        help=f"Override SYSTEMATIC_FWHM_FLOOR_GEV (default: {SYSTEMATIC_FWHM_FLOOR_GEV}). "
             "Use 0 for the pre-floor legacy state, 50 for the older intermediate state.",
    )
    p.add_argument(
        "--legacy-bifur-conversion", action="store_true",
        help="Use the buggy bifur conversion FWHM/2.3548 (instead of FWHM/1.1774). "
             "For reproducing pre-fix bifur prior values only.",
    )
    p.add_argument(
        "--peak-mode", choices=["fix-mwr", "fix-pred", "constrain-pred"],
        default="constrain-pred",
        help="Treatment of mu in the toy fit. "
             "'constrain-pred': mu floats with Gaussian prior N(peak_onshell, sigma_peak) (default). "
             "sigma_peak = bootstrap stdev (100 toys) ⊕ bin-resolution floor Δm/√12 in quadrature. "
             "'fix-pred': mu = peak_onshell fixed. "
             "'fix-mwr': mu = M_WR fixed (legacy).",
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)

    # Apply legacy-mode overrides (these shift module-level constants used by
    # predict_fwhm and the bifur fit; they must run before any fit work).
    global SYSTEMATIC_FWHM_FLOOR_GEV, FWHM_TO_BIFUR_SIGMA
    if args.systematic_floor_gev is not None:
        SYSTEMATIC_FWHM_FLOOR_GEV = float(args.systematic_floor_gev)
        logger.info("Override: SYSTEMATIC_FWHM_FLOOR_GEV = %.1f GeV",
                    SYSTEMATIC_FWHM_FLOOR_GEV)
    if args.legacy_bifur_conversion:
        FWHM_TO_BIFUR_SIGMA = FWHM_TO_GAUSS_SIGMA
        logger.info("Override: legacy bifur conversion (FWHM_TO_BIFUR_SIGMA = "
                    "FWHM_TO_GAUSS_SIGMA = %.4f)", FWHM_TO_GAUSS_SIGMA)

    era = args.era
    M_WR = float(args.wr)
    M_N = float(args.n)
    x = M_N / M_WR
    channel = args.channel
    sig_tag = f"WR{args.wr}_N{args.n}"

    info = load_lumi(era)
    com = info.get("com", 13.0)

    results_json = Path(args.results_json or
        repo_root() / "signal_fitting" / "outputs" / era / "fwhm" / "fits" / "results.json")
    with open(results_json) as f:
        results = json.load(f)
    model_entry = results[channel]["models"]["a_linear"]
    fwhm_param, fwhm_err = predict_fwhm(model_entry, x, M_WR)

    input_dirs, _ = input_dirs_for_era(era, repo_root(), args.dir)
    region = build_region_name(channel, args.topology)
    mass_var = "mass_twoobject" if args.topology == "boosted" else "mass_fourobject"
    hist_key = build_hist_key(region, mass_var)
    edges_native, vals_native, var_native = load_and_combine_signal(
        input_dirs, hist_key, sig_tag,
    )
    # Native binning (10 GeV) for toy event sampling — gives finer resolution.
    # Rebinned histogram (60 GeV, matching Phase 1) for peak measurement and
    # peak bootstrap, so they are consistent with how Phase 1 measured FWHM.
    edges, vals, _ = rebin_histogram(edges_native, vals_native, var_native, 6)
    logger.info(
        "MC histogram: native %d bins (10 GeV), rebinned to %d bins (60 GeV) for shape measurements; integral=%.1f",
        len(vals_native), len(vals), float(vals.sum()),
    )

    # Two FWHM values are in play:
    #   fwhm_param   = parameterization a*x + b*M_WR (used for the µ prior)
    #   fwhm_boot    = per-cell bootstrap-mean MC FWHM (used for width prior)
    # The µ-prior calibration (MU_PRIOR_ALPHA) was scanned against fwhm_param;
    # the width-prior calibration (WIDTH_PRIOR_ALPHA) was scanned with fwhm_boot
    # as the prior central. Keep them separate so neither calibration shifts.
    fwhm_boot, fwhm_boot_unc = bootstrap_fwhm_estimate(
        edges, vals, sig_tag, channel, args.topology, n_toys=100, seed=args.seed,
    )
    width_alpha = WIDTH_PRIOR_ALPHA[channel]
    # Below run_fit takes (fwhm_pred, fwhm_err) as the *width* prior. The
    # central is FWHM_boot (per-cell MC truth); the σ is α * FWHM_param
    # (the parameterization), matching the µ-prior σ convention.
    fwhm_pred = fwhm_boot                        # width-prior central
    fwhm_err  = width_alpha * fwhm_param         # width-prior σ
    logger.info(
        "FWHM: param=%.1f ± %.1f GeV (a*x+b*M_WR, used for µ prior); "
        "boot=%.1f ± %.1f GeV (per-cell MC bootstrap, used for width prior)",
        fwhm_param, fwhm_err, fwhm_boot, fwhm_boot_unc,
    )
    logger.info(
        "  -> width prior: central=%.1f GeV (FWHM_boot), "
        "sigma=%.2f*FWHM_param=%.1f GeV",
        fwhm_boot, width_alpha, fwhm_err,
    )

    # Peak from MC histogram. Bootstrap to get (central, sigma):
    #   central = mean of peak_onshell across 100 Poisson-resampled toys
    #             (smoother than the single-realization measurement; reflects
    #              that this MC sample is a noisy proxy for the true peak,
    #              especially since real data fit has different statistics).
    #   sigma   = bootstrap stdev ⊕ binning-floor (Δm/√12) in quadrature.
    sp = compute_shape_params(edges, vals, sig_tag, channel, args.topology)
    peak_orig = sp.peak_onshell
    mu_boot, sigma_peak_boot = bootstrap_peak_estimate(
        edges, vals, sig_tag, channel, args.topology,
        n_toys=100, seed=args.seed,
    )
    mu_sigma: float | None = None
    mu_alpha = MU_PRIOR_ALPHA[channel]
    if args.peak_mode == "fix-mwr":
        mu_central = M_WR
    elif args.peak_mode == "fix-pred":
        mu_central = mu_boot
    else:  # constrain-pred
        mu_central = mu_boot
        # FWHM-proportional µ prior; calibrated against fwhm_param (the
        # parameterization a*x + b*M_WR), not fwhm_boot. See MU_PRIOR_ALPHA.
        mu_sigma = mu_alpha * fwhm_param
    logger.info(
        "Peak: M_WR=%.0f, peak_orig=%.0f, peak_boot=%.1f ± %.1f, "
        "--peak-mode=%s -> mu_central=%.1f%s",
        M_WR, peak_orig, mu_boot, sigma_peak_boot,
        args.peak_mode, mu_central,
        f" (constrained, sigma_mu={mu_sigma:.1f}={mu_alpha:.2f}*FWHM_param)"
        if mu_sigma is not None else "",
    )

    fit_lo = ONSHELL_WINDOW_LO_FRAC * M_WR
    fit_hi = ONSHELL_WINDOW_HI_FRAC * M_WR
    # Sample toy events from the NATIVE-binned histogram (10 GeV) for finer
    # event-position resolution than the 60 GeV shape-measurement histogram.
    centers_n = 0.5 * (edges_native[:-1] + edges_native[1:])
    in_win = (centers_n >= fit_lo) & (centers_n <= fit_hi)
    where = np.where(in_win)[0]
    edges_win = edges_native[where[0]:where[-1] + 2]
    vals_win = vals_native[where]

    rng = np.random.default_rng(args.seed)
    events = sample_from_hist(edges_win, vals_win, args.n_events, rng)
    logger.info(
        "Sampled %d events in [%.0f, %.0f] GeV (range %.0f .. %.0f)",
        len(events), fit_lo, fit_hi, float(events.min()), float(events.max()),
    )

    out_dir = Path(args.output_dir or
        repo_root() / "signal_fitting" / "outputs" / era / "signal_toy_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    common_prior = {
        "FWHM_prior": {
            "FWHM_pred_GeV": float(fwhm_pred),
            "FWHM_err_GeV": float(fwhm_err),
            "systematic_floor_GeV": SYSTEMATIC_FWHM_FLOOR_GEV,
            "gauss_sigma_pred": float(fwhm_pred / FWHM_TO_GAUSS_SIGMA),
            "gauss_sigma_sigma": float(fwhm_err / FWHM_TO_GAUSS_SIGMA),
            "bifur_Sigma_pred": float(fwhm_pred / FWHM_TO_BIFUR_SIGMA),
            "bifur_Sigma_sigma": float(fwhm_err / FWHM_TO_BIFUR_SIGMA),
        },
        "mu_prior": {
            "peak_orig": float(peak_orig),
            "peak_boot_mean": float(mu_boot),
            "sigma_peak_boot_GeV": float(sigma_peak_boot),
        },
    }

    if args.only_config is not None:
        # Legacy single-panel path: use --peak-mode to determine mu treatment.
        m, mode = args.only_config.split("_", 1)
        # Translate legacy "constrain" -> new "constrained" spelling.
        width_mode_arg = "constrained" if mode == "constrain" else "free"
        logger.info("Fitting: %s / %s (single-panel)", m, mode)
        fit = run_fit(
            m, width_mode_arg, events, M_WR, fwhm_pred, fwhm_err, fit_lo, fit_hi,
            mu_mode=("constrained" if mu_sigma is not None else "fixed"),
            mu_central=mu_central, mu_sigma=mu_sigma,
        )
        fits = {(m, mode): fit}
        stem = f"signal_toy_{args.only_config}_{sig_tag}_{channel}_seed{args.seed}_n{args.n_events}"
        make_compare_plot(
            fits, events, edges, vals,
            output_path=out_dir / f"{stem}.pdf",
            channel=channel, era=era, com=com,
            M_WR=M_WR, M_N=M_N, n_events=args.n_events,
        )
        summary = {
            "signal_tag": sig_tag, "channel": channel, "era": era,
            "M_WR": M_WR, "M_N": M_N, "x": x,
            "n_events": int(args.n_events), "seed": int(args.seed),
            **common_prior,
            "events": [float(e) for e in events],
            "fits": {f"{m}_{mode}": fit},
        }
        with open(out_dir / f"{stem}.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)
        logger.info("Wrote %s", out_dir / f"{stem}.pdf")
        return

    # Default: 2x2 prior-grid comparison, one plot per PDF (gauss + bifur).
    # mu_central / mu_sigma always come from the per-point bootstrap
    # (mu_central = bootstrap mean; mu_sigma = bootstrap stdev ⊕ Δm/√12).
    for model in ("gauss", "bifur"):
        fits_grid: dict[str, dict] = {}
        for cfg_name, mu_mode_cfg, width_mode_cfg, _label in COMPARE_CONFIGS:
            logger.info("Fitting: %s / %s (mu=%s, width=%s)",
                        model, cfg_name, mu_mode_cfg, width_mode_cfg)
            fit = run_fit(
                model, width_mode_cfg, events, M_WR, fwhm_pred, fwhm_err,
                fit_lo, fit_hi,
                mu_mode=mu_mode_cfg, mu_central=mu_boot, mu_sigma=sigma_peak_boot,
                suffix_extra=f"_{cfg_name}",
            )
            fits_grid[cfg_name] = fit
            if model == "gauss":
                logger.info(
                    "  -> mu=%.1f±%.1f, sigma=%.1f±%.1f, NLL=%.2f",
                    fit["params"]["mu"], fit["errors"]["mu"],
                    fit["params"]["sigma"], fit["errors"]["sigma"],
                    fit["min_nll"],
                )
            else:
                logger.info(
                    "  -> mu=%.1f±%.1f, Sigma=%.1f±%.1f, Delta=%.1f±%.1f, NLL=%.2f",
                    fit["params"]["mu"], fit["errors"]["mu"],
                    fit["params"]["Sigma"], fit["errors"]["Sigma"],
                    fit["params"]["Delta"], fit["errors"]["Delta"],
                    fit["min_nll"],
                )

        stem = f"signal_toy_compare_{model}_{sig_tag}_{channel}_seed{args.seed}_n{args.n_events}"
        make_compare_plot(
            fits_grid, events, edges, vals,
            output_path=out_dir / f"{stem}.pdf",
            channel=channel, era=era, com=com,
            M_WR=M_WR, M_N=M_N, n_events=args.n_events,
            model=model,
        )
        summary = {
            "signal_tag": sig_tag, "channel": channel, "era": era,
            "M_WR": M_WR, "M_N": M_N, "x": x,
            "n_events": int(args.n_events), "seed": int(args.seed),
            "model": model,
            **common_prior,
            "events": [float(e) for e in events],
            "fits": fits_grid,
        }
        with open(out_dir / f"{stem}.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)
        logger.info("Wrote %s", out_dir / f"{stem}.pdf")


if __name__ == "__main__":
    main()
