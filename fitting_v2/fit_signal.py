#!/usr/bin/env python3
"""
Stage 1: Characterize signal shape by fitting a parametric model to signal MC m_lljj.

Outputs the fitted mean, sigma, and blinding window [mean - n*sigma, mean + n*sigma]
for use in subsequent sideband background fits.

Usage:
    # Run default models (gaussian, bifur-gauss, crystal-ball):
    python fitting_v2/fit_signal.py \\
        --era RunIII2024Summer24 \\
        --dir 20260223_nlo_dy \\
        --signal WR2000_N1100

    # Run specific model(s):
    python fitting_v2/fit_signal.py \\
        --era RunIII2024Summer24 \\
        --dir 20260223_nlo_dy \\
        --signal WR2000_N1100 \\
        --model voigtian crystal-ball

    # Fine-tune binning or blinding window:
    python fitting_v2/fit_signal.py \\
        --era RunIII2024Summer24 \\
        --dir 20260223_nlo_dy \\
        --signal WR2000_N1100 \\
        --rebin 3 \\
        --n-sigma 2.5
"""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

# Add repo root to path so wrplotter and fitting_v2 imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import ROOT
except ImportError:
    sys.exit(
        "ERROR: PyROOT is not available. Please set up a ROOT-enabled environment.\n"
        "  Options: cmsenv (CMSSW), conda install root, or source LCG views."
    )

from wrplotter.cli_utils import setup_logging
from wrplotter.config import load_lumi
from wrplotter.paths import input_dirs_for_era, repo_root

from fitting_v2.fit_utils import (
    add_common_args,
    build_region_name,
    build_hist_key,
    load_signal,
    create_roodatahist,
    run_fit,
    extract_fit_results,
    save_results_json,
    plot_fit,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_wr_mass(signal_tag: str) -> float:
    """Extract M_WR in GeV from a signal tag like 'WR2000_N1100' → 2000.0."""
    match = re.match(r"WR(\d+)_N\d+", signal_tag)
    if not match:
        raise ValueError(
            f"Cannot parse W_R mass from signal tag '{signal_tag}'. "
            "Expected format: WR<mass>_N<mass>, e.g. WR2000_N1100."
        )
    return float(match.group(1))


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_gaussian_model(
    observable: ROOT.RooRealVar,
    n_data: float,
    m_wr: float,
) -> tuple[ROOT.RooExtendPdf, dict[str, ROOT.RooRealVar]]:
    """Build a simple Gaussian signal model wrapped in RooExtendPdf."""
    mean = ROOT.RooRealVar(
        "mean", "Gaussian mean [GeV]",
        m_wr,
        m_wr * 0.5,
        m_wr * 1.5,
    )
    sigma = ROOT.RooRealVar(
        "sigma", "Gaussian sigma [GeV]",
        m_wr * 0.05,   # initial: 5% of M_WR
        1.0,
        m_wr * 0.5,
    )
    n_sig = ROOT.RooRealVar("n_sig", "signal yield", n_data, 0.0, 1e7)

    shape = ROOT.RooGaussian(
        "gauss", "Gaussian signal shape",
        observable, mean, sigma,
    )
    ROOT.SetOwnership(shape, False)

    pdf = ROOT.RooExtendPdf("ext_gauss", "extended Gaussian", shape, n_sig)

    return pdf, {"mean": mean, "sigma": sigma, "n_sig": n_sig}


def build_crystal_ball_model(
    observable: ROOT.RooRealVar,
    n_data: float,
    m_wr: float,
) -> tuple[ROOT.RooExtendPdf, dict[str, ROOT.RooRealVar]]:
    """Build a Crystal Ball signal model wrapped in RooExtendPdf.

    The Crystal Ball function is a Gaussian core with a power-law tail on the
    low-mass side, which is the standard signal model in HEP for distributions
    with detector resolution effects and radiation losses:

        f(m) = exp(-(m-mean)^2 / 2sigma^2)        if (m-mean)/sigma > -alpha
             = (n/alpha)^n * exp(-alpha^2/2)
               * ((n/alpha - alpha) - (m-mean)/sigma)^-n  otherwise

    Parameters
    ----------
    observable : mass variable (RooRealVar)
    n_data     : total signal events in fit range (used to seed n_sig)
    m_wr       : W_R mass in GeV (used to seed mean and set parameter ranges)

    RooCBShape parameter conventions:
        mean  : peak position
        sigma : Gaussian core width
        alpha : transition point (in units of sigma, on the low side); must be > 0
        cb_n  : power-law exponent of the tail; must be > 0
    """
    mean = ROOT.RooRealVar(
        "mean", "CB mean [GeV]",
        m_wr,
        m_wr * 0.5,
        m_wr * 1.5,
    )
    sigma = ROOT.RooRealVar(
        "sigma", "CB sigma [GeV]",
        m_wr * 0.05,   # initial: 5% of M_WR
        1.0,
        m_wr * 0.5,
    )
    # alpha: where the Gaussian ends and the power-law tail begins.
    # Typical values 0.5–3. Start at 1.5.
    alpha = ROOT.RooRealVar("alpha", "CB alpha", 1.5, 0.1, 10.0)
    # cb_n: power-law exponent. Typical values 1–10. Start at 2.
    cb_n = ROOT.RooRealVar("cb_n", "CB n", 2.0, 0.1, 50.0)
    n_sig = ROOT.RooRealVar("n_sig", "signal yield", n_data, 0.0, 1e7)

    shape = ROOT.RooCBShape(
        "crystal_ball", "Crystal Ball signal shape",
        observable, mean, sigma, alpha, cb_n,
    )
    ROOT.SetOwnership(shape, False)  # prevent Python GC; RooExtendPdf holds a ref

    pdf = ROOT.RooExtendPdf("ext_cb", "extended Crystal Ball", shape, n_sig)

    return pdf, {"mean": mean, "sigma": sigma, "alpha": alpha, "cb_n": cb_n, "n_sig": n_sig}


def build_voigtian_model(
    observable: ROOT.RooRealVar,
    n_data: float,
    m_wr: float,
) -> tuple[ROOT.RooExtendPdf, dict[str, ROOT.RooRealVar]]:
    """Build a Voigtian (Breit-Wigner convolved with Gaussian) signal model.

    RooVoigtian parameter conventions:
        mean  : peak position (resonance mass)
        width : Breit-Wigner natural width (gamma); controls the heavy tail
        sigma : Gaussian resolution width (detector smearing)

    For WR → ℓℓjj the ~13% fractional width arises from cascade decay
    kinematics, not from the resonance width or detector resolution alone.
    The Voigtian distributes this width between the BW and Gaussian components,
    which may give a better shape description than a pure Gaussian or Crystal Ball.
    """
    mean = ROOT.RooRealVar(
        "mean", "Voigtian mean [GeV]",
        m_wr,
        m_wr * 0.5,
        m_wr * 1.5,
    )
    # BW natural width — start at 10% of M_WR; allow up to 50%
    width = ROOT.RooRealVar(
        "width", "BW width [GeV]",
        m_wr * 0.10,
        1.0,
        m_wr * 0.5,
    )
    # Gaussian smearing — start at 5% of M_WR; allow up to 30%
    sigma = ROOT.RooRealVar(
        "sigma", "Gaussian sigma [GeV]",
        m_wr * 0.05,
        1.0,
        m_wr * 0.3,
    )
    n_sig = ROOT.RooRealVar("n_sig", "signal yield", n_data, 0.0, 1e7)

    shape = ROOT.RooVoigtian(
        "voigt", "Voigtian signal shape",
        observable, mean, width, sigma,
    )
    ROOT.SetOwnership(shape, False)

    pdf = ROOT.RooExtendPdf("ext_voigt", "extended Voigtian", shape, n_sig)

    return pdf, {"mean": mean, "width": width, "sigma": sigma, "n_sig": n_sig}


def build_breit_wigner_model(
    observable: ROOT.RooRealVar,
    n_data: float,
    m_wr: float,
) -> tuple[ROOT.RooExtendPdf, dict[str, ROOT.RooRealVar]]:
    """Build a pure Breit-Wigner signal model.

    RooBreitWigner(x, mean, width) — relativistic Breit-Wigner:
        f(m) ∝ 1 / ((m^2 - mean^2)^2 + mean^2 * width^2)

    Note: BW has much heavier tails than a Gaussian. ±2*width contains only
    ~70% of events (vs 95% for ±2σ Gaussian). The blinding window will be
    correspondingly narrower; increase --n-sigma if needed.
    """
    mean = ROOT.RooRealVar(
        "mean", "BW mean [GeV]",
        m_wr,
        m_wr * 0.5,
        m_wr * 1.5,
    )
    # Start at 13% of M_WR (matching observed Gaussian sigma); allow up to 50%
    width = ROOT.RooRealVar(
        "width", "BW width [GeV]",
        m_wr * 0.13,
        1.0,
        m_wr * 0.5,
    )
    n_sig = ROOT.RooRealVar("n_sig", "signal yield", n_data, 0.0, 1e7)

    shape = ROOT.RooBreitWigner(
        "bw", "Breit-Wigner signal shape",
        observable, mean, width,
    )
    ROOT.SetOwnership(shape, False)

    pdf = ROOT.RooExtendPdf("ext_bw", "extended Breit-Wigner", shape, n_sig)

    return pdf, {"mean": mean, "width": width, "n_sig": n_sig}


def build_bifur_gauss_model(
    observable: ROOT.RooRealVar,
    n_data: float,
    m_wr: float,
) -> tuple[ROOT.RooExtendPdf, dict[str, ROOT.RooRealVar]]:
    """Build a bifurcated (asymmetric) Gaussian signal model.

    RooBifurGauss(x, mean, sigmaL, sigmaR) — Gaussian with independent
    widths on each side of the peak:
        f(m) ∝ exp(-(m-mean)^2 / (2*sigmaL^2))   if m < mean
        f(m) ∝ exp(-(m-mean)^2 / (2*sigmaR^2))   if m ≥ mean

    WR → ℓℓjj kinematics produce a low-mass tail from the cascade decay,
    so sigmaL > sigmaR is expected. Blinding window uses each sigma separately:
        lo = mean - n_sigma * sigmaL
        hi = mean + n_sigma * sigmaR
    """
    mean = ROOT.RooRealVar(
        "mean", "BifurGauss mean [GeV]",
        m_wr,
        m_wr * 0.5,
        m_wr * 1.5,
    )
    # Left (low-mass) sigma — expect larger due to cascade kinematics
    sigma_lo = ROOT.RooRealVar(
        "sigma_lo", "left sigma [GeV]",
        m_wr * 0.15,   # 15% of M_WR as initial guess
        1.0,
        m_wr * 0.5,
    )
    # Right (high-mass) sigma — expect smaller
    sigma_hi = ROOT.RooRealVar(
        "sigma_hi", "right sigma [GeV]",
        m_wr * 0.10,   # 10% of M_WR as initial guess
        1.0,
        m_wr * 0.5,
    )
    n_sig = ROOT.RooRealVar("n_sig", "signal yield", n_data, 0.0, 1e7)

    shape = ROOT.RooBifurGauss(
        "bifur", "Bifurcated Gaussian signal shape",
        observable, mean, sigma_lo, sigma_hi,
    )
    ROOT.SetOwnership(shape, False)

    pdf = ROOT.RooExtendPdf("ext_bifur", "extended Bifurcated Gaussian", shape, n_sig)

    return pdf, {"mean": mean, "sigma_lo": sigma_lo, "sigma_hi": sigma_hi, "n_sig": n_sig}


def build_dscb_model(
    observable: ROOT.RooRealVar,
    n_data: float,
    m_wr: float,
) -> tuple[ROOT.RooExtendPdf, dict[str, ROOT.RooRealVar]]:
    """Build a Double-sided Crystal Ball (DSCB) signal model.

    RooCrystalBall(x, x0, sigmaLR, alphaL, nL, alphaR, nR) — Gaussian core
    with independent power-law tails on both sides:

        f(m) ∝ exp(-(m-mean)^2 / 2σ²)                     if -αL < (m-mean)/σ < αR
             ∝ (nL/αL)^nL * exp(-αL²/2) * (...)^{-nL}      if (m-mean)/σ ≤ -αL
             ∝ (nR/αR)^nR * exp(-αR²/2) * (...)^{-nR}      if (m-mean)/σ ≥ αR

    This is the standard CMS/ATLAS signal model for resonance searches,
    as it captures both the low-mass tail (cascade decay kinematics) and
    the high-mass tail (radiation, resolution effects).
    """
    mean = ROOT.RooRealVar(
        "mean", "DSCB mean [GeV]",
        m_wr,
        m_wr * 0.5,
        m_wr * 1.5,
    )
    sigma = ROOT.RooRealVar(
        "sigma", "DSCB sigma [GeV]",
        m_wr * 0.05,
        1.0,
        m_wr * 0.5,
    )
    # Left tail: transition point and power-law exponent
    alpha_lo = ROOT.RooRealVar("alpha_lo", "DSCB alpha left", 1.5, 0.1, 10.0)
    n_lo = ROOT.RooRealVar("n_lo", "DSCB n left", 5.0, 0.1, 50.0)
    # Right tail: transition point and power-law exponent
    alpha_hi = ROOT.RooRealVar("alpha_hi", "DSCB alpha right", 1.5, 0.1, 10.0)
    n_hi = ROOT.RooRealVar("n_hi", "DSCB n right", 5.0, 0.1, 50.0)
    n_sig = ROOT.RooRealVar("n_sig", "signal yield", n_data, 0.0, 1e7)

    shape = ROOT.RooCrystalBall(
        "dscb", "Double-sided Crystal Ball signal shape",
        observable, mean, sigma, alpha_lo, n_lo, alpha_hi, n_hi,
    )
    ROOT.SetOwnership(shape, False)

    pdf = ROOT.RooExtendPdf("ext_dscb", "extended DSCB", shape, n_sig)

    return pdf, {
        "mean": mean, "sigma": sigma,
        "alpha_lo": alpha_lo, "n_lo": n_lo,
        "alpha_hi": alpha_hi, "n_hi": n_hi,
        "n_sig": n_sig,
    }


# ---------------------------------------------------------------------------
# iminuit-based fits (for comparison with RooFit)
# ---------------------------------------------------------------------------

def _extract_hist_arrays(
    hist: ROOT.TH1D,
    mass_lo: float,
    mass_hi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract bin centers, contents, and errors from a TH1 within the mass range."""
    centers, contents, errors = [], [], []
    for i in range(1, hist.GetNbinsX() + 1):
        c = hist.GetBinCenter(i)
        if mass_lo <= c <= mass_hi:
            centers.append(c)
            contents.append(hist.GetBinContent(i))
            errors.append(hist.GetBinError(i))
    return np.array(centers), np.array(contents), np.array(errors)


def run_iminuit_fit(
    model_name: str,
    bin_centers: np.ndarray,
    counts: np.ndarray,
    errors: np.ndarray,
    bin_width: float,
    m_wr: float,
    n_data: float,
    mass_range: list[float],
) -> dict:
    """Run a chi2 fit using iminuit (Minuit2 migrad) as a cross-check.

    Uses the same Minuit2 minimizer as RooFit's chi2FitTo, with the same
    model definitions, initial parameters, and parameter ranges.  This
    isolates any remaining differences to the model implementation
    (scipy PDF vs RooFit PDF) rather than the minimization algorithm.
    """
    from iminuit import Minuit
    from scipy.integrate import quad
    from scipy.special import voigt_profile
    from scipy.stats import crystalball as cb_dist
    from scipy.stats import norm

    # --- Define model function, initial params, and bounds per model ---
    if model_name == "gaussian":
        param_names = ["mean", "sigma", "n_sig"]
        p0 = [m_wr, m_wr * 0.05, n_data]
        bounds = [(m_wr * 0.5, m_wr * 1.5), (1.0, m_wr * 0.5), (0.0, 1e7)]

        def model(x, mean, sigma, n_sig):
            return n_sig * norm.pdf(x, loc=mean, scale=sigma) * bin_width

    elif model_name == "crystal-ball":
        param_names = ["mean", "sigma", "alpha", "cb_n", "n_sig"]
        p0 = [m_wr, m_wr * 0.05, 1.5, 2.0, n_data]
        bounds = [
            (m_wr * 0.5, m_wr * 1.5), (1.0, m_wr * 0.5),
            (0.1, 10.0), (1.01, 50.0), (0.0, 1e7),
        ]

        def model(x, mean, sigma, alpha, cb_n, n_sig):
            return n_sig * cb_dist.pdf(x, alpha, cb_n, loc=mean, scale=sigma) * bin_width

    elif model_name == "voigtian":
        param_names = ["mean", "width", "sigma", "n_sig"]
        p0 = [m_wr, m_wr * 0.10, m_wr * 0.05, n_data]
        bounds = [
            (m_wr * 0.5, m_wr * 1.5), (1.0, m_wr * 0.5),
            (1.0, m_wr * 0.3), (0.0, 1e7),
        ]

        def model(x, mean, width, sigma, n_sig):
            gamma = width / 2.0  # Lorentzian HWHM
            return n_sig * voigt_profile(x - mean, sigma, gamma) * bin_width

    elif model_name == "breit-wigner":
        param_names = ["mean", "width", "n_sig"]
        p0 = [m_wr, m_wr * 0.13, n_data]
        bounds = [(m_wr * 0.5, m_wr * 1.5), (1.0, m_wr * 0.5), (0.0, 1e7)]

        def model(x, mean, width, n_sig):
            bw = 1.0 / ((x**2 - mean**2)**2 + mean**2 * width**2)
            norm_val, _ = quad(
                lambda m: 1.0 / ((m**2 - mean**2)**2 + mean**2 * width**2),
                mass_range[0], mass_range[1],
            )
            return n_sig * (bw / norm_val) * bin_width

    elif model_name == "bifur-gauss":
        param_names = ["mean", "sigma_lo", "sigma_hi", "n_sig"]
        p0 = [m_wr, m_wr * 0.15, m_wr * 0.10, n_data]
        bounds = [
            (m_wr * 0.5, m_wr * 1.5), (1.0, m_wr * 0.5),
            (1.0, m_wr * 0.5), (0.0, 1e7),
        ]

        def model(x, mean, sigma_lo, sigma_hi, n_sig):
            pdf_vals = np.where(
                x < mean,
                np.exp(-0.5 * ((x - mean) / sigma_lo) ** 2),
                np.exp(-0.5 * ((x - mean) / sigma_hi) ** 2),
            )
            norm_val = np.sqrt(2 * np.pi) * (sigma_lo + sigma_hi) / 2
            return n_sig * (pdf_vals / norm_val) * bin_width

    elif model_name == "dscb":
        param_names = ["mean", "sigma", "alpha_lo", "n_lo", "alpha_hi", "n_hi", "n_sig"]
        p0 = [m_wr, m_wr * 0.05, 1.5, 5.0, 1.5, 5.0, n_data]
        bounds = [
            (m_wr * 0.5, m_wr * 1.5), (1.0, m_wr * 0.5),
            (0.1, 10.0), (0.1, 50.0),
            (0.1, 10.0), (0.1, 50.0),
            (0.0, 1e7),
        ]

        def model(x, mean, sigma, alpha_lo, n_lo, alpha_hi, n_hi, n_sig):
            from scipy.integrate import quad
            t = (x - mean) / sigma

            def _dscb_core(t_val):
                if t_val < -alpha_lo:
                    A = (n_lo / alpha_lo) ** n_lo * np.exp(-0.5 * alpha_lo ** 2)
                    B = n_lo / alpha_lo - alpha_lo
                    return A * (B - t_val) ** (-n_lo)
                elif t_val > alpha_hi:
                    A = (n_hi / alpha_hi) ** n_hi * np.exp(-0.5 * alpha_hi ** 2)
                    B = n_hi / alpha_hi - alpha_hi
                    return A * (B + t_val) ** (-n_hi)
                else:
                    return np.exp(-0.5 * t_val ** 2)

            pdf_vals = np.array([_dscb_core(ti) for ti in t])
            norm_val, _ = quad(
                lambda m: _dscb_core((m - mean) / sigma),
                mass_range[0], mass_range[1],
            )
            norm_val *= sigma  # Jacobian: dt = dm/sigma
            return n_sig * (pdf_vals / norm_val) * bin_width

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # --- Mask empty bins (same as RooFit post-hoc chi2) ---
    mask = (counts > 0) & (errors > 0)
    x_fit = bin_centers[mask]
    y_fit = counts[mask]
    e_fit = errors[mask]

    # --- Chi2 cost function for iminuit ---
    def chi2_cost(*args):
        predicted = model(x_fit, *args)
        return float(np.sum(((y_fit - predicted) / e_fit) ** 2))

    # --- Set up Minuit with same initial values and bounds ---
    m = Minuit(chi2_cost, *p0, name=param_names)
    for i, (lo, hi) in enumerate(bounds):
        m.limits[i] = (lo, hi)

    # --- Run migrad (same algorithm as RooFit's Minuit2) ---
    m.migrad()
    m.hesse()
    converged = m.valid

    if not converged:
        logger.warning("iminuit fit (%s) did not converge.", model_name)

    # --- Extract results ---
    popt = np.array(m.values)
    perr = np.array(m.errors)
    chi2 = float(m.fval) if converged else float("inf")
    ndf = int(mask.sum() - len(param_names))
    chi2_ndf = chi2 / ndf if ndf > 0 else float("inf")

    # --- Evaluate smooth curve for plotting ---
    x_curve = np.linspace(mass_range[0], mass_range[1], 500)
    y_curve = model(x_curve, *popt) if converged else np.zeros_like(x_curve)

    # --- Package results ---
    parameters = {}
    for name, val, err in zip(param_names, popt, perr):
        parameters[name] = {"value": float(val), "error": float(err)}

    # Correlation matrix from Hesse
    if converged and np.all(perr > 0):
        cov = np.array(m.covariance)
        corr = (cov / np.outer(perr, perr)).tolist()
    else:
        corr = np.eye(len(param_names)).tolist()

    return {
        "converged": converged,
        "parameters": parameters,
        "correlation_matrix": corr,
        "goodness_of_fit": {
            "chi2_per_ndf": chi2_ndf,
            "ndf": ndf,
            "chi2": chi2,
        },
        "curve": (x_curve, y_curve),
    }


def run_scipy_fit(
    model_name: str,
    bin_centers: np.ndarray,
    counts: np.ndarray,
    errors: np.ndarray,
    bin_width: float,
    m_wr: float,
    n_data: float,
    mass_range: list[float],
) -> dict:
    """Run scipy.optimize.curve_fit (chi2 least-squares) as a cross-check.

    Uses the same model definitions, initial parameters, and parameter ranges
    as the RooFit versions to enable a direct comparison.

    Key difference from RooFit MLE: curve_fit minimizes chi2 (least-squares
    with errors), while RooFit minimizes -log(L) (binned extended maximum
    likelihood).  Results may differ, especially in low-statistics tails.
    """
    from scipy.integrate import quad
    from scipy.optimize import curve_fit
    from scipy.special import voigt_profile
    from scipy.stats import crystalball as cb_dist
    from scipy.stats import norm

    # --- Define model function, initial params, and bounds per model ---
    if model_name == "gaussian":
        param_names = ["mean", "sigma", "n_sig"]
        p0 = [m_wr, m_wr * 0.05, n_data]
        bounds = ([m_wr * 0.5, 1.0, 0.0],
                  [m_wr * 1.5, m_wr * 0.5, 1e7])

        def model(x, mean, sigma, n_sig):
            return n_sig * norm.pdf(x, loc=mean, scale=sigma) * bin_width

    elif model_name == "crystal-ball":
        param_names = ["mean", "sigma", "alpha", "cb_n", "n_sig"]
        p0 = [m_wr, m_wr * 0.05, 1.5, 2.0, n_data]
        bounds = ([m_wr * 0.5, 1.0, 0.1, 1.01, 0.0],
                  [m_wr * 1.5, m_wr * 0.5, 10.0, 50.0, 1e7])

        def model(x, mean, sigma, alpha, cb_n, n_sig):
            return n_sig * cb_dist.pdf(x, alpha, cb_n, loc=mean, scale=sigma) * bin_width

    elif model_name == "voigtian":
        param_names = ["mean", "width", "sigma", "n_sig"]
        p0 = [m_wr, m_wr * 0.10, m_wr * 0.05, n_data]
        bounds = ([m_wr * 0.5, 1.0, 1.0, 0.0],
                  [m_wr * 1.5, m_wr * 0.5, m_wr * 0.3, 1e7])

        def model(x, mean, width, sigma, n_sig):
            gamma = width / 2.0  # Lorentzian HWHM
            return n_sig * voigt_profile(x - mean, sigma, gamma) * bin_width

    elif model_name == "breit-wigner":
        param_names = ["mean", "width", "n_sig"]
        p0 = [m_wr, m_wr * 0.13, n_data]
        bounds = ([m_wr * 0.5, 1.0, 0.0],
                  [m_wr * 1.5, m_wr * 0.5, 1e7])

        def model(x, mean, width, n_sig):
            bw = 1.0 / ((x**2 - mean**2)**2 + mean**2 * width**2)
            norm_val, _ = quad(
                lambda m: 1.0 / ((m**2 - mean**2)**2 + mean**2 * width**2),
                mass_range[0], mass_range[1],
            )
            return n_sig * (bw / norm_val) * bin_width

    elif model_name == "bifur-gauss":
        param_names = ["mean", "sigma_lo", "sigma_hi", "n_sig"]
        p0 = [m_wr, m_wr * 0.15, m_wr * 0.10, n_data]
        bounds = ([m_wr * 0.5, 1.0, 1.0, 0.0],
                  [m_wr * 1.5, m_wr * 0.5, m_wr * 0.5, 1e7])

        def model(x, mean, sigma_lo, sigma_hi, n_sig):
            pdf_vals = np.where(
                x < mean,
                np.exp(-0.5 * ((x - mean) / sigma_lo) ** 2),
                np.exp(-0.5 * ((x - mean) / sigma_hi) ** 2),
            )
            norm_val = np.sqrt(2 * np.pi) * (sigma_lo + sigma_hi) / 2
            return n_sig * (pdf_vals / norm_val) * bin_width

    elif model_name == "dscb":
        param_names = ["mean", "sigma", "alpha_lo", "n_lo", "alpha_hi", "n_hi", "n_sig"]
        p0 = [m_wr, m_wr * 0.05, 1.5, 5.0, 1.5, 5.0, n_data]
        bounds = ([m_wr * 0.5, 1.0, 0.1, 0.1, 0.1, 0.1, 0.0],
                  [m_wr * 1.5, m_wr * 0.5, 10.0, 50.0, 10.0, 50.0, 1e7])

        def model(x, mean, sigma, alpha_lo, n_lo, alpha_hi, n_hi, n_sig):
            t = (x - mean) / sigma

            # Vectorized DSCB
            A_lo = (n_lo / alpha_lo) ** n_lo * np.exp(-0.5 * alpha_lo ** 2)
            B_lo = n_lo / alpha_lo - alpha_lo
            A_hi = (n_hi / alpha_hi) ** n_hi * np.exp(-0.5 * alpha_hi ** 2)
            B_hi = n_hi / alpha_hi - alpha_hi

            pdf_vals = np.where(
                t < -alpha_lo,
                A_lo * (B_lo - t) ** (-n_lo),
                np.where(
                    t > alpha_hi,
                    A_hi * (B_hi + t) ** (-n_hi),
                    np.exp(-0.5 * t ** 2),
                ),
            )
            norm_val, _ = quad(
                lambda m: model(np.array([m]), mean, sigma, alpha_lo, n_lo, alpha_hi, n_hi, 1.0)[0] / bin_width,
                mass_range[0], mass_range[1],
            )
            return n_sig * (pdf_vals / (norm_val + 1e-300)) * bin_width

    else:
        raise ValueError(f"Unknown scipy model: {model_name}")

    # --- Fit (chi2 least-squares with observed errors) ---
    mask = errors > 0  # skip empty bins

    try:
        popt, pcov = curve_fit(
            model, bin_centers[mask], counts[mask],
            p0=p0, sigma=errors[mask], absolute_sigma=True,
            bounds=bounds, maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcov))
        converged = True
    except (RuntimeError, ValueError) as e:
        logger.warning("Scipy fit (%s) did not converge: %s", model_name, e)
        popt = np.array(p0)
        perr = np.full(len(p0), float("nan"))
        converged = False

    # --- Chi2 ---
    if converged:
        predicted = model(bin_centers[mask], *popt)
        chi2 = float(np.sum(((counts[mask] - predicted) / errors[mask]) ** 2))
    else:
        chi2 = float("inf")
    ndf = int(mask.sum() - len(param_names))
    chi2_ndf = chi2 / ndf if ndf > 0 else float("inf")

    # --- Evaluate smooth curve for plotting ---
    x_curve = np.linspace(mass_range[0], mass_range[1], 500)
    if converged:
        y_curve = model(x_curve, *popt)
    else:
        y_curve = np.zeros_like(x_curve)

    # --- Package results ---
    parameters = {}
    for name, val, err in zip(param_names, popt, perr):
        parameters[name] = {"value": float(val), "error": float(err)}

    if converged and np.all(np.isfinite(perr)) and np.all(perr > 0):
        corr = (pcov / np.outer(perr, perr)).tolist()
    else:
        corr = np.eye(len(param_names)).tolist()

    return {
        "converged": converged,
        "parameters": parameters,
        "correlation_matrix": corr,
        "goodness_of_fit": {
            "chi2_per_ndf": chi2_ndf,
            "ndf": ndf,
            "chi2": chi2,
        },
        "curve": (x_curve, y_curve),
    }



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_MODEL_LABELS = {
    "gaussian": "Gaussian",
    "crystal-ball": "Crystal Ball",
    "dscb": "Double-sided Crystal Ball",
    "voigtian": "Voigtian",
    "breit-wigner": "Breit-Wigner",
    "bifur-gauss": "Bifurcated Gaussian",
}


def compute_blinding_window(
    hist: ROOT.TH1D,
    n_sigma: float,
) -> tuple[float, float, float, float]:
    """Return (lo, hi, mean, rms) blinding window from histogram statistics.

    Uses the histogram mean and RMS directly (no parametric model),
    matching the logic in signal_window.py:
        lo = mean - n_sigma * RMS
        hi = mean + n_sigma * RMS
    """
    mean = hist.GetMean()
    rms = hist.GetRMS()
    return mean - n_sigma * rms, mean + n_sigma * rms, mean, rms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1: Fit a parametric model to signal MC m_lljj to characterize "
            "the signal peak and define a blinding window."
        )
    )
    add_common_args(parser)
    _all_models = ["gaussian", "crystal-ball", "dscb", "voigtian", "breit-wigner", "bifur-gauss"]
    _default_models = ["gaussian", "bifur-gauss", "crystal-ball", "dscb"]
    parser.add_argument(
        "--model",
        nargs="*",
        choices=_all_models,
        default=_default_models,
        metavar="MODEL",
        help=(
            "Signal shape model(s) to fit. "
            f"Choices: {', '.join(_all_models)}. "
            f"Default: {' '.join(_default_models)}"
        ),
    )
    parser.add_argument(
        "--fit-method",
        choices=["mle", "chi2"],
        default="mle",
        help=(
            "RooFit minimization method (default: mle). "
            "'mle' uses binned extended max-likelihood (fitTo) with a scipy "
            "curve_fit cross-check. "
            "'chi2' normalizes MC to N_eff, uses chi2FitTo with Poisson "
            "errors, and cross-checks with iminuit (same Minuit2 migrad)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    if not args.signal:
        sys.exit("ERROR: --signal is required. Example: --signal WR2000_N1100")

    ROOT.gROOT.SetBatch(True)
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

    # Parse W_R mass from signal tag for parameter initialization
    m_wr = parse_wr_mass(args.signal)
    logger.info("Signal: %s  (M_WR = %.0f GeV)", args.signal, m_wr)

    # --- Resolve input path ---
    info = load_lumi(args.era)
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)
    input_dir = input_dirs[0]
    logger.info("Input directory: %s", input_dir)

    # --- Output directories ---
    base_output_dir = Path(
        args.output_dir
        or str(repo_root() / "fitting_v2" / "outputs" / args.era / "signal_shape" / args.signal)
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    models = args.model
    channels = [args.channel] if args.channel else ["ee", "mumu"]

    use_chi2 = args.fit_method == "chi2"

    for model_name in models:
      model_output_dir = base_output_dir / model_name
      model_output_dir.mkdir(parents=True, exist_ok=True)

      for channel in channels:
          logger.info("=" * 60)
          logger.info("Channel: %s  Model: %s  [fit-method: %s]", channel, model_name, args.fit_method)

          # --- Build histogram key ---
          region = build_region_name(channel, args.topology)
          mass_var = "mass_twoobject" if args.topology == "boosted" else "mass_fourobject"
          hist_key = build_hist_key(region, mass_var)
          logger.info("Histogram key: %s", hist_key)

          # --- Load signal histogram ---
          sig_hist = load_signal(input_dir, hist_key, args.signal)

          # --- Rebin ---
          rebinned = sig_hist.Rebin(args.rebin, f"h_sig_{model_name}_{channel}_rebinned")
          bin_width_gev = rebinned.GetBinWidth(1)
          logger.info(
              "Rebinned: %d bins (factor %d, %.0f GeV/bin)",
              rebinned.GetNbinsX(), args.rebin, bin_width_gev,
          )

          if use_chi2:
              # --- Normalize MC to look like real data ---
              # Weighted MC inflates the apparent event count beyond the
              # true statistical power.  Rescale to N_eff = (sumw)^2 / sumw2
              # and set errors to sqrt(content), so the chi2 fit sees a
              # histogram with proper Poisson statistics — matching what
              # will happen on real data.
              sum_w = rebinned.Integral()
              sum_w2 = sum(
                  rebinned.GetBinError(i) ** 2
                  for i in range(1, rebinned.GetNbinsX() + 1)
              )
              if sum_w2 > 0:
                  n_eff = sum_w ** 2 / sum_w2
                  scale = n_eff / sum_w if sum_w > 0 else 1.0
              else:
                  n_eff = sum_w
                  scale = 1.0

              logger.info(
                  "MC normalization: sum_w=%.1f, N_eff=%.1f, scale=%.4f",
                  sum_w, n_eff, scale,
              )
              rebinned.Scale(scale)

              # Reset errors to sqrt(content) (Poisson).  Empty bins get
              # a very large error so chi2FitTo skips them (matching
              # iminuit's mask = counts > 0 behavior).
              for i in range(0, rebinned.GetNbinsX() + 2):
                  content = max(rebinned.GetBinContent(i), 0.0)
                  if content > 0:
                      rebinned.SetBinError(i, np.sqrt(content))
                  else:
                      rebinned.SetBinError(i, 1e10)

          # --- Blinding window (histogram mean ± n_sigma * RMS) ---
          blo, bhi, hist_mean, hist_rms = compute_blinding_window(rebinned, args.n_sigma)

          # --- Define observable: fit only within the signal window ---
          mass_lo, mass_hi = blo, bhi
          mass = ROOT.RooRealVar("mass", r"m_{lljj}", mass_lo, mass_hi, "GeV")
          logger.info(
              "Fit range (%.0f*RMS window): [%.1f, %.1f] GeV  (mean=%.1f, RMS=%.1f)",
              args.n_sigma, mass_lo, mass_hi, hist_mean, hist_rms,
          )

          # --- Create RooDataHist ---
          data_hist = create_roodatahist(rebinned, mass, name=f"data_sig_{model_name}_{channel}")
          n_data = data_hist.sumEntries()
          logger.info("Signal events in fit range: %.1f", n_data)

          if n_data < 1:
              logger.warning("Too few signal events in %s — skipping.", channel)
              continue

          # --- Build model ---
          model_builders = {
              "gaussian": build_gaussian_model,
              "crystal-ball": build_crystal_ball_model,
              "dscb": build_dscb_model,
              "voigtian": build_voigtian_model,
              "breit-wigner": build_breit_wigner_model,
              "bifur-gauss": build_bifur_gauss_model,
          }
          pdf, params = model_builders[model_name](mass, n_data, m_wr)

          # --- Fit ---
          fit_result = run_fit(pdf, data_hist, params, use_chi2=use_chi2)

          # --- Extract results ---
          fit_results = extract_fit_results(fit_result, params)

          # --- Chi2/ndf ---
          if use_chi2:
              # Compute chi2 manually, skipping empty bins to match iminuit.
              n_free = len(params)
              chi2 = 0.0
              n_bins = 0
              for i in range(1, rebinned.GetNbinsX() + 1):
                  center = rebinned.GetBinCenter(i)
                  if center < mass_lo or center > mass_hi:
                      continue
                  obs = rebinned.GetBinContent(i)
                  if obs <= 0:
                      continue
                  err = rebinned.GetBinError(i)
                  mass.setVal(center)
                  exp = pdf.getVal(ROOT.RooArgSet(mass)) * n_data * bin_width_gev
                  chi2 += ((obs - exp) / err) ** 2
                  n_bins += 1
              ndf = n_bins - n_free
              chi2_ndf = chi2 / ndf if ndf > 0 else float("inf")
          else:
              # Use RooPlot's chiSquare (includes all bins in range).
              tmp_frame = mass.frame()
              data_hist.plotOn(tmp_frame, ROOT.RooFit.Name("tmp_data"))
              pdf.plotOn(tmp_frame, ROOT.RooFit.Name("tmp_fit"))
              roo_hist = tmp_frame.getHist("tmp_data")
              n_bins = roo_hist.GetN()
              ndf = n_bins - len(params)
              chi2_ndf = tmp_frame.chiSquare("tmp_fit", "tmp_data", len(params))
              chi2 = chi2_ndf * ndf

          fit_results["goodness_of_fit"] = {
              "chi2_per_ndf": chi2_ndf,
              "ndf": ndf,
              "n_bins": n_bins,
              "chi2": chi2,
          }

          # --- Assemble full results dict ---
          results = {
              "metadata": {
                  "era": args.era,
                  "channel": channel,
                  "topology": args.topology,
                  "region": region,
                  "signal": args.signal,
                  "variable": mass_var,
                  "model": model_name,
                  "fit_method": args.fit_method,
                  "mass_range": [mass_lo, mass_hi],
                  "rebin": args.rebin,
                  "bin_width_gev": bin_width_gev,
                  "n_events_signal": float(n_data),
                  "timestamp": datetime.now().isoformat(),
              },
              "blinding_window": {
                  "lo": blo,
                  "hi": bhi,
                  "n_sigma": args.n_sigma,
                  "hist_mean": hist_mean,
                  "hist_rms": hist_rms,
              },
              "roofit": {
                  **fit_results,
              },
          }

          json_path = model_output_dir / f"sig_fit_{model_name}_{channel}.json"
          save_results_json(results, json_path)

          # --- Plot ---
          plot_path = model_output_dir / f"sig_fit_{model_name}_{channel}.pdf"
          plot_fit(
              mass, data_hist, pdf, fit_result, params, plot_path,
              lumi=info["lumi"],
              com=info.get("com", 13.6),
              channel=channel,
              topology=args.topology,
              era=args.era,
              model_label=_MODEL_LABELS[model_name],
              bin_width_gev=bin_width_gev,
              data_label=f"MC signal ({args.signal.replace('_', ', ')})",
              blinding_window=(blo, bhi),
              y_range=(0, rebinned.GetMaximum() * 1.4),
              chi2_ndf_override=chi2_ndf,
              x_range=args.mass_range,
              extra_data=rebinned,
              box_left=args.box_left,
          )

          # --- Summary ---
          logger.info("=" * 60)
          logger.info("Signal shape fit: %s %s  [model: %s]", channel, args.signal, model_name)

          # RooFit results
          logger.info("  RooFit:")
          logger.info("    Status:    %d (0 = converged)", fit_results["fit_status"]["status"])
          logger.info("    covQual:   %d (3 = full)", fit_results["fit_status"]["cov_quality"])
          logger.info("    chi2/ndf:  %.2f (%d dof)", chi2_ndf, ndf)
          for pname in sorted(params.keys()):
              p = params[pname]
              if pname.startswith("n_"):
                  logger.info("    %-10s %.0f +/- %.0f", pname + ":", p.getVal(), p.getError())
              elif pname in ("mean", "sigma", "sigma_lo", "sigma_hi", "width"):
                  logger.info("    %-10s %.1f +/- %.1f GeV", pname + ":", p.getVal(), p.getError())
              else:
                  logger.info("    %-10s %.3f +/- %.3f", pname + ":", p.getVal(), p.getError())

          # Blinding window (shared, histogram-based)
          logger.info("  Window (hist mean +/- %.0f*RMS):", args.n_sigma)
          logger.info("    Mean:      %.1f GeV", hist_mean)
          logger.info("    RMS:       %.1f GeV", hist_rms)
          logger.info("    Window:    [%.1f, %.1f] GeV", blo, bhi)

          logger.info("  Output:    %s", base_output_dir)
          logger.info("=" * 60)


if __name__ == "__main__":
    main()
