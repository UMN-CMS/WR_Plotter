#!/usr/bin/env python3
"""Stage 10 shared fit core -- S+B fit with prior-constrained signal shape and
a Poisson-likelihood option.

Extends (does NOT modify) the Stage 5-8 fixed-shape chi2 fit in
../shared/sb_fit.py along the two axes the earlier stages showed we need:

  1. statistic  --  `stat="chi2"` reproduces the Stage 5-8 convention (Neyman
     chi2, sigma = sqrt(n), bins with n > 0 only).  `stat="poisson"` minimizes
     the Baker-Cousins -2 ln lambda INCLUDING EMPTY BINS, which is the correct
     treatment of the sparse high-mass windows where the chi2 fit collapses
     (Stage 6: n_ok 919 @ 3600 -> 504 @ 4200 -> 105 @ 5000).

  2. signal shape  --  the Gaussian (mu, sigma) can be fixed (Stage 5-8), FLOATED
     WITH A GAUSSIAN PRIOR  ((mu - mu0)/s_mu)^2 + ((sigma - sigma0)/s_sigma)^2
     added to the objective, or floated free inside physical bounds.  The prior
     centre (mu0, sigma0) is the Stage-2 linear window parameterization; the
     physics motivation for floating sigma is Stage 1's width-variation result
     (sigma varies ~55-92 % vs x = m_N/m_WR, so the median under-covers the
     x-extremes).

The background model, seeding, parameter limits and quality checks are the
Stage-4 ones (bkg_fit_lib), evaluated through fast per-function closures rather
than TF1 formula strings because the objective is a hand-rolled FCN minimized
with Minuit2 (ROOT.Math).  For stat="chi2" with s_mu = s_sigma = 0 the fit is
statistically identical to sb_fit.fit_splusb (cross-checked by
1_nsp_diagnostics).

Source LCG_106 before importing (PyROOT)."""
from __future__ import annotations

import array
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent                      # 10_limit_refinement
sys.path.insert(0, str(_HERE.parent / "shared"))             # measure_fwhm loaders
sys.path.insert(0, str(_HERE.parent / "4_background_fits"))  # bkg_fit_lib

import ROOT                                                  # noqa: E402
ROOT.TH1.AddDirectory(False)
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError

from bkg_fit_lib import FUNCS, PAR_LIMITS, _seed, predict    # noqa: E402

# Objective is 2*NLL (or chi2) in both modes -> 1-sigma errors at Delta = 1.
_ERRORDEF = 1.0
_FLOOR = 1e-12          # model floor inside logs (empty-model protection)

# Physical bounds when mu/sigma float (units of sigma0, the prior centre).
# mu stays inside the +-3 sigma window with headroom (largest observed true-peak
# offset is ~0.8 sigma0); sigma covers the measured true-width range
# sigma_true/sigma0 in [0.67, 1.77] (Stage-1 U-shape) with margin, kept away
# from 0 (a collapsing Gaussian absorbs single-bin fluctuations) and from
# absurdly wide (degenerate with the background).
MU_BOUND_NSIG = 1.5          # |mu - mu0| <= 1.5 sigma0
SIGMA_BOUND_LO = 0.5         # sigma >= 0.5 sigma0
SIGMA_BOUND_HI = 2.5         # sigma <= 2.5 sigma0

CHECK_KEYS = ["valid_minimum", "cov_ok", "no_param_at_limit", "monotonic"]


# ---------------------------------------------------------------------------
# Fast background evaluators (same math as bkg_fit_lib.predict / TF1 formulas,
# specialized per function with the centred slope terms precomputed)
# ---------------------------------------------------------------------------

def _bkg_eval_factory(name, centers, m_c):
    """Return f(theta) -> per-bin background values on `centers`."""
    s = (centers - m_c) / 1000.0
    if name in ("powlaw", "powexp"):
        lr = np.log(centers / m_c)
    if name == "expo":
        return lambda t: np.exp(t[0] + t[1] * s)
    if name == "expo2":
        return lambda t: np.exp(t[0] + t[1] * s + t[2] * s * s)
    if name == "expo3":
        return lambda t: np.exp(t[0] + t[1] * s + t[2] * s * s + t[3] * s ** 3)
    if name == "powlaw":
        return lambda t: np.exp(t[0] + t[1] * lr)
    if name == "powexp":
        return lambda t: np.exp(t[0] + t[1] * lr + t[2] * s)
    if name == "dexp":
        return lambda t: (np.exp(t[0] + t[1] * s) + np.exp(t[2] + t[3] * s))
    raise ValueError(name)


def gauss_template(centers, mu, sigma, binwidth):
    """Per-bin events of a unit-yield Gaussian (same as sb_fit.gauss_per_bin)."""
    amp = binwidth / (sigma * math.sqrt(2.0 * math.pi))
    return amp * np.exp(-0.5 * ((centers - mu) / sigma) ** 2)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

def fit_splusb_v2(name, edges, data, lo, hi, m_c, sigma_win, mu0, sigma0,
                  binwidth, *, stat="poisson", s_mu=0.0, s_sigma=0.0,
                  bkg_constraints=None, nsig_bound="auto", strategy=1,
                  tolerance=0.1):
    """S+B fit of `data` inside [lo, hi]: Stage-4 background + Gaussian signal.

    stat     : "poisson" -- Baker-Cousins -2 ln lambda over ALL bins in range
               (empty bins included); "chi2" -- Neyman chi2 over bins with
               n > 0 (the Stage 5-8 convention).
    s_mu     : 0 -> mu fixed at mu0;  > 0 -> floats with Gaussian prior width
               s_mu (GeV);  None/inf -> floats free inside the mu bounds.
    s_sigma  : same for sigma about sigma0.
    nsig_bound : lower bound on N_sig.  "auto" (default) = -5 * max(observed
               count within mu0 +- 2 sigma0, 1) in poisson mode -- an unguarded
               Poisson likelihood on a near-empty window lets N_sig run away
               to huge negative values balanced by an inflated background
               (verified: RMS(N_sp) 6000+ at m_WR=4200 without the bound).
               None = unbounded (the chi2 legacy behaviour; also the default
               when stat="chi2").  A float is used as-is.
    bkg_constraints : None, or a list of npar entries anchoring the BACKGROUND
               parameters from outside the window (the trusted-range envelope):
               each entry is None (parameter free, default behaviour), or a
               tuple (centre, width) -- width > 0 adds a Gaussian penalty
               ((theta_i - centre)/width)^2, width == 0 FIXES the parameter at
               `centre`.  Constrained/fixed parameters are also SEEDED at
               `centre` (the in-window log-linear seed is unreliable exactly
               where anchoring is needed).

    Window/sideband bookkeeping matches sb_fit.fit_splusb: the window used for
    seeding is [m_c - k*sigma_win, m_c + k*sigma_win] implied by [lo, hi]; the
    background seed is the log-linear LS on the sideband bins (fallback: all).

    Returns a dict (nsig, nsig_err, mu, mu_err, sigma, sigma_err, params, perr,
    checks, passed, status, cov, edm, nll, nfloat, nbins_used, nbins_pop) or
    None when the window cannot support the fit.
    """
    centers = 0.5 * (edges[:-1] + edges[1:])
    npar = FUNCS[name][2]
    inrange = (centers >= lo) & (centers <= hi)
    pop = inrange & (data > 0)

    bc = list(bkg_constraints) if bkg_constraints is not None else [None] * npar
    if len(bc) != npar:
        raise ValueError(f"bkg_constraints needs {npar} entries for {name}")
    fixed_bkg = [i for i, c in enumerate(bc) if c is not None and c[1] == 0.0]
    prior_bkg = [i for i, c in enumerate(bc) if c is not None and c[1] > 0.0]

    float_mu = s_mu is None or (s_mu is not None and not (s_mu == 0.0))
    float_sigma = s_sigma is None or (s_sigma is not None and not (s_sigma == 0.0))
    prior_mu = (s_mu is not None) and float_mu and math.isfinite(s_mu)
    prior_sigma = (s_sigma is not None) and float_sigma and math.isfinite(s_sigma)
    # nfloat counts FREE parameters (bin-support checks); ntot counts all
    # REGISTERED ones -- Minuit passes fixed variables to the FCN too, so the
    # functor dimension and the X()/Errors() arrays use ntot.
    nfloat = npar - len(fixed_bkg) + 1 + int(float_mu) + int(float_sigma)
    ntot = npar + 1 + int(float_mu) + int(float_sigma)

    if stat == "chi2":
        sel = pop
        if sel.sum() <= nfloat:
            return None
    else:
        sel = inrange
        # need some information: enough bins and at least one observed event
        if sel.sum() <= nfloat or data[sel].sum() <= 0:
            return None

    # --- background seed: log-linear LS on the populated bins ----------------
    # (sb_fit tries the sidebands first, but in the in-window fit the fit range
    # IS the k*sigma window, so the sideband set is always empty and the seed
    # falls back to all populated bins -- do that directly here).  Anchored
    # parameters are seeded at their constraint centre instead; when EVERY
    # background parameter is anchored the LS seed is not needed at all.
    bseed = None
    if pop.sum() > 1:
        bseed = _seed(name, centers[pop], data[pop],
                      np.clip(data[pop], 1.0, None), m_c)
    if bseed is None:
        if all(c is not None for c in bc):
            bseed = [0.0] * npar
        else:
            return None
    bseed = [bc[i][0] if bc[i] is not None else bseed[i] for i in range(npar)]

    c_sel = centers[sel]
    n_sel = data[sel].astype(float)
    pos = n_sel > 0
    n_pos = n_sel[pos]
    nlogn = n_pos * np.log(n_pos)          # constant part of Baker-Cousins
    bkg_of = _bkg_eval_factory(name, c_sel, m_c)

    lims = PAR_LIMITS.get(name, [])

    # fixed-shape template cache (used when neither mu nor sigma floats)
    fixed_tpl = None
    if not (float_mu or float_sigma):
        fixed_tpl = gauss_template(c_sel, mu0, sigma0, binwidth)

    i_mu = npar + 1 if float_mu else -1
    i_sigma = (npar + 1 + int(float_mu)) if float_sigma else -1

    def fcn(par):
        t = [par[i] for i in range(npar)]
        nsig = par[npar]
        if fixed_tpl is not None:
            spb = nsig * fixed_tpl
        else:
            mu = par[i_mu] if float_mu else mu0
            sg = par[i_sigma] if float_sigma else sigma0
            spb = nsig * gauss_template(c_sel, mu, sg, binwidth)
        f = bkg_of(t) + spb
        f = np.clip(f, _FLOOR, 1e30)          # also caps exp() overflow -> inf
        if stat == "chi2":
            obj = float(np.sum((n_sel - f) ** 2 / n_sel))
        else:
            # 2 * sum[ f - n + n ln(n/f) ]; empty bins contribute 2f
            obj = 2.0 * float(np.sum(f) - n_pos.sum()
                              + np.sum(nlogn - n_pos * np.log(f[pos])))
        if prior_mu:
            obj += ((par[i_mu] - mu0) / s_mu) ** 2
        if prior_sigma:
            obj += ((par[i_sigma] - sigma0) / s_sigma) ** 2
        for i in prior_bkg:
            obj += ((par[i] - bc[i][0]) / bc[i][1]) ** 2
        return obj if math.isfinite(obj) else 1e30

    functor = ROOT.Math.Functor(fcn, ntot)
    mnz = ROOT.Math.Factory.CreateMinimizer("Minuit2", "Migrad")
    ROOT.SetOwnership(mnz, True)
    mnz.SetErrorDef(_ERRORDEF)
    mnz.SetStrategy(strategy)
    mnz.SetTolerance(tolerance)
    mnz.SetMaxFunctionCalls(50000)
    mnz.SetPrintLevel(-1)
    mnz.SetFunction(functor)

    names = []
    for i in range(npar):
        s0 = float(bseed[i])
        if i in fixed_bkg:
            mnz.SetFixedVariable(i, f"b{i}", s0)
        elif i < len(lims) and lims[i] is not None:
            s0 = min(max(s0, lims[i][0]), lims[i][1])
            mnz.SetLimitedVariable(i, f"b{i}", s0, 0.01,
                                   lims[i][0], lims[i][1])
        else:
            mnz.SetVariable(i, f"b{i}", s0, 0.01)
        names.append(f"b{i}")
    nsig_step = max(1.0, math.sqrt(max(data[sel].sum(), 1.0)) / 4.0)
    nsig_lo = nsig_bound
    if nsig_bound == "auto":
        if stat == "poisson":
            core = (centers >= mu0 - 2 * sigma0) & (centers <= mu0 + 2 * sigma0)
            nsig_lo = -5.0 * max(float(data[core & inrange].sum()), 1.0)
        else:
            nsig_lo = None                    # chi2 legacy: unbounded
    if nsig_lo is not None:
        mnz.SetLowerLimitedVariable(npar, "nsig", 0.0, nsig_step, float(nsig_lo))
    else:
        mnz.SetVariable(npar, "nsig", 0.0, nsig_step)
    names.append("nsig")
    if float_mu:
        mnz.SetLimitedVariable(i_mu, "mu", mu0, 0.1 * sigma0,
                               mu0 - MU_BOUND_NSIG * sigma0,
                               mu0 + MU_BOUND_NSIG * sigma0)
        names.append("mu")
    if float_sigma:
        mnz.SetLimitedVariable(i_sigma, "sigma", sigma0, 0.1 * sigma0,
                               SIGMA_BOUND_LO * sigma0, SIGMA_BOUND_HI * sigma0)
        names.append("sigma")

    ok = mnz.Minimize()
    if not (ok and int(mnz.Status()) == 0):
        # one retry from the first attempt's endpoint with the careful strategy
        mnz.SetStrategy(2)
        ok = mnz.Minimize()
    mnz.Hesse()

    par = np.array([mnz.X()[i] for i in range(ntot)])
    err = np.array([mnz.Errors()[i] for i in range(ntot)])
    status = int(mnz.Status())
    covstat = int(mnz.CovMatrixStatus())
    edm = float(mnz.Edm())

    # --- quality checks (Stage-4 conventions, background component only) -----
    bkg_params = par[:npar]
    at_limit = False
    for i in range(npar):
        if i in fixed_bkg:
            continue
        if i < len(lims) and lims[i] is not None:
            rng = lims[i][1] - lims[i][0]
            if (bkg_params[i] - lims[i][0]) < 1e-3 * rng \
                    or (lims[i][1] - bkg_params[i]) < 1e-3 * rng:
                at_limit = True
    # floated shape parameters railed against their physical boxes: reported as
    # per-fit flags (scan metrics) but NOT folded into no_param_at_limit -- a
    # railed sigma is a property of the prior config under study, not a reason
    # to silently drop the toy.
    sigma_railed = mu_railed = nsig_railed = False
    if float_sigma:
        s_v = par[i_sigma]
        rng = (SIGMA_BOUND_HI - SIGMA_BOUND_LO) * sigma0
        sigma_railed = ((s_v - SIGMA_BOUND_LO * sigma0) < 1e-3 * rng
                        or (SIGMA_BOUND_HI * sigma0 - s_v) < 1e-3 * rng)
    if float_mu:
        m_v = par[i_mu]
        rng = 2 * MU_BOUND_NSIG * sigma0
        mu_railed = ((m_v - (mu0 - MU_BOUND_NSIG * sigma0)) < 1e-3 * rng
                     or ((mu0 + MU_BOUND_NSIG * sigma0) - m_v) < 1e-3 * rng)
    if nsig_lo is not None:
        nsig_railed = (par[npar] - nsig_lo) < 1e-3 * max(abs(nsig_lo), 1.0)
    mgrid = np.linspace(lo, hi, 200)
    fmono = predict(name, bkg_params, mgrid, m_c)
    monotonic = bool(np.all(fmono[1:] <= fmono[:-1] * (1.0 + 1e-6)))
    checks = {"valid_minimum": bool(ok) and status == 0,
              "cov_ok": covstat == 3,
              "no_param_at_limit": not at_limit,
              "monotonic": monotonic}

    return dict(
        nsig=float(par[npar]), nsig_err=float(err[npar]),
        mu=float(par[i_mu]) if float_mu else float(mu0),
        mu_err=float(err[i_mu]) if float_mu else 0.0,
        sigma=float(par[i_sigma]) if float_sigma else float(sigma0),
        sigma_err=float(err[i_sigma]) if float_sigma else 0.0,
        params=par, perr=err, names=names,
        checks=checks, passed=all(checks.values()),
        mu_railed=mu_railed, sigma_railed=sigma_railed,
        nsig_railed=nsig_railed,
        status=status, cov=covstat, edm=edm, nll=float(mnz.MinValue()),
        nfloat=ntot, nbins_used=int(sel.sum()), nbins_pop=int(pop.sum()))


# ---------------------------------------------------------------------------
# Toy machinery (ROOT RNG + ROOT summary stats, per working preferences)
# ---------------------------------------------------------------------------

def poisson_toy(rng, mu):
    """Bin-wise ROOT Poisson draw of an expectation array."""
    return np.array([rng.Poisson(m) for m in mu], dtype=float)


def toy_seed(base, mWR, tagint):
    """Deterministic per-point seed (same scheme as Stage 6/8)."""
    return base * 1_000_003 + int(mWR) * 1009 + int(tagint)


def summarize(values):
    """mean, RMS, and the (2.5, 16, 50, 84, 97.5) percentiles via ROOT.TMath."""
    n = len(values)
    if n == 0:
        return None
    arr = array.array("d", sorted(float(v) for v in values))
    mean = float(ROOT.TMath.Mean(n, arr))
    rms = float(ROOT.TMath.RMS(n, arr))
    probs = array.array("d", [0.025, 0.16, 0.50, 0.84, 0.975])
    quants = array.array("d", [0.0] * 5)
    ROOT.TMath.Quantiles(n, 5, arr, quants, probs, True)
    return {"n": n, "mean": mean, "rms": rms,
            "q025": float(quants[0]), "q16": float(quants[1]),
            "q50": float(quants[2]), "q84": float(quants[3]),
            "q975": float(quants[4])}


def gaussianity(values):
    """Normality proxies for a toy sample: skewness, excess kurtosis,
    Jarque-Bera p-value (TMath.Prob), and quantile/RMS width ratios."""
    s = summarize(values)
    if s is None or s["n"] < 20 or not (s["rms"] > 0):
        return None
    x = np.asarray(values, float)
    z = (x - s["mean"]) / s["rms"]
    skew = float(np.mean(z ** 3))
    exkurt = float(np.mean(z ** 4)) - 3.0
    jb = s["n"] / 6.0 * (skew ** 2 + exkurt ** 2 / 4.0)
    p_jb = float(ROOT.TMath.Prob(jb, 2))
    r68 = (s["q84"] - s["q16"]) / 2.0 / s["rms"]          # 1.0 for a Gaussian
    r95 = (s["q975"] - s["q025"]) / 3.9199 / s["rms"]     # 1.0 for a Gaussian
    return {**s, "skew": skew, "exkurt": exkurt, "jb_pvalue": p_jb,
            "r68": r68, "r95": r95}


# ---------------------------------------------------------------------------
# Per-toy CLs upper limit (asymptotic convention of Stage 7)
# ---------------------------------------------------------------------------

def cls_upper_limit(n_hat, sigma, alpha=0.05):
    """One-sided CLs asymptotic UL for estimator n_hat with std sigma
    (identical formula to 7_limit_plots/expected_limit.py)."""
    if not (sigma > 0.0 and math.isfinite(sigma) and math.isfinite(n_hat)):
        return float("nan")
    p = 1.0 - alpha * ROOT.TMath.Freq(n_hat / sigma)
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return n_hat + sigma * ROOT.TMath.NormQuantile(p)
