#!/usr/bin/env python3
"""Shared background-fit toolkit for Stage 4 (in_window_fit).

Candidate functions of the mass m, recentered at the window centre m_c so the
slope term runs in (m - m_c)/1000 (power laws use m/m_c):

  expo    f = e^{a+b(m-m_c)/1000}               (2 par)
  expo2   f = e^{a+b(m-m_c)/1000+c[(m-m_c)/1000]^2}  (3 par)
  expo3   f = e^{a+...+d[(m-m_c)/1000]^3}        (4 par)
  powlaw  f = e^a (m/m_c)^b                      (2 par, pure power law)
  powexp  f = (m/m_c)^b e^{a+c(m-m_c)/1000}      (3 par, power law x exp)
  dexp    f = e^{a1+b1(m-m_c)/1000} + e^{a2+b2(m-m_c)/1000}  (4 par, double exp)

Recentering at m_c makes a = log(f) at the centre and de-correlates a from b, so
the covariance comes back accurate instead of forced positive-definite; the
fitted curve is unchanged. All functions are fit with ROOT: a TF1 chi-square
fit (Minuit2 Migrad) to the histogram bins in the requested range. Empty bins
(zero error) are skipped by the chi2, matching the populated-bins-only
convention. Migrad is seeded from a weighted log-space linear least-squares
solution so it starts near the minimum.

Every fit returns a FitResult carrying four quality checks (the older EDM /
call-limit / Hesse checks are folded into valid_minimum; see README):
  1. valid_minimum      FitResult::IsValid() -- a genuinely converged minimum.
                        IsValid() = State().IsValid() && !IsAboveMaxEdm() &&
                        !HasReachedCallLimit(), so it already requires EDM below
                        Minuit's tolerance, the call limit not reached, and a
                        Hesse-produced valid covariance.
  2. cov_ok             CovMatrixStatus() == 3 -- the covariance is accurate,
                        not merely forced positive-definite (status 2).
  3. no_param_at_limit  no parameter railed against a set bound (the slope <= 0
                        constraint, or dexp's normalization fence). A railed
                        slope flags a sparse window where the fit wanted to rise.
  4. monotonic          the fitted background falls across the whole window, with
                        no local rise. Trivial for expo/powlaw (single slope <=0);
                        can fail for the curved models (expo2/expo3/powexp/dexp),
                        where higher-order terms turn the slope positive away
                        from the pivot even when the leading slope is <= 0.
`passed` is the AND of all four. Raw status/cov_status/edm/ncalls are kept on the
FitResult as diagnostics -- they tell you *how* a fit failed.

Histogram I/O stays on uproot (pure-Python ROOT file reading).
"""
from __future__ import annotations

import collections
import csv
import ctypes
import dataclasses
import json
import logging
import statistics
from pathlib import Path

import numpy as np
import uproot

try:
    import ROOT
except ImportError:
    raise SystemExit("ERROR: PyROOT unavailable. Source LCG_106 first.")
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning
ROOT.TH1.AddDirectory(ROOT.kFALSE)
ROOT.Math.MinimizerOptions.SetDefaultMinimizer("Minuit2", "Migrad")

logger = logging.getLogger(__name__)

T0 = 1000.0          # GeV; fit variable t = m / T0
COEF_SYMS = "abcd"
# Generous Migrad call budget. Exhausting it is not a separate check: it sets
# HasReachedCallLimit(), which makes FitResult::IsValid() (valid_minimum) false.
MAX_CALLS = 50000
ROOT.Math.MinimizerOptions.SetDefaultMaxFunctionCalls(MAX_CALLS)

# name -> (color, f(m) formula for display, n_parameters). Order = plot order.
# Fits are recentered at the window centre m_c and the slope runs in
# (m - m_c)/1000; the power-law factor is (m/m_c); a is then log(f) at the centre
# m = m_c. The display labels below spell the slope term out in full (no "u").
FUNCS = {
    "expo":   ("#000000", r"$e^{\,a+b(m-m_{c})/1000}$",                          2),
    "expo2":  ("#5790fc",
               r"$e^{\,a+b(m-m_{c})/1000+c[(m-m_{c})/1000]^{2}}$",               3),
    "expo3":  ("#e42536",
               r"$e^{\,a+b(m-m_{c})/1000+c[(m-m_{c})/1000]^{2}+d[(m-m_{c})/1000]^{3}}$",
               4),
    "powlaw": ("#964a8b", r"$e^{a}\,(m/m_{c})^{\,b}$",                           2),
    "powexp": ("#f89c20", r"$(m/m_{c})^{\,b}\,e^{a+c(m-m_{c})/1000}$",           3),
    "dexp":   ("#2ca02c",
               r"$e^{a_1+b_1(m-m_{c})/1000}+e^{a_2+b_2(m-m_{c})/1000}$",         4),
}

# TF1 formulas, recentered at the window centre m0 = m_c (a float substituted
# in at fit time via .format). The slope term is (x - m0)/1000 and the
# power-law factor is x/m0. With this pivot a = log(f) at m = m_c (the yield at
# the centre) and a/b are de-correlated, so the covariance is well-conditioned
# (no forced positive-definite step). Coefficients match the display formulas.
TF1_FORMULA = {
    "expo":   "exp([0]+[1]*((x-{m0})/1.e3))",
    "expo2":  "exp([0]+[1]*((x-{m0})/1.e3)+[2]*pow((x-{m0})/1.e3,2))",
    "expo3":  ("exp([0]+[1]*((x-{m0})/1.e3)+[2]*pow((x-{m0})/1.e3,2)"
               "+[3]*pow((x-{m0})/1.e3,3))"),
    "powlaw": "exp([0])*pow(x/{m0},[1])",
    "powexp": "pow(x/{m0},[1])*exp([0]+[2]*((x-{m0})/1.e3))",
    "dexp":   "exp([0]+[1]*((x-{m0})/1.e3))+exp([2]+[3]*((x-{m0})/1.e3))",
}

# Parameter limits. The slope parameters are constrained <= 0: the background is
# a falling distribution, so its (leading) slope cannot be positive -- otherwise
# an unconstrained fit in a sparse high-mass window returns a rising slope (e.g.
# ee resolved 5.6 TeV, expo b = +0.97). Normalizations are left free. dexp also
# keeps a +-100 divergence fence on its two normalizations (interchangeable
# terms / collapse to -inf). `None` means that parameter is unconstrained. A fit
# that rails against a bound is flagged by the at-limit check (the constraint
# binding is itself a warning that the window is too sparse there).
#
# Per-function parameter order (recentered at m_c, slope in (m-m_c)/1000):
#   expo [a,b]  expo2 [a,b,c]  expo3 [a,b,c,d]  powlaw [a,b]
#   powexp [a,b,c]  dexp [a1,b1,a2,b2]
# Slopes constrained <= 0: expo/expo2/expo3/powlaw b; powexp b & c; dexp b1 & b2.
PAR_LIMITS = {
    "expo":   [None, (-100.0, 0.0)],
    "expo2":  [None, (-100.0, 0.0), None],
    "expo3":  [None, (-100.0, 0.0), None, None],
    "powlaw": [None, (-100.0, 0.0)],
    "powexp": [None, (-100.0, 0.0), (-100.0, 0.0)],
    "dexp":   [(-100.0, 100.0), (-100.0, 0.0), (-100.0, 100.0), (-100.0, 0.0)],
}

# Display constants.
MASS_VAR = {"resolved": "mass_fourobject", "boosted": "mass_twoobject"}
MASS_LABEL = {
    "resolved": r"$m_{\ell\ell jj}$ [GeV]",
    "boosted": r"$m_{\ell J}$ [GeV]",
}
CH_LAB = {"ee": "ee", "mumu": r"$\mu\mu$", "emu": r"$e\mu$"}
TOPO_LAB = {"resolved": "Resolved", "boosted": "Boosted"}

BKG_SAMPLES = ["DYJets", "tt_tW", "Nonprompt", "Other"]
BASELINE_FIT_RANGE = "[0.7,1.3]"   # Stage-1 width-table seed window


def func_label(name):
    return f"{name}:  {FUNCS[name][1]}"


def coef_text(params, perr=None):
    """Per-parameter lines 'a = v ± e' (± omitted when no/infinite error)."""
    lines = []
    for i, v in enumerate(params):
        if i >= 4:
            break
        if perr is not None and i < len(perr) and np.isfinite(perr[i]):
            lines.append(rf"${COEF_SYMS[i]}={v:.3g} \pm {perr[i]:.2g}$")
        else:
            lines.append(rf"${COEF_SYMS[i]}={v:.3g}$")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Seeding — weighted log-space linear LS (internal; Migrad does the real fit)
# ---------------------------------------------------------------------------

def _design(name, m, m0):
    s = (m - m0) / T0                 # centred slope term (m - m_c)/1000, m0 = m_c
    o = np.ones_like(s)
    if name == "expo":   return np.column_stack([o, s])
    if name == "expo2":  return np.column_stack([o, s, s ** 2])
    if name == "expo3":  return np.column_stack([o, s, s ** 2, s ** 3])
    if name == "powlaw": return np.column_stack([o, np.log(m / m0)])
    if name == "powexp": return np.column_stack([o, np.log(m / m0), s])
    raise ValueError(name)


def _loglin_seed(name, m, n, var, m0):
    Phi = _design(name, m, m0)
    if Phi.shape[0] <= Phi.shape[1]:
        return None
    sw = n / np.sqrt(var)
    try:
        coef, *_ = np.linalg.lstsq(Phi * sw[:, None], np.log(n) * sw, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return list(coef) if np.all(np.isfinite(coef)) else None


def _seed(name, m, n, var, m0):
    if name == "dexp":
        s = _loglin_seed("expo", m, n, var, m0)
        if s is None:
            return None
        a, b = s
        return [a, 0.7 * b, a - 1.5, 1.5 * b]   # one shallow, one steep
    return _loglin_seed(name, m, n, var, m0)


# ---------------------------------------------------------------------------
# ROOT fit
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FitResult:
    params: np.ndarray
    perr: np.ndarray
    cov: np.ndarray
    chi2: float
    ndf: int
    edm: float
    ncalls: int
    status: int          # Minuit2: 0 ok, 1 cov made posdef, 2 Hesse failed,
    cov_status: int      #          3 EDM above max, 4 call limit, 5 other
    checks: dict
    passed: bool


_UID = 0


def fit_model(name, edges, values, variances, lo, hi, m0):
    """Chi-square TF1 fit (Minuit2) to the histogram bins in [lo, hi].

    The model is recentered at the pivot m0 = m_c (the window centre): the slope
    term is (m - m0)/1000 and the power-law factor is m/m0. This makes
    a = log(f) at m = m_c and de-correlates a from b, so Minuit's covariance is
    accurate rather than forced positive-definite. The fitted curve (chi2,
    predictions) is identical to the m=0 pivot; only the parameter basis differs.

    Returns a FitResult, or None when the range has too few populated bins
    (need > npar) or no seed can be built.
    """
    global _UID
    _UID += 1
    npar = FUNCS[name][2]
    centers = 0.5 * (edges[:-1] + edges[1:])
    pop = ((centers >= lo) & (centers <= hi) & (values > 0) & (variances > 0))
    if pop.sum() <= npar:
        return None
    seed = _seed(name, centers[pop], values[pop], variances[pop], m0)
    if seed is None:
        return None

    h = ROOT.TH1D(f"_bkgfit_h{_UID}", "", len(values),
                  np.ascontiguousarray(edges, np.float64))
    for i, (v, e) in enumerate(zip(values, np.sqrt(variances)), start=1):
        h.SetBinContent(i, v)
        h.SetBinError(i, e if v > 0 else 0.0)   # zero error -> bin skipped

    f = ROOT.TF1(f"_bkgfit_f{_UID}", TF1_FORMULA[name].format(m0=f"{m0:.10g}"),
                 lo, hi)
    lims = PAR_LIMITS.get(name, [])
    for i, s in enumerate(seed):
        if i < len(lims) and lims[i] is not None:   # clamp the seed into the box
            s = min(max(s, lims[i][0]), lims[i][1])
        f.SetParameter(i, s)
    for i, lim in enumerate(lims):
        if lim is not None:
            f.SetParLimits(i, lim[0], lim[1])

    rptr = h.Fit(f, "RNSQ")    # Range, No-store, Save result, Quiet
    r = rptr.Get()
    if not r:
        return None

    params = np.array([r.Parameter(i) for i in range(npar)])
    perr = np.array([r.ParError(i) for i in range(npar)])
    cov = np.array([[r.CovMatrix(i, j) for j in range(npar)]
                    for i in range(npar)])
    status, cov_status = int(r.Status()), int(r.CovMatrixStatus())

    at_limit = False
    for i in range(npar):
        plo, phi = ctypes.c_double(), ctypes.c_double()
        f.GetParLimits(i, plo, phi)
        if phi.value > plo.value:               # limits were set
            rng = phi.value - plo.value
            if (params[i] - plo.value) < 1e-3 * rng \
                    or (phi.value - params[i]) < 1e-3 * rng:
                at_limit = True

    # monotonicity: the fitted background must FALL across the whole window (no
    # local rise). The slope constraint guarantees this for expo/powlaw (single
    # slope <= 0) but not for the curved models (expo2/expo3/powexp/dexp), where
    # the higher-order terms can turn the slope positive away from the pivot.
    mgrid = np.linspace(lo, hi, 200)
    fmono = predict(name, params, mgrid, m0)
    monotonic = bool(np.all(fmono[1:] <= fmono[:-1] * (1.0 + 1e-6)))

    checks = {
        "valid_minimum": bool(r.IsValid()),
        "cov_ok": cov_status == 3,
        "no_param_at_limit": not at_limit,
        "monotonic": monotonic,
    }
    return FitResult(
        params=params, perr=perr, cov=cov,
        chi2=float(r.Chi2()), ndf=int(r.Ndf()),
        edm=float(r.Edm()), ncalls=int(r.NCalls()),
        status=status, cov_status=cov_status,
        checks=checks, passed=all(checks.values()),
    )


# ---------------------------------------------------------------------------
# Evaluation helpers (NumPy; identical math to the TF1 formulas)
# ---------------------------------------------------------------------------

def predict(name, params, m, m0):
    if name == "dexp":
        s = (np.asarray(m, float) - m0) / T0
        return np.exp(params[0] + params[1] * s) + np.exp(params[2] + params[3] * s)
    return np.exp(_design(name, np.asarray(m, float), m0) @ np.asarray(params))


def band(name, params, cov, m, m0, eps=1e-5):
    """Central f and the 1-sigma log-uncertainty s (numerical Jacobian)."""
    f0 = predict(name, params, m, m0)
    lnf0 = np.log(f0)
    J = np.zeros((len(m), len(params)))
    for j in range(len(params)):
        dp = np.array(params, float)
        hstep = eps * (abs(params[j]) + eps)
        dp[j] += hstep
        J[:, j] = (np.log(predict(name, dp, m, m0)) - lnf0) / hstep
    var_lnf = np.einsum("ij,jk,ik->i", J, cov, J)
    return f0, np.sqrt(np.clip(var_lnf, 0.0, None))


def b_window(name, params, cov, m_win, m0):
    """Sum of f over the given bins and its uncertainty (Jacobian . cov)."""
    fvals = predict(name, params, m_win, m0)
    g = np.zeros(len(params))
    for j in range(len(params)):
        dp = np.array(params, float)
        hstep = 1e-5 * (abs(params[j]) + 1e-5)
        dp[j] += hstep
        g[j] = ((predict(name, dp, m_win, m0) - fvals) / hstep).sum()
    gcg = g @ cov @ g
    err = float(np.sqrt(gcg)) if np.isfinite(gcg) and gcg >= 0 else float("nan")
    return float(fvals.sum()), err


# ---------------------------------------------------------------------------
# Inputs: Stage-1 widths and the summed MC background (uproot I/O)
# ---------------------------------------------------------------------------

def load_grid_widths(width_csv: Path, channel: str, topology: str,
                     agg: str) -> dict[float, tuple[float, float]]:
    """{m_WR: (mu, sigma)} from the Stage-1 Gaussian width table.

    mu_gaus and sigma_gaus are aggregated (median/mean/...) over the M_N points
    at the baseline on-shell window; the window is centered on the fitted peak.
    """
    mu_by_mwr: dict[float, list[float]] = collections.defaultdict(list)
    sig_by_mwr: dict[float, list[float]] = collections.defaultdict(list)
    with open(width_csv) as fh:
        for r in csv.DictReader(fh):
            if (r["channel"] == channel and r["category"] == topology
                    and r["fit_range"] == BASELINE_FIT_RANGE):
                m = float(r["mWR"])
                mu_by_mwr[m].append(float(r["mu_gaus"]))
                sig_by_mwr[m].append(float(r["sigma_gaus"]))
    aggfn = {"median": statistics.median, "mean": statistics.mean,
             "max": max, "min": min}[agg]
    return {m: (aggfn(mu_by_mwr[m]), aggfn(sig_by_mwr[m]))
            for m in sorted(mu_by_mwr)}


def load_window_params(params_json: Path, channel: str, topology: str,
                       sigma_kind: str = "median",
                       ) -> tuple[list[float], list[float]]:
    """Linear-fit coefficients ([slope, intercept]) for mu and sigma from the
    Stage-2 wr/ window parameterization JSON, for one (channel, topology).

    `sigma_kind` selects "median" or "conservative" (see parameterize_window.py).
    """
    with open(params_json) as fh:
        p = json.load(fh)[channel][topology]
    return p["mu"], p[f"sigma_{sigma_kind}"]


def grid_widths_from_params(params_json: Path, channel: str, topology: str,
                            masses, sigma_kind: str = "median",
                            ) -> dict[float, tuple[float, float]]:
    """{m_WR: (mu, sigma)} from the Stage-2 LINEAR window parameterization.

    Unlike `load_grid_widths` (the per-m_WR MEASURED median over M_N), this
    evaluates the smooth linear fits mu(m_WR) = a + b*m_WR and
    sigma(m_WR) = a + b*m_WR at each requested m_WR -- the window the analysis
    is meant to use, well-defined at off-grid masses too.
    """
    (b_mu, a_mu), (b_sig, a_sig) = load_window_params(
        params_json, channel, topology, sigma_kind)
    return {float(m): (a_mu + b_mu * float(m), a_sig + b_sig * float(m))
            for m in masses}


def load_summed_background(input_dirs, region, variable, factor):
    """Sum the four background MC samples across sub-era dirs; rebin by factor."""
    hist_key = f"{region}/{variable}_{region}"
    edges = values = variances = None
    for d in input_dirs:
        for s in BKG_SAMPLES:
            fpath = d / f"WRAnalyzer_{s}.root"
            if not fpath.exists():
                logger.warning("Missing file: %s", fpath)
                continue
            h = uproot.open(fpath)[hist_key]
            e, v, var = h.axes[0].edges(), h.values(), h.variances()
            if edges is None:
                edges, values, variances = e, v.copy(), var.copy()
            else:
                values += v
                variances += var
    if edges is None:
        raise RuntimeError("No background histograms found")
    if factor > 1:
        n = (len(values) // factor) * factor
        values = values[:n].reshape(-1, factor).sum(axis=1)
        variances = variances[:n].reshape(-1, factor).sum(axis=1)
        edges = edges[::factor][: len(values) + 1]
    return edges, values, variances
