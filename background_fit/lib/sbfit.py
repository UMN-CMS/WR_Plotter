"""The full-range S+B fit of background_fit: background function plus a Gaussian signal.

  model(x) = f_bkg(x; p)  +  N_sig x [Phi((x + w/2 - mu)/sigma) - Phi((x - w/2 - mu)/sigma)]

evaluated at the bin centre x of the BIN_WIDTH = w histogram, so the signal term is the exact
Gaussian content of the bin. f_bkg is a form of the published_functions registry in counts
per bin with the pivot PIVOT; every background parameter, the background normalisation
(inside f_bkg) and N_sig float. The fit is the Poisson likelihood (TH1::Fit "L") over
[FIT_LO, FIT_HI]. N_sig floats without limits (nsig_limits; config.NSIG_LIMITED restores the earlier
guard at -0.8 x the background within +-SIG_CORE_K sigma of the peak). The signal mean and width come
from the Stage-2 window map mu(m_WR), sigma(m_WR) of the channel.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config as C                                                       # noqa: E402
import published_functions as pf                                         # noqa: E402
import ROOT                                                              # noqa: E402
from fits import _uid, fit_poisson, make_tf1, seeds_for, th1             # noqa: E402
from spectrum import centers                                             # noqa: E402

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning
ROOT.TH1.AddDirectory(ROOT.kFALSE)
FIT_HI = 8000.0


def window_map(channel):
    p = json.loads(Path(C.WINDOW_JSON).read_text())[channel][C.TOPOLOGY]
    return p["mu"], p["sigma_median"]                 # [slope, intercept] each


def signal_pars(channel, m):
    (b_mu, a_mu), (b_s, a_s) = window_map(channel)
    return a_mu + b_mu * m, a_s + b_s * m


def signal_shape(edges, mu, sigma):
    """Per-bin content of a unit-yield Gaussian."""
    from math import erf, sqrt
    lo, hi = edges[:-1], edges[1:]
    cdf = lambda x: 0.5 * (1.0 + erf((x - mu) / (sigma * sqrt(2.0))))
    return np.array([cdf(b) - cdf(a) for a, b in zip(lo, hi)])


def pivot(name, mu=None):
    """The pivot of the form's parameter basis: PIVOT, or the signal mean of the fit for the forms of
    config.PIVOT_AT_SIGNAL (written around the window; the pivot never changes the curves)."""
    return mu if (mu is not None and name in C.PIVOT_AT_SIGNAL) else C.PIVOT


def sb_formula(name, mu, sigma):
    npar = pf.FUNCS[name][0]
    bkg = pf.formula(name, pivot(name, mu), C.SQRTS)
    w2 = C.BIN_WIDTH / 2.0
    s2 = sigma * math.sqrt(2.0)
    sig = (f"[{npar}]*0.5*(TMath::Erf((x+{w2:.10g}-{mu:.10g})/{s2:.10g})"
           f"-TMath::Erf((x-{w2:.10g}-{mu:.10g})/{s2:.10g}))")
    return f"({bkg})+({sig})", npar


def bkg_only(name, edges, values, variances, lo, hi, parents, m0=None):
    """Background-only Poisson fit of `name` to a spectrum (seeds from fitted parents); m0 = the pivot
    (PIVOT unless given)."""
    c = centers(edges)
    m0 = C.PIVOT if m0 is None else m0
    seeds = seeds_for(name, c, values, np.where(variances > 0, variances, np.maximum(values, 1e-9)),
                      m0, parents)
    return fit_poisson(name, edges, values, variances, lo, hi, m0, seeds)


def nsig_limits(b_core):
    """The limits on N_sig for a signal with b_core background events in its core: None (N_sig is free) unless
    config.NSIG_LIMITED restores the earlier guard."""
    return (-0.8 * b_core, max(5.0 * b_core + 50.0, 200.0)) if C.NSIG_LIMITED else None


class SBModel:
    """The S+B TF1 of one candidate at one signal mass, built once and reused for every toy
    (building a TF1 from a formula string compiles it; that was the whole cost of a toy fit)."""

    def __init__(self, name, mu, sigma, lo, hi):
        self.name, self.mu, self.sigma, self.lo, self.hi = name, mu, sigma, lo, hi
        formula, self.npar = sb_formula(name, mu, sigma)
        self.f = ROOT.TF1(f"_v3sb{_uid()}", formula, lo, hi)
        self.fb = ROOT.TF1(f"_v3sbb{_uid()}", pf.formula(name, pivot(name, mu), C.SQRTS), lo, hi)
        self.box = pf.limits(name, lo, hi)
        self.grid = np.linspace(lo, hi, 120)

    def fit(self, edges, data, bkg_seed, nsig_seed, nsig_bounds):
        f, npar, box = self.f, self.npar, self.box
        for i, sv in enumerate(list(bkg_seed)[:npar]):
            if box[i] is not None:
                sv = min(max(sv, box[i][0] + 1e-9), box[i][1] - 1e-9)
            f.SetParameter(i, float(sv))
        for i, b in enumerate(box):
            if b is not None:
                f.SetParLimits(i, b[0], b[1])
        if nsig_bounds is None:                       # N_sig free: (0, 0) releases the parameter of the reused TF1
            f.SetParameter(npar, float(nsig_seed))
            f.SetParLimits(npar, 0.0, 0.0)
        else:
            f.SetParameter(npar, float(min(max(nsig_seed, nsig_bounds[0]), nsig_bounds[1])))
            f.SetParLimits(npar, nsig_bounds[0], nsig_bounds[1])
        h = th1(edges, np.where(data > 0, data, 0.0), np.sqrt(np.maximum(data, 0.0)))
        r = h.Fit(f, "LRNSQ").Get()
        if not r or not math.isfinite(r.MinFcnValue()):
            return None
        params = np.array([r.Parameter(i) for i in range(npar + 1)])
        perr = np.array([r.ParError(i) for i in range(npar + 1)])
        at_limit = any(b is not None and (abs(params[i] - b[0]) < 1e-3 * (b[1] - b[0])
                                          or abs(b[1] - params[i]) < 1e-3 * (b[1] - b[0]))
                       for i, b in enumerate(box))
        if nsig_bounds is None:
            nsig_at_bound = False
        else:
            span = nsig_bounds[1] - nsig_bounds[0]
            nsig_at_bound = (abs(params[npar] - nsig_bounds[0]) < 1e-3 * span
                             or abs(nsig_bounds[1] - params[npar]) < 1e-3 * span)
        for i in range(npar):
            self.fb.SetParameter(i, float(params[i]))
        fg = np.array([self.fb.Eval(x) for x in self.grid])
        checks = {"valid": bool(r.IsValid()), "cov": int(r.CovMatrixStatus()) == 3,
                  "limit": not at_limit and not nsig_at_bound,
                  "mono": bool(np.all(fg[1:] <= fg[:-1] * (1.0 + 1e-6)))}
        return {"nsig": float(params[npar]), "nsig_err": float(perr[npar]), "bkg": params[:npar],
                "checks": checks, "passed": all(checks.values()), "dev": 2.0 * float(r.MinFcnValue())}


def fit_sb(name, edges, data, lo, hi, mu, sigma, bkg_seed, nsig_seed, nsig_bounds):
    """One-off S+B fit (builds the model; use SBModel for repeated fits)."""
    return SBModel(name, mu, sigma, lo, hi).fit(edges, data, bkg_seed, nsig_seed, nsig_bounds)
