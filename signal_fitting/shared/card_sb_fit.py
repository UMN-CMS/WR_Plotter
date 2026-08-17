#!/usr/bin/env python3
"""Homemade card-model S+B fit core -- the combine card's likelihood, exactly.

The Stage-6 toy machinery historically fit the Stage-4 TF1 chi2 model
(sqrt(N) bin errors, empty bins skipped). This module instead implements the
statistical model of the ACTUAL combine cards (8_combine_limits/optimization),
so homemade toys and combine run the same fit -- same binning, same window
snap, same constraints:

  binning     the card bin width (50 GeV for the k5_bw50 winner), snapped to
              bins whose CENTER lies in [fit_lo, fit_hi]
  likelihood  extended binned Poisson over ALL window bins (empty bins
              contribute nu_i -- nothing is skipped),
              -2 lnL = 2 sum_i [ nu_i - n_i ln nu_i ]  (+ const)
  signal      fixed-shape Gaussian(m_c, sigma), continuous normalization over
              the window, bin-center evaluation (RooFit binned convention);
              r is the signal yield in EVENTS (= r_fb x rate)

Two background modes, mirroring the card variants:

  float   nu_i = r S_i + B f_i(b)                    <-> card k5_bw50_float
          (single recentered expo, norm B and slope b free)
  fcr     nu_i = r S_i + mu_tt T_tt t_i + B f_i(b)   <-> card k5_bw50_fcr
          plus a one-bin flavor-CR channel nu_cr = mu_tt T_cr + C_other;
          t_i is the tt expo with the slope FIXED to b_tt (prepare_fcr
          component fit), so the tt norm is anchored by the CR bin through
          the shared mu_tt -- the homemade rateParam.

The likelihood machinery (bin probabilities, Minuit2 minimization) follows
explainer/combine_from_scratch.py CardModel; this module generalizes it to
the two-channel CR-anchored structure and exposes a fit_splusb-style call for
the toy scripts.

Source LCG_106 before importing.
"""
from __future__ import annotations

import math

import numpy as np
import ROOT

SQRT2 = math.sqrt(2.0)


def rebin_snap(edges, values, bw, fit_lo, fit_hi):
    """Native (10 GeV) -> card bin width, then keep bins whose center lies in
    [fit_lo, fit_hi]. Returns (snapped edges, snapped values)."""
    f = max(1, round(bw / float(edges[1] - edges[0])))
    n = (len(values) // f) * f
    vals = values[:n].reshape(-1, f).sum(axis=1)
    e = edges[0:n + 1:f]
    centers = 0.5 * (e[:-1] + e[1:])
    sel = (centers >= fit_lo) & (centers <= fit_hi)
    return np.append(e[:-1][sel], e[1:][sel][-1]), vals[sel]


class CardSB:
    """Card-model S+B fit at one mass point.

    Parameters
    ----------
    edges, m_c, sigma : snapped window bin edges and the signal Gaussian
    mode              : "float" or "fcr"
    fcr               : for mode "fcr", dict with T_tt, T_cr, C_other, b_tt
                        (the prepare_fcr json entry for this mass/variant)
    """

    def __init__(self, edges, m_c, sigma, mode="float", fcr=None):
        self.edges = np.asarray(edges, float)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.width = float(self.edges[1] - self.edges[0])
        self.lo, self.hi = float(self.edges[0]), float(self.edges[-1])
        self.m_c, self.sigma = float(m_c), float(sigma)
        self.mode = mode
        if mode == "fcr":
            self.T_tt, self.T_cr = float(fcr["T_tt"]), float(fcr["T_cr"])
            self.C_other, self.b_tt = float(fcr["C_other"]), float(fcr["b_tt"])
            self.t = self._expo_shape(self.b_tt)     # fixed tt bin probabilities
        g = (np.exp(-0.5 * ((self.centers - m_c) / sigma) ** 2)
             / (sigma * math.sqrt(2 * math.pi)))
        norm = 0.5 * (math.erf((self.hi - m_c) / (SQRT2 * sigma))
                      - math.erf((self.lo - m_c) / (SQRT2 * sigma)))
        self.S = g * self.width / norm               # signal bin probabilities

    def _expo_shape(self, b):
        """Recentered-expo bin probabilities, RooFit binned convention
        (pdf(center) * width / continuous norm over the window)."""
        vals = np.exp(b * (self.centers - self.m_c) / 1000.0)
        if abs(b) < 1e-9:
            norm = self.hi - self.lo
        else:
            k = b / 1000.0
            norm = (math.exp(k * (self.hi - self.m_c))
                    - math.exp(k * (self.lo - self.m_c))) / k
        return vals * self.width / norm

    # -- expected yields ------------------------------------------------------

    def nu(self, r, pars):
        """SR expectation per bin. pars = (b, B) for float,
        (mu_tt, b, B) for fcr."""
        if self.mode == "float":
            b, B = pars
            return np.clip(r * self.S + B * self._expo_shape(b), 1e-12, None)
        mu_tt, b, B = pars
        return np.clip(r * self.S + mu_tt * self.T_tt * self.t
                       + B * self._expo_shape(b), 1e-12, None)

    def nu_cr(self, pars):
        """Flavor-CR one-bin expectation (fcr mode only)."""
        mu_tt = pars[0]
        return max(mu_tt * self.T_cr + self.C_other, 1e-12)

    def nll2(self, r, pars, data, data_cr=None):
        """-2 lnL, extended binned Poisson over all bins (+ the CR bin in fcr
        mode), additive constants dropped. In fcr mode, data_cr=None drops the
        CR term (an SR-only likelihood of the split model -- diagnostics)."""
        v = self.nu(r, pars)
        out = 2.0 * float(np.sum(v) - np.sum(data * np.log(v)))
        if self.mode == "fcr" and data_cr is not None:
            vc = self.nu_cr(pars)
            out += 2.0 * (vc - float(data_cr) * math.log(vc))
        return out

    # -- fits ------------------------------------------------------------------

    def _minimize(self, fcn, start, limits):
        mnz = ROOT.Math.Factory.CreateMinimizer("Minuit2", "Migrad")
        ROOT.SetOwnership(mnz, True)
        mnz.SetErrorDef(1.0)
        mnz.SetStrategy(1)
        mnz.SetTolerance(0.05)
        mnz.SetMaxFunctionCalls(50000)
        mnz.SetPrintLevel(-1)
        functor = ROOT.Math.Functor(fcn, len(start))
        mnz.SetFunction(functor)
        for i, (s0, lim) in enumerate(zip(start, limits)):
            if lim is None:
                mnz.SetVariable(i, f"p{i}", s0, 0.1)
            else:
                mnz.SetLimitedVariable(i, f"p{i}", s0, 0.1, lim[0], lim[1])
        ok = mnz.Minimize()
        if not (ok and mnz.Status() == 0):
            mnz.SetStrategy(2)
            ok = mnz.Minimize()
        pars = [mnz.X()[i] for i in range(len(start))]
        errs = [mnz.Errors()[i] for i in range(len(start))]
        return pars, errs, float(mnz.MinValue()), int(mnz.Status())

    def _starts_limits(self, data):
        ntot = max(float(np.sum(data)), 1.0)
        if self.mode == "float":
            return ([-3.0, ntot],
                    [(-100.0, 0.0), (0.0, 10.0 * ntot + 50.0)])
        return ([1.0, -3.0, max(ntot - self.T_tt, 1.0)],
                [(0.0, 5.0), (-100.0, 0.0), (0.0, 10.0 * ntot + 50.0)])

    def fit(self, data, data_cr=None, r_bounds=(-2000.0, 2000.0)):
        """Global S+B fit (r floating). Returns a fit_splusb-style dict:
        nsig/nsig_err are r-hat (EVENTS) and its Minuit error; params/perr
        hold the background parameters ((b,B) or (mu_tt,b,B))."""
        s0, lims = self._starts_limits(data)
        nb = len(s0)

        def fcn(p):
            # p is a raw double* from Minuit2 -- index element-wise, never slice
            v = self.nll2(p[0], tuple(p[i + 1] for i in range(nb)),
                          data, data_cr)
            return v if math.isfinite(v) else 1e30
        pars, errs, nll, status = self._minimize(
            fcn, [0.0] + s0, [r_bounds] + lims)
        return dict(nsig=pars[0], nsig_err=errs[0],
                    params=np.array(pars[1:]), perr=np.array(errs[1:]),
                    nll2=nll, status=status)

    def fit_profiled(self, r, data, data_cr=None):
        """Fix r, minimize over the background parameters."""
        s0, lims = self._starts_limits(data)
        nb = len(s0)

        def fcn(p):
            v = self.nll2(r, tuple(p[i] for i in range(nb)), data, data_cr)
            return v if math.isfinite(v) else 1e30
        pars, errs, nll, status = self._minimize(fcn, s0, lims)
        return dict(params=np.array(pars), perr=np.array(errs),
                    nll2=nll, status=status)
