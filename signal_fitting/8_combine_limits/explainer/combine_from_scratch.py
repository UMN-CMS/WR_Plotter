#!/usr/bin/env python3
"""Shared engine -- CMS combine rebuilt from scratch, one object.

Implements, in ~300 lines of plain numpy + ROOT-Minuit2, the exact statistical
model of the run2 m_WR = 2000 FLOAT card `card_k5_bw50_float_m2000.txt` (the
10.9 optimization geometry: k5 window [1150, 2700], 50 GeV bins, floating
expo background, fixed Gaussian signal), so each step*.py script can reproduce
one piece of what combine does internally and check it against genuine combine
numbers on the identical card (reference_fitdiag.json, key `k5f_m2000`).

Units: combine's POI on this card is sigma x BR in fb (rate = 24.34 ev/fb);
here we work in EVENTS -- r_events = r_fb * rate, exactly equivalent in the
likelihood (r_fb * rate * S_i == r_events * S_i) -- so every likelihood
picture stays in yields, the language of the homemade N_sp machinery. The
reference numbers are pre-converted to events (fb originals kept under "fb").

  model       nu_i(r, b, B) = r * S_i  +  B * f_i(b)          per 50 GeV bin
              S_i = Gaussian(m_c, sigma) bin probability (continuous
              normalization over the window, bin-centre evaluation -- the
              RooFit convention for a binned dataset),
              f_i(b) = recentered falling expo, same convention.
  likelihood  binned extended Poisson:
              -2 lnL = 2 * sum_i [ nu_i - n_i ln nu_i ]        (+ const)
              n_i are the MC window contents ("data"); empty bins contribute
              2*nu_i -- nothing is skipped.
  profiling   at fixed r, minimize over the background (b, B) -> "conditional"
              MLEs theta-hat-hat(r); global fit floats everything.
  q_mu tilde  the LHC test statistic (CCGV eq. 16):
                mu-hat > mu :  0
                mu-hat < 0  :  2[lnL(mu-hat=0 profiled) - lnL(mu profiled)]
                else        :  2[lnL(mu-hat global)     - lnL(mu profiled)]
  toys        bin-wise Poisson draws of nu_i at the tested hypothesis with the
              background at its CONDITIONAL MLE (the LHC-style frequentist
              convention combine uses).
  CLs         p_{s+b} / (1 - p_b) from the two q_mu tail probabilities.
  Asimov      data_i = nu_i(r = 0, theta-hat(0));  sigma_A = mu / sqrt(q_mu(A)).

Source LCG_106 before importing.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                        # explainer/
STAGE9 = HERE.parent / "baseline"
OPT = HERE.parent / "optimization"                            # 10.9 run2 inputs
FLOAT_VAR = "k5_bw50"                                         # the winner card
PARITY = HERE.parents[1] / "archived" / "10_limit_refinement" / "6_combine_parity"

import ROOT                                                   # noqa: E402
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError

SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Card model
# ---------------------------------------------------------------------------

class CardModel:
    """The run2 k5_bw50 float-card statistical model at one mass (floating
    expo background, fixed-shape Gaussian signal; r in EVENTS = fb x rate)."""

    def __init__(self, mass=2000, channel="ee", topology="resolved"):
        tag = f"{channel}_{topology}"
        with open(OPT / "inputs" / f"{tag}.json") as fh:
            m = json.load(fh)["masses"][str(int(mass))]
        v = m["vars"][FLOAT_VAR]
        self.mass, self.m_c, self.sigma = float(mass), m["m_c"], m["sigma"]
        self.fit_lo, self.fit_hi = v["fit_lo"], v["fit_hi"]
        self.rate = v["rate_per_fb"]              # ev per fb of sigma x BR

        import uproot
        h = uproot.open(OPT / "inputs" / f"{tag}.root")["bkg_native"]
        e10, v10 = h.axes[0].edges(), h.values()  # native 10 GeV bins
        f = max(1, round(v["bw"] / float(e10[1] - e10[0])))
        nfull = (len(v10) // f) * f               # rebin native -> card bw
        edges = e10[0:nfull + 1:f]
        vals = np.clip(v10[:nfull].reshape(-1, f).sum(axis=1), 0.0, None)
        centers = 0.5 * (edges[:-1] + edges[1:])
        # snap: bins whose CENTER is inside [fit_lo, fit_hi] (TH1::Fit rule,
        # same as make_opt_workspaces.snap)
        sel = (centers >= self.fit_lo) & (centers <= self.fit_hi)
        self.centers = centers[sel]
        self.width = float(edges[1] - edges[0])
        self.lo = float(edges[:-1][sel][0])
        self.hi = float(edges[1:][sel][-1])
        self.n = vals[sel].astype(float)          # the "data" (MC Asimov-like)
        self.nbins = len(self.n)

        # signal bin probabilities: pdf(center)*width / continuous norm over
        # [lo, hi]  (RooFit binned-dataset convention)
        g = self._gauss(self.centers)
        norm = 0.5 * (math.erf((self.hi - self.m_c) / (SQRT2 * self.sigma))
                      - math.erf((self.lo - self.m_c) / (SQRT2 * self.sigma)))
        self.S = g * self.width / norm

    def _gauss(self, x):
        return (np.exp(-0.5 * ((x - self.m_c) / self.sigma) ** 2)
                / (self.sigma * math.sqrt(2 * math.pi)))

    def bkg_shape(self, b):
        """f_i(b): expo bin probabilities, same convention (analytic norm)."""
        s = (self.centers - self.m_c) / 1000.0
        vals = np.exp(b * s)
        if abs(b) < 1e-9:
            norm = (self.hi - self.lo)
        else:
            k = b / 1000.0
            norm = (math.exp(k * (self.hi - self.m_c))
                    - math.exp(k * (self.lo - self.m_c))) / k
        return vals * self.width / norm

    def nu(self, r, b, B):
        """Expected events per bin."""
        return np.clip(r * self.S + B * self.bkg_shape(b), 1e-12, None)

    # -- likelihood ----------------------------------------------------------

    def nll2(self, r, b, B, data=None):
        """-2 lnL (extended binned Poisson, additive constant dropped)."""
        n = self.n if data is None else data
        v = self.nu(r, b, B)
        return 2.0 * float(np.sum(v) - np.sum(n * np.log(v)))

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
            mnz.Minimize()
        pars = [mnz.X()[i] for i in range(len(start))]
        errs = [mnz.Errors()[i] for i in range(len(start))]
        return pars, errs, float(mnz.MinValue())

    def fit_global(self, data=None, r_bounds=(-2000.0, 2000.0)):
        """Float everything: returns (r_hat, b_hat, B_hat), errors, -2lnL_min."""
        n = self.n if data is None else data
        ntot = max(float(np.sum(n)), 1.0)

        def fcn(p):
            v = self.nll2(p[0], p[1], p[2], data)
            return v if math.isfinite(v) else 1e30
        return self._minimize(
            fcn, [0.0, -3.0, ntot],
            [r_bounds, (-100.0, 0.0), (0.0, 10.0 * ntot + 50.0)])

    def fit_profiled(self, r, data=None):
        """Fix r, minimize over (b, B): returns (b, B), errors, -2lnL(r)."""
        n = self.n if data is None else data
        ntot = max(float(np.sum(n)), 1.0)

        def fcn(p):
            v = self.nll2(r, p[0], p[1], data)
            return v if math.isfinite(v) else 1e30
        return self._minimize(
            fcn, [-3.0, ntot], [(-100.0, 0.0), (0.0, 10.0 * ntot + 50.0)])

    # -- test statistic --------------------------------------------------------

    def qmu_tilde(self, mu, data=None):
        """The LHC test statistic q~_mu (CCGV eq. 16) for one dataset.
        Returns (q, mu_hat)."""
        (r_hat, b_hat, B_hat), _, nll_glob = self.fit_global(data)
        _, _, nll_mu = self.fit_profiled(mu, data)
        if r_hat > mu:
            return 0.0, r_hat
        if r_hat < 0.0:
            _, _, nll_0 = self.fit_profiled(0.0, data)
            return max(nll_mu - nll_0, 0.0), r_hat
        return max(nll_mu - nll_glob, 0.0), r_hat

    # -- toys ------------------------------------------------------------------

    def toys(self, r, ntoys, seed, data=None):
        """LHC-style toys of hypothesis r: background at its conditional MLE
        theta-hat-hat(r) for the given dataset, bin-wise Poisson draws."""
        (b_c, B_c), _, _ = self.fit_profiled(r, data)
        v = self.nu(r, b_c, B_c)
        rng = ROOT.TRandom3(seed)
        return [np.array([rng.Poisson(x) for x in v], dtype=float)
                for _ in range(ntoys)]

    def cls_toys(self, mu, ntoys, seed, data=None):
        """CLs at hypothesis mu from toy distributions of q~_mu.
        Returns dict(cls, clsb, clb, q_obs, q_sb, q_b)."""
        q_obs, _ = self.qmu_tilde(mu, data)
        q_sb = np.array([self.qmu_tilde(mu, t)[0]
                         for t in self.toys(mu, ntoys, seed, data)])
        q_b = np.array([self.qmu_tilde(mu, t)[0]
                        for t in self.toys(0.0, ntoys, seed + 1, data)])
        clsb = float(np.mean(q_sb >= q_obs))          # p_{s+b}
        clb = float(np.mean(q_b >= q_obs))            # 1 - p_b
        cls = clsb / clb if clb > 0 else 1.0
        return dict(cls=cls, clsb=clsb, clb=clb, q_obs=q_obs,
                    q_sb=q_sb, q_b=q_b)

    # -- Asimov / asymptotics ---------------------------------------------------

    def asimov(self, r=0.0):
        """The Asimov dataset of hypothesis r (background at theta-hat(r) for
        the nominal data)."""
        (b_c, B_c), _, _ = self.fit_profiled(r)
        return self.nu(r, b_c, B_c)

    def sigma_A(self, mu):
        """CCGV: sigma_A = mu / sqrt(q_mu(Asimov(0)))."""
        qa, _ = self.qmu_tilde(mu, data=self.asimov(0.0))
        return mu / math.sqrt(qa) if qa > 0 else float("inf")

    def asymptotic_limit(self, alpha=0.05, quantile=None, tol=1e-3):
        """AsymptoticLimits from scratch: solve CLs(mu)=alpha with the CCGV
        asymptotic formulas, re-deriving sigma_A(mu) at every tested mu (this
        iteration is exactly what combine does).  quantile=None -> observed
        (on self.n); else the expected band point at that background quantile
        N = Phi^-1(quantile)."""
        def cls_asym(mu):
            s = self.sigma_A(mu)
            if quantile is None:
                q_obs, _ = self.qmu_tilde(mu)
                if q_obs <= 0:
                    return 1.0
                sq = math.sqrt(q_obs)
                # CCGV eqs. 66/59 for q~_mu (with the q > mu^2/sigma^2 branch)
                if q_obs <= (mu / s) ** 2:
                    clsb = 1.0 - ROOT.TMath.Freq(sq)
                    clb = 1.0 - ROOT.TMath.Freq(sq - mu / s)
                else:
                    clsb = 1.0 - ROOT.TMath.Freq(
                        (q_obs + (mu / s) ** 2) / (2 * mu / s))
                    clb = 1.0 - ROOT.TMath.Freq(
                        (q_obs - (mu / s) ** 2) / (2 * mu / s))
                return clsb / clb if clb > 0 else 1.0
            N = ROOT.TMath.NormQuantile(quantile)
            clsb = 1.0 - ROOT.TMath.Freq(mu / s - N)
            clb = 1.0 - ROOT.TMath.Freq(-N)
            return clsb / clb if clb > 0 else 1.0
        lo_r, hi_r = 0.0, 10.0
        while cls_asym(hi_r) > alpha and hi_r < 1e5:
            hi_r *= 2
        while hi_r - lo_r > tol * (1 + hi_r):
            mid = 0.5 * (lo_r + hi_r)
            if cls_asym(mid) > alpha:
                lo_r = mid
            else:
                hi_r = mid
        return 0.5 * (lo_r + hi_r)


# ---------------------------------------------------------------------------
# Reference numbers from the 10.6 parity run (the answer key)
# ---------------------------------------------------------------------------

def load_reference(mass=2000, variant="k5f", channel="ee",
                   topology="resolved"):
    """Combine's numbers for this card (events).

    variant "k5f" (default): everything from reference_fitdiag.json -- the
    run2 k5_bw50 float card (FitDiagnostics + the 10.9 AsymptoticLimits scan),
    pre-converted to events. Legacy variants ("fixed", "prior", "anchored")
    fall back to the archived 10.6 parity CSV (k3_bw100, run3).
    """
    out = {}
    ref_json = HERE / "reference_fitdiag.json"
    if ref_json.exists():
        with open(ref_json) as fh:
            fd = json.load(fh).get(f"{variant}_m{int(mass)}")
        if fd:
            out["fitdiag"] = fd
            if "ul_med" in fd:
                out["asymptotic"] = {k: v for k, v in fd.items()
                                     if k.startswith("ul_")}
    if variant in ("fixed", "prior", "anchored"):
        import csv
        path = PARITY / f"parity_table_{channel}_{topology}.csv"
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                if int(float(r["mWR"])) == int(mass) and r["variant"] == variant:
                    out[r["method"]] = {
                        k: float(v) for k, v in r.items()
                        if k.startswith("ul_") and v not in ("", None)}
    return out


def check(label, ours, ref, tol=0.05):
    """Print a pass/fail line comparing our number with combine's."""
    if ref is None or not math.isfinite(ours):
        print(f"  CHECK {label:38s} ours={ours:9.3f}   (no reference)")
        return
    d = ours / ref - 1.0
    ok = "PASS" if abs(d) <= tol else "FAIL"
    print(f"  CHECK {label:38s} ours={ours:9.3f}  combine={ref:9.3f}  "
          f"({d:+.1%})  {ok}")


def check_abs(label, ours, ref, scale, tol=0.10):
    """Pass/fail on |ours - ref| <= tol * scale -- for quantities whose
    natural size is `scale` (e.g. a near-zero best fit measured in sigma):
    a relative check on a number compatible with zero is meaningless."""
    if ref is None or not math.isfinite(ours):
        print(f"  CHECK {label:38s} ours={ours:9.3f}   (no reference)")
        return
    d = (ours - ref) / scale
    ok = "PASS" if abs(d) <= tol else "FAIL"
    print(f"  CHECK {label:38s} ours={ours:9.3f}  combine={ref:9.3f}  "
          f"({d:+.3f} of scale {scale:.3g})  {ok}")
