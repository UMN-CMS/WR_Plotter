#!/usr/bin/env python3
"""Registry of published background functions for falling mass spectra.

Each entry is a functional form that a CMS or ATLAS search used as the
background model of a smoothly falling mass spectrum (fitted to simulation or
to data), written as a ROOT TF1 formula of the mass x in GeV with a pivot m0
(GeV) and the centre-of-mass energy sqrts (GeV). Variables used in the
formulas and labels:

  u = (x - m0)/1000   TeV offset from the pivot (slope terms)
  t = x/1000          mass in TeV
  r = x/m0            mass over the pivot (power laws)
  z = x/sqrts         the dijet variable

The pivot only changes the parameter basis (a = log f at m0), never the curve.
Normalisation is always the free parameter [0] (log-normalisation) so every
form is "shape x free norm", as the papers use them. Parameter limits are
sanity boxes only; monotonicity is checked after the fit, not imposed, so the
published forms are tested as published.

`PARENT[name]` lists the forms a function reduces to when one parameter is
removed (the nesting used for F-tests). `SOURCE[name]` names the analyses that
used the form.
"""
from __future__ import annotations

import math

import numpy as np

T0 = 1000.0

# name -> (npar, display label, parameter names)
FUNCS = {
    "expo":       (2, r"$e^{a+bu}$", ["a", "b"]),
    "expo2":      (3, r"$e^{a+bu+cu^{2}}$", ["a", "b", "c"]),
    "expo3":      (4, r"$e^{a+bu+cu^{2}+du^{3}}$", ["a", "b", "c", "d"]),
    # expo2 with a linear normalisation at the pivot, falling slope and curvature
    # positive; the same curves as expo2 (N = e^a, b -> -b, c -> -c), another parameter basis
    "expo2n":     (3, r"$N\,e^{-bu-cu^{2}}$", ["N", "b", "c"]),
    "powlaw":     (2, r"$e^{a}\,r^{b}$", ["a", "b"]),
    "powexp":     (3, r"$e^{a+cu}\,r^{b}$", ["a", "b", "c"]),
    # the same two forms in the bare mass (GeV): exp(a + c m) and m^b exp(a + c m); the fit is invariant to the
    # reparametrisation, only Minuit's numerics differ
    "exporaw":    (2, r"$e^{a+cm}$", ["a", "c"]),
    "powexpraw":  (3, r"$m^{b}\,e^{a+cm}$", ["a", "b", "c"]),
    "dilep8":     (4, r"$e^{a+cu+du^{2}}\,r^{b}$", ["a", "b", "c", "d"]),
    "dilep16":    (5, r"$e^{a+cu+du^{2}+eu^{3}}\,r^{b}$", ["a", "b", "c", "d", "e"]),
    "wprime7":    (3, r"$e^{a}\,[(x+b)/(m_{0}+b)]^{-c}$", ["a", "b", "c"]),
    "dijet3":     (3, r"$e^{a}\,(1-z)^{b}\,r^{-c}$", ["a", "b", "c"]),
    "dijet4":     (4, r"$e^{a}\,(1-z)^{b}\,r^{-(c+d\ln r)}$", ["a", "b", "c", "d"]),
    "dijet5":     (5, r"$e^{a}\,(1-z)^{b}\,r^{-(c+d\ln r+e\ln^{2}r)}$", ["a", "b", "c", "d", "e"]),
    "levexp":     (3, r"$e^{a}\,e^{-t/(b+ct)}$", ["a", "b", "c"]),
    "expinv":     (3, r"$e^{a+bt+c/t}$", ["a", "b", "c"]),
    "atlasemu":   (3, r"$e^{a}\,r^{\,b+c\ln r}$", ["a", "b", "c"]),
    "atlasdilep": (5, r"$e^{a}\,(1-z^{b})^{c}\,r^{\,d+e\ln r}$", ["a", "b", "c", "d", "e"]),
    "dexp":       (4, r"$e^{a+bu}+e^{c+du}$", ["a", "b", "c", "d"]),
}

# the analyses that used each form (public papers; the survey notes hold the equation numbers)
SOURCE = {
    "expo":       "CMS W_R 7 TeV EXO-11-091 (per process); the production anchor; hybrid tail",
    "expo2":      "bkg_fit_lib expo2 (exponent quadratic); EXO-16-047 segments without the power",
    "expo3":      "bkg_fit_lib expo3 (exponent cubic); EXO-15-005 exponent",
    "expo2n":     "expo2 written as [0]*exp(-([1]/1000)(x-m0)-([2]/1e6)(x-m0)^2) (a parameter-basis check)",
    "powlaw":     "ATLAS LQ 2006.05872 tt tail a m^b; B2G-19-002 power-law tails",
    "powexp":     "CMS dilepton 7 TeV EXO-11-019 m^-k e^-am; ATLAS boosted W_R 1904.12679 A e^-Bu/u^C",
    "exporaw":    "expo written as exp(a + c m), m in GeV: no pivot, no TeV scaling (a numerics check)",
    "powexpraw":  "powexp written as m^b exp(a + c m), m in GeV (a numerics check)",
    "dilep8":     "CMS dilepton 8 TeV EXO-12-061; W' 8 TeV EXO-12-060; PAS EXO-25-021; UA2 dijet form",
    "dilep16":    "CMS dilepton EXO-15-005 / EXO-16-047 (cubic exponent x power)",
    "wprime7":    "CMS W' 7 TeV EXO-11-024 a/(M_T+b)^c",
    "dijet3":     "dijet function, 3 parameters (CMS EXO-16-056 alternative)",
    "dijet4":     "dijet function, 4 parameters (CMS EXO-19-012, EXO-16-056)",
    "dijet5":     "dijet function, 5 parameters (CMS EXO-19-012 alternative)",
    "levexp":     "levelled exponential e^{-x/(a+bx)} (B2G alpha method: B2G-16-026, B2G-17-004)",
    "expinv":     "e^{ax+b/x} (B2G alpha method)",
    "atlasemu":   "ATLAS e-mu 1807.06573 a m^{b+c ln m}",
    "atlasdilep": "ATLAS dilepton 1903.06248 (1-z^c)^b z^{p0+p1 ln z}, Breit-Wigner factor dropped",
    "dexp":       "sum of two exponentials (CMS HIG-17-012 tail); bkg_fit_lib dexp",
}

# child -> parents with one parameter fewer (nested forms for the F-test)
PARENT = {
    "expo2": ["expo"], "expo3": ["expo2"],
    "powexp": ["expo", "powlaw"],
    "powexpraw": ["exporaw"],
    "dilep8": ["powexp"], "dilep16": ["dilep8"],
    "wprime7": ["powlaw"],
    "dijet4": ["dijet3"], "dijet5": ["dijet4"],
    "levexp": ["expo"], "expinv": ["expo"],
    "atlasemu": ["powlaw"], "atlasdilep": ["dijet4"],
    "dexp": ["expo"],
}

# plotting: one colour per family, line style within the family
COLOR = {
    "expo": "#000000", "expo2": "#5790fc", "expo3": "#5790fc", "expo2n": "#e42536",
    "powlaw": "#964a8b", "powexp": "#f89c20", "dilep8": "#f89c20", "dilep16": "#f89c20",
    "exporaw": "#5790fc", "powexpraw": "#e42536",
    "wprime7": "#964a8b",
    "dijet3": "#e42536", "dijet4": "#e42536", "dijet5": "#e42536",
    "levexp": "#2ca02c", "expinv": "#2ca02c",
    "atlasemu": "#9c9ca1", "atlasdilep": "#7a21dd",
    "dexp": "#717581",
}
LS = {
    "expo": "-", "expo2": "-", "expo3": "--", "expo2n": "--",
    "powlaw": "-", "powexp": "-", "dilep8": "--", "dilep16": ":",
    "exporaw": "--", "powexpraw": "--",
    "wprime7": "--",
    "dijet3": "-", "dijet4": "--", "dijet5": ":",
    "levexp": "-", "expinv": "--",
    "atlasemu": "-", "atlasdilep": "-",
    "dexp": "-",
}

ORDER = list(FUNCS)


def _u(m0):
    return f"((x-{m0:.10g})/1.e3)"


def formula(name, m0, sqrts):
    """ROOT TF1 formula string of the mass x in GeV."""
    u, t, r, z = _u(m0), "(x/1.e3)", f"(x/{m0:.10g})", f"(x/{sqrts:.10g})"
    if name == "expo":
        return f"exp([0]+[1]*{u})"
    if name == "expo2":
        return f"exp([0]+[1]*{u}+[2]*pow({u},2))"
    if name == "expo3":
        return f"exp([0]+[1]*{u}+[2]*pow({u},2)+[3]*pow({u},3))"
    if name == "expo2n":
        return f"[0]*exp(-(([1]/1000.0)*(x-{m0:.10g}))-([2]/1000000)*(x-{m0:.10g})*(x-{m0:.10g}))"
    if name == "powlaw":
        return f"exp([0])*pow({r},[1])"
    if name == "powexp":
        return f"exp([0]+[2]*{u})*pow({r},[1])"
    if name == "exporaw":
        return "exp([0]+[1]*x)"
    if name == "powexpraw":
        return "exp([0]+[2]*x)*pow(x,[1])"
    if name == "dilep8":
        return f"exp([0]+[2]*{u}+[3]*pow({u},2))*pow({r},[1])"
    if name == "dilep16":
        return f"exp([0]+[2]*{u}+[3]*pow({u},2)+[4]*pow({u},3))*pow({r},[1])"
    if name == "wprime7":
        return f"exp([0])*pow((x+[1])/({m0:.10g}+[1]),-[2])"
    # dijet forms: the power terms run in r = x/m0 = z/z0 (a pivot at m0); the family
    # is unchanged (ln z = ln r + ln z0 is absorbed by a and c), the Hessian is conditioned
    if name == "dijet3":
        return f"exp([0])*pow(1.-{z},[1])*pow({r},-[2])"
    if name == "dijet4":
        return f"exp([0])*pow(1.-{z},[1])*pow({r},-([2]+[3]*log({r})))"
    if name == "dijet5":
        return (f"exp([0])*pow(1.-{z},[1])*pow({r},-([2]+[3]*log({r})"
                f"+[4]*pow(log({r}),2)))")
    if name == "levexp":
        return f"exp([0])*exp(-{t}/([1]+[2]*{t}))"
    if name == "expinv":
        return f"exp([0]+[1]*{t}+[2]/{t})"
    if name == "atlasemu":
        return f"exp([0])*pow({r},[1]+[2]*log({r}))"
    if name == "atlasdilep":
        return f"exp([0])*pow(1.-pow({z},[1]),[2])*pow({r},[3]+[4]*log({r}))"
    if name == "dexp":
        return f"exp([0]+[1]*{u})+exp([2]+[3]*{u})"
    raise KeyError(name)


def limits(name, lo, hi):
    """Sanity boxes per parameter (None = free). Only what Minuit needs to stay
    away from poles and sign flips of the formula; no slope-sign constraints."""
    n = FUNCS[name][0]
    box = [None] * n
    if name == "wprime7":
        box[1] = (-(lo - 50.0), 50000.0)     # keep x + b > 0 over the fit range
        box[2] = (0.0, 300.0)
    elif name == "levexp":
        box[1] = (1e-3, 50.0)                # b in TeV > 0
        box[2] = (0.0, 20.0)                 # c >= 0: no pole, slope "levels" at high mass
    elif name == "atlasdilep":
        box[1] = (0.05, 20.0)
    elif name == "dexp":
        box[0] = (-100.0, 100.0)
        box[2] = (-100.0, 100.0)
    return box


# ---------------------------------------------------------------------------
# log-linear least-squares seeds (the forms that are linear in log space)
# ---------------------------------------------------------------------------

def _design(name, m, m0, sqrts):
    u = (m - m0) / T0
    t = m / T0
    r = m / m0
    z = m / sqrts
    o = np.ones_like(m)
    if name == "expo":     return np.column_stack([o, u])
    if name == "exporaw":  return np.column_stack([o, m])
    if name == "powexpraw": return np.column_stack([o, np.log(m), m])
    if name == "expo2":    return np.column_stack([o, u, u ** 2])
    if name == "expo3":    return np.column_stack([o, u, u ** 2, u ** 3])
    if name == "powlaw":   return np.column_stack([o, np.log(r)])
    if name == "powexp":   return np.column_stack([o, np.log(r), u])
    if name == "dilep8":   return np.column_stack([o, np.log(r), u, u ** 2])
    if name == "dilep16":  return np.column_stack([o, np.log(r), u, u ** 2, u ** 3])
    if name == "dijet3":   return np.column_stack([o, np.log(1 - z), -np.log(r)])
    if name == "dijet4":   return np.column_stack([o, np.log(1 - z), -np.log(r), -np.log(r) ** 2])
    if name == "dijet5":   return np.column_stack([o, np.log(1 - z), -np.log(r), -np.log(r) ** 2,
                                                   -np.log(r) ** 3])
    if name == "expinv":   return np.column_stack([o, t, 1.0 / t])
    if name == "atlasemu": return np.column_stack([o, np.log(r), np.log(r) ** 2])
    return None


def _loglin(name, m, n, var, m0, sqrts):
    Phi = _design(name, m, m0, sqrts)
    if Phi is None or Phi.shape[0] <= Phi.shape[1]:
        return None
    sw = n / np.sqrt(var)
    try:
        coef, *_ = np.linalg.lstsq(Phi * sw[:, None], np.log(n) * sw, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return list(coef) if np.all(np.isfinite(coef)) else None


def seed(name, m, n, var, m0, sqrts):
    """Starting values from a weighted log-space linear fit on the populated
    bins (m, n, var), or derived from the expo / dijet4 seeds."""
    if name == "expo2n":                                 # the expo2 seed in the expo2n basis
        e2 = _loglin("expo2", m, n, var, m0, sqrts)
        return [math.exp(e2[0]), -e2[1], -e2[2]] if e2 is not None else None
    direct = _loglin(name, m, n, var, m0, sqrts)
    if direct is not None:
        return direct
    e = _loglin("expo", m, n, var, m0, sqrts)
    if e is None:
        return None
    a, b = e
    t0 = m0 / T0
    if name == "levexp":
        b = min(b, -1e-3)
        return [a - b * t0 + 0.0, -1.0 / b, 0.01]        # e^{a+b(t-t0)} = e^{a-bt0} e^{-t/(-1/b)}
    if name == "wprime7":
        p = _loglin("powlaw", m, n, var, m0, sqrts)
        return [p[0], 0.0, max(-p[1], 1.0)] if p else [a, 0.0, 5.0]
    if name == "atlasdilep":
        d4 = _loglin("dijet4", m, n, var, m0, sqrts)
        if d4 is None:
            return None
        a4, b4, c4, d4_ = d4
        return [a4, 1.0, b4, -c4, -d4_]
    if name == "dexp":
        return [a, 0.7 * b, a - 1.5, 1.5 * b]
    return None


def embed(child, parent, params, m0):
    """Start values of `child` that reproduce the fitted `parent` curve exactly
    (the extra parameter switched off), or None when no embedding exists."""
    p = list(map(float, params))
    t0 = m0 / T0
    key = (child, parent)
    if key in {("expo2", "expo"), ("expo3", "expo2"), ("dilep8", "powexp"),
               ("dilep16", "dilep8"), ("dijet4", "dijet3"), ("dijet5", "dijet4"),
               ("powexp", "powlaw"), ("atlasemu", "powlaw")}:
        return p + [0.0]
    if key == ("powexp", "expo"):
        return [p[0], 0.0, p[1]]
    if key == ("powexpraw", "exporaw"):
        return [p[0], 0.0, p[1]]
    if key == ("wprime7", "powlaw"):
        return [p[0], 0.0, -p[1]]
    if key == ("levexp", "expo"):
        b = min(p[1], -1e-3)
        return [p[0] - b * t0, -1.0 / b, 0.0]
    if key == ("expinv", "expo"):
        return [p[0] - p[1] * t0, p[1], 0.0]
    if key == ("atlasdilep", "dijet4"):
        return [p[0], 1.0, p[1], -p[2], -p[3]]
    if key == ("dexp", "expo"):
        return [p[0], p[1], p[0] - 4.0, p[1]]
    return None


def param_text(name, params, perr=None):
    """Lines 'a = v +- e' for a stat box."""
    syms = FUNCS[name][2]
    out = []
    for i, v in enumerate(params):
        if perr is not None and i < len(perr) and np.isfinite(perr[i]) and perr[i] > 0:
            out.append(rf"${syms[i]}={v:.3g} \pm {perr[i]:.2g}$")
        else:
            out.append(rf"${syms[i]}={v:.3g}$")
    return "\n".join(out)


def is_monotonic(f_vals):
    f = np.asarray(f_vals, float)
    return bool(np.all(f[1:] <= f[:-1] * (1.0 + 1e-6)))


def f_pvalue(chi2_parent, chi2_child, npar_parent, npar_child, nbins):
    """Fisher F-test p-value for adding (npar_child - npar_parent) parameters."""
    import ROOT
    dnp = npar_child - npar_parent
    dof2 = nbins - npar_child
    if dnp <= 0 or dof2 <= 0 or chi2_child <= 0:
        return float("nan")
    F = ((chi2_parent - chi2_child) / dnp) / (chi2_child / dof2)
    if F <= 0:
        return 1.0
    return float(ROOT.Math.fdistribution_cdf_c(F, dnp, dof2))


def chi2_pvalue(chi2, ndf):
    import ROOT
    if ndf <= 0 or not math.isfinite(chi2):
        return float("nan")
    return float(ROOT.Math.chisquared_cdf_c(chi2, ndf))
