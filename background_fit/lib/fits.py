"""Full-range fits of the published forms to one spectrum (steps 1-3 of background_fit).

Two fits per form, with different jobs:

  Poisson  TH1::Fit "L" of the form to the lumi-scaled yields (counts per BIN_WIDTH), every
           bin in range including empty ones; negative bins floored at zero. Gives the
           fitted yields, the tails, the extrapolation and, through 2 x the minimum FCN
           (the Baker-Cousins deviance against the saturated model), the -2 ln(likelihood
           ratio) a dataset of this luminosity would show on average: the expected
           data-level F statistic between nested forms and the expected data-level
           discrimination from the spectrum ("dev").
  Pearson  chi2 with the MC-statistics variance f_i x w_i (w_i the local per-bin event weight
           of the raw MC), minimised by iteratively reweighted least squares: the
           function-selection statistic at MC precision, with the F-test between nested forms.

The forms, their seeds, boxes and nesting come from published_functions.py (the registry).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config as C                                                       # noqa: E402
import published_functions as pf                                         # noqa: E402
import ROOT                                                              # noqa: E402
from spectrum import centers                                             # noqa: E402

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning
ROOT.TH1.AddDirectory(ROOT.kFALSE)
ROOT.Math.MinimizerOptions.SetDefaultMinimizer("Minuit2", "Migrad")
ROOT.Math.MinimizerOptions.SetDefaultMaxFunctionCalls(100000)
_UID = 0


def _uid():
    global _UID
    _UID += 1
    return _UID


def th1(edges, values, errors):
    h = ROOT.TH1D(f"_v3h{_uid()}", "", len(values), np.ascontiguousarray(edges, np.float64))
    for i, (v, e) in enumerate(zip(values, errors), start=1):
        h.SetBinContent(i, float(v))
        h.SetBinError(i, float(e))
    return h


def bin_weights(edges, values, variances, half=250.0):
    """Local per-bin event weight w_i = sum(var) / sum(v) over the bins within +-half GeV;
    inf where no yield is available (the Pearson statistic then skips the bin)."""
    c = centers(edges)
    w = np.full(len(values), np.inf)
    for i, x in enumerate(c):
        m = (c >= x - half) & (c <= x + half) & (values > 0)
        s, s2 = float(values[m].sum()), float(variances[m].sum())
        if s > 0 and s2 > 0:
            w[i] = s2 / s
    return w


def make_tf1(name, m0, lo, hi, seed):
    f = ROOT.TF1(f"_v3f{_uid()}", pf.formula(name, m0, C.SQRTS), lo, hi)
    box = pf.limits(name, lo, hi)
    for i, s in enumerate(seed[: pf.FUNCS[name][0]]):
        if box[i] is not None:
            s = min(max(s, box[i][0] + 1e-9), box[i][1] - 1e-9)
        f.SetParameter(i, float(s))
    for i, b in enumerate(box):
        if b is not None:
            f.SetParLimits(i, b[0], b[1])
    return f, box


def eval_bins(f, c):
    return np.array([f.Eval(x) for x in c])


def checks_of(r, f, box, lo, hi, npar):
    params = np.array([r.Parameter(i) for i in range(npar)])
    at_limit = any(b is not None and (abs(params[i] - b[0]) < 1e-3 * (b[1] - b[0])
                                      or abs(b[1] - params[i]) < 1e-3 * (b[1] - b[0]))
                   for i, b in enumerate(box))
    grid = np.linspace(lo, hi, 400)
    fg = np.array([f.Eval(x) for x in grid])
    mono = bool(np.all(fg[1:] <= fg[:-1] * (1.0 + 1e-6)))
    return {"valid": bool(r.IsValid()), "cov": int(r.CovMatrixStatus()) == 3,
            "limit": not at_limit, "mono": mono}


def seeds_for(name, c, values, variances, m0, parents):
    """Start points: the log-linear seed and the embeddings of fitted parents."""
    pop = (values > 0) & (variances > 0)
    out = []
    s = pf.seed(name, c[pop], values[pop], variances[pop], m0, C.SQRTS)
    if s is not None:
        out.append(s)
    for par in pf.PARENT.get(name, []):
        if par in parents and parents[par] is not None:
            e = pf.embed(name, par, parents[par]["params"], m0)
            if e is not None:
                out.append(e)
    return out


def fit_poisson(name, edges, values, variances, lo, hi, m0, seeds):
    """Poisson fit of `name` over [lo, hi]; the best of the start points. None if impossible."""
    npar = pf.FUNCS[name][0]
    c = centers(edges)
    insel = (c >= lo) & (c <= hi)
    if int(((values > 0) & insel).sum()) <= npar or not seeds:
        return None
    h = th1(edges, np.where(values > 0, values, 0.0), np.sqrt(np.maximum(variances, 0.0)))
    best = None
    for sv in seeds:
        f, box = make_tf1(name, m0, lo, hi, sv)
        r = h.Fit(f, "LRNSQ").Get()
        if not r or not math.isfinite(r.MinFcnValue()):
            continue
        if best is None or r.MinFcnValue() < best[0].MinFcnValue() - 1e-9:
            best = (r, f, box)
    if best is None:
        return None
    r, f, box = best
    params = np.array([r.Parameter(i) for i in range(npar)])
    cov = np.array([[r.CovMatrix(i, j) for j in range(npar)] for i in range(npar)])
    chk = checks_of(r, f, box, lo, hi, npar)
    return {"name": name, "npar": npar, "stat": "poisson", "params": params,
            "perr": np.array([r.ParError(i) for i in range(npar)]), "cov": cov,
            "checks": chk, "passed": all(chk.values()), "dev": 2.0 * float(r.MinFcnValue()),
            "lo": lo, "hi": hi, "m0": m0, "tf1": f, "model": eval_bins(f, c)}


def fit_pearson(name, edges, values, w_arr, lo, hi, m0, seeds, iters=4):
    """Pearson chi2 fit, sum (v - f)^2 / (f w): iteratively reweighted chi2 fits with the
    bin errors sqrt(f w) taken from the previous iteration's model (first from the seed)."""
    npar = pf.FUNCS[name][0]
    c = centers(edges)
    insel = (c >= lo) & (c <= hi)
    usable = insel & np.isfinite(w_arr)
    if int(usable.sum()) <= npar or not seeds:
        return None
    def chi2_of(model):
        sel = usable & (model > 0)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            val = float(np.sum((values[sel] - model[sel]) ** 2 / (model[sel] * w_arr[sel])))
        return (val if math.isfinite(val) else float("inf")), int(sel.sum())

    best = None
    for sv in seeds:
        f, box = make_tf1(name, m0, lo, hi, sv)
        model = eval_bins(f, c)
        # iteratively reweighted: errors from the previous model; keep the best iteration
        # (a grossly wrong form can run away when its model collapses in a bin)
        for _ in range(iters):
            err = np.where(usable & (model > 0),
                           np.sqrt(np.maximum(model, 1e-12) * np.where(np.isfinite(w_arr), w_arr, 1.0)), 0.0)
            h = th1(edges, values, err)                 # error 0 -> the bin is skipped by ROOT
            r = h.Fit(f, "RNSQ").Get()
            if not r:
                break
            model = eval_bins(f, c)
            chi2, nb = chi2_of(model)
            if best is None or chi2 < best[0] - 1e-9:
                best = (chi2, nb, r, f, box, model.copy(),
                        np.array([r.Parameter(i) for i in range(npar)]),
                        np.array([r.ParError(i) for i in range(npar)]))
    if best is None:
        return None
    chi2, nb, r, f, box, model, params_best, perr_best = best
    for i, v in enumerate(params_best):               # put the best iteration back into the TF1
        f.SetParameter(i, float(v))
    chk = checks_of(r, f, box, lo, hi, npar)
    ndf = max(nb - npar, 1)
    return {"name": name, "npar": npar, "stat": "pearson", "params": params_best, "perr": perr_best,
            "checks": chk, "passed": all(chk.values()), "chi2": chi2, "nbins": nb, "ndf": ndf,
            "p": pf.chi2_pvalue(chi2, ndf), "lo": lo, "hi": hi, "m0": m0, "tf1": f, "model": model}


def pearson_at(model, values, w_arr, c, lo, hi, npar):
    sel = (c >= lo) & (c <= hi) & (model > 0) & np.isfinite(w_arr)
    chi2 = float(np.sum((values[sel] - model[sel]) ** 2 / (model[sel] * w_arr[sel])))
    return chi2, max(int(sel.sum()) - npar, 1)


def integral(fit, a, b):
    """Events predicted between a and b (the TF1 is counts per BIN_WIDTH)."""
    return float(fit["tf1"].Integral(a, b)) / C.BIN_WIDTH


def f_test(chi2_p, chi2_c, np_p, np_c, nbins):
    return pf.f_pvalue(chi2_p, chi2_c, np_p, np_c, nbins)


def wilks(dev_p, dev_c, dnp):
    d = max(dev_p - dev_c, 0.0)
    return d, pf.chi2_pvalue(d, dnp)


def fit_all(edges, values, variances, lo, hi, m0, menu=None, w_arr=None):
    """Poisson and Pearson fits of every form of the menu, parents first; returns
    ({name: poisson fit}, {name: pearson fit}, w_arr)."""
    menu = menu or C.MENU
    c = centers(edges)
    if w_arr is None:
        w_arr = bin_weights(edges, values, variances)
    L, P = {}, {}
    order = sorted(menu, key=lambda n: pf.FUNCS[n][0])          # parents (fewer parameters) first
    for name in order:
        seeds = seeds_for(name, c, values, variances, m0, L)
        L[name] = fit_poisson(name, edges, values, variances, lo, hi, m0, seeds)
        seeds_p = seeds_for(name, c, values, variances, m0, P)
        if L[name] is not None:
            seeds_p = seeds_p + [list(L[name]["params"])]
        P[name] = fit_pearson(name, edges, values, w_arr, lo, hi, m0, seeds_p)
    return L, P, w_arr
