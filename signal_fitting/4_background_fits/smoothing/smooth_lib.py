#!/usr/bin/env python3
"""Background-smoothing toolkit (Stage 4 / smoothing).

The MC background histograms carry single-high-weight-event fluctuations in
the high-mass tail (relative MC-stat error 30-50% at 3-3.5 TeV, 70-100% above
4 TeV, negative bins inside the k5 windows for m >= 3400). Those bins are real
MC events but not typical ones; this module builds a smooth per-component
expectation with four techniques so they can be compared and fed to the limit
inputs:

  kernel   local-polynomial Gaussian-kernel regression in u = ln m; the
           bandwidth per target bin is the smallest that reaches an effective
           kernel-weighted count (sum K v)^2 / (sum K^2 var) >= n_target.
           Kernel weights only (no inverse-variance weights: Neyman bias).
           No extrapolation beyond the MC; total yield conserved by rescale.
  hybrid   raw MC below a per-component trust mass m_T (running N_eff over a
           500 GeV window drops below trust_neff), parametric tail above,
           linear blend over [m_T - l_blend, m_T]. Tail yield = the fit.
  param    one full-range parametric fit per component (expo/powexp/expo2,
           lowest chi2/ndf among the fits that pass bkg_fit_lib's checks).
           The per-component twin of the production anchor.
  kde      ROOT RooKeysPdf (adaptive KDE) on a weighted RooDataSet whose
           entries are the populated bins (weight = bin content), in m
           (default since Oct 5 2026; ln m was the original choice), with
           optional mirroring at the 800 GeV threshold. Shape only:
           normalization = the raw total. sumw2 never enters.

Every smoother takes the native 10 GeV (edges, values, variances) of ONE
component and returns (smoothed values on the same grid, diagnostics dict).
Everything below the 800 GeV selection threshold is zero (the SR is empty
there in every component).

Conventions: ROOT for fits/RNG/stats (TF1, TRandom3, TMath); numpy only as
array plumbing. Histogram I/O stays on uproot.
"""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                  # 4_background_fits/smoothing
SIGFIT = HERE.parents[1]                                # signal_fitting
sys.path.insert(0, str(SIGFIT.parent))                  # repo root (wrplotter)
sys.path.insert(0, str(SIGFIT / "4_background_fits"))
sys.path.insert(0, str(SIGFIT / "3_background_plots" / "mc_shapes"))
sys.path.insert(0, str(SIGFIT / "shared"))

import ROOT                                                              # noqa: E402
from bkg_fit_lib import (FUNCS, PAR_LIMITS, TF1_FORMULA, MASS_VAR,       # noqa: E402
                         fit_model, predict)
from plot_mc_backgrounds import load_component, rebin                    # noqa: E402

ROOT.gROOT.SetBatch(True)
ROOT.TH1.AddDirectory(ROOT.kFALSE)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

ERA, BKG_DIR = "RunIISummer20UL18", "20260714_run2_bkgs"
CHANNEL, TOPOLOGY = "ee", "resolved"
REGION = f"wr_{CHANNEL}_{TOPOLOGY}_sr"
VARIABLE = MASS_VAR[TOPOLOGY]
THRESHOLD = 800.0          # SR mass threshold: every component is empty below
NATIVE_BW = 10.0

# component -> sample files (order = plot / table order)
COMPONENTS = {
    "tt_tW": ["tt_tW"],
    "DYJets": ["DYJets"],
    "Nonprompt": ["Nonprompt"],
    "Other": ["Other"],
}
COMPONENTS_GROUPED = {
    "tt_tW": ["tt_tW"],
    "DYJets": ["DYJets"],
    "rest": ["Nonprompt", "Other"],
}
# per-process split (WrCoffea rerun with --split-groups Nonprompt Other):
# Nonprompt -> WJets / TTSemiLep / SingleTop, Other -> Diboson / Triboson / ttV
COMPONENTS_SPLIT = {
    "tt_tW": ["tt_tW"],
    "DYJets": ["DYJets"],
    "WJets": ["WJets"],
    "TTSemiLep": ["TTSemiLep"],
    "SingleTop": ["SingleTop"],
    "Diboson": ["Diboson"],
    "Triboson": ["Triboson"],
    "ttV": ["ttV"],
}
# the six processes inside Nonprompt + Other only (no DY, no tt+tW)
COMPONENTS_NP_OTHER = {k: v for k, v in COMPONENTS_SPLIT.items()
                       if k not in ("tt_tW", "DYJets")}
COMPONENTS_OTHER = {k: COMPONENTS_SPLIT[k] for k in ("Diboson", "Triboson", "ttV")}
COMPONENTS_NONPROMPT = {k: COMPONENTS_SPLIT[k]
                        for k in ("WJets", "TTSemiLep", "SingleTop")}
COMPONENT_SETS = {"legacy4": COMPONENTS, "grouped3": COMPONENTS_GROUPED,
                  "split8": COMPONENTS_SPLIT, "np_other6": COMPONENTS_NP_OTHER,
                  "other3": COMPONENTS_OTHER, "nonprompt3": COMPONENTS_NONPROMPT}
# what the sum of a component set is called on plots
TOTAL_LABEL = {"legacy4": "Total background", "grouped3": "Total background",
               "split8": "Total background", "np_other6": "Nonprompt + Other",
               "other3": "Other (sum)", "nonprompt3": "Nonprompt (sum)"}
COMP_LABEL = {"tt_tW": r"$t\bar{t}$+tW", "DYJets": "DY+jets",
              "Nonprompt": "Nonprompt", "Other": "Other",
              "rest": "Nonprompt+Other", "sum": "Total background",
              "total": "Total background",
              "WJets": "W+jets", "TTSemiLep": r"$t\bar{t}$ semileptonic",
              "SingleTop": "single top", "Diboson": "diboson",
              "Triboson": "triboson", "ttV": r"$t\bar{t}$V"}
# house colors for the four groups (data/sample_groups.yaml); the split
# processes take shades of their parent group's color
COMP_COLOR = {"tt_tW": "#f89c20", "DYJets": "#5790fc",
              "Nonprompt": "#e42536", "Other": "#964a8b", "rest": "#964a8b",
              "sum": "#000000", "total": "#000000",
              "WJets": "#e42536", "TTSemiLep": "#f28e8e", "SingleTop": "#8c1d1d",
              "Diboson": "#964a8b", "Triboson": "#c9a0c4", "ttV": "#5a2d57"}
# line styles for overlaid per-process curves: the split processes share their
# parent's hue, so the second and third of each family are dashed / dotted
COMP_LS = {"TTSemiLep": "--", "SingleTop": ":", "Triboson": "--", "ttV": ":"}
# bottom -> top stacking order (smallest processes at the bottom, DY on top
# like the analysis stack plots)
STACK_ORDER = {"legacy4": ["Other", "Nonprompt", "tt_tW", "DYJets"],
               "grouped3": ["rest", "tt_tW", "DYJets"],
               "split8": ["Triboson", "ttV", "Diboson", "SingleTop", "TTSemiLep",
                          "WJets", "tt_tW", "DYJets"],
               "np_other6": ["Triboson", "ttV", "Diboson", "SingleTop",
                             "TTSemiLep", "WJets"],
               "other3": ["Triboson", "ttV", "Diboson"],
               "nonprompt3": ["SingleTop", "TTSemiLep", "WJets"]}

METHODS = ["kernel", "hybrid", "param", "kde"]
METHOD_COLOR = {"kernel": "#5790fc", "hybrid": "#e42536",
                "param": "#964a8b", "kde": "#f89c20"}
METHOD_LABEL = {"kernel": "kernel (local quadratic, ln m)",
                "hybrid": "hybrid (raw + expo tail)",
                "param": "parametric (full range)",
                "kde": "RooKeysPdf (adaptive KDE)"}


def load_components(input_dirs, *, group_rest=False, region=REGION,
                    variable=VARIABLE, component_set=None):
    """{component: (edges, values, variances)} at the native 10 GeV binning.
    `component_set` is a key of COMPONENT_SETS (overrides group_rest)."""
    if component_set is not None:
        comps = COMPONENT_SETS[component_set]
    else:
        comps = COMPONENTS_GROUPED if group_rest else COMPONENTS
    out = {}
    for name, samples in comps.items():
        hvv = load_component(input_dirs, samples, region, variable)
        if hvv is None:
            raise RuntimeError(f"no histograms for component {name}")
        out[name] = hvv
    return out


def sum_components(comps):
    edges = next(iter(comps.values()))[0]
    v = sum(c[1] for c in comps.values())
    var = sum(c[2] for c in comps.values())
    return edges, v, var


def centers_of(edges):
    return 0.5 * (edges[:-1] + edges[1:])


# ---------------------------------------------------------------------------
# MC-statistics diagnostics
# ---------------------------------------------------------------------------

def neff_per_bin(values, variances):
    """Effective entries per bin, v^2/var (0 for empty / non-positive bins)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.where((variances > 0) & (values > 0),
                     values ** 2 / variances, 0.0)
    return n


def neff_sum(values, variances):
    """Effective entries of a sum of bins, (sum v)^2 / sum var."""
    s, sv = float(np.sum(values)), float(np.sum(variances))
    return (s * s / sv) if sv > 0 and s > 0 else 0.0


def running_neff(edges, values, variances, width, step=50.0, start=THRESHOLD):
    """N_eff of the sliding window [m, m + width) for m = start, start+step, ...
    Returns (m_lo array, N_eff array)."""
    c = centers_of(edges)
    ms = np.arange(start, edges[-1] - width + 1e-9, step)
    out = np.empty(len(ms))
    for i, m in enumerate(ms):
        sel = (c >= m) & (c < m + width)
        out[i] = neff_sum(values[sel], variances[sel])
    return ms, out


def last_populated(edges, values):
    """Centre of the last bin with positive content."""
    c = centers_of(edges)
    pos = np.nonzero(values > 0)[0]
    return float(c[pos[-1]]) if pos.size else float(c[0])


def trust_mass(edges, values, variances, n_thr, width, step=50.0,
               start=THRESHOLD):
    """Lowest window start m (on the `step` grid) where the running N_eff over
    [m, m + width) falls below n_thr; the last populated bin if it never does."""
    ms, ne = running_neff(edges, values, variances, width, step, start)
    below = np.nonzero(ne < n_thr)[0]
    if below.size:
        return float(ms[below[0]])
    return math.floor(last_populated(edges, values) / step) * step


def clip_negative(values):
    """Clip non-finite / negative entries to 0; returns (clipped, n_clipped)."""
    bad = ~np.isfinite(values) | (values < 0)
    return np.where(bad, 0.0, values), int(bad.sum())


def normalize_total(smooth, target, mask):
    """Rescale `smooth[mask]` so its sum equals `target`; returns (values, r)."""
    s = float(np.sum(smooth[mask]))
    if s <= 0 or target <= 0:
        return smooth, float("nan")
    r = target / s
    out = smooth.copy()
    out[mask] *= r
    return out, r


# ---------------------------------------------------------------------------
# (a) kernel: local-polynomial Gaussian-kernel regression in ln m
# ---------------------------------------------------------------------------

def _kernel_neff(K, v, var):
    num = float(np.dot(K, v))
    den = float(np.dot(K * K, var))
    return (num * num / den) if den > 0 and num > 0 else 0.0


def smooth_kernel(edges, values, variances, *, degree=2, h_min=0.05,
                  h_max=0.4, n_target=20.0, threshold=THRESHOLD,
                  n_bisect=25):
    """Local-polynomial kernel regression of the bin contents on u = ln m.

    For every target bin j (centre >= threshold) the bandwidth h_j is the
    smallest value in [h_min, h_max] (ln-m units) whose kernel-weighted
    effective count reaches n_target (bisection; h_max if never reached).
    The estimate is the intercept of the weighted least-squares polynomial of
    degree `degree` in (u_i - u_j). Bins beyond the last MC event stay in the
    regression as zeros, so the estimate decays to zero with the MC (no
    extrapolation). Negative raw bins are kept in the regression; the OUTPUT is
    clipped >= 0 and rescaled so the total above threshold equals the raw
    total (negatives included).
    """
    c = centers_of(edges)
    inside = c >= threshold
    u = np.log(c[inside])
    v, var = values[inside], variances[inside]
    n = len(u)
    s = np.zeros(n)
    h_used = np.zeros(n)
    neff_used = np.zeros(n)
    for j in range(n):
        du = u - u[j]

        def neff_at(h):
            return _kernel_neff(np.exp(-0.5 * (du / h) ** 2), v, var)

        if neff_at(h_min) >= n_target:
            h = h_min
        elif neff_at(h_max) < n_target:
            h = h_max
        else:
            lo, hi = h_min, h_max
            for _ in range(n_bisect):
                mid = 0.5 * (lo + hi)
                if neff_at(mid) >= n_target:
                    hi = mid
                else:
                    lo = mid
            h = hi
        K = np.exp(-0.5 * (du / h) ** 2)
        # weighted LS of v on [1, du, du^2, ...]; intercept = estimate at u_j
        X = np.vander(du, degree + 1, increasing=True)
        sw = np.sqrt(K)
        coef, *_ = np.linalg.lstsq(X * sw[:, None], v * sw, rcond=None)
        s[j] = coef[0]
        h_used[j] = h
        neff_used[j] = neff_at(h)

    out = np.zeros_like(values)
    out[inside] = s
    total_raw = float(np.sum(values[inside]))
    total_pre = float(np.sum(out[inside]))
    out, n_clipped = clip_negative(out)
    out, r = normalize_total(out, total_raw, inside)
    diag = {
        "degree": degree, "h_min": h_min, "h_max": h_max, "n_target": n_target,
        "total_raw": total_raw, "total_pre_rescale": total_pre, "rescale": r,
        "n_clipped": n_clipped,
        "n_at_hmax": int(np.sum(h_used >= h_max * (1 - 1e-9))),
        "h_of_m": {"m": c[inside].tolist(), "h": h_used.tolist(),
                   "neff": neff_used.tolist()},
    }
    return out, diag


# ---------------------------------------------------------------------------
# ROOT fits shared by hybrid and param
# ---------------------------------------------------------------------------

_UID = 0


def _th1(edges, values, variances, name="h"):
    global _UID
    _UID += 1
    h = ROOT.TH1D(f"_smooth_{name}{_UID}", "", len(values),
                  np.ascontiguousarray(edges, np.float64))
    for i, (v, e) in enumerate(zip(values, np.sqrt(np.maximum(variances, 0))),
                               start=1):
        h.SetBinContent(i, float(v))
        h.SetBinError(i, float(e))
    return h


def fit_tf1_poisson(name, edges, values, variances, lo, hi, m0, option="L"):
    """Binned likelihood TF1 fit with bkg_fit_lib's formula / limits for
    function `name`, pivot m0, all bins in range including empty ones.
    option "L"  = Poisson likelihood on the (fractional) contents -- the
                  production anchor's convention (Pearson-like: no Neyman
                  bias; the absolute errors are meaningless for weighted MC);
    option "WL" = ROOT's weighted likelihood (effective counts v^2/var per
                  bin -- the statistically right one for weighted MC).
    Returns a dict or None."""
    global _UID
    _UID += 1
    npar = FUNCS[name][2]
    seed = None
    r0 = fit_model(name, edges, values, variances, lo, hi, m0)   # chi2 seed
    if r0 is not None:
        seed = list(r0.params)
    c = centers_of(edges)
    sel = (c >= lo) & (c <= hi)
    if np.sum(values[sel] > 0) <= npar:
        return None
    h = _th1(edges, np.where(values > 0, values, 0.0), variances, "pois")
    f = ROOT.TF1(f"_pois_f{_UID}", TF1_FORMULA[name].format(m0=f"{m0:.10g}"),
                 lo, hi)
    if seed is None:                       # crude: log of the local mean
        seed = [math.log(max(float(np.mean(values[sel])), 1e-6))] + [0.0] * (npar - 1)
        if name in ("expo", "expo2", "expo3", "powlaw", "powexp"):
            seed[1] = -3.0
    lims = PAR_LIMITS.get(name, [])
    for i, sv in enumerate(seed[:npar]):
        if i < len(lims) and lims[i] is not None:
            sv = min(max(sv, lims[i][0]), lims[i][1])
        f.SetParameter(i, sv)
    for i, lim in enumerate(lims):
        if lim is not None:
            f.SetParLimits(i, lim[0], lim[1])
    rptr = h.Fit(f, option + "RNSQ")
    r = rptr.Get()
    if not r:
        return None
    params = [r.Parameter(i) for i in range(npar)]
    perr = [r.ParError(i) for i in range(npar)]
    return {"name": name, "params": params, "perr": perr,
            "chi2": float(r.Chi2()), "ndf": int(r.Ndf()),
            "ok": bool(r.IsValid()), "stat": {"L": "poisson", "WL": "wpoisson"}[option],
            "range": [lo, hi], "m0": m0}


def fit_tf1_neff(name, edges, values, variances, lo, hi, m0):
    """Scaled-Poisson ("effective count") likelihood fit for weighted MC:
    per bin n_i = v_i / w_i events with expectation f_i / w_i, w_i the
    per-bin weight (bin_weight; the local mean weight where the bin is
    empty). All bins in range enter. Unlike TH1::Fit "L" this gives a bin
    the statistical power it actually has -- a single high-weight event does
    not outvote twenty low-weight ones -- and unlike the chi2 fit it carries
    no Neyman bias. Implemented as TH1::Fit "L" of the effective-count
    histogram with a python TF1 = formula(x) / w(bin of x)."""
    global _UID
    _UID += 1
    npar = FUNCS[name][2]
    c = centers_of(edges)
    w = bin_weight(edges, values, variances)
    n_eff = np.where(values > 0, values / w, 0.0)
    sel = (c >= lo) & (c <= hi)
    if np.sum(values[sel] > 0) <= npar:
        return None
    r0 = fit_model(name, edges, values, variances, lo, hi, m0)
    seed = list(r0.params) if r0 is not None else None
    if seed is None:
        seed = [math.log(max(float(np.mean(values[sel])), 1e-6))] + [0.0] * (npar - 1)
        if name != "dexp":
            seed[1] = -3.0
    shape = ROOT.TF1(f"_neff_shape{_UID}",
                     TF1_FORMULA[name].format(m0=f"{m0:.10g}"), lo, hi)
    e0, bw = float(edges[0]), float(edges[1] - edges[0])
    w_arr = np.ascontiguousarray(w, np.float64)
    nb = len(w_arr)

    def model(x, p):
        for i in range(npar):
            shape.SetParameter(i, p[i])
        j = min(max(int((x[0] - e0) / bw), 0), nb - 1)
        return shape.Eval(x[0]) / w_arr[j]

    f = ROOT.TF1(f"_neff_f{_UID}", model, lo, hi, npar)
    lims = PAR_LIMITS.get(name, [])
    for i, sv in enumerate(seed[:npar]):
        if i < len(lims) and lims[i] is not None:
            sv = min(max(sv, lims[i][0]), lims[i][1])
        f.SetParameter(i, sv)
    for i, lim in enumerate(lims):
        if lim is not None:
            f.SetParLimits(i, lim[0], lim[1])
    h = _th1(edges, n_eff, np.where(n_eff > 0, n_eff, 0.0), "neff")
    rptr = h.Fit(f, "LRNSQ")
    r = rptr.Get()
    if not r:
        return None
    params = [r.Parameter(i) for i in range(npar)]
    perr = [r.ParError(i) for i in range(npar)]
    return {"name": name, "params": params, "perr": perr,
            "chi2": float(r.Chi2()), "ndf": int(r.Ndf()),
            "ok": bool(r.IsValid()), "stat": "neff", "range": [lo, hi],
            "m0": m0}


def fit_component(name, edges, values, variances, lo, hi, m0, stat="chi2"):
    """Uniform wrapper: chi2 (bkg_fit_lib.fit_model, populated bins),
    poisson (TH1::Fit "L", all bins), wpoisson ("WL": same central values,
    corrected covariance) or neff (scaled-Poisson effective-count
    likelihood, all bins). Returns a dict with params / chi2 / ndf / ok /
    passed / checks, or None."""
    if stat == "neff":
        r = fit_tf1_neff(name, edges, values, variances, lo, hi, m0)
        if r is not None:
            r["passed"] = r["ok"]
            r["checks"] = {"valid_minimum": r["ok"]}
        return r
    if stat in ("poisson", "wpoisson"):
        r = fit_tf1_poisson(name, edges, values, variances, lo, hi, m0,
                            option={"poisson": "L", "wpoisson": "WL"}[stat])
        if r is not None:
            r["passed"] = r["ok"]
            r["checks"] = {"valid_minimum": r["ok"]}
        return r
    r = fit_model(name, edges, values, variances, lo, hi, m0)
    if r is None:
        return None
    return {"name": name, "params": list(map(float, r.params)),
            "perr": list(map(float, r.perr)), "chi2": r.chi2, "ndf": r.ndf,
            "ok": bool(r.checks["valid_minimum"]), "passed": bool(r.passed),
            "checks": {k: bool(x) for k, x in r.checks.items()},
            "stat": "chi2", "range": [lo, hi], "m0": m0}


def eval_fit(fit, m, bw_fit, bw_out):
    """Evaluate a fit done on `bw_fit`-wide bins as content per `bw_out`-wide
    bin at masses m (smooth functions: content scales with the bin width)."""
    return predict(fit["name"], np.array(fit["params"]), np.asarray(m, float),
                   fit["m0"]) * (bw_out / bw_fit)


def local_weight(edges, values, variances, half=250.0):
    """Mean MC event weight per bin, sum(var)/sum(v) over +-half GeV (the
    Pearson-error scale: var_i ~ w_i * mu_i). NaN where nothing populated."""
    c = centers_of(edges)
    out = np.full(len(c), np.nan)
    for i in range(len(c)):
        sel = (c >= c[i] - half) & (c <= c[i] + half) & (values > 0)
        sv, ss = float(np.sum(variances[sel])), float(np.sum(values[sel]))
        if ss > 0 and sv > 0:
            out[i] = sv / ss
    return out


def bin_weight(edges, values, variances, half=250.0):
    """Per-bin mean MC event weight w_i = var_i / v_i where the bin is
    populated, the local mean weight (+-half GeV) where it is empty or
    negative, the global mean weight where even that is undefined. This is
    the scale of the bin's statistical power: n_i = v_i / w_i effective
    events, and the Pearson variance of a smooth expectation f_i is f_i w_i."""
    w_loc = local_weight(edges, values, variances, half)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where((values > 0) & (variances > 0), variances / values, w_loc)
    fin = np.isfinite(w) & (w > 0)
    if not fin.any():
        return np.ones_like(values)
    glob = float(np.sum(variances[values > 0]) / np.sum(values[values > 0])) \
        if np.any(values > 0) else 1.0
    return np.where(fin, w, glob)


def pearson_chi2(edges, values, variances, model, lo, hi):
    """Pearson chi2 of `model` vs the raw bins in [lo, hi]: sum (v - f)^2 /
    (f w_i), w_i the per-bin weight (bin_weight). Returns (chi2, nbins).
    Bins with f <= 0 are skipped."""
    c = centers_of(edges)
    w = bin_weight(edges, values, variances)
    sel = (c >= lo) & (c <= hi) & (model > 0) & np.isfinite(w)
    chi2 = float(np.sum((values[sel] - model[sel]) ** 2 / (model[sel] * w[sel])))
    return chi2, int(sel.sum())


# ---------------------------------------------------------------------------
# (b) hybrid: raw below the trust mass, parametric tail above
# ---------------------------------------------------------------------------

def smooth_hybrid(edges, values, variances, *, trust_neff=20.0,
                  trust_width=500.0, l_below=1000.0, l_above=500.0,
                  l_blend=200.0, family="expo", stat="poisson", fit_bw=50.0,
                  conserve_total=False, threshold=THRESHOLD):
    """Raw MC below m_T, the tail fit above, blended over [m_T - l_blend, m_T].

    m_T = first 50 GeV grid point where the running N_eff over the next
    `trust_width` GeV drops below `trust_neff`. The tail is fit on the
    `fit_bw`-rebinned component over [m_T - l_below, m_T + l_above] (chi2,
    populated bins) or [m_T - l_below, last populated] (poisson, all bins),
    pivot m_T; a 3-parameter family that fails falls back to expo. Default
    stat = poisson: the chi2 fit with observed (Neyman) errors is biased low
    on this weighted MC wherever N_eff per bin is small.
    """
    c = centers_of(edges)
    f = int(round(fit_bw / NATIVE_BW))
    e50, v50, var50 = rebin(edges, values, variances, f)
    m_T = trust_mass(e50, v50, var50, trust_neff, trust_width, step=fit_bw)
    m_last = last_populated(edges, values)
    lo = max(m_T - l_below, threshold)
    hi = (m_T + l_above) if stat == "chi2" else max(m_last, m_T + l_above)
    fit = fit_component(family, e50, v50, var50, lo, hi, m_T, stat)
    fallback = None
    if (fit is None or not fit["passed"]) and family != "expo":
        fallback = family
        fit = fit_component("expo", e50, v50, var50, lo, hi, m_T, stat)
    if fit is None:
        raise RuntimeError(f"hybrid: no tail fit possible (m_T={m_T:.0f})")

    tail = eval_fit(fit, c, fit_bw, NATIVE_BW)
    t = np.clip((c - (m_T - l_blend)) / l_blend, 0.0, 1.0)     # 0 raw .. 1 fit
    out = (1.0 - t) * values + t * tail
    out[c < threshold] = 0.0
    inside = c >= threshold
    total_raw = float(np.sum(values[inside]))
    total_pre = float(np.sum(out[inside]))
    out, n_clipped = clip_negative(out)
    r = float("nan")
    if conserve_total:
        out, r = normalize_total(out, total_raw, inside)
    # seam diagnostic: raw vs fit over the blend band, in raw sigma
    band = (c >= m_T - l_blend) & (c <= m_T)
    sig = math.sqrt(max(float(np.sum(variances[band])), 0.0))
    seam_pull = ((float(np.sum(values[band])) - float(np.sum(tail[band]))) / sig
                 if sig > 0 else float("nan"))
    above = c >= m_T
    sig_a = math.sqrt(max(float(np.sum(variances[above])), 0.0))
    tail_raw, tail_fit = float(np.sum(values[above])), float(np.sum(tail[above]))
    diag = {
        "trust_neff": trust_neff, "trust_width": trust_width,
        "l_below": l_below, "l_above": l_above, "l_blend": l_blend,
        "family": fit["name"], "family_requested": family,
        "fallback_from": fallback, "stat": stat, "m_T": m_T, "m_last": m_last,
        "fit": fit, "seam_pull": seam_pull,
        "tail_raw": tail_raw, "tail_fit": tail_fit, "tail_raw_err": sig_a,
        "tail_pull": ((tail_raw - tail_fit) / sig_a) if sig_a > 0 else float("nan"),
        "total_raw": total_raw, "total_pre_rescale": total_pre, "rescale": r,
        "n_clipped": n_clipped,
    }
    return out, diag


# ---------------------------------------------------------------------------
# (c) param: one full-range parametric fit per component
# ---------------------------------------------------------------------------

def smooth_param(edges, values, variances, *, families=("expo", "powexp", "expo2"),
                 lo=THRESHOLD, hi=None, pivot=2000.0, stat="poisson", fit_bw=50.0,
                 threshold=THRESHOLD):
    """Fit each family over [lo, hi] (hi default: last populated bin) on the
    `fit_bw`-rebinned component; keep the lowest Pearson chi2/ndf among the
    fits that pass (all fits if none pass, flagged). Output = f at the 10 GeV
    centres. Default stat = poisson (see smooth_hybrid)."""
    c = centers_of(edges)
    f = int(round(fit_bw / NATIVE_BW))
    e50, v50, var50 = rebin(edges, values, variances, f)
    m_last = last_populated(edges, values)
    hi = m_last if hi is None else hi
    c50 = centers_of(e50)
    fits = {}
    for fam in families:
        r = fit_component(fam, e50, v50, var50, lo, hi, pivot, stat)
        if r is None:
            continue
        chi2p, nb = pearson_chi2(e50, v50, var50, eval_fit(r, c50, fit_bw, fit_bw),
                                 lo, hi)
        r["chi2_pearson"], r["ndf_pearson"] = chi2p, max(nb - FUNCS[fam][2], 1)
        fits[fam] = r
    if not fits:
        raise RuntimeError("param: no family could be fit")
    passed = {k: r for k, r in fits.items() if r["passed"]}
    pool = passed if passed else fits
    best = min(pool, key=lambda k: pool[k]["chi2_pearson"] / pool[k]["ndf_pearson"])
    fit = fits[best]
    out = eval_fit(fit, c, fit_bw, NATIVE_BW)
    out[c < threshold] = 0.0
    inside = c >= threshold
    total_raw = float(np.sum(values[inside]))
    total_pre = float(np.sum(out[inside]))
    out, n_clipped = clip_negative(out)
    # turn-on region quality: Pearson chi2 over [threshold, 1000]
    chi2_turnon, n_turnon = pearson_chi2(
        e50, v50, var50, eval_fit(fit, c50, fit_bw, fit_bw), threshold, 1000.0)
    ton = np.zeros(n_turnon)
    diag = {
        "families": list(families), "lo": lo, "hi": hi, "pivot": pivot,
        "stat": stat, "best": best, "any_passed": bool(passed),
        "fits": fits, "fit": fit,
        "chi2_turnon": chi2_turnon, "n_turnon": int(ton.sum()),
        "total_raw": total_raw, "total_pre_rescale": total_pre,
        "rescale": float("nan"), "n_clipped": n_clipped,
        "total_ratio": (total_pre / total_raw) if total_raw > 0 else float("nan"),
    }
    return out, diag


# ---------------------------------------------------------------------------
# (d) kde: RooKeysPdf on a weighted RooDataSet
# ---------------------------------------------------------------------------

_MIRROR = {"NoMirror": ROOT.RooKeysPdf.NoMirror,
           "MirrorLeft": ROOT.RooKeysPdf.MirrorLeft,
           "MirrorRight": ROOT.RooKeysPdf.MirrorRight,
           "MirrorBoth": ROOT.RooKeysPdf.MirrorBoth,
           "MirrorAsymLeft": ROOT.RooKeysPdf.MirrorAsymLeft,
           "MirrorAsymBoth": ROOT.RooKeysPdf.MirrorAsymBoth}


def smooth_kde(edges, values, variances, *, var_kind="m", mirror="MirrorLeft",
               rho=1.0, resample=1, threshold=THRESHOLD, xmax=None):
    """Adaptive KDE of the populated bins: entries at the bin centres (or
    `resample` equally spaced sub-positions per bin) with weight = content,
    in x = ln m (`logm`) or x = m. Negative bins are dropped from the dataset
    but kept in the normalization target. Output s_j = N_raw pdf(x_j) dx_j."""
    global _UID
    _UID += 1
    c = centers_of(edges)
    xmax = float(edges[-1]) if xmax is None else xmax
    tr = (lambda m: np.log(m)) if var_kind == "logm" else (lambda m: m)
    x = ROOT.RooRealVar(f"x{_UID}", "x", tr(threshold), tr(xmax))
    w = ROOT.RooRealVar(f"w{_UID}", "w", 0.0, 1e9)
    ds = ROOT.RooDataSet(f"ds{_UID}", "", ROOT.RooArgSet(x, w),
                         ROOT.RooFit.WeightVar(w.GetName()))
    inside = c >= threshold
    n_entries = 0
    for i in np.nonzero(inside & (values > 0))[0]:
        lo_e, hi_e = float(edges[i]), float(edges[i + 1])
        subs = (np.array([c[i]]) if resample <= 1
                else lo_e + (np.arange(resample) + 0.5) * (hi_e - lo_e) / resample)
        for m in subs:
            x.setVal(float(tr(m)))
            ds.add(ROOT.RooArgSet(x), float(values[i]) / len(subs))
            n_entries += 1
    pdf = ROOT.RooKeysPdf(f"kde{_UID}", "", x, ds, _MIRROR[mirror], rho)
    nset = ROOT.RooArgSet(x)
    out = np.zeros_like(values)
    for j in np.nonzero(inside)[0]:
        x.setVal(float(tr(c[j])))
        dx = float(tr(edges[j + 1]) - tr(edges[j]))
        out[j] = pdf.getVal(nset) * dx
    total_raw = float(np.sum(values[inside]))
    n_dropped = int(np.sum(inside & (values < 0)))
    out, n_clipped = clip_negative(out)
    total_pre = float(np.sum(out[inside]))
    out, r = normalize_total(out, total_raw, inside)
    diag = {
        "var_kind": var_kind, "mirror": mirror, "rho": rho, "resample": resample,
        "n_entries": n_entries, "n_negative_dropped": n_dropped,
        "sum_weights": float(ds.sumEntries()),
        "total_raw": total_raw, "total_pre_rescale": total_pre, "rescale": r,
        "n_clipped": n_clipped,
    }
    return out, diag


def load_component_events(input_dirs, samples, region=REGION):
    """(mass, weight) of the selected MC events of `samples` in `region`, from
    the `events_<region>` TTrees written by run_analysis.py --save-events."""
    import uproot
    ms, ws = [], []
    for d in input_dirs:
        for smp in samples:
            fpath = d / f"WRAnalyzer_{smp}.root"
            if not fpath.exists():
                logger.warning("Missing file: %s", fpath)
                continue
            f = uproot.open(fpath)
            key = f"events_{region}"
            if key not in f:
                raise KeyError(f"{fpath.name} has no {key} tree (run with --save-events)")
            a = f[key].arrays(library="np")
            ms.append(np.asarray(a["mass"], float))
            ws.append(np.asarray(a["weight"], float))
    if not ms:
        raise RuntimeError(f"no event lists for {samples} in {region}")
    return np.concatenate(ms), np.concatenate(ws)


def load_components_events(input_dirs, component_set, region=REGION):
    """{component: (mass, weight)} for a COMPONENT_SETS key."""
    return {name: load_component_events(input_dirs, samples, region)
            for name, samples in COMPONENT_SETS[component_set].items()}


def smooth_kde_events(edges, mass, weight, *, var_kind="m", mirror="MirrorLeft",
                      rho=1.0, threshold=THRESHOLD, xmax=None):
    """RooKeysPdf with ONE ENTRY PER MC EVENT (the textbook construction):
    entries at the true masses, weight = the event weight. Events outside
    [threshold, xmax) are ignored; negative-weight events are dropped from the
    dataset but kept in the normalization target. Returns the curve as
    content per bin of `edges` (same grid as the histograms) and a diag dict."""
    global _UID
    _UID += 1
    c = centers_of(edges)
    xmax = float(edges[-1]) if xmax is None else xmax
    tr = (lambda m: np.log(m)) if var_kind == "logm" else (lambda m: m)
    inrange = (mass >= threshold) & (mass < xmax)
    use = inrange & (weight > 0)
    x = ROOT.RooRealVar(f"x{_UID}", "x", tr(threshold), tr(xmax))
    w = ROOT.RooRealVar(f"w{_UID}", "w", 0.0, 1e9)
    ds = ROOT.RooDataSet(f"ds{_UID}", "", ROOT.RooArgSet(x, w),
                         ROOT.RooFit.WeightVar(w.GetName()))
    for m, wt in zip(mass[use], weight[use]):
        x.setVal(float(tr(m)))
        ds.add(ROOT.RooArgSet(x), float(wt))
    pdf = ROOT.RooKeysPdf(f"kde{_UID}", "", x, ds, _MIRROR[mirror], rho)
    nset = ROOT.RooArgSet(x)
    inside = c >= threshold
    out = np.zeros(len(c))
    for j in np.nonzero(inside)[0]:
        x.setVal(float(tr(c[j])))
        out[j] = pdf.getVal(nset) * float(tr(edges[j + 1]) - tr(edges[j]))
    total_raw = float(weight[inrange].sum())
    out, n_clipped = clip_negative(out)
    total_pre = float(np.sum(out[inside]))
    out, r = normalize_total(out, total_raw, inside)
    diag = {
        "entries": "events", "var_kind": var_kind, "mirror": mirror, "rho": rho,
        "n_entries": int(use.sum()), "n_negative_dropped": int((inrange & (weight < 0)).sum()),
        "n_events_in_range": int(inrange.sum()),
        "sum_weights": float(weight[use].sum()),
        "total_raw": total_raw, "total_pre_rescale": total_pre, "rescale": r,
        "n_clipped": n_clipped,
    }
    return out, diag


SMOOTHERS = {"kernel": smooth_kernel, "hybrid": smooth_hybrid,
             "param": smooth_param, "kde": smooth_kde}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_smoothed_root(path, edges, smooth, raw):
    """smoothed_<method>.root: one TH1D per component (content = smoothed,
    error = sqrt(raw sumw2)), `bkg_smooth` / `bkg_raw` (sums, with errors) and
    `bkg_native` (= bkg_smooth clipped >= 0, no errors -- the drop-in twin of
    the optimization-inputs histogram)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fout = ROOT.TFile(str(path), "RECREATE")
    e = np.ascontiguousarray(edges, np.float64)
    n = len(e) - 1

    def th1(name, v, var=None):
        h = ROOT.TH1D(name, ";m [GeV];events", n, e)
        for i in range(n):
            h.SetBinContent(i + 1, float(v[i]))
            if var is not None:
                h.SetBinError(i + 1, float(math.sqrt(max(var[i], 0.0))))
        h.Write()

    s_sum = np.zeros(n)
    v_sum = np.zeros(n)
    var_sum = np.zeros(n)
    for name, s in smooth.items():
        _, v, var = raw[name]
        th1(name, s, var)
        th1(f"{name}_raw", v, var)
        s_sum += s
        v_sum += v
        var_sum += var
    th1("bkg_smooth", s_sum, var_sum)
    th1("bkg_raw", v_sum, var_sum)
    native, _ = clip_negative(s_sum)
    th1("bkg_native", native)
    fout.Close()
    return path


def jsonable(obj):
    """Recursively convert numpy / non-finite values for json.dump."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
