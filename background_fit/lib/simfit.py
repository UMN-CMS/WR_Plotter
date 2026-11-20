"""The simultaneous SR + e-mu flavour-CR fit of background_fit (tried and dropped; kept for the record).

  SR(m) = TF x f_tt(m; theta) + f_rest(m; phi) + N_sig x S(m)
  CR(m) = f_tt(m; theta)

f_tt is one form of the registry fitted to the whole e-mu CR (93 % tt+tW in 2024) and shared
with the SR, where it stands for everything that is not DY (tt+tW, nonprompt, other) scaled by
the transfer factor TF = SR non-DY / CR, either fixed to the truth value or free; f_rest is a
second form for the DY. The two regions are fitted together as ONE histogram: the SR bins of
[lo, HI] followed by the CR bins shifted by HI - lo, with one TF1 whose formula switches at
x = HI. Poisson likelihood (TH1::Fit "L"), every parameter free except TF when fixed. This is
a two-component background whose tt-like part is measured in the CR; it runs over the full
range with any pair of forms.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config as C                                                       # noqa: E402
import published_functions as pf                                         # noqa: E402
import ROOT                                                              # noqa: E402
import uproot                                                            # noqa: E402
from fits import _uid, th1                                               # noqa: E402
from spectrum import centers, input_dir                                  # noqa: E402

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning
ROOT.TH1.AddDirectory(ROOT.kFALSE)
HI = 8000.0
CR_REGION = "wr_resolved_flavor_cr"
CR_DATA_STREAM = "Muon"            # the stream that fills the resolved flavour CR (both hold it; use one)
NONDY = ["tt_tW", "Nonprompt", "Other"]


def cr_key():
    return f"{CR_REGION}/{C.VARIABLE}_{CR_REGION}"


def load_cr(variant="lo_ht"):
    """{component: (edges, values, variances)} of the e-mu CR at the native binning, plus 'sum'."""
    d = input_dir()
    files = {"DYJets": C.DY_VARIANTS[variant], **{g: f"WRAnalyzer_{g}.root" for g in C.GROUPS}}
    out = {}
    for comp, fname in files.items():
        h = uproot.open(d / fname)[cr_key()]
        out[comp] = (h.axes[0].edges(), h.values().astype(float), h.variances().astype(float))
    e = out["DYJets"][0]
    out["sum"] = (e, sum(out[c][1] for c in files), sum(out[c][2] for c in files))
    return out


def load_cr_data(stream=CR_DATA_STREAM):
    h = uproot.open(input_dir() / f"WRAnalyzer_{stream}.root")[cr_key()]
    return h.axes[0].edges(), h.values().astype(float), h.variances().astype(float)


def shift_formula(formula, par_offset, x_expr):
    """Renumber the [k] parameters by par_offset and replace the variable x by x_expr."""
    f = re.sub(r"\[(\d+)\]", lambda m: f"[{int(m.group(1)) + par_offset}]", formula)
    return re.sub(r"\bx\b", f"({x_expr})", f)


def segment(edges, lo, hi=HI):
    """Slice of the bins whose centres lie in (lo, hi), and their edges."""
    c = centers(edges)
    idx = np.nonzero((c > lo) & (c < hi))[0]
    sl = slice(int(idx[0]), int(idx[-1]) + 1)
    return sl, edges[sl.start:sl.stop + 1]


def concat_edges(edges, lo):
    sl, e_seg = segment(edges, lo)
    return sl, np.concatenate([e_seg, e_seg[1:] + (HI - lo)])


def concat_values(sr, cr, sl):
    return np.concatenate([sr[sl], cr[sl]])


class SimModel:
    """The two-region TF1 of one configuration at one signal mass, built once and reused."""

    def __init__(self, tt_name, dy_name, tf_mode, mu, sigma, lo):
        self.tt, self.dy, self.tf_mode, self.lo = tt_name, dy_name, tf_mode, lo
        self.ntt, self.ndy = pf.FUNCS[tt_name][0], pf.FUNCS[dy_name][0]
        self.itf = self.ntt + self.ndy
        self.isig = self.itf + 1
        self.npar = self.isig + 1
        self.shift = HI - lo
        ftt = pf.formula(tt_name, C.PIVOT, C.SQRTS)
        fdy = shift_formula(pf.formula(dy_name, C.PIVOT, C.SQRTS), self.ntt, "x")
        # each segment's variable is clamped to its own mass range: the other segment's formula is
        # multiplied by zero but still evaluated, and a fractional power or a log of a negative
        # mass would turn that zero into NaN
        x_sr = f"TMath::Min(x,{HI:.1f})"                       # doubles on both sides: the templated
        x_cr = f"TMath::Max(x-{self.shift:.1f},{lo:.1f})"       # TMath::Min/Max cannot take an int literal
        ftt_sr = shift_formula(ftt, 0, x_sr)
        fdy_sr = shift_formula(fdy, 0, x_sr)
        ftt_cr = shift_formula(ftt, 0, x_cr)
        w2, s2 = C.BIN_WIDTH / 2.0, sigma * math.sqrt(2.0)
        sig = (f"[{self.isig}]*0.5*(TMath::Erf(({x_sr}+{w2:.10g}-{mu:.10g})/{s2:.10g})"
               f"-TMath::Erf(({x_sr}-{w2:.10g}-{mu:.10g})/{s2:.10g}))")
        sr_bkg = f"[{self.itf}]*({ftt_sr})+({fdy_sr})"
        formula = f"(x<{HI:.10g})*(({sr_bkg})+({sig}))+(x>={HI:.10g})*({ftt_cr})"
        self.f = ROOT.TF1(f"_v3sim{_uid()}", formula, lo, 2 * HI - lo)
        if not self.f.IsValid():
            raise RuntimeError(f"invalid two-region formula for {tt_name}+{dy_name}: {formula[:200]}")
        self.f_sr = ROOT.TF1(f"_v3simsr{_uid()}", f"[{self.itf}]*({ftt})+({fdy})", lo, HI)   # SR background
        self.f_cr = ROOT.TF1(f"_v3simcr{_uid()}", ftt, lo, HI)                              # CR shape
        self.box = list(pf.limits(tt_name, lo, HI)) + list(pf.limits(dy_name, lo, HI))
        self.grid = np.linspace(lo, HI, 120)

    def fit(self, e_cat, data_cat, tt_seed, dy_seed, tf_seed, nsig_seed, nsig_bounds):
        f = self.f
        seeds = list(tt_seed)[: self.ntt] + list(dy_seed)[: self.ndy]
        for i, sv in enumerate(seeds):
            b = self.box[i]
            if b is not None:
                sv = min(max(sv, b[0] + 1e-9), b[1] - 1e-9)
                f.SetParLimits(i, b[0], b[1])
            f.SetParameter(i, float(sv))
        tf_lo, tf_hi = 0.2 * tf_seed, 5.0 * tf_seed
        if self.tf_mode == "fix":
            f.FixParameter(self.itf, float(tf_seed))
        else:
            f.ReleaseParameter(self.itf)
            f.SetParameter(self.itf, float(tf_seed))
            f.SetParLimits(self.itf, tf_lo, tf_hi)
        f.SetParameter(self.isig, float(min(max(nsig_seed, nsig_bounds[0]), nsig_bounds[1])))
        f.SetParLimits(self.isig, nsig_bounds[0], nsig_bounds[1])
        h = th1(e_cat, np.where(data_cat > 0, data_cat, 0.0), np.sqrt(np.maximum(data_cat, 0.0)))
        r = h.Fit(f, "LRNSQ").Get()
        if not r or not math.isfinite(r.MinFcnValue()):
            return None
        params = np.array([r.Parameter(i) for i in range(self.npar)])
        perr = np.array([r.ParError(i) for i in range(self.npar)])
        at_limit = any(b is not None and (abs(params[i] - b[0]) < 1e-3 * (b[1] - b[0])
                                          or abs(b[1] - params[i]) < 1e-3 * (b[1] - b[0]))
                       for i, b in enumerate(self.box))
        span = nsig_bounds[1] - nsig_bounds[0]
        nsig_at_bound = (abs(params[self.isig] - nsig_bounds[0]) < 1e-3 * span
                         or abs(nsig_bounds[1] - params[self.isig]) < 1e-3 * span)
        tf_at_bound = self.tf_mode == "free" and (params[self.itf] < tf_lo * 1.001 or params[self.itf] > tf_hi * 0.999)
        for i in range(self.itf + 1):
            self.f_sr.SetParameter(i, float(params[i]))
        for i in range(self.ntt):
            self.f_cr.SetParameter(i, float(params[i]))
        g_sr = np.array([self.f_sr.Eval(x) for x in self.grid])
        g_cr = np.array([self.f_cr.Eval(x) for x in self.grid])
        mono = bool(np.all(g_sr[1:] <= g_sr[:-1] * (1.0 + 1e-6)) and np.all(g_cr[1:] <= g_cr[:-1] * (1.0 + 1e-6)))
        checks = {"valid": bool(r.IsValid()), "cov": int(r.CovMatrixStatus()) == 3,
                  "limit": not at_limit and not nsig_at_bound and not tf_at_bound, "mono": mono}
        return {"nsig": float(params[self.isig]), "nsig_err": float(perr[self.isig]),
                "tf": float(params[self.itf]), "tf_err": float(perr[self.itf]),
                "tt": params[: self.ntt], "dyp": params[self.ntt: self.itf],
                "checks": checks, "passed": all(checks.values()), "dev": 2.0 * float(r.MinFcnValue())}
