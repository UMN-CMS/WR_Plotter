#!/usr/bin/env python3
"""Fit candidate parameterizations of the on-shell width vs (x, M_WR).

For each width definition (σ_gauss^on, σ_FWHM^on), channel and topology, fit a
suite of models of y = σ/M_WR and record their fitted parameters + overlay plots.
This script ONLY fits — model scoring / cross-validation lives in
`validate_models.py`, which imports the model registry from here.

Models (x = M_N/M_WR, m = (M_WR-4000)/4000):
  pol2 / pol3 / pol4 : polynomials in x                      (ROOT TF1, weighted)
  physics            : p0+p1 x+p2 x^2+p3/(1-x+p4)            (resolved only)
  spline             : TSpline3 through x-binned medians     (1D)
  poly3+mass         : pol3 + b1 m + b2 x m                  (2D, lstsq)
  fxgx               : pol4 + m(b0+b1 x+b2 x^2+b3 x^3)       (2D, lstsq)
  +m2 / +x2m         : poly3+mass + b3 m^2 / + b3 x^2 m      (2D, lstsq)
  spline2d           : per-M_WR spline in x, interp. in M_WR (2D)

Reads the Stage-1 width tables from 1_signal_widths/{gaussian,fwhm}/. Writes
beside itself (this script lives in 2_width_parameterization/parameterizations/):
  parameterization_params.json   fitted parameters per (width, channel, cat, model)
  <1d|2d>/<model>/<width>/{ch}_{cat}.{png,pdf}   data + that model's fit, grouped
                                 by 1D-in-x vs 2D-in-(x,M_WR), one folder per model,
                                 gauss/ & fwhm/ inside each
  all_models/<width>/...         every model overlaid (2D models at m=0)

(The σ_gauss vs σ_FWHM comparison lives in 1_signal_widths/gauss_vs_fwhm/.)

Setup:
    source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import ROOT
except ImportError:
    sys.exit("ERROR: PyROOT unavailable. Source LCG_106 first.")
ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPalette(ROOT.kViridis)

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from wrplotter.paths import repo_root  # noqa: E402,F401

logger = logging.getLogger(__name__)

BASE = "[0.8,1.2]"
CHANNELS = ["ee", "mumu"]
CATEGORIES = ["resolved", "boosted"]
GAUSS = r"\sigma_{\rm gauss}^{\rm on}"
FWHM = r"\sigma_{\rm FWHM}^{\rm on}"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _pick(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def load_width_table(path: Path, width: str):
    """Load a width table, auto-detect columns, validate, drop/flag bad rows.

    Returns samples[(channel, category)] = dict(x, y, yerr|None, mWR, m) and a
    per-sample QA report.
    """
    df = pd.read_csv(path)
    cols = set(df.columns)
    mwr_c = _pick(cols, ["mWR", "M_WR", "mwr"])
    mn_c = _pick(cols, ["mN", "M_N", "mn"])
    ch_c = _pick(cols, ["channel"])
    cat_c = _pick(cols, ["category", "topology"])
    if width == "gauss":
        sig_c = _pick(cols, ["sigma_gauss_on", "sigma_gaus", "sigma_gauss"])
        err_c = _pick(cols, ["sigma_err"])
    else:
        sig_c = _pick(cols, ["sigma_fwhm_on", "sigma_fwhm"])
        err_c = None
    for need, name in [(mwr_c, "mWR"), (mn_c, "mN"), (ch_c, "channel"),
                       (cat_c, "category"), (sig_c, "sigma")]:
        if need is None:
            raise ValueError(f"{path.name}: missing required column for {name}")
    fr_c = _pick(cols, ["fit_range"])
    if fr_c:
        df = df[df[fr_c] == BASE]
    logger.info("[%s] detected: mWR=%s mN=%s sigma=%s err=%s status=%s/%s",
                width, mwr_c, mn_c, sig_c, err_c,
                "fit_status" in cols, "cov_status" in cols)

    samples, report = {}, {}
    for ch in CHANNELS:
        for cat in CATEGORIES:
            sub = df[(df[ch_c] == ch) & (df[cat_c] == cat)].copy()
            n0 = len(sub)
            reasons = {"bad_fit": 0, "nonpos": 0, "missing": 0}
            mwr = sub[mwr_c].to_numpy(float)
            mn = sub[mn_c].to_numpy(float)
            sig = sub[sig_c].to_numpy(float)
            y = sig / mwr
            x = mn / mwr
            good = np.isfinite(y) & np.isfinite(x)
            reasons["missing"] = int((~good).sum())
            pos = sig > 0
            reasons["nonpos"] = int((good & ~pos).sum())
            good &= pos
            if {"fit_status", "cov_status"} <= cols:
                ok = (sub["fit_status"].to_numpy() == 0) & \
                     (sub["cov_status"].to_numpy() == 3)
                reasons["bad_fit"] = int((good & ~ok).sum())
                good &= ok
            yerr = None
            if err_c:
                yerr = (sub[err_c].to_numpy(float) / mwr)
            samples[(ch, cat)] = {
                "x": x[good], "y": y[good], "mWR": mwr[good],
                "m": (mwr[good] - 4000.0) / 4000.0,
                "yerr": (yerr[good] if yerr is not None else None),
            }
            report[(ch, cat)] = {"n_total": n0, "n_kept": int(good.sum()),
                                 **reasons}
            logger.info("  [%s/%s] kept %d/%d (drop: %s)", ch, cat,
                        int(good.sum()), n0, reasons)
    return samples, report


# ---------------------------------------------------------------------------
# Model registry
#   Each fit_model() call returns a dict fit-state; predict_model() consumes it.
#   1D models ignore mwr in predict; 2D models use m = (mwr-4000)/4000.
# ---------------------------------------------------------------------------

def _poly(p, x):
    return sum(c * x ** i for i, c in enumerate(p))


def _d_mass(x, m):
    return np.column_stack([np.ones_like(x), x, x**2, x**3, m, m*x])


def _d_fxgx(x, m):
    return np.column_stack([np.ones_like(x), x, x**2, x**3, x**4,
                            m, m*x, m*x**2, m*x**3])


def _d_mass_m2(x, m):                       # poly3+mass + b3 m^2
    return np.column_stack([np.ones_like(x), x, x**2, x**3, m, m*x, m**2])


def _d_mass_x2m(x, m):                      # poly3+mass + b3 x^2 m
    return np.column_stack([np.ones_like(x), x, x**2, x**3, m, m*x, x**2*m])


MODELS = [
    {"name": "pol2", "npar": 3, "kind": "root", "root": "pol2",
     "pred": _poly, "cats": "both"},
    {"name": "pol3", "npar": 4, "kind": "root", "root": "pol3",
     "pred": _poly, "cats": "both"},
    {"name": "pol4", "npar": 5, "kind": "root", "root": "pol4",
     "pred": _poly, "cats": "both"},
    {"name": "physics", "npar": 5, "kind": "root",
     "root": "[0]+[1]*x+[2]*x*x+[3]/(1-x+[4])",
     "pred": lambda p, x: p[0] + p[1]*x + p[2]*x*x + p[3]/(1 - x + p[4]),
     "inits": {0: 0.05, 1: 0.0, 2: 0.0, 3: 1e-3, 4: 0.05},
     "limits": {4: (0.01, 0.5)}, "cats": "resolved"},
    {"name": "spline", "kind": "spline", "cats": "both"},
    {"name": "poly3+mass", "kind": "linear", "design": _d_mass, "npar": 6,
     "cats": "both"},
    {"name": "fxgx", "kind": "linear", "design": _d_fxgx, "npar": 9,
     "cats": "both"},
    {"name": "+m2", "kind": "linear", "design": _d_mass_m2, "npar": 7,
     "cats": "both"},
    {"name": "+x2m", "kind": "linear", "design": _d_mass_x2m, "npar": 7,
     "cats": "both"},
    {"name": "spline2d", "kind": "spline2d", "npar": None, "cats": "both"},
]

MODEL_NAMES = [m["name"] for m in MODELS]
SPEC = {m["name"]: m for m in MODELS}


def model_dim(spec):
    """'2d' for models that use M_WR (linear / spline2d), else '1d' (x-only)."""
    return "2d" if spec["kind"] in ("linear", "spline2d") else "1d"


def root_fit(x, y, ey, spec, xr):
    n = len(x)
    xa = np.ascontiguousarray(x, np.float64)
    ya = np.ascontiguousarray(y, np.float64)
    if ey is not None:
        gr = ROOT.TGraphErrors(n, xa, ya, np.zeros(n),
                               np.ascontiguousarray(ey, np.float64))
    else:
        gr = ROOT.TGraph(n, xa, ya)
    f = ROOT.TF1(f"f_{spec['name']}_{id(x)}", spec["root"], xr[0], xr[1])
    for i, v in spec.get("inits", {}).items():
        f.SetParameter(i, v)
    for i, (a, b) in spec.get("limits", {}).items():
        f.SetParLimits(i, a, b)
    gr.Fit(f, "QNR")
    params = [f.GetParameter(i) for i in range(spec["npar"])]
    chi2, ndf = f.GetChisquare(), f.GetNDF()
    chi2ndf = (chi2 / ndf) if (ey is not None and ndf > 0) else float("nan")
    return params, chi2ndf


def spline_build(x, y, nbins):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    edges = np.linspace(xs.min(), xs.max(), nbins + 1)
    idx = np.clip(np.digitize(xs, edges) - 1, 0, nbins - 1)
    bx, by = [], []
    for b in range(nbins):
        sel = idx == b
        if sel.sum():
            bx.append(np.median(xs[sel]))
            by.append(np.median(ys[sel]))
    bx, by = np.array(bx), np.array(by)
    keep = np.concatenate([[True], np.diff(bx) > 1e-9])
    bx, by = bx[keep], by[keep]
    sp = ROOT.TSpline3("sp", np.ascontiguousarray(bx, np.float64),
                       np.ascontiguousarray(by, np.float64), len(bx))
    return sp, (float(bx.min()), float(bx.max())), (bx, by)


class Spline2D:
    """Per-M_WR 1D interpolator (TSpline3 if >=4 pts, else linear), interpolated
    linearly between adjacent M_WR slices; clamped at the grid ends."""
    def __init__(self):
        self.grid = None
        self.slices = {}

    def fit(self, x, y, mwr):
        self.slices = {}
        for mw in np.unique(mwr):
            sel = mwr == mw
            order = np.argsort(x[sel])
            xs, ys = x[sel][order], y[sel][order]
            ux, inv = np.unique(xs, return_inverse=True)
            uy = np.array([ys[inv == k].mean() for k in range(len(ux))])
            if len(ux) >= 4:
                sp = ROOT.TSpline3(f"sp_{mw}_{id(self)}",
                                   np.ascontiguousarray(ux, np.float64),
                                   np.ascontiguousarray(uy, np.float64), len(ux))
                self.slices[float(mw)] = ("spline", sp, ux, uy)
            else:
                self.slices[float(mw)] = ("linear", None, ux, uy)
        self.grid = np.array(sorted(self.slices))
        return self

    def _eval1d(self, mw, xq):
        kind, sp, ux, uy = self.slices[mw]
        xc = float(np.clip(xq, ux.min(), ux.max()))
        return sp.Eval(xc) if kind == "spline" else float(np.interp(xc, ux, uy))

    def predict(self, x, mwr):
        x = np.asarray(x, float)
        mwr = np.asarray(mwr, float)
        g = self.grid
        out = np.empty(len(x))
        for i in range(len(x)):
            mw, xq = mwr[i], x[i]
            if mw <= g[0]:
                out[i] = self._eval1d(float(g[0]), xq)
            elif mw >= g[-1]:
                out[i] = self._eval1d(float(g[-1]), xq)
            else:
                j = int(np.searchsorted(g, mw))
                lo, hi = float(g[j-1]), float(g[j])
                w = (mw - lo) / (hi - lo)
                out[i] = (1-w)*self._eval1d(lo, xq) + w*self._eval1d(hi, xq)
        return out


def fit_model(spec, x, y, ey, mwr, xr):
    """Fit one model. Returns a fit-state dict consumed by predict_model."""
    kind = spec["kind"]
    if kind == "spline":
        nb = max(4, min(15, len(x) // 8))
        sp, rng, binned = spline_build(x, y, nb)
        return {"kind": "spline", "sp": sp, "rng": rng, "binned": binned}
    if kind == "linear":
        m = (np.asarray(mwr, float) - 4000.0) / 4000.0
        coef, *_ = np.linalg.lstsq(spec["design"](x, m), y, rcond=None)
        return {"kind": "linear", "coef": list(coef)}
    if kind == "spline2d":
        return {"kind": "spline2d", "model": Spline2D().fit(x, y, mwr)}
    params, chi2ndf = root_fit(x, y, ey, spec, xr)
    return {"kind": "root", "params": params, "chi2ndf": chi2ndf}


def predict_model(fit, spec, x, mwr=None):
    """Evaluate a fit. `mwr` is required for the 2D models (linear, spline2d)."""
    k = fit["kind"]
    if k == "spline":
        lo, hi = fit["rng"]
        xc = np.clip(x, lo, hi)
        return np.array([fit["sp"].Eval(float(v)) for v in xc])
    if k == "linear":
        m = (np.asarray(mwr, float) - 4000.0) / 4000.0
        return spec["design"](x, m) @ np.asarray(fit["coef"])
    if k == "spline2d":
        return fit["model"].predict(x, mwr)
    return spec["pred"](fit["params"], x)


def params_json(fit):
    """JSON-serializable parameters for a fit-state."""
    k = fit["kind"]
    if k == "root":
        return {"params": fit["params"], "chi2_ndf": fit["chi2ndf"]}
    if k == "linear":
        return {"coef": list(fit["coef"])}
    if k == "spline":
        bx, by = fit["binned"]
        return {"binned_x": bx.tolist(), "binned_y": by.tolist()}
    if k == "spline2d":
        sl = {str(mw): {"x": ux.tolist(), "y": uy.tolist()}
              for mw, (_kind, _sp, ux, uy) in fit["model"].slices.items()}
        return {"grid": fit["model"].grid.tolist(), "slices": sl}
    return None


# ---------------------------------------------------------------------------
# Plotting (matplotlib + mplhep, CMS publication style)
# ---------------------------------------------------------------------------

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mplhep as hep  # noqa: E402

PLT_CH = {"ee": "ee", "mumu": r"$\mu\mu$"}
MODEL_COLORS = {"pol2": "#bd1f01", "pol3": "#3f90da", "pol4": "#2ca02c",
                "physics": "#832db6", "spline": "#e76300",
                "poly3+mass": "#555555", "fxgx": "#e76300", "spline2d": "#92dadd",
                "+m2": "#b9ac70", "+x2m": "#964a8b"}
COM = 13


def _cms(ax, text):
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  com=COM, fontsize=16)
    ax.text(0.035, 0.96, text, transform=ax.transAxes, fontsize=15,
            va="top", ha="left")


def _save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=140, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_main(s, fits, width, ch, cat, xr, out):
    """All model curves overlaid on the data (2D models drawn at m=0)."""
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    sc = ax.scatter(s["x"], s["y"], c=s["mWR"], cmap="viridis", s=28,
                    edgecolor="black", linewidth=0.3, zorder=2)
    xg = np.linspace(xr[0], xr[1], 300)
    mwr0 = np.full_like(xg, 4000.0)              # m = 0 slice for 2D models
    for name, (spec, fit) in fits.items():
        is2d = fit["kind"] in ("linear", "spline2d")
        yg = predict_model(fit, spec, xg, mwr0 if is2d else None)
        ax.plot(xg, yg, lw=2.0, color=MODEL_COLORS.get(name, "k"), zorder=3,
                label=name + (" (m=0)" if is2d else ""))
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=16)
    ax.set_xlabel(r"$x = M_N / M_{W_R}$", fontsize=18)
    ax.set_ylabel(r"$\sigma^{\rm on} / M_{W_R}$", fontsize=18)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11, loc="upper left", bbox_to_anchor=(0.02, 0.90),
              framealpha=0.9, title="model")
    _cms(ax, f"{PLT_CH[ch]}  {cat}  ({width})")
    _save(fig, out)


def plot_model_fit(s, spec, fit, model_name, width, ch, cat, out):
    """Data (colored by M_WR) with this one model's fit overlaid: a single curve
    for 1D-in-x models, one curve per M_WR for the 2D models."""
    hep.style.use("CMS")
    vmin, vmax = float(s["mWR"].min()), float(s["mWR"].max())
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    sc = ax.scatter(s["x"], s["y"], c=s["mWR"], cmap="viridis", s=28,
                    edgecolor="black", linewidth=0.3, vmin=vmin, vmax=vmax,
                    zorder=2)
    if fit["kind"] in ("linear", "spline2d"):
        norm = plt.Normalize(vmin, vmax)
        cmap = plt.cm.viridis
        for mw in np.unique(s["mWR"]):
            xx = s["x"][s["mWR"] == mw]
            if len(xx) < 2:
                continue
            xg = np.linspace(xx.min(), xx.max(), 120)
            yg = predict_model(fit, spec, xg, np.full_like(xg, mw))
            ax.plot(xg, yg, color=cmap(norm(mw)), lw=1.5, alpha=0.95, zorder=3)
        sub = rf"{model_name} fit (one curve per $M_{{W_R}}$)"
    else:
        xg = np.linspace(float(s["x"].min()), float(s["x"].max()), 300)
        ax.plot(xg, predict_model(fit, spec, xg),
                color=MODEL_COLORS.get(model_name, "#bd1f01"), lw=2.4, zorder=3)
        sub = f"{model_name} fit"
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=16)
    ax.set_xlabel(r"$x = M_N/M_{W_R}$", fontsize=18)
    ax.set_ylabel(r"$\sigma^{\rm on}/M_{W_R}$", fontsize=18)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    _cms(ax, f"{PLT_CH[ch]}  {cat}  ({width})\n{sub}")
    _save(fig, out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def fit_width(width, samples, out_dir, params_out):
    """Fit every model for each (channel, category); store params + fit plots."""
    plots = out_dir
    for ch in CHANNELS:
        for cat in CATEGORIES:
            s = samples[(ch, cat)]
            if len(s["x"]) < 8:
                logger.warning("[%s/%s/%s] too few points (%d), skipping",
                               width, ch, cat, len(s["x"]))
                continue
            xr = (float(s["x"].min()), float(s["x"].max()))
            fits = {}
            for spec in MODELS:
                if spec["cats"] == "resolved" and cat != "resolved":
                    continue
                try:
                    fit = fit_model(spec, s["x"], s["y"], s["yerr"],
                                    s["mWR"], xr)
                except Exception as exc:
                    logger.warning("[%s/%s/%s] model %s failed: %s",
                                   width, ch, cat, spec["name"], exc)
                    continue
                name = spec["name"]
                fits[name] = (spec, fit)
                pj = params_out.setdefault(width, {}).setdefault(
                    f"{ch}_{cat}", {})
                pj[name] = {"params": params_json(fit), "x_range": xr,
                            "n_points": int(len(s["x"]))}
                # grouped 1d/2d, one folder per model, split by width:
                #   <1d|2d>/<model>/<width>/{ch}_{cat}
                try:
                    plot_model_fit(s, spec, fit, name, width, ch, cat,
                                   plots / model_dim(spec) / name / width
                                   / f"{ch}_{cat}")
                except Exception as exc:
                    logger.warning("[%s/%s/%s] %s plot failed: %s",
                                   width, ch, cat, name, exc)
            if not fits:
                continue
            try:
                plot_main(s, fits, width, ch, cat, xr,
                          plots / "all_models" / width / f"{ch}_{cat}")
            except Exception as exc:
                logger.warning("[%s/%s/%s] overlay plot failed: %s",
                               width, ch, cat, exc)
            logger.info("[%s/%s/%s] fit %d models", width, ch, cat, len(fits))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--widths-dir", type=Path,
                    default=Path(__file__).resolve().parents[3] / "1_signal_widths",
                    help="Stage-1 dir holding gaussian/ and fwhm/ width tables.")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent,
                    help="Where to write outputs (default: this script's folder).")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sg, _ = load_width_table(args.widths_dir / "gaussian" / "gauss_fit_table.csv",
                             "gauss")
    sf, _ = load_width_table(args.widths_dir / "fwhm" / "fwhm_table.csv", "fwhm")

    params_out = {}
    fit_width("gauss", sg, out_dir, params_out)
    fit_width("fwhm", sf, out_dir, params_out)

    with open(out_dir / "parameterization_params.json", "w") as fh:
        json.dump(params_out, fh, indent=2,
                  default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print(f"Wrote {out_dir/'parameterization_params.json'}")
    print(f"Per-model fit plots in {out_dir}/")


if __name__ == "__main__":
    main()
