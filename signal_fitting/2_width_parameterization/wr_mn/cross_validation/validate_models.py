#!/usr/bin/env python3
"""Score the width-parameterization models by leave-one-M_WR-out cross-validation.

Companion to `parameterize_width.py` (which only fits): this imports that script's
model registry and, for every (width, channel, category, model), measures
generalization — hold out each M_WR slice entirely, fit on the rest, predict the
held-out slice. The held-out (out-of-fold) fractional residual is the honest test
of whether a model interpolates across masses.

Per model it reports: in-sample residual diagnostics, the leave-one-M_WR-out CV
median / q95 |frac residual|, and how much residual structure still tracks M_WR
(resid_R2_m). It recommends the lowest-CV-median model per cell.

Reads the Stage-1 tables via parameterize_width.load_width_table. Writes beside
itself (this script lives in 2_width_parameterization/cross_validation/):
  cv_summary.csv                        all metrics per (width, ch, cat, model)
                                        (+ median_sigma_meas_gev for the cell)
  predictions.csv                       per-point measured + predicted σ (GeV)
  cv_comparison/<width>.{png,pdf}       CV-median bar chart, all models per cell
  residuals/<1d|2d>/<model>/<width>/{ch}_{cat}.*       held-out residual, per model
  pred_vs_meas/<1d|2d>/<model>/<width>/{ch}_{cat}.*    measured vs predicted σ (y=x)
  best_model_residual/<width>/{ch}_{cat}.*   full-size held-out residual, best model

Setup:
    source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parameterizations"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from parameterize_width import (  # noqa: E402
    load_width_table, fit_model, predict_model, model_dim, MODELS, SPEC,
    MODEL_NAMES, CHANNELS, CATEGORIES, PLT_CH,
)

logger = logging.getLogger(__name__)

# Key models for the comparison grid (baseline + the flexible 2D forms).
GRID_LABELS = {
    "pol3": r"pol3 (x only)",
    "poly3+mass": r"poly3+mass: pol3 $+ b_1 m + b_2 x m$",
    "fxgx": "fxgx:  " + r"$y = f_4(x) + m\,g_3(x)$",
    "spline2d": r"spline2d (per-$M_{W_R}$ spline, interp. in $M_{W_R}$)",
}
MODEL_COLORS = {"pol2": "#7f7f7f", "pol3": "#999999", "pol4": "#9467bd",
                "physics": "#832db6", "spline": "#e76300",
                "poly3+mass": "#3f90da", "fxgx": "#bd1f01", "spline2d": "#2ca02c",
                "+m2": "#b9ac70", "+x2m": "#964a8b"}


# ---------------------------------------------------------------------------
# Metrics + cross-validation
# ---------------------------------------------------------------------------

def metrics(y, yp, ey, npar):
    fr = (yp - y) / y
    af = np.abs(fr)
    d = {"median_abs_resid": float(np.median(af)),
         "rms_resid": float(np.sqrt(np.mean(fr**2))),
         "q68_abs_resid": float(np.quantile(af, 0.68)),
         "q95_abs_resid": float(np.quantile(af, 0.95)),
         "max_abs_resid": float(af.max())}
    if ey is not None and npar is not None:
        ndf = len(y) - npar
        d["chi2_ndf"] = (float(np.sum(((yp - y)/ey)**2)/ndf)
                         if ndf > 0 else float("nan"))
    else:
        d["chi2_ndf"] = float("nan")
    return d


def resid_R2_m(frac, m):
    """Fraction of the residual variance explained by a linear trend in m."""
    A = np.column_stack([np.ones_like(m), m])
    c, *_ = np.linalg.lstsq(A, frac, rcond=None)
    ss_res = np.sum((frac - A @ c)**2)
    ss_tot = np.sum((frac - frac.mean())**2)
    return float(max(1 - ss_res/ss_tot, 0.0)) if ss_tot > 0 else 0.0


def oof_predict(spec, s):
    """Leave-one-M_WR-out out-of-fold prediction for every point: each point is
    predicted by the model fit with its whole M_WR slice held out."""
    x, y, ey, mwr = s["x"], s["y"], s["yerr"], s["mWR"]
    xr = (float(x.min()), float(x.max()))
    need = (spec.get("npar") or 6) + 2
    yp = np.full(len(x), np.nan)
    for mw in np.unique(mwr):
        tr = mwr != mw
        if tr.sum() < need or np.unique(mwr[tr]).size < 3:
            continue
        eytr = ey[tr] if ey is not None else None
        try:
            fit = fit_model(spec, x[tr], y[tr], eytr, mwr[tr], xr)
            yp[~tr] = predict_model(fit, spec, x[~tr], mwr[~tr])
        except Exception as exc:
            logger.debug("OOF %s mWR=%.0f failed: %s", spec["name"], mw, exc)
    return yp


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save(fig, out, dpi=140):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def bar_chart(rows, width, out):
    hep.style.use("CMS")
    cells = [f"{ch}\n{cat}" for ch in CHANNELS for cat in CATEGORIES]
    keys = [(ch, cat) for ch in CHANNELS for cat in CATEGORIES]
    names = [n for n in MODEL_NAMES]
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    w = 0.8 / len(names)
    for j, name in enumerate(names):
        vals = []
        for (ch, cat) in keys:
            r = [x for x in rows if x["width_definition"] == width
                 and x["channel"] == ch and x["category"] == cat
                 and x["model"] == name]
            vals.append(r[0]["cv_median_abs_resid"]*100 if r else np.nan)
        ax.bar(np.arange(len(keys)) + (j - (len(names)-1)/2)*w, vals, w,
               label=name, color=MODEL_COLORS.get(name, "k"),
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(cells, fontsize=14)
    ax.set_ylabel("CV median |frac. residual| [%]", fontsize=16)
    ax.legend(fontsize=12, title="model", ncol=3)
    ax.grid(alpha=0.3, axis="y")
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  com=13, fontsize=15)
    ax.text(0.02, 0.93, f"width: {width}", transform=ax.transAxes, fontsize=15)
    _save(fig, out)


def single_residual(s, fr, cvm, model_label, width, ch, cat, out, ylim=None):
    """Full-size single-panel held-out fractional residual vs x for one model."""
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(11, 8.5), constrained_layout=True)
    g = np.isfinite(fr)
    sc = ax.scatter(s["x"][g], fr[g], c=s["mWR"][g], cmap="viridis", s=38,
                    edgecolor="black", linewidth=0.3,
                    vmin=s["mWR"].min(), vmax=s["mWR"].max(), zorder=2)
    ax.axhline(0, color="red", lw=1.5, alpha=0.8)
    for yv in (0.05, -0.05):
        ax.axhline(yv, color="red", ls="--", lw=1.2, alpha=0.6)
    lim = ylim if ylim is not None else max(
        0.15, float(np.quantile(np.abs(fr[g]), 0.99)) * 1.35)
    ax.set_ylim(-lim, lim)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=17)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel(r"$x = M_N/M_{W_R}$", fontsize=19)
    ax.set_ylabel(r"$(\sigma^{\rm heldout}_{\rm pred}-\sigma_{\rm meas})/"
                  r"\sigma_{\rm meas}$", fontsize=18)
    ax.tick_params(labelsize=15)
    ax.grid(alpha=0.3)
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  com=13, fontsize=17)
    ax.text(0.035, 0.965,
            f"{PLT_CH[ch]}  {cat}  ({width})\n{model_label}\n"
            f"held-out median = {cvm*100:.2f}%",
            transform=ax.transAxes, fontsize=15, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))
    noff = int(np.sum(np.abs(fr[g]) > lim))
    if noff:
        ax.text(0.98, 0.03, f"{noff} off-scale", transform=ax.transAxes,
                fontsize=12, color="gray", ha="right", va="bottom")
    _save(fig, out)


def pred_vs_meas(s, fr, model_label, cvm, width, ch, cat, out, lims):
    """Held-out predicted vs measured on-shell width σ (GeV), with the y=x line.
    σ = (σ/M_WR) · M_WR; predicted uses the leave-one-M_WR-out fold."""
    hep.style.use("CMS")
    g = np.isfinite(fr)
    sig_meas = s["y"][g] * s["mWR"][g]
    sig_pred = s["y"][g] * (1.0 + fr[g]) * s["mWR"][g]
    fig, ax = plt.subplots(figsize=(9, 8.5), constrained_layout=True)
    ax.plot(lims, lims, color="red", lw=1.4, alpha=0.85, zorder=1)
    sc = ax.scatter(sig_meas, sig_pred, c=s["mWR"][g], cmap="viridis", s=34,
                    edgecolor="black", linewidth=0.3,
                    vmin=s["mWR"].min(), vmax=s["mWR"].max(), zorder=2)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_aspect("equal", "box")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=16)
    ax.set_xlabel(r"measured $\sigma^{\rm on}$ [GeV]", fontsize=18)
    ax.set_ylabel(r"held-out predicted $\sigma^{\rm on}$ [GeV]", fontsize=18)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.3)
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  com=13, fontsize=16)
    ax.text(0.04, 0.96,
            f"{PLT_CH[ch]}  {cat}  ({width})\n{model_label}\n"
            f"held-out median = {cvm*100:.2f}%",
            transform=ax.transAxes, fontsize=13, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))
    _save(fig, out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

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

    sg, _ = load_width_table(args.widths_dir/"gaussian"/"gauss_fit_table.csv",
                             "gauss")
    sf, _ = load_width_table(args.widths_dir/"fwhm"/"fwhm_table.csv", "fwhm")

    rows = []
    pred_rows = []     # per-point measured + predicted width, for predictions.csv
    best_single = []   # (s, model, fr, cvm, width, ch, cat) for shared-ylim plots
    hdr = ("%-6s %-5s %-9s %-11s | %6s %6s | %6s %6s | %5s" %
           ("width", "ch", "cat", "model", "med%", "rms%", "CVmed%", "CVq95%",
            "R2_m"))
    print("\n" + hdr + "\n" + "-"*len(hdr))
    for width, samp in (("gauss", sg), ("fwhm", sf)):
        for ch in CHANNELS:
            for cat in CATEGORIES:
                s = samp[(ch, cat)]
                if len(s["x"]) < 12:
                    continue
                xr = (float(s["x"].min()), float(s["x"].max()))
                med_sig_meas = float(np.median(s["y"] * s["mWR"]))  # GeV, per cell
                frs, meds = {}, {}
                cell_rows = []
                for spec in MODELS:
                    if spec["cats"] == "resolved" and cat != "resolved":
                        continue
                    name = spec["name"]
                    try:
                        fit = fit_model(spec, s["x"], s["y"], s["yerr"],
                                        s["mWR"], xr)
                        yp_in = predict_model(fit, spec, s["x"], s["mWR"])
                    except Exception as exc:
                        logger.warning("[%s/%s/%s] %s fit failed: %s",
                                       width, ch, cat, name, exc)
                        continue
                    mt = metrics(s["y"], yp_in, s["yerr"], spec.get("npar"))
                    fr = (oof_predict(spec, s) - s["y"]) / s["y"]   # held-out
                    g = np.isfinite(fr)
                    cvm = float(np.median(np.abs(fr[g]))) if g.any() else float("nan")
                    cvq = (float(np.quantile(np.abs(fr[g]), 0.95))
                           if g.any() else float("nan"))
                    r2 = resid_R2_m(fr[g], s["m"][g]) if g.any() else float("nan")
                    frs[name], meds[name] = fr, cvm
                    row = {
                        "width_definition": width, "channel": ch, "category": cat,
                        "model": name, "n_points": int(len(s["x"])),
                        "median_abs_resid": mt["median_abs_resid"],
                        "rms_resid": mt["rms_resid"],
                        "q68_abs_resid": mt["q68_abs_resid"],
                        "q95_abs_resid": mt["q95_abs_resid"],
                        "max_abs_resid": mt["max_abs_resid"],
                        "chi2_ndf": mt["chi2_ndf"],
                        "cv_median_abs_resid": cvm, "cv_q95_abs_resid": cvq,
                        "resid_R2_m": r2,
                        "median_sigma_meas_gev": med_sig_meas,
                    }
                    rows.append(row); cell_rows.append(row)

                    # per-point measured + predicted width (GeV) for predictions.csv
                    yp_oof = s["y"] * (1.0 + fr)
                    for i in range(len(s["x"])):
                        mwr_i = float(s["mWR"][i])
                        pred_rows.append({
                            "width_definition": width, "channel": ch,
                            "category": cat, "model": name,
                            "M_WR": mwr_i, "M_N": round(float(s["x"][i]) * mwr_i),
                            "x": float(s["x"][i]),
                            "sigma_meas_gev": float(s["y"][i] * mwr_i),
                            "sigma_pred_insample_gev": float(yp_in[i] * mwr_i),
                            "sigma_pred_heldout_gev": (
                                float(yp_oof[i] * mwr_i)
                                if np.isfinite(fr[i]) else ""),
                            "frac_resid_heldout": (
                                float(fr[i]) if np.isfinite(fr[i]) else ""),
                        })
                    print("%-6s %-5s %-9s %-11s | %5.2f%% %5.2f%% | %5.2f%% "
                          "%5.2f%% | %4.2f" %
                          (width, ch, cat, name, mt["median_abs_resid"]*100,
                           mt["rms_resid"]*100, cvm*100, cvq*100, r2))
                if not cell_rows:
                    continue
                # one held-out residual plot per model (all models), shared
                # y-range within the cell so models are directly comparable.
                cellf = np.concatenate(
                    [np.abs(frs[m][np.isfinite(frs[m])]) for m in frs])
                cell_ylim = max(0.08, float(np.quantile(cellf, 0.985)) * 1.2)
                for name in frs:
                    single_residual(
                        s, frs[name], meds[name], GRID_LABELS.get(name, name),
                        width, ch, cat,
                        out_dir / "residuals" / model_dim(SPEC[name]) / name
                        / width / f"{ch}_{cat}",
                        ylim=cell_ylim)
                # measured-vs-predicted width (GeV) per model, shared axes per cell
                hi = float((s["y"] * s["mWR"]).max())
                for name in frs:
                    gg = np.isfinite(frs[name])
                    if gg.any():
                        hi = max(hi, float(np.max(
                            s["y"][gg] * (1.0 + frs[name][gg]) * s["mWR"][gg])))
                pvm_lims = (0.0, hi * 1.05)
                for name in frs:
                    pred_vs_meas(
                        s, frs[name], GRID_LABELS.get(name, name), meds[name],
                        width, ch, cat,
                        out_dir / "pred_vs_meas" / model_dim(SPEC[name]) / name
                        / width / f"{ch}_{cat}",
                        lims=pvm_lims)
                best = min(cell_rows, key=lambda r: r["cv_median_abs_resid"])
                best_single.append((s, best["model"], frs[best["model"]],
                                    best["cv_median_abs_resid"], width, ch, cat))
                print()
        bar_chart(rows, width, out_dir / "cv_comparison" / width)

    # full-size held-out residual for each cell's best model, shared y-range.
    if best_single:
        allfr = np.concatenate([np.abs(fr[np.isfinite(fr)])
                                for (_s, _m, fr, *_r) in best_single])
        shared = float(np.max(allfr)) * 1.05
        for (s, name, fr, cvm, width, ch, cat) in best_single:
            single_residual(s, fr, cvm, GRID_LABELS.get(name, name), width, ch,
                            cat, out_dir / "best_model_residual" / width / f"{ch}_{cat}",
                            ylim=shared)

    with open(out_dir / "cv_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    with open(out_dir / "predictions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pred_rows[0].keys()))
        w.writeheader(); w.writerows(pred_rows)

    print("=" * 72 + "\nRECOMMENDED model per cell (min CV median |frac resid|):")
    for width in ("gauss", "fwhm"):
        for ch in CHANNELS:
            for cat in CATEGORIES:
                cand = [r for r in rows if r["width_definition"] == width
                        and r["channel"] == ch and r["category"] == cat]
                if not cand:
                    continue
                b = min(cand, key=lambda r: r["cv_median_abs_resid"])
                base = [r for r in cand if r["model"] == "poly3+mass"]
                ref = (f"  (poly3+mass={base[0]['cv_median_abs_resid']*100:4.2f}%)"
                       if base else "")
                print(f"  {width:5} {ch:5} {cat:9} -> {b['model']:11} "
                      f"CV={b['cv_median_abs_resid']*100:4.2f}%{ref}")
    print(f"\nWrote {out_dir/'cv_summary.csv'} + plots in {out_dir}/")


if __name__ == "__main__":
    main()
