#!/usr/bin/env python3
"""Stage A pull study: many-toy fits across the prior-grid configs.

For each (channel, mass, n_events, model, prior_config) cell, run N_toys
signal-only toys; record fit values, errors, and Minuit convergence flags.

Output:
  outputs/<era>/pull_study/results.csv     — flat one-row-per-fit table
  outputs/<era>/pull_study/summary_<channel>.pdf — convergence rate, pull
    mean, pull width versus n_events for each (model, prior config).

Usage:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
  python signal_fitting/pull_study.py                   # ee, 100 toys (default)
  python signal_fitting/pull_study.py --n-toys 500
  python signal_fitting/pull_study.py --channels ee mumu --n-toys 200
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

try:
    import ROOT
except ImportError:
    sys.exit("ERROR: PyROOT not available. source LCG_106 first.")

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kFatal     # suppress per-fit RooFit info spam

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wrplotter.cli_utils import setup_logging
from wrplotter.config import load_lumi
from wrplotter.paths import input_dirs_for_era, repo_root

from measure_fwhm import (
    ONSHELL_WINDOW_LO_FRAC,
    ONSHELL_WINDOW_HI_FRAC,
    build_hist_key,
    build_region_name,
    compute_shape_params,
    load_and_combine_signal,
    parse_masses,
    rebin_histogram,
)
from fit_signal_toy import (
    COMPARE_CONFIGS,
    FWHM_TO_GAUSS_SIGMA,
    FWHM_TO_BIFUR_SIGMA,
    MU_PRIOR_ALPHA,
    WIDTH_PRIOR_ALPHA,
    bootstrap_fwhm_estimate,
    bootstrap_peak_estimate,
    predict_fwhm,
    run_fit,
    sample_from_hist,
)

logger = logging.getLogger(__name__)

DEFAULT_MASSES = ["WR2400_N1200", "WR4000_N2000", "WR5600_N2800"]
DEFAULT_N_EVENTS = [6, 10, 20]
DEFAULT_N_TOYS = 100

# Suppress RooFit output before any fits run.
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.FATAL)


# ---------------------------------------------------------------------------
# Per-cell truth + toy loop
# ---------------------------------------------------------------------------

def compute_truth(edges_native, vals_native, edges, vals,
                  sig_tag, channel, topology, M_WR,
                  fwhm_param, fwhm_err, fit_lo, fit_hi):
    """MC truth values used as pull denominators (deterministic, fixed seed)."""
    # Peak via bootstrap (seed=0 → deterministic across pull-study runs)
    mu_truth, mu_truth_err = bootstrap_peak_estimate(
        edges, vals, sig_tag, channel, topology, n_toys=100, seed=0,
    )

    # FWHM-derived widths from the per-point MC measurement
    sp = compute_shape_params(edges, vals, sig_tag, channel, topology)
    sigma_truth = sp.fwhm_onshell / FWHM_TO_GAUSS_SIGMA
    Sigma_truth = sp.fwhm_onshell / FWHM_TO_BIFUR_SIGMA

    # Delta from a high-stats bifur fit (5000 events sampled from MC)
    centers_n = 0.5 * (edges_native[:-1] + edges_native[1:])
    in_win = (centers_n >= fit_lo) & (centers_n <= fit_hi)
    where = np.where(in_win)[0]
    edges_win = edges_native[where[0]:where[-1] + 2]
    vals_win = vals_native[where]
    rng = np.random.default_rng(0)
    events_hi = sample_from_hist(edges_win, vals_win, 5000, rng)
    try:
        truth_fit = run_fit(
            "bifur", "constrained", events_hi, M_WR, fwhm_param, fwhm_err,
            fit_lo, fit_hi,
            mu_mode="constrained",
            mu_central=mu_truth, mu_sigma=mu_truth_err,
            suffix_extra="_truth",
        )
        Delta_truth = float(truth_fit["params"]["Delta"])
    except Exception as e:
        logger.warning("Truth bifur fit failed for %s/%s: %s", sig_tag, channel, e)
        Delta_truth = 0.0

    return {
        "mu_truth": mu_truth,
        "sigma_truth": sigma_truth,
        "Sigma_truth": Sigma_truth,
        "Delta_truth": Delta_truth,
    }


def per_cell_loop(channel, sig_tag, M_WR, M_N, n_events, n_toys,
                  edges_native, vals_native,
                  fwhm_param, fwhm_boot,
                  truth, mu_boot, sigma_peak_boot,
                  fit_lo, fit_hi, results):
    """Run all (toys × configs × PDFs) for one (channel, mass, n_events) cell.

    Caller is responsible for providing the loaded histogram, MC truth,
    fwhm_param (parameterization, used for µ prior), fwhm_boot (per-cell
    bootstrap mean, used for width prior), and the µ bootstrap.
    """
    centers_n = 0.5 * (edges_native[:-1] + edges_native[1:])
    in_win = (centers_n >= fit_lo) & (centers_n <= fit_hi)
    where = np.where(in_win)[0]
    edges_win = edges_native[where[0]:where[-1] + 2]
    vals_win = vals_native[where]

    # Per-channel α tunings. Both prior σ's scale against the parameterization
    # FWHM_param (consistent "size" reference); the width-prior *central* is
    # the per-cell bootstrap mean FWHM_boot to put the prior on MC truth.
    mu_prior_sigma     = MU_PRIOR_ALPHA[channel]    * fwhm_param
    width_prior_central = fwhm_boot
    width_prior_sigma   = WIDTH_PRIOR_ALPHA[channel] * fwhm_param

    for seed in range(1, n_toys + 1):
        rng = np.random.default_rng(seed)
        events = sample_from_hist(edges_win, vals_win, n_events, rng)

        for model in ("gauss", "bifur"):
            for cfg_name, mu_mode_cfg, width_mode_cfg, _label in COMPARE_CONFIGS:
                try:
                    fit = run_fit(
                        model, width_mode_cfg, events, M_WR,
                        width_prior_central, width_prior_sigma,
                        fit_lo, fit_hi,
                        mu_mode=mu_mode_cfg,
                        mu_central=mu_boot, mu_sigma=mu_prior_sigma,
                        suffix_extra=f"_n{n_events}_s{seed}_{cfg_name}",
                    )
                except Exception as e:
                    logger.debug("seed=%d %s/%s exception: %s",
                                 seed, model, cfg_name, e)
                    continue

                width_key = "sigma" if model == "gauss" else "Sigma"
                truth_width = (truth["sigma_truth"] if model == "gauss"
                               else truth["Sigma_truth"])

                results.append({
                    "channel": channel, "mass": sig_tag, "n_events": n_events,
                    "model": model, "config": cfg_name, "seed": seed,
                    "status": int(fit["minuit_status"]),
                    "covqual": int(fit["covqual"]),
                    "mu_truth": truth["mu_truth"],
                    "mu_fit": fit["params"]["mu"],
                    "mu_err": fit["errors"]["mu"],
                    "width_truth": truth_width,
                    "width_fit": fit["params"][width_key],
                    "width_err": fit["errors"][width_key],
                    "delta_truth": (truth["Delta_truth"] if model == "bifur"
                                    else float("nan")),
                    "delta_fit": fit["params"].get("Delta", float("nan")),
                    "delta_err": fit["errors"].get("Delta", float("nan")),
                    "n_sig_fit": fit["params"]["n_sig"],
                    "n_sig_err": fit["errors"]["n_sig"],
                    "min_nll": fit["min_nll"],
                })


# ---------------------------------------------------------------------------
# Summary plots
# ---------------------------------------------------------------------------

CONFIG_COLORS = {
    "no_priors":  "#1f77b4",
    "mu_only":    "#2ca02c",
    "width_only": "#ff7f0e",
    "both":       "#d62728",
}
CONFIG_LABELS = {
    "no_priors":  "No Priors",
    "mu_only":    r"$\mu$ constrained",
    "width_only": "Width constrained",
    "both":       "Both constrained",
}


def _compute_pull_stats(sub_ok, fit_key, err_key, truth_key):
    """Robust median and 1.4826*MAD of pulls, excluding fits where err=0."""
    err = sub_ok[err_key]
    keep = err > 0
    if keep.sum() < 5:
        return np.nan, np.nan
    pulls = (sub_ok[fit_key][keep] - sub_ok[truth_key][keep]) / err[keep]
    med = float(np.median(pulls))
    rms = float(1.4826 * np.median(np.abs(pulls - med)))  # robust σ-equivalent
    return med, rms


def make_summary_plots(rows, out_dir):
    """One figure per (channel, mass, PDF): bias and error-calibration diagnostics.

    Two columns:
      • left  = "Median pull"   — bias indicator (0 = unbiased).
      • right = "Pull spread"   — error calibration (1 = post-fit errors match the
                                    actual toy-to-toy scatter; using 1.4826 × MAD
                                    so railed-fit outliers don't dominate).

    One row per fit parameter (mu, sigma/Sigma, and Delta for bifur). Each panel
    plots the metric vs n_events with one line per prior config. Reference lines
    drawn at 0 (median) and 1 (spread).
    """
    if not rows:
        logger.warning("No results to plot.")
        return
    arr = np.array(
        [(r["channel"], r["mass"], r["n_events"], r["model"], r["config"],
          r["status"], r["covqual"],
          r["mu_truth"], r["mu_fit"], r["mu_err"],
          r["width_truth"], r["width_fit"], r["width_err"],
          r["delta_truth"], r["delta_fit"], r["delta_err"]) for r in rows],
        dtype=[("channel", "U8"), ("mass", "U20"), ("n_events", int),
               ("model", "U6"), ("config", "U16"),
               ("status", int), ("covqual", int),
               ("mu_truth", float), ("mu_fit", float), ("mu_err", float),
               ("width_truth", float), ("width_fit", float), ("width_err", float),
               ("delta_truth", float), ("delta_fit", float), ("delta_err", float)],
    )

    channels = sorted(set(arr["channel"]))
    masses = sorted(set(arr["mass"]))
    n_events_list = sorted(set(arr["n_events"]))

    n_toys_per_cell = (
        len(arr[(arr["channel"] == channels[0]) & (arr["mass"] == masses[0])])
        // (len(n_events_list) * 8)
    ) if channels and masses else 0

    PARAMS = {
        "gauss": [
            (r"$\mu$",    "mu_fit",    "mu_err",    "mu_truth"),
            (r"$\sigma$", "width_fit", "width_err", "width_truth"),
        ],
        "bifur": [
            (r"$\mu$",    "mu_fit",    "mu_err",    "mu_truth"),
            (r"$\Sigma$", "width_fit", "width_err", "width_truth"),
            (r"$\Delta$", "delta_fit", "delta_err", "delta_truth"),
        ],
    }
    PDF_DESC = {"gauss": "Single Gaussian", "bifur": "Bifurcated Gaussian"}

    out_subdir = out_dir / "per_mass"
    out_subdir.mkdir(parents=True, exist_ok=True)

    for channel in channels:
        for mass in masses:
            for model in ("gauss", "bifur"):
                params = PARAMS[model]
                n_rows_panels = len(params)
                fig, axes = plt.subplots(
                    n_rows_panels, 2,
                    figsize=(13, 3.5 * n_rows_panels + 1.5),
                    sharex=True,
                )
                if n_rows_panels == 1:
                    axes = axes.reshape(1, 2)
                for ridx, (plabel, fit_key, err_key, truth_key) in enumerate(params):
                    ax_med, ax_rms = axes[ridx, 0], axes[ridx, 1]
                    for cfg in ["no_priors", "mu_only", "width_only", "both"]:
                        meds, rmss = [], []
                        for ne in n_events_list:
                            mask = ((arr["channel"] == channel) &
                                    (arr["mass"] == mass) &
                                    (arr["n_events"] == ne) &
                                    (arr["model"] == model) &
                                    (arr["config"] == cfg))
                            sub = arr[mask]
                            if len(sub) == 0:
                                meds.append(np.nan); rmss.append(np.nan); continue
                            ok = (sub["status"] == 0) & (sub["covqual"] == 3)
                            med, rms = _compute_pull_stats(
                                sub[ok], fit_key, err_key, truth_key,
                            )
                            meds.append(med); rmss.append(rms)
                        ax_med.plot(n_events_list, meds, marker="o", linewidth=2,
                                    color=CONFIG_COLORS[cfg],
                                    label=CONFIG_LABELS[cfg])
                        ax_rms.plot(n_events_list, rmss, marker="o", linewidth=2,
                                    color=CONFIG_COLORS[cfg])
                    ax_med.axhline(0.0, color="gray", linestyle=":", linewidth=1.2)
                    ax_rms.axhline(1.0, color="gray", linestyle=":", linewidth=1.2)
                    ax_med.set_ylabel(f"Median pull on {plabel}")
                    ax_rms.set_ylabel(f"Pull spread on {plabel}")
                    if ridx == 0:
                        ax_med.set_title("Bias  (0 = unbiased)", fontsize=14)
                        ax_rms.set_title(
                            "Error calibration  (1 = post-fit errors match scatter)",
                            fontsize=14,
                        )
                    if ridx == n_rows_panels - 1:
                        ax_med.set_xlabel(r"$N_\mathrm{events}$")
                        ax_rms.set_xlabel(r"$N_\mathrm{events}$")
                    ax_med.grid(alpha=0.3)
                    ax_rms.grid(alpha=0.3)

                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper center", ncol=4,
                           bbox_to_anchor=(0.5, 1.0), fontsize=12)

                ch_lab = {"ee": "ee", "mumu": r"$\mu\mu$"}.get(channel, channel)
                fig.suptitle(
                    f"Pull diagnostics — {PDF_DESC[model]}, {ch_lab} channel, "
                    f"{mass}  ({n_toys_per_cell} toys / cell)",
                    fontsize=13, y=1.02,
                )
                fig.text(
                    0.5, -0.02,
                    "Pull = (fit − truth) / fit_error.  "
                    "Spread = 1.4826 × MAD (robust σ-equivalent, immune to outliers).",
                    ha="center", fontsize=11, style="italic",
                )
                fig.tight_layout()
                out = out_subdir / f"pull_summary_{channel}_{mass}_{model}.pdf"
                fig.savefig(out, bbox_inches="tight")
                fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
                plt.close(fig)
            logger.info("Pull-summary plots for %s/%s done", channel, mass)


def _per_cell_pulls(rows):
    """Compute per-cell pull lists for mu/width/delta.

    Returns dict[(channel, model, config, n_events, mass, pname)] -> list of pulls.
    Excludes fixed parameters (err <= 0), non-finite values, and unconverged
    toys (status != 0 or covqual < 3).
    """
    from collections import defaultdict
    PARAMS = {
        "mu":    ("mu_fit",    "mu_err",    "mu_truth"),
        "width": ("width_fit", "width_err", "width_truth"),
        "delta": ("delta_fit", "delta_err", "delta_truth"),
    }
    pulls = defaultdict(list)
    for r in rows:
        if int(r["status"]) != 0 or int(r["covqual"]) < 3:
            continue
        for pname, (fit_k, err_k, truth_k) in PARAMS.items():
            try:
                err = float(r[err_k])
                if err <= 0 or not np.isfinite(err):
                    continue
                fit = float(r[fit_k]); truth = float(r[truth_k])
                if not (np.isfinite(fit) and np.isfinite(truth)):
                    continue
            except (KeyError, ValueError):
                continue
            key = (r["channel"], r["model"], r["config"],
                   int(r["n_events"]), r["mass"], pname)
            pulls[key].append((fit - truth) / err)
    return pulls


def make_pull_bias_summary(rows, out_dir):
    """Aggregate pull-bias plot, mirroring make_convergence_plot.

    For each (channel, parameter) where parameter ∈ {mu, width, delta}:
      • 2 panels (gauss left, bifur right; delta plot omits gauss).
      • x-axis: n_events.
      • y-axis: median (across masses) of the per-mass median pull.
      • shaded band: 25–75 percentile across masses (mass-to-mass spread).
      • reference line at 0 (unbiased).
    Pulls excluded for fits where the parameter was fixed (err = 0) or
    the truth/fit is non-finite, and for unconverged toys (status != 0
    or covqual < 3).
    """
    if not rows:
        return
    import matplotlib.patches as mpatches

    PARAMS = {
        "mu":    r"$\mu$",
        "width": "width",
        "delta": r"$\Delta$",
    }

    pulls = _per_cell_pulls(rows)
    cell_medians = {k: float(np.median(v)) for k, v in pulls.items() if v}

    channels = sorted({r["channel"] for r in rows})
    n_events_list = sorted({int(r["n_events"]) for r in rows})
    n_toys = max(int(r["seed"]) for r in rows)

    hep.style.use("CMS")
    for channel in channels:
        n_masses = len({r["mass"] for r in rows if r["channel"] == channel})
        for pname, plabel in PARAMS.items():
            # delta only exists for bifur
            models = ["bifur"] if pname == "delta" else ["gauss", "bifur"]
            ncols = len(models)

            # First pass: gather median + 25/75 band per (model, cfg, n_events).
            curves = {}  # (model, cfg) -> (means, p25, p75)
            for model in models:
                for cfg in ["no_priors", "mu_only", "width_only", "both"]:
                    means, p25, p75 = [], [], []
                    for n in n_events_list:
                        per_mass = [v for k, v in cell_medians.items()
                                    if k[0] == channel and k[1] == model
                                    and k[2] == cfg and k[3] == n
                                    and k[5] == pname]
                        per_mass = np.array(per_mass)
                        if len(per_mass) == 0:
                            means.append(np.nan)
                            p25.append(np.nan); p75.append(np.nan)
                            continue
                        means.append(float(np.median(per_mass)))
                        p25.append(float(np.percentile(per_mass, 25)))
                        p75.append(float(np.percentile(per_mass, 75)))
                    curves[(model, cfg)] = (means, p25, p75)

            if not any(np.any(np.isfinite(c[0])) for c in curves.values()):
                continue

            # ylim from actual plotted curves: median + IQR bands across all
            # configs/n_events, ignoring NaN. Floor at 1 so a perfect fit
            # still has visible scale; small headroom for legend.
            plotted = np.concatenate([
                np.array(arr) for c in curves.values() for arr in c
            ])
            plotted = plotted[np.isfinite(plotted)]
            ymax = max(1.0, 1.15 * np.max(np.abs(plotted))) if plotted.size else 1.0

            fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 8))
            if ncols == 1:
                axes = [axes]
            ch_lab = {"ee": "ee", "mumu": r"$\mu\mu$"}.get(channel, channel)

            for col, model in enumerate(models):
                ax = axes[col]
                for cfg in ["no_priors", "mu_only", "width_only", "both"]:
                    means, p25, p75 = curves[(model, cfg)]
                    color = CONFIG_COLORS[cfg]
                    ax.fill_between(n_events_list, p25, p75, color=color, alpha=0.22)
                    ax.plot(n_events_list, means, marker="o", linewidth=2.5,
                            markersize=8, color=color, label=CONFIG_LABELS[cfg])
                ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.2)
                ax.set_xlabel(r"$N_\mathrm{events}$", fontsize=22)
                ax.set_ylabel(f"Median pull on {plabel}", fontsize=22)
                ax.set_xticks(n_events_list)
                ax.set_xlim(min(n_events_list) - 0.5, max(n_events_list) + 0.5)
                ax.set_ylim(-ymax, ymax * 1.6)  # extra headroom on top for legend
                ax.tick_params(labelsize=18)
                ax.grid(alpha=0.3)
                hep.cms.label(loc=0, ax=ax, data=False,
                              label="Work in Progress", com=13, fontsize=18)

                pdf_name = ("Gaussian" if model == "gauss"
                            else "Bifurcated Gaussian")
                ax.text(
                    0.04, 0.96,
                    f"{ch_lab}\nResolved SR\nRunIISummer20UL18\n"
                    f"{n_masses} mass points × {n_toys} toys\n"
                    f"{pdf_name}",
                    transform=ax.transAxes, fontsize=14, verticalalignment="top",
                )
                handles, labels = ax.get_legend_handles_labels()
                band_patch = mpatches.Patch(
                    facecolor="gray", alpha=0.30,
                    label="25–75% across mass points",
                )
                handles.append(band_patch)
                labels.append("25–75% across mass points")
                ax.legend(handles, labels,
                          loc="upper right", bbox_to_anchor=(0.98, 0.98),
                          fontsize=14, framealpha=0.85)
            fig.tight_layout()
            out = out_dir / f"pull_bias_{pname}_{channel}.pdf"
            fig.savefig(out, bbox_inches="tight")
            fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
            plt.close(fig)
            logger.info("Wrote %s", out)


def make_pull_spread_summary(rows, out_dir):
    """Aggregate pull-spread (error-calibration) plot.

    Per cell (channel, model, config, n_events, mass, parameter):
      • spread = 1.4826 × MAD(pulls) — robust σ-equivalent.
      • A well-calibrated fit has spread ≈ 1; > 1 means errors underestimated,
        < 1 means errors overestimated (or pulled by a tight prior).

    Layout matches make_pull_bias_summary: 2 panels (gauss/bifur), one figure
    per (channel, parameter), reference line at 1.0.
    """
    if not rows:
        return
    import matplotlib.patches as mpatches

    PARAMS = {
        "mu":    r"$\mu$",
        "width": "width",
        "delta": r"$\Delta$",
    }

    pulls = _per_cell_pulls(rows)
    cell_spreads = {}
    for k, v in pulls.items():
        if len(v) < 5:
            continue
        arr = np.asarray(v)
        mad = np.median(np.abs(arr - np.median(arr)))
        cell_spreads[k] = float(1.4826 * mad)

    channels = sorted({r["channel"] for r in rows})
    n_events_list = sorted({int(r["n_events"]) for r in rows})
    n_toys = max(int(r["seed"]) for r in rows)

    hep.style.use("CMS")
    for channel in channels:
        n_masses = len({r["mass"] for r in rows if r["channel"] == channel})
        for pname, plabel in PARAMS.items():
            models = ["bifur"] if pname == "delta" else ["gauss", "bifur"]
            ncols = len(models)

            curves = {}
            for model in models:
                for cfg in ["no_priors", "mu_only", "width_only", "both"]:
                    means, p25, p75 = [], [], []
                    for n in n_events_list:
                        per_mass = [v for k, v in cell_spreads.items()
                                    if k[0] == channel and k[1] == model
                                    and k[2] == cfg and k[3] == n
                                    and k[5] == pname]
                        per_mass = np.array(per_mass)
                        if len(per_mass) == 0:
                            means.append(np.nan)
                            p25.append(np.nan); p75.append(np.nan)
                            continue
                        means.append(float(np.median(per_mass)))
                        p25.append(float(np.percentile(per_mass, 25)))
                        p75.append(float(np.percentile(per_mass, 75)))
                    curves[(model, cfg)] = (means, p25, p75)

            if not any(np.any(np.isfinite(c[0])) for c in curves.values()):
                continue

            fig, axes = plt.subplots(1, ncols, figsize=(9 * ncols, 8))
            if ncols == 1:
                axes = [axes]
            ch_lab = {"ee": "ee", "mumu": r"$\mu\mu$"}.get(channel, channel)

            for col, model in enumerate(models):
                ax = axes[col]
                # Per-panel y-limits so an extreme outlier in one model doesn't
                # squash the other; clip band edges at the 90th percentile of
                # the medians so a single bad cell doesn't dominate.
                medians = np.concatenate([
                    np.array(curves[(model, c)][0])
                    for c in ["no_priors", "mu_only", "width_only", "both"]
                ])
                medians = medians[np.isfinite(medians)]
                # Robust upper bound: 75th percentile × 1.5 — ignores 1-2 extreme
                # cells (e.g. width-only on bifur at N=5 where errors blow up).
                ymax_data = (float(np.percentile(medians, 75)) * 1.5
                             if medians.size else 2.0)
                ymax = max(2.0, ymax_data)

                for cfg in ["no_priors", "mu_only", "width_only", "both"]:
                    means, p25, p75 = curves[(model, cfg)]
                    color = CONFIG_COLORS[cfg]
                    ax.fill_between(n_events_list, p25, p75, color=color, alpha=0.22)
                    ax.plot(n_events_list, means, marker="o", linewidth=2.5,
                            markersize=8, color=color, label=CONFIG_LABELS[cfg])
                ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.2)
                ax.set_xlabel(r"$N_\mathrm{events}$", fontsize=22)
                ax.set_ylabel(f"Pull spread on {plabel}  "
                              r"(1.4826 $\times$ MAD)", fontsize=22)
                ax.set_xticks(n_events_list)
                ax.set_xlim(min(n_events_list) - 0.5, max(n_events_list) + 0.5)
                ax.set_ylim(0.0, ymax * 1.4)
                ax.tick_params(labelsize=18)
                ax.grid(alpha=0.3)
                hep.cms.label(loc=0, ax=ax, data=False,
                              label="Work in Progress", com=13, fontsize=18)

                pdf_name = ("Gaussian" if model == "gauss"
                            else "Bifurcated Gaussian")
                ax.text(
                    0.04, 0.96,
                    f"{ch_lab}\nResolved SR\nRunIISummer20UL18\n"
                    f"{n_masses} mass points × {n_toys} toys\n"
                    f"{pdf_name}",
                    transform=ax.transAxes, fontsize=14, verticalalignment="top",
                )
                handles, labels = ax.get_legend_handles_labels()
                band_patch = mpatches.Patch(
                    facecolor="gray", alpha=0.30,
                    label="25–75% across mass points",
                )
                handles.append(band_patch)
                labels.append("25–75% across mass points")
                ax.legend(handles, labels,
                          loc="upper right", bbox_to_anchor=(0.98, 0.98),
                          fontsize=14, framealpha=0.85)
            fig.tight_layout()
            out = out_dir / f"pull_spread_{pname}_{channel}.pdf"
            fig.savefig(out, bbox_inches="tight")
            fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
            plt.close(fig)
            logger.info("Wrote %s", out)


def make_pull_vs_mwr(rows, out_dir, model="bifur", config="both"):
    """Drill-down: pull bias and spread vs M_WR for one (model, config).

    For each channel writes two figures (bias, spread), each with 3 panels
    (µ, width, Δ). Within a panel: x = M_WR, y = median (across M_N and
    toys) of pull bias or 1.4826×MAD; one curve per n_events. Shaded band
    is the 25–75 percentile across M_N points at the same M_WR — i.e.
    "how much does the pull depend on M_N once M_WR is fixed."

    Default (bifur, both) is the production config from the pull study,
    but `model`/`config` are exposed so the same machinery can spot-check
    other cells if needed.
    """
    if not rows:
        return
    import re
    from collections import defaultdict
    import matplotlib.patches as mpatches

    PARAMS = {
        "mu":    r"$\mu$",
        "width": "width",
        "delta": r"$\Delta$",
    }
    mass_re = re.compile(r"WR(\d+)_N(\d+)")

    pulls = _per_cell_pulls(rows)
    cell_med = {k: float(np.median(v)) for k, v in pulls.items() if v}
    cell_spr = {}
    for k, v in pulls.items():
        if len(v) < 5: continue
        a = np.asarray(v)
        cell_spr[k] = float(1.4826 * np.median(np.abs(a - np.median(a))))

    channels = sorted({r["channel"] for r in rows})
    n_events_list = sorted({int(r["n_events"]) for r in rows})
    n_toys = max(int(r["seed"]) for r in rows)

    # n_events ordered colormap (Viridis)
    cmap = plt.cm.viridis
    n_colors = {n: cmap(i / max(1, len(n_events_list) - 1))
                for i, n in enumerate(n_events_list)}

    hep.style.use("CMS")
    for channel in channels:
        # All M_WR values present in this channel
        mwr_set = sorted({int(mass_re.match(r["mass"]).group(1))
                          for r in rows if r["channel"] == channel
                          and mass_re.match(r["mass"])})

        for metric, store, ylab_fmt, ref_y in (
            ("bias", cell_med,
             lambda lab: f"Median pull on {lab}", 0.0),
            ("spread", cell_spr,
             lambda lab: f"Pull spread on {lab}  (1.4826 × MAD)", 1.0),
        ):
            # Per-(M_WR, n_events) marginal across M_N
            #   key: (n, M_WR, pname) -> (median, p25, p75)
            cells = defaultdict(list)
            for k, val in store.items():
                ch, mod, cfg, n, mass, pname = k
                if ch != channel or mod != model or cfg != config:
                    continue
                m = mass_re.match(mass)
                if m is None: continue
                M_WR = int(m.group(1))
                cells[(n, M_WR, pname)].append(val)

            curves = {}
            for (n, M_WR, pname), arr in cells.items():
                arr = np.asarray(arr)
                if not arr.size: continue
                curves[(n, M_WR, pname)] = (
                    float(np.median(arr)),
                    float(np.percentile(arr, 25)),
                    float(np.percentile(arr, 75)),
                )

            ch_lab = {"ee": "ee", "mumu": r"$\mu\mu$"}.get(channel, channel)
            fig, axes = plt.subplots(1, 3, figsize=(27, 8))

            for col, (pname, plabel) in enumerate(PARAMS.items()):
                ax = axes[col]
                # No δ for gauss
                if pname == "delta" and model == "gauss":
                    ax.text(0.5, 0.5, "Δ exists for bifurcated only",
                            transform=ax.transAxes, ha="center", va="center",
                            fontsize=14)
                    ax.set_xticks([]); ax.set_yticks([])
                    continue

                # Robust per-panel y-bounds across all curves shown
                panel_meds = []
                for n in n_events_list:
                    xs, meds, p25, p75 = [], [], [], []
                    for M_WR in mwr_set:
                        c = curves.get((n, M_WR, pname))
                        if c is None: continue
                        xs.append(M_WR)
                        meds.append(c[0]); p25.append(c[1]); p75.append(c[2])
                    if not xs: continue
                    panel_meds.extend(meds)
                    color = n_colors[n]
                    ax.fill_between(xs, p25, p75, color=color, alpha=0.18)
                    ax.plot(xs, meds, marker="o", linewidth=2.0,
                            markersize=6, color=color,
                            label=rf"$N={n}$")

                if not panel_meds:
                    continue
                if metric == "bias":
                    ymax = max(1.0, 1.20 * np.max(np.abs(panel_meds)))
                    ax.set_ylim(-ymax * 1.4, ymax * 1.4)
                else:
                    ymax = max(2.0,
                               1.5 * float(np.percentile(panel_meds, 75)))
                    ax.set_ylim(0.0, ymax * 1.3)

                ax.axhline(ref_y, color="gray", linestyle=":", linewidth=1.2)
                ax.set_xlabel(r"$M_{W_R}$ [GeV]", fontsize=20)
                ax.set_ylabel(ylab_fmt(plabel), fontsize=20)
                ax.tick_params(labelsize=15)
                ax.grid(alpha=0.3)
                hep.cms.label(loc=0, ax=ax, data=False,
                              label="Work in Progress", com=13, fontsize=16)

                pdf_name = ("Gaussian" if model == "gauss"
                            else "Bifurcated Gaussian")
                cfg_lab = CONFIG_LABELS.get(config, config)
                ax.text(
                    0.04, 0.96,
                    f"{ch_lab}\nResolved SR\nRunIISummer20UL18\n"
                    f"{pdf_name} / {cfg_lab}\n{n_toys} toys per cell",
                    transform=ax.transAxes, fontsize=13,
                    verticalalignment="top",
                )
                handles, labels = ax.get_legend_handles_labels()
                band_patch = mpatches.Patch(
                    facecolor="gray", alpha=0.30,
                    label=r"25–75% across $M_N$",
                )
                handles.append(band_patch)
                labels.append(r"25–75% across $M_N$")
                ax.legend(handles, labels,
                          loc="upper right", bbox_to_anchor=(0.98, 0.98),
                          fontsize=13, framealpha=0.85, ncol=2)

            fig.suptitle(
                f"{metric.capitalize()} vs $M_{{W_R}}$  ·  "
                f"{model} / {config}  ·  {ch_lab}",
                fontsize=18, y=1.02,
            )
            fig.tight_layout()
            out = out_dir / f"pull_vs_mwr_{metric}_{model}_{config}_{channel}.pdf"
            fig.savefig(out, bbox_inches="tight")
            fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
            plt.close(fig)
            logger.info("Wrote %s", out)


def make_outlier_mass_scan(rows, out_dir):
    """Per-channel (M_WR, M_N) scatter plots highlighting outlier mass points.

    For each channel writes one figure with 3 panels:
      1. Convergence rate at N=5, bifur/no_priors  — failure map of the
         unconstrained fit, the most stress-test scenario.
      2. Worst |median pull on µ| across n_events, bifur/both  — bias map of
         the production-quality setup.
      3. Worst pull spread on Δ across n_events, bifur/both — error-calibration
         map of the production setup.
    Each marker is one mass; color encodes the metric. The diagonal M_N = M_WR
    is dashed for reference (kinematic boundary).
    """
    if not rows:
        return
    from collections import defaultdict
    import re
    from matplotlib import colors as mcolors

    mass_re = re.compile(r"WR(\d+)_N(\d+)")

    def parse(tag):
        m = mass_re.match(tag)
        return (int(m.group(1)), int(m.group(2))) if m else (None, None)

    pulls = _per_cell_pulls(rows)
    cell_medians = {k: float(np.median(v)) for k, v in pulls.items() if v}
    cell_spreads = {}
    for k, v in pulls.items():
        if len(v) < 5: continue
        a = np.asarray(v)
        cell_spreads[k] = float(1.4826 * np.median(np.abs(a - np.median(a))))

    conv = defaultdict(lambda: {"ok": 0, "tot": 0})
    for r in rows:
        key = (r["channel"], r["model"], r["config"],
               int(r["n_events"]), r["mass"])
        conv[key]["tot"] += 1
        if int(r["status"]) == 0 and int(r["covqual"]) == 3:
            conv[key]["ok"] += 1

    channels = sorted({r["channel"] for r in rows})
    hep.style.use("CMS")
    for channel in channels:
        masses = sorted({r["mass"] for r in rows if r["channel"] == channel})
        coords = [parse(m) for m in masses]
        coords = [(m, x, y) for m, (x, y) in zip(masses, coords) if x is not None]

        # Panel data: list of (mass, M_WR, M_N, value)
        p1, p2, p3 = [], [], []
        for m, M_WR, M_N in coords:
            v1 = conv[(channel, "bifur", "no_priors", 5, m)]
            if v1["tot"]:
                p1.append((M_WR, M_N, v1["ok"] / v1["tot"]))

            biases = [abs(cell_medians.get((channel, "bifur", "both", n, m, "mu"),
                                            np.nan))
                      for n in (5, 10, 15, 20)]
            biases = [b for b in biases if np.isfinite(b)]
            if biases:
                p2.append((M_WR, M_N, max(biases)))

            spreads = [cell_spreads.get((channel, "bifur", "both", n, m, "delta"),
                                         np.nan)
                       for n in (5, 10, 15, 20)]
            spreads = [s for s in spreads if np.isfinite(s)]
            if spreads:
                p3.append((M_WR, M_N, max(spreads)))

        ch_lab = {"ee": "ee", "mumu": r"$\mu\mu$"}.get(channel, channel)
        fig, axes = plt.subplots(1, 3, figsize=(27, 8))

        panels = [
            (p1, "Convergence rate", "RdYlGn", 0.0, 1.0,
             "Bifur / No Priors / N=5", False),
            (p2, r"Max $|$median pull$|$ on $\mu$", "viridis", None, None,
             "Bifur / Both Constrained", False),
            (p3, r"Max pull spread on $\Delta$", "magma", None, None,
             "Bifur / Both Constrained", True),
        ]
        for ax, (pts, title, cmap, vmin, vmax, sub, clip) in zip(axes, panels):
            if not pts:
                ax.text(0.5, 0.5, "no data",
                        transform=ax.transAxes, ha="center", va="center")
                continue
            xs, ys, vs = zip(*pts)
            xs, ys, vs = np.array(xs), np.array(ys), np.array(vs)
            if clip:
                # Δ-spread tail can dominate the colorbar — clip at p95.
                vmax = float(np.percentile(vs, 95))
            sc = ax.scatter(xs, ys, c=vs, cmap=cmap, s=42,
                            vmin=vmin, vmax=vmax, edgecolor="black",
                            linewidth=0.4)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=14)
            xmax = max(xs.max(), ys.max()) * 1.05
            ax.plot([0, xmax], [0, xmax], "--", color="gray", linewidth=1)
            ax.set_xlim(0, xmax)
            ax.set_ylim(0, xmax)
            ax.set_xlabel(r"$M_{W_R}$ [GeV]", fontsize=20)
            ax.set_ylabel(r"$M_{N}$ [GeV]", fontsize=20)
            ax.tick_params(labelsize=15)
            ax.set_title(title, fontsize=18)
            ax.text(0.04, 0.96, f"{ch_lab}\nResolved SR\n{sub}",
                    transform=ax.transAxes, fontsize=13,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="gray", alpha=0.9))
            ax.grid(alpha=0.25)

        fig.suptitle("Outlier-mass scan  ·  RunIISummer20UL18",
                     fontsize=20, y=1.02)
        fig.tight_layout()
        out = out_dir / f"outlier_mass_scan_{channel}.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info("Wrote %s", out)


def make_convergence_xy(rows, out_dir, model="bifur", config="both"):
    """Convergence rate per mass point, plotted in the (x, M_WR) plane —
    same style as the FWHM-fit data plots in fit_fwhm_parameterization.

    For each (channel, n_events): one figure with x-axis = M_N / M_WR,
    y-axis = convergence rate, marker colour = M_WR (gradient).

    Default `model`/`config` is the production bifur/both setting; pass
    other values to inspect alternative setups.
    """
    if not rows: return
    import re
    from collections import defaultdict
    from matplotlib import cm, colors as mcolors

    mass_re = re.compile(r"WR(\d+)_N(\d+)")

    agg = defaultdict(lambda: {"ok": 0, "tot": 0})
    for r in rows:
        if r["model"] != model or r["config"] != config:
            continue
        key = (r["channel"], int(r["n_events"]), r["mass"])
        agg[key]["tot"] += 1
        if int(r["status"]) == 0 and int(r["covqual"]) == 3:
            agg[key]["ok"] += 1

    channels = sorted({k[0] for k in agg})
    n_events_list = sorted({k[1] for k in agg})

    # Discover M_WR range across all masses for a uniform colour scale
    all_mwrs = []
    for k in agg:
        m = mass_re.match(k[2])
        if m: all_mwrs.append(int(m.group(1)))
    mwr_lo, mwr_hi = (min(all_mwrs), max(all_mwrs)) if all_mwrs else (2000, 6000)
    norm = mcolors.Normalize(vmin=mwr_lo, vmax=mwr_hi)
    cmap = cm.viridis

    pdf_name = "Bifurcated Gaussian" if model == "bifur" else "Gaussian"
    cfg_lab = {"no_priors": "No Priors", "mu_only": r"$\mu$ Constrained",
               "width_only": "Width Constrained",
               "both": "Both Constrained"}.get(config, config)

    hep.style.use("CMS")
    for channel in channels:
        ch_lab = {"ee": "ee", "mumu": r"$\mu\mu$"}.get(channel, channel)
        for n in n_events_list:
            xs, ys, mwrs = [], [], []
            for (ch, nn, mass), v in agg.items():
                if ch != channel or nn != n: continue
                if v["tot"] == 0: continue
                m = mass_re.match(mass)
                if not m: continue
                M_WR = int(m.group(1)); M_N = int(m.group(2))
                xs.append(M_N / M_WR)
                ys.append(v["ok"] / v["tot"])
                mwrs.append(M_WR)
            if not xs: continue
            xs, ys, mwrs = np.array(xs), np.array(ys), np.array(mwrs)
            order = np.argsort(mwrs)
            xs, ys, mwrs = xs[order], ys[order], mwrs[order]

            fig, ax = plt.subplots(figsize=(11, 8))
            sc = ax.scatter(xs, ys, c=mwrs, cmap=cmap, norm=norm,
                            s=55, edgecolor="black", linewidth=0.4, zorder=3)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
            cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=18)
            cbar.ax.tick_params(labelsize=14)

            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2,
                       alpha=0.6, zorder=1)
            ax.set_xlabel(r"$M_N / M_{W_R}$", fontsize=22)
            ax.set_ylabel("Convergence rate", fontsize=22)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(-0.02, 1.05)
            ax.tick_params(labelsize=16)
            ax.grid(alpha=0.3)
            hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                          com=13, fontsize=18)

            ax.text(
                0.04, 0.40,
                f"{ch_lab}\nResolved SR\nRunIISummer20UL18\n"
                f"{pdf_name} / {cfg_lab}\n"
                rf"$N_{{\rm events}} = {n}$ × 100 toys per cell",
                transform=ax.transAxes, fontsize=13, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="gray", alpha=0.9),
            )
            fig.tight_layout()
            out = (out_dir /
                   f"convergence_xy_{model}_{config}_{channel}_n{n}.pdf")
            fig.savefig(out, bbox_inches="tight")
            fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
            plt.close(fig)
            logger.info("Wrote %s", out)


def make_pull_xy(rows, out_dir, param="mu", metric="bias",
                 model="bifur", config="both"):
    """Per-mass-point pull bias or spread, plotted in (x, M_WR) with M_WR as
    marker colour. Same style as make_convergence_xy.

    For each (channel, n_events): one figure with x-axis = M_N / M_WR,
    y-axis = median pull (metric='bias') or half-68% spread (metric='spread'),
    marker colour = M_WR. Default model/config is the production bifur/both.
    `param` is one of 'mu', 'width', 'delta'.
    """
    if not rows: return
    import re
    from collections import defaultdict
    from matplotlib import cm, colors as mcolors

    PARAMS = {
        "mu":    ("mu_fit",    "mu_err",    "mu_truth",    r"\mu"),
        "width": ("width_fit", "width_err", "width_truth", r"\mathrm{width}"),
        "delta": ("delta_fit", "delta_err", "delta_truth", r"\Delta"),
    }
    if param not in PARAMS:
        raise ValueError(f"unknown param {param!r}")
    if metric not in ("bias", "spread"):
        raise ValueError(f"metric must be 'bias' or 'spread', not {metric!r}")
    fit_k, err_k, truth_k, plabel = PARAMS[param]
    mass_re = re.compile(r"WR(\d+)_N(\d+)")

    cell_pulls = defaultdict(list)
    for r in rows:
        if r["model"] != model or r["config"] != config: continue
        if int(r["status"]) != 0 or int(r["covqual"]) < 3: continue
        try:
            err = float(r[err_k])
            if err <= 0 or not np.isfinite(err): continue
            fit = float(r[fit_k]); truth = float(r[truth_k])
            if not (np.isfinite(fit) and np.isfinite(truth)): continue
        except (KeyError, ValueError):
            continue
        cell_pulls[(r["channel"], int(r["n_events"]), r["mass"])].append(
            (fit - truth) / err)

    cell_metric = {}
    for k, vs in cell_pulls.items():
        if len(vs) < 5: continue
        a = np.asarray(vs)
        if metric == "bias":
            cell_metric[k] = float(np.median(a))
        else:
            p16, p84 = np.percentile(a, [16, 84])
            cell_metric[k] = 0.5 * float(p84 - p16)

    channels = sorted({k[0] for k in cell_metric})
    n_events_list = sorted({k[1] for k in cell_metric})

    all_mwrs = []
    for k in cell_metric:
        m = mass_re.match(k[2])
        if m: all_mwrs.append(int(m.group(1)))
    if not all_mwrs: return
    mwr_lo, mwr_hi = min(all_mwrs), max(all_mwrs)
    norm = mcolors.Normalize(vmin=mwr_lo, vmax=mwr_hi)
    cmap = cm.viridis

    pdf_name = "Bifurcated Gaussian" if model == "bifur" else "Gaussian"
    cfg_lab = {"no_priors": "No Priors", "mu_only": r"$\mu$ Constrained",
               "width_only": "Width Constrained",
               "both": "Both Constrained"}.get(config, config)
    metric_label = (rf"Median pull on ${plabel}$" if metric == "bias"
                    else rf"Pull spread on ${plabel}$  (half-68%)")
    ref_y = 0.0 if metric == "bias" else 1.0

    hep.style.use("CMS")
    for channel in channels:
        ch_lab = {"ee": "ee", "mumu": r"$\mu\mu$"}.get(channel, channel)
        for n in n_events_list:
            xs, ys, mwrs = [], [], []
            for (ch, nn, mass), v in cell_metric.items():
                if ch != channel or nn != n: continue
                m = mass_re.match(mass)
                if not m: continue
                M_WR = int(m.group(1)); M_N = int(m.group(2))
                xs.append(M_N / M_WR)
                ys.append(v)
                mwrs.append(M_WR)
            if not xs: continue
            xs, ys, mwrs = np.array(xs), np.array(ys), np.array(mwrs)
            order = np.argsort(mwrs)
            xs, ys, mwrs = xs[order], ys[order], mwrs[order]

            fig, ax = plt.subplots(figsize=(11, 8))
            sc = ax.scatter(xs, ys, c=mwrs, cmap=cmap, norm=norm,
                            s=55, edgecolor="black", linewidth=0.4, zorder=3)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
            cbar.set_label(r"$M_{W_R}$ [GeV]", fontsize=18)
            cbar.ax.tick_params(labelsize=14)

            ax.axhline(ref_y, color="gray", linestyle="--",
                       linewidth=1.2, alpha=0.6, zorder=1)
            ax.set_xlabel(r"$M_N / M_{W_R}$", fontsize=22)
            ax.set_ylabel(metric_label, fontsize=22)
            ax.set_xlim(0.0, 1.0)

            if metric == "bias":
                ymax = max(1.0, 1.20 * float(np.max(np.abs(ys))))
                ax.set_ylim(-ymax, ymax)
            else:
                p99 = float(np.percentile(np.abs(ys - 1), 99))
                ymax = max(0.5, 1.5 * p99)
                ax.set_ylim(max(0.0, 1 - ymax), 1 + ymax)

            ax.tick_params(labelsize=16)
            ax.grid(alpha=0.3)
            hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                          com=13, fontsize=18)

            ax.text(
                0.04, 0.96,
                f"{ch_lab}\nResolved SR\nRunIISummer20UL18\n"
                f"{pdf_name} / {cfg_lab}\n"
                rf"$N_{{\rm events}} = {n}$ × 100 toys per cell",
                transform=ax.transAxes, fontsize=13, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="gray", alpha=0.9),
            )
            fig.tight_layout()
            out = (out_dir /
                   f"pull_xy_{metric}_{param}_{model}_{config}_{channel}_n{n}.pdf")
            fig.savefig(out, bbox_inches="tight")
            fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
            plt.close(fig)
            logger.info("Wrote %s", out)


def make_convergence_plot(rows, out_dir):
    """Aggregate convergence-rate summary: 2 panels per channel (gauss, bifur),
    each showing convergence rate vs n_events with one curve per prior config.
    Mean line = average across all masses; band = 25–75 percentile spread;
    triangles = worst-mass convergence at each n_events.
    """
    if not rows: return
    from collections import defaultdict
    agg = defaultdict(lambda: {"ok": 0, "tot": 0})
    for r in rows:
        key = (r["channel"], r["config"], int(r["n_events"]),
               r["model"], r["mass"])
        agg[key]["tot"] += 1
        if int(r["status"]) == 0 and int(r["covqual"]) == 3:
            agg[key]["ok"] += 1

    channels = sorted({r["channel"] for r in rows})
    n_events_list = sorted({int(r["n_events"]) for r in rows})
    n_toys = max(int(r["seed"]) for r in rows)

    hep.style.use("CMS")
    for channel in channels:
        n_masses = len({r["mass"] for r in rows if r["channel"] == channel})
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        ch_lab = {"ee": "ee", "mumu": r"$\mu\mu$"}.get(channel, channel)
        for col, model in enumerate(["gauss", "bifur"]):
            ax = axes[col]
            for cfg in ["no_priors", "mu_only", "width_only", "both"]:
                means, p25, p75 = [], [], []
                for n in n_events_list:
                    rates = [v["ok"] / v["tot"] for k, v in agg.items()
                             if k[0] == channel and k[1] == cfg
                             and k[2] == n and k[3] == model]
                    rates = np.array(rates)
                    means.append(rates.mean() if len(rates) else np.nan)
                    p25.append(np.percentile(rates, 25) if len(rates) else np.nan)
                    p75.append(np.percentile(rates, 75) if len(rates) else np.nan)
                color = CONFIG_COLORS[cfg]
                ax.fill_between(n_events_list, p25, p75, color=color, alpha=0.22)
                ax.plot(n_events_list, means, marker="o", linewidth=2.5,
                        markersize=8, color=color, label=CONFIG_LABELS[cfg])
            ax.set_xlabel(r"$N_\mathrm{events}$", fontsize=22)
            ax.set_ylabel("Convergence rate", fontsize=22)
            ax.set_xticks(n_events_list)
            ax.set_xlim(min(n_events_list) - 0.5, max(n_events_list) + 0.5)
            ax.set_ylim(0.4, 1.40)  # headroom for top-right legend
            ax.set_yticks(np.arange(0.4, 1.01, 0.1))
            ax.tick_params(labelsize=18)
            ax.grid(alpha=0.3)

            hep.cms.label(loc=0, ax=ax, data=False,
                          label="Work in Progress",
                          com=13, fontsize=18)

            # Analysis info + PDF name as the last line of the same text block.
            pdf_name = "Gaussian" if model == "gauss" else "Bifurcated Gaussian"
            ax.text(
                0.04, 0.96,
                f"{ch_lab}\nResolved SR\nRunIISummer20UL18\n"
                f"{n_masses} mass points × {n_toys} toys\n"
                f"{pdf_name}",
                transform=ax.transAxes, fontsize=14, verticalalignment="top",
            )

            # Legend (with extra band-patch entry) on both panels.
            import matplotlib.patches as mpatches
            handles, labels = ax.get_legend_handles_labels()
            band_patch = mpatches.Patch(
                facecolor="gray", alpha=0.30,
                label="25–75% across mass points",
            )
            handles.append(band_patch)
            labels.append("25–75% across mass points")
            ax.legend(handles, labels,
                      loc="upper right", bbox_to_anchor=(0.98, 0.98),
                      fontsize=14, framealpha=0.85)

        fig.tight_layout()
        out = out_dir / f"convergence_summary_{channel}.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
        plt.close(fig)
        logger.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--dir", default="20260406_signals")
    p.add_argument("--channels", nargs="+", default=["ee"])
    p.add_argument("--masses", nargs="+", default=DEFAULT_MASSES)
    p.add_argument("--all", action="store_true",
                   help="Run all FWHM-fit-eligible mass points per channel "
                        "(reads the per-channel point list from results.json).")
    p.add_argument("--n-events", nargs="+", type=int, default=DEFAULT_N_EVENTS)
    p.add_argument("--n-toys", type=int, default=DEFAULT_N_TOYS,
                   help=f"Number of toys per cell (default: {DEFAULT_N_TOYS}).")
    p.add_argument("--topology", choices=["resolved", "boosted"], default="resolved")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--plot-only", action="store_true",
                   help="Skip the fit loop; replot from existing results.csv.")
    p.add_argument("--skip-per-mass", action="store_true",
                   help="Under --plot-only, skip the slow per-mass plots and "
                        "regenerate only the aggregate summaries.")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore any existing results.csv (don't resume).")
    p.add_argument("--max-masses-per-run", type=int, default=0,
                   help="Exit cleanly after this many *new* (channel,mass) cells "
                        "have been processed. 0 = no limit. Used to chunk runs "
                        "around RooFit/cppyy memory leaks; combined with --resume, "
                        "the whole job can be completed by repeated invocations.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    hep.style.use("CMS")

    out_dir = Path(args.output_dir or
                   repo_root() / "signal_fitting" / "outputs" / args.era /
                   "pull_study")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"

    if args.plot_only:
        logger.info("Replotting from %s", csv_path)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = []
            for r in reader:
                for k in ("n_events", "seed", "status", "covqual"):
                    r[k] = int(r[k])
                for k in ("mu_truth", "mu_fit", "mu_err",
                          "width_truth", "width_fit", "width_err",
                          "delta_truth", "delta_fit", "delta_err",
                          "n_sig_fit", "n_sig_err", "min_nll"):
                    r[k] = float(r[k])
                rows.append(r)
        make_convergence_plot(rows, out_dir)
        make_convergence_xy(rows, out_dir, model="bifur", config="both")
        make_pull_bias_summary(rows, out_dir)
        make_pull_spread_summary(rows, out_dir)
        make_outlier_mass_scan(rows, out_dir)
        make_pull_vs_mwr(rows, out_dir, model="bifur", config="both")
        if not args.skip_per_mass:
            make_summary_plots(rows, out_dir)
        return

    # Resume support: if results.csv exists, preload it and skip cells that
    # are already complete. This lets a crashed/killed run pick up where it
    # left off — just re-run with the same args.
    results: list[dict] = []
    completed: set = set()  # (channel, mass, n_events) cells already done
    if csv_path.exists() and not args.fresh:
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                for k in ("n_events", "seed", "status", "covqual"):
                    r[k] = int(r[k])
                for k in ("mu_truth", "mu_fit", "mu_err",
                          "width_truth", "width_fit", "width_err",
                          "delta_truth", "delta_fit", "delta_err",
                          "n_sig_fit", "n_sig_err", "min_nll"):
                    r[k] = float(r[k])
                results.append(r)
                completed.add((r["channel"], r["mass"], r["n_events"]))
        logger.info("Resume: loaded %d rows, %d (channel,mass,n) cells already complete",
                    len(results), len(completed))

    results_json_path = (repo_root() / "signal_fitting" / "outputs" / args.era /
                         "fwhm" / "fits" / "results.json")
    with open(results_json_path) as f:
        param_results_all = json.load(f)
    input_dirs, _ = input_dirs_for_era(args.era, repo_root(), args.dir)

    new_cells_done = 0  # for --max-masses-per-run early exit
    for channel in args.channels:
        # Mass list: --all → use Phase-2 point list for this channel
        if args.all:
            mass_list = [p["tag"] for p in param_results_all[channel]["points"]]
        else:
            mass_list = list(args.masses)
        n_cells_outer = len(mass_list)
        logger.info("[%s] %d mass points × %d n_events × %d toys × 8 configs = %d fits",
                    channel, n_cells_outer, len(args.n_events), args.n_toys,
                    n_cells_outer * len(args.n_events) * args.n_toys * 8)

        region = build_region_name(channel, args.topology)
        mass_var = ("mass_twoobject" if args.topology == "boosted"
                    else "mass_fourobject")
        hist_key = build_hist_key(region, mass_var)
        model_entry = param_results_all[channel]["models"]["a_linear"]

        for outer_i, sig_tag in enumerate(mass_list, start=1):
            try:
                M_WR, M_N = parse_masses(sig_tag)
                M_WR = float(M_WR); M_N = float(M_N)
            except Exception:
                logger.warning("Bad mass tag %r, skipping", sig_tag)
                continue
            # Skip the whole mass if every n_events cell is already in the CSV.
            missing_n_events = [n for n in args.n_events
                                if (channel, sig_tag, n) not in completed]
            if not missing_n_events:
                continue
            x = M_N / M_WR

            # Predict FWHM at this point.
            fwhm_param, fwhm_err = predict_fwhm(model_entry, x, M_WR)

            # Histogram (loaded once per (channel, mass)).
            try:
                edges_native, vals_native, var_native = load_and_combine_signal(
                    input_dirs, hist_key, sig_tag,
                )
            except Exception as e:
                logger.warning("Failed to load %s/%s: %s", channel, sig_tag, e)
                continue
            edges, vals, _ = rebin_histogram(
                edges_native, vals_native, var_native, 6,
            )

            fit_lo = ONSHELL_WINDOW_LO_FRAC * M_WR
            fit_hi = ONSHELL_WINDOW_HI_FRAC * M_WR

            # Truth values + bootstrap (computed once per cell, fixed seed=0
            # so the prior central is the same across all toys).
            truth = compute_truth(
                edges_native, vals_native, edges, vals,
                sig_tag, channel, args.topology, M_WR,
                fwhm_param, fwhm_err, fit_lo, fit_hi,
            )
            mu_boot, sigma_peak_boot = bootstrap_peak_estimate(
                edges, vals, sig_tag, channel, args.topology,
                n_toys=100, seed=0,
            )
            # Bootstrap-mean MC FWHM for the width-prior central (fixed seed=0
            # so the prior is deterministic across toys, like mu_boot).
            fwhm_boot, _fwhm_boot_unc = bootstrap_fwhm_estimate(
                edges, vals, sig_tag, channel, args.topology,
                n_toys=100, seed=0,
            )
            logger.info(
                "[%s][%d/%d] %s  truth: µ=%.0f σ=%.0f Σ=%.0f Δ=%+.0f  "
                "(µ_boot=%.0f±%.0f, FWHM_boot=%.0f, FWHM_param=%.0f)",
                channel, outer_i, n_cells_outer, sig_tag,
                truth["mu_truth"], truth["sigma_truth"],
                truth["Sigma_truth"], truth["Delta_truth"],
                mu_boot, sigma_peak_boot, fwhm_boot, fwhm_param,
            )

            for n_events in missing_n_events:
                per_cell_loop(
                    channel, sig_tag, M_WR, M_N, n_events, args.n_toys,
                    edges_native, vals_native,
                    fwhm_param, fwhm_boot,
                    truth, mu_boot, sigma_peak_boot,
                    fit_lo, fit_hi, results,
                )
                completed.add((channel, sig_tag, n_events))

            new_cells_done += 1
            if (args.max_masses_per_run > 0 and
                    new_cells_done >= args.max_masses_per_run):
                logger.info("Reached --max-masses-per-run=%d. Exiting cleanly "
                            "(re-run with same args to continue).",
                            args.max_masses_per_run)
                # Final flush before exit
                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=list(results[0].keys()))
                    writer.writeheader()
                    writer.writerows(results)
                logger.info("Wrote %s (%d rows)", csv_path, len(results))
                return

            # Periodic checkpoint: write the partial CSV every 5 mass points
            # so a crash mid-job doesn't lose progress.
            if outer_i % 5 == 0 and results:
                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=list(results[0].keys()))
                    writer.writeheader()
                    writer.writerows(results)
                logger.info("  [checkpoint] wrote %s (%d rows so far)",
                            csv_path, len(results))

    if results:
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        logger.info("Wrote %s (%d rows)", csv_path, len(results))

    make_convergence_plot(results, out_dir)
    make_convergence_xy(results, out_dir, model="bifur", config="both")
    make_pull_bias_summary(results, out_dir)
    make_pull_spread_summary(results, out_dir)
    make_outlier_mass_scan(results, out_dir)
    make_pull_vs_mwr(results, out_dir, model="bifur", config="both")
    make_summary_plots(results, out_dir)


if __name__ == "__main__":
    main()
