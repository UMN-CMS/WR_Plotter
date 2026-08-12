#!/usr/bin/env python3
"""Quantify how the gauss-fit pulls depend on the prior widths (α_µ, α_σ).

Reads the 2D prior scan CSV (e.g. scan_2d_root_rng_v2.csv) and produces three
diagnostic plot types in a nested layout:

  sensitivity/
  └── n{N}/
      ├── alpha_mu_slice/         1D scan of α_µ at fixed α_σ
      │   ├── ee_{bias,spread}_{mu,width}.{pdf,png}
      │   └── mumu_{bias,spread}_{mu,width}.{pdf,png}
      ├── alpha_sigma_slice/      1D scan of α_σ at fixed α_µ
      │   └── (same 8 files)
      ├── comfort_zone/           2D heatmap of max(|spread-1|, |bias|)
      │   ├── ee.{pdf,png}
      │   └── mumu.{pdf,png}
      └── prior_fraction/         data vs prior contribution to fit variance
          └── {channel}_{param}.{pdf,png}

Also writes a CSV summary of numerical sensitivities (∂pull/∂log α) at the
operating point.

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from pull_stats import gaussian_pull_fit

logger = logging.getLogger(__name__)

CH_LAB = {"ee": "ee", "mumu": r"$\mu\mu$"}
PARAM_LAB = {"mu": r"\mu", "width": r"\sigma"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True,
                   help="One or more 2D-scan CSVs (will be concatenated).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--alpha-mu-op", type=float, default=1.0,
                   help="Operating point α_µ (for the α_σ slice and comfort marker).")
    p.add_argument("--alpha-sigma-op", type=float, default=0.20,
                   help="Operating point α_σ.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def _gfit_mu(x):
    return gaussian_pull_fit(x)["mu"]


def _gfit_sigma(x):
    return gaussian_pull_fit(x)["sigma"]


def load_and_aggregate(input_paths):
    """Read CSV(s), compute pulls, return per-(channel, N, α_µ, α_σ) aggregates
    across bulk cells & toys."""
    dfs = [pd.read_csv(p) for p in input_paths]
    df = pd.concat(dfs, ignore_index=True)
    # Remove duplicates if any combos are in multiple files
    df = df.drop_duplicates(subset=["channel", "mass", "n_events",
                                     "alpha_mu", "alpha_sigma", "seed"])
    logger.info("Loaded %d rows from %d CSV(s)", len(df), len(input_paths))
    df = df[(df.status == 0) & (df.covqual == 3)
            & (df.mu_err > 0) & (df.sigma_err > 0)]
    df["mu_pull"] = (df.mu_fit - df.mu_truth) / df.mu_err
    df["w_pull"]  = (df.sigma_fit - df.sigma_truth) / df.sigma_err
    rows = []
    for (ch, ne, am, asg), grp in df.groupby(
        ["channel", "n_events", "alpha_mu", "alpha_sigma"]):
        # Per-cell bias/width from a Gaussian fit to that cell's pulls, then
        # aggregate across the bulk cells with a (robust) median.
        per_cell = grp.groupby("mass").agg(
            mu_b=("mu_pull", _gfit_mu),
            w_b=("w_pull", _gfit_mu),
            mu_s=("mu_pull", _gfit_sigma),
            w_s=("w_pull", _gfit_sigma),
            mean_mu_err=("mu_err", "mean"),
            mean_w_err=("sigma_err", "mean"),
            mean_mu_prior=("mu_prior_sigma", "mean"),
            mean_w_prior=("sigma_prior_sigma", "mean"),
        ).reset_index()
        rows.append({
            "channel": ch, "n_events": int(ne),
            "alpha_mu": float(am), "alpha_sigma": float(asg),
            "mu_bias":     float(per_cell.mu_b.median()),
            "mu_spread":   float(per_cell.mu_s.median()),
            "w_bias":      float(per_cell.w_b.median()),
            "w_spread":    float(per_cell.w_s.median()),
            "mean_mu_err":   float(per_cell.mean_mu_err.mean()),
            "mean_w_err":    float(per_cell.mean_w_err.mean()),
            "mean_mu_prior": float(per_cell.mean_mu_prior.mean()),
            "mean_w_prior":  float(per_cell.mean_w_prior.mean()),
        })
    agg = pd.DataFrame(rows)
    logger.info("Aggregated to %d (channel, N, α_µ, α_σ) points", len(agg))
    return agg


# --------------------------------------------------------------------------- #
# (A) 1D slices through the operating point
# --------------------------------------------------------------------------- #

def plot_1d_slice(agg, scan_axis, fixed_axis_val, fixed_axis_name,
                  channel, n, param, metric,
                  out_path, op_x):
    """One 1D slice plot for (channel, n, param, metric)."""
    sub = agg[(agg.channel == channel) & (agg.n_events == n)
              & (agg[fixed_axis_name] == fixed_axis_val)].sort_values(scan_axis)
    if len(sub) < 2:
        return False
    col = f"{'mu' if param == 'mu' else 'w'}_{'bias' if metric == 'bias' else 'spread'}"
    target = 0.0 if metric == "bias" else 1.0

    plt.style.use("default")
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(sub[scan_axis], sub[col], "o-", color="#1f77b4", markersize=8, linewidth=1.8)
    ax.axhline(target, color="grey", linestyle="--", linewidth=1.2, alpha=0.7,
               label=f"target {target:.0f}")
    # Mark operating point
    op_row = sub[(sub[scan_axis] - op_x).abs() < 1e-9]
    if len(op_row):
        ax.plot(op_x, op_row[col].iloc[0], "r*", markersize=18,
                markeredgecolor="black", markeredgewidth=0.5,
                label="operating point", zorder=5)
    ax.set_xscale("log")
    plabel = PARAM_LAB[param]
    sub_letter = "\\mu" if scan_axis == "alpha_mu" else "\\sigma"
    ax.set_xlabel(rf"$\alpha_{{{sub_letter}}}$", fontsize=18)
    ax.set_ylabel((rf"pull bias on ${plabel}$  (Gaussian-fit $\mu$)" if metric == "bias"
                   else rf"pull width on ${plabel}$  (Gaussian-fit $\sigma$)"), fontsize=18)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=0.3)
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  com=13, fontsize=16)
    fixed_label = (rf"$\alpha_{{\sigma}} = {fixed_axis_val:.2f}$"
                   if fixed_axis_name == "alpha_sigma"
                   else rf"$\alpha_{{\mu}} = {fixed_axis_val:.2f}$")
    ax.text(0.04, 0.96,
            f"{CH_LAB[channel]}\nResolved SR, RunIISummer20UL18\n"
            f"Gaussian / Both Constrained\n"
            rf"$N_{{\rm events}} = {n}$"
            f"\n{fixed_label}",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.9))
    ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


def do_1d_slices(agg, out_dir, alpha_mu_op, alpha_sigma_op, n_events_list, channels):
    """Two 1D scans through (alpha_mu_op, alpha_sigma_op) per (channel, N)."""
    n_made = 0
    for n in n_events_list:
        for channel in channels:
            for param in ("mu", "width"):
                for metric in ("bias", "spread"):
                    # α_µ slice at fixed α_σ
                    out_path = (out_dir / f"n{n}" / "alpha_mu_slice"
                                / f"{channel}_{metric}_{param}.png")
                    if plot_1d_slice(agg, "alpha_mu", alpha_sigma_op, "alpha_sigma",
                                      channel, n, param, metric, out_path, alpha_mu_op):
                        n_made += 1
                    # α_σ slice at fixed α_µ
                    out_path = (out_dir / f"n{n}" / "alpha_sigma_slice"
                                / f"{channel}_{metric}_{param}.png")
                    if plot_1d_slice(agg, "alpha_sigma", alpha_mu_op, "alpha_mu",
                                      channel, n, param, metric, out_path, alpha_sigma_op):
                        n_made += 1
    logger.info("1D slices: wrote %d files", n_made)


# --------------------------------------------------------------------------- #
# (B) Comfort-zone 2D heatmap
# --------------------------------------------------------------------------- #

def plot_comfort_zone(agg, channel, n, spread_param, out_path,
                      alpha_mu_op, alpha_sigma_op):
    """Heatmap of a single pull spread (µ or σ) vs (α_µ, α_σ).

    Target is 1.0; values >1.10 mean errors underestimated, <0.90 mean
    errors overestimated (prior too tight). Divergent colormap centered
    at 1.0; dashed contours at the 0.90 / 1.10 comfort bounds.
    """
    sub = agg[(agg.channel == channel) & (agg.n_events == n)].copy()
    if len(sub) < 4:
        return False
    col = "mu_spread" if spread_param == "mu" else "w_spread"
    plabel = r"\mu" if spread_param == "mu" else r"\sigma"
    am = sorted(sub.alpha_mu.unique())
    asg = sorted(sub.alpha_sigma.unique())
    grid = np.full((len(asg), len(am)), np.nan)
    for _, r in sub.iterrows():
        grid[asg.index(float(r.alpha_sigma)), am.index(float(r.alpha_mu))] = r[col]

    plt.style.use("default")
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdBu_r",
                   vmin=0.7, vmax=1.3,
                   extent=[-0.5, len(am) - 0.5, -0.5, len(asg) - 0.5])
    ax.set_xticks(range(len(am)));   ax.set_xticklabels([f"{a:g}" for a in am],
                                                         fontsize=11, rotation=0)
    ax.set_yticks(range(len(asg)));  ax.set_yticklabels([f"{a:g}" for a in asg],
                                                         fontsize=11)
    ax.set_xlabel(r"$\alpha_\mu$", fontsize=18)
    ax.set_ylabel(r"$\alpha_\sigma$", fontsize=18)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(rf"${plabel}_{{\rm spread}}$", fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    # Cell annotations (above the operating-point marker).
    for i in range(len(asg)):
        for j in range(len(am)):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color=("white" if abs(v - 1.0) > 0.20 else "black"),
                        zorder=6)
    # Operating point: hollow black star (red was hard to see on red cells).
    if alpha_mu_op in am and alpha_sigma_op in asg:
        ax.plot(am.index(alpha_mu_op), asg.index(alpha_sigma_op),
                marker="*", markersize=24,
                markerfacecolor="none", markeredgecolor="black",
                markeredgewidth=2.0, linestyle="None",
                zorder=5)
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  com=13, fontsize=15)
    ax.text(0.04, 0.06,
            rf"{CH_LAB[channel]}, $N_{{\rm events}}={n}$",
            transform=ax.transAxes, fontsize=11, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.9))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


def do_comfort_zone(agg, out_dir, alpha_mu_op, alpha_sigma_op, n_events_list, channels):
    n_made = 0
    for n in n_events_list:
        for channel in channels:
            for spread_param in ("mu", "sigma"):
                out_path = (out_dir / f"n{n}" / "comfort_zone"
                            / f"{channel}_{spread_param}.png")
                if plot_comfort_zone(agg, channel, n, spread_param, out_path,
                                     alpha_mu_op, alpha_sigma_op):
                    n_made += 1
    logger.info("Comfort-zone: wrote %d files", n_made)


# --------------------------------------------------------------------------- #
# (C) Prior fraction: how much of σ_fit comes from the prior vs the data
# --------------------------------------------------------------------------- #

def _prior_fraction(sigma_fit, sigma_prior):
    """Fraction of 1/sigma_fit² that comes from the prior.

    1/σ_fit² = 1/σ_data² + 1/σ_prior²  →  prior_frac = σ_fit² / σ_prior²
    """
    return (sigma_fit / sigma_prior) ** 2


def plot_prior_fraction(agg, channel, param, alpha_mu_op, alpha_sigma_op,
                        out_path, n_events_list):
    """For fixed α at operating point: plot prior_fraction vs N, per channel × param.

    Also overlay a few alternative α values to show how it shifts.
    """
    err_col = "mean_mu_err" if param == "mu" else "mean_w_err"
    prior_col = "mean_mu_prior" if param == "mu" else "mean_w_prior"

    plt.style.use("default")
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(9, 6))

    # Operating point curve
    sub = agg[(agg.channel == channel) & (agg.alpha_mu == alpha_mu_op)
              & (agg.alpha_sigma == alpha_sigma_op)].sort_values("n_events")
    if len(sub) >= 2:
        pf = _prior_fraction(sub[err_col], sub[prior_col])
        ax.plot(sub.n_events, pf * 100.0, "o-", color="red", markersize=10, linewidth=2.5,
                label=rf"operating point  $\alpha_\mu={alpha_mu_op:g}$, $\alpha_\sigma={alpha_sigma_op:g}$",
                zorder=4)

    # Overlay a few other α values for context
    other_alphas = []
    if param == "mu":
        for am in sorted(agg.alpha_mu.unique()):
            if am == alpha_mu_op: continue
            other_alphas.append(("alpha_mu", am, alpha_sigma_op,
                                  rf"$\alpha_\mu = {am:g}$"))
    else:
        for asg in sorted(agg.alpha_sigma.unique()):
            if asg == alpha_sigma_op: continue
            other_alphas.append(("alpha_sigma", alpha_mu_op, asg,
                                  rf"$\alpha_\sigma = {asg:g}$"))

    cmap = plt.get_cmap("Blues")
    for i, (axis, am, asg, label) in enumerate(other_alphas):
        sub_o = agg[(agg.channel == channel) & (agg.alpha_mu == am)
                    & (agg.alpha_sigma == asg)].sort_values("n_events")
        if len(sub_o) < 2: continue
        pf = _prior_fraction(sub_o[err_col], sub_o[prior_col])
        c = cmap(0.3 + 0.6 * i / max(1, len(other_alphas) - 1))
        ax.plot(sub_o.n_events, pf * 100.0, "o-", color=c, markersize=5,
                linewidth=1.0, alpha=0.7, label=label)

    ax.set_xscale("log")
    ax.set_xlabel(r"$N_{\rm events}$", fontsize=18)
    ax.set_ylabel(rf"prior contribution to $1/\sigma^2_{{{PARAM_LAB[param]},{{\rm fit}}}}$ [%]",
                  fontsize=15)
    ax.set_ylim(0, 105)
    ax.axhline(50, color="grey", ls=":", lw=1, alpha=0.5)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=0.3)
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  com=13, fontsize=16)
    ax.text(0.04, 0.96,
            f"{CH_LAB[channel]}\nResolved SR, RunIISummer20UL18\n"
            f"Gaussian / Both Constrained\n"
            rf"prior on ${PARAM_LAB[param]}$",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.9))
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


def do_prior_fraction(agg, out_dir, alpha_mu_op, alpha_sigma_op, channels):
    """One plot per (channel, param) showing prior-fraction vs N."""
    out_sub = out_dir / "prior_fraction"
    n_made = 0
    for channel in channels:
        for param in ("mu", "width"):
            out_path = out_sub / f"{channel}_{param}.png"
            if plot_prior_fraction(agg, channel, param, alpha_mu_op, alpha_sigma_op,
                                    out_path, sorted(agg.n_events.unique())):
                n_made += 1
    logger.info("Prior-fraction: wrote %d files", n_made)


# --------------------------------------------------------------------------- #
# Numerical sensitivity summary (∂pull / ∂log α) at operating point
# --------------------------------------------------------------------------- #

def numerical_sensitivities(agg, alpha_mu_op, alpha_sigma_op, out_path):
    rows = []
    cols = ["mu_bias", "mu_spread", "w_bias", "w_spread"]
    for n in sorted(agg.n_events.unique()):
        for ch in sorted(agg.channel.unique()):
            for axis, op_val, fixed_axis, fixed_val in [
                ("alpha_mu", alpha_mu_op, "alpha_sigma", alpha_sigma_op),
                ("alpha_sigma", alpha_sigma_op, "alpha_mu", alpha_mu_op),
            ]:
                sub = agg[(agg.channel == ch) & (agg.n_events == n)
                          & (agg[fixed_axis] == fixed_val)].sort_values(axis)
                if len(sub) < 2:
                    continue
                # Finite-difference around op_val: use closest neighbors
                op_idx = sub[axis].sub(op_val).abs().idxmin()
                op_row = sub.loc[op_idx]
                op_pos = sub.index.get_loc(op_idx)
                if op_pos == 0:
                    lo_pos, hi_pos = 0, 1
                elif op_pos == len(sub) - 1:
                    lo_pos, hi_pos = len(sub) - 2, len(sub) - 1
                else:
                    lo_pos, hi_pos = op_pos - 1, op_pos + 1
                a_lo = sub.iloc[lo_pos][axis]
                a_hi = sub.iloc[hi_pos][axis]
                dlog = np.log(a_hi / a_lo)
                for c in cols:
                    d = (sub.iloc[hi_pos][c] - sub.iloc[lo_pos][c]) / dlog
                    rows.append({"channel": ch, "n_events": int(n),
                                  "axis": axis, "metric": c,
                                  "d_dlog_alpha": float(d),
                                  "op_value": float(op_row[c])})
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Sensitivities: %d rows → %s", len(df), out_path)
    # Print summary
    print(f"\nNumerical sensitivities at α_µ={alpha_mu_op}, α_σ={alpha_sigma_op}:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    agg = load_and_aggregate(args.inputs)
    n_events_list = sorted(agg.n_events.unique())
    channels = sorted(agg.channel.unique())

    do_1d_slices(agg, args.output_dir, args.alpha_mu_op, args.alpha_sigma_op,
                 n_events_list, channels)
    do_comfort_zone(agg, args.output_dir, args.alpha_mu_op, args.alpha_sigma_op,
                    n_events_list, channels)
    do_prior_fraction(agg, args.output_dir, args.alpha_mu_op, args.alpha_sigma_op,
                      channels)
    numerical_sensitivities(agg, args.alpha_mu_op, args.alpha_sigma_op,
                             args.output_dir / "sensitivities_at_op.csv")


if __name__ == "__main__":
    main()
