#!/usr/bin/env python3
"""2D-heatmap plots for the (alpha_mu, alpha_sigma) joint prior scan.

Reads scan_priors.py output CSV with multiple alpha_mu × alpha_sigma values
and produces, per channel × n_events:

  bulk-aggregated heatmaps of
    (a) median µ-pull bias  vs (α_µ, α_σ)
    (b) µ-pull spread       vs (α_µ, α_σ)
    (c) median σ-pull bias  vs (α_µ, α_σ)
    (d) σ-pull spread       vs (α_µ, α_σ)

  combined "spread fitness" heatmap:
    sqrt((µ_spread - 1)² + (σ_spread - 1)²)

Plus a summary table that prints the (α_µ, α_σ) minimizing the spread fitness
for each (channel, N).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from pull_stats import gaussian_pull_fit

HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=HERE / "outputs" / "scan_2d_plots")
    return p.parse_args()


def compute_pulls(df):
    df = df.copy()
    df["mu_pull"] = (df["mu_fit"] - df["mu_truth"]) / df["mu_err"]
    df["sigma_pull"] = (df["sigma_fit"] - df["sigma_truth"]) / df["sigma_err"]
    df.loc[df["mu_err"] <= 0, "mu_pull"] = np.nan
    df.loc[df["sigma_err"] <= 0, "sigma_pull"] = np.nan
    return df


def bulk_aggregate(df):
    """Per (channel, n_events, alpha_mu, alpha_sigma): median across cells and toys."""
    rows = []
    keys = ["channel", "n_events", "alpha_mu", "alpha_sigma"]
    for k, grp in df.groupby(keys, sort=False):
        ok = grp[(grp.status == 0) & (grp.covqual == 3)]
        rec = dict(zip(keys, k))
        rec["n_toys_total"] = len(grp)
        rec["n_toys_conv"] = len(ok)
        gf_mu = gaussian_pull_fit(ok.mu_pull) if len(ok) else None
        gf_s  = gaussian_pull_fit(ok.sigma_pull) if len(ok) else None
        rec["mu_bias"]    = gf_mu["mu"] if gf_mu else np.nan
        rec["mu_spread"]  = gf_mu["sigma"] if gf_mu else np.nan
        rec["sigma_bias"]   = gf_s["mu"] if gf_s else np.nan
        rec["sigma_spread"] = gf_s["sigma"] if gf_s else np.nan
        rec["spread_fitness"] = np.sqrt(
            (rec["mu_spread"] - 1) ** 2 + (rec["sigma_spread"] - 1) ** 2)
        rec["abs_bias_fitness"] = np.sqrt(
            rec["mu_bias"] ** 2 + rec["sigma_bias"] ** 2)
        rows.append(rec)
    return pd.DataFrame(rows)


def plot_heatmap(ax, df, value_col, *, vcenter=None, cmap="RdBu_r", title="",
                 vmin=None, vmax=None, mark_min_abs=False):
    am = sorted(df.alpha_mu.unique())
    as_ = sorted(df.alpha_sigma.unique())
    grid = np.full((len(as_), len(am)), np.nan)
    for _, r in df.iterrows():
        i = as_.index(float(r["alpha_sigma"]))
        j = am.index(float(r["alpha_mu"]))
        grid[i, j] = r[value_col]
    if vcenter is not None and vmin is None and vmax is None:
        amp = float(np.nanmax(np.abs(grid - vcenter)))
        if not np.isfinite(amp) or amp == 0:
            amp = 1.0
        vmin = vcenter - amp
        vmax = vcenter + amp
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax,
                   extent=[-0.5, len(am) - 0.5, -0.5, len(as_) - 0.5])
    ax.set_xticks(range(len(am)))
    ax.set_xticklabels([f"{x:.2g}" for x in am], fontsize=6, rotation=45)
    ax.set_yticks(range(len(as_)))
    ax.set_yticklabels([f"{y:.2g}" for y in as_], fontsize=6)
    ax.set_xlabel(r"$\alpha_\mu$", fontsize=8)
    ax.set_ylabel(r"$\alpha_\sigma$", fontsize=8)
    ax.set_title(title, fontsize=8)

    # Annotate each cell
    for i in range(len(as_)):
        for j in range(len(am)):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}" if vcenter is not None else f"{v:.2f}",
                        ha="center", va="center", fontsize=5)

    if mark_min_abs:
        flat = np.where(np.isfinite(grid), np.abs(grid - (vcenter or 0.0)), np.inf)
        i_min, j_min = np.unravel_index(np.argmin(flat), flat.shape)
        ax.plot(j_min, i_min, marker="*", color="lime", markersize=12,
                markeredgecolor="black", markeredgewidth=0.5)
    return im


def make_per_n_figures(bulk_df, output_dir):
    """One figure per (channel, n_events) — 5 heatmaps in a row."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for (ch, n_ev), sub in bulk_df.groupby(["channel", "n_events"], sort=False):
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        plot_heatmap(axes[0], sub, "mu_bias", vcenter=0.0,
                     title=f"median µ-pull bias  ({ch}, N={int(n_ev)})")
        plot_heatmap(axes[1], sub, "mu_spread", vcenter=1.0,
                     title=f"µ-pull spread  ({ch}, N={int(n_ev)})",
                     mark_min_abs=True)
        plot_heatmap(axes[2], sub, "sigma_bias", vcenter=0.0,
                     title=f"median σ-pull bias  ({ch}, N={int(n_ev)})")
        plot_heatmap(axes[3], sub, "sigma_spread", vcenter=1.0,
                     title=f"σ-pull spread  ({ch}, N={int(n_ev)})",
                     mark_min_abs=True)
        plot_heatmap(axes[4], sub, "spread_fitness", vmin=0.0, vmax=None,
                     cmap="viridis_r",
                     title=f"spread fitness √(Δμ²+Δσ²) ({ch}, N={int(n_ev)})",
                     mark_min_abs=False)
        # Mark the spread-fitness minimum
        flat = sub.set_index(["alpha_sigma", "alpha_mu"])["spread_fitness"]
        as_, am = flat.idxmin()
        as_idx = sorted(sub.alpha_sigma.unique()).index(float(as_))
        am_idx = sorted(sub.alpha_mu.unique()).index(float(am))
        axes[4].plot(am_idx, as_idx, marker="*", color="white",
                     markersize=12, markeredgecolor="black", markeredgewidth=0.5)

        fig.tight_layout()
        out = output_dir / f"heatmap_{ch}_n{int(n_ev)}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  wrote {out}")


def print_best_per_n(bulk_df):
    print("\nBest (α_µ, α_σ) minimizing spread fitness √((µ_sp-1)² + (σ_sp-1)²)")
    print(f"{'channel':>7s} {'N':>4s}  {'α_µ':>5s} {'α_σ':>5s}  "
          f"{'µ_bias':>7s} {'µ_spr':>6s}  {'σ_bias':>7s} {'σ_spr':>6s}  "
          f"{'fit':>5s}")
    for (ch, n_ev), sub in bulk_df.groupby(["channel", "n_events"], sort=False):
        # pick row minimizing spread fitness
        i_min = sub["spread_fitness"].idxmin()
        r = sub.loc[i_min]
        print(f"{ch:>7s} {int(n_ev):>4d}  {r.alpha_mu:>5.2f} {r.alpha_sigma:>5.2f}  "
              f"{r.mu_bias:+7.2f} {r.mu_spread:>6.2f}  "
              f"{r.sigma_bias:+7.2f} {r.sigma_spread:>6.2f}  "
              f"{r.spread_fitness:>5.2f}")


def print_best_overall(bulk_df):
    """Single (α_µ, α_σ) that gives best aggregated fitness across all N values."""
    print("\nSingle best (α_µ, α_σ) across N (sum of spread_fitness² over N=5..100)")
    agg = bulk_df.groupby(["channel", "alpha_mu", "alpha_sigma"]).agg(
        rms_fitness=("spread_fitness", lambda x: float(np.sqrt(np.nanmean(x ** 2)))),
        max_fitness=("spread_fitness", "max"),
        mean_mu_spread=("mu_spread", "mean"),
        mean_sigma_spread=("sigma_spread", "mean"),
        mean_mu_bias=("mu_bias", "mean"),
        mean_sigma_bias=("sigma_bias", "mean"),
    ).reset_index()
    for ch in sorted(agg.channel.unique()):
        sub = agg[agg.channel == ch].sort_values("rms_fitness")
        top = sub.head(5)
        print(f"\n  channel = {ch}, top 5 by RMS spread fitness across N:")
        print(top.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


def main():
    args = parse_args()
    df = pd.read_csv(args.input)
    print(f"Read {len(df):,} rows from {args.input}")
    df = compute_pulls(df)
    bulk_df = bulk_aggregate(df)

    out_summary = args.output_dir / "bulk_summary.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bulk_df.to_csv(out_summary, index=False)
    print(f"Wrote {out_summary}")

    print_best_per_n(bulk_df)
    print_best_overall(bulk_df)

    print(f"\nWriting heatmaps → {args.output_dir}/")
    make_per_n_figures(bulk_df, args.output_dir)


if __name__ == "__main__":
    main()
