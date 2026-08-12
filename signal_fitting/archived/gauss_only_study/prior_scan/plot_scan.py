#!/usr/bin/env python3
"""Aggregate and plot the gauss prior-width scan.

Reads scan_priors.py output CSV and produces:

  1. bias_spread_vs_alpha_mu.{pdf,png}
       For each channel × N, two panels: median µ-pull vs α_µ, and
       µ-pull spread vs α_µ. One curve per cell (8 bulk cells), plus a
       bulk-median curve.
  2. bias_spread_vs_alpha_sigma.{pdf,png}  (only if --scan sigma)
       Same shape for σ-pull vs α_σ.
  3. summary.csv
       Per (channel, n_events, α_µ, α_σ): bulk-aggregated median pull
       and spread on µ and σ, plus convergence rate.

Pulls are computed against the bootstrap centrals (mu_truth = mu_boot,
sigma_truth = FWHM_boot / 2.3548) written by the scan script.

Usage:
  python plot_scan.py --input outputs/scan_alpha_mu.csv \\
      --output-dir outputs/ --scan mu
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True,
                   help="Scan CSV produced by scan_priors.py.")
    p.add_argument("--output-dir", type=Path, default=HERE / "outputs",
                   help="Where to write plots and summary.csv.")
    p.add_argument("--scan", choices=["mu", "sigma"], default="mu",
                   help="Which alpha was varied. Selects x-axis variable.")
    return p.parse_args()


def robust_spread(x):
    x = np.asarray(x)
    if x.size < 3:
        return np.nan
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def compute_pulls(df):
    """Add mu_pull and sigma_pull columns, filtering err > 0."""
    df = df.copy()
    df["mu_pull"] = (df["mu_fit"] - df["mu_truth"]) / df["mu_err"]
    df["sigma_pull"] = (df["sigma_fit"] - df["sigma_truth"]) / df["sigma_err"]
    df.loc[df["mu_err"] <= 0, "mu_pull"] = np.nan
    df.loc[df["sigma_err"] <= 0, "sigma_pull"] = np.nan
    return df


def aggregate_cell(df_cell):
    converged = df_cell[(df_cell.status == 0) & (df_cell.covqual == 3)]
    n_total = len(df_cell)
    n_conv = len(converged)
    return {
        "n_total": n_total,
        "n_conv": n_conv,
        "conv_rate": n_conv / n_total if n_total else np.nan,
        "mu_bias": float(np.nanmedian(converged.mu_pull)) if n_conv else np.nan,
        "mu_spread": robust_spread(converged.mu_pull.dropna()),
        "sigma_bias": float(np.nanmedian(converged.sigma_pull)) if n_conv else np.nan,
        "sigma_spread": robust_spread(converged.sigma_pull.dropna()),
    }


def make_summary(df, scan_var):
    """One row per (channel, n_events, alpha_mu, alpha_sigma, mass)."""
    keys = ["channel", "n_events", "alpha_mu", "alpha_sigma", "mass"]
    rows = []
    for k, g in df.groupby(keys, sort=False):
        rec = dict(zip(keys, k))
        rec.update(aggregate_cell(g))
        rows.append(rec)
    cell_df = pd.DataFrame(rows)

    # Bulk-aggregated (median across cells within a channel).
    bulk_rows = []
    for (ch, n_ev, am, as_), g in cell_df.groupby(
        ["channel", "n_events", "alpha_mu", "alpha_sigma"], sort=False
    ):
        bulk_rows.append({
            "channel": ch, "n_events": int(n_ev),
            "alpha_mu": float(am), "alpha_sigma": float(as_),
            "n_cells": len(g),
            "mean_conv_rate": float(np.nanmean(g.conv_rate)),
            "median_mu_bias_across_cells": float(np.nanmedian(g.mu_bias)),
            "median_mu_spread_across_cells": float(np.nanmedian(g.mu_spread)),
            "max_abs_mu_bias_across_cells": float(np.nanmax(np.abs(g.mu_bias))),
            "median_sigma_bias_across_cells": float(np.nanmedian(g.sigma_bias)),
            "median_sigma_spread_across_cells": float(np.nanmedian(g.sigma_spread)),
            "max_abs_sigma_bias_across_cells": float(np.nanmax(np.abs(g.sigma_bias))),
        })
    bulk_df = pd.DataFrame(bulk_rows)
    return cell_df, bulk_df


def plot_scan_panels(cell_df, bulk_df, scan_var, param, output_path):
    """Two-panel × one-row-per-channel × one-column-per-N figure.

    param ∈ {"mu", "sigma"} selects which pull's bias and spread to plot.
    scan_var ∈ {"alpha_mu", "alpha_sigma"} is the x-axis.
    """
    channels = sorted(cell_df.channel.unique())
    n_events_list = sorted(cell_df.n_events.unique())
    masses = sorted(cell_df.mass.unique())
    cmap = plt.get_cmap("tab10")
    mass_colors = {m: cmap(i % 10) for i, m in enumerate(masses)}

    n_rows = 2 * len(channels)  # one bias row + one spread row per channel
    n_cols = len(n_events_list)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.0 * n_cols, 2.5 * n_rows),
                             sharex=True, squeeze=False)

    bias_key = f"{param}_bias"
    spread_key = f"{param}_spread"
    bulk_bias_key = f"median_{param}_bias_across_cells"
    bulk_spread_key = f"median_{param}_spread_across_cells"
    param_label = r"$\mu$" if param == "mu" else r"$\sigma$"

    for row_ch, ch in enumerate(channels):
        for col, n_ev in enumerate(n_events_list):
            ax_bias = axes[2 * row_ch, col]
            ax_spread = axes[2 * row_ch + 1, col]
            sub_cell = cell_df[(cell_df.channel == ch) & (cell_df.n_events == n_ev)]
            sub_bulk = bulk_df[(bulk_df.channel == ch) & (bulk_df.n_events == n_ev)]
            for m in masses:
                sub_m = sub_cell[sub_cell.mass == m].sort_values(scan_var)
                if not len(sub_m):
                    continue
                ax_bias.plot(sub_m[scan_var], sub_m[bias_key],
                             "o-", color=mass_colors[m], alpha=0.5, lw=0.8,
                             ms=3, label=(m if (row_ch == 0 and col == 0) else None))
                ax_spread.plot(sub_m[scan_var], sub_m[spread_key],
                               "o-", color=mass_colors[m], alpha=0.5, lw=0.8,
                               ms=3)
            sub_bulk = sub_bulk.sort_values(scan_var)
            ax_bias.plot(sub_bulk[scan_var], sub_bulk[bulk_bias_key],
                         "k-", lw=2.0, label=("bulk median" if (row_ch == 0 and col == 0) else None))
            ax_spread.plot(sub_bulk[scan_var], sub_bulk[bulk_spread_key],
                           "k-", lw=2.0)
            ax_bias.axhline(0, color="grey", ls="--", lw=0.5)
            ax_spread.axhline(1, color="grey", ls="--", lw=0.5)
            ax_bias.set_xscale("log")
            ax_spread.set_xscale("log")
            ax_bias.set_title(f"{ch}, N={n_ev}", fontsize=9)
            if col == 0:
                ax_bias.set_ylabel(f"median {param_label}-pull")
                ax_spread.set_ylabel(f"{param_label}-pull spread")
            if 2 * row_ch + 1 == n_rows - 1:
                ax_spread.set_xlabel(scan_var)

    # Legend on the first bias panel only.
    axes[0, 0].legend(fontsize=6, loc="best", ncol=2)

    fig.suptitle(f"gauss prior scan — {scan_var} (bias and spread on {param_label})",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path.with_suffix(".png"), dpi=110)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {output_path}.{{png,pdf}}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.input}")
    df = pd.read_csv(args.input)
    print(f"  rows: {len(df):,}")
    df = compute_pulls(df)
    scan_var = "alpha_mu" if args.scan == "mu" else "alpha_sigma"

    cell_df, bulk_df = make_summary(df, scan_var)

    summary_path = args.output_dir / "summary.csv"
    bulk_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    print(f"\nbulk-aggregated summary (median across 8 cells per channel):")
    keep = ["channel", "n_events", "alpha_mu", "alpha_sigma",
            "mean_conv_rate",
            "median_mu_bias_across_cells", "median_mu_spread_across_cells",
            "max_abs_mu_bias_across_cells",
            "median_sigma_bias_across_cells", "median_sigma_spread_across_cells",
            "max_abs_sigma_bias_across_cells"]
    pd.set_option("display.float_format", "{:+.3f}".format)
    pd.set_option("display.width", 200)
    print(bulk_df[keep].to_string(index=False))

    print(f"\nWriting plots → {args.output_dir}")
    plot_scan_panels(
        cell_df, bulk_df, scan_var, "mu",
        args.output_dir / f"bias_spread_mu_vs_{scan_var}",
    )
    plot_scan_panels(
        cell_df, bulk_df, scan_var, "sigma",
        args.output_dir / f"bias_spread_sigma_vs_{scan_var}",
    )


if __name__ == "__main__":
    main()
