#!/usr/bin/env python3
"""Compute gauss-vs-bifur diagnostics on the existing pull-study CSV.

Inputs:
  outputs/RunIISummer20UL18/pull_study/results.csv

Per (channel, mass, n_events, config), and per seed within a cell, we have
parallel gauss+bifur fits sharing the same toy. That lets us pair them and
compute:

  * Convergence rate (status==0 & covqual==3)
  * Median pull (bias) and 1.4826*MAD pull spread on µ
  * Median pull (bias) and spread on width:
      gauss width_truth = FWHM_onshell / 2.3548  (σ)
      bifur width_truth = FWHM_onshell / 1.1774  (Σ)
  * ΔNLL = 2 * (min_nll_gauss − min_nll_bifur), positive if bifur fits better

Output: bulk_cell_summary.csv and a console table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV_IN = HERE.parent / "results.csv"
CSV_OUT = HERE / "bulk_cell_summary.csv"

# The 8 bulk cells per channel (M_WR in {3..6} TeV × M_N/M_WR in {0.3, 0.5}).
BULK_MASSES = [
    "WR3000_N1000", "WR3000_N1400",
    "WR4000_N1200", "WR4000_N2000",
    "WR5000_N1400", "WR5000_N2400",
    "WR6000_N1800", "WR6000_N3000",
]


def robust_spread(x):
    """1.4826 * MAD — robust σ estimator."""
    x = np.asarray(x)
    if x.size < 5:
        return np.nan
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def pull_stats(df, fit_key, err_key, truth_key):
    """Return (median, spread, n) of (fit - truth) / err over valid rows."""
    err = df[err_key].to_numpy()
    valid = np.isfinite(err) & (err > 0)
    sub = df[valid]
    pulls = (sub[fit_key].to_numpy() - sub[truth_key].to_numpy()) / sub[err_key].to_numpy()
    pulls = pulls[np.isfinite(pulls)]
    if pulls.size < 5:
        return np.nan, np.nan, pulls.size
    return float(np.median(pulls)), robust_spread(pulls), pulls.size


def summarize_cell(df_cell):
    """df_cell holds one (channel, mass, n_events, config), both models, all seeds."""
    out = {}
    converged_mask = (df_cell.status == 0) & (df_cell.covqual == 3)
    for model in ("gauss", "bifur"):
        sub = df_cell[df_cell.model == model]
        n_total = len(sub)
        ok = sub[(sub.status == 0) & (sub.covqual == 3)]
        out[f"{model}_n_toys"] = n_total
        out[f"{model}_n_conv"] = len(ok)
        out[f"{model}_conv_rate"] = len(ok) / n_total if n_total else np.nan

        med_mu, spr_mu, n_mu = pull_stats(ok, "mu_fit", "mu_err", "mu_truth")
        out[f"{model}_mu_bias"] = med_mu
        out[f"{model}_mu_spread"] = spr_mu
        out[f"{model}_mu_n"] = n_mu

        med_w, spr_w, n_w = pull_stats(ok, "width_fit", "width_err", "width_truth")
        out[f"{model}_width_bias"] = med_w
        out[f"{model}_width_spread"] = spr_w
        out[f"{model}_width_n"] = n_w

    # Paired ΔNLL: match seeds within (config, n_events), keep only seeds where
    # both models converged.
    df_g = df_cell[(df_cell.model == "gauss")
                   & (df_cell.status == 0) & (df_cell.covqual == 3)]
    df_b = df_cell[(df_cell.model == "bifur")
                   & (df_cell.status == 0) & (df_cell.covqual == 3)]
    merged = df_g[["seed", "min_nll"]].merge(
        df_b[["seed", "min_nll"]], on="seed", suffixes=("_g", "_b"))
    if len(merged) >= 5:
        dnll = 2.0 * (merged["min_nll_g"].to_numpy() - merged["min_nll_b"].to_numpy())
        dnll = dnll[np.isfinite(dnll)]
        out["dnll_n_paired"] = len(dnll)
        out["dnll_median"] = float(np.median(dnll))
        out["dnll_mean"] = float(np.mean(dnll))
        out["dnll_p16"] = float(np.percentile(dnll, 16))
        out["dnll_p84"] = float(np.percentile(dnll, 84))
        out["dnll_frac_pos"] = float(np.mean(dnll > 0))   # bifur better
        out["dnll_frac_gt4"] = float(np.mean(dnll > 4.0)) # 2σ preference
        out["dnll_frac_gt1"] = float(np.mean(dnll > 1.0)) # 1σ preference
    else:
        for k in ("dnll_n_paired", "dnll_median", "dnll_mean", "dnll_p16",
                  "dnll_p84", "dnll_frac_pos", "dnll_frac_gt4", "dnll_frac_gt1"):
            out[k] = np.nan
    return out


def main():
    print(f"Reading {CSV_IN} ...")
    df = pd.read_csv(CSV_IN)
    print(f"  total rows: {len(df):,}")

    df = df[df.mass.isin(BULK_MASSES)]
    print(f"  bulk-cell rows: {len(df):,}  ({df.mass.nunique()} masses)")

    rows = []
    for (ch, mass, n_ev, cfg), df_cell in df.groupby(
        ["channel", "mass", "n_events", "config"], sort=False
    ):
        rec = {"channel": ch, "mass": mass, "n_events": int(n_ev), "config": cfg}
        rec.update(summarize_cell(df_cell))
        rows.append(rec)

    out_df = pd.DataFrame(rows).sort_values(
        ["config", "channel", "mass", "n_events"]).reset_index(drop=True)
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(CSV_OUT, index=False)
    print(f"\nWrote {CSV_OUT}")
    print(f"  shape: {out_df.shape}")

    # --- pretty console tables (production config only: "both") ---
    for cfg in ("both", "no_priors"):
        print(f"\n{'=' * 96}")
        print(f"Config = {cfg}  (production = both)")
        print(f"{'=' * 96}")
        sub = out_df[out_df.config == cfg]
        for ch in ("ee", "mumu"):
            sub_c = sub[sub.channel == ch]
            print(f"\n--- channel = {ch}  ({cfg}) ---")
            print(f"{'cell':14s} {'N':>3s}  "
                  f"{'g_conv':>6s} {'b_conv':>6s}  "
                  f"{'g_µ_bias':>9s} {'b_µ_bias':>9s}  "
                  f"{'g_µ_spr':>8s} {'b_µ_spr':>8s}  "
                  f"{'g_w_bias':>9s} {'b_w_bias':>9s}  "
                  f"{'g_w_spr':>8s} {'b_w_spr':>8s}  "
                  f"{'dnll_med':>9s} {'P(b>g)':>7s} {'P(>4)':>6s}")
            for _, r in sub_c.iterrows():
                def fmt(x, w, p=2, sign=False):
                    if not np.isfinite(x):
                        return f"{'nan':>{w}s}"
                    if sign:
                        return f"{x:+{w}.{p}f}"
                    return f"{x:{w}.{p}f}"
                print(
                    f"{r['mass']:14s} {r['n_events']:>3d}  "
                    f"{fmt(r['gauss_conv_rate'], 6, 2)} "
                    f"{fmt(r['bifur_conv_rate'], 6, 2)}  "
                    f"{fmt(r['gauss_mu_bias'], 9, 2, sign=True)} "
                    f"{fmt(r['bifur_mu_bias'], 9, 2, sign=True)}  "
                    f"{fmt(r['gauss_mu_spread'], 8, 2)} "
                    f"{fmt(r['bifur_mu_spread'], 8, 2)}  "
                    f"{fmt(r['gauss_width_bias'], 9, 2, sign=True)} "
                    f"{fmt(r['bifur_width_bias'], 9, 2, sign=True)}  "
                    f"{fmt(r['gauss_width_spread'], 8, 2)} "
                    f"{fmt(r['bifur_width_spread'], 8, 2)}  "
                    f"{fmt(r['dnll_median'], 9, 2, sign=True)} "
                    f"{fmt(r['dnll_frac_pos'], 7, 2)} "
                    f"{fmt(r['dnll_frac_gt4'], 6, 2)}"
                )


if __name__ == "__main__":
    main()
