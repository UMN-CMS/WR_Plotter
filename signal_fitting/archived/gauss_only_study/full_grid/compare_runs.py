#!/usr/bin/env python3
"""Compare two full-grid results CSVs side-by-side on bias/spread metrics.

Usage:
  python compare_runs.py --old outputs/results.csv --new outputs/results_chsplit.csv

Prints per-(channel, N) bulk statistics for both runs and a per-mass diff table.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--n-events", type=int, default=20)
    p.add_argument("--label-old", default="baseline")
    p.add_argument("--label-new", default="ch-split")
    return p.parse_args()


def half68(x):
    x = np.asarray(x); x = x[np.isfinite(x)]
    if x.size < 5: return np.nan
    p16, p84 = np.percentile(x, [16, 84])
    return 0.5 * (p84 - p16)


def cell_stats(df):
    rows = []
    for (ch, mass, n_ev), grp in df.groupby(["channel", "mass", "n_events"]):
        ok = grp[(grp.status == 0) & (grp.covqual == 3) & (grp.mu_err > 0) & (grp.width_err > 0)]
        if len(ok) < 5:
            continue
        mu_pull = (ok.mu_fit - ok.mu_truth) / ok.mu_err
        w_pull = (ok.width_fit - ok.width_truth) / ok.width_err
        rows.append({
            "channel": ch, "mass": mass, "n_events": int(n_ev),
            "conv_rate": len(ok) / len(grp),
            "mu_bias": float(np.nanmedian(mu_pull)),
            "mu_spread": half68(mu_pull),
            "width_bias": float(np.nanmedian(w_pull)),
            "width_spread": half68(w_pull),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    print(f"Reading {args.old} ...")
    old = pd.read_csv(args.old)
    print(f"Reading {args.new} ...")
    new = pd.read_csv(args.new)
    s_old = cell_stats(old)
    s_new = cell_stats(new)

    print("\n" + "=" * 90)
    print(f"Bulk medians across 369 masses per channel, N={args.n_events}")
    print("=" * 90)
    print(f"{'ch':>4s} {'run':>10s}  {'µ_bias':>7s} {'µ_spr':>7s}  {'w_bias':>7s} {'w_spr':>7s}  "
          f"{'|µ_spr-1|':>10s} {'|w_spr-1|':>10s}")
    for ch in ["ee", "mumu"]:
        for label, s in [(args.label_old, s_old), (args.label_new, s_new)]:
            sub = s[(s.channel == ch) & (s.n_events == args.n_events)]
            mu_b = float(np.nanmedian(sub.mu_bias))
            mu_s = float(np.nanmedian(sub.mu_spread))
            w_b = float(np.nanmedian(sub.width_bias))
            w_s = float(np.nanmedian(sub.width_spread))
            print(f"{ch:>4s} {label:>10s}  {mu_b:>+7.2f} {mu_s:>7.2f}  "
                  f"{w_b:>+7.2f} {w_s:>7.2f}  {abs(mu_s-1):>10.2f} {abs(w_s-1):>10.2f}")

    print("\n" + "=" * 90)
    print(f"Fraction of masses with |spread − 1| ≤ tol at N={args.n_events}")
    print("=" * 90)
    print(f"{'ch':>4s} {'run':>10s}  "
          f"{'µ_spr ±0.1':>11s} {'µ_spr ±0.2':>11s}  "
          f"{'w_spr ±0.1':>11s} {'w_spr ±0.2':>11s}")
    for ch in ["ee", "mumu"]:
        for label, s in [(args.label_old, s_old), (args.label_new, s_new)]:
            sub = s[(s.channel == ch) & (s.n_events == args.n_events)]
            n = len(sub)
            f_mu_10 = ((sub.mu_spread - 1).abs() <= 0.1).sum() / n
            f_mu_20 = ((sub.mu_spread - 1).abs() <= 0.2).sum() / n
            f_w_10 = ((sub.width_spread - 1).abs() <= 0.1).sum() / n
            f_w_20 = ((sub.width_spread - 1).abs() <= 0.2).sum() / n
            print(f"{ch:>4s} {label:>10s}  "
                  f"{100*f_mu_10:>10.0f}% {100*f_mu_20:>10.0f}%  "
                  f"{100*f_w_10:>10.0f}% {100*f_w_20:>10.0f}%")

    print("\n" + "=" * 90)
    print(f"Spread/bias ranges at N={args.n_events}")
    print("=" * 90)
    print(f"{'ch':>4s} {'run':>10s}  "
          f"{'µ_spr range':>16s}  {'w_spr range':>16s}  "
          f"{'µ_bias range':>17s}  {'w_bias range':>17s}")
    for ch in ["ee", "mumu"]:
        for label, s in [(args.label_old, s_old), (args.label_new, s_new)]:
            sub = s[(s.channel == ch) & (s.n_events == args.n_events)]
            ms_lo, ms_hi = sub.mu_spread.min(), sub.mu_spread.max()
            ws_lo, ws_hi = sub.width_spread.min(), sub.width_spread.max()
            mb_lo, mb_hi = sub.mu_bias.min(), sub.mu_bias.max()
            wb_lo, wb_hi = sub.width_bias.min(), sub.width_bias.max()
            print(f"{ch:>4s} {label:>10s}  "
                  f"[{ms_lo:.2f}, {ms_hi:.2f}]  "
                  f"[{ws_lo:.2f}, {ws_hi:.2f}]  "
                  f"[{mb_lo:+.2f}, {mb_hi:+.2f}]  "
                  f"[{wb_lo:+.2f}, {wb_hi:+.2f}]")


if __name__ == "__main__":
    main()
