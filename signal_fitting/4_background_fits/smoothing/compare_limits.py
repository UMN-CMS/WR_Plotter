#!/usr/bin/env python3
"""Stage 4 / smoothing -- what the smoothed background does to the limit.

Joins, per method, the combine runs made by run_smooth_limits.sh (collected
with 8_combine_limits/production/collect_refined.py --table-suffix _<label>
into run2/ee_resolved/refined_limit_table_ee_resolved_<label>.csv) with the production reference
(8_combine_limits/production/refined_limit_table_ee_resolved.csv), the
card-model spurious-signal toys (6_spurious_signal_toys/card_model_toys.py
--bkg-hist-file, run2/ee_resolved/toys_<method>/ vs toys_raw/), the window
yields (compare_methods.py) and the anchored cards' B_env (parsed from the
card comments), and writes

  limit_ratio.csv          one row per (mass, label)
  plots/limit_ratio.*      three panels vs m_WR: expected-limit median ratio
                           smooth/raw (asymptotic medians on both sides, so
                           the anchored tail compares like with like), the
                           window background (raw, smooth, anchor B_env raw
                           and re-fit), and the spurious signal (Asimov N_sp
                           and toy mean, float regime)

Labels are the run_smooth_limits.sh labels: kernel, hybrid, param, kde,
<method>_anchraw (anchors fit on the raw MC: data_obs effect only) and
<method>_normwin (anchored cards normalized to the smoothed window yield
instead of the transported anchor B_env: the 'mcmax' test).

  PYTHONPATH=/opt/homebrew/lib/root ../../../.venv-fitting/bin/python compare_limits.py -v
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from smooth_lib import METHOD_COLOR, METHOD_LABEL                        # noqa: E402
METHOD_COLOR = {**METHOD_COLOR, "raw": "#555555"}
METHOD_LABEL = {**METHOD_LABEL, "raw": "raw MC through the same path (regression)"}
from smooth_background import _cms, _save                                # noqa: E402
from wrplotter.cli_utils import setup_logging                            # noqa: E402
from wrplotter.config import load_lumi                                   # noqa: E402
from wrplotter.plotting_helpers import custom_log_formatter              # noqa: E402
import smooth_lib as sl                                                  # noqa: E402

logger = logging.getLogger("compare_limits")

SIGFIT = HERE.parents[1]
PROD = SIGFIT / "8_combine_limits" / "production"
RAW_TOYS = SIGFIT / "6_spurious_signal_toys" / "run2" / "card"
BAND = ["m2s", "m1s", "med", "p1s", "p2s"]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=HERE / "run2" / "ee_resolved")
    p.add_argument("--labels", nargs="+",
                   default=["raw", "kernel", "hybrid", "param", "kde", "hybrid_anchraw"])
    p.add_argument("--tag", default="ee_resolved")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def read_limit_table(path):
    """{mass: row} with floats; empty if the file is missing."""
    if not path.exists():
        logger.warning("missing %s", path)
        return {}
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            m = int(float(r["mWR"]))
            out[m] = {k: (float(v) if re.fullmatch(r"-?[\d.]+(e-?\d+)?|nan", v or "")
                          else v) for k, v in r.items()}
    return out


def read_toys(path):
    """{mass: row} of the card_float rows."""
    if not path.exists():
        return {}
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["function"] != "card_float":
                continue
            m = int(float(r["mWR"]))
            out[m] = {k: (float(v) if v not in ("", None) else float("nan"))
                      for k, v in r.items() if k in ("B_window", "nsp_asimov",
                                                     "nsp_asimov_err", "mean_Nsp",
                                                     "rms_Nsp", "n_ok")}
    return out


def read_card_benv(cards_dir):
    """{mass: (B_env central, window MC)} from the anchored cards' comment
    '# background anchored: member=central B_env=X (window MC Y)'."""
    out = {}
    for card in sorted(cards_dir.glob("card_anchored_m*.txt")):
        m = int(re.search(r"_m(\d+)\.txt", card.name).group(1))
        mt = re.search(r"member=central B_env=([\d.eE+-]+) \(window MC ([\d.eE+-]+)\)",
                       card.read_text())
        if mt:
            out[m] = (float(mt.group(1)), float(mt.group(2)))
    return out


def read_window_yields(path):
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {int(float(r["mass"])): {k: float(v) for k, v in r.items()
                                        if k not in ("regime",)}
                for r in csv.DictReader(fh)}


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    info = load_lumi(sl.ERA)
    lumi, com = info["lumi"], info.get("com", 13.0)
    ind = args.input_dir

    ref = read_limit_table(PROD / f"refined_limit_table_{args.tag}.csv")
    ref_benv = read_card_benv(PROD / "cards" / args.tag)
    raw_toys = read_toys(ind / "toys_raw" / f"card_toy_table_{args.tag}.csv") \
        or read_toys(RAW_TOYS / f"card_toy_table_{args.tag}.csv")
    wy = read_window_yields(ind / "window_yields.csv")

    rows = []
    for label in args.labels:
        method = label.replace("_anchraw", "").replace("_normwin", "")
        lim = read_limit_table(ind / f"refined_limit_table_{args.tag}_{label}.csv")
        benv = read_card_benv(ind / f"cards_{label}" / args.tag)
        toys = read_toys(ind / f"toys_{method}" / f"card_toy_table_{args.tag}.csv")
        for m in sorted(lim):
            r, r0 = lim[m], ref.get(m)
            if r0 is None:
                continue
            row = dict(mass=m, label=label, method=method, regime=r["regime"],
                       asym_med_raw=r0["asym_med_fb"], asym_med_smooth=r["asym_med_fb"],
                       ratio_med=(r["asym_med_fb"] / r0["asym_med_fb"]
                                  if r0["asym_med_fb"] > 0 else float("nan")),
                       model_spread_raw=r0["model_spread_fb"],
                       model_spread_smooth=r["model_spread_fb"])
            for b in BAND:                      # band edges (asymptotic or hybrid)
                row[f"fb_{b}_raw"], row[f"fb_{b}_smooth"] = r0[f"fb_{b}"], r[f"fb_{b}"]
            if r["regime"] == "float":
                for b in ("m1s", "p1s"):
                    row[f"ratio_{b}"] = (r[f"fb_{b}"] / r0[f"fb_{b}"]
                                         if r0[f"fb_{b}"] > 0 else float("nan"))
            if m in wy:
                row["B_window_raw"] = wy[m]["B_raw"]
                row["B_window_raw_err"] = wy[m]["B_raw_err"]
                row["B_window_smooth"] = wy[m].get(f"B_{method}", float("nan"))
            if m in ref_benv:
                row["B_env_raw"], row["window_MC_raw"] = ref_benv[m]
            if m in benv:
                row["B_env_smooth"], row["window_MC_smooth"] = benv[m]
            if m in raw_toys and m in toys:
                for k in ("nsp_asimov", "mean_Nsp", "rms_Nsp", "B_window"):
                    row[f"{k}_toyraw"] = raw_toys[m][k]
                    row[f"{k}_toysmooth"] = toys[m][k]
            rows.append(row)
            logger.info("  %-14s m=%d %-11s median %.3f -> %.3f (x%.3f)%s%s", label, m,
                        r["regime"], r0["asym_med_fb"], r["asym_med_fb"],
                        row["ratio_med"],
                        f"  B_env {row['B_env_raw']:.3f} -> {row['B_env_smooth']:.3f}"
                        if "B_env_smooth" in row else "",
                        f"  Nsp {row['nsp_asimov_toyraw']:+.1f} -> "
                        f"{row['nsp_asimov_toysmooth']:+.1f}"
                        if "nsp_asimov_toysmooth" in row else "")
    if not rows:
        logger.error("nothing to compare -- run run_smooth_limits.sh + collect_refined.py")
        sys.exit(1)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(ind / "limit_ratio.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    logger.info("  wrote %s", ind / "limit_ratio.csv")

    # ---- plot -----------------------------------------------------------
    hep.style.use("CMS")
    fig, (a1, a2, a3) = plt.subplots(3, 1, sharex=True, figsize=(10, 14),
                                     gridspec_kw={"hspace": 0.08,
                                                  "height_ratios": [1.2, 1.2, 1]})
    style = {"anchraw": {"linestyle": "--", "marker": "s", "markerfacecolor": "none"}}
    for label in args.labels:
        rr = [r for r in rows if r["label"] == label]
        if not rr:
            continue
        method = rr[0]["method"]
        st = style["anchraw"] if label != method else {"marker": "o"}
        ms = [r["mass"] for r in rr]
        a1.plot(ms, [r["ratio_med"] for r in rr], color=METHOD_COLOR[method],
                linewidth=1.6, markersize=5,
                label=f"{METHOD_LABEL[method]}"
                + (" (anchors on raw MC)" if label.endswith("_anchraw") else "")
                + (" (anchored norm = window yield)" if label.endswith("_normwin") else ""),
                **st)
        if label == method:
            a2.plot(ms, [r.get("B_window_smooth", np.nan) for r in rr],
                    color=METHOD_COLOR[method], linewidth=1.4, marker="o", markersize=4)
            be = [(r["mass"], r["B_env_smooth"]) for r in rr if "B_env_smooth" in r]
            if be:
                a2.plot([x for x, _ in be], [y for _, y in be], color=METHOD_COLOR[method],
                        linestyle="none", marker="s", markersize=6, markerfacecolor="none")
            fl = [r for r in rr if "nsp_asimov_toysmooth" in r]
            if fl:
                a3.errorbar([r["mass"] + 15 for r in fl],
                            [r["nsp_asimov_toysmooth"] for r in fl],
                            yerr=[r["rms_Nsp_toysmooth"] for r in fl],
                            color=METHOD_COLOR[method], fmt="o", markersize=5,
                            capsize=2, linewidth=1)
                a3.plot([r["mass"] + 15 for r in fl], [r["mean_Nsp_toysmooth"] for r in fl],
                        color=METHOD_COLOR[method], linestyle="none", marker="_",
                        markersize=12, markeredgewidth=2)
    # raw references
    rr0 = {r["mass"]: r for r in rows}
    ms0 = sorted(rr0)
    a2.errorbar(ms0, [rr0[m].get("B_window_raw", np.nan) for m in ms0],
                yerr=[rr0[m].get("B_window_raw_err", 0) for m in ms0], fmt="o",
                color="black", markersize=5, label="raw MC window (data_obs)", zorder=5)
    be0 = [(m, rr0[m]["B_env_raw"]) for m in ms0 if "B_env_raw" in rr0[m]]
    if be0:
        a2.plot([x for x, _ in be0], [y for _, y in be0], "s", color="black",
                markersize=7, markerfacecolor="none", label="production anchor $B_{env}$")
    fl0 = [rr0[m] for m in ms0 if "nsp_asimov_toyraw" in rr0[m]]
    if fl0:
        a3.errorbar([r["mass"] - 15 for r in fl0], [r["nsp_asimov_toyraw"] for r in fl0],
                    yerr=[r["rms_Nsp_toyraw"] for r in fl0], color="black", fmt="o",
                    markersize=5, capsize=2, linewidth=1, label="raw: Asimov $N_{sp}$ ± toy RMS")
        a3.plot([r["mass"] - 15 for r in fl0], [r["mean_Nsp_toyraw"] for r in fl0],
                color="black", linestyle="none", marker="_", markersize=12,
                markeredgewidth=2, label="toy mean $\\langle N_{sp}\\rangle$")
    a1.axhline(1, color="black", linewidth=0.8)
    a1.set_ylabel("expected median: smooth / raw", fontsize=15)
    a1.set_ylim(0.6, 1.75)
    a1.legend(loc="upper left", fontsize=9, ncol=2)
    a1.text(0.97, 0.05, "run2 2018 ee resolved, asymptotic medians\n"
            "dashed/hollow: anchors on raw MC or anchored norm = window yield",
            transform=a1.transAxes, fontsize=11, ha="right", va="bottom")
    a2.set_yscale("log")
    a2.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    a2.set_ylabel("background in window", fontsize=15)
    a2.legend(loc="upper right", fontsize=10.5)
    a2.text(0.03, 0.05, "lines: smoothed window yield; squares: $B_{env}$ of the "
            "anchor re-fit on the smoothed spectrum", transform=a2.transAxes,
            fontsize=10.5, va="bottom")
    a3.axhline(0, color="black", linewidth=0.8)
    a3.set_ylabel("spurious signal $N_{sp}$ (float cards)", fontsize=14)
    a3.set_xlabel(r"$m_{W_R}$ [GeV]")
    a3.legend(loc="upper left", fontsize=10.5)
    _cms(a1, lumi, com, fontsize=14)
    _save(fig, ind / "plots" / "limit_ratio")

    hdr = f"{'label':<15}{'m':>6} {'regime':<12}{'med raw':>9}{'med sm':>9}{'ratio':>7}" \
          f"{'Benv raw':>10}{'Benv sm':>9}{'Nsp raw':>9}{'Nsp sm':>8}"
    print(hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(f"{r['label']:<15}{r['mass']:>6} {r['regime']:<12}"
              f"{r['asym_med_raw']:>9.3f}{r['asym_med_smooth']:>9.3f}{r['ratio_med']:>7.3f}"
              f"{r.get('B_env_raw', float('nan')):>10.3f}{r.get('B_env_smooth', float('nan')):>9.3f}"
              f"{r.get('nsp_asimov_toyraw', float('nan')):>9.1f}"
              f"{r.get('nsp_asimov_toysmooth', float('nan')):>8.1f}")


if __name__ == "__main__":
    main()
