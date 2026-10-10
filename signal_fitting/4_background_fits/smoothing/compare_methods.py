#!/usr/bin/env python3
"""Stage 4 / smoothing -- truth-free comparison of the smoothing methods.

Reads run2/ee_resolved/smoothed_<method>.{root,json} (and scan/ for the
hyperparameter variants) written by smooth_background.py and produces, all at
the card binning:

  pull_stats.csv + plots/pulls_<method>.*
      per component and for the sum: the per-bin pull (raw - smooth) / sigma
      on [800, last populated bin], with two sigma conventions --
      Neyman (sqrt of the raw sumw2 of that bin) and Pearson
      (sigma^2 = w_i * smooth, w_i the bin's mean MC weight var/v, the local
      mean weight where the bin is empty) -- mean, RMS,
      chi2/n, a linear trend (ROOT pol1 fit of pull vs m), the number of sign
      runs and the bins with |pull| > 3. A good smoother has Pearson pulls
      ~ N(0,1) with no trend.
  window_yields.csv + plots/window_yields.*
      the background yield in EVERY production window (1000-6000, the k5
      50 GeV windows of the float regime from ../../8_combine_limits/
      optimization/inputs, the k3 100 GeV windows of the anchored regimes
      from ../../8_combine_limits/baseline/run2/inputs), raw +- sqrt(sumw2)
      vs each smoother, plus the production anchor members' B_env (the
      transported anchor fit) from production/cards/ee_resolved/anchors.json.
  smoothing_systematic.csv + plots/smoothing_systematic.*
      per (window, method): the largest relative change of the window yield
      across the scan/ hyperparameter variants, and the between-method spread
      of the defaults.
  toy_closure_<truth>.csv (--toys N, --toy-truth kernel param)
      self-consistency toys: truth = a method's default expectation; each
      bin is fluctuated as w_i * Poisson(s_i / w_i) with the raw per-bin MC
      weight w_i (TRandom3); every smoother is re-run per toy; bias and RMS
      of the window yields vs the truth. Measures each method's variance /
      bias under a smooth truth -- not a proof of the truth.

Split-sample / jackknife checks are NOT possible here: the inputs are
lumi-scaled histograms with sumw2 only (no per-event weights or ids).

  PYTHONPATH=/opt/homebrew/lib/root ../../../.venv-fitting/bin/python compare_methods.py -v --toys 200
"""
from __future__ import annotations

import argparse
import array
import csv
import json
import logging
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "8_combine_limits" / "production"))

import ROOT                                                              # noqa: E402
import smooth_lib as sl                                                  # noqa: E402
from smooth_lib import (COMP_LABEL, METHODS, METHOD_COLOR, METHOD_LABEL,   # noqa: E402
                        centers_of, rebin)
from smooth_background import COMP_COLOR, XMIN, XMAX, _cms, _save         # noqa: E402
from card_sb_fit import rebin_snap                                       # noqa: E402
from make_refined_workspaces import (MEMBERS, PIVOT, member_window_norm,  # noqa: E402
                                     snap_range)
from wrplotter.cli_utils import setup_logging                            # noqa: E402
from wrplotter.config import load_lumi                                   # noqa: E402
from wrplotter.plotting_helpers import custom_log_formatter              # noqa: E402

logger = logging.getLogger("compare_methods")

SIGFIT = HERE.parents[1]
OPT_INPUTS = SIGFIT / "8_combine_limits" / "optimization" / "inputs"
BASE_INPUTS = SIGFIT / "8_combine_limits" / "baseline" / "run2" / "inputs"
ANCHORS = SIGFIT / "8_combine_limits" / "production" / "cards" / "ee_resolved" / "anchors.json"
FLOAT_MIN, FLOAT_MAX, FLOAT_VAR = 1400, 3200, "k5_bw50"
PROD_MASSES = list(range(1000, 6001, 200))


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, default=HERE / "run2" / "ee_resolved")
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--bw", type=float, default=50.0, help="pull bin width (card)")
    p.add_argument("--toys", type=int, default=0)
    p.add_argument("--toy-truth", nargs="+", default=["kernel", "param"],
                   choices=METHODS,
                   help="which default expectation plays the truth in the "
                        "self-consistency toys (kernel: dies with the MC; "
                        "param: has a tail everywhere)")
    p.add_argument("--include-diag", action="store_true",
                   help="count the diag_* scan variants (known-biased fit "
                        "statistics) in the smoothing systematic")
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def read_smoothed(path):
    """{comp: (smooth 10 GeV, raw values, raw variances)}, edges, json meta."""
    meta = json.loads(path.with_suffix(".json").read_text())
    f = uproot.open(path)
    out, edges = {}, None
    for comp in meta["components"]:
        hs, hr = f[comp], f[f"{comp}_raw"]
        edges = hs.axes[0].edges()
        out[comp] = (hs.values(), hr.values(), hr.variances())
    return out, edges, meta


def production_windows():
    """{mass: (regime, bw, lo, hi)} exactly as the production cards use."""
    opt = json.loads((OPT_INPUTS / "ee_resolved.json").read_text())["masses"]
    base = json.loads((BASE_INPUTS / "ee_resolved.json").read_text())["masses"]
    win = {}
    for m in PROD_MASSES:
        if FLOAT_MIN <= m <= FLOAT_MAX:
            v = opt[str(m)]["vars"][FLOAT_VAR]
            win[m] = ("float", v["bw"], v["fit_lo"], v["fit_hi"])
        else:
            b = base[str(m)]
            regime = "anch_low" if m < FLOAT_MIN else "anch_sparse"
            win[m] = (regime, 100.0, b["fit_lo"], b["fit_hi"])
    return win


def window_sum(edges, values, bw, lo, hi):
    _, vals = rebin_snap(edges, values, bw, lo, hi)
    return float(vals.sum())


def anchor_b_env(edges, raw_sum, windows):
    """{mass: {member: B_env}} from the production anchors.json, transported
    into the anchored-regime windows like make_refined_workspaces does (100
    GeV summed-MC histogram, centre-in-range snap)."""
    if not ANCHORS.exists():
        logger.warning("no %s -- B_env columns skipped", ANCHORS)
        return {}
    anchors = json.loads(ANCHORS.read_text())
    e100, v100, _ = rebin(edges, raw_sum, np.zeros_like(raw_sum), 10)
    h = ROOT.TH1D("h_full_cmp", "", len(v100), np.ascontiguousarray(e100, np.float64))
    for i, x in enumerate(v100, start=1):
        h.SetBinContent(i, float(x))
    out = {}
    for m, (regime, bw, lo, hi) in windows.items():
        i0, i1 = snap_range(h, lo, hi)
        out[m] = {mb: member_window_norm(a, h, i0, i1) for mb, a in anchors.items()}
    return out


# ---------------------------------------------------------------------------
# 1. pulls
# ---------------------------------------------------------------------------

def pull_stats(edges, smooth, raw_v, raw_var, bw):
    """Per-bin pulls on [XMIN, last populated] at width bw, both conventions,
    plus trend / runs / outliers. Returns (row dict, pulls_pearson, centres)."""
    f = int(round(bw / sl.NATIVE_BW))
    e5, v5, var5 = rebin(edges, raw_v, raw_var, f)
    _, s5, _ = rebin(edges, smooth, np.zeros_like(smooth), f)
    c5 = centers_of(e5)
    m_last = sl.last_populated(edges, raw_v)
    w_loc = sl.bin_weight(e5, v5, var5)          # per-bin MC weight
    sel = (c5 >= XMIN) & (c5 <= m_last)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_ney = np.where(sel & (var5 > 0), (v5 - s5) / np.sqrt(var5), np.nan)
        p_pea = np.where(sel & (s5 > 0) & np.isfinite(w_loc),
                         (v5 - s5) / np.sqrt(s5 * w_loc), np.nan)

    def stats(p):
        x = p[np.isfinite(p)]
        if x.size < 3:
            return dict(n=int(x.size), mean=float("nan"), rms=float("nan"),
                        chi2_n=float("nan"))
        a = array.array("d", x)
        return dict(n=int(x.size), mean=float(ROOT.TMath.Mean(len(x), a)),
                    rms=float(ROOT.TMath.RMS(len(x), a)),
                    chi2_n=float(np.mean(x * x)))

    sn, sp = stats(p_ney), stats(p_pea)
    ok = np.isfinite(p_pea)
    trend = dict(slope=float("nan"), slope_err=float("nan"), pull=float("nan"))
    if ok.sum() >= 4:
        g = ROOT.TGraph(int(ok.sum()), array.array("d", (c5[ok] - 2000.0) / 1000.0),
                        array.array("d", p_pea[ok]))
        r = g.Fit("pol1", "SQN")
        if int(r) == 0:
            se = r.ParError(1)
            trend = dict(slope=r.Parameter(1), slope_err=se,
                         pull=(r.Parameter(1) / se) if se > 0 else float("nan"))
    sgn = np.sign(p_pea[ok])
    n_runs = int(1 + np.sum(sgn[1:] != sgn[:-1])) if ok.sum() else 0
    big = [(float(c5[i]), float(p_pea[i])) for i in np.nonzero(ok & (np.abs(p_pea) > 3))[0]]
    row = dict(n_bins=sp["n"], m_last=m_last,
               mean_neyman=sn["mean"], rms_neyman=sn["rms"], chi2n_neyman=sn["chi2_n"],
               mean_pearson=sp["mean"], rms_pearson=sp["rms"], chi2n_pearson=sp["chi2_n"],
               trend_slope=trend["slope"], trend_slope_err=trend["slope_err"],
               trend_pull=trend["pull"], n_runs=n_runs,
               n_runs_expected=(ok.sum() + 1) / 2.0, n_abs_gt3=len(big),
               bins_abs_gt3=" ".join(f"{m:.0f}:{p:+.1f}" for m, p in big))
    return row, p_pea, c5


def plot_pulls(method, per_comp, *, lumi, com, out):
    """per_comp: {comp: (centres, pulls, row)} -> one panel per component."""
    names = list(per_comp)
    hep.style.use("CMS")
    fig, axes = plt.subplots(len(names), 1, sharex=True,
                             figsize=(10, 2.2 * len(names) + 1.5))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        c, p, row = per_comp[name]
        ok = np.isfinite(p)
        ax.bar(c[ok], p[ok], width=45, color=COMP_COLOR[name], alpha=0.85)
        for y in (-2, 2):
            ax.axhline(y, color="gray", linestyle=":", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        if np.isfinite(row["trend_slope"]):
            xx = np.array([XMIN, row["m_last"]])
            ax.plot(xx, row["mean_pearson"] + row["trend_slope"] * 0
                    + row["trend_slope"] * ((xx - 2000.0) / 1000.0), color="black",
                    linewidth=1.0, linestyle="--")
        ax.set_ylim(-5, 5)
        ax.set_ylabel("pull", fontsize=13)
        ax.text(0.02, 0.92,
                f"{COMP_LABEL[name]}:  mean {row['mean_pearson']:+.2f}, "
                f"RMS {row['rms_pearson']:.2f}, trend {row['trend_pull']:+.1f}$\\sigma$, "
                f"|pull|>3: {row['n_abs_gt3']}", transform=ax.transAxes,
                fontsize=11, va="top")
    axes[-1].text(0.98, 0.06, f"pull = (raw $-$ {method}) / "
                  f"$\\sqrt{{w_i\\,\\mathrm{{smooth}}}}$  (Pearson, $w_i$ = bin MC weight)",
                  transform=axes[-1].transAxes, fontsize=11, ha="right", va="bottom")
    axes[-1].set_xlabel(r"$m_{\ell\ell jj}$ [GeV]")
    axes[-1].set_xlim(XMIN, XMAX)
    _cms(axes[0], lumi, com, fontsize=13)
    _save(fig, out)


# ---------------------------------------------------------------------------
# 2./3. window yields and the hyperparameter systematic
# ---------------------------------------------------------------------------

def plot_window_yields(rows, methods, b_env, *, lumi, com, out):
    ms = np.array([r["mass"] for r in rows], float)
    braw = np.array([r["B_raw"] for r in rows])
    berr = np.array([r["B_raw_err"] for r in rows])
    hep.style.use("CMS")
    fig, (ax, axr) = plt.subplots(2, 1, sharex=True, figsize=(10, 10),
                                  gridspec_kw={"height_ratios": [2, 1.3],
                                               "hspace": 0.06})
    ax.errorbar(ms, braw, yerr=berr, fmt="o", color="black", markersize=5,
                label="raw MC $\\pm\\sqrt{\\Sigma w^2}$", zorder=5)
    for m in methods:
        b = np.array([r[f"B_{m}"] for r in rows])
        ax.plot(ms, b, color=METHOD_COLOR[m], linewidth=1.6, marker=".",
                label=METHOD_LABEL[m])
        with np.errstate(divide="ignore", invalid="ignore"):
            axr.plot(ms, b / braw, color=METHOD_COLOR[m], linewidth=1.6, marker=".")
    if b_env:
        be = np.array([b_env.get(int(r["mass"]), {}).get("central", np.nan)
                       for r in rows])
        anch = np.array([r["regime"] != "float" for r in rows])
        ax.plot(ms[anch], be[anch], "s", color="gray", markersize=5,
                markerfacecolor="none", label="production anchor $B_{env}$ (central)")
        with np.errstate(divide="ignore", invalid="ignore"):
            axr.plot(ms[anch], (be / braw)[anch], "s", color="gray", markersize=5,
                     markerfacecolor="none")
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = berr / braw
    axr.fill_between(ms, 1 - rel, 1 + rel, color="gray", alpha=0.25, step="mid",
                     label="raw MC stat.")
    axr.axhline(1, color="black", linewidth=0.8)
    for x in (FLOAT_MIN - 100, FLOAT_MAX + 100):
        for a in (ax, axr):
            a.axvline(x, color="gray", linestyle=":", linewidth=0.8)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylabel("background in the window")
    ax.text(0.04, 0.05, "ee resolved SR, production windows\n"
            "anch_low | float (k5, 50 GeV) | anch_sparse", transform=ax.transAxes,
            fontsize=12, va="bottom")
    ax.legend(loc="upper right", fontsize=11)
    axr.set_ylim(0, 2.5)
    axr.set_ylabel("smooth / raw", fontsize=16)
    axr.set_xlabel(r"$m_{W_R}$ [GeV]")
    axr.legend(loc="upper left", fontsize=11)
    _cms(ax, lumi, com)
    _save(fig, out)


def plot_systematic(sys_rows, methods, *, lumi, com, out):
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 7))
    for m in methods:
        rr = [r for r in sys_rows if r["method"] == m]
        ax.plot([r["mass"] for r in rr], [100 * r["max_rel_change"] for r in rr],
                color=METHOD_COLOR[m], marker="o", markersize=4, linewidth=1.5,
                label=f"{m}: max |ΔB|/B over its variants")
    rr = [r for r in sys_rows if r["method"] == methods[0]]
    ax.plot([r["mass"] for r in rr], [100 * r["between_method_spread"] for r in rr],
            color="black", linestyle="--", linewidth=1.5,
            label="between-method spread of the defaults")
    ax.set_yscale("log")
    ax.set_ylim(0.1, 300)
    ax.set_xlim(XMIN, XMAX + 200)
    ax.set_xlabel(r"$m_{W_R}$ [GeV]")
    ax.set_ylabel("window-yield variation [%]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=11)
    _cms(ax, lumi, com)
    _save(fig, out)


# ---------------------------------------------------------------------------
# 4. self-consistency toys
# ---------------------------------------------------------------------------

def toy_closure(truth_by_comp, raw_by_comp, edges, kwargs_by_method, windows,
                ntoys, seed):
    """truth_by_comp: {comp: s (10 GeV)}; raw_by_comp: {comp: (v, var)}."""
    c = centers_of(edges)
    rng = ROOT.TRandom3(seed)
    comps = list(truth_by_comp)
    # per-bin MC weight: var/v where populated, else the local mean weight
    wts = {}
    for name in comps:
        v, var = raw_by_comp[name]
        w_loc = sl.local_weight(edges, v, var)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where((v > 0) & (var > 0), var / v, w_loc)
        w = np.where(np.isfinite(w) & (w > 0), w, np.nanmedian(w_loc[np.isfinite(w_loc)]))
        wts[name] = w
    methods = list(kwargs_by_method)
    b_truth = {m: window_sum(edges, sum(truth_by_comp.values()), bw, lo, hi)
               for m, (_, bw, lo, hi) in windows.items()}
    acc = {mth: {m: [] for m in windows} for mth in methods + ["raw"]}
    n_fail = {mth: 0 for mth in methods}
    for it in range(ntoys):
        toy = {}
        for name in comps:
            s, w = truth_by_comp[name], wts[name]
            n = np.array([rng.Poisson(x) if x > 0 else 0
                          for x in np.where(c >= XMIN, s / w, 0.0)], float)
            toy[name] = (w * n, w * w * n)
        raw_sum = sum(t[0] for t in toy.values())
        for m, (_, bw, lo, hi) in windows.items():
            acc["raw"][m].append(window_sum(edges, raw_sum, bw, lo, hi))
        for mth in methods:
            try:
                sm = sum(sl.SMOOTHERS[mth](edges, t[0], t[1], **kwargs_by_method[mth])[0]
                         for t in toy.values())
            except Exception as exc:                       # noqa: BLE001
                n_fail[mth] += 1
                logger.debug("toy %d %s failed: %s", it, mth, exc)
                continue
            for m, (_, bw, lo, hi) in windows.items():
                acc[mth][m].append(window_sum(edges, sm, bw, lo, hi))
        if (it + 1) % 25 == 0:
            logger.info("  toys %d/%d", it + 1, ntoys)
    rows = []
    for mth in ["raw"] + methods:
        for m in windows:
            x = acc[mth][m]
            bt = b_truth[m]
            if len(x) < 3 or bt <= 0:
                continue
            a = array.array("d", x)
            mean = float(ROOT.TMath.Mean(len(x), a))
            rms = float(ROOT.TMath.RMS(len(x), a))
            rows.append(dict(method=mth, mass=m, regime=windows[m][0], B_truth=bt,
                             n_toys=len(x), mean=mean, rms=rms,
                             bias=(mean - bt) / bt, rel_rms=rms / bt,
                             bias_err=rms / math.sqrt(len(x)) / bt))
    for mth, nf in n_fail.items():
        if nf:
            logger.warning("  %s: %d/%d toys failed", mth, nf, ntoys)
    return rows


# ---------------------------------------------------------------------------

def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    logger.info("  wrote %s", path)


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    info = load_lumi(sl.ERA)
    lumi, com = info["lumi"], info.get("com", 13.0)
    ind = args.input_dir
    pdir = ind / "plots"

    data, edges, metas = {}, None, {}
    for m in args.methods:
        data[m], edges, metas[m] = read_smoothed(ind / f"smoothed_{m}.root")
    comps = list(next(iter(data.values())))
    raw = {n: (data[args.methods[0]][n][1], data[args.methods[0]][n][2]) for n in comps}
    raw_sum = sum(r[0] for r in raw.values())
    raw_var_sum = sum(r[1] for r in raw.values())
    windows = production_windows()
    b_env = anchor_b_env(edges, raw_sum, windows)

    # 1. pulls -------------------------------------------------------------
    pull_rows = []
    for m in args.methods:
        per_comp = {}
        for n in comps + ["sum"]:
            if n == "sum":
                s, v, var = sum(data[m][k][0] for k in comps), raw_sum, raw_var_sum
            else:
                s, v, var = data[m][n]
            row, p, c = pull_stats(edges, s, v, var, args.bw)
            pull_rows.append({"method": m, "component": n, **row})
            per_comp[n] = (c, p, row)
            logger.info("  pulls %-6s %-9s n=%3d Pearson mean %+.2f rms %.2f  "
                        "Neyman mean %+.2f rms %.2f  trend %+.1fs  |p|>3: %d",
                        m, n, row["n_bins"], row["mean_pearson"], row["rms_pearson"],
                        row["mean_neyman"], row["rms_neyman"], row["trend_pull"],
                        row["n_abs_gt3"])
        plot_pulls(m, per_comp, lumi=lumi, com=com, out=pdir / f"pulls_{m}")
    write_csv(ind / "pull_stats.csv", pull_rows)

    # 2. window yields -----------------------------------------------------
    wy_rows = []
    for mass, (regime, bw, lo, hi) in windows.items():
        b_raw = window_sum(edges, raw_sum, bw, lo, hi)
        b_err = math.sqrt(max(window_sum(edges, raw_var_sum, bw, lo, hi), 0.0))
        row = dict(mass=mass, regime=regime, bw=bw, lo=lo, hi=hi,
                   B_raw=b_raw, B_raw_err=b_err,
                   neff_raw=(b_raw ** 2 / b_err ** 2) if b_err > 0 else float("nan"))
        for m in args.methods:
            b = window_sum(edges, sum(data[m][k][0] for k in comps), bw, lo, hi)
            row[f"B_{m}"] = b
            row[f"ratio_{m}"] = (b / b_raw) if b_raw > 0 else float("nan")
            row[f"pull_{m}"] = ((b_raw - b) / b_err) if b_err > 0 else float("nan")
        for mb, val in b_env.get(mass, {}).items():
            row[f"B_env_{mb}"] = val
        wy_rows.append(row)
        logger.info("  window m=%d %-11s [%.0f,%.0f] B_raw=%.3f+-%.3f  %s%s", mass,
                    regime, lo, hi, b_raw, b_err,
                    "  ".join(f"{m}={row[f'B_{m}']:.3f}" for m in args.methods),
                    f"  B_env={b_env[mass]['central']:.3f}"
                    if mass in b_env and regime != "float" else "")
    write_csv(ind / "window_yields.csv", wy_rows)
    plot_window_yields(wy_rows, args.methods, b_env, lumi=lumi, com=com,
                       out=pdir / "window_yields")

    # 3. hyperparameter systematic ------------------------------------------
    sdir = ind / "scan"
    sys_rows = []
    if sdir.exists():
        variants = {m: {} for m in args.methods}
        for p in sorted(sdir.glob("*.root")):
            m, vname = p.stem.split("__", 1)
            if vname.startswith("diag_") and not args.include_diag:
                continue
            if m in variants:
                d, _, _ = read_smoothed(p)
                variants[m][vname] = sum(d[k][0] for k in comps)
        for mass, (regime, bw, lo, hi) in windows.items():
            defaults = {m: window_sum(edges, sum(data[m][k][0] for k in comps),
                                      bw, lo, hi) for m in args.methods}
            vals = [x for x in defaults.values() if x > 0]
            spread = ((max(vals) - min(vals)) / (sum(vals) / len(vals))
                      if len(vals) > 1 else float("nan"))
            for m in args.methods:
                b0 = defaults[m]
                worst, wname = 0.0, ""
                for vname, sm in variants[m].items():
                    b = window_sum(edges, sm, bw, lo, hi)
                    if b0 > 0 and abs(b - b0) / b0 > worst:
                        worst, wname = abs(b - b0) / b0, vname
                sys_rows.append(dict(mass=mass, regime=regime, method=m, B_default=b0,
                                     n_variants=len(variants[m]),
                                     max_rel_change=worst, worst_variant=wname,
                                     between_method_spread=spread))
        write_csv(ind / "smoothing_systematic.csv", sys_rows)
        plot_systematic(sys_rows, args.methods, lumi=lumi, com=com,
                        out=pdir / "smoothing_systematic")
        for m in args.methods:
            rr = [r for r in sys_rows if r["method"] == m]
            logger.info("  systematic %-6s (%d variants): max|dB|/B = %s", m,
                        rr[0]["n_variants"],
                        " ".join(f"{r['mass']}:{100 * r['max_rel_change']:.0f}%"
                                 for r in rr))
    else:
        logger.warning("no scan/ directory -- run smooth_background.py --scan first")

    # 4. toys ------------------------------------------------------------------
    for truth_m in (args.toy_truth if args.toys > 0 else []):
        if truth_m not in data:
            logger.error("toy truth %s not among the loaded methods", truth_m)
            continue
        truth = {n: data[truth_m][n][0] for n in comps}
        kw = {m: dict(metas[m]["kwargs"]) for m in args.methods}
        for m in kw:                            # json lists -> tuples
            if "families" in kw[m]:
                kw[m]["families"] = tuple(kw[m]["families"])
        logger.info("  self-consistency toys: %d toys, truth = %s default",
                    args.toys, truth_m)
        rows = toy_closure(truth, raw, edges, kw, windows, args.toys, args.seed)
        for r in rows:
            r["truth"] = truth_m
        write_csv(ind / f"toy_closure_{truth_m}.csv", rows)
        for mth in ["raw"] + args.methods:
            rr = [r for r in rows if r["method"] == mth]
            logger.info("  toys[%s] %-6s bias%%: %s", truth_m, mth,
                        " ".join(f"{r['mass']}:{100 * r['bias']:+.0f}+-{100 * r['rel_rms']:.0f}"
                                 for r in rr))


if __name__ == "__main__":
    main()
