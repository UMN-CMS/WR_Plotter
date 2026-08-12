#!/usr/bin/env python3
"""Window-optimization, Part B -- sensitivity sigma_UL vs window width k.

Combines the two k-dependent pieces into the expected cross-section limit,

    sigma_UL(m, k) = N_UL(m, k) / (1000 * L[fb^-1] * eff(m, k)),

    * eff(m, k)   from Part A (efficiency_vs_k.py): wider window -> more signal.
    * N_UL(m, k)  = cls_band(0, RMS(N_sp), alpha)[median], the centre-zero CLs
                    median from the Stage-6 toy spurious-signal width at each k
                    (wider window -> more sideband -> smaller RMS ... until the
                    fit destabilizes and RMS blows up).

Both improve with wider k in the reliable region, so sigma_UL FALLS with k up to
an optimum, then the toy distribution stops being reliable (survivor bias at
narrow-high-mass, runaway tail at wide-high-mass) and the limit is no longer
trustworthy. We therefore pick, per mass, the k that minimizes sigma_UL AMONG the
reliable k only (reliability-capped optimum).

Reliability per (mass, k), read straight from the Stage-6 CSV columns:
    converged : n_ok / ntoys      >= --conv-min   (else survivor bias)
    clean tail: RMS / q95(|N_sp|) <= --tail-max    (clean ~0.5; broken >1)

Inputs:
    Part A csv : <out>/efficiency_vs_k_{ch}_{topo}.csv
    Stage-6    : <out>/stage6/k{k:g}/spurious_toy_table_{ch}_{topo}.csv  (per k)

Outputs (per function, namespaced {ch}_{topo}):
    sensitivity_vs_k_{ch}_{topo}.csv        per (mass,fn): k_opt, sigma_UL, gain
    sigma_vs_k/{ch}_{topo}/{fn}.*           sigma_UL vs k per mass, optimum starred
    kopt_vs_mass/{ch}_{topo}/{fn}.*         reliability-capped optimal k vs mass
    reach/{ch}_{topo}/{fn}.*                sigma_UL vs mass: k=3 / k=5 / optimal + theory
    gain_vs_mass/{ch}_{topo}/{fn}.*         sigma_UL(k=3)/sigma_UL(k_opt) vs mass

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                       # repo root
sys.path.insert(0, str(HERE.parents[1] / "4_background_fits"))  # bkg_fit_lib
sys.path.insert(0, str(HERE.parents[1] / "shared"))           # loaders
sys.path.insert(0, str(HERE.parent))                          # xsec_limit / expected_limit

from wrplotter.cli_utils import setup_logging                            # noqa: E402
from wrplotter.config import load_lumi                                   # noqa: E402
from wrplotter.plotting_helpers import custom_log_formatter              # noqa: E402
from bkg_fit_lib import FUNCS, CH_LAB, TOPO_LAB                          # noqa: E402
from measure_fwhm import parse_masses                                    # noqa: E402
from xsec_limit import default_signal_config                            # noqa: E402
from expected_limit import cls_band                                     # noqa: E402

logger = logging.getLogger("sens_vs_k")
KREF = 3.0
FUNC_COL = {"expo": "#1f77b4", "powlaw": "#e42536", "expo2": "#2ca02c"}


def _save(fig, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _cms(ax, com, lumi):
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  lumi=f"{lumi:.1f}", com=com, fontsize=16)


def _logfmt(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))


def load_efficiency(csv_path):
    """{mWR: {k: eff}} and {mWR: signal_tag} from the Part A table."""
    eff, tags = {}, {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            m = float(r["mWR"])
            eff.setdefault(m, {})[float(r["k"])] = float(r["eff"])
            tags[m] = r["signal_tag"]
    return eff, tags


def load_stage6(csv_path, function):
    """{mWR: dict(rms, nsp_asimov, n_ok, ntoys, q95)} for one function, reliable
    rows only kept downstream. Missing/blank RMS rows are skipped."""
    out = {}
    if not csv_path.exists():
        return out
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r["function"] != function or not r["rms_Nsp"]:
                continue
            out[float(r["mWR"])] = {
                "rms": float(r["rms_Nsp"]),
                "mean": float(r["mean_Nsp"]) if r["mean_Nsp"] else float("nan"),
                "n_ok": int(r["n_ok"]), "ntoys": int(r["ntoys"]),
                "q95": float(r["q95_abs_Nsp"]) if r["q95_abs_Nsp"] else float("nan"),
            }
    return out


def reliable(s6, conv_min, tail_max, bias_max):
    """Three failure modes: non-convergence (survivor bias), runaway tail (RMS
    not a faithful width), and mismodeling BIAS -- the wide-window fixed function
    fakes a signal. The bias |mean(N_sp)|/RMS is the spurious-signal acceptance
    (standard threshold ~0.5): it is NOT in the centre-zero band, so a large bias
    means the limit is optimistic even though the band width is 'clean'."""
    conv = s6["n_ok"] / s6["ntoys"] if s6["ntoys"] else 0.0
    tail = s6["rms"] / s6["q95"] if s6["q95"] > 0 else float("inf")
    bias = abs(s6["mean"]) / s6["rms"] if s6["rms"] > 0 else float("inf")
    return conv >= conv_min and tail <= tail_max and bias <= bias_max


def load_theory(cfg_path, tags, bfrac):
    """{mWR: sigma_theory_pb (x bfrac)} from the signal config."""
    with open(cfg_path) as fh:
        cfg = {v["dataset"]: v for v in json.load(fh).values()}
    out = {}
    for mWR, tag in tags.items():
        mwr, mn = parse_masses(tag)
        key = f"WRtoNLtoLLJJ_MWR{mwr}_MN{mn}"
        if key in cfg:
            out[mWR] = bfrac * float(cfg[key]["xsec"])
    return out


# ---------------------------------------------------------------------------
def plot_sigma_vs_k(rows_by_mass, fn, out, *, channel, topology, com, lumi):
    """sigma_UL vs k, one line per mass; reliable points filled, unreliable open,
    the reliability-capped optimum starred."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    masses = sorted(rows_by_mass)
    cmap = plt.cm.viridis(np.linspace(0, 0.92, len(masses)))
    for mWR, c in zip(masses, cmap):
        pts = rows_by_mass[mWR]
        ks = [p["k"] for p in pts]
        sig = [p["sigma_UL"] for p in pts]
        ax.plot(ks, sig, "-", color=c, lw=1.2, zorder=2)
        rel = [(p["k"], p["sigma_UL"]) for p in pts if p["reliable"]]
        unr = [(p["k"], p["sigma_UL"]) for p in pts if not p["reliable"]]
        if rel:
            ax.plot(*zip(*rel), "o", color=c, ms=4, zorder=3)
        if unr:
            ax.plot(*zip(*unr), "o", color=c, ms=4, mfc="none", zorder=3)
        opt = min((p for p in pts if p["reliable"]),
                  key=lambda p: p["sigma_UL"], default=None)
        if opt:
            ax.plot(opt["k"], opt["sigma_UL"], "*", color=c, ms=13,
                    mec="black", mew=0.5, zorder=5)
    _logfmt(ax)
    ax.set_xlabel(r"Window half-width $k$  ($\pm k\sigma$)")
    ax.set_ylabel(r"Expected $\sigma_{\rm UL}$ [pb]")
    ax.text(0.04, 0.30, f"{CH_LAB[channel]}  {TOPO_LAB[topology]}  ({fn})\n"
            r"$m_N=m_{W_R}/2$" "\nfilled=reliable, open=unreliable\n"
            r"$\star$ = reliability-capped optimum",
            transform=ax.transAxes, fontsize=12, va="top")
    _cms(ax, com, lumi)
    _save(fig, out)


def plot_kopt_vs_mass(summary, fn, out, *, channel, topology, com, lumi):
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    ms = sorted(summary)
    ax.plot([m / 1000 for m in ms], [summary[m]["k_opt"] for m in ms], "o-",
            color=FUNC_COL.get(fn, "#1f77b4"), ms=6, lw=1.6)
    ax.axhline(KREF, color="0.6", lw=0.8, ls=":", label=f"current k={KREF:g}")
    ax.set_xlabel(r"$m_{W_R}$ [TeV]")
    ax.set_ylabel(r"Reliability-capped optimal $k$")
    ax.set_ylim(1.8, 5.2)
    ax.text(0.04, 0.12, f"{CH_LAB[channel]}  {TOPO_LAB[topology]}  ({fn})",
            transform=ax.transAxes, fontsize=13, va="bottom")
    ax.legend(loc="upper right", fontsize=12, frameon=False)
    _cms(ax, com, lumi)
    _save(fig, out)


def plot_reach(rows_by_mass, summary, theory, fn, out, *, channel, topology,
               com, lumi):
    """sigma_UL vs mass for fixed k=3, fixed k=5, and the optimal-k envelope,
    with the theory xsec overlaid -- the crossings are the expected reaches."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()

    def curve(kpick):
        xs, ys = [], []
        for m in sorted(rows_by_mass):
            p = next((q for q in rows_by_mass[m]
                      if abs(q["k"] - kpick) < 1e-6 and q["reliable"]), None)
            if p:
                xs.append(m / 1000); ys.append(p["sigma_UL"])
        return xs, ys

    for kpick, col, lab in ((3.0, "#1f77b4", "fixed $k=3$"),
                            (5.0, "#ff7f0e", "fixed $k=5$")):
        xs, ys = curve(kpick)
        if xs:
            ax.plot(xs, ys, "o-", color=col, ms=4, lw=1.4, label=lab)
    xo = [m / 1000 for m in sorted(summary)]
    yo = [summary[m]["sigma_UL_opt"] for m in sorted(summary)]
    ax.plot(xo, yo, "*-", color="#2ca02c", ms=11, lw=1.8, mec="black",
            mew=0.4, label="optimal $k$ (capped)", zorder=5)
    if theory:
        xt = [m / 1000 for m in sorted(theory)]
        yt = [theory[m] for m in sorted(theory)]
        ax.plot(xt, yt, "-", color="#d62728", lw=2.0, label=r"theory $\sigma\times$BR")
    _logfmt(ax)
    ax.set_xlabel(r"$m_{W_R}$ [TeV]")
    ax.set_ylabel(r"Expected $\sigma_{\rm UL}$ [pb]")
    ax.text(0.04, 0.30, f"{CH_LAB[channel]}  {TOPO_LAB[topology]}  ({fn})\n"
            r"$m_N=m_{W_R}/2$", transform=ax.transAxes, fontsize=13, va="top")
    ax.legend(loc="upper right", fontsize=12, frameon=False)
    _cms(ax, com, lumi)
    _save(fig, out)


def plot_gain_vs_mass(summary, fn, out, *, channel, topology, com, lumi):
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    ms = sorted(summary)
    ax.plot([m / 1000 for m in ms], [summary[m]["gain"] for m in ms], "o-",
            color=FUNC_COL.get(fn, "#1f77b4"), ms=6, lw=1.6)
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(r"$m_{W_R}$ [TeV]")
    ax.set_ylabel(r"$\sigma_{\rm UL}(k{=}3)\,/\,\sigma_{\rm UL}(k_{\rm opt})$")
    ax.text(0.04, 0.90, f"{CH_LAB[channel]}  {TOPO_LAB[topology]}  ({fn})\n"
            ">1 = optimized window is tighter",
            transform=ax.transAxes, fontsize=13, va="top")
    _cms(ax, com, lumi)
    _save(fig, out)


def plot_decomposition(rows_by_mass, fn, k_ref, k_tgt, out, *, channel,
                       topology, com, lumi, mass_range=None):
    """Split the limit tightening sigma_UL(k_ref)/sigma_UL(k_tgt) into its two
    multiplicative pieces vs mass, over the trusted region, so it is visible that
    N_UL (background pinning) does most of the work, not the efficiency:
        sigma_UL(k_ref)/sigma_UL(k_tgt) = N_UL(k_ref)/N_UL(k_tgt) x eff(k_tgt)/eff(k_ref)
    Linear-y (restricted range has no high-mass crash); the aggregate log-share of
    each component over the region is stamped on the plot."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    ms, e_fac, n_fac, tot = [], [], [], []
    for mWR in sorted(rows_by_mass):
        if mass_range and not (mass_range[0] <= mWR <= mass_range[1]):
            continue
        pr = {p["k"]: p for p in rows_by_mass[mWR]}
        if k_ref not in pr or k_tgt not in pr:
            continue
        a, b = pr[k_ref], pr[k_tgt]
        ms.append(mWR / 1000)
        e_fac.append(b["eff"] / a["eff"])            # efficiency tightening
        n_fac.append(a["n_ul"] / b["n_ul"])          # N_UL (background) tightening
        tot.append(a["sigma_UL"] / b["sigma_UL"])    # total
    # aggregate log-share of each component over the region
    l_nul = sum(math.log(n) for n in n_fac)
    l_eff = sum(math.log(e) for e in e_fac)
    l_tot = l_nul + l_eff
    nul_pct, eff_pct = 100 * l_nul / l_tot, 100 * l_eff / l_tot
    geo = math.exp(l_tot / len(tot))                 # geo-mean tightening

    ax.plot(ms, tot, "o-", color="black", ms=6, lw=2.2, zorder=4,
            label=r"total $\sigma_{\rm UL}(k{=}3)/\sigma_{\rm UL}(k{=}5)$")
    ax.plot(ms, n_fac, "s-", color="#2ca02c", ms=5, lw=1.7, zorder=3,
            label=fr"from $N_{{\rm UL}}$ (background): {nul_pct:.0f}%")
    ax.plot(ms, e_fac, "^-", color="#1f77b4", ms=5, lw=1.7, zorder=3,
            label=fr"from efficiency: {eff_pct:.0f}%")
    ax.axhline(1.0, color="0.6", lw=0.9, ls=":")
    ax.set_ylim(0.9, max(tot) * 1.12)
    ax.set_xlabel(r"$m_{W_R}$ [TeV]")
    ax.set_ylabel(fr"limit tightening  $\sigma_{{\rm UL}}(k{{=}}{k_ref:g})/"
                  fr"\sigma_{{\rm UL}}(k{{=}}{k_tgt:g})$")
    ax.text(0.04, 0.95, f"{CH_LAB[channel]}  {TOPO_LAB[topology]}  ({fn})\n"
            r"$m_N=m_{W_R}/2$,  trusted region" "\n"
            fr"mean tightening $\times{geo:.2f}$: "
            fr"${nul_pct:.0f}\%$ background, ${eff_pct:.0f}\%$ efficiency",
            transform=ax.transAxes, fontsize=12, va="top")
    ax.legend(loc="upper left", bbox_to_anchor=(0.04, 0.80), fontsize=12,
              frameon=False)
    _cms(ax, com, lumi)
    _save(fig, out)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved", choices=["resolved", "boosted"])
    p.add_argument("--functions", nargs="+", default=["expo", "powlaw"])
    p.add_argument("--k-grid", nargs="+", type=float,
                   default=[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
    p.add_argument("--cl", type=float, default=0.95)
    p.add_argument("--conv-min", type=float, default=0.85,
                   help="min converged-toy fraction for a (mass,k) to be reliable")
    p.add_argument("--tail-max", type=float, default=1.0,
                   help="max RMS/q95(|N_sp|); >~1 means a runaway tail")
    p.add_argument("--bias-max", type=float, default=0.5,
                   help="max |mean(N_sp)|/RMS (spurious-signal acceptance); the "
                        "centre-zero band omits this bias, so a wide window that "
                        "mismodels the background is rejected here. Set to a large "
                        "value (e.g. 99) to recover the RMS-only optimization.")
    p.add_argument("--decomp-k", nargs=2, type=float, default=[3.0, 5.0],
                   help="the (k_ref, k_tgt) pair for the eff-vs-N_UL "
                        "decomposition plot")
    p.add_argument("--decomp-range", nargs=2, type=float, default=[1200.0, 3000.0],
                   help="mass range (trusted region) for the decomposition plot")
    p.add_argument("--channel-bfrac", type=float, default=0.5)
    p.add_argument("--signal-config", type=Path, default=None)
    p.add_argument("--signal-era", default=None)
    p.add_argument("--input-dir", type=Path, default=None,
                   help="dir holding efficiency_vs_k_*.csv and stage6/ (default: "
                        "<script dir>/<run2|run3>)")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    ch, topo = args.channel, args.topology
    info = load_lumi(args.era)
    lumi, com = info["lumi"], info.get("com", 13.6)
    run_sub = {"RunII": "run2", "Run3": "run3"}[str(info["run"])]
    base = args.input_dir or (HERE / run_sub)
    out_dir = args.output_dir or base
    alpha = 1.0 - args.cl
    tag = f"{ch}_{topo}"
    kgrid = sorted(args.k_grid)

    eff, tags = load_efficiency(base / f"efficiency_vs_k_{tag}.csv")
    cfg_path = args.signal_config or default_signal_config(args.signal_era or args.era)
    theory = load_theory(cfg_path, tags, args.channel_bfrac)

    out_rows = []
    for fn in args.functions:
        s6 = {k: load_stage6(base / "stage6" / f"k{k:g}"
                             / f"spurious_toy_table_{tag}.csv", fn) for k in kgrid}
        rows_by_mass = {}
        for mWR in sorted(eff):
            pts = []
            for k in kgrid:
                if k not in eff[mWR] or mWR not in s6.get(k, {}):
                    continue
                e = eff[mWR][k]
                rec = s6[k][mWR]
                n_ul = cls_band(0.0, rec["rms"], alpha)[0]
                if not (e > 0 and math.isfinite(n_ul)):
                    continue
                sigma_ul = n_ul / (1000.0 * lumi * e)
                pts.append({"k": k, "eff": e, "rms": rec["rms"], "n_ul": n_ul,
                            "sigma_UL": sigma_ul,
                            "reliable": reliable(rec, args.conv_min, args.tail_max,
                                                 args.bias_max)})
            if pts:
                rows_by_mass[mWR] = pts

        # per-mass reliability-capped optimum
        summary = {}
        for mWR, pts in rows_by_mass.items():
            rel = [p for p in pts if p["reliable"]]
            if not rel:
                logger.info("  [%s] m=%.0f: no reliable k, skip", fn, mWR)
                continue
            opt = min(rel, key=lambda p: p["sigma_UL"])
            base_k3 = next((p for p in pts if abs(p["k"] - KREF) < 1e-6), None)
            gain = (base_k3["sigma_UL"] / opt["sigma_UL"]) if base_k3 else float("nan")
            summary[mWR] = {"k_opt": opt["k"], "sigma_UL_opt": opt["sigma_UL"],
                            "sigma_UL_k3": base_k3["sigma_UL"] if base_k3 else float("nan"),
                            "gain": gain,
                            "reliable_ks": [p["k"] for p in rel]}
            out_rows.append({"channel": ch, "topology": topo, "function": fn,
                             "mWR": int(mWR), "k_opt": opt["k"],
                             "sigma_UL_opt_pb": f"{opt['sigma_UL']:.5g}",
                             "sigma_UL_k3_pb": f"{summary[mWR]['sigma_UL_k3']:.5g}",
                             "gain_k3_over_opt": f"{gain:.3f}",
                             "reliable_ks": "|".join(f"{k:g}" for k in summary[mWR]["reliable_ks"])})
            logger.info("  [%s] m=%.0f -> k_opt=%.1f  sigma_UL=%.4g pb "
                        "(k3=%.4g, gain x%.2f)  reliable k in [%s]", fn, mWR,
                        opt["k"], opt["sigma_UL"], summary[mWR]["sigma_UL_k3"],
                        gain, ",".join(f"{k:g}" for k in summary[mWR]["reliable_ks"]))

        if not summary:
            logger.warning("  [%s] no masses with a reliable optimum", fn); continue

        # global single-best k (informational; the headline is per-mass)
        by_k_gain = {}
        for mWR, pts in rows_by_mass.items():
            k3 = next((p for p in pts if abs(p["k"] - KREF) < 1e-6 and p["reliable"]), None)
            for p in pts:
                if p["reliable"] and k3:
                    by_k_gain.setdefault(p["k"], []).append(k3["sigma_UL"] / p["sigma_UL"])
        if by_k_gain:
            gk = {k: float(np.mean(v)) for k, v in by_k_gain.items() if len(v) >= 3}
            best = max(gk, key=gk.get) if gk else None
            if best is not None:
                logger.info("  [%s] single global best k ~ %.1f (mean gain x%.2f "
                            "over masses reliable at both k and 3)", fn, best, gk[best])

        plot_sigma_vs_k(rows_by_mass, fn, out_dir / "sigma_vs_k" / tag / fn,
                        channel=ch, topology=topo, com=com, lumi=lumi)
        plot_kopt_vs_mass(summary, fn, out_dir / "kopt_vs_mass" / tag / fn,
                          channel=ch, topology=topo, com=com, lumi=lumi)
        plot_reach(rows_by_mass, summary, theory, fn, out_dir / "reach" / tag / fn,
                   channel=ch, topology=topo, com=com, lumi=lumi)
        plot_gain_vs_mass(summary, fn, out_dir / "gain_vs_mass" / tag / fn,
                          channel=ch, topology=topo, com=com, lumi=lumi)
        plot_decomposition(rows_by_mass, fn, args.decomp_k[0], args.decomp_k[1],
                           out_dir / "decomposition" / tag / fn,
                           channel=ch, topology=topo, com=com, lumi=lumi,
                           mass_range=tuple(args.decomp_range))

    if out_rows:
        csv_path = out_dir / f"sensitivity_vs_k_{tag}.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader(); w.writerows(out_rows)
        logger.info("  wrote %s", csv_path)
    logger.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
