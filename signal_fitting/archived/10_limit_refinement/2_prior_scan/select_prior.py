#!/usr/bin/env python3
"""Stage 10.2b -- apply the documented selection rule to the prior-scan table.

Reads prior_scan_table_{ch}_{topo}.csv (merging shard tables if --table points
at a directory of shards) and picks the recommended (alpha_mu, alpha_sigma).

Selection (explicit; TRUSTED masses only, default {1400, 2000, 2800, 3200}).

The additive background-mismodeling spurious signal (the null mean, e.g. ~-19
events at m=1400 with the Poisson core) sits identically in the null and the
injection cells, and NO signal-shape prior can remove it -- so both the
injection gate and the ranking use the RECOVERY DEFICIT

    deficit(cfg, m, x) = [mean_Nsig(N=9) - mean_Nsig(null)] - 9

(the injection-specific response above the config's own null), never the raw
bias.  The TARGET is the IN-WINDOW injected yield 9*W (W = template fraction
inside [fit_lo, fit_hi], computed here from the same templates the scan
injected): the fit can only ever see the in-window part of the signal, and the
out-of-window loss 9*(1-W) -- up to ~45% for the compressed x=min shapes at
high mass -- is bookkept downstream in the xsec efficiency, so charging it to
the prior config would (measurably) swamp the shape-mismatch signal the scan
is after.  Thresholds are SEM-aware: with ~300 toys the deficit SEM at low
mass is ~3 events, so the gate is max(2, 2*SEM).

  Hard gates -- every trusted cell must satisfy
    G1  conv >= 0.90 (catastrophic-collapse guard; the 3200 null sits at 0.94)
    G2  frac_mu_rail <= 0.10 and frac_sigma_rail <= 0.10
    G3  null degradation vs fixed: |mean_null(cfg) - mean_null(fixed)|
        <= max(2, 2*SEM) and half68_null(cfg)/half68_null(fixed) <= 1.5
    G4  every N=9 injection cell (all x):
        |deficit| = |mean9 - mean_null - 9*W| <= max(2, 2*SEM)

  Ranking among survivors (lexicographic, tie tolerances in brackets):
    R1  minimize B = max over trusted (mass, x) of |deficit|         [0.5 evt]
    R2  minimize S = median over trusted masses of
        half68_null(cfg) / half68_null(fixed)                        [0.05]
    R3  minimize C = median over trusted cells of |pull_width - 1|   [0.05]
    R4  prefer smaller alpha_sigma, then smaller alpha_mu (least departure
        from the fixed baseline)

Outputs: chosen_prior_{ch}_{topo}.json (winner + runners-up + gate report),
heatmaps of B/S/C over the (alpha_mu, alpha_sigma) grid, and bias/spread
vs mass overlays for the winner vs the fixed baseline.

Setup:
  source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))                      # repo root
from wrplotter.cli_utils import setup_logging                 # noqa: E402

logger = logging.getLogger("select_prior")


def _f(x, default=float("nan")):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_table(path, tag):
    """Read the scan table; if `path` is a directory, merge shards/m*/ tables."""
    path = Path(path)
    files = ([path] if path.is_file()
             else sorted(path.glob(f"shards/m*/prior_scan_table_{tag}.csv"))
             or [path / f"prior_scan_table_{tag}.csv"])
    rows = []
    for f in files:
        with open(f, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    logger.info("read %d rows from %d file(s)", len(rows), len(files))
    return rows


def cfg_key(r):
    return (r["alpha_mu"], r["alpha_sigma"])


def a_sort(v):
    return float("inf") if v == "inf" else float(v)


def window_fractions(rows, trusted):
    """{(mass, x_frac): W} -- in-window fraction of each injected template,
    recomputed from the same signal MC the scan injected."""
    sys.path.insert(0, str(HERE.parent))
    from toy_engine import Inputs
    cells = sorted({(_f(r["mWR"]), r["x_frac"]) for r in rows
                    if _f(r["N_inj"]) > 0 and _f(r["mWR"]) in trusted})
    if not cells:
        return {}
    ch, topo = rows[0]["channel"], rows[0]["topology"]
    inp = Inputs(channel=ch, topology=topo)
    out = {}
    for m, xf in cells:
        frac = 0.05 if xf == "min" else float(xf)
        stag = inp.signal_tag(m, frac)
        shape = inp.signal_shape(stag) if stag else None
        if shape is None:
            continue
        m_c, sw, lo, hi = inp.fit_range(m)
        win = (inp.centers >= lo) & (inp.centers <= hi)
        out[(m, xf)] = float(shape[win].sum())
    return out


def evaluate(rows, trusted, wfrac, gate4=2.0, min_conv=0.90):
    """Per-config gate results + ranking metrics (deficit-based, SEM-aware)."""
    cfgs = sorted({cfg_key(r) for r in rows},
                  key=lambda c: (a_sort(c[0]), a_sort(c[1])))
    fixed_null = {}          # mass -> (mean, half68) of the fixed baseline null
    for r in rows:
        if (cfg_key(r) == ("0.0", "0.0") and _f(r["N_inj"]) == 0
                and _f(r["mWR"]) in trusted and r.get("half68")):
            fixed_null[_f(r["mWR"])] = (_f(r["mean_Nsig"]), _f(r["half68"]))
    out = []
    for cfg in cfgs:
        sub = [r for r in rows if cfg_key(r) == cfg and _f(r["mWR"]) in trusted]
        if not sub:
            continue
        g1 = all(_f(r["conv"]) >= min_conv for r in sub)
        g2 = all(_f(r.get("frac_mu_rail"), 0) <= 0.10
                 and _f(r.get("frac_sigma_rail"), 0) <= 0.10 for r in sub
                 if r.get("half68"))
        nulls = {_f(r["mWR"]): r for r in sub
                 if _f(r["N_inj"]) == 0 and r.get("half68")}
        # G3: null degradation vs the fixed baseline (paired toys -> the
        # difference of means is the mean paired difference)
        g3 = True
        S_terms = []
        for m, r in nulls.items():
            if m not in fixed_null:
                continue
            f_mean, f_h68 = fixed_null[m]
            sem = _f(r["bias_evt_err"], _f(r["half68"]) / math.sqrt(
                max(_f(r["n_ok"], 300), 1)))
            if abs(_f(r["mean_Nsig"]) - f_mean) > max(2.0, 2 * sem):
                g3 = False
            ratio = _f(r["half68"]) / f_h68 if f_h68 > 0 else float("nan")
            S_terms.append(ratio)
            if math.isfinite(ratio) and ratio > 1.5:
                g3 = False
        # G4: injection-recovery deficit above the config's own null
        inj = [r for r in sub if _f(r["N_inj"]) == 9 and r.get("half68")]
        deficits = []
        g4 = bool(inj)
        for r in inj:
            m = _f(r["mWR"])
            if m not in nulls:
                continue
            null_mean = _f(nulls[m]["mean_Nsig"])
            target = 9.0 * wfrac.get((m, r["x_frac"]), 1.0)
            d = (_f(r["mean_Nsig"]) - null_mean) - target
            n_ok = max(_f(r["n_ok"], 300), 1)
            sem = math.sqrt(_f(r["rms_Nsig"]) ** 2 / n_ok
                            + _f(nulls[m]["rms_Nsig"]) ** 2
                            / max(_f(nulls[m]["n_ok"], 300), 1))
            deficits.append(abs(d))
            if abs(d) > max(gate4, 2 * sem):
                g4 = False
        B = max(deficits) if deficits else float("nan")
        S = statistics.median(S_terms) if S_terms else float("nan")
        C_terms = [abs(_f(r["pull_width"]) - 1.0) for r in sub
                   if r.get("pull_width")]
        C = statistics.median(C_terms) if C_terms else float("nan")
        out.append({"alpha_mu": cfg[0], "alpha_sigma": cfg[1],
                    "G1": g1, "G2": g2, "G3": g3, "G4": g4,
                    "pass_all": g1 and g2 and g3 and g4,
                    "B": B, "S": S, "C": C})
    return out


def rank(survivors):
    """Lexicographic with tie tolerances: B (0.5), S (0.05), C (0.05), then
    smallest alpha_sigma, then alpha_mu."""
    pool = list(survivors)
    if not pool:
        return []
    bmin = min(s["B"] for s in pool)
    pool = [s for s in pool if s["B"] <= bmin + 0.5]
    smin = min(s["S"] for s in pool)
    pool = [s for s in pool if s["S"] <= smin + 0.05]
    cmin = min(s["C"] for s in pool)
    pool = [s for s in pool if s["C"] <= cmin + 0.05]
    return sorted(pool, key=lambda s: (a_sort(s["alpha_sigma"]),
                                       a_sort(s["alpha_mu"])))


def heatmap(evals, metric, out, tag):
    amus = sorted({e["alpha_mu"] for e in evals}, key=a_sort)
    asigs = sorted({e["alpha_sigma"] for e in evals}, key=a_sort)
    grid = np.full((len(asigs), len(amus)), np.nan)
    for e in evals:
        grid[asigs.index(e["alpha_sigma"]), amus.index(e["alpha_mu"])] = e[metric]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(amus)), amus)
    ax.set_yticks(range(len(asigs)), asigs)
    ax.set_xlabel(r"$\alpha_\mu$ (prior width on $\mu$ / $\sigma_0$)")
    ax.set_ylabel(r"$\alpha_\sigma$ (prior width on $\sigma$ / $\sigma_0$)")
    ax.set_title(f"{metric}  ({tag})")
    for e in evals:
        i, j = asigs.index(e["alpha_sigma"]), amus.index(e["alpha_mu"])
        v = e[metric]
        txt = f"{v:.2f}" if math.isfinite(v) else "-"
        if not e["pass_all"]:
            txt += "*"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                color="white" if not math.isfinite(v) or v > np.nanmean(grid)
                else "black")
    fig.colorbar(im, ax=ax)
    fig.text(0.01, 0.01, "* = fails a hard gate", fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def overlay_vs_mass(rows, winner, out, tag):
    """Winner vs fixed baseline: null half68 and N=9 recovery bias vs mass."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for cfg, color, label in [(("0.0", "0.0"), "#e42536", "fixed (0,0)"),
                              (winner, "#5790fc",
                               f"winner ({winner[0]},{winner[1]})")]:
        nulls = sorted([(r, _f(r["mWR"])) for r in rows
                        if cfg_key(r) == cfg and _f(r["N_inj"]) == 0
                        and r.get("half68")], key=lambda t: t[1])
        axs[0].plot([m for _, m in nulls], [_f(r["half68"]) for r, _ in nulls],
                    "o-", color=color, label=label)
        null_mean = {m: _f(r["mean_Nsig"]) for r, m in nulls}
        inj = [(r, _f(r["mWR"])) for r in rows
               if cfg_key(r) == cfg and _f(r["N_inj"]) == 9 and r.get("half68")]
        for xf, ls in (("min", ":"), ("0.2", "-."), ("0.5", "-"), ("0.9", "--")):
            pts = sorted([(m, _f(r["mean_Nsig"]) - null_mean[m] - 9)
                          for r, m in inj
                          if r["x_frac"] == xf and m in null_mean])
            if pts:
                axs[1].plot(*zip(*pts), ls, marker="o", ms=3, color=color,
                            label=f"{label} x={xf}")
    axs[0].set_xlabel(r"$m_{W_R}$ [GeV]")
    axs[0].set_ylabel("null half68 spread [events]")
    axs[0].legend(fontsize=9)
    axs[1].axhline(0, color="grey", lw=0.8, ls="--")
    axs[1].axhspan(-2, 2, color="#2ca02c", alpha=0.12)
    axs[1].set_xlabel(r"$m_{W_R}$ [GeV]")
    axs[1].set_ylabel(
        r"recovery deficit $[\langle N\rangle_{9} - \langle N\rangle_{0}] - 9$")
    axs[1].legend(fontsize=8)
    fig.suptitle(tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--channel", default="ee", choices=["ee", "mumu"])
    p.add_argument("--topology", default="resolved",
                   choices=["resolved", "boosted"])
    p.add_argument("--table", type=Path, default=HERE,
                   help="scan table CSV, or a directory containing shards/")
    p.add_argument("--trusted", nargs="+", type=float,
                   default=[1400, 2000, 2800, 3200])
    p.add_argument("--output-dir", type=Path, default=HERE)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    tag = f"{args.channel}_{args.topology}"
    rows = load_table(args.table, tag)
    trusted = set(args.trusted)
    wfrac = window_fractions(rows, trusted)
    for (m, xf), w in sorted(wfrac.items()):
        logger.info("  W(m=%.0f, x=%s) = %.3f -> target %.2f", m, xf, w, 9 * w)

    evals = evaluate(rows, trusted, wfrac)
    survivors = [e for e in evals if e["pass_all"]]
    gate4_note = ""
    if not survivors:
        logger.warning("G4 (|deficit|<=2) empties the field -- relaxing to 3.0")
        evals = evaluate(rows, trusted, wfrac, gate4=3.0)
        survivors = [e for e in evals if e["pass_all"]]
        gate4_note = "G4 relaxed to 3.0"
    ranked = rank(survivors)

    for e in evals:
        logger.info("  amu=%-5s asig=%-5s gates %s%s%s%s  B=%5.2f S=%5.2f C=%5.2f",
                    e["alpha_mu"], e["alpha_sigma"],
                    *["+" if e[g] else "-" for g in ("G1", "G2", "G3", "G4")],
                    e["B"], e["S"], e["C"])
    if not ranked:
        logger.error("No config passes the gates even after relaxing G4. "
                     "The blocker is the estimator, not the prior widths.")
        result = {"winner": None, "note": "no survivor; see table",
                  "evals": evals}
    else:
        winner = ranked[0]
        logger.info("WINNER: alpha_mu=%s alpha_sigma=%s  (B=%.2f S=%.2f C=%.2f) %s",
                    winner["alpha_mu"], winner["alpha_sigma"],
                    winner["B"], winner["S"], winner["C"], gate4_note)
        result = {"winner": {"alpha_mu": winner["alpha_mu"],
                             "alpha_sigma": winner["alpha_sigma"]},
                  "runners_up": [{"alpha_mu": r["alpha_mu"],
                                  "alpha_sigma": r["alpha_sigma"]}
                                 for r in ranked[1:3]],
                  "note": gate4_note, "trusted_masses": sorted(trusted),
                  "metrics": ranked[0], "n_survivors": len(survivors)}
        overlay_vs_mass(rows, (winner["alpha_mu"], winner["alpha_sigma"]),
                        args.output_dir / "winner_vs_fixed" / tag, tag)

    for metric in ("B", "S", "C"):
        heatmap(evals, metric, args.output_dir / "heatmaps" / f"{tag}_{metric}",
                tag)
    out = args.output_dir / f"chosen_prior_{tag}.json"
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
