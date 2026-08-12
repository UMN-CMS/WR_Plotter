#!/usr/bin/env python3
"""Plot driver for the full-grid gauss study.

Reads outputs/results.csv (produced by scan_full.py — same schema as
signal_fitting/outputs/.../pull_study/results.csv) and produces:

  --what pull_xy     pull_xy_{bias,spread}_{mu,width}_gauss_both_{channel}_n{N}.{pdf,png}
                     (4 metric/param combos × 2 channels × selected N values)

  --what cell_offsets cell_offsets_{mu,width}_{mass}_{channel}_gauss_both_n{N}.{pdf,png}
                     One per (channel, mass) at the requested N (default 20).
                     ~1480 figures across both params.

  --what pull_demo   pull_demo_{mu,width}_{cell}.png for outliers only.
                     Re-runs a single-toy fit with the same prior conventions
                     as scan_full.py and makes a pedagogy plot for the worst
                     cells.

  --what all         Run all three.
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))

from wrplotter.cli_utils import setup_logging
from wrplotter.paths import input_dirs_for_era, repo_root

from measure_fwhm import (
    ONSHELL_WINDOW_LO_FRAC, ONSHELL_WINDOW_HI_FRAC,
    build_hist_key, build_region_name,
    load_and_combine_signal, parse_masses, rebin_histogram,
)
from fit_signal_toy import (
    FWHM_TO_GAUSS_SIGMA,
    bootstrap_fwhm_estimate, bootstrap_peak_estimate,
    evaluate_fit_curve, run_fit, sample_from_hist_root,
)
from pull_stats import gauss_amp, gaussian_pull_fit

logger = logging.getLogger(__name__)

PARAMS = {
    "mu":    ("mu_fit",    "mu_err",    "mu_truth",    r"\mu"),
    "width": ("width_fit", "width_err", "width_truth", r"\sigma"),
}
CH_LAB = {"ee": "ee", "mumu": r"$\mu\mu$"}

# Curated demo masses: 5 M_WR × 3 M_N/M_WR ratios (low / mid / high in [0.2, 0.9]).
# Used by --curated for cell_offsets and pull_demo to avoid plotting all 369 masses.
CURATED_MASSES = [
    "WR2000_N600",  "WR2000_N1000", "WR2000_N1600",
    "WR3000_N1000", "WR3000_N1600", "WR3000_N2400",
    "WR4000_N1200", "WR4000_N2000", "WR4000_N3200",
    "WR5000_N1400", "WR5000_N2400", "WR5000_N4000",
    "WR6000_N1800", "WR6000_N3000", "WR6000_N4800",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True,
                   help="Path to results.csv produced by scan_full.py.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--what", default="all",
                   choices=["2d_pulls", "1d_pulls", "cell_offsets", "pull_demo", "all"],
                   help="2d_pulls + 1d_pulls are the production plots; "
                        "cell_offsets and pull_demo are legacy flat-layout "
                        "outputs (only run when explicitly requested).")
    p.add_argument("--n-events", type=int, nargs="+", default=[20],
                   help="N values to plot for cell_offsets / pull_demo. "
                        "pull_xy plots all N found in the CSV automatically.")
    p.add_argument("--era", default="RunIISummer20UL18")
    p.add_argument("--dir", default="20260406_signals")
    p.add_argument("--alpha-mu", type=float, default=0.10,
                   help="α_µ used in the scan (for pull_demo prior reconstruction).")
    p.add_argument("--alpha-sigma", type=float, default=0.25,
                   help="α_σ used in the scan (for pull_demo prior reconstruction).")
    p.add_argument("--alpha-sigma-mumu", type=float, default=None,
                   help="If set, overrides --alpha-sigma for the mumu channel "
                        "(used by pull_demo when production uses channel-split α_σ).")
    p.add_argument("--use-moments", action="store_true",
                   help="Use windowed (mean, RMS) for the pull_demo truth + "
                        "prior centrals (matches scan_full.py --use-moments).")
    p.add_argument("--n-outliers", type=int, default=8,
                   help="Number of outlier cells to make pull_demo plots for, "
                        "per (channel, param). Sorted by combined |bias|+|spread-1|.")
    p.add_argument("--curated", action="store_true",
                   help="Use the 15-cell curated demo list (5 M_WR × 3 M_N) for "
                        "cell_offsets and pull_demo instead of all-masses / outliers.")
    p.add_argument("--topology", default="resolved")
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args()


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# pull_xy: trivial wrapper around the existing function
# --------------------------------------------------------------------------- #

def do_2d_pulls(rows, out_dir, *, model="gauss", config="both"):
    """Per-N (M_N/M_WR vs metric) scatter plots, nested layout:
        out_dir/n{N}/{channel}_{metric}_{param}.{pdf,png}

    Uniform y-axis across all (channel × N) panels within the same (metric,
    param). One PDF + one PNG per file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for param in ("mu", "width"):
        for metric in ("bias", "spread"):
            ymin, ymax = _compute_uniform_yrange(rows, param=param,
                                                  metric=metric,
                                                  model=model, config=config)
            logger.info("2d_pulls: param=%s metric=%s  ylim=[%.2f, %.2f]",
                        param, metric, ymin, ymax)
            _make_pull_xy_uniform(rows, out_dir,
                                  param=param, metric=metric,
                                  ymin=ymin, ymax=ymax,
                                  model=model, config=config,
                                  nested=True)


def _per_cell_value(rows, *, param, metric, model, config):
    """Returns dict (channel, n_events, mass) -> value (bias or spread)."""
    fit_k, err_k, truth_k, _ = {
        "mu":    ("mu_fit",    "mu_err",    "mu_truth",    r"\mu"),
        "width": ("width_fit", "width_err", "width_truth", r"\sigma"),
    }[param]
    by_cell = defaultdict(list)
    for r in rows:
        if r["model"] != model or r["config"] != config:
            continue
        if int(r["status"]) != 0 or int(r["covqual"]) < 3:
            continue
        try:
            err = float(r[err_k])
            if err <= 0 or not np.isfinite(err):
                continue
            fit = float(r[fit_k]); truth = float(r[truth_k])
            if not (np.isfinite(fit) and np.isfinite(truth)):
                continue
        except (KeyError, ValueError):
            continue
        by_cell[(r["channel"], int(r["n_events"]), r["mass"])].append(
            (fit - truth) / err)
    out = {}
    for k, vs in by_cell.items():
        if len(vs) < 5:
            continue
        gf = gaussian_pull_fit(vs)
        out[k] = gf["mu"] if metric == "bias" else gf["sigma"]
    return out


def _compute_uniform_yrange(rows, *, param, metric, model, config):
    """Return (ymin, ymax) covering every (channel, N, mass) cell."""
    vals = list(_per_cell_value(rows, param=param, metric=metric,
                                 model=model, config=config).values())
    if not vals:
        return (-1.0, 1.0) if metric == "bias" else (0.0, 2.0)
    a = np.asarray(vals)
    if metric == "bias":
        amp = float(np.max(np.abs(a))) * 1.10
        amp = max(amp, 1.0)
        return (-amp, amp)
    else:
        # spread: center on 1, expand to cover the worst cell
        amp = float(np.max(np.abs(a - 1.0))) * 1.10
        amp = max(amp, 0.5)
        return (max(0.0, 1.0 - amp), 1.0 + amp)


def _make_pull_xy_uniform(rows, out_dir, *, param, metric,
                           ymin, ymax, model, config, nested=False):
    """Mirror pull_study.make_pull_xy but with externally fixed y-limits.

    If `nested=True`, save to `out_dir/n{N}/{channel}_{metric}_{param}.{pdf,png}`
    (2d_pulls layout). Otherwise use the legacy flat filename.
    """
    import re
    from matplotlib import cm, colors as mcolors

    cell_metric = _per_cell_value(rows, param=param, metric=metric,
                                   model=model, config=config)
    if not cell_metric:
        return
    channels = sorted({k[0] for k in cell_metric})
    n_events_list = sorted({k[1] for k in cell_metric})
    mass_re = re.compile(r"WR(\d+)_N(\d+)")

    all_mwrs = []
    for k in cell_metric:
        m = mass_re.match(k[2])
        if m: all_mwrs.append(int(m.group(1)))
    if not all_mwrs:
        return
    mwr_lo, mwr_hi = min(all_mwrs), max(all_mwrs)
    norm = mcolors.Normalize(vmin=mwr_lo, vmax=mwr_hi)
    cmap = cm.viridis

    plabel = r"\mu" if param == "mu" else r"\sigma"
    pdf_name = "Gaussian"
    cfg_lab = "Both Constrained" if config == "both" else config
    metric_label = (rf"Pull bias on ${plabel}$  (Gaussian-fit $\mu$)" if metric == "bias"
                    else rf"Pull width on ${plabel}$  (Gaussian-fit $\sigma$)")
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
            if not xs:
                continue
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
            ax.set_ylim(ymin, ymax)
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
            if nested:
                cell_dir = out_dir / f"n{n}"
                cell_dir.mkdir(parents=True, exist_ok=True)
                out = cell_dir / f"{channel}_{metric}_{param}.pdf"
            else:
                out = (out_dir /
                       f"pull_xy_{metric}_{param}_{model}_{config}_{channel}_n{n}.pdf")
            fig.savefig(out, bbox_inches="tight")
            fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
            plt.close(fig)
            logger.info("Wrote %s", out)


# --------------------------------------------------------------------------- #
# cell_offsets: per-cell histogram of (fit - truth) in GeV.  Closely mirrors
# pull_cell_demo.py's main(), but converted to a reusable function so we can
# iterate over every (channel, mass) without reinvoking Python each time.
# --------------------------------------------------------------------------- #

def _cell_pulls(rows_filter):
    """Yield (offset_gev, pull, mu_or_sigma_err) tuples for a filter dict."""
    pass  # not used; we inline below


def plot_cell_offsets_one(channel, mass, n_events, param, cell_rows, out_dir=None,
                          config="both", era="RunIISummer20UL18", out_path=None,
                          alpha_mu=None, alpha_sigma=None):
    """Per-cell pull histogram in σ-units; reference is N(0, 1).

    cell_rows: pre-filtered list for (channel, mass, n_events,
               model='gauss', config=config).

    Pass `out_dir` for the legacy flat layout (cell_offsets_*.png), or
    `out_path` for an explicit destination (e.g. the nested 1d_pulls/ layout).
    """
    fit_k, err_k, truth_k, plabel = PARAMS[param]
    pulls = []
    for r in cell_rows:
        if int(r["status"]) != 0 or int(r["covqual"]) < 3:
            continue
        try:
            err = float(r[err_k])
            fit = float(r[fit_k]); truth = float(r[truth_k])
        except (KeyError, ValueError):
            continue
        if not (err > 0 and np.isfinite(err)
                and np.isfinite(fit) and np.isfinite(truth)):
            continue
        pulls.append((fit - truth) / err)
    if len(pulls) < 5:
        return False
    pulls_a = np.asarray(pulls)
    gf = gaussian_pull_fit(pulls_a)
    counts, edges, centers = gf["counts"], gf["edges"], gf["centers"]
    bw = edges[1] - edges[0]
    mu_g, mu_g_err = gf["mu"], gf["mu_err"]
    sig_g, sig_g_err = gf["sigma"], gf["sigma_err"]

    plt.style.use("default")
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(centers, counts, width=bw, color="#1f77b4", alpha=0.85,
           edgecolor="white", linewidth=0.6, label=f"{len(pulls_a)} toys")

    xs = np.linspace(-4, 4, 400)
    ref = (np.exp(-0.5 * xs ** 2) / np.sqrt(2 * np.pi)) * len(pulls_a) * bw
    ax.plot(xs, ref, color="black", linewidth=1.8, linestyle="--",
            label=r"reference $\mathcal{N}(0,1)$")

    # Gaussian fit to the pulls (standard HEP pull metric).
    gfit = None
    if np.isfinite(mu_g) and np.isfinite(sig_g):
        gfit = gauss_amp(xs, gf["amp"], mu_g, sig_g)
        ax.plot(xs, gfit, color="#2ca02c", linewidth=2.4,
                label=rf"Gaussian fit: $\mu={mu_g:+.2f}$, $\sigma={sig_g:.2f}$")
        ax.axvline(mu_g, color="#2ca02c", linewidth=1.6, linestyle="-",
                   alpha=0.8)

    ymax = float(max(counts.max() if counts.size else 1.0, ref.max(),
                     np.max(gfit) if gfit is not None else 0.0))
    ax.axvline(0, color="grey", linewidth=1.0, linestyle=":")

    ax.set_xlabel(
        rf"pull on ${plabel}$  $= ({plabel}_{{\mathrm{{fit}}}} - "
        rf"{plabel}_{{\mathrm{{truth}}}})/\sigma_{{{plabel}, \mathrm{{fit}}}}$",
        fontsize=18,
    )
    ax.set_ylabel("toys / 0.25", fontsize=18)
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, ymax * 1.55)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.3)
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                  com=13, fontsize=18)
    extra_lines = ""
    if alpha_mu is not None:
        extra_lines += rf"\n$\alpha_\mu = {alpha_mu:g}$"
    if alpha_sigma is not None:
        extra_lines += rf"\n$\alpha_\sigma = {alpha_sigma:g}$"
    if np.isfinite(mu_g):
        extra_lines += (rf"\nfit $\mu = {mu_g:+.2f} \pm {mu_g_err:.2f}$"
                        rf"\nfit $\sigma = {sig_g:.2f} \pm {sig_g_err:.2f}$")
    ax.text(
        0.04, 0.96,
        (f"{CH_LAB[channel]}\nResolved SR\n{era}\n"
         f"{mass.replace('_', ', ')}\n"
         f"Gaussian / Both Constrained\n"
         rf"$N_{{\rm events}} = {n_events}$").replace("\\n", "\n")
        + extra_lines.replace("\\n", "\n"),
        transform=ax.transAxes, fontsize=12, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="gray", alpha=0.9),
    )
    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98),
              fontsize=11, framealpha=0.90)
    fig.tight_layout()
    if out_path is None:
        out_path = (out_dir / f"cell_offsets_{param}_{mass}_{channel}_"
                              f"gauss_{config}_n{n_events}.png")
    fig.savefig(out_path, dpi=110)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return True


def do_1d_pulls(rows, out_dir, n_events_list, *,
                alpha_mu, alpha_sigma_by_channel,
                era, dir_, topology,
                model="gauss", config="both",
                use_moments=False):
    """Per-cell 1D pull histograms + single-toy pull_demos in nested layout:
        out_dir/n{N}/{mass_lower}/{channel}_{param}.png       (pull histogram)
        out_dir/n{N}/{mass_lower}/{channel}_{param}_demo.png  (single-toy demo)

    Restricted to the curated 15 cells (5 M_WR × 3 M_N ratios) for each
    channel. Each cell directory has 8 files: ee/mumu × mu/width × pull/demo.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for r in rows:
        if r["model"] != model or r["config"] != config:
            continue
        grouped[(r["channel"], r["mass"], int(r["n_events"]))].append(r)
    channels = sorted({ch for (ch, _, _) in grouped})
    logger.info("1d_pulls: %d curated masses × %d channels × %d N × 2 params × {pull, demo}",
                len(CURATED_MASSES), len(channels), len(n_events_list))

    n_pulls = 0; n_demos = 0
    for n_ev in n_events_list:
        for mass in CURATED_MASSES:
            cell_dir = out_dir / f"n{n_ev}" / mass.lower()
            cell_dir.mkdir(parents=True, exist_ok=True)
            for channel in channels:
                cell_rows = grouped.get((channel, mass, n_ev), [])
                if not cell_rows:
                    continue
                alpha_sigma = alpha_sigma_by_channel.get(
                    channel, alpha_sigma_by_channel.get("default", 0.20))
                for param in ("mu", "width"):
                    # 1D pull histogram from CSV
                    pull_path = cell_dir / f"{channel}_{param}.png"
                    if plot_cell_offsets_one(
                        channel, mass, n_ev, param, cell_rows,
                        out_path=pull_path, config=config):
                        n_pulls += 1
                    # Single-toy pull_demo (re-runs one fit)
                    demo_path = cell_dir / f"{channel}_{param}_demo.png"
                    if make_pull_demo_one(
                        channel, mass, n_ev, param,
                        alpha_mu, alpha_sigma, None,
                        era, dir_, topology,
                        use_moments=use_moments, out_path=demo_path):
                        n_demos += 1
        logger.info("  N=%d done — pulls=%d demos=%d", n_ev, n_pulls, n_demos)
    logger.info("1d_pulls: wrote %d pull histograms + %d single-toy demos",
                n_pulls, n_demos)


def do_cell_offsets(rows, out_dir, n_events_list,
                    model="gauss", config="both", curated=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Group rows by (channel, mass, n_events) — single pass over CSV.
    grouped = defaultdict(list)
    for r in rows:
        if r["model"] != model or r["config"] != config:
            continue
        grouped[(r["channel"], r["mass"], int(r["n_events"]))].append(r)
    cells = sorted({(ch, m) for (ch, m, _) in grouped})
    if curated:
        cells = [(ch, m) for (ch, m) in cells if m in set(CURATED_MASSES)]
    logger.info("cell_offsets: %d (channel, mass) cells × %d N × 2 params"
                "%s", len(cells), len(n_events_list),
                "  [curated]" if curated else "")
    n_made = 0
    n_total_attempts = len(cells) * len(n_events_list) * 2
    for (ch, mass) in cells:
        for n_ev in n_events_list:
            cell_rows = grouped.get((ch, mass, n_ev), [])
            for param in ("mu", "width"):
                ok = plot_cell_offsets_one(
                    ch, mass, n_ev, param, cell_rows, out_dir, config=config)
                if ok:
                    n_made += 1
                    if n_made % 200 == 0:
                        logger.info("  cell_offsets progress: %d / %d files",
                                    n_made, n_total_attempts)
    logger.info("cell_offsets: wrote %d files", n_made)


# --------------------------------------------------------------------------- #
# Outlier identification + pull_demo
# --------------------------------------------------------------------------- #

def per_cell_stats(rows, n_events, model="gauss", config="both"):
    """Returns dict (channel, mass) → {mu_bias, mu_spread, sigma_bias, sigma_spread}."""
    by_cell = defaultdict(list)
    for r in rows:
        if (r["model"] != model or r["config"] != config
                or int(r["n_events"]) != n_events):
            continue
        if int(r["status"]) != 0 or int(r["covqual"]) < 3:
            continue
        try:
            mu_err = float(r["mu_err"]); s_err = float(r["width_err"])
            if mu_err <= 0 or s_err <= 0:
                continue
            mu_pull = (float(r["mu_fit"]) - float(r["mu_truth"])) / mu_err
            s_pull = (float(r["width_fit"]) - float(r["width_truth"])) / s_err
            if np.isfinite(mu_pull) and np.isfinite(s_pull):
                by_cell[(r["channel"], r["mass"])].append((mu_pull, s_pull))
        except (KeyError, ValueError):
            continue
    out = {}
    for k, pairs in by_cell.items():
        if len(pairs) < 10:
            continue
        a = np.asarray(pairs)
        gf_mu = gaussian_pull_fit(a[:, 0])
        gf_w  = gaussian_pull_fit(a[:, 1])
        out[k] = {
            "mu_bias":      gf_mu["mu"],
            "mu_spread":    gf_mu["sigma"],
            "width_bias":   gf_w["mu"],
            "width_spread": gf_w["sigma"],
        }
    return out


def find_outliers(stats, channel, param, n_top=8):
    """Top n_top cells by |bias| + |spread - 1| for (channel, param)."""
    cands = [(mass, st) for (ch, mass), st in stats.items() if ch == channel]
    bias_k = f"{param}_bias"
    spread_k = f"{param}_spread"
    cands.sort(key=lambda x: abs(x[1][bias_k]) + abs(x[1][spread_k] - 1.0),
               reverse=True)
    return cands[:n_top]


def make_pull_demo_one(channel, mass, n_events, param,
                       alpha_mu, alpha_sigma, out_dir,
                       era, dir_, topology, seed=12345, bin_width=200.0,
                       use_moments=False, out_path=None,
                       width_mode="constrained"):
    """Re-run one toy fit and plot a pedagogy figure for `param` ∈ {"mu","width"}.

    Mirrors signal_fitting/pull_demo.py and pull_demo_width.py: two stacked
    panels — top shows the toy + fitted PDF + truth/fit lines; bottom shows
    the posterior on the parameter with σ ticks and a pull arrow.

    Adapted for gauss + new prior conventions:
      prior centrals from bootstrap (mu_boot, FWHM_boot)
      prior widths      α × FWHM_boot
    """
    try:
        import ROOT
    except ImportError:
        logger.warning("ROOT unavailable, skipping pull_demo for %s/%s", channel, mass)
        return False

    M_WR, M_N = parse_masses(mass)
    M_WR, M_N = float(M_WR), float(M_N)
    input_dirs, _ = input_dirs_for_era(era, repo_root(), dir_)
    region = build_region_name(channel, topology)
    hist_key = build_hist_key(
        region, "mass_twoobject" if topology == "boosted" else "mass_fourobject")
    try:
        edges_n, vals_n, var_n = load_and_combine_signal(input_dirs, hist_key, mass)
    except Exception as e:
        logger.warning("load fail %s/%s: %s", channel, mass, e)
        return False
    edges, vals, _ = rebin_histogram(edges_n, vals_n, var_n, 6)

    fit_lo = ONSHELL_WINDOW_LO_FRAC * M_WR
    fit_hi = ONSHELL_WINDOW_HI_FRAC * M_WR
    centers_n = 0.5 * (edges_n[:-1] + edges_n[1:])
    in_win = (centers_n >= fit_lo) & (centers_n <= fit_hi)
    where = np.where(in_win)[0]
    edges_win = edges_n[where[0]:where[-1] + 2]
    vals_win = vals_n[where]

    if use_moments:
        # Moment-based truth via ROOT TH1::GetMean / GetStdDev.
        n_b = len(vals_n)
        ed_arr = np.ascontiguousarray(edges_n, dtype=np.float64)
        h_truth = ROOT.TH1D("", "", n_b, ed_arr)
        h_truth.SetDirectory(0)
        for i in range(n_b):
            h_truth.SetBinContent(i + 1, max(float(vals_n[i]), 0.0))
        h_truth.GetXaxis().SetRangeUser(fit_lo, fit_hi)
        mu_center = float(h_truth.GetMean())
        sigma_truth = float(h_truth.GetStdDev())
        fwhm_for_priors = sigma_truth * FWHM_TO_GAUSS_SIGMA
        mu_prior_sigma = alpha_mu * sigma_truth
        sigma_prior_sigma_fwhm = alpha_sigma * sigma_truth * FWHM_TO_GAUSS_SIGMA
    else:
        mu_center, _ = bootstrap_peak_estimate(edges, vals, mass, channel, topology,
                                                n_toys=100, seed=0)
        fwhm_for_priors, _ = bootstrap_fwhm_estimate(edges, vals, mass, channel, topology,
                                                     n_toys=100, seed=0)
        sigma_truth = fwhm_for_priors / FWHM_TO_GAUSS_SIGMA
        mu_prior_sigma = alpha_mu * fwhm_for_priors
        sigma_prior_sigma_fwhm = alpha_sigma * fwhm_for_priors

    events = sample_from_hist_root(edges_win, vals_win, n_events, seed)
    try:
        fit = run_fit("gauss", width_mode, events, M_WR,
                      fwhm_for_priors, sigma_prior_sigma_fwhm,
                      fit_lo, fit_hi,
                      mu_mode="constrained",
                      mu_central=mu_center, mu_sigma=mu_prior_sigma,
                      suffix_extra=f"_pulldemo_{param}_{channel}_{mass}_n{n_events}")
    except Exception as e:
        logger.warning("fit fail %s/%s: %s", channel, mass, e)
        return False

    mu_fit = fit["params"]["mu"]; mu_err = fit["errors"]["mu"]
    sigma_fit = fit["params"]["sigma"]; sigma_err = fit["errors"]["sigma"]
    n_sig_fit = fit["params"]["n_sig"]
    mu_truth = mu_center  # whatever convention (peak or mean) is the truth

    if param == "mu":
        val, err, truth = mu_fit, mu_err, mu_truth
        plabel = r"\mu"
        pull = (mu_fit - mu_truth) / mu_err
    else:
        val, err, truth = sigma_fit, sigma_err, sigma_truth
        plabel = r"\sigma"
        pull = (sigma_fit - sigma_truth) / sigma_err

    # ---- Layout (matches signal_fitting/pull_demo.py)
    n_bins = int(round((fit_hi - fit_lo) / bin_width))
    plot_edges = np.linspace(fit_lo, fit_hi, n_bins + 1)
    plot_centers = 0.5 * (plot_edges[:-1] + plot_edges[1:])
    h_obs, _ = np.histogram(events, bins=plot_edges)
    err_obs = np.sqrt(np.maximum(h_obs, 1.0))
    xs_dense = np.linspace(fit_lo, fit_hi, 2000)
    pdf_fit = (np.exp(-0.5 * ((xs_dense - mu_fit) / sigma_fit) ** 2)
               / (sigma_fit * np.sqrt(2 * np.pi)))

    # Pearson χ²/ndf comparing binned data to the fitted gauss.
    # Model expectation per bin: n_sig × ∫_bin gauss ≈ n_sig × gauss(center) × bin_width.
    gauss_at_centers = (np.exp(-0.5 * ((plot_centers - mu_fit) / sigma_fit) ** 2)
                        / (sigma_fit * np.sqrt(2 * np.pi)))
    exp_per_bin = gauss_at_centers * n_sig_fit * bin_width
    chi2_mask = exp_per_bin > 0.5
    n_free_params = 3  # µ, σ, n_sig
    if chi2_mask.sum() > n_free_params:
        chi2_val = float(np.sum(((h_obs[chi2_mask] - exp_per_bin[chi2_mask]) ** 2)
                                / exp_per_bin[chi2_mask]))
        ndf = int(chi2_mask.sum()) - n_free_params
        chi2_per_ndf = chi2_val / ndf
    else:
        chi2_val = float("nan"); ndf = 0; chi2_per_ndf = float("nan")

    hep.style.use("CMS")
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 11),
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.30},
    )

    # ---- Top panel
    # MC truth shape (scaled to n_events), as in pull_demo.py — for both mu and width.
    mc_per_plot = np.zeros(n_bins)
    centers_in = 0.5 * (edges[:-1] + edges[1:])
    for i in range(n_bins):
        mask = (centers_in >= plot_edges[i]) & (centers_in < plot_edges[i + 1])
        mc_per_plot[i] = float(vals[mask].sum())
    if mc_per_plot.sum() > 0:
        ax_top.stairs(mc_per_plot / mc_per_plot.sum() * n_events, plot_edges,
                      color="black", alpha=0.30, linewidth=1.5,
                      label=f"MC shape (scaled to {n_events} events)")
    ax_top.errorbar(plot_centers, h_obs, yerr=err_obs, marker="o", linestyle="",
                    color="black", markersize=7, label=f"toy ({n_events} events)")
    ax_top.plot(xs_dense, pdf_fit * n_sig_fit * bin_width,
                color="red", linewidth=2.2, label="Gaussian fit")

    if param == "mu":
        ax_top.axvline(mu_truth, color="black", linestyle="--", linewidth=1.5,
                       label=rf"truth $\mu_{{\rm truth}}={mu_truth:.0f}$ GeV")
        ax_top.axvline(mu_fit, color="red", linestyle="-", linewidth=1.5)
        ax_top.axvspan(mu_fit - mu_err, mu_fit + mu_err, color="red", alpha=0.15,
                       label=rf"fit $\mu_{{\rm fit}}={mu_fit:.0f}\pm{mu_err:.0f}$ GeV")
    else:
        # Width pedagogy: overlay a "truth-width" Gaussian at mu_fit with sigma=sigma_truth
        pdf_truth_w = (np.exp(-0.5 * ((xs_dense - mu_fit) / sigma_truth) ** 2)
                       / (sigma_truth * np.sqrt(2 * np.pi)))
        ax_top.plot(xs_dense, pdf_truth_w * n_sig_fit * bin_width,
                    color="black", linestyle="--", linewidth=2.0,
                    label=rf"truth-width  ($\sigma_{{\rm truth}}={sigma_truth:.0f}$ GeV)")
        # Re-draw fit label with σ-style annotation
        ax_top.lines[-2].set_label(
            rf"fit  ($\sigma_{{\rm fit}}={sigma_fit:.0f}\pm{sigma_err:.0f}$ GeV)")

    ax_top.set_ylabel(f"Events / {bin_width:.0f} GeV", fontsize=18)
    ax_top.set_xlim(fit_lo, fit_hi)
    ax_top.set_ylim(bottom=0)
    ax_top.grid(alpha=0.3)
    hep.cms.label(loc=0, ax=ax_top, data=False, label="Work in Progress",
                  com=13, fontsize=18)

    chi2_label = (rf"$\chi^2/\mathrm{{ndf}} = {chi2_val:.1f}/{ndf} = {chi2_per_ndf:.2f}$"
                  if np.isfinite(chi2_per_ndf) else r"$\chi^2/\mathrm{ndf} = -$")
    ax_top.text(
        0.04, 0.96,
        f"{CH_LAB[channel]}\nResolved SR\n{era}\n"
        rf"$M_{{W_R}}={M_WR:.0f}$ GeV, $M_N={M_N:.0f}$ GeV"
        f"\nGaussian / Both Constrained"
        f"\n" + rf"$\alpha_\mu={alpha_mu:.2f},\ \alpha_\sigma={alpha_sigma:.2f}$"
        f"\nseed = {seed}"
        f"\n{chi2_label}",
        transform=ax_top.transAxes, fontsize=11, verticalalignment="top",
    )
    ax_top.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98),
                  fontsize=11, framealpha=0.90)

    pull_color = ("tab:green" if abs(pull) < 1
                  else "tab:orange" if abs(pull) < 2
                  else "tab:red")
    ax_top.text(
        0.98, 0.55,
        "Parameter pull\n"
        rf"pull$_{{{plabel}}}$ = $({plabel}_{{\rm fit}} - "
        rf"{plabel}_{{\rm truth}})/\sigma_{{{plabel}}}$"
        "\n"
        rf"        = $({val:.1f} - {truth:.1f})/{err:.1f}$"
        "\n"
        rf"        = $\mathbf{{{pull:+.2f}\,\sigma}}$",
        transform=ax_top.transAxes, fontsize=12,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=pull_color, linewidth=2.0, alpha=0.95),
    )

    # ---- Bottom panel: posterior on the parameter
    half_pull = max(abs(pull) + 1.0, 3.5)
    x_lo = val - half_pull * err
    x_hi = val + half_pull * err
    xs_g = np.linspace(x_lo, x_hi, 1000)
    gauss = np.exp(-0.5 * ((xs_g - val) / err) ** 2)
    ax_bot.fill_between(xs_g, 0, gauss, color="red", alpha=0.18)
    ax_bot.plot(xs_g, gauss, color="red", linewidth=2.0,
                label=rf"fit posterior on ${plabel}$:  "
                      rf"$\mathcal{{N}}({plabel}_{{\rm fit}}, \sigma_{{{plabel}}})$")
    ax_bot.axvline(truth, color="black", linestyle="--", linewidth=1.5,
                   label=rf"truth ${plabel}_{{\rm truth}}$")
    ax_bot.axvline(val, color="red", linestyle="-", linewidth=1.2, alpha=0.7)

    sigma_range = range(-4, 5) if param == "width" else range(-3, 4)
    for k in sigma_range:
        x = val + k * err
        if not (x_lo <= x <= x_hi):
            continue
        ax_bot.axvline(x, color="red", alpha=0.20, linewidth=0.8)
        ax_bot.text(x, 1.05, f"{k:+d}σ" if k != 0 else rf"${plabel}_{{\rm fit}}$",
                    color="red", fontsize=11, ha="center", va="bottom",
                    transform=ax_bot.get_xaxis_transform())

    arrow_y = 0.45
    ax_bot.annotate(
        "",
        xy=(truth, arrow_y), xytext=(val, arrow_y),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.6),
    )
    midx = 0.5 * (val + truth)
    ax_bot.text(
        midx, arrow_y + 0.03,
        rf"pull = $\mathbf{{{pull:+.2f}\,\sigma}}$",
        ha="center", va="bottom", fontsize=13,
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                  edgecolor="black", alpha=0.95),
    )

    ax_bot.set_xlabel(rf"${plabel}$ [GeV]", fontsize=18)
    ax_bot.set_ylabel("Fit posterior  (a.u.)", fontsize=14)
    ax_bot.set_xlim(x_lo, x_hi)
    ax_bot.set_ylim(0.0, 1.18)
    ax_bot.set_yticks([])
    ax_bot.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98),
                  fontsize=11, framealpha=0.9)

    name = "mu" if param == "mu" else "width"
    if out_path is None:
        out_path = (out_dir / f"pull_demo_{name}_{mass}_{channel}_gauss_both_"
                              f"seed{seed}_n{n_events}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return True


def do_pull_demo_outliers(rows, out_dir, n_events_list, *,
                          alpha_mu, alpha_sigma_by_channel,
                          n_outliers, era, dir_, topology,
                          curated=False, use_moments=False):
    """If curated=True, use CURATED_MASSES instead of outliers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for n_ev in n_events_list:
        if not curated:
            stats = per_cell_stats(rows, n_ev)
            if not stats:
                logger.warning("pull_demo: no stats at N=%d, skipping", n_ev)
                continue
            channels = sorted({ch for (ch, _) in stats})
        else:
            channels = sorted({r["channel"] for r in rows})
        for channel in channels:
            alpha_sigma = alpha_sigma_by_channel.get(
                channel, alpha_sigma_by_channel.get("default", 0.05))
            for param in ("mu", "width"):
                if curated:
                    masses_to_plot = [(m, None) for m in CURATED_MASSES]
                else:
                    masses_to_plot = find_outliers(stats, channel, param, n_outliers)
                logger.info("pull_demo: N=%d  %s  param=%s  α_σ=%.3f  cells=%s",
                            n_ev, channel, param, alpha_sigma,
                            [t[0] for t in masses_to_plot])
                for (mass, _) in masses_to_plot:
                    ok = make_pull_demo_one(
                        channel, mass, n_ev, param,
                        alpha_mu, alpha_sigma, out_dir,
                        era, dir_, topology,
                        use_moments=use_moments,
                    )
                    if not ok:
                        logger.warning("  failed: %s/%s/%s", channel, mass, param)


def main():
    args = parse_args()
    setup_logging(args.verbose)

    logger.info("Reading %s", args.input)
    rows = load_rows(args.input)
    logger.info("  %d rows", len(rows))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    alpha_sigma_by_channel = {"default": args.alpha_sigma,
                              "ee": args.alpha_sigma}
    if args.alpha_sigma_mumu is not None:
        alpha_sigma_by_channel["mumu"] = args.alpha_sigma_mumu

    if args.what in ("2d_pulls", "all"):
        do_2d_pulls(rows, args.output_dir / "2d_pulls")
    if args.what in ("1d_pulls", "all"):
        do_1d_pulls(rows, args.output_dir / "1d_pulls", args.n_events,
                    alpha_mu=args.alpha_mu,
                    alpha_sigma_by_channel=alpha_sigma_by_channel,
                    era=args.era, dir_=args.dir, topology=args.topology,
                    use_moments=args.use_moments)
    if args.what == "cell_offsets":  # legacy, not in "all"
        do_cell_offsets(rows, args.output_dir / "cell_offsets", args.n_events,
                        curated=args.curated)
    if args.what == "pull_demo":  # legacy, not in "all"
        do_pull_demo_outliers(rows, args.output_dir / "pull_demo",
                              args.n_events,
                              alpha_mu=args.alpha_mu,
                              alpha_sigma_by_channel=alpha_sigma_by_channel,
                              n_outliers=args.n_outliers,
                              era=args.era, dir_=args.dir,
                              topology=args.topology,
                              curated=args.curated,
                              use_moments=args.use_moments)


if __name__ == "__main__":
    main()
