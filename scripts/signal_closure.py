#!/usr/bin/env python3
"""
signal_closure.py — Signal closure validation for Run 2 → Run 3 proxy usage.

Validates whether Run 2 UL18 signal MC can substitute for Run 3 2024 signal MC
by comparing distributions across campaigns and between gen/reco levels.

Produces:
  1) Run 2 vs Run 3 gen-level shape overlays with chi2/ndf + KS p-values
  2) Run 2 vs Run 3 reco-level shape overlays with chi2/ndf + KS p-values
  3) Gen vs reco overlays within each campaign (acceptance x efficiency effects)
  4) Cutflow efficiency comparison (bar chart)
  5) Summary CSV with all quantitative metrics across mass points

Usage:
    python WR_Plotter/scripts/signal_closure.py
    python WR_Plotter/scripts/signal_closure.py --mass WR2000_N1900
    python WR_Plotter/scripts/signal_closure.py --mass WR2000_N1900 --output-dir ./my_plots
    python WR_Plotter/scripts/signal_closure.py --no-normalize
"""

import argparse
import csv
import logging
import re
from pathlib import Path
import sys

import uproot
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wrplotter.io import repo_root, save_figure
from wrplotter.histogram_utils import extract_hist_data, rebin_histogram
from wrplotter.cli_utils import setup_logging

setup_logging()

hep.style.use("CMS")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COMMON_POINTS = [
    "WR2000_N100",
    "WR2000_N1900",
    "WR4000_N100",
    "WR4000_N3900",
    "WR6000_N100",
    "WR6000_N5900",
]

ROOTFILE_DIR = repo_root() / "rootfiles"

RUN2_ERA = "RunIISummer20UL18"
RUN3_ERA = "RunIII2024Summer24"

RUN2_LABEL = "Run 2 UL18"
RUN3_LABEL = "Run 3 2024"

RUN2_COLOR = "C0"
RUN3_COLOR = "C1"

GEN_COLOR = "C2"
RECO_COLOR = "C3"

# Gen-level variables to compare.
# Format: (gen_name, reco_name, label, unit, rebin_factor)
VARIABLE_PAIRS = [
    ("gen_pt_leading_lepton",    "pt_leading_lepton",    r"$p_{T}$ leading lepton",    "GeV", 5),
    ("gen_eta_leading_lepton",   "eta_leading_lepton",   r"$\eta$ leading lepton",     "",    2),
    ("gen_pt_subleading_lepton", "pt_subleading_lepton", r"$p_{T}$ subleading lepton", "GeV", 5),
    ("gen_eta_subleading_lepton","eta_subleading_lepton",r"$\eta$ subleading lepton",  "",    2),
    ("gen_pt_leading_jet",       "pt_leading_jet",       r"$p_{T}$ leading jet",       "GeV", 5),
    ("gen_eta_leading_jet",      "eta_leading_jet",      r"$\eta$ leading jet",        "",    2),
    ("gen_pt_subleading_jet",    "pt_subleading_jet",    r"$p_{T}$ subleading jet",    "GeV", 5),
    ("gen_eta_subleading_jet",   "eta_subleading_jet",   r"$\eta$ subleading jet",     "",    2),
    ("gen_mass_dilepton",        "mass_dilepton",        r"$m_{\ell\ell}$",            "GeV", 50),
    ("gen_mass_dijet",           "mass_dijet",           r"$m_{jj}$",                  "GeV", 10),
    ("gen_mass_fourobject",      "mass_fourobject",      r"$m_{\ell\ell jj}$",         "GeV", 20),
]

# Reco-only variables (no gen counterpart) — only for Run2 vs Run3 reco comparison.
RECO_ONLY_VARIABLES = [
    ("pt_fourobject", r"$p_{T,\ell\ell jj}$", "GeV", 20),
]

GEN_REGION = "gen_inclusive"

RECO_REGIONS = [
    ("wr_ee_resolved_sr",        r"$ee$ resolved SR"),
    ("wr_mumu_resolved_sr",      r"$\mu\mu$ resolved SR"),
    ("wr_ee_resolved_dy_cr_1",   r"$ee$ resolved DY CR (60-120)"),
    ("wr_mumu_resolved_dy_cr_1", r"$\mu\mu$ resolved DY CR (60-120)"),
]

CUTFLOW_STEPS = [
    r"$\geq$2 jets (p$_{T}$/$\eta$)",
    r"$\geq$2 jets (ID)",
    r"2 leptons (p$_{T}$/$\eta$)",
    r"2 leptons (ID)",
    "Trigger",
    r"$\Delta R > 0.4$",
    r"$m_{\ell\ell jj} > 800$",
    r"$m_{\ell\ell} > 200$",
    r"$m_{\ell\ell} > 400$",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def root_file_path(era, mass):
    """Build the path to the closure ROOT file for a given era and mass point."""
    if era == RUN2_ERA:
        return ROOTFILE_DIR / "RunII" / "2018" / era / "closure" / f"ClosureStudy_signal_{mass}.root"
    else:
        return ROOTFILE_DIR / "Run3" / "2024" / era / "closure" / f"ClosureStudy_signal_{mass}.root"


def load_hist_1d(fpath, region, hist_name):
    """Load a histogram from a ROOT file and project out the process axis if present."""
    key = f"{region}/{hist_name}_{region}"
    try:
        with uproot.open(fpath) as f:
            h = f[key].to_hist()
    except Exception as e:
        logging.warning("Could not load %s from %s: %s", key, fpath, e)
        return None

    import hist as hist_mod
    num_axes = [i for i, ax in enumerate(h.axes)
                if isinstance(ax, (hist_mod.axis.Regular, hist_mod.axis.Variable))]
    if len(num_axes) == 1 and h.ndim > 1:
        h = h.project(num_axes[0])

    return h


def normalize_to_unity(vals, errs, bin_widths):
    """Normalise histogram values and errors so that the integral equals 1."""
    integral = np.sum(vals * bin_widths)
    if integral <= 0:
        return vals, errs
    return vals / integral, errs / integral


def extract(h):
    """Extract edges, values, errors from a hist.Hist object."""
    edges, _, vals, errs = extract_hist_data(h)
    return edges, vals, errs


def format_mass(mass):
    """Format mass point: WR2000_N1900 -> (m_WR, m_N) = (2000, 1900) GeV."""
    m = re.search(r"WR(\d+)_N(\d+)", mass)
    if m:
        return rf"$(m_{{W_R}}, m_N) = ({m.group(1)}, {m.group(2)})$ GeV"
    return mass


# ---------------------------------------------------------------------------
# Quantitative tests
# ---------------------------------------------------------------------------
def chi2_test(vals1, errs1, vals2, errs2):
    """
    Compute chi2/ndf between two binned distributions.

    Uses the combined bin variance as the denominator:
        chi2 = sum( (v1 - v2)^2 / (sig1^2 + sig2^2) )
    Bins where both histograms are zero are excluded.

    Returns (chi2, ndf, chi2_per_ndf).
    """
    var_sum = errs1**2 + errs2**2
    mask = var_sum > 0
    if np.sum(mask) == 0:
        return 0.0, 0, 0.0

    chi2 = np.sum((vals1[mask] - vals2[mask])**2 / var_sum[mask])
    ndf = int(np.sum(mask)) - 1  # -1 if normalised
    chi2_per_ndf = chi2 / ndf if ndf > 0 else 0.0
    return chi2, ndf, chi2_per_ndf


def ks_test(vals1, vals2):
    """
    Kolmogorov-Smirnov test between two binned distributions.

    Converts bin contents to pseudo-samples for scipy.stats.ks_2samp.
    Returns (KS statistic, p-value).
    """
    # Build CDF from bin values (treat each bin as weighted)
    cdf1 = np.cumsum(vals1)
    cdf2 = np.cumsum(vals2)
    total1 = cdf1[-1] if cdf1[-1] > 0 else 1.0
    total2 = cdf2[-1] if cdf2[-1] > 0 else 1.0
    cdf1 = cdf1 / total1
    cdf2 = cdf2 / total2

    ks_stat = np.max(np.abs(cdf1 - cdf2))

    # Effective number of entries for p-value approximation
    n_eff = total1 * total2 / (total1 + total2) if (total1 + total2) > 0 else 1.0
    # Kolmogorov distribution approximation
    lam = (np.sqrt(n_eff) + 0.12 + 0.11 / np.sqrt(n_eff)) * ks_stat
    # Two-sided p-value from Kolmogorov distribution (series approximation)
    if lam < 1e-10:
        p_value = 1.0
    else:
        p_value = 2.0 * np.sum([(-1)**(k-1) * np.exp(-2 * k**2 * lam**2)
                                 for k in range(1, 101)])
        p_value = max(0.0, min(1.0, p_value))

    return ks_stat, p_value


def compute_metrics(h1, h2, rebin_n=1, normalize=True):
    """
    Rebin, optionally normalise, and compute chi2 + KS between two histograms.

    Returns dict with keys: chi2, ndf, chi2_ndf, ks_stat, ks_pval.
    Returns None if either histogram is None.
    """
    if h1 is None or h2 is None:
        return None

    hr1 = rebin_histogram(h1, rebin_n) if rebin_n > 1 else h1
    hr2 = rebin_histogram(h2, rebin_n) if rebin_n > 1 else h2

    edges1, vals1, errs1 = extract(hr1)
    edges2, vals2, errs2 = extract(hr2)

    bw = np.diff(edges1)

    if normalize:
        vals1, errs1 = normalize_to_unity(vals1, errs1, bw)
        vals2, errs2 = normalize_to_unity(vals2, errs2, bw)

    chi2, ndf, chi2_ndf = chi2_test(vals1, errs1, vals2, errs2)
    ks_stat, ks_pval = ks_test(vals1, vals2)

    return dict(chi2=chi2, ndf=ndf, chi2_ndf=chi2_ndf,
                ks_stat=ks_stat, ks_pval=ks_pval)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _annotate_metrics(ax, metrics, y_start=0.70, fontsize=12):
    """Add chi2/ndf and KS p-value text to a plot axis."""
    if metrics is None:
        return
    chi2_str = (rf"$\chi^2$/ndf = {metrics['chi2']:.1f}/{metrics['ndf']}"
                rf" = {metrics['chi2_ndf']:.2f}")
    ks_str = f"KS p-value = {metrics['ks_pval']:.3f}"
    ax.text(0.05, y_start, chi2_str,
            transform=ax.transAxes, fontsize=fontsize, va="top")
    ax.text(0.05, y_start - 0.05, ks_str,
            transform=ax.transAxes, fontsize=fontsize, va="top")


def plot_overlay(h_a, h_b, *,
                 xlabel, label_a, label_b, color_a, color_b,
                 region_label, mass_label,
                 ratio_label="B / A",
                 normalize=True, rebin_n=1, logy=False,
                 metrics=None):
    """
    Generic overlay of two 1D histograms with a ratio panel.

    Used for both Run2-vs-Run3 and gen-vs-reco comparisons.
    """
    ha = rebin_histogram(h_a, rebin_n) if rebin_n > 1 else h_a
    hb = rebin_histogram(h_b, rebin_n) if rebin_n > 1 else h_b

    edges_a, vals_a, errs_a = extract(ha)
    edges_b, vals_b, errs_b = extract(hb)

    bw = np.diff(edges_a)

    if normalize:
        vals_a, errs_a = normalize_to_unity(vals_a, errs_a, bw)
        vals_b, errs_b = normalize_to_unity(vals_b, errs_b, bw)

    centers = 0.5 * (edges_a[:-1] + edges_a[1:])

    fig, (ax, rax) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.10},
        figsize=(8, 7),
    )

    # --- Main panel ---
    ax.step(edges_a, np.append(vals_a, vals_a[-1]), where="post",
            color=color_a, linewidth=1.5, label=label_a)
    ax.errorbar(centers, vals_a, yerr=errs_a,
                fmt="none", capsize=2, color=color_a)

    ax.step(edges_b, np.append(vals_b, vals_b[-1]), where="post",
            color=color_b, linewidth=1.5, label=label_b)
    ax.errorbar(centers, vals_b, yerr=errs_b,
                fmt="none", capsize=2, color=color_b)

    if logy:
        ax.set_yscale("log")
    ylabel = "Normalised" if normalize else "Events"
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=14, loc="upper right")

    ax.text(0.05, 0.96, f"{region_label}\n{mass_label}",
            transform=ax.transAxes, fontsize=14, va="top")
    _annotate_metrics(ax, metrics, y_start=0.82, fontsize=12)

    hep.cms.label(loc=0, ax=ax, data=False,
                  label="Work in Progress", fontsize=14)
    ax.tick_params(labelbottom=False)

    # --- Ratio panel ---
    mask = vals_a > 0
    ratio = np.ones_like(vals_a)
    ratio_err = np.zeros_like(vals_a)
    ratio[mask] = vals_b[mask] / vals_a[mask]
    ratio_err[mask] = np.sqrt(
        (errs_b[mask] / vals_a[mask])**2 +
        (vals_b[mask] * errs_a[mask] / (vals_a[mask]**2))**2
    )

    rax.errorbar(centers, ratio, yerr=ratio_err,
                 fmt="o", markersize=3, capsize=2, color="black")
    rax.axhline(1.0, ls="--", color="gray", linewidth=1)
    rax.set_xlabel(xlabel)
    rax.set_ylabel(ratio_label, fontsize=10)
    rax.set_ylim(0.5, 1.5)
    rax.set_xlim(edges_a[0], edges_a[-1])

    fig.subplots_adjust(left=0.14, right=0.96, top=0.92, bottom=0.12, hspace=0.10)
    return fig


def plot_cutflow(run2_file, run3_file, mass, output_dir):
    """Compare cumulative cutflow efficiencies between campaigns."""
    flavors = ["ee", "mumu"]
    flavor_labels = {
        "ee": r"$ee$ channel",
        "mumu": r"$\mu\mu$ channel",
    }

    for flavor in flavors:
        key = f"cutflow/{flavor}/cumulative"

        try:
            with uproot.open(run2_file) as f2:
                h2 = f2[key].to_hist()
            with uproot.open(run3_file) as f3:
                h3 = f3[key].to_hist()
        except Exception as e:
            logging.warning("Could not load cutflow for %s: %s", flavor, e)
            continue

        def _project_to_1d(h):
            import hist as hist_mod
            if h.ndim > 1:
                int_axes = [i for i, ax in enumerate(h.axes)
                            if isinstance(ax, (hist_mod.axis.IntCategory,
                                               hist_mod.axis.Regular,
                                               hist_mod.axis.Variable))]
                if int_axes:
                    return h.project(*int_axes)
                return h[sum, :]
            return h

        h2_1d = _project_to_1d(h2)
        h3_1d = _project_to_1d(h3)

        vals2 = h2_1d.values().flatten()
        vals3 = h3_1d.values().flatten()

        if vals2[0] > 0:
            eff2 = vals2 / vals2[0]
        else:
            eff2 = np.zeros_like(vals2)
        if vals3[0] > 0:
            eff3 = vals3 / vals3[0]
        else:
            eff3 = np.zeros_like(vals3)

        n_steps = min(len(eff2), len(eff3), len(CUTFLOW_STEPS))
        eff2 = eff2[:n_steps]
        eff3 = eff3[:n_steps]
        labels = CUTFLOW_STEPS[:n_steps]

        x = np.arange(n_steps)
        width = 0.35

        fig, (ax, rax) = plt.subplots(
            2, 1, sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.10},
            figsize=(12, 7),
        )

        ax.bar(x - width / 2, eff2, width, label=RUN2_LABEL, color=RUN2_COLOR, alpha=0.7)
        ax.bar(x + width / 2, eff3, width, label=RUN3_LABEL, color=RUN3_COLOR, alpha=0.7)

        ax.set_ylabel("Cumulative efficiency (relative to first cut)")
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 2.0)
        ax.legend(fontsize=14)
        ax.text(0.05, 0.96, f"{flavor_labels[flavor]}\n{format_mass(mass)}",
                transform=ax.transAxes, fontsize=14, va="top")
        hep.cms.label(loc=0, ax=ax, data=False,
                      label="Work in Progress", fontsize=14)

        mask = eff2 > 0
        ratio = np.ones_like(eff2)
        ratio[mask] = eff3[mask] / eff2[mask]

        rax.bar(x, ratio, 2 * width, color="gray", alpha=0.5)
        rax.axhline(1.0, ls="--", color="black", linewidth=1)
        rax.set_ylabel(f"{RUN3_LABEL} / {RUN2_LABEL}", fontsize=10)
        rax.set_ylim(0.5, 1.5)
        rax.set_xticks(x)
        rax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

        fig.subplots_adjust(left=0.14, right=0.96, top=0.92, bottom=0.12, hspace=0.10)

        outpath = output_dir / f"cutflow_{flavor}_{mass}.pdf"
        save_figure(fig, outpath)
        logging.info("    Saved %s", outpath)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------
def write_summary_csv(rows, output_path):
    """Write the summary metrics table to a CSV file."""
    fieldnames = ["mass", "comparison", "region", "variable",
                  "chi2", "ndf", "chi2_ndf", "ks_stat", "ks_pval"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Summary CSV written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Signal closure validation: Run 2 vs Run 3 + gen vs reco.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mass", "-m", type=str, default=None,
        help="Single mass point (e.g. WR2000_N1900). Default: all common points.",
    )
    p.add_argument(
        "--output-dir", "-o", type=str, default=None,
        help="Output directory. Default: WR_Plotter/plots/closure/",
    )
    p.add_argument(
        "--normalize", action="store_true", default=True,
        help="Normalise distributions to unit area (default).",
    )
    p.add_argument(
        "--no-normalize", action="store_false", dest="normalize",
        help="Do not normalise — plot absolute rates.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    masses = [args.mass] if args.mass else COMMON_POINTS

    if args.output_dir:
        out_base = Path(args.output_dir)
    else:
        out_base = repo_root() / "plots" / "closure"

    summary_rows = []

    for mass in masses:
        logging.info("=== %s ===", mass)
        mass_label = format_mass(mass)

        run2_file = root_file_path(RUN2_ERA, mass)
        run3_file = root_file_path(RUN3_ERA, mass)

        if not run2_file.exists():
            logging.error("  Missing Run 2 file: %s", run2_file)
            continue
        if not run3_file.exists():
            logging.error("  Missing Run 3 file: %s", run3_file)
            continue

        # =============================================================
        # 1) Run 2 vs Run 3: Gen-level
        # =============================================================
        logging.info("  [1/4] Gen-level: Run 2 vs Run 3")
        for gen_name, _, var_label, var_unit, rebin_n in VARIABLE_PAIRS:
            h2 = load_hist_1d(run2_file, GEN_REGION, gen_name)
            h3 = load_hist_1d(run3_file, GEN_REGION, gen_name)
            if h2 is None or h3 is None:
                logging.warning("    Skipping %s (missing histogram)", gen_name)
                continue

            metrics = compute_metrics(h2, h3, rebin_n, args.normalize)
            summary_rows.append(dict(
                mass=mass, comparison="Run2_vs_Run3_gen",
                region=GEN_REGION, variable=gen_name, **metrics,
            ))

            xlabel = f"Gen {var_label} [{var_unit}]" if var_unit else f"Gen {var_label}"
            fig = plot_overlay(
                h2, h3, xlabel=xlabel,
                label_a=RUN2_LABEL, label_b=RUN3_LABEL,
                color_a=RUN2_COLOR, color_b=RUN3_COLOR,
                region_label="Gen-level (inclusive)",
                mass_label=mass_label,
                ratio_label=f"{RUN3_LABEL} / {RUN2_LABEL}",
                normalize=args.normalize, rebin_n=rebin_n,
                metrics=metrics,
            )
            outpath = out_base / mass / "run2_vs_run3" / "gen" / f"{gen_name}_{mass}.pdf"
            save_figure(fig, outpath)
            logging.info("    %s  chi2/ndf=%.2f  KS p=%.3f",
                         gen_name, metrics["chi2_ndf"], metrics["ks_pval"])
            plt.close(fig)

        # =============================================================
        # 2) Run 2 vs Run 3: Reco-level
        # =============================================================
        logging.info("  [2/4] Reco-level: Run 2 vs Run 3")
        all_reco_vars = (
            [(name, label, unit, rb) for _, name, label, unit, rb in VARIABLE_PAIRS]
            + list(RECO_ONLY_VARIABLES)
        )

        for region_name, region_label in RECO_REGIONS:
            for reco_name, var_label, var_unit, rebin_n in all_reco_vars:
                h2 = load_hist_1d(run2_file, region_name, reco_name)
                h3 = load_hist_1d(run3_file, region_name, reco_name)
                if h2 is None or h3 is None:
                    continue

                metrics = compute_metrics(h2, h3, rebin_n, args.normalize)
                summary_rows.append(dict(
                    mass=mass, comparison="Run2_vs_Run3_reco",
                    region=region_name, variable=reco_name, **metrics,
                ))

                xlabel = f"{var_label} [{var_unit}]" if var_unit else var_label
                fig = plot_overlay(
                    h2, h3, xlabel=xlabel,
                    label_a=RUN2_LABEL, label_b=RUN3_LABEL,
                    color_a=RUN2_COLOR, color_b=RUN3_COLOR,
                    region_label=region_label, mass_label=mass_label,
                    ratio_label=f"{RUN3_LABEL} / {RUN2_LABEL}",
                    normalize=args.normalize, rebin_n=rebin_n,
                    metrics=metrics,
                )
                outpath = (out_base / mass / "run2_vs_run3" / region_name
                           / f"{reco_name}_{mass}.pdf")
                save_figure(fig, outpath)
                logging.info("    %s/%s  chi2/ndf=%.2f  KS p=%.3f",
                             region_name, reco_name,
                             metrics["chi2_ndf"], metrics["ks_pval"])
                plt.close(fig)

        # =============================================================
        # 3) Gen vs Reco (within each campaign)
        # =============================================================
        logging.info("  [3/4] Gen vs Reco (per campaign)")
        for era, era_label, era_color, root_file in [
            (RUN2_ERA, RUN2_LABEL, RUN2_COLOR, run2_file),
            (RUN3_ERA, RUN3_LABEL, RUN3_COLOR, run3_file),
        ]:
            era_short = "run2" if era == RUN2_ERA else "run3"
            for region_name, region_label in RECO_REGIONS:
                for gen_name, reco_name, var_label, var_unit, rebin_n in VARIABLE_PAIRS:
                    h_gen = load_hist_1d(root_file, GEN_REGION, gen_name)
                    h_reco = load_hist_1d(root_file, region_name, reco_name)
                    if h_gen is None or h_reco is None:
                        continue

                    metrics = compute_metrics(h_gen, h_reco, rebin_n, args.normalize)
                    summary_rows.append(dict(
                        mass=mass,
                        comparison=f"gen_vs_reco_{era_short}",
                        region=region_name, variable=reco_name, **metrics,
                    ))

                    xlabel = f"{var_label} [{var_unit}]" if var_unit else var_label
                    fig = plot_overlay(
                        h_gen, h_reco, xlabel=xlabel,
                        label_a=f"Gen ({era_label})",
                        label_b=f"Reco ({era_label})",
                        color_a=GEN_COLOR, color_b=RECO_COLOR,
                        region_label=region_label, mass_label=mass_label,
                        ratio_label="Reco / Gen",
                        normalize=args.normalize, rebin_n=rebin_n,
                        metrics=metrics,
                    )
                    outpath = (out_base / mass / f"gen_vs_reco_{era_short}"
                               / region_name / f"{reco_name}_{mass}.pdf")
                    save_figure(fig, outpath)
                    logging.info("    %s %s/%s  chi2/ndf=%.2f  KS p=%.3f",
                                 era_short, region_name, reco_name,
                                 metrics["chi2_ndf"], metrics["ks_pval"])
                    plt.close(fig)

        # =============================================================
        # 4) Cutflow comparison
        # =============================================================
        logging.info("  [4/4] Cutflow comparison")
        cutflow_dir = out_base / mass / "cutflow"
        plot_cutflow(run2_file, run3_file, mass, cutflow_dir)

    # =================================================================
    # Write summary CSV
    # =================================================================
    if summary_rows:
        csv_path = out_base / "closure_summary.csv"
        write_summary_csv(summary_rows, csv_path)

        # Print a quick digest to terminal
        logging.info("")
        logging.info("=== SUMMARY ===")
        logging.info("%-16s %-24s %-30s %-24s %8s %8s",
                     "Mass", "Comparison", "Region", "Variable", "chi2/ndf", "KS p")
        logging.info("-" * 120)
        for row in summary_rows:
            logging.info("%-16s %-24s %-30s %-24s %8.2f %8.3f",
                         row["mass"], row["comparison"], row["region"],
                         row["variable"], row["chi2_ndf"], row["ks_pval"])

    logging.info("Done.")


if __name__ == "__main__":
    main()
