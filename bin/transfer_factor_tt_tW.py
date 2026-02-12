#!/usr/bin/env python3
"""
Compute and plot the tt+tW transfer factor: the bin-by-bin ratio of
mass_fourobject histograms between resolved signal regions (ee and mumu)
and the resolved flavor control region (emu).

The plot includes constant fit lines and chi2/ndf annotations for both channels.

Usage:
    python3 bin/transfer_factor_tt_tW.py --era RunIII2024Summer24
    python3 bin/transfer_factor_tt_tW.py --era RunIISummer20UL18 --dir subdir
"""

# ── Standard library ────────────────────────────────────────────────────────────
import sys
import logging
import argparse
from pathlib import Path

# ── Third-party ────────────────────────────────────────────────────────────────
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wrplotter.io import repo_root, save_figure
from wrplotter.config import list_eras, load_lumi, load_plot_settings, index_plot_settings, get_var_cfg
from wrplotter.histogram_utils import rebin_histogram, load_histogram
from wrplotter.cli_utils import setup_logging

hep.style.use("CMS")

logger = logging.getLogger(__name__)

_ERA_CHOICES = list_eras()

# Colors from sample_groups/base.yaml
COLOR_EE = "#7a21dd"
COLOR_MUMU = "#e42536"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute tt+tW transfer factors: ratio of resolved SR (ee/mumu) to resolved flavor CR (emu)"
    )
    parser.add_argument(
        "--era", dest="era", type=str, choices=_ERA_CHOICES, required=True,
        help="Specify the era (e.g., RunIII2024Summer24, RunIISummer20UL18)"
    )
    parser.add_argument(
        "--dir", dest="dir", type=str, default="",
        help="Optional subdirectory under the input path"
    )
    parser.add_argument(
        "--output", "-o", dest="output", type=str, default=None,
        help="Output file path (default: plots/<era>/ratio_mass_fourobject_tt_tW.pdf)"
    )
    parser.add_argument(
        "--rebin", dest="rebin", type=int, default=None,
        help="Override rebin factor (integer). If not specified, uses value from YAML config."
    )
    parser.add_argument(
        "--variable-bins", dest="variable_bins", action="store_true",
        help="Use variable bin widths instead of uniform rebinning from YAML config."
    )
    parser.add_argument(
        "--numerator", dest="numerator", type=str, default="wr_ee_resolved_sr",
        help="Numerator region (default: wr_ee_resolved_sr)"
    )
    parser.add_argument(
        "--denominator", dest="denominator", type=str, default="wr_resolved_flavor_cr",
        help="Denominator region (default: wr_resolved_flavor_cr)"
    )
    return parser.parse_args()


def _load_hist(filepath: Path, region: str, variable: str = "mass_fourobject"):
    """Load a histogram by region/variable convention, returning None on failure."""
    hist_key = f"{region}/{variable}_{region}"
    try:
        return load_histogram(filepath, hist_key)
    except Exception as e:
        logger.warning("Could not load %s from %s: %s", hist_key, filepath, e)
        return None


def compute_ratio(numerator_hist, denominator_hist):
    """
    Compute bin-by-bin ratio of two histograms.

    Returns:
        ratio_values: numpy array of ratio values
        ratio_errors: numpy array of ratio uncertainties
        edges: bin edges
    """
    num_vals = numerator_hist.values()
    num_vars = numerator_hist.variances()
    den_vals = denominator_hist.values()
    den_vars = denominator_hist.variances()
    edges = numerator_hist.axes[0].edges

    # Suppress divide-by-zero warnings; bins with zero denominator get ratio=0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(den_vals > 0, num_vals / den_vals, 0.0)

        # Error propagation: sigma_r = r * sqrt((sigma_n/n)^2 + (sigma_d/d)^2)
        rel_err_num = np.where(num_vals > 0, np.sqrt(num_vars) / num_vals, 0.0)
        rel_err_den = np.where(den_vals > 0, np.sqrt(den_vars) / den_vals, 0.0)
        ratio_errors = ratio * np.sqrt(rel_err_num**2 + rel_err_den**2)

        ratio_errors = np.where(ratio > 0, ratio_errors, 0.0)

    return ratio, ratio_errors, edges


def _draw_ratio_panel(ax, centers, xerr, ratio_ee, ratio_ee_errors,
                      ratio_mumu, ratio_mumu_errors, era, lumi, com):
    """Draw the common elements shared by both zoomed and full ratio plots."""
    ax.errorbar(
        centers, ratio_ee, yerr=ratio_ee_errors, xerr=xerr,
        fmt='o', color=COLOR_EE, markersize=6, capsize=3,
        label=r'$\frac{\mathrm{Resolved\ SR\ (ee)}}{\mathrm{Resolved\ Flavor\ CR\ (e\mu)}}$'
    )
    ax.errorbar(
        centers, ratio_mumu, yerr=ratio_mumu_errors, xerr=xerr,
        fmt='s', color=COLOR_MUMU, markersize=6, capsize=3,
        label=r'$\frac{\mathrm{Resolved\ SR\ (\mu\mu)}}{\mathrm{Resolved\ Flavor\ CR\ (e\mu)}}$'
    )

    ax.set_xlabel(r'$m_{lljj}$ [GeV]', fontsize=20)
    ax.set_ylabel(r'Resolved SR / Resolved Flavor CR (e$\mu$)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    hep.cms.label(
        loc=0, ax=ax, data=False,
        text="Work in Progress",
        lumi=f"{lumi:.1f}",
        com=com,
        fontsize=20
    )
    ax.text(0.07, 0.95, era, transform=ax.transAxes,
            fontsize=20, ha='left', va='top')
    ax.text(0.07, 0.90, r'Simulated $t\bar{t}$ + tW', transform=ax.transAxes,
            fontsize=20, ha='left', va='top')


def _constant(x, c):
    """Constant model for curve_fit."""
    return np.full_like(x, c, dtype=float)


def chi2_fit_constant(values, errors, centers, fit_range=(0, 8000)):
    """
    Fit a constant to binned ratio data using chi-squared minimization.

    Uses scipy.optimize.curve_fit with absolute_sigma=True so the covariance
    matrix is scaled by the actual bin uncertainties (i.e. true chi-squared
    weighting), not rescaled to chi2/ndf=1. Bins with large uncertainties
    are automatically downweighted by the chi-squared fit.

    Returns:
        fit_val: best-fit constant
        fit_err: 1-sigma uncertainty on the constant
        chi2: chi-squared value
        ndf: number of degrees of freedom
    """
    mask = ((centers >= fit_range[0]) & (centers <= fit_range[1])
            & (values > 0) & (errors > 0))

    if np.sum(mask) < 2:
        return 0.0, 0.0, 0.0, 0

    x = centers[mask]
    y = values[mask]
    sigma = errors[mask]

    popt, pcov = curve_fit(_constant, x, y, p0=[np.mean(y)],
                           sigma=sigma, absolute_sigma=True)
    fit_val = popt[0]
    fit_err = np.sqrt(pcov[0, 0])

    residuals = (y - fit_val) / sigma
    chi2 = np.sum(residuals ** 2)
    ndf = len(y) - 1  # 1 free parameter

    return fit_val, fit_err, chi2, ndf


def plot_ratio(ratio_ee, ratio_ee_errors, ratio_mumu, ratio_mumu_errors, edges, era, lumi, output_path, com=13.6):
    """
    Create the ratio plot with both ee and mumu channels, including fit lines and chi2/ndf.
    """
    centers = 0.5 * (edges[:-1] + edges[1:])
    xerr = 0.5 * np.diff(edges)

    # Chi-squared fit for a constant transfer factor (full range)
    fit_ee, fit_ee_err, chi2_ee, ndf_ee = chi2_fit_constant(
        ratio_ee, ratio_ee_errors, centers)
    fit_mumu, fit_mumu_err, chi2_mumu, ndf_mumu = chi2_fit_constant(
        ratio_mumu, ratio_mumu_errors, centers)

    logger.info("Chi-squared constant fit (full range):")
    if ndf_ee > 0:
        logger.info("  ee:   %.4f +/- %.4f  (chi2/ndf = %.1f/%d = %.2f)",
                     fit_ee, fit_ee_err, chi2_ee, ndf_ee, chi2_ee / ndf_ee)
    else:
        logger.info("  ee:   fit failed (not enough bins)")
    if ndf_mumu > 0:
        logger.info("  mumu: %.4f +/- %.4f  (chi2/ndf = %.1f/%d = %.2f)",
                     fit_mumu, fit_mumu_err, chi2_mumu, ndf_mumu, chi2_mumu / ndf_mumu)
    else:
        logger.info("  mumu: fit failed (not enough bins)")

    # Create plot with fit lines and chi2/ndf
    fig, ax = plt.subplots()
    _draw_ratio_panel(ax, centers, xerr, ratio_ee, ratio_ee_errors,
                      ratio_mumu, ratio_mumu_errors, era, lumi, com)

    # Draw fit lines and uncertainty bands
    if ndf_ee > 0:
        ax.axhline(y=fit_ee, color=COLOR_EE, linestyle='--', linewidth=1.5)
        ax.axhspan(fit_ee - fit_ee_err, fit_ee + fit_ee_err,
                   color=COLOR_EE, alpha=0.12)
    if ndf_mumu > 0:
        ax.axhline(y=fit_mumu, color=COLOR_MUMU, linestyle='--', linewidth=1.5)
        ax.axhspan(fit_mumu - fit_mumu_err, fit_mumu + fit_mumu_err,
                   color=COLOR_MUMU, alpha=0.12)

    # Set x-axis range starting at 800 GeV
    ax.set_xlim(800, edges[-1])

    # Dynamically scale y-axis based on data + errors
    max_ee = np.max(ratio_ee + ratio_ee_errors)
    max_mumu = np.max(ratio_mumu + ratio_mumu_errors)
    y_max = max(max_ee, max_mumu) * 1.15  # Add 15% headroom
    ax.set_ylim(0, y_max)

    ax.legend(loc='upper right', fontsize=24)

    # Annotation text with fit value and chi2/ndf
    y_text = 0.80
    if ndf_mumu > 0:
        ax.text(0.07, y_text,
                rf'$\mu\mu$ fit: {fit_mumu:.3f} $\pm$ {fit_mumu_err:.3f}'
                rf'  ($\chi^2$/ndf = {chi2_mumu:.1f}/{ndf_mumu})',
                transform=ax.transAxes, fontsize=16, ha='left', va='top', color=COLOR_MUMU)
        y_text -= 0.05
    if ndf_ee > 0:
        ax.text(0.07, y_text,
                rf'ee fit: {fit_ee:.3f} $\pm$ {fit_ee_err:.3f}'
                rf'  ($\chi^2$/ndf = {chi2_ee:.1f}/{ndf_ee})',
                transform=ax.transAxes, fontsize=16, ha='left', va='top', color=COLOR_EE)

    fig.tight_layout()
    save_figure(fig, output_path)
    logger.info("Saved plot to: %s", output_path)
    plt.close(fig)

    # Create zoomed-in version with smaller y-axis range
    fig_zoom, ax_zoom = plt.subplots()
    _draw_ratio_panel(ax_zoom, centers, xerr, ratio_ee, ratio_ee_errors,
                      ratio_mumu, ratio_mumu_errors, era, lumi, com)

    # Draw fit lines and uncertainty bands
    if ndf_ee > 0:
        ax_zoom.axhline(y=fit_ee, color=COLOR_EE, linestyle='--', linewidth=1.5)
        ax_zoom.axhspan(fit_ee - fit_ee_err, fit_ee + fit_ee_err,
                        color=COLOR_EE, alpha=0.12)
    if ndf_mumu > 0:
        ax_zoom.axhline(y=fit_mumu, color=COLOR_MUMU, linestyle='--', linewidth=1.5)
        ax_zoom.axhspan(fit_mumu - fit_mumu_err, fit_mumu + fit_mumu_err,
                        color=COLOR_MUMU, alpha=0.12)

    ax_zoom.set_xlim(800, 4000)
    ax_zoom.set_ylim(0, 1.4)  # Zoomed y-axis
    ax_zoom.legend(loc='upper right', fontsize=24)

    # Same annotations for zoomed plot
    y_text = 0.80
    if ndf_mumu > 0:
        ax_zoom.text(0.07, y_text,
                     rf'$\mu\mu$ fit: {fit_mumu:.3f} $\pm$ {fit_mumu_err:.3f}'
                     rf'  ($\chi^2$/ndf = {chi2_mumu:.1f}/{ndf_mumu})',
                     transform=ax_zoom.transAxes, fontsize=16, ha='left', va='top', color=COLOR_MUMU)
        y_text -= 0.05
    if ndf_ee > 0:
        ax_zoom.text(0.07, y_text,
                     rf'ee fit: {fit_ee:.3f} $\pm$ {fit_ee_err:.3f}'
                     rf'  ($\chi^2$/ndf = {chi2_ee:.1f}/{ndf_ee})',
                     transform=ax_zoom.transAxes, fontsize=16, ha='left', va='top', color=COLOR_EE)

    fig_zoom.tight_layout()
    output_path_zoom = output_path.parent / (output_path.stem + "_zoom" + output_path.suffix)
    save_figure(fig_zoom, output_path_zoom)
    logger.info("Saved zoomed plot to: %s", output_path_zoom)
    plt.close(fig_zoom)


def main():
    args = parse_args()
    setup_logging()

    # Load era info
    info = load_lumi(args.era)
    run = info["run"]
    year = info["year"]
    lumi = info["lumi"]
    com  = info.get("com", 13.6)

    # Build input path
    input_dir = repo_root() / "rootfiles" / run / str(year) / args.era
    if args.dir:
        input_dir = input_dir / args.dir

    filepath = input_dir / "WRAnalyzer_tt_tW.root"

    if not filepath.exists():
        logger.error("ROOT file not found: %s", filepath)
        sys.exit(1)

    logger.info("Loading histograms from: %s", filepath)

    # Load histograms for ee channel
    ee_num_hist = _load_hist(filepath, "wr_ee_resolved_sr")
    den_hist = _load_hist(filepath, "wr_resolved_flavor_cr")

    # Load histograms for mumu channel
    mumu_num_hist = _load_hist(filepath, "wr_mumu_resolved_sr")

    if ee_num_hist is None:
        logger.error("Could not load ee numerator histogram from region 'wr_ee_resolved_sr'")
        sys.exit(1)
    if mumu_num_hist is None:
        logger.error("Could not load mumu numerator histogram from region 'wr_mumu_resolved_sr'")
        sys.exit(1)
    if den_hist is None:
        logger.error("Could not load denominator histogram from region 'wr_resolved_flavor_cr'")
        sys.exit(1)

    logger.info("Loaded ee numerator: wr_ee_resolved_sr/mass_fourobject")
    logger.info("Loaded mumu numerator: wr_mumu_resolved_sr/mass_fourobject")
    logger.info("Loaded denominator: wr_resolved_flavor_cr/mass_fourobject")

    # Rebin histograms
    if args.variable_bins:
        variable_edges = [0, 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 8000]
        ee_num_hist = rebin_histogram(ee_num_hist, variable_edges)
        mumu_num_hist = rebin_histogram(mumu_num_hist, variable_edges)
        den_hist = rebin_histogram(den_hist, variable_edges)
        logger.info("Rebinned with variable edges: %s", variable_edges)
    elif args.rebin:
        ee_num_hist = rebin_histogram(ee_num_hist, args.rebin)
        mumu_num_hist = rebin_histogram(mumu_num_hist, args.rebin)
        den_hist = rebin_histogram(den_hist, args.rebin)
        logger.info("Rebinned with factor: %d", args.rebin)
    else:
        plot_settings = load_plot_settings(args.era)
        region_cfgs, common_vars = index_plot_settings(plot_settings)
        num_vcfg = get_var_cfg(region_cfgs, common_vars, "wr_ee_resolved_sr", "mass_fourobject")
        rebin_factor = num_vcfg.get('rebin', 1) if num_vcfg else 1
        ee_num_hist = rebin_histogram(ee_num_hist, rebin_factor)
        mumu_num_hist = rebin_histogram(mumu_num_hist, rebin_factor)
        den_hist = rebin_histogram(den_hist, rebin_factor)
        logger.info("Rebinned from YAML config: %s", rebin_factor)

    # Compute ratios
    ratio_ee, ratio_ee_errors, edges = compute_ratio(ee_num_hist, den_hist)
    ratio_mumu, ratio_mumu_errors, _ = compute_ratio(mumu_num_hist, den_hist)

    logger.info("Ratio values per bin (ee):")
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        logger.info("  [%.0f, %.0f): %.4f +/- %.4f", low, high, ratio_ee[i], ratio_ee_errors[i])

    logger.info("Ratio values per bin (mumu):")
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        logger.info("  [%.0f, %.0f): %.4f +/- %.4f", low, high, ratio_mumu[i], ratio_mumu_errors[i])

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = repo_root() / "plots" / run / str(year) / args.era
        if args.dir:
            output_dir = output_dir / args.dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"ratio_mass_fourobject_tt_tW_{args.era}.pdf"

    # Create plot
    plot_ratio(
        ratio_ee, ratio_ee_errors,
        ratio_mumu, ratio_mumu_errors,
        edges, args.era, lumi,
        output_path, com=com
    )


if __name__ == '__main__':
    main()
