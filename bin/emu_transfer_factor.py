#!/usr/bin/env python3
"""
Compute and plot the emu transfer factor (R_emu): the bin-by-bin ratio of
mass histograms between same-flavor signal regions (ee and mumu) and the
emu flavor sideband control region.

The plot includes constant fit lines and chi2/ndf annotations for both channels.

Usage:
    # All eras, resolved + boosted:
    python3 bin/emu_transfer_factor.py

    # Single era, resolved + boosted:
    python3 bin/emu_transfer_factor.py --era RunIII2024Summer24

    # Resolved only:
    python3 bin/emu_transfer_factor.py --era RunIII2024Summer24 --resolved

    # TF study: combine _tf_low (m < 800) + standard SR (m > 800) with split fits:
    python3 bin/emu_transfer_factor.py --era RunIII2024Summer24 --tf-regions --input WRAnalyzer_TTbar.root

    # Boosted TF study:
    python3 bin/emu_transfer_factor.py --era RunIII2024Summer24 --boosted --tf-regions --input WRAnalyzer_TTbar.root
"""

# ── Standard library ────────────────────────────────────────────────────────────
import sys
import logging
import argparse
import subprocess
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
from wrplotter.cli_utils import setup_logging, add_era_args

hep.style.use("CMS")

logger = logging.getLogger(__name__)

COLOR_EE = "red"
COLOR_MUMU = "blue"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute emu transfer factor (R_emu): ratio of SR (ee/mumu) to flavor sideband CR (emu)"
    )
    add_era_args(parser, required=False)
    parser.add_argument(
        "--output", "-o", dest="output", type=str, default=None,
        help="Output file path (default: plots/<era>/ratio_<variable>_tt_tW.pdf)"
    )
    parser.add_argument(
        "--rebin", dest="rebin", type=int, default=None,
        help="Override rebin factor (integer). If not specified, uses value from YAML config."
    )
    parser.add_argument(
        "--variable-bins", dest="variable_bins", nargs="?", const="default",
        metavar="EDGES",
        help="Use variable bin widths. Optionally provide comma-separated edges "
             "(e.g. '0,200,400,600,800,1200,2000,8000'). "
             "Without an argument, uses built-in defaults appropriate for --tf-regions or standard mode."
    )
    parser.add_argument(
        "--input", dest="input_file", type=str, default=None,
        help="Override ROOT file name (default: WRAnalyzer_tt_tW.root)"
    )
    parser.add_argument(
        "--tf-regions", dest="tf_regions", action="store_true",
        help="Load _tf_low (m < 800) and standard SR (m > 800) regions; fit separately below/above 800 GeV"
    )
    parser.add_argument(
        "--boosted", dest="boosted", action="store_true",
        help="Boosted only (default: both resolved and boosted)"
    )
    parser.add_argument(
        "--resolved", dest="resolved", action="store_true",
        help="Resolved only (default: both resolved and boosted)"
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
                      ratio_mumu, ratio_mumu_errors, edges, era, lumi, com,
                      boosted=False):
    """Draw the common elements shared by both zoomed and full ratio plots."""
    cat = "Boosted" if boosted else "Resolved"
    mass_label = r'$m_{lj}$' if boosted else r'$m_{lljj}$'

    ax.stairs(ratio_ee, edges, color=COLOR_EE, linewidth=2,
              label=r'ee/e$\mu$')
    ax.errorbar(centers, ratio_ee, yerr=ratio_ee_errors,
                fmt='none', ecolor=COLOR_EE, capsize=3)

    ax.stairs(ratio_mumu, edges, color=COLOR_MUMU, linewidth=2,
              label=r'$\mu\mu$/e$\mu$')
    ax.errorbar(centers, ratio_mumu, yerr=ratio_mumu_errors,
                fmt='none', ecolor=COLOR_MUMU, capsize=3)

    ax.set_xlabel(f'{mass_label} [GeV]', fontsize=20)
    ax.set_ylabel(rf'{cat} SR / {cat} Flavor CR (e$\mu$)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    hep.cms.label(
        loc=0, ax=ax, data=False,
        label="Work in Progress",
        lumi=f"{lumi:.1f}",
        com=com,
        fontsize=20
    )
    ax.text(0.07, 0.95, era, transform=ax.transAxes,
            fontsize=20, ha='left', va='top')
    ax.text(0.07, 0.90, r'Simulated $t\bar{t}$', transform=ax.transAxes,
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


def _draw_fits_and_annotations(ax, fit_ranges, range_labels, fits_ee, fits_mumu):
    """Draw dotted fit lines with p0 legend entries on a ratio panel."""
    for i, (fr, rl) in enumerate(zip(fit_ranges, range_labels)):
        x_lo, x_hi = fr
        fv_ee, ferr_ee, chi2_ee, ndf_ee = fits_ee[i]
        fv_mumu, ferr_mumu, chi2_mumu, ndf_mumu = fits_mumu[i]
        prefix = f"{rl}: " if len(range_labels) > 1 else ""

        if ndf_ee > 0:
            ax.plot([x_lo, x_hi], [fv_ee, fv_ee], ':',
                    color=COLOR_EE, linewidth=1.5,
                    label=rf'{prefix}$p_{{0}}$ ={fv_ee:.3f} $\pm$ {ferr_ee:.5f} '
                          rf'($\chi^2$ = {chi2_ee:.2f})')
        if ndf_mumu > 0:
            ax.plot([x_lo, x_hi], [fv_mumu, fv_mumu], ':',
                    color=COLOR_MUMU, linewidth=1.5,
                    label=rf'{prefix}$p_{{0}}$ ={fv_mumu:.3f} $\pm$ {ferr_mumu:.5f} '
                          rf'($\chi^2$ = {chi2_mumu:.2f})')

    if len(fit_ranges) > 1:
        ax.axvline(x=fit_ranges[0][1], color='gray', linestyle=':', linewidth=1, alpha=0.5)


def compute_fits(ratio_ee, ratio_ee_errors, ratio_mumu, ratio_mumu_errors,
                 edges, tf_regions=False, fit_ranges=None, range_labels=None):
    """Compute constant fits to the ratio data.

    Returns:
        fits_ee: list of (fit_val, fit_err, chi2, ndf) per range
        fits_mumu: list of (fit_val, fit_err, chi2, ndf) per range
        fit_ranges: list of (lo, hi) tuples
        range_labels: list of label strings
    """
    centers = 0.5 * (edges[:-1] + edges[1:])

    if fit_ranges is None:
        if tf_regions:
            fit_ranges = [(0, 800), (800, 8000)]
            range_labels = ["low-mass", "high-mass"]
        else:
            fit_ranges = [(0, 8000)]
            range_labels = ["full"]

    fits_ee = [chi2_fit_constant(ratio_ee, ratio_ee_errors, centers, fit_range=fr)
               for fr in fit_ranges]
    fits_mumu = [chi2_fit_constant(ratio_mumu, ratio_mumu_errors, centers, fit_range=fr)
                 for fr in fit_ranges]

    for label, fe, fm in zip(range_labels, fits_ee, fits_mumu):
        logger.info("Chi-squared constant fit (%s range):", label)
        fv, ferr, chi2, ndf = fe
        if ndf > 0:
            logger.info("  ee:   %.4f +/- %.4f  (chi2/ndf = %.1f/%d = %.2f)",
                         fv, ferr, chi2, ndf, chi2 / ndf)
        else:
            logger.info("  ee:   fit failed (not enough bins)")
        fv, ferr, chi2, ndf = fm
        if ndf > 0:
            logger.info("  mumu: %.4f +/- %.4f  (chi2/ndf = %.1f/%d = %.2f)",
                         fv, ferr, chi2, ndf, chi2 / ndf)
        else:
            logger.info("  mumu: fit failed (not enough bins)")

    return fits_ee, fits_mumu, fit_ranges, range_labels


def plot_ratio(ratio_ee, ratio_ee_errors, ratio_mumu, ratio_mumu_errors,
               edges, era, lumi, output_path, fits_ee, fits_mumu,
               fit_ranges, range_labels, com=13.6,
               boosted=False, xlim=None):
    """
    Create the ratio plot with both ee and mumu channels, including fit lines and chi2/ndf.
    """
    centers = 0.5 * (edges[:-1] + edges[1:])
    xerr = 0.5 * np.diff(edges)

    if xlim is not None:
        x_start, x_end = xlim
    else:
        x_start, x_end = 800, edges[-1]

    fig, ax = plt.subplots()
    _draw_ratio_panel(ax, centers, xerr, ratio_ee, ratio_ee_errors,
                      ratio_mumu, ratio_mumu_errors, edges, era, lumi, com,
                      boosted=boosted)
    _draw_fits_and_annotations(ax, fit_ranges, range_labels, fits_ee, fits_mumu)

    ax.set_xlim(x_start, x_end)

    visible = (centers >= x_start) & (centers <= x_end)
    max_vals = []
    if np.any(visible & (ratio_ee > 0)):
        max_vals.append(np.max((ratio_ee + ratio_ee_errors)[visible & (ratio_ee > 0)]))
    if np.any(visible & (ratio_mumu > 0)):
        max_vals.append(np.max((ratio_mumu + ratio_mumu_errors)[visible & (ratio_mumu > 0)]))
    y_max = max(max_vals) * 1.15 if max_vals else 1.0
    ax.set_ylim(0, y_max)

    ax.legend(loc='upper right', fontsize=16)

    fig.tight_layout()
    save_figure(fig, output_path)
    logger.info("Saved plot to: %s", output_path)
    plt.close(fig)


def plot_channel_ratio(ratio, ratio_errors, edges, era, lumi, output_path,
                       fits, fit_ranges, range_labels,
                       channel_label, num_label, den_label,
                       color, marker='o', com=13.6, xlim=None):
    """Ratio plot for a single boosted jet-flavor channel (eJ or muJ)."""
    centers = 0.5 * (edges[:-1] + edges[1:])

    if xlim is not None:
        x_start, x_end = xlim
    else:
        x_start, x_end = 800, edges[-1]

    def _make_panel(ax):
        ax.stairs(ratio, edges, color=color, linewidth=2,
                  label=rf'$\frac{{\mathrm{{{num_label}}}}}{{\mathrm{{{den_label}}}}}$')
        ax.errorbar(centers, ratio, yerr=ratio_errors,
                    fmt='none', ecolor=color, capsize=3)

        for i, fr in enumerate(fit_ranges):
            fv, ferr, chi2, ndf = fits[i]
            prefix = f"{range_labels[i]}: " if len(range_labels) > 1 else ""
            if ndf > 0:
                ax.plot([fr[0], fr[1]], [fv, fv], ':',
                        color=color, linewidth=1.5,
                        label=rf'{prefix}$p_{{0}}$ ={fv:.3f} $\pm$ {ferr:.5f} '
                              rf'($\chi^2$ = {chi2:.2f})')
        if len(fit_ranges) > 1:
            ax.axvline(x=fit_ranges[0][1], color='gray', linestyle=':',
                       linewidth=1, alpha=0.5)

        ax.set_xlabel(r'$m_{\ell J}$ [GeV]', fontsize=20)
        ax.set_ylabel(r'Boosted SR / Boosted Flavor CR', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress",
                      lumi=f"{lumi:.1f}", com=com, fontsize=20)
        ax.text(0.07, 0.95, era, transform=ax.transAxes,
                fontsize=20, ha='left', va='top')
        ax.text(0.07, 0.90,
                rf'Simulated $t\bar{{t}}$ ({channel_label})',
                transform=ax.transAxes, fontsize=20, ha='left', va='top')

    fig, ax = plt.subplots()
    _make_panel(ax)
    ax.set_xlim(x_start, x_end)

    visible = (centers >= x_start) & (centers <= x_end)
    maxv = []
    if np.any(visible & (ratio > 0)):
        maxv.append(np.max((ratio + ratio_errors)[visible & (ratio > 0)]))
    ax.set_ylim(0, max(maxv) * 1.15 if maxv else 1.0)

    ax.legend(loc='upper right', fontsize=16)
    fig.tight_layout()
    save_figure(fig, output_path)
    logger.info("Saved %s plot to: %s", channel_label, output_path)
    plt.close(fig)


def _fmt_fit(fit_val, fit_err):
    """Format a fit result as '$value \\pm error$' for LaTeX."""
    if fit_val == 0.0:
        return "---"
    return f"${fit_val:.3f} \\pm {fit_err:.3f}$"


def write_latex_table(fits_ee, fits_mumu, fit_ranges, range_labels,
                      era, year, table_path, boosted=False):
    """Write a LaTeX table of R_emu fit results (like AN Table 59).

    When ``--tf-regions`` is used, fit_ranges has two entries (low/high mass)
    and the table gets one row per range.  Otherwise a single row.
    """
    category = "Boosted" if boosted else "Resolved"
    mass_var = r"$m_{\ell J}$" if boosted else r"$m_{\ell\ell jj}$"

    # Build row labels matching AN convention
    row_labels = []
    for lo, hi in fit_ranges:
        if lo == 0 and hi == 800:
            row_labels.append(f"{mass_var} $\\leq$ 800 GeV")
        elif lo == 800:
            row_labels.append(f"{mass_var} $>$ 800 GeV")
        else:
            row_labels.append(f"{category}")

    lines = [
        r"\begin{table}[htp]",
        rf"  \topcaption{{{era} ({year}): $e\mu$ transfer factor ($R_{{e\mu}}$) "
        rf"constant fit results for the {category.lower()} category, "
        r"with statistical uncertainties.}",
        rf"  \label{{tab:emu_tf_{category.lower()}_{era}}}",
        r"  \centering",
        r"  \begin{tabular}{l | c c}",
        r"    \hline",
        rf"    Mass region & ee & $\mu\mu$ \\",
        r"    \hline",
    ]

    for i, label in enumerate(row_labels):
        ee_str = _fmt_fit(*fits_ee[i][:2])
        mumu_str = _fmt_fit(*fits_mumu[i][:2])
        lines.append(f"    {label} & {ee_str} & {mumu_str} \\\\")

    lines += [
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]

    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines))
    logger.info("Saved LaTeX table to: %s", table_path)


def write_channel_latex_table(ejet_fits, mujet_fits, fit_ranges, range_labels,
                               era, year, table_path):
    """Write LaTeX table for per-channel boosted R_emu results (eJ vs muJ)."""
    mass_var = r"$m_{\ell J}$"
    row_labels = []
    for lo, hi in fit_ranges:
        if lo == 0 and hi == 800:
            row_labels.append(f"{mass_var} $\\leq$ 800 GeV")
        elif lo == 800:
            row_labels.append(f"{mass_var} $>$ 800 GeV")
        else:
            row_labels.append("Boosted")

    lines = [
        r"\begin{table}[htp]",
        rf"  \topcaption{{{era} ({year}): per-channel $e\mu$ transfer factor ($R_{{e\mu}}$) "
        r"constant fit results for the boosted category, with statistical uncertainties.}",
        rf"  \label{{tab:emu_tf_boosted_perchannel_{era}}}",
        r"  \centering",
        r"  \begin{tabular}{l | c c}",
        r"    \hline",
        r"    Mass region & $eJ$ (ee/$\mu$e) & $\mu J$ ($\mu\mu$/e$\mu$) \\",
        r"    \hline",
    ]

    for i, label in enumerate(row_labels):
        ejet_str = _fmt_fit(*ejet_fits[i][:2])
        mujet_str = _fmt_fit(*mujet_fits[i][:2])
        lines.append(f"    {label} & {ejet_str} & {mujet_str} \\\\")

    lines += [
        r"    \hline",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]

    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines))
    logger.info("Saved per-channel LaTeX table to: %s", table_path)


def compile_preview_pdf(table_dir, table_files):
    """Write a standalone preview.tex that inputs all generated tables and compile to PDF."""
    inputs = "\n".join(
        rf"\input{{{f.name}}}" + "\n" + r"\vspace{3em}"
        for f in table_files
    )
    preview_tex = rf"""\documentclass[12pt]{{article}}
\usepackage[margin=0.5in]{{geometry}}
\usepackage{{amsmath}}
\usepackage{{float}}

%% CMS style stubs
\usepackage{{caption}}
\captionsetup{{font=large}}
\newcommand{{\topcaption}}[1]{{\caption{{#1}}}}
\setlength{{\floatsep}}{{4pt}}
\setlength{{\textfloatsep}}{{4pt}}
\setlength{{\intextsep}}{{4pt}}

\begin{{document}}

{inputs}

\end{{document}}
"""
    preview_path = table_dir / "preview.tex"
    preview_path.write_text(preview_tex)
    logger.info("Wrote %s", preview_path)

    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "preview.tex"],
        cwd=str(table_dir), capture_output=True, text=True,
    )
    if result.returncode == 0:
        logger.info("Compiled preview PDF: %s", table_dir / "preview.pdf")
    else:
        logger.error("pdflatex failed:")
        for line in result.stdout.splitlines()[-10:]:
            logger.error("  %s", line)


def process_topology(args, era, boosted):
    """Process a single era + topology (resolved or boosted).

    Returns:
        (table_dir, list[Path]): table output directory and paths of written .tex files,
        or (None, []) if the topology was skipped.
    """
    info = load_lumi(era)
    run = info["run"]
    year = info["year"]
    lumi = info["lumi"]
    com = info.get("com", 13.6)

    category = "boosted" if boosted else "resolved"
    tf_suffix = "_tf" if args.tf_regions else ""

    # Build input path
    input_dir = repo_root() / "rootfiles" / run / str(year) / era
    if args.dir:
        input_dir = input_dir / args.dir

    if args.input_file:
        filepath = input_dir / args.input_file
    else:
        filepath = input_dir / "WRAnalyzer_tt_tW.root"

    if not filepath.exists():
        logger.warning("ROOT file not found: %s — skipping %s %s", filepath, era, category)
        return None, []

    logger.info("Loading histograms from: %s (%s)", filepath, category)

    # Determine region names and variable
    if boosted:
        ee_region = "wr_ee_boosted_sr"
        mumu_region = "wr_mumu_boosted_sr"
        den_regions = [
            "wr_emu_boosted_flavor_cr",
            "wr_mue_boosted_flavor_cr",
        ]
        variable = "mass_twoobject"
    else:
        ee_region = "wr_ee_resolved_sr"
        mumu_region = "wr_mumu_resolved_sr"
        den_regions = ["wr_resolved_flavor_cr"]
        variable = "mass_fourobject"

    # Load histograms.  When --tf-regions, load _tf_low (m < 800) and
    # standard SR (m > 800) histograms separately.
    if args.tf_regions:
        ee_low = _load_hist(filepath, ee_region + "_tf_low", variable)
        mumu_low = _load_hist(filepath, mumu_region + "_tf_low", variable)
        den_low = [_load_hist(filepath, r + "_tf_low", variable) for r in den_regions]

        ee_high = _load_hist(filepath, ee_region, variable)
        mumu_high = _load_hist(filepath, mumu_region, variable)
        den_high = [_load_hist(filepath, r, variable) for r in den_regions]

        if any(h is None for h in [ee_low, mumu_low] + den_low):
            logger.warning("Could not load _tf_low histograms — skipping %s %s", era, category)
            return None, []
        if any(h is None for h in [ee_high, mumu_high] + den_high):
            logger.warning("Could not load standard SR histograms — skipping %s %s", era, category)
            return None, []

        logger.info("Loaded _tf_low and standard SR histograms for %s %s", era, category)

        hist_sets = [
            {
                "suffix": "_tf_low",
                "ee": ee_low, "mumu": mumu_low, "den": den_low,
                "xlim": (0, 1000),
                "fit_ranges": [(0, 800)], "range_labels": ["low-mass"],
            },
            {
                "suffix": "_tf_high",
                "ee": ee_high, "mumu": mumu_high, "den": den_high,
                "xlim": None,
                "fit_ranges": [(800, 8000)], "range_labels": ["high-mass"],
            },
        ]
    else:
        ee_num_hist = _load_hist(filepath, ee_region, variable)
        mumu_num_hist = _load_hist(filepath, mumu_region, variable)
        den_hists = [_load_hist(filepath, r, variable) for r in den_regions]

        if ee_num_hist is None:
            logger.warning("Could not load ee numerator '%s' — skipping %s %s", ee_region, era, category)
            return None, []
        if mumu_num_hist is None:
            logger.warning("Could not load mumu numerator '%s' — skipping %s %s", mumu_region, era, category)
            return None, []
        if any(h is None for h in den_hists):
            logger.warning("Could not load denominator(s) %s — skipping %s %s", den_regions, era, category)
            return None, []

        logger.info("Loaded ee numerator: %s/%s", ee_region, variable)
        logger.info("Loaded mumu numerator: %s/%s", mumu_region, variable)
        for r in den_regions:
            logger.info("Loaded denominator: %s/%s", r, variable)

        hist_sets = [
            {
                "suffix": "",
                "ee": ee_num_hist, "mumu": mumu_num_hist, "den": den_hists,
                "xlim": None,
                "fit_ranges": None, "range_labels": None,
            },
        ]

    # Output paths
    output_dir = repo_root() / "plots" / run / str(year) / era
    if args.dir:
        output_dir = output_dir / args.dir
    output_dir.mkdir(parents=True, exist_ok=True)

    table_dir = repo_root() / "tables" / run / str(year) / era
    if args.dir:
        table_dir = table_dir / args.dir

    table_files = []

    for hs in hist_sets:
        suffix = hs["suffix"]
        ee_num_hist = hs["ee"]
        mumu_num_hist = hs["mumu"]
        den_hists = hs["den"]
        xlim = hs["xlim"]
        hs_fit_ranges = hs["fit_ranges"]
        hs_range_labels = hs["range_labels"]

        # Rebin histograms (rebin individual denominators before summing so
        # per-channel boosted plots can reuse them)
        if args.variable_bins is not None:
            if args.variable_bins == "default":
                if args.tf_regions:
                    variable_edges = [0, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1600, 2000, 2800, 4000, 8000]
                else:
                    variable_edges = [800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 8000]
            else:
                variable_edges = [float(x) for x in args.variable_bins.split(",")]
            ee_num_hist = rebin_histogram(ee_num_hist, variable_edges)
            mumu_num_hist = rebin_histogram(mumu_num_hist, variable_edges)
            den_hists = [rebin_histogram(h, variable_edges) for h in den_hists]
            logger.info("Rebinned with variable edges: %s", variable_edges)
        elif args.rebin:
            ee_num_hist = rebin_histogram(ee_num_hist, args.rebin)
            mumu_num_hist = rebin_histogram(mumu_num_hist, args.rebin)
            den_hists = [rebin_histogram(h, args.rebin) for h in den_hists]
            logger.info("Rebinned with factor: %d", args.rebin)
        else:
            plot_settings = load_plot_settings(era)
            region_cfgs, common_vars = index_plot_settings(plot_settings)
            num_vcfg = get_var_cfg(region_cfgs, common_vars, ee_region, variable)
            rebin_factor = num_vcfg.get('rebin', 1) if num_vcfg else 1
            ee_num_hist = rebin_histogram(ee_num_hist, rebin_factor)
            mumu_num_hist = rebin_histogram(mumu_num_hist, rebin_factor)
            den_hists = [rebin_histogram(h, rebin_factor) for h in den_hists]
            logger.info("Rebinned from YAML config: %s", rebin_factor)

        # Sum denominator histograms (boosted has separate emu and mue regions)
        den_hist = den_hists[0]
        for h in den_hists[1:]:
            den_hist = den_hist + h

        # Compute ratios
        ratio_ee, ratio_ee_errors, edges = compute_ratio(ee_num_hist, den_hist)
        ratio_mumu, ratio_mumu_errors, _ = compute_ratio(mumu_num_hist, den_hist)

        logger.info("Ratio values per bin (ee, %s):", suffix or "full")
        for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            logger.info("  [%.0f, %.0f): %.4f +/- %.4f", low, high, ratio_ee[i], ratio_ee_errors[i])

        logger.info("Ratio values per bin (mumu, %s):", suffix or "full")
        for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            logger.info("  [%.0f, %.0f): %.4f +/- %.4f", low, high, ratio_mumu[i], ratio_mumu_errors[i])

        # Compute fits
        fits_ee, fits_mumu, fit_ranges, range_labels = compute_fits(
            ratio_ee, ratio_ee_errors, ratio_mumu, ratio_mumu_errors,
            edges, fit_ranges=hs_fit_ranges, range_labels=hs_range_labels,
        )

        # Plot
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = output_dir / f"emu_transfer_factor_{category}{suffix}_{era}.pdf"

        plot_ratio(
            ratio_ee, ratio_ee_errors,
            ratio_mumu, ratio_mumu_errors,
            edges, era, lumi,
            output_path, fits_ee, fits_mumu,
            fit_ranges, range_labels, com=com,
            boosted=boosted, xlim=xlim,
        )

        # Write LaTeX table
        table_path = table_dir / f"emu_transfer_factor_{category}{suffix}.tex"
        write_latex_table(
            fits_ee, fits_mumu, fit_ranges, range_labels,
            era, year, table_path, boosted=boosted,
        )
        table_files.append(table_path)

        # Per-channel boosted plots: match by jet flavor
        #   e jet (eJ):  ee SR / mue CR  (both have electron in AK8 jet)
        #   mu jet (muJ): mumu SR / emu CR (both have muon in AK8 jet)
        if boosted:
            emu_cr_hist = den_hists[0]   # wr_emu_boosted_flavor_cr: e (resolved) + mu (in jet)
            mue_cr_hist = den_hists[1]   # wr_mue_boosted_flavor_cr: mu (resolved) + e (in jet)
            centers_ch = 0.5 * (edges[:-1] + edges[1:])

            # e jet: ee SR / mue CR
            ratio_ejet, ratio_ejet_err, _ = compute_ratio(ee_num_hist, mue_cr_hist)
            fits_ejet = [chi2_fit_constant(ratio_ejet, ratio_ejet_err,
                         centers_ch, fit_range=fr) for fr in fit_ranges]

            ejet_path = output_dir / f"emu_transfer_factor_boosted_ejet{suffix}_{era}.pdf"
            plot_channel_ratio(
                ratio_ejet, ratio_ejet_err, edges, era, lumi,
                ejet_path, fits_ejet, fit_ranges, range_labels,
                channel_label="eJ",
                num_label=r"Boosted\ SR\ (ee)",
                den_label=r"Boosted\ Flavor\ CR\ (\mu e)",
                color=COLOR_EE, com=com, xlim=xlim,
            )

            # mu jet: mumu SR / emu CR
            ratio_mujet, ratio_mujet_err, _ = compute_ratio(mumu_num_hist, emu_cr_hist)
            fits_mujet = [chi2_fit_constant(ratio_mujet, ratio_mujet_err,
                          centers_ch, fit_range=fr) for fr in fit_ranges]

            mujet_path = output_dir / f"emu_transfer_factor_boosted_mujet{suffix}_{era}.pdf"
            plot_channel_ratio(
                ratio_mujet, ratio_mujet_err, edges, era, lumi,
                mujet_path, fits_mujet, fit_ranges, range_labels,
                channel_label=r"$\mu$J",
                num_label=r"Boosted\ SR\ (\mu\mu)",
                den_label=r"Boosted\ Flavor\ CR\ (e\mu)",
                color=COLOR_MUMU, marker='s', com=com, xlim=xlim,
            )

            # Per-channel LaTeX table
            ch_table_path = table_dir / f"emu_transfer_factor_boosted_perchannel{suffix}.tex"
            write_channel_latex_table(
                fits_ejet, fits_mujet, fit_ranges, range_labels,
                era, year, ch_table_path,
            )
            table_files.append(ch_table_path)

    return table_dir, table_files


def main():
    args = parse_args()
    setup_logging(args.verbose)

    # Determine eras
    eras = [args.era] if args.era else _ERA_CHOICES

    # Determine topologies
    if args.boosted and not args.resolved:
        topologies = [True]         # boosted only
    elif args.resolved and not args.boosted:
        topologies = [False]        # resolved only
    else:
        topologies = [False, True]  # both (default)

    for era in eras:
        table_dir = None
        all_table_files = []
        for boosted in topologies:
            tdir, tfiles = process_topology(args, era, boosted)
            if tdir is not None:
                table_dir = tdir
            all_table_files.extend(tfiles)

        if table_dir and all_table_files:
            compile_preview_pdf(table_dir, all_table_files)


if __name__ == '__main__':
    main()
