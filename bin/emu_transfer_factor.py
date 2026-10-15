#!/usr/bin/env python3
"""
Compute and plot the emu transfer factor (R_emu): the bin-by-bin ratio of
mass histograms between same-flavor signal regions (ee and mumu) and the
emu flavor sideband control region, with constant fits and chi2/ndf.

Only tt+tW MC is read (WRAnalyzer_tt_tW.root, or the file given with
--input), so there is no --dy option.  --era takes a single era, a year
(2022, 2023, 2024) or Run3: a combination adds the histograms of its eras
(data/lumi.yaml sub_eras), each read from rootfiles/<Run>/<Year>/<era>/<dir>.

Outputs (<plots> is plots/ with --local-plots, else EOS, or outputs/ when EOS
is not mounted; [_tf] in --tf-study mode, which also adds _lowzoom views):
    <plots>/<Run>/<Year>/<Era>/<dir>/emu_transfer_factor_{resolved,boosted}[_tf]_<Era>.pdf
    <plots>/<Run>/<Year>/<Era>/<dir>/emu_transfer_factor_boosted_{ejet,mujet}[_tf]_<Era>.pdf
    tables/<Run>/<Year>/<Era>/<dir>/emu_transfer_factor_{resolved,boosted}[_tf].tex
    tables/<Run>/<Year>/<Era>/<dir>/emu_transfer_factor_boosted_perchannel[_tf].tex
    tables/<Run>/<Year>/<Era>/<dir>/preview.tex (and preview.pdf)

Usage:
    # Resolved + boosted (nominal SR / flavor CR, m > 800 GeV):
    python3 bin/emu_transfer_factor.py --era RunIII2024Summer24 --dir 20261008_skims_v2 --local-plots

    # A year or all of Run 3, resolved only:
    python3 bin/emu_transfer_factor.py --era Run3 --dir 20261008_skims_v2 --resolved --local-plots

    # Transfer-factor study (AN-19-083 App. J closure): read the ``_nomass``
    # regions written by ``run_analysis.py --study transfer-factor
    # --split-groups tt_tW`` and fit below / above 800 GeV separately:
    python3 bin/emu_transfer_factor.py --era RunIII2024Summer24 --dir tf_study --tf-study --input WRAnalyzer_TTbar.root

Study mode reads one histogram per region (SR / flavor CR without the
reconstructed-mass cut) and slices it at MASS_SPLIT.  800 GeV must remain a
bin edge after any rebinning; the script refuses otherwise.  Files from the
older analyzer that only contain ``_tf_low`` regions are read by summing
``_tf_low`` with the nominal region (decided per era file).
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

from wrplotter.cli_utils import add_era_args, add_output_args, setup_logging
from wrplotter.config import era_label, get_var_cfg, index_plot_settings, load_lumi, load_plot_settings
from wrplotter.histogram_cache import load_and_rebin
from wrplotter.histogram_utils import add_hists, sum_hists
from wrplotter.inputs import make_inputs, root_file
from wrplotter.paths import plots_base, repo_root, save_figure
from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header
from wrplotter.stack import hist_key

hep.style.use("CMS")

logger = logging.getLogger(__name__)

COLOR_EE = "#e42536"     # CMS (Petroff) palette red
COLOR_MUMU = "#5790fc"   # CMS (Petroff) palette blue

# Boundary between the closure region (below) and the SR (above), in GeV.
MASS_SPLIT = 800.0
NOMASS_SUFFIX = "_nomass"
LEGACY_INCLUSIVE_SUFFIX = "_tf"      # older analyzer: same mass-inclusive region under another name
LEGACY_LOW_SUFFIX = "_tf_low"        # older analyzer: m < 800 copy, to be summed with the nominal region

DEFAULT_SAMPLE = "tt_tW"
STUDY_SAMPLE = "TTbar"               # --split-groups tt_tW output, preferred in --tf-study mode

# Built-in --variable-bins edges (study mode must keep MASS_SPLIT as an edge).
DEFAULT_STUDY_EDGES = [0, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1600, 2000, 2800, 4000, 8000]
DEFAULT_EDGES = [800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 8000]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute emu transfer factor (R_emu): ratio of SR (ee/mumu) to flavor sideband CR (emu)"
    )
    add_era_args(parser)
    add_output_args(parser)
    parser.add_argument(
        "--output", "-o", dest="output", type=str, default=None,
        help="Path of the main ratio plot of each topology, instead of "
             "<plots>/.../emu_transfer_factor_<category>[_tf]_<era>.pdf "
             "(use with --resolved or --boosted)"
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
             "Without an argument, uses built-in defaults appropriate for --tf-study or standard mode."
    )
    parser.add_argument(
        "--input", dest="input_file", type=str, default=None,
        help="ROOT file to read in each era's --dir directory, WRAnalyzer_<sample>.root "
             "(default: WRAnalyzer_tt_tW.root; in --tf-study mode WRAnalyzer_TTbar.root is "
             "preferred when every era has it)"
    )
    parser.add_argument(
        "--tf-study", dest="tf_study", action="store_true",
        help="Transfer-factor study: read the _nomass regions (SR / flavor CR without the "
             "m > 800 GeV cut, from run_analysis.py --study transfer-factor) and fit the "
             "ratio separately below and above 800 GeV."
    )
    parser.add_argument(
        "--boosted", dest="boosted", action="store_true",
        help="Boosted only (default: both resolved and boosted)"
    )
    parser.add_argument(
        "--resolved", dest="resolved", action="store_true",
        help="Resolved only (default: both resolved and boosted)"
    )
    args = parser.parse_args(argv)
    name = args.input_file
    if name is not None and ("/" in name or not (name.startswith("WRAnalyzer_")
                                                 and name.endswith(".root"))):
        parser.error(f"--input takes a file name WRAnalyzer_<sample>.root (read from each era's "
                     f"--dir directory), got '{name}'")
    return args


def input_sample(inputs, input_file=None, tf_study=False) -> str:
    """Sample to read: from --input, else TTbar in study mode when every era has it, else tt_tW."""
    if input_file:
        return input_file.removeprefix("WRAnalyzer_").removesuffix(".root")
    if tf_study and all(root_file(d, STUDY_SAMPLE).is_file() for d in inputs.dirs_for(STUDY_SAMPLE)):
        return STUDY_SAMPLE
    return DEFAULT_SAMPLE


def file_loader(directory: Path, sample: str, rebin):
    """``load(hist_key)``: one histogram of one era's file, rebinned, or None if absent."""
    def load(key):
        return load_and_rebin([directory], sample, key, rebin)
    return load


def load_study_hist(load, region: str, variable: str, where=""):
    """Load the mass-inclusive histogram of *region* for the transfer-factor study.

    *load(hist_key)* reads one file (see :func:`file_loader`); *where* names it
    in messages.  Prefers ``<region>_nomass``.  Legacy files are supported in
    two forms: ``<region>_tf`` (the same mass-inclusive region under its older
    name) and ``<region>_tf_low`` (m < 800), which is summed with the nominal
    region (m > 800) since together they cover the same events.  Returns None
    if none of these is available.
    """
    name = Path(where).name if where else "the input"
    h = load(hist_key(region + NOMASS_SUFFIX, variable))
    if h is not None:
        return h
    h = load(hist_key(region + LEGACY_INCLUSIVE_SUFFIX, variable))
    if h is not None:
        logger.warning("%s%s not found in %s; using legacy mass-inclusive %s%s.",
                       region, NOMASS_SUFFIX, name, region, LEGACY_INCLUSIVE_SUFFIX)
        return h
    low = load(hist_key(region + LEGACY_LOW_SUFFIX, variable))
    high = load(hist_key(region, variable))
    if low is None or high is None:
        logger.warning("Could not load %s%s (nor legacy %s%s + %s) from %s",
                       region, NOMASS_SUFFIX, region, LEGACY_LOW_SUFFIX, region, where or name)
        return None
    logger.warning("%s%s not found in %s; using legacy %s%s + nominal region.",
                   region, NOMASS_SUFFIX, name, region, LEGACY_LOW_SUFFIX)
    return add_hists(low, high)


def load_hists(inputs, sample: str, regions, variable: str, rebin, *, study=False):
    """``{region: histogram}`` of *sample*, rebinned and summed over the era files.

    Every era file must provide every region (in study mode possibly through
    a legacy form, chosen per file).  Returns None, with a warning, otherwise
    or when no era has the file.
    """
    files = inputs.files_for(sample)
    if not files:
        logger.warning("%s not found in %s", root_file("", sample).name,
                       ", ".join(str(d) for d in inputs.dirs_for(sample)))
        return None
    per_file = []
    for path in files:
        logger.info("Loading histograms from: %s", path)
        load = file_loader(path.parent, sample, rebin)
        hists = {}
        for region in regions:
            if study:
                h = load_study_hist(load, region, variable, path)
            else:
                h = load(hist_key(region, variable))
                if h is None:
                    logger.warning("Could not load %s from %s", hist_key(region, variable), path)
            if h is None:
                return None
            hists[region] = h
        per_file.append(hists)
    return {r: sum_hists(h[r] for h in per_file) for r in regions}


def require_edge(edges, value=MASS_SPLIT):
    """Raise unless *value* is one of the bin *edges* (so slices at it are exact)."""
    edges = np.asarray(edges, dtype=float)
    if not np.any(np.isclose(edges, value, rtol=0, atol=1e-6)):
        raise ValueError(
            f"{value:g} GeV must be a bin edge to split the fit there; got edges {edges.tolist()}. "
            "Choose a rebin factor that keeps it, or pass --variable-bins with it included."
        )
    return edges


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


def _draw_points(ax, centers, ratio, errors, color, marker, label):
    """One ratio as markers with vertical error bars (uniform bins: no horizontal bars, PubGuidelines);
    bins whose flavor CR is empty (ratio 0) are not drawn."""
    ok = ratio > 0
    ax.errorbar(centers[ok], ratio[ok], yerr=errors[ok], fmt=marker, color=color, markersize=8,
                elinewidth=1.5, capsize=0, label=label)


def _fit_label(prefix, fv, ferr, chi2, ndf):
    return rf"{prefix}fit: {fv:.3f} $\pm$ {ferr:.3f}, $\chi^2$/ndf = {chi2:.1f}/{ndf}"


def _set_ranges(ax, edges, series, fit_values, xlim):
    """x from the first edge (or *xlim*) to the last filled bin, rounded up to 500 GeV; y from 0 to
    1.25 x the highest well-measured point with its error (relative uncertainty under 30 %), at
    least every marker measured better than 100 %, and twice the largest fit value.  Points
    consistent with zero and the error bars of low-statistics bins may run off the top rather
    than squash the transfer factor."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    if xlim is not None:
        x0, x1 = xlim
    else:
        filled = np.flatnonzero(np.any([r > 0 for r, _ in series], axis=0))
        x0 = 800.0
        x1 = float(edges[filled[-1] + 1]) if filled.size else float(edges[-1])
        x1 = min(float(edges[-1]), 500.0 * np.ceil(x1 / 500.0))
    ax.set_xlim(x0, x1)
    visible = (centers >= x0) & (centers <= x1)
    highs = [np.max((r + e)[m]) for r, e in series if np.any(m := visible & (r > 0) & (e < 0.3 * r))]
    markers = [1.1 * np.max(r[m]) for r, e in series if np.any(m := visible & (r > 0) & (e < r))]
    y_max = max([1.25 * h for h in highs] + markers + [2.0 * v for v in fit_values if v > 0] + [1.0])
    ax.set_ylim(0, y_max)


def _legend(ax):
    """The ratios (error-bar markers) first, then the fits; an opaque white box, so the clipped error
    bars of the low-statistics bins pass behind it."""
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: "fit:" in labels[i])
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper right',
              fontsize=FONT_SIZE_LEGEND, frameon=True, facecolor="white", edgecolor="none", framealpha=1)


def _draw_ratio_panel(ax, centers, xerr, ratio_ee, ratio_ee_errors,
                      ratio_mumu, ratio_mumu_errors, edges, era, lumi, com,
                      boosted=False):
    """Draw the common elements shared by both zoomed and full ratio plots."""
    cat = "Boosted" if boosted else "Resolved"
    mass_label = r'$m_{\ell J}$' if boosted else r'$m_{\ell\ell jj}$'

    _draw_points(ax, centers, ratio_ee, ratio_ee_errors, COLOR_EE, "o", r"ee / e$\mu$")
    _draw_points(ax, centers, ratio_mumu, ratio_mumu_errors, COLOR_MUMU, "s", r"$\mu\mu$ / e$\mu$")

    ax.set_xlabel(f'{mass_label} [GeV]')
    ax.set_ylabel(rf'{cat} SR / {cat} Flavor CR (e$\mu$)')

    cms_label(ax, lumi, com, data=False, text=r'Simulated $t\bar{t}$', year=era)


def _constant(x, c):
    """Constant model for curve_fit."""
    return np.full_like(x, c, dtype=float)


def chi2_fit_constant(values, errors, edges, fit_range=None):
    """
    Fit a constant to binned ratio data using chi-squared minimization.

    A bin enters the fit only if it lies entirely inside *fit_range*
    (``lo_edge >= fit_range[0]`` and ``hi_edge <= fit_range[1]``), so two
    adjacent ranges sharing an edge use disjoint bins and a bin straddling
    the boundary is used by neither.  ``fit_range=None`` uses every bin.

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
    edges = np.asarray(edges, dtype=float)
    lo_edges, hi_edges = edges[:-1], edges[1:]
    centers = 0.5 * (lo_edges + hi_edges)
    in_range = np.ones(len(centers), dtype=bool)
    if fit_range is not None:
        in_range = (lo_edges >= fit_range[0] - 1e-6) & (hi_edges <= fit_range[1] + 1e-6)
    mask = in_range & (values > 0) & (errors > 0)

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
    """Draw the constant fits (ee dashed, mumu dash-dotted) with their value and chi2/ndf in the legend."""
    for i, (fr, rl) in enumerate(zip(fit_ranges, range_labels)):
        x_lo, x_hi = fr
        prefix = f"{rl} " if len(range_labels) > 1 else ""
        for (fv, ferr, chi2, ndf), color, ls, chan in ((fits_ee[i], COLOR_EE, "--", "ee"),
                                                      (fits_mumu[i], COLOR_MUMU, "-.", r"$\mu\mu$")):
            if ndf > 0:
                ax.plot([x_lo, x_hi], [fv, fv], ls, color=color, linewidth=2,
                        label=_fit_label(f"{prefix}{chan} ", fv, ferr, chi2, ndf))

    if len(fit_ranges) > 1:
        ax.axvline(x=fit_ranges[0][1], color='gray', linestyle=':', linewidth=1, alpha=0.5)


def compute_fits(ratio, ratio_errors, edges, fit_ranges, range_labels, label=""):
    """Constant fits of one ratio in each of *fit_ranges*.

    Returns a list of ``(fit_val, fit_err, chi2, ndf)``, one per range.
    """
    fits = [chi2_fit_constant(ratio, ratio_errors, edges, fit_range=fr) for fr in fit_ranges]
    for rl, (fv, ferr, chi2, ndf) in zip(range_labels, fits):
        if ndf > 0:
            logger.info("Chi-squared constant fit (%s, %s range): %.4f +/- %.4f  (chi2/ndf = %.1f/%d = %.2f)",
                        label, rl, fv, ferr, chi2, ndf, chi2 / ndf)
        else:
            logger.info("Chi-squared constant fit (%s, %s range): fit failed (not enough bins)", label, rl)
    return fits


def study_fit_ranges(edges, split=MASS_SPLIT):
    """Fit ranges for the study: below and above *split*, on validated edges."""
    edges = require_edge(edges, split)
    return [(float(edges[0]), split), (split, float(edges[-1]))], ["low-mass", "high-mass"]


def plot_ratio(ratio_ee, ratio_ee_errors, ratio_mumu, ratio_mumu_errors,
               edges, era, lumi, output_path, fits_ee, fits_mumu,
               fit_ranges, range_labels, com=13.6,
               boosted=False, xlim=None):
    """
    Create the ratio plot with both ee and mumu channels, including fit lines and chi2/ndf.
    """
    centers = 0.5 * (edges[:-1] + edges[1:])
    xerr = 0.5 * np.diff(edges)

    fig, ax = plt.subplots(figsize=(10, 10))
    _draw_ratio_panel(ax, centers, xerr, ratio_ee, ratio_ee_errors,
                      ratio_mumu, ratio_mumu_errors, edges, era, lumi, com,
                      boosted=boosted)
    _draw_fits_and_annotations(ax, fit_ranges, range_labels, fits_ee, fits_mumu)
    _set_ranges(ax, edges, [(ratio_ee, ratio_ee_errors), (ratio_mumu, ratio_mumu_errors)],
                [f[0] for f in fits_ee + fits_mumu], xlim)

    _legend(ax)
    fit_header(ax, ignore_above=True)
    save_figure(fig, output_path)
    logger.info("Saved plot to: %s", output_path)
    plt.close(fig)


def plot_channel_ratio(ratio, ratio_errors, edges, era, lumi, output_path,
                       fits, fit_ranges, range_labels,
                       channel_label, num_label, den_label,
                       color, marker='o', com=13.6, xlim=None):
    """Ratio plot for a single boosted jet-flavor channel (eJ or muJ)."""
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(10, 10))
    _draw_points(ax, centers, ratio, ratio_errors, color, marker, f"{num_label} / {den_label}")
    for i, fr in enumerate(fit_ranges):
        fv, ferr, chi2, ndf = fits[i]
        prefix = f"{range_labels[i]} " if len(range_labels) > 1 else ""
        if ndf > 0:
            ax.plot([fr[0], fr[1]], [fv, fv], "--", color=color, linewidth=2,
                    label=_fit_label(prefix, fv, ferr, chi2, ndf))
    if len(fit_ranges) > 1:
        ax.axvline(x=fit_ranges[0][1], color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel(r'$m_{\ell J}$ [GeV]')
    ax.set_ylabel(r'Boosted SR / Boosted Flavor CR')
    cms_label(ax, lumi, com, data=False, text=rf'Simulated $t\bar{{t}}$, {channel_label}', year=era)
    _set_ranges(ax, edges, [(ratio, ratio_errors)], [f[0] for f in fits], xlim)

    _legend(ax)
    fit_header(ax, ignore_above=True)
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

    When ``--tf-study`` is used, fit_ranges has two entries (low/high mass)
    and the table gets one row per range.  Otherwise a single row.
    """
    category = "Boosted" if boosted else "Resolved"
    mass_var = r"$m_{\ell J}$" if boosted else r"$m_{\ell\ell jj}$"

    # Build row labels matching AN convention
    row_labels = []
    for lo, hi in fit_ranges:
        if hi == MASS_SPLIT:
            row_labels.append(f"{mass_var} $<$ {MASS_SPLIT:g} GeV")
        elif lo == MASS_SPLIT:
            row_labels.append(f"{mass_var} $>$ {MASS_SPLIT:g} GeV")
        else:
            row_labels.append(f"{category}")

    lines = [
        r"\begin{table}[htp]",
        rf"  \topcaption{{{era_label(era)}: $e\mu$ transfer factor ($R_{{e\mu}}$) "
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
        if hi == MASS_SPLIT:
            row_labels.append(f"{mass_var} $<$ {MASS_SPLIT:g} GeV")
        elif lo == MASS_SPLIT:
            row_labels.append(f"{mass_var} $>$ {MASS_SPLIT:g} GeV")
        else:
            row_labels.append("Boosted")

    lines = [
        r"\begin{table}[htp]",
        rf"  \topcaption{{{era_label(era)}: per-channel $e\mu$ transfer factor ($R_{{e\mu}}$) "
        r"constant fit results for the boosted category, with statistical uncertainties.}",
        rf"  \label{{tab:emu_tf_boosted_perchannel_{era}}}",
        r"  \centering",
        r"  \begin{tabular}{l | c c}",
        r"    \hline",
        r"    Mass region & ee / $\mu$($e$-jet) & $\mu\mu$ / $e$($\mu$-jet) \\",
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


def rebin_spec(args, era, ee_region, variable):
    """Rebinning of every histogram: --variable-bins edges, --rebin, or the era's plot settings."""
    if args.variable_bins is not None:
        if args.variable_bins == "default":
            edges = DEFAULT_STUDY_EDGES if args.tf_study else DEFAULT_EDGES
        else:
            edges = [float(x) for x in args.variable_bins.split(",")]
        logger.info("Rebinning with variable edges: %s", edges)
        return edges
    if args.rebin:
        logger.info("Rebinning with factor: %d", args.rebin)
        return args.rebin
    region_cfgs, common_vars = index_plot_settings(load_plot_settings(era))
    num_vcfg = get_var_cfg(region_cfgs, common_vars, ee_region, variable)
    rebin_factor = num_vcfg.get('rebin', 1) if num_vcfg else 1
    logger.info("Rebinning from YAML config: %s", rebin_factor)
    return rebin_factor


def _log_ratio(label, ratio, errors, edges):
    logger.info("Ratio values per bin (%s):", label)
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        logger.info("  [%.0f, %.0f): %.4f +/- %.4f", low, high, ratio[i], errors[i])


def process_topology(args, inputs, sample, boosted, plot_dir, table_dir):
    """Process one era (or combination) + topology (resolved or boosted).

    Plots go to *plot_dir*, LaTeX tables to *table_dir*.

    Returns:
        list[Path]: the written .tex files, empty if the topology was skipped.
    """
    era = inputs.era
    info = load_lumi(era)
    year = info["year"]
    lumi = info["lumi"]
    com = info.get("com", 13.6)
    label = era_label(era)

    category = "boosted" if boosted else "resolved"
    suffix = "_tf" if args.tf_study else ""

    # Region names and variable
    if boosted:
        ee_region = "wr_ee_boosted_sr"
        mumu_region = "wr_mumu_boosted_sr"
        den_regions = ["wr_emu_boosted_flavor_cr", "wr_mue_boosted_flavor_cr"]
        variable = "mass_twoobject"
    else:
        ee_region = "wr_ee_resolved_sr"
        mumu_region = "wr_mumu_resolved_sr"
        den_regions = ["wr_resolved_flavor_cr"]
        variable = "mass_fourobject"

    rebin = rebin_spec(args, era, ee_region, variable)
    hists = load_hists(inputs, sample, [ee_region, mumu_region] + den_regions, variable, rebin,
                       study=args.tf_study)
    if hists is None:
        logger.warning("Skipping %s %s", era, category)
        return []
    for r in hists:
        logger.info("Loaded %s/%s%s", r, variable, NOMASS_SUFFIX if args.tf_study else "")

    ee_num_hist = hists[ee_region]
    mumu_num_hist = hists[mumu_region]
    den_hists = [hists[r] for r in den_regions]

    # Sum denominator histograms (boosted has separate emu and mue regions)
    den_hist = sum_hists(den_hists)

    # Ratios
    ratio_ee, ratio_ee_errors, edges = compute_ratio(ee_num_hist, den_hist)
    ratio_mumu, ratio_mumu_errors, _ = compute_ratio(mumu_num_hist, den_hist)
    _log_ratio("ee", ratio_ee, ratio_ee_errors, edges)
    _log_ratio("mumu", ratio_mumu, ratio_mumu_errors, edges)

    # Fit ranges: split at MASS_SPLIT for the study, otherwise the full axis.
    if args.tf_study:
        fit_ranges, range_labels = study_fit_ranges(edges)
    else:
        fit_ranges, range_labels = [(float(edges[0]), float(edges[-1]))], ["full"]

    fits_ee = compute_fits(ratio_ee, ratio_ee_errors, edges, fit_ranges, range_labels, "ee")
    fits_mumu = compute_fits(ratio_mumu, ratio_mumu_errors, edges, fit_ranges, range_labels, "mumu")

    table_files = []

    # Plots: full range, plus a low-mass zoom in study mode.
    views = [("", None if not args.tf_study else (0, edges[-1]))]
    if args.tf_study:
        views.append(("_lowzoom", (0, 1000)))
    for view_suffix, xlim in views:
        if args.output and view_suffix == "":
            output_path = Path(args.output)
        else:
            output_path = plot_dir / f"emu_transfer_factor_{category}{suffix}{view_suffix}_{era}.pdf"
        plot_ratio(
            ratio_ee, ratio_ee_errors,
            ratio_mumu, ratio_mumu_errors,
            edges, label, lumi,
            output_path, fits_ee, fits_mumu,
            fit_ranges, range_labels, com=com,
            boosted=boosted, xlim=xlim,
        )

    # LaTeX table (one row per fit range)
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
        emu_cr_hist, mue_cr_hist = den_hists[0], den_hists[1]
        channels = [
            ("ejet", "electron in the jet", ee_num_hist, mue_cr_hist, "ee", r"$\mu$(e-jet)", COLOR_EE, 'o'),
            ("mujet", "muon in the jet", mumu_num_hist, emu_cr_hist, r"$\mu\mu$", r"e($\mu$-jet)", COLOR_MUMU, 's'),
        ]
        channel_fits = {}
        for key, ch_label, num, den, num_label, den_label, color, marker in channels:
            ratio, ratio_err, _ = compute_ratio(num, den)
            channel_fits[key] = compute_fits(ratio, ratio_err, edges, fit_ranges, range_labels, key)
            for view_suffix, xlim in views:
                path = plot_dir / f"emu_transfer_factor_boosted_{key}{suffix}{view_suffix}_{era}.pdf"
                plot_channel_ratio(
                    ratio, ratio_err, edges, label, lumi,
                    path, channel_fits[key], fit_ranges, range_labels,
                    channel_label=ch_label, num_label=num_label, den_label=den_label,
                    color=color, marker=marker, com=com, xlim=xlim,
                )

        ch_table_path = table_dir / f"emu_transfer_factor_boosted_perchannel{suffix}.tex"
        write_channel_latex_table(
            channel_fits["ejet"], channel_fits["mujet"], fit_ranges, range_labels,
            era, year, ch_table_path,
        )
        table_files.append(ch_table_path)

    return table_files


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.verbose)

    try:
        (inputs,) = make_inputs(args.era, args.dir, None, with_dy=False)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(2)
    sample = input_sample(inputs, args.input_file, args.tf_study)
    logger.info("Era %s, sample %s, from %s", args.era, sample,
                ", ".join(str(d) for d in inputs.dirs_for(sample)))

    info = load_lumi(args.era)
    run, year = info["run"], str(info["year"])
    plot_dir = plots_base(run, year, args.era, args.dir or None, local=args.local_plots)
    table_dir = repo_root() / "tables" / run / year / args.era / args.dir

    # Determine topologies
    if args.boosted and not args.resolved:
        topologies = [True]         # boosted only
    elif args.resolved and not args.boosted:
        topologies = [False]        # resolved only
    else:
        topologies = [False, True]  # both (default)

    table_files = []
    for boosted in topologies:
        table_files += process_topology(args, inputs, sample, boosted, plot_dir, table_dir)

    if table_files:
        compile_preview_pdf(table_dir, table_files)


if __name__ == '__main__':
    main()
