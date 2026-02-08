#!/usr/bin/env python3
"""
Plot the bin-by-bin ratio of mass_fourobject histograms between
ee_resolved_sr and resolved_flavor_cr regions.

Usage:
    python3 bin/plot_ratio.py --era RunIII2024Summer24
    python3 bin/plot_ratio.py --era RunIISummer20UL18 --dir subdir
"""

# ── Standard library ────────────────────────────────────────────────────────────
import sys
import argparse
from pathlib import Path

# ── Third-party ────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wrplotter.io import repo_root, save_figure
from wrplotter.config import list_eras, load_lumi, load_plot_settings, index_plot_settings, get_var_cfg
from wrplotter.histogram_utils import rebin_histogram, load_histogram

hep.style.use("CMS")

_ERA_CHOICES = list_eras()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot bin-by-bin ratio of mass_fourobject between wr_ee_resolved_sr and wr_resolved_flavor_cr"
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
        print(f"Warning: Could not load {hist_key} from {filepath}: {e}")
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

    # Compute ratio where denominator > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(den_vals > 0, num_vals / den_vals, 0.0)

        # Error propagation: sigma_r/r = sqrt((sigma_n/n)^2 + (sigma_d/d)^2)
        # => sigma_r = r * sqrt((sigma_n/n)^2 + (sigma_d/d)^2)
        rel_err_num = np.where(num_vals > 0, np.sqrt(num_vars) / num_vals, 0.0)
        rel_err_den = np.where(den_vals > 0, np.sqrt(den_vars) / den_vals, 0.0)
        ratio_errors = ratio * np.sqrt(rel_err_num**2 + rel_err_den**2)

        # Set errors to 0 where ratio is 0
        ratio_errors = np.where(ratio > 0, ratio_errors, 0.0)

    return ratio, ratio_errors, edges


def plot_ratio(ratio_ee, ratio_ee_errors, ratio_mumu, ratio_mumu_errors, edges, era, lumi, output_path, com=13.6):
    """
    Create the ratio plot with both ee and mumu channels.
    Generates two versions: zoomed (with avg lines) and full (no avg lines, auto-scaled).
    """
    # Plot ratio as error bars at bin centers
    centers = 0.5 * (edges[:-1] + edges[1:])
    xerr = 0.5 * np.diff(edges)

    # Calculate average ratio for bins between 0 and 4 TeV (4000 GeV)
    # Use weighted average with inverse variance weighting for bins with data
    mask_ee = (centers <= 4000) & (ratio_ee > 0) & (ratio_ee_errors > 0)
    mask_mumu = (centers <= 4000) & (ratio_mumu > 0) & (ratio_mumu_errors > 0)

    if np.any(mask_ee):
        weights_ee = 1.0 / (ratio_ee_errors[mask_ee] ** 2)
        avg_ee = np.average(ratio_ee[mask_ee], weights=weights_ee)
        avg_ee_err = np.sqrt(1.0 / np.sum(weights_ee))
    else:
        avg_ee = 0.0
        avg_ee_err = 0.0

    if np.any(mask_mumu):
        weights_mumu = 1.0 / (ratio_mumu_errors[mask_mumu] ** 2)
        avg_mumu = np.average(ratio_mumu[mask_mumu], weights=weights_mumu)
        avg_mumu_err = np.sqrt(1.0 / np.sum(weights_mumu))
    else:
        avg_mumu = 0.0
        avg_mumu_err = 0.0

    print(f"\nWeighted average ratio (0-4 TeV):")
    print(f"  ee:   {avg_ee:.4f} +/- {avg_ee_err:.4f}")
    print(f"  mumu: {avg_mumu:.4f} +/- {avg_mumu_err:.4f}")

    # Colors from sample_groups/base.yaml
    color_ee = "#7a21dd"    # darker blue
    color_mumu = "#e42536"  # red (nonprompt color)

    # --- Plot 1: Zoomed version with average lines ---
    fig, ax = plt.subplots()

    # Plot ee ratio as error bars
    ax.errorbar(
        centers, ratio_ee, yerr=ratio_ee_errors, xerr=xerr,
        fmt='o', color=color_ee, markersize=6, capsize=3,
        label=r'$\frac{\mathrm{Resolved\ SR\ (ee)}}{\mathrm{Resolved\ Flavor\ CR\ (e\mu)}}$'
    )

    # Plot mumu ratio as error bars
    ax.errorbar(
        centers, ratio_mumu, yerr=ratio_mumu_errors, xerr=xerr,
        fmt='s', color=color_mumu, markersize=6, capsize=3,
        label=r'$\frac{\mathrm{Resolved\ SR\ (\mu\mu)}}{\mathrm{Resolved\ Flavor\ CR\ (e\mu)}}$'
    )

    # Draw horizontal dashed lines at average values
    xlim = (800, 4000)
    ylim = (0, 1.2)
    ax.axhline(y=avg_ee, color=color_ee, linestyle='--', linewidth=1.5)
    ax.axhline(y=avg_mumu, color=color_mumu, linestyle='--', linewidth=1.5)

    # Cosmetics
    ax.set_xlabel(r'$m_{lljj}$ [GeV]', fontsize=20)
    ax.set_ylabel(r'Resolved SR / Resolved Flavor CR (e$\mu$)', fontsize=20)
    ax.set_xlim(*xlim)

    # Set y-axis limits
    ax.set_ylim(*ylim)

    ax.tick_params(axis='both', which='major', labelsize=20)

    # Place legend below the avg labels
    ax.legend(loc='upper right', fontsize=24)

    # CMS label
    hep.cms.label(
        loc=0, ax=ax, data=False,
        label="Work in Progress",
        lumi=f"{lumi:.1f}",
        com=com,
        fontsize=20
    )

    # Add sample label in upper left
    ax.text(
        0.07, 0.95, era,
        transform=ax.transAxes,
        fontsize=20, ha='left', va='top'
    )

    # Add sample label in upper left
    ax.text(
        0.07, 0.90, r'Simulated $t\bar{t}$ + tW',
        transform=ax.transAxes,
        fontsize=20, ha='left', va='top'
    )


    # Add text labels for average values in top left (below sample label)
    ax.text(
        0.07, 0.80, f'$\\mu\\mu$ avg: {avg_mumu:.3f}',
        transform=ax.transAxes,
        fontsize=20, ha='left', va='top', color=color_mumu
    )

    ax.text(
        0.07, 0.75, f'ee avg: {avg_ee:.3f}',
        transform=ax.transAxes,
        fontsize=20, ha='left', va='top', color=color_ee
    )

    fig.tight_layout()
    save_figure(fig, output_path)
    print(f"Saved plot to: {output_path}")
    plt.close(fig)

    # --- Plot 2: Full version without average lines, auto-scaled axes ---
    fig2, ax2 = plt.subplots()

    # Plot ee ratio as error bars
    ax2.errorbar(
        centers, ratio_ee, yerr=ratio_ee_errors, xerr=xerr,
        fmt='o', color=color_ee, markersize=6, capsize=3,
        label=r'$\frac{\mathrm{Resolved\ SR\ (ee)}}{\mathrm{Resolved\ Flavor\ CR\ (e\mu)}}$'
    )

    # Plot mumu ratio as error bars
    ax2.errorbar(
        centers, ratio_mumu, yerr=ratio_mumu_errors, xerr=xerr,
        fmt='s', color=color_mumu, markersize=6, capsize=3,
        label=r'$\frac{\mathrm{Resolved\ SR\ (\mu\mu)}}{\mathrm{Resolved\ Flavor\ CR\ (e\mu)}}$'
    )

    # No dashed average lines in this version

    # Cosmetics
    ax2.set_xlabel(r'$m_{lljj}$ [GeV]', fontsize=20)
    ax2.set_ylabel(r'Resolved SR / Resolved Flavor CR (e$\mu$)', fontsize=20)

    # Auto-scale axes to capture all data points with some padding
    # Find range of data with non-zero values
    valid_mask = (ratio_ee > 0) | (ratio_mumu > 0)
    if np.any(valid_mask):
        ax2.set_xlim(800, 5600)

        # Y-axis: find max of ratio + error, add padding
        all_ratios = np.concatenate([ratio_ee, ratio_mumu])
        all_errors = np.concatenate([ratio_ee_errors, ratio_mumu_errors])
        y_max = np.max(all_ratios)
        ax2.set_ylim(0, 10)

    ax2.tick_params(axis='both', which='major', labelsize=20)

    # Place legend
    ax2.legend(loc='best', fontsize=24)

    # CMS label
    hep.cms.label(
        loc=0, ax=ax2, data=False,
        label="Work in Progress",
        lumi=f"{lumi:.1f}",
        com=com,
        fontsize=20
    )

    # Add sample label in upper left
    ax2.text(
        0.07, 0.95, era,
        transform=ax2.transAxes,
        fontsize=20, ha='left', va='top'
    )

    # Add sample label in upper left
    ax2.text(
        0.07, 0.90, r'Simulated $t\bar{t}$ + tW',
        transform=ax2.transAxes,
        fontsize=20, ha='left', va='top'
    )

    # No average value labels in this version

    fig2.tight_layout()

    # Generate output path for full version
    output_path_full = output_path.parent / (output_path.stem + "_full" + output_path.suffix)
    save_figure(fig2, output_path_full)
    print(f"Saved full plot to: {output_path_full}")
    plt.close(fig2)


def main():
    args = parse_args()

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
        print(f"Error: ROOT file not found: {filepath}")
        sys.exit(1)

    print(f"Loading histograms from: {filepath}")

    # Load histograms for ee channel
    ee_num_hist = _load_hist(filepath, "wr_ee_resolved_sr")
    den_hist = _load_hist(filepath, "wr_resolved_flavor_cr")

    # Load histograms for mumu channel
    mumu_num_hist = _load_hist(filepath, "wr_mumu_resolved_sr")

    if ee_num_hist is None:
        print(f"Error: Could not load ee numerator histogram from region 'wr_ee_resolved_sr'")
        sys.exit(1)
    if mumu_num_hist is None:
        print(f"Error: Could not load mumu numerator histogram from region 'wr_mumu_resolved_sr'")
        sys.exit(1)
    if den_hist is None:
        print(f"Error: Could not load denominator histogram from region 'wr_resolved_flavor_cr'")
        sys.exit(1)

    print(f"Loaded ee numerator: wr_ee_resolved_sr/mass_fourobject")
    print(f"Loaded mumu numerator: wr_mumu_resolved_sr/mass_fourobject")
    print(f"Loaded denominator: wr_resolved_flavor_cr/mass_fourobject")

    # Rebin histograms
    if args.variable_bins:
        # Use variable binning for mass_fourobject (same as in histo.py)
        variable_edges = [0, 800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 8000]
        ee_num_hist = rebin_histogram(ee_num_hist, variable_edges)
        mumu_num_hist = rebin_histogram(mumu_num_hist, variable_edges)
        den_hist = rebin_histogram(den_hist, variable_edges)
        print(f"Rebinned with variable edges: {variable_edges}")
    elif args.rebin:
        # Use explicit rebin factor
        ee_num_hist = rebin_histogram(ee_num_hist, args.rebin)
        mumu_num_hist = rebin_histogram(mumu_num_hist, args.rebin)
        den_hist = rebin_histogram(den_hist, args.rebin)
        print(f"Rebinned with factor: {args.rebin}")
    else:
        # Use rebin from YAML config (default behavior)
        plot_settings = load_plot_settings(args.era)
        region_cfgs, common_vars = index_plot_settings(plot_settings)

        num_vcfg = get_var_cfg(region_cfgs, common_vars, "wr_ee_resolved_sr", "mass_fourobject")
        rebin_factor = num_vcfg.get('rebin', 1) if num_vcfg else 1

        ee_num_hist = rebin_histogram(ee_num_hist, rebin_factor)
        mumu_num_hist = rebin_histogram(mumu_num_hist, rebin_factor)
        den_hist = rebin_histogram(den_hist, rebin_factor)
        print(f"Rebinned from YAML config: {rebin_factor}")

    # Compute ratios
    ratio_ee, ratio_ee_errors, edges = compute_ratio(ee_num_hist, den_hist)
    ratio_mumu, ratio_mumu_errors, _ = compute_ratio(mumu_num_hist, den_hist)

    print(f"\nRatio values per bin (ee):")
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        print(f"  [{low:.0f}, {high:.0f}): {ratio_ee[i]:.4f} +/- {ratio_ee_errors[i]:.4f}")

    print(f"\nRatio values per bin (mumu):")
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        print(f"  [{low:.0f}, {high:.0f}): {ratio_mumu[i]:.4f} +/- {ratio_mumu_errors[i]:.4f}")

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = repo_root() / "plots" / run / str(year) / args.era
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
