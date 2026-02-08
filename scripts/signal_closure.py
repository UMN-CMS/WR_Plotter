#!/usr/bin/env python3
"""
compare_closure.py

Compare Run 2 (RunIISummer20UL18) and Run 3 (RunIII2024Summer24) signal
distributions for the 6 common mass points.

Produces:
  1) Gen-level shape overlays (normalised to unit area) + ratio panel
  2) Reco-level shape overlays (normalised to unit area) + ratio panel
  3) Cutflow efficiency comparison (bar chart)

Usage:
    python WR_Plotter/bin/compare_closure.py
    python WR_Plotter/bin/compare_closure.py --mass WR2000_N1900
    python WR_Plotter/bin/compare_closure.py --mass WR2000_N1900 --output-dir ./my_plots
"""

import argparse
import logging
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

# Gen-level variables to compare.
# Format: (name, label, unit, rebin_factor)
# Bin counts from analyzer: pT -> 200 bins, eta -> 60, phi -> 80,
# mass_dilepton -> 5000, mass_dijet -> 500, mass_fourobject -> 800, pt_dilepton -> 200
GEN_VARIABLES = [
    ("gen_pt_leading_lepton",       r"Gen $p_{T}$ leading lepton",       "GeV", 5),
    ("gen_eta_leading_lepton",      r"Gen $\eta$ leading lepton",        "",     2),
    ("gen_pt_subleading_lepton",    r"Gen $p_{T}$ subleading lepton",    "GeV", 5),
    ("gen_eta_subleading_lepton",   r"Gen $\eta$ subleading lepton",     "",     2),
    ("gen_pt_leading_jet",          r"Gen $p_{T}$ leading jet",          "GeV", 5),
    ("gen_eta_leading_jet",         r"Gen $\eta$ leading jet",           "",     2),
    ("gen_pt_subleading_jet",       r"Gen $p_{T}$ subleading jet",       "GeV", 5),
    ("gen_eta_subleading_jet",      r"Gen $\eta$ subleading jet",        "",     2),
    ("gen_mass_dilepton",           r"Gen $m_{\ell\ell}$",               "GeV", 50),
    ("gen_mass_dijet",              r"Gen $m_{jj}$",                     "GeV", 10),
    ("gen_mass_fourobject",         r"Gen $m_{\ell\ell jj}$",            "GeV", 20),
]

# Reco-level variables to compare.
RECO_VARIABLES = [
    ("pt_leading_lepton",           r"$p_{T}$ leading lepton",           "GeV", 5),
    ("eta_leading_lepton",          r"$\eta$ leading lepton",            "",     2),
    ("pt_subleading_lepton",        r"$p_{T}$ subleading lepton",        "GeV", 5),
    ("eta_subleading_lepton",       r"$\eta$ subleading lepton",         "",     2),
    ("pt_leading_jet",              r"$p_{T}$ leading jet",              "GeV", 5),
    ("eta_leading_jet",             r"$\eta$ leading jet",               "",     2),
    ("pt_subleading_jet",           r"$p_{T}$ subleading jet",           "GeV", 5),
    ("eta_subleading_jet",          r"$\eta$ subleading jet",            "",     2),
    ("mass_dilepton",               r"$m_{\ell\ell}$",                   "GeV", 50),
    ("mass_dijet",                  r"$m_{jj}$",                         "GeV", 10),
    ("mass_fourobject",             r"$m_{\ell\ell jj}$",                "GeV", 20),
    ("pt_fourobject",               r"$p_{T,\ell\ell jj}$",             "GeV", 20),
]

# Gen region.
GEN_REGION = "gen_inclusive"

# Reco regions (resolved SR).
RECO_REGIONS = [
    ("wr_ee_resolved_sr",       r"$ee$ resolved SR"),
    ("wr_mumu_resolved_sr",     r"$\mu\mu$ resolved SR"),
    ("wr_ee_resolved_dy_cr_1",  r"$ee$ resolved DY CR (60–120)"),
    ("wr_mumu_resolved_dy_cr_1", r"$\mu\mu$ resolved DY CR (60–120)"),
]

# Cutflow step labels (must match analyzer_closure.py chain order).
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


def load_hist_1d(root_file_path, region, hist_name):
    """Load a histogram from a ROOT file and project out the process axis if present."""
    key = f"{region}/{hist_name}_{region}"
    try:
        with uproot.open(root_file_path) as f:
            h = f[key].to_hist()
    except Exception as e:
        logging.warning(f"Could not load {key} from {root_file_path}: {e}")
        return None

    # Project down to 1D: keep only the Regular (numeric) axis,
    # discarding any StrCategory axes (process, region, syst, etc.).
    # After split_hists_with_syst, the saved histogram typically has
    # a StrCategory "xaxis" (process) + Regular "yaxis" (physics variable).
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


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------
def plot_overlay(h_run2, h_run3, *,
                 xlabel, region_label, mass_label,
                 normalize=True, rebin_n=1, logy=False):
    """Overlay two 1D histograms with a ratio panel (Run 3 / Run 2)."""
    h2 = rebin_histogram(h_run2, rebin_n) if rebin_n > 1 else h_run2
    h3 = rebin_histogram(h_run3, rebin_n) if rebin_n > 1 else h_run3

    edges2, vals2, errs2 = extract(h2)
    edges3, vals3, errs3 = extract(h3)

    bw = np.diff(edges2)

    if normalize:
        vals2, errs2 = normalize_to_unity(vals2, errs2, bw)
        vals3, errs3 = normalize_to_unity(vals3, errs3, bw)

    centers = 0.5 * (edges2[:-1] + edges2[1:])

    fig, (ax, rax) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.10},
        figsize=(8, 7),
    )

    # --- Main panel ---
    ax.step(edges2, np.append(vals2, vals2[-1]), where="post",
            color=RUN2_COLOR, linewidth=1.5, label=RUN2_LABEL)
    ax.errorbar(centers, vals2, yerr=errs2,
                fmt="none", capsize=2, color=RUN2_COLOR)

    ax.step(edges3, np.append(vals3, vals3[-1]), where="post",
            color=RUN3_COLOR, linewidth=1.5, label=RUN3_LABEL)
    ax.errorbar(centers, vals3, yerr=errs3,
                fmt="none", capsize=2, color=RUN3_COLOR)

    if logy:
        ax.set_yscale("log")
    ylabel = "Normalised" if normalize else "Events"
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=14, loc="upper right")

    # Region + mass annotation.
    ax.text(0.05, 0.96, f"{region_label}\n{mass_label}",
            transform=ax.transAxes, fontsize=14, va="top")

    hep.cms.label(loc=0, ax=ax, data=False,
                  label="Work in Progress", fontsize=14)

    ax.tick_params(labelbottom=False)

    # --- Ratio panel (Run 3 / Run 2) ---
    mask = vals2 > 0
    ratio = np.ones_like(vals2)
    ratio_err = np.zeros_like(vals2)
    ratio[mask] = vals3[mask] / vals2[mask]
    ratio_err[mask] = np.sqrt(
        (errs3[mask] / vals2[mask]) ** 2 +
        (vals3[mask] * errs2[mask] / (vals2[mask] ** 2)) ** 2
    )

    rax.errorbar(centers, ratio, yerr=ratio_err,
                 fmt="o", markersize=3, capsize=2, color="black")
    rax.axhline(1.0, ls="--", color="gray", linewidth=1)
    rax.set_xlabel(xlabel)
    rax.set_ylabel(f"{RUN3_LABEL} / {RUN2_LABEL}", fontsize=10)
    rax.set_ylim(0.5, 1.5)
    rax.set_xlim(edges2[0], edges2[-1])

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
            logging.warning(f"Could not load cutflow for {flavor}: {e}")
            continue

        # The histogram has axes: (StrCategory process/campaign, IntCategory step).
        # Project out any StrCategory axes to get 1D (step only).
        def _project_to_1d(h):
            import hist as hist_mod
            if h.ndim > 1:
                int_axes = [i for i, ax in enumerate(h.axes)
                            if isinstance(ax, (hist_mod.axis.IntCategory,
                                               hist_mod.axis.Regular,
                                               hist_mod.axis.Variable))]
                if int_axes:
                    return h.project(*int_axes)
                # Fallback: sum over first axis
                return h[sum, :]
            return h

        h2_1d = _project_to_1d(h2)
        h3_1d = _project_to_1d(h3)

        vals2 = h2_1d.values().flatten()
        vals3 = h3_1d.values().flatten()

        # Normalise to first step to get relative efficiency.
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

        # Ratio panel.
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
        logging.info(f"  Saved {outpath}")
        plt.close(fig)


def format_mass(mass):
    """Format mass point for plot labels: WR2000_N1900 -> (m_WR, m_N) = (2000, 1900) GeV."""
    import re
    m = re.search(r"WR(\d+)_N(\d+)", mass)
    if m:
        return rf"$(m_{{W_R}}, m_N) = ({m.group(1)}, {m.group(2)})$ GeV"
    return mass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Compare Run 2 / Run 3 closure study results.")
    p.add_argument(
        "--mass", "-m",
        type=str, default=None,
        help="Single mass point to plot (e.g. WR2000_N1900). Default: all 6 common points.",
    )
    p.add_argument(
        "--output-dir", "-o",
        type=str, default=None,
        help="Output directory for plots. Default: WR_Plotter/plots/closure/",
    )
    p.add_argument(
        "--normalize", action="store_true", default=True,
        help="Normalise distributions to unit area (default: True).",
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

    for mass in masses:
        logging.info(f"=== {mass} ===")
        mass_label = format_mass(mass)

        run2_file = root_file_path(RUN2_ERA, mass)
        run3_file = root_file_path(RUN3_ERA, mass)

        if not run2_file.exists():
            logging.error(f"  Missing Run 2 file: {run2_file}")
            continue
        if not run3_file.exists():
            logging.error(f"  Missing Run 3 file: {run3_file}")
            continue

        # --- Gen-level plots ---
        logging.info("  Gen-level distributions")
        for var_name, var_label, var_unit, rebin_n in GEN_VARIABLES:
            h2 = load_hist_1d(run2_file, GEN_REGION, var_name)
            h3 = load_hist_1d(run3_file, GEN_REGION, var_name)
            if h2 is None or h3 is None:
                logging.warning(f"    Skipping {var_name} (missing histogram)")
                continue

            xlabel = f"{var_label} [{var_unit}]" if var_unit else var_label
            fig = plot_overlay(
                h2, h3,
                xlabel=xlabel,
                region_label="Gen-level (inclusive)",
                mass_label=mass_label,
                normalize=args.normalize,
                rebin_n=rebin_n,
            )

            outpath = out_base / mass / "gen" / f"{var_name}_{mass}.pdf"
            save_figure(fig, outpath)
            logging.info(f"    {var_name}")
            plt.close(fig)

        # --- Reco-level plots ---
        for region_name, region_label in RECO_REGIONS:
            logging.info(f"  Reco: {region_name}")
            for var_name, var_label, var_unit, rebin_n in RECO_VARIABLES:
                h2 = load_hist_1d(run2_file, region_name, var_name)
                h3 = load_hist_1d(run3_file, region_name, var_name)
                if h2 is None or h3 is None:
                    continue

                xlabel = f"{var_label} [{var_unit}]" if var_unit else var_label
                fig = plot_overlay(
                    h2, h3,
                    xlabel=xlabel,
                    region_label=region_label,
                    mass_label=mass_label,
                    normalize=args.normalize,
                    rebin_n=rebin_n,
                )

                outpath = out_base / mass / region_name / f"{var_name}_{mass}.pdf"
                save_figure(fig, outpath)
                logging.info(f"    {var_name}")
                plt.close(fig)

        # --- Cutflow comparison ---
        logging.info("  Cutflows")
        cutflow_dir = out_base / mass / "cutflow"
        plot_cutflow(run2_file, run3_file, mass, cutflow_dir)

    logging.info("Done.")


if __name__ == "__main__":
    main()
