#!/usr/bin/env python3
"""
compare_dy.py

Compare DYJets histograms in three modes:

  lo-nlo      Compare LO vs NLO ROOT files for a single era.
              Loads from {era}/lo/ and {era}/nlo/ subdirectories.

  mll-vs-ht   Compare 2024 NLO mll-binned DY against 2022 LO HT-binned DY.
              Loads 2024 from a dedicated file (with hist-key translation)
              and 2022 from dy_ht subdirectories (combining sub-eras).

  cross-era   Compare DYJets between two eras (--era vs --ref-era).
              Loads WRAnalyzer_DYJets.root from each era's directory.

Usage:
    python bin/compare_dy.py --mode lo-nlo     --era RunIII2024Summer24
    python bin/compare_dy.py --mode mll-vs-ht  --era 2022
    python bin/compare_dy.py --mode cross-era  --era RunIII2024Summer24 --ref-era RunIISummer20UL18
"""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wrplotter.histogram_utils import rebin_histogram, load_histogram, extract_hist_data
from wrplotter.plotting_helpers import FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_LEGEND
from wrplotter.paths import repo_root, output_dir, save_figure
from wrplotter.regions import regions_for_era
from wrplotter.variables import Variable
from wrplotter.config import load_plot_settings, load_kfactors, load_lumi, list_eras
from wrplotter.cli_utils import setup_logging

# 2024 NLO mll-binned DY sample (used by mll-vs-ht mode)
DY_MLL_2024_FILE = (
    repo_root()
    / "rootfiles"
    / "Run3"
    / "2024"
    / "RunIII2024Summer24"
    / "WRAnalyzer_DYJets.root"
)

_DY_COMP_KFACTORS = load_kfactors().get("dy_comparison", {})

VARIABLES = [
    Variable('mass_fourobject',      r'$m_{lljj}$',            'GeV'),
    Variable('mass_twoobject',       r'$m_{lJ}$',              'GeV'),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Compare DYJets histograms")
    p.add_argument(
        "--mode", "-m",
        required=True,
        choices=["lo-nlo", "mll-vs-ht", "cross-era", "three-way"],
        help="lo-nlo: LO vs NLO within one era. mll-vs-ht: 2024 NLO mll-binned vs 2022 LO HT-binned. cross-era: compare DYJets between two eras. three-way: compare LO inc, NLO inc, and LO HT DY.",
    )
    p.add_argument(
        "--era", "-e",
        type=str,
        choices=list_eras(),
        default=None,
        help="Era to process (numerator in cross-era). Defaults to RunIII2024Summer24 (lo-nlo) or 2022 (mll-vs-ht/cross-era).",
    )
    p.add_argument(
        "--ref-era",
        type=str,
        choices=list_eras(),
        default=None,
        help="Reference era (denominator) for cross-era mode.",
    )
    p.add_argument(
        "--plot-config", "-c",
        dest="plot_config",
        type=str,
        default="data/plot_settings/dy.yaml",
        help="YAML plot settings file (rebin/xlim/ylim per region/variable).",
    )
    p.add_argument(
        "--date", "-d",
        type=str,
        default=None,
        help="Date prefix for three-way mode (e.g. 20260313).",
    )
    p.add_argument(
        "--variable-rebin",
        action="store_true",
        help="Use variable-width bin edges (from rebin_variable in YAML) where available.",
    )
    args = p.parse_args()

    # mode-dependent default era
    if args.era is None:
        args.era = "RunIII2024Summer24" if args.mode == "lo-nlo" else "2022"

    if args.mode == "cross-era" and args.ref_era is None:
        p.error("--ref-era is required for cross-era mode")

    if args.mode == "three-way" and args.date is None:
        p.error("--date is required for three-way mode")

    return args


# ---------------------------------------------------------------------------
# Histogram loaders
# ---------------------------------------------------------------------------

def load_one(n_rebin, root_file: Path, hist_key: str, extra_scale: float = 1.0):
    """Load a single histogram from a ROOT file and rebin."""
    if not root_file.is_file():
        logging.error(f"Missing ROOT file: {root_file}")
        return None
    try:
        raw = load_histogram(root_file, hist_key)
    except Exception as e:
        logging.error(f"Failed reading {hist_key} from {root_file}: {e}")
        return None
    h = rebin_histogram(raw, n_rebin)
    if extra_scale != 1.0:
        h = h * extra_scale
    return h


def load_and_combine(input_dirs, input_lumis, n_rebin, hist_subdir, hist_key):
    """Load + rebin DYJets from multiple sub-era directories, applying k-factors."""
    combined = None
    for indir, sublumi in zip(input_dirs, input_lumis):
        fp = indir / hist_subdir / 'WRAnalyzer_DYJets.root'
        try:
            raw = load_histogram(fp, hist_key)
        except FileNotFoundError:
            continue
        except Exception:
            logging.warning(f"Failed reading {hist_key} from {fp}", exc_info=True)
            continue
        h = rebin_histogram(raw, n_rebin)
        k = _DY_COMP_KFACTORS.get(hist_subdir, 1.0)
        h = h * k
        combined = h if combined is None else combined + h
    return combined


def load_mll_2024(n_rebin, hist_key):
    """
    Load the 2024 NLO mll-binned DY histogram, translating 2022-style
    (region, variable) names to the 2024 naming scheme.

    Examples:
      2022: wr_mumu_resolved_dy_cr/pt_leading_jet_wr_mumu_resolved_dy_cr
      2024: wr_resolved_flavor_cr/pt_leadjet_wr_resolved_flavor_cr
    """
    import uproot

    if "/" in hist_key:
        old_region = hist_key.split("/")[0]
        base = hist_key.split("/")[-1]
    else:
        old_region = None
        base = hist_key

    varname_old = base
    region_suffix_old = None
    if "_wr_" in base:
        varname_old, region_suffix_old = base.split("_wr_", 1)

    var_alias = {
        "pt_leading_jet":     "pt_leadjet",
        "pt_leading_lepton":  "pt_leadlep",
        "phi_leading_jet":    "phi_leadjet",
        "phi_leading_lepton": "phi_leadlep",
    }

    try:
        with uproot.open(DY_MLL_2024_FILE) as f:
            keys_no_cycle = [k.split(";")[0] for k in f.keys()]

            if hist_key in keys_no_cycle:
                raw = f[hist_key].to_hist()
            elif base in keys_no_cycle:
                raw = f[base].to_hist()
            else:
                region_dirs = sorted(
                    {k.split("/", 1)[0] for k in keys_no_cycle if "/" in k}
                )

                new_region = None
                if region_suffix_old is not None:
                    if region_suffix_old.endswith("resolved_dy_cr"):
                        for r in region_dirs:
                            if "resolved" in r and "flavor" in r and "cr" in r:
                                new_region = r
                                break
                    elif region_suffix_old.endswith("resolved_sr"):
                        for r in region_dirs:
                            if "resolved" in r and "sr" in r:
                                new_region = r
                                break

                if new_region is None:
                    if old_region in region_dirs:
                        new_region = old_region
                    elif region_dirs:
                        new_region = region_dirs[0]

                var_core_new = var_alias.get(varname_old, varname_old)

                candidates = []
                if new_region:
                    candidates.append(f"{new_region}/{var_core_new}_wr_{new_region}")
                    candidates.append(f"{var_core_new}_wr_{new_region}")
                candidates.append(var_core_new)

                raw = None
                for cand in candidates:
                    if cand in keys_no_cycle:
                        for k_full in f.keys():
                            if k_full.split(";")[0] == cand:
                                logging.info(f"[2024 NLO] Using '{k_full.split(';')[0]}' for {hist_key}")
                                raw = f[k_full].to_hist()
                                break
                    if raw is not None:
                        break

                if raw is None:
                    logging.error(
                        f"None of the candidate keys {candidates} were found in {DY_MLL_2024_FILE}"
                    )
                    return None

    except Exception:
        logging.exception(f"Failed to load {hist_key} from {DY_MLL_2024_FILE}")
        return None

    return rebin_histogram(raw, n_rebin)


def load_three_way_sample(n_rebin, era_key, date, subdir_suffix, hist_key):
    """Load DYJets from dated subdirectories, combining sub-eras if present.

    For example, with era_key='2022', date='20260313', subdir_suffix='lo_inc_dy',
    loads from Run3Summer22/20260313_lo_inc_dy/ and Run3Summer22EE/20260313_lo_inc_dy/.
    """
    info = load_lumi(era_key)
    run, year = info["run"], info["year"]
    sub_eras = info.get("sub_eras", [era_key])

    combined = None
    for se in sub_eras:
        fp = repo_root() / 'rootfiles' / run / year / se / f'{date}_{subdir_suffix}' / 'WRAnalyzer_DYJets.root'
        try:
            raw = load_histogram(fp, hist_key)
        except FileNotFoundError:
            logging.warning(f"File not found: {fp}")
            continue
        except Exception:
            logging.warning(f"Failed reading {hist_key} from {fp}", exc_info=True)
            continue
        h = rebin_histogram(raw, n_rebin)
        combined = h if combined is None else combined + h
    return combined


def load_era_dy(n_rebin, era_key, hist_key):
    """Load DYJets from an era, combining sub-eras if present."""
    info = load_lumi(era_key)
    run, year = info["run"], info["year"]
    sub_eras  = info.get("sub_eras", [era_key])
    sub_lumis = info.get("sub_lumis", [info["lumi"]])

    combined = None
    for se, sl in zip(sub_eras, sub_lumis):
        fp = repo_root() / 'rootfiles' / run / year / se / 'WRAnalyzer_DYJets.root'
        try:
            raw = load_histogram(fp, hist_key)
        except FileNotFoundError:
            continue
        except Exception:
            logging.warning(f"Failed reading {hist_key} from {fp}", exc_info=True)
            continue
        combined = rebin_histogram(raw, n_rebin) if combined is None else combined + rebin_histogram(raw, n_rebin)
    return combined


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overlays(region, variable, h_num, h_den, *,
                  xlim, ylim, lumi, com, num_label, den_label, ratio_label):
    edges, cen_n, val_n, err_n = extract_hist_data(h_num)
    _,     cen_d, val_d, err_d = extract_hist_data(h_den)

    # Normalize to unit area
    bin_widths = np.diff(edges)
    area_n = np.sum(val_n * bin_widths)
    area_d = np.sum(val_d * bin_widths)
    if area_n > 0:
        val_n, err_n = val_n / area_n, err_n / area_n
    if area_d > 0:
        val_d, err_d = val_d / area_d, err_d / area_d

    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.10},
    )
    hep.style.use("CMS")

    for vals, errs, color, label in [
        (val_n, err_n, 'C1', num_label),
        (val_d, err_d, 'C2', den_label),
    ]:
        ax.step(edges, np.append(vals, vals[-1]), where='post',
                color=color, label=label)
        ax.errorbar(0.5 * (edges[:-1] + edges[1:]), vals, yerr=errs,
                    fmt='none', capsize=2, color=color)

    xlabel = variable.tlatex_alias + (f" [{variable.unit}]" if variable.unit else "")
    ax.set_ylabel("Normalized to unit area")
    ax.set_yscale("log")
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right')

    alias = region.tlatex_alias.replace("\\n", "\n")
    ax.text(0.05, 0.96, alias, transform=ax.transAxes,
            fontsize=FONT_SIZE_TITLE, va='top')

    hep.cms.label(
        loc=0, ax=ax,
        data=region.unblind_data,
        label="Work in Progress",
        lumi=f"{lumi:.1f}",
        com=com,
        fontsize=FONT_SIZE_LABEL,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.tick_params(labelbottom=False)

    # ratio panel
    mask = val_d > 0
    ratio = np.zeros_like(val_d)
    ratio[mask] = val_n[mask] / val_d[mask]

    err_ratio = np.zeros_like(err_n)
    err_ratio[mask] = np.sqrt(
        (err_n[mask] / val_d[mask]) ** 2 +
        (val_n[mask] * err_d[mask] / val_d[mask] ** 2) ** 2
    )

    axr.errorbar(cen_n, ratio, yerr=err_ratio, xerr=True,
                 fmt='o', capsize=2, color='black')
    axr.axhline(1, color='black', linestyle='--', linewidth=1)
    axr.set_xlabel(xlabel)
    axr.set_ylabel(ratio_label)
    axr.set_xlim(*xlim)
    axr.set_ylim(0.5, 1.6)

    return fig


def plot_three_overlays(region, variable, histograms, *,
                        xlim, ylim, lumi, com, labels, colors=None):
    """Overlay three histograms with ratio panel (ratios to first histogram)."""
    if colors is None:
        colors = ['C0', 'C1', 'C2']

    hist_data = [extract_hist_data(h) for h in histograms]

    # Normalize each histogram to unit area
    normalized = []
    for edges, cen, vals, errs in hist_data:
        bin_widths = np.diff(edges)
        area = np.sum(vals * bin_widths)
        if area > 0:
            normalized.append((edges, cen, vals / area, errs / area))
        else:
            normalized.append((edges, cen, vals, errs))
    hist_data = normalized

    # NLO is index 0 (numerator for ratios)
    nlo_edges, nlo_cen, nlo_val, nlo_err = hist_data[0]

    hep.style.use("CMS")
    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 10),
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.10},
    )

    for (edges, cen, vals, errs), color, label in zip(hist_data, colors, labels):
        ax.step(edges, np.append(vals, vals[-1]), where='post',
                color=color, label=label)
        ax.errorbar(0.5 * (edges[:-1] + edges[1:]), vals, yerr=errs,
                    fmt='none', capsize=2, color=color)

    xlabel = variable.tlatex_alias + (f" [{variable.unit}]" if variable.unit else "")
    ax.set_ylabel("Normalized to unit area")
    ax.set_yscale("log")
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right')

    alias = region.tlatex_alias.replace("\\n", "\n")
    ax.text(0.05, 0.96, alias, transform=ax.transAxes,
            fontsize=FONT_SIZE_TITLE, va='top')

    hep.cms.label(
        loc=0, ax=ax,
        data=region.unblind_data,
        label="Work in Progress",
        lumi=f"{lumi:.1f}",
        com=com,
        fontsize=FONT_SIZE_LABEL,
    )
    ax.set_xlim(800,5000)
    ax.set_ylim(1e-6, 1e-0)
    ax.tick_params(labelbottom=False)

    # ratio panel: NLO / each LO sample
    for (edges, cen, den_val, den_err), color, label in zip(hist_data[1:], colors[1:], labels[1:]):
        mask = den_val > 0
        ratio = np.zeros_like(nlo_val)
        ratio[mask] = nlo_val[mask] / den_val[mask]
        err_ratio = np.zeros_like(nlo_err)
        err_ratio[mask] = np.sqrt(
            (nlo_err[mask] / den_val[mask]) ** 2 +
            (nlo_val[mask] * den_err[mask] / den_val[mask] ** 2) ** 2
        )
        axr.errorbar(cen, ratio, yerr=err_ratio, xerr=True,
                     fmt='o', capsize=2, color=color, label=f"NLO / {label}", markersize=3)

    axr.axhline(1, color='black', linestyle='--', linewidth=1)
    axr.set_xlabel(xlabel)
    axr.set_ylabel("NLO / LO")
    axr.set_xlim(800,5000)
    axr.set_ylim(0.0, 2.5)
    axr.legend(fontsize=FONT_SIZE_LEGEND - 2, loc='upper right')

    return fig


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    setup_logging()

    era_info = load_lumi(args.era)
    run  = era_info["run"]
    year = era_info["year"]
    era  = args.era
    lumi = era_info["lumi"]
    com  = era_info.get("com", 13.6)

    regions = regions_for_era(args.era)
    for r in regions:
        r.print()

    # sub-era dirs for combining (used by mll-vs-ht)
    sub_eras  = era_info.get("sub_eras", [args.era])
    sub_lumis = era_info.get("sub_lumis", [lumi])
    input_dirs  = [repo_root() / 'rootfiles' / run / year / se for se in sub_eras]
    input_lumis = sub_lumis

    # lo-nlo file paths
    base_dir = repo_root() / 'rootfiles' / run / year / args.era
    lo_file  = base_dir / "lo" / "WRAnalyzer_DYJets.root"
    nlo_file = base_dir / "nlo" / "WRAnalyzer_DYJets.root"

    # mode-dependent labels
    if args.mode == "lo-nlo":
        num_label   = "NLO"
        den_label   = "LO"
        ratio_label = "NLO / LO"
    elif args.mode == "mll-vs-ht":
        num_label   = r"2024 NLO $m_{\ell\ell}$ Binned"
        den_label   = "2022 LO HT Binned"
        ratio_label = "NLO / LO"
    elif args.mode == "three-way":
        three_way_labels = ["NLO Inclusive", "LO Inclusive", "LO HT-Binned"]
        three_way_subdirs = ["nlo_inc_dy", "lo_inc_dy", "lo_ht_dy"]
    else:  # cross-era
        num_label   = args.era
        den_label   = args.ref_era
        ratio_label = f"{args.era} / {args.ref_era}"

    plot_settings = load_plot_settings(args.plot_config)

    # For three-way mode, only process SR regions
    if args.mode == "three-way":
        regions = [r for r in regions if r.is_signal_region]

    for region in regions:
        logging.info(f"Region: {region.name}")
        vars_cfg = plot_settings.get(region.name, {})
        if not vars_cfg:
            logging.warning(f"No YAML for {region.name}; skipping.")
            continue

        for variable in VARIABLES:
            name = variable.name
            if name not in vars_cfg:
                continue
            cfg = vars_cfg[name]

            if args.variable_rebin and 'rebin_variable' in cfg:
                n_rebin = cfg['rebin_variable']
            else:
                n_rebin = cfg['rebin']
            xlim = (float(cfg['xlim'][0]), float(cfg['xlim'][1]))
            ylim = (float(cfg['ylim'][0]), float(cfg['ylim'][1]))

            hist_key = f"{region.name}/{name}_{region.name}"

            if args.mode == "three-way":
                histograms = []
                for subdir in three_way_subdirs:
                    h = load_three_way_sample(n_rebin, args.era, args.date, subdir, hist_key)
                    histograms.append(h)
                if any(h is None for h in histograms):
                    missing = [l for l, h in zip(three_way_labels, histograms) if h is None]
                    logging.error(f"{name}: missing samples {missing}; skipping.")
                    continue

                three_way_xlim = (xlim[0], 8000.0)
                fig = plot_three_overlays(
                    region, variable, histograms,
                    xlim=three_way_xlim, ylim=ylim, lumi=lumi, com=com,
                    labels=three_way_labels,
                )
                outpath = (output_dir(run, year, era, f"{args.date}_compare_dy_three_way")
                           / f"{region.name}_{region.primary_dataset}"
                           / f"{name}_{region.name}.pdf")
                save_figure(fig, outpath)
                print(outpath)
                plt.close(fig)
                continue

            if args.mode == "lo-nlo":
                h_num = load_one(n_rebin, nlo_file, hist_key)
                h_den = load_one(n_rebin, lo_file,  hist_key)
            elif args.mode == "mll-vs-ht":
                h_num = load_mll_2024(n_rebin, hist_key)
                h_den = load_and_combine(input_dirs, input_lumis, n_rebin, 'dy_ht', hist_key)
            else:  # cross-era
                h_num = load_era_dy(n_rebin, args.era, hist_key)
                h_den = load_era_dy(n_rebin, args.ref_era, hist_key)

            if h_num is None or h_den is None:
                logging.error(f"{name} missing; skipping.")
                continue

            fig = plot_overlays(
                region, variable, h_num, h_den,
                xlim=xlim, ylim=ylim, lumi=lumi, com=com,
                num_label=num_label, den_label=den_label,
                ratio_label=ratio_label,
            )

            outpath = (output_dir(run, year, era, "compare_dy")
                       / f"{region.name}_{region.primary_dataset}"
                       / f"{name}_{region.name}.pdf")
            save_figure(fig, outpath)
            print(outpath)
            plt.close(fig)


if __name__ == "__main__":
    main()
