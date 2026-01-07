#!/usr/bin/env python3
"""
compare_dy.py

Compare DYJets histograms processed by dy_binned_nlo, dy_ht and dy_ptll_nlo pipelines
for Run3Summer22, Run3Summer22EE, or their combination ("2022").
Overlays histograms with error bars, applies per‐variable plot settings
(rebin, xlim, ylim) from YAML, and saves into EOS under “compare_dy”.
Draws all regions and variables defined via Region and Variable objects.
Default era is 2022.  Now with ratio panel (NLO/HT and PTLL/HT).
"""

import argparse
import logging
from pathlib import Path
import sys

import yaml
import uproot
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

# allow imports of Plotter, Variable & Region
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from python.plotter import Plotter
from src.plotting_helpers import custom_log_formatter, set_y_label, plot_stack
from python.io import data_path, read_yaml, read_json,output_dir, save_figure
from python.histo import load_and_rebin
from python.regions import regions_for_era, expand_region_requests
from python.variables import Variable, build_variables
from python.sample_groups import load_sample_groups
from python.config import list_eras,load_lumi,load_plot_settings,load_kfactors,get_kfactor,index_plot_settings, get_var_cfg

# where your ROOT files live
WORKING_DIR = Path('/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter')

# EOS output layout
RUN  = "Run3"
YEAR = "2022"

# 2024 NLO mll-binned DY sample
DY_MLL_2024_FILE = (
    WORKING_DIR
    / "rootfiles"
    / "Run3"
    / "2024"
    / "RunIII2024Summer24"
    / "WRAnalyzer_DYJets.root"
)

# era → run/year/lumi
ERA_CONFIG = {
    "Run3Summer22":   {"run": "Run3", "year": "2022", "lumi":  7.9804},
    "Run3Summer22EE": {"run": "Run3", "year": "2022", "lumi": 26.6717},
    "2022":           {"run": "Run3", "year": "2022", "lumi":  7.9804 + 26.6717},
    "RunIII2024Summer24": {"run": "Run3", "year": "2024", "lumi":  108.09},
}

# Font size constants
FONT_SIZE_TITLE  = 20
FONT_SIZE_LABEL  = 20
FONT_SIZE_LEGEND = 18

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare DYJets histograms between pipelines"
    )
    p.add_argument(
        "--era", "-e",
        dest="era",
        type=str,
        choices=list(ERA_CONFIG.keys()),
        default="2022",
        help="Era to process (default: 2022). Options: Run3Summer22, Run3Summer22EE, or 2022"
    )
    return p.parse_args()

def load_plot_settings(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def load_histogram(path: Path, hist_key: str):
    with uproot.open(path) as f:
        return f[hist_key].to_hist()

def extract_hist_data(h):
    edges = h.axes[0].edges
    vals  = h.values()
    errs  = np.sqrt(h.variances()) if h.variances() is not None else np.sqrt(vals)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers, vals, errs

def load_mll_2024(plotter, hist_key):
    """
    Load the 2024 NLO mll-binned DY histogram, translating 2022-style
    (region, variable) names to the 2024 naming scheme.

    Examples:
      2022: wr_mumu_resolved_dy_cr/pt_leading_jet_wr_mumu_resolved_dy_cr
      2024: wr_resolved_flavor_cr/pt_leadjet_wr_resolved_flavor_cr
    """
    import uproot

    # --- parse the 2022-style key ---
    # e.g. "wr_mumu_resolved_dy_cr/pt_leading_jet_wr_mumu_resolved_dy_cr"
    if "/" in hist_key:
        old_region = hist_key.split("/")[0]
        base = hist_key.split("/")[-1]
    else:
        old_region = None
        base = hist_key

    # base ~ "pt_leading_jet_wr_mumu_resolved_dy_cr"
    varname_old = base
    region_suffix_old = None
    if "_wr_" in base:
        varname_old, region_suffix_old = base.split("_wr_", 1)

    # --- variable name aliases: 2022 → 2024 ---
    var_alias = {
        "pt_leading_jet":     "pt_leadjet",
        "pt_leading_lepton":  "pt_leadlep",
        "phi_leading_jet":    "phi_leadjet",
        "phi_leading_lepton": "phi_leadlep",
        # keep others unchanged by default
    }

    try:
        with uproot.open(DY_MLL_2024_FILE) as f:
            # strip cycle numbers (;1, ;2, ...)
            keys_no_cycle = [k.split(";")[0] for k in f.keys()]

            # 1) Try exact original key first (unlikely, but cheap)
            if hist_key in keys_no_cycle:
                raw = f[hist_key].to_hist()
            else:
                # 2) Try just the base name (no directory)
                if base in keys_no_cycle:
                    raw = f[base].to_hist()
                else:
                    # 3) Build 2024-style region + variable name and try that
                    #    Discover available region dirs, e.g. "wr_resolved_flavor_cr"
                    region_dirs = sorted(
                        {k.split("/", 1)[0] for k in keys_no_cycle if "/" in k}
                    )

                    new_region = None

                    if region_suffix_old is not None:
                        # map DY CR regions (ee/mumu) → resolved_flavor CR
                        if region_suffix_old.endswith("resolved_dy_cr"):
                            for r in region_dirs:
                                if "resolved" in r and "flavor" in r and "cr" in r:
                                    new_region = r
                                    break
                        # map SR regions (ee/mumu) → common resolved SR
                        elif region_suffix_old.endswith("resolved_sr"):
                            for r in region_dirs:
                                if "resolved" in r and "sr" in r:
                                    new_region = r
                                    break

                    # sensible fallbacks if we didn't find a mapping
                    if new_region is None:
                        if old_region in region_dirs:
                            new_region = old_region
                        elif region_dirs:
                            new_region = region_dirs[0]  # last resort

                    # variable rename
                    var_core_new = var_alias.get(varname_old, varname_old)

                    # candidate keys in 2024 file
                    candidates = []

                    if new_region:
                        # full dir + new_region suffix
                        candidates.append(
                            f"{new_region}/{var_core_new}_wr_{new_region}"
                        )
                        # also try just basename with new suffix
                        candidates.append(
                            f"{var_core_new}_wr_{new_region}"
                        )

                    # last-resort: just the var_core_new by itself
                    candidates.append(var_core_new)

                    raw = None
                    for cand in candidates:
                        if cand in keys_no_cycle:
                            # find the exact key including cycle
                            for k_full in f.keys():
                                if k_full.split(";")[0] == cand:
                                    logging.info(
                                        f"[2024 NLO] Using '{k_full.split(';')[0]}' for {hist_key}"
                                    )
                                    raw = f[k_full].to_hist()
                                    break
                        if raw is not None:
                            break

                    if raw is None:
                        logging.error(
                            f"None of the candidate keys {candidates} were found in "
                            f"{DY_MLL_2024_FILE}"
                        )
                        return None

    except Exception:
        logging.exception(f"Failed to load {hist_key} from {DY_MLL_2024_FILE}")
        return None

    # Rebin + scale
    h = plotter.rebin_hist(raw)
    h = plotter.scale_hist(h)
    return h

def load_and_combine(plotter, hist_subdir, hist_key):
    combined  = None
    orig_lumi = plotter.lumi
    for indir, sublumi in zip(plotter.input_directory, plotter.input_lumis):
        fp = indir / hist_subdir / 'WRAnalyzer_DYJets.root'
        try:
            raw = load_histogram(fp, hist_key)
        except Exception:
            continue
        h = plotter.rebin_hist(raw)
        plotter.lumi = sublumi
        h = plotter.scale_hist(h)
        if hist_subdir == "dy_ht":
            h = h * 1.4
        elif hist_subdir == "dy_ptll_nlo":
            h = h * 0.941
        combined = h if combined is None else combined + h
    plotter.lumi = orig_lumi
    return combined

def plot_overlays(plotter, region, variable, h_nlo_mll, h_ht):
    # extract data
    edges,   cen_p, val_p, err_p = extract_hist_data(h_nlo_mll)
    _,   cen_h, val_h, err_h = extract_hist_data(h_ht)

    # build figure with two pads
    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={'height_ratios':[3,1], 'hspace':0.10},
    )
    hep.style.use("CMS")

    # --- main overlay on ax ---
    for vals, errs, color, label in [
        (val_p, err_p, 'C1', r"2024 NLO $m_{\ell\ell}$ Binned"),
        (val_h, err_h, 'C2', f"2022 LO HT Binned"),
    ]:
        ax.step(edges, np.append(vals, vals[-1]), where='post',
                color=color, label=label)
        ax.errorbar(0.5*(edges[:-1]+edges[1:]), vals, yerr=errs,
                    fmt='none', capsize=2, color=color)

    xlabel = variable.tlatex_alias + (f" [{variable.unit}]" if variable.unit else "")
    ax.set_ylabel("Events")
    ax.set_yscale("log")
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right')
    alias = region.tlatex_alias.replace("\\n", "\n")
    alias = alias.replace("2022", "DY Comparison")
    ax.text(
        0.05, 0.96, alias,
        transform=ax.transAxes,
        fontsize=20,
        va="top"
    )
    hep.cms.label(
        loc=0, 
        ax=ax,
        data=region.unblind_data,
        label="Work in Progress",
        lumi=f"{plotter.lumi:.1f}",
        com=13.6,
        fontsize=FONT_SIZE_LABEL
    )
    ax.set_xlim(*plotter.xlim)
    ax.set_ylim(*plotter.ylim)
    ax.tick_params(labelbottom=False)

    # --- ratio panel on axr ---
    mask = val_h > 0
    ratio_p = np.zeros_like(val_h)
    ratio_p[mask] = val_p[mask] / val_h[mask]

    err_ratio_p = np.zeros_like(err_p)
    err_ratio_p[mask] = np.sqrt((err_p[mask]/val_h[mask])**2 +
                                 (val_p[mask]*err_h[mask]/val_h[mask]**2)**2)

    axr.errorbar(cen_p, ratio_p, yerr=err_ratio_p, xerr=True, fmt='o', capsize=2, color='black')

    axr.axhline(1, color='black', linestyle='--', linewidth=1)
    axr.set_xlabel(xlabel)
    axr.set_ylabel(r"NLO / LO")
    axr.set_xlim(*plotter.xlim)
    axr.set_ylim(0.5,1.6)
    axr.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right')

    return fig

def main():
    args = parse_args()
    setup_logging()

    # initialize Plotter
    plotter      = Plotter()
    cfg0         = ERA_CONFIG[args.era]
    plotter.run  = cfg0["run"]
    plotter.year = cfg0["year"]
    plotter.era  = args.era
    plotter.lumi = cfg0["lumi"]
    plotter.scale = True

    regions = regions_for_era(args.era)
    plotter.regions_to_draw = regions
    plotter.print_regions()


    # define Variables
    plotter.variables_to_draw = [
        Variable('mass_fourobject',            r'$m_{lljj}$',                   'GeV'),
        Variable('pt_dilepton',                r'$p_{T}^{ll}$',                 'GeV'),
    ]

    # build input dirs & lumis
    if args.era == "2022":
        d1 = WORKING_DIR / 'rootfiles' / plotter.run / plotter.year / 'Run3Summer22'
        d2 = WORKING_DIR / 'rootfiles' / plotter.run / plotter.year / 'Run3Summer22EE'
        plotter.input_directory = [d1, d2]
        plotter.input_lumis      = [
            ERA_CONFIG["Run3Summer22"]["lumi"],
            ERA_CONFIG["Run3Summer22EE"]["lumi"],
        ]
    else:
        base = WORKING_DIR / 'rootfiles' / plotter.run / plotter.year / args.era
        plotter.input_directory = [base]
        plotter.input_lumis      = [cfg0["lumi"]]

    # load YAML
    cfg_path = Path(f"data/plot_settings/dy.yaml")
    if not cfg_path.is_file():
        logging.error(f"Plot-config YAML not found: {cfg_path}")
        sys.exit(1)
    plot_settings = load_plot_settings(cfg_path)

    # loop and draw
    for region in plotter.regions_to_draw:
        logging.info(f"Region: {region.name}")
        vars_cfg = plot_settings.get(region.name, {})
        if not vars_cfg:
            logging.warning(f"No YAML for {region.name}; skipping.")
            continue

        for variable in plotter.variables_to_draw:
            name = variable.name
            if name not in vars_cfg:
                continue
            cfg = vars_cfg[name]

            plotter.configure_axes(
                nrebin=int(cfg['rebin']),
                xlim=(float(cfg['xlim'][0]), float(cfg['xlim'][1])),
                ylim=(float(cfg['ylim'][0]), float(cfg['ylim'][1])),
            )

            hist_key = f"{region.name}/{name}_{region.name}"
            h_nlo  = load_mll_2024(plotter, hist_key)
            h_ht    = load_and_combine(plotter, 'dy_ht',         hist_key)

            if any(h is None for h in (h_nlo, h_ht)):
                logging.error(f"{name} missing; skipping.")
                continue

            fig = plot_overlays(plotter, region, variable, h_nlo, h_ht)

            outpath = (Path(f"/eos/user/w/wijackso/{RUN}/{YEAR}/{plotter.era}")
                       / 'compare_dy'
                       / f"{region.name}_{region.primary_dataset}"
                       / f"{name}_{region.name}.pdf")
            save_figure(fig, outpath)
            print(outpath)
            plt.close(fig)

if __name__ == "__main__":
    main()
