#!/usr/bin/env python3
"""
compare_dy.py

Compare DYJets histograms between LO and NLO ROOT files for RunIII2024Summer24.
Overlays histograms with error bars, applies per-variable plot settings
(rebin, xlim, ylim) from YAML, and saves into EOS under “compare_dy”.
Draws all regions and variables defined via Region and Variable objects.
Now with ratio panel (NLO/LO).
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
from python.io import save_figure
from python.regions import regions_for_era
from python.variables import Variable

# where your ROOT files live
WORKING_DIR = Path('/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter')

# === NEW: explicit DY inputs (2024 Summer24) ===
DY_LO_FILE  = Path("/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/lo/WRAnalyzer_DYJets.root")
DY_NLO_FILE = Path("/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter/rootfiles/Run3/2024/RunIII2024Summer24/nlo/WRAnalyzer_DYJets.root")

# EOS output layout
RUN  = "Run3"
YEAR = "2024"

# If your Plotter needs a lumi value for scaling, set it here.
# Ratio is insensitive to this as long as both use the same lumi/scaling rules.
ERA_CONFIG = {
    "RunIII2024Summer24": {"run": "Run3", "year": "2024", "lumi": 109.08},
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
    p = argparse.ArgumentParser(description="Compare DYJets histograms between LO and NLO files (RunIII2024Summer24)")
    p.add_argument(
        "--era", "-e",
        dest="era",
        type=str,
        choices=list(ERA_CONFIG.keys()),
        default="RunIII2024Summer24",
        help="Era to process (default: RunIII2024Summer24)"
    )
    p.add_argument(
        "--plot-config", "-c",
        dest="plot_config",
        type=str,
        default="data/plot_settings/dy.yaml",
        help="YAML plot settings file (rebin/xlim/ylim per region/variable). Default kept as 2022.yaml."
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

def load_one(plotter, root_file: Path, hist_key: str, extra_scale: float = 1.0):
    """
    Load a single histogram from a specific ROOT file, then rebin and scale using Plotter.
    """
    if not root_file.is_file():
        logging.error(f"Missing ROOT file: {root_file}")
        return None

    try:
        raw = load_histogram(root_file, hist_key)
    except Exception as e:
        logging.error(f"Failed reading {hist_key} from {root_file}: {e}")
        return None

    h = plotter.rebin_hist(raw)
    if plotter.scale:
        h = plotter.scale_hist(h)
    if extra_scale != 1.0:
        h = h * extra_scale
    return h

def plot_overlays(plotter, region, variable, h_nlo, h_lo):
    # extract data
    edges, cen_n, val_n, err_n = extract_hist_data(h_nlo)
    _,     cen_l, val_l, err_l = extract_hist_data(h_lo)

    # build figure with two pads
    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={'height_ratios':[3,1], 'hspace':0.10},
    )
    hep.style.use("CMS")

    # --- main overlay on ax ---
    for vals, errs, color, label in [
        (val_n, err_n, 'C1', "NLO"),
        (val_l, err_l, 'C2', "LO"),
    ]:
        ax.step(edges, np.append(vals, vals[-1]), where='post',
                color=color, label=label)
        ax.errorbar(0.5*(edges[:-1]+edges[1:]), vals, yerr=errs,
                    fmt='none', capsize=2, color=color)

    xlabel = variable.tlatex_alias + (f" [{variable.unit}]" if variable.unit else "")
    ax.set_ylabel("Events")
    ax.set_yscale("log")
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right')
    print(region.tlatex_alias)
    label = region.tlatex_alias.replace(r"\n", "\n")
    ax.text(0.05, 0.96, label, transform=ax.transAxes,fontsize=FONT_SIZE_TITLE, va="top")
#    ax.text(0.05, 0.96, region.tlatex_alias, transform=ax.transAxes, fontsize=FONT_SIZE_TITLE, va='top')

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

    # --- ratio panel on axr (NLO/LO) ---
    mask = val_l > 0
    ratio = np.zeros_like(val_l)
    ratio[mask] = val_n[mask] / val_l[mask]

    err_ratio = np.zeros_like(err_n)
    err_ratio[mask] = np.sqrt(
        (err_n[mask] / val_l[mask])**2 +
        (val_n[mask] * err_l[mask] / (val_l[mask]**2))**2
    )

    axr.errorbar(cen_n, ratio, yerr=err_ratio, xerr=True, fmt='o', capsize=2, color='black')
    axr.axhline(1, color='black', linestyle='--', linewidth=1)
    axr.set_xlabel(xlabel)
    axr.set_ylabel("NLO / LO")
    axr.set_xlim(*plotter.xlim)
    axr.set_ylim(0.1, 2.0)

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

    # regions
    regions = regions_for_era(args.era)
    plotter.regions_to_draw = regions
    plotter.print_regions()

    # variables
    plotter.variables_to_draw = [
        Variable('mass_fourobject',            r'$m_{lljj}$',                   'GeV'),
        Variable('pt_leading_jet',             r'$p_{T}$ leading jet',          'GeV'),
        Variable('pt_dilepton',                r'$p_{T}^{ll}$',                 'GeV'),
        Variable('phi_leading_lepton',         r'$\phi$ leading lepton',        ''),
        Variable('phi_leading_jet',            r'$\phi$ leading jet',           ''),
    ]

    # load YAML
    cfg_path = Path(args.plot_config)
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

            h_nlo = load_one(plotter, DY_NLO_FILE, hist_key)
            h_lo  = load_one(plotter, DY_LO_FILE,  hist_key)

            if any(h is None for h in (h_nlo, h_lo)):
                logging.error(f"{name} missing in LO/NLO; skipping.")
                continue

            fig = plot_overlays(plotter, region, variable, h_nlo, h_lo)

            outpath = (Path(f"/eos/user/w/wijackso/{RUN}/{YEAR}/{plotter.era}")
                       / 'compare_dy'
                       / f"{region.name}_{region.primary_dataset}"
                       / f"{name}_{region.name}.pdf")
            save_figure(fig, outpath)
            print(outpath)
            plt.close(fig)

if __name__ == "__main__":
    main()
