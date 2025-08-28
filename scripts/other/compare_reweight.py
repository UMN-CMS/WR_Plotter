#!/usr/bin/env python3
"""
compare_reweight.py

Compare DYJets histograms processed by mjj-reweight and p_T-reweight pipelines
for 2018 UL.  Overlays histograms with error bars, applies per-variable plot settings
(rebin, xlim, ylim) from YAML, and saves into EOS under “compare_reweight”.
Draws all regions and variables defined via Region and Variable objects.
Default era is 2018.  Now with ratio panel (p_T / mjj).
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
from python.plotter import Plotter, Variable, Region
from src import save_figure

# where your ROOT files live
WORKING_DIR = Path('/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter')

# EOS output layout
RUN  = "RunII"
YEAR = "2018"

# era → run/year/lumi
ERA_CONFIG = {
    "2018": {"run": "RunII", "year": "2018", "lumi": 59.83},
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
        description="Compare DYJets histograms between reweight pipelines"
    )
    p.add_argument(
        "--era", "-e",
        dest="era",
        type=str,
        choices=list(ERA_CONFIG.keys()),
        default="2018",
        help="Era to process (default: 2018)"
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

def load_and_combine(plotter, hist_subdir, hist_key):
    combined = None
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
        combined = h if combined is None else combined + h
    plotter.lumi = orig_lumi
    return combined

def plot_overlays(plotter, region: Region, variable: Variable, h_pt, h_m):
    # extract data
    edges,   cen_p, val_p, err_p = extract_hist_data(h_pt)
    _,       _,   val_m, err_m = extract_hist_data(h_m)

    # build figure with two pads
    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={'height_ratios':[3,1], 'hspace':0.10},
    )
    hep.style.use("CMS")

    # --- main overlay on ax ---
    for vals, errs, color, label in [
        (val_p, err_p, 'C1', f"DY w/ $p_{{T}}$ reweight"),
        (val_m, err_m, 'C2', f"DY w/ $m_{{jj}}$ reweight"),
    ]:
        ax.step(edges, np.append(vals, vals[-1]), where='post',
                color=color, label=label)
        ax.errorbar(0.5*(edges[:-1]+edges[1:]), vals, yerr=errs,
                    fmt='none', capsize=2, color=color)

    xlabel = variable.tlatex_alias + (f" [{variable.unit}]" if variable.unit else "")
    ax.set_ylabel("Events")
    ax.set_yscale("log")
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right')
    ax.text(0.05, 0.96, region.tlatex_alias,
            transform=ax.transAxes, fontsize=FONT_SIZE_TITLE, va='top')
    hep.cms.label(
        loc=0, ax=ax, data=region.unblind_data,
        label="Work in Progress",
        lumi=f"{plotter.lumi:.1f}", com=13.6,
        fontsize=FONT_SIZE_LABEL
    )
    ax.set_xlim(*plotter.xlim)
    ax.set_ylim(*plotter.ylim)
    ax.tick_params(labelbottom=False)

    # --- ratio panel on axr ---
    mask = val_m > 0
    ratio_pm = np.zeros_like(val_m)
    ratio_pm[mask] = val_p[mask] / val_m[mask]

    err_ratio = np.zeros_like(err_p)
    err_ratio[mask] = np.sqrt((err_p[mask]/val_m[mask])**2 +
                              (val_p[mask]*err_m[mask]/val_m[mask]**2)**2)

    axr.errorbar(cen_p, ratio_pm, yerr=err_ratio, fmt='o',
                 capsize=2, color='black')
    axr.axhline(1, color='black', linestyle='--', linewidth=1)
    axr.set_xlabel(xlabel)
    axr.set_ylabel(f"$p_{{T}} / m_{{jj}}$")
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

    # define Regions
    plotter.regions_to_draw = [
        Region('wr_mumu_resolved_sr', 'Muon', unblind_data=True,
               tlatex_alias=f"$\\mu\\mu$\nResolved SR\n{args.era}"),
        Region('wr_ee_resolved_sr', 'EGamma', unblind_data=True,
               tlatex_alias=f"ee\nResolved SR\n{args.era}"),
    ]

    # define Variables
    plotter.variables_to_draw = [
        Variable('mass_fourobject',            r'$m_{lljj}$',                   'GeV'),
#        Variable('pt_leading_jet',             r'$p_{T}$ leading jet',          'GeV'),
#        Variable('mass_dijet',                 r'$m_{jj}$',                     'GeV'),
#        Variable('pt_leading_lepton',          r'$p_{T}$ leading lepton',       'GeV'),
#        Variable('eta_leading_lepton',         r'$\eta$ leading lepton',        ''),
#        Variable('phi_leading_lepton',         r'$\phi$ leading lepton',        ''),
#        Variable('pt_subleading_lepton',       r'$p_{T}$ subleading lepton',    'GeV'),
#        Variable('eta_subleading_lepton',      r'$\eta$ subleading lepton',     ''),
#        Variable('phi_subleading_lepton',      r'$\phi$ subleading lepton',     ''),
#        Variable('eta_leading_jet',            r'$\eta$ leading jet',           ''),
#        Variable('phi_leading_jet',            r'$\phi$ leading jet',           ''),
#        Variable('pt_subleading_jet',          r'$p_{T}$ subleading jet',       'GeV'),
#        Variable('eta_subleading_jet',         r'$\eta$ subleading jet',        ''),
#        Variable('phi_subleading_jet',         r'$\phi$ subleading jet',        ''),
#        Variable('mass_dilepton',              r'$m_{ll}$',                     'GeV'),
#        Variable('pt_dilepton',                r'$p_{T}^{ll}$',                 'GeV'),
#        Variable('pt_dijet',                   r'$p_{T}^{jj}$',                 'GeV'),
#        Variable('mass_threeobject_leadlep',   r'$m_{l_{\mathrm{pri}}jj}$',     'GeV'),
#        Variable('pt_threeobject_leadlep',     r'$p^{T}_{l_{\mathrm{pri}}jj}$', 'GeV'),
#        Variable('mass_threeobject_subleadlep',r'$m_{l_{\mathrm{sec}}jj}$',     'GeV'),
#        Variable('pt_threeobject_subleadlep',  r'$p^{T}_{l_{\mathrm{sec}}jj}$', 'GeV'),
#        Variable('pt_fourobject',              r'$p^{T}_{lljj}$',               'GeV'),
    ]

    # single input dir for 2018 UL
    base = WORKING_DIR / 'rootfiles' / plotter.run / plotter.year / 'RunIISummer20UL18'
    plotter.input_directory = [base]
    plotter.input_lumis      = [cfg0["lumi"]]

    # load YAML
    cfg_path = Path(f"data/{RUN}/{YEAR}/RunIISummer20UL18/compare_reweight.yaml")
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
            h_pt = load_and_combine(plotter, 'dy_ht_leadjet_rewght',      hist_key)
            h_m  = load_and_combine(plotter, 'dy_ht_lo_dijetmass_rewght', hist_key)

            if any(h is None for h in (h_pt, h_m)):
                logging.error(f"{name} missing; skipping.")
                continue

            fig = plot_overlays(plotter, region, variable, h_pt, h_m)

            outpath = (
                Path(f"/eos/user/w/wijackso/{RUN}/{YEAR}/{plotter.era}")
                / 'compare_reweight'
                / f"{region.name}_{region.primary_dataset}"
                / f"{name}_{region.name}.pdf"
            )
            save_figure(fig, outpath)
            plt.close(fig)

if __name__ == "__main__":
    main()
