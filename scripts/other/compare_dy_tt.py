#!/usr/bin/env python3
"""
compare_dy_tt.py

Overlay DYJets and TTbar histograms from the SAME directory (per era),
optionally combining Run3Summer22 and Run3Summer22EE when era="2022".
Applies per-variable plot settings (rebin, xlim, ylim) from YAML, and
saves into EOS under “compare_dy”.

Draws all regions and variables defined via Region and Variable objects.
Default era is 2022.  Ratio panel shows DY / TT.
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
RUN  = "Run3"
YEAR = "2022"

# era → run/year/lumi
ERA_CONFIG = {
    "Run3Summer22":   {"run": "Run3", "year": "2022", "lumi":  7.9804},
    "Run3Summer22EE": {"run": "Run3", "year": "2022", "lumi": 26.6717},
    "2022":           {"run": "Run3", "year": "2022", "lumi":  7.9804 + 26.6717},
}

# Font size constants
FONT_SIZE_TITLE  = 20
FONT_SIZE_LABEL  = 20
FONT_SIZE_LEGEND = 18

# explicit filenames for samples we overlay
SAMPLE_TO_FILE = {
    "DYJets": "WRAnalyzer_DYJets.root",
    "TTbar":  "WRAnalyzer_TTbar.root",
}

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def parse_args():
    p = argparse.ArgumentParser(
        description="Overlay DYJets and TTbar histograms from the same directory"
    )
    p.add_argument(
        "--era", "-e",
        dest="era",
        type=str,
        choices=list(ERA_CONFIG.keys()),
        default="2022",
        help="Era to process (default: 2022). Options: Run3Summer22, Run3Summer22EE, or 2022"
    )
    p.add_argument(
        "--dir",
        dest="subdir",
        type=str,
        default="",
        help="Optional subdirectory inside the era folder (e.g. 'met')"
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
    errs  = np.sqrt(h.variances()) if h.variances() is not None else np.sqrt(np.clip(vals, 0, None))
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers, vals, errs

def load_and_combine_sample(plotter: Plotter, sample: str, hist_key: str):
    """
    For each input_directory (possibly multiple eras when era == '2022'),
    load WRAnalyzer_<sample>.root directly in that directory (no subdirs),
    rebin and scale to the sub-lumi, and sum.
    """
    filename = SAMPLE_TO_FILE[sample]
    combined  = None
    orig_lumi = plotter.lumi
    for indir, sublumi in zip(plotter.input_directory, plotter.input_lumis):
        fp = indir / filename
        try:
            raw = load_histogram(fp, hist_key)
        except Exception as exc:
            logging.warning(f"Missing or unreadable file for {sample}: {fp} ({exc}); continuing.")
            continue
        h = plotter.rebin_hist(raw)
        plotter.lumi = sublumi  # scale each part to its own lumi
        h = plotter.scale_hist(h)
        combined = h if combined is None else combined + h
    plotter.lumi = orig_lumi
    return combined

def plot_overlays(plotter, region: Region, variable: Variable, h_dy, h_tt):
    # extract data
    edges, cen_dy, val_dy, err_dy = extract_hist_data(h_dy)
    _,     cen_tt, val_tt, err_tt = extract_hist_data(h_tt)

    # build figure with two pads
    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={'height_ratios':[3,1], 'hspace':0.10},
    )
    hep.style.use("CMS")

    # --- main overlay on ax ---
    # DYJets
    ax.step(edges, np.append(val_dy, val_dy[-1]), where='post',
            color='C0', label="DYJets")
    ax.errorbar(0.5*(edges[:-1]+edges[1:]), val_dy, yerr=err_dy,
                fmt='none', capsize=2, color='C0')

    # TTbar
    ax.step(edges, np.append(val_tt, val_tt[-1]), where='post',
            color='C1', label=r"$t\bar{t}$")
    ax.errorbar(0.5*(edges[:-1]+edges[1:]), val_tt, yerr=err_tt,
                fmt='none', capsize=2, color='C1')

    xlabel = variable.tlatex_alias + (f" [{variable.unit}]" if variable.unit else "")
    ax.set_ylabel("Events")
    ax.set_yscale("log")
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper right')
    ax.text(0.05, 0.96, region.tlatex_alias, transform=ax.transAxes, fontsize=FONT_SIZE_TITLE, va='top')
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

    # --- ratio panel on axr (DY / TT) ---
    mask = val_tt > 0
    ratio = np.zeros_like(val_tt, dtype=float)
    ratio_err = np.zeros_like(val_tt, dtype=float)
    ratio[mask] = val_dy[mask] / val_tt[mask]
    # propagate uncorrelated errors: R = A/B
    ratio_err[mask] = np.sqrt(
        (err_dy[mask] / val_tt[mask])**2 +
        (val_dy[mask] * err_tt[mask] / (val_tt[mask]**2))**2
    )

    axr.errorbar(cen_dy, ratio, yerr=ratio_err, xerr=True, fmt='o', capsize=2, color='black')
    axr.axhline(1.0, color='black', linestyle='--', linewidth=1)
    axr.set_xlabel(xlabel)
    axr.set_ylabel("DY / TT")
    axr.set_xlim(*plotter.xlim)
    axr.set_ylim(0.5, 1.6)

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
#        Region('wr_mumu_resolved_dy_cr', 'Muon', unblind_data=True,
#               tlatex_alias=f"$\\mu\\mu$\nResolved DY CR\n{args.era}"),
#        Region('wr_ee_resolved_dy_cr', 'EGamma', unblind_data=True,
#               tlatex_alias=f"ee\nResolved DY CR\n{args.era}"),
        Region('wr_mumu_resolved_sr', 'Muon',   unblind_data=True,
               tlatex_alias=f"$\\mu\\mu$\nResolved SR\n{plotter.era}"),
        Region('wr_ee_resolved_sr',   'EGamma', unblind_data=True,
               tlatex_alias=f"ee\nResolved SR\n{plotter.era}"),
    ]

    # define Variables
    plotter.variables_to_draw = [
        Variable('mass_fourobject',            r'$m_{lljj}$',                   'GeV'),
        Variable('met',            r'$E_{T}^{\text{miss}}$',                   'GeV'),
    ]

    if args.era == "2022":
        d1 = WORKING_DIR / 'rootfiles' / plotter.run / plotter.year / 'Run3Summer22'   / args.subdir
        d2 = WORKING_DIR / 'rootfiles' / plotter.run / plotter.year / 'Run3Summer22EE' / args.subdir
        plotter.input_directory = [d1, d2]
        plotter.input_lumis     = [
            ERA_CONFIG["Run3Summer22"]["lumi"],
            ERA_CONFIG["Run3Summer22EE"]["lumi"],
        ]
    else:
        base = WORKING_DIR / 'rootfiles' / plotter.run / plotter.year / args.era / args.subdir
        plotter.input_directory = [base]
        plotter.input_lumis     = [cfg0["lumi"]]

    # load YAML
    cfg_path = Path(f"data/{RUN}/{YEAR}/{YEAR}/dy_compare.yaml")
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

            h_dy = load_and_combine_sample(plotter, "DYJets", hist_key)
            h_tt = load_and_combine_sample(plotter, "TTbar",  hist_key)

            if any(h is None for h in (h_dy, h_tt)):
                logging.error(f"{name}: missing DYJets or TTbar; skipping.")
                continue

            fig = plot_overlays(plotter, region, variable, h_dy, h_tt)

            outpath = (Path(f"/eos/user/w/wijackso/{RUN}/{YEAR}/{plotter.era}")
                       / 'met'
                       / f"{region.name}_{region.primary_dataset}"
                       / f"{name}_{region.name}.pdf")
            save_figure(fig, outpath)
            plt.close(fig)

if __name__ == "__main__":
    main()
