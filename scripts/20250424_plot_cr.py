#!/usr/bin/env python3
"""
plot_cr.py

Generate stacked control‐region plots (DY_CR or Flavor_CR) for a given era.
Usage example:
    ./plot_cr.py --era Run3Summer22EE --cat dy_cr --plot-config path/to/plot_settings.yaml
"""

# Standard library
import sys
import logging
from pathlib import Path
import argparse

# Third-party
import yaml
import numpy as np
import hist
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mplhep as hep
from hist.intervals import ratio_uncertainty

# Custom imports
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
from python.plotter import SampleGroup, Variable, Region, Plotter
from python import predefined_samples as ps
from src import custom_log_formatter, save_figure, set_y_label

# Configuration for supported eras
ERA_CONFIG = {
    "RunIISummer20UL18":   {"run": "RunII", "year": "2018", "lumi": 59.832422397},
    "Run3Summer22":        {"run": "Run3",  "year": "2022", "lumi": 7.9804},
    "Run3Summer22EE":      {"run": "Run3",  "year": "2022", "lumi": 26.6717},
    "2022":                {"run": "Run3",  "year": "2022", "lumi": 26.6717+7.9804},
}

# Set of data-like sample groups
DATA_GROUPS = {"EGamma", "SingleMuon", "Muon"}

# Font size constants
FONT_SIZE_TITLE  = 24
FONT_SIZE_LABEL  = 20
FONT_SIZE_LEGEND = 18

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CR plot commands')
    parser.add_argument(
        '--era',
        dest='era',
        type=str,
        choices=list(ERA_CONFIG.keys()),
        required=True, 
        help='Specify the era (e.g. Run3Summer22EE)'
    )
    parser.add_argument(
        '--cat', 
        dest='category', 
        type=str, 
        choices=["dy_cr", "flavor_cr"], 
        required=True, 
        help="Control region: dy_cr or flavor_cr"
    )
    parser.add_argument(
        "--dir", 
        dest="dir", 
        type=str, 
        default="",
        help="Optional subdirectory under the input & default EOS output paths"
    )
    parser.add_argument(
        '--name', 
        type=str, 
        default="",
        help='Append a suffix to the filenames'
    )
    parser.add_argument(
        '--plot-config', 
        dest='plot_config',
        type=str,
        default='data/RunII/2018/RunIISummer20UL18/plot_settings.yaml',
        help='YAML file with rebin/xlim/ylim for each (region,variable)'
    )
    return parser.parse_args()

def setup_logging():
    """
    Configure root logger to INFO level using a simple default format.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def load_plot_settings(config_path: str) -> dict:
    """
    Load YAML configuration for rebin/xlim/ylim.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)

def setup_plotter(args) -> Plotter:
    """
    Create and configure a Plotter instance based on CLI args.
    - Sets run/year/lumi from ERA_CONFIG
    - Builds input_directory and output_directory
    - Defines sample_groups, regions_to_draw, and variables_to_draw
    """
    working_dir = Path('/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter')
    config = ERA_CONFIG[args.era]

    plotter = Plotter()
    plotter.era = args.era
    plotter.run  = config["run"]
    plotter.year = config["year"]
    plotter.lumi = config["lumi"]
    plotter.scale = True

    # Set directories
    if args.dir:
        plotter.input_directory = str(working_dir / 'rootfiles' / plotter.run / plotter.year / plotter.era / args.dir)
        plotter.output_directory = Path(f"/eos/user/w/wijackso/{plotter.run}/{plotter.year}/{plotter.era}/{args.dir}")
    else:
        plotter.input_directory = str(working_dir / 'rootfiles' / plotter.run / plotter.year / plotter.era)
        plotter.output_directory = f"/eos/user/w/wijackso/{plotter.run}/{plotter.year}/{plotter.era}"

    # Define sample group ordering
    base_groups = [
        f'SampleGroup_{args.era}_Other',
        f'SampleGroup_{args.era}_Nonprompt',
        f'SampleGroup_{args.era}_TTbar',
        f'SampleGroup_{args.era}_DY',
        f'SampleGroup_{args.era}_EGamma',
        f'SampleGroup_{args.era}_Muon',
    ]
    if args.category == "dy_cr":
        sample_group_names = base_groups.copy()
    else:  # flavor_cr → swap TTbar and DY
        sample_group_names = base_groups.copy()
        sample_group_names[2], sample_group_names[3] = sample_group_names[3], sample_group_names[2]

    # Turn names into SampleGroup objects
    plotter.sample_groups = []
    for name in sample_group_names:
        try:
            sg = getattr(ps, name)
        except AttributeError:
            logging.error(f"SampleGroup '{name}' not found in predefined_samples.py")
            sys.exit(1)
        plotter.sample_groups.append(sg)

    if args.category == "dy_cr":
        plotter.regions_to_draw = [
            Region(
                'WR_MuMu_Resolved_DYCR', 
                'Muon', 
                unblind_data = True, 
                tlatex_alias='$\mu\mu$\nResolved DY CR\nRun3Summer22EE\nNLO DY, NNLO x-sec'
            ),
            Region(
                'WR_EE_Resolved_DYCR', 
                'EGamma', 
                unblind_data = True, 
                tlatex_alias='ee\nResolved DY CR\nRun3Summer22EE\nNLO DY, NNLO x-sec'
            ),
        ]
    elif args.category == "flavor_cr":
        plotter.regions_to_draw = [
            Region('WR_Resolved_FlavorCR', 'Muon', unblind_data = True, tlatex_alias='$e\mu$\nResolved Flavor CR'),
            Region('WR_Resolved_FlavorCR', 'EGamma', unblind_data = True, tlatex_alias='$e\mu$\nResolved Flavor CR'),
        ]

    plotter.print_samples()
    plotter.print_regions()

    # Define Variables
    plotter.variables_to_draw = [
        Variable('mass_fourobject', r'$m_{lljj}$', 'GeV'),
        Variable('pt_leading_jet', r'$p_{T}$ of the leading jet', 'GeV'),
        Variable('mass_dijet', r'$m_{jj}$', 'GeV'),
        Variable('pt_leading_lepton', r'$p_{T}$ of the leading lepton', 'GeV'),
        Variable('eta_leading_lepton', r'$\eta$ of the leading lepton', ''),
        Variable('phi_leading_lepton', r'$\phi$ of the leading lepton', ''),
        Variable('pt_subleading_lepton', r'$p_{T}$ of the subleading lepton', 'GeV'),
        Variable('eta_subleading_lepton', r'$\eta$ of the subleading lepton', ''),
        Variable('phi_subleading_lepton', r'$\phi$ of the subleading lepton', ''),
        Variable('eta_leading_jet', r'$\eta$ of the leading jet', ''),
        Variable('phi_leading_jet', r'$\phi$ of the leading jet', ''),
        Variable('pt_subleading_jet', r'$p_{T}$ of the subleading jet', 'GeV'),
        Variable('eta_subleading_jet', r'$\eta$ of the subleading jet', ''),
        Variable('phi_subleading_jet', r'$\phi$ of the subleading jet', ''),
        Variable('mass_dilepton', r'$m_{ll}$', 'GeV'),
        Variable('pt_dilepton', r'$p_{T}^{ll}$', 'GeV'),
        Variable('pt_dijet', r'$p_{T}^{jj}$', 'GeV'),
        Variable('mass_threeobject_leadlep', r'$m_{l_{\mathrm{lead}}jj}$', 'GeV'),
        Variable('pt_threeobject_leadlep', r'$p^{T}_{l_{\mathrm{lead}}jj}$', 'GeV'),
        Variable('mass_threeobject_subleadlep', r'$m_{l_{\mathrm{sublead}}jj}$', 'GeV'),
        Variable('pt_threeobject_subleadlep', r'$p^{T}_{l_{\mathrm{sublead}}jj}$', 'GeV'),
        Variable('pt_fourobject', r'$p^{T}_{lljj}$', 'GeV'),
    ]
    plotter.print_variables()

    return plotter

def load_and_rebin(input_dir: Path, sample: str, hist_key: str, plotter: Plotter):
    """
    Try to open WRAnalyzer_{sample}.root from input_dir, extract hist_key,
    rebin and scale it. Returns a rebinned (and scaled) hist or None on failure.
    """
    fp = input_dir / f"WRAnalyzer_{sample}.root"
    try:
        with uproot.open(fp) as f:
            raw_hist = f[hist_key].to_hist()
    except (FileNotFoundError, KeyError) as e:
        logging.warning(f"{e.__class__.__name__} for {fp}: {e}")
        return None

    rebinned = plotter.rebin_hist(raw_hist)
    return rebinned

def reorder_legend(ax, priority_labels=("MC stat. unc.", "Data"), fontsize=FONT_SIZE_LEGEND):
    """
    Reorder the legend so that:
      1) 'MC stat. unc.' appears first
      2) 'Data' appears second
      3) All other backgrounds appear in reverse order
    """
    handles, labels = ax.get_legend_handles_labels()
    idx_map = {label: i for i, label in enumerate(labels)}
    mc_idx = idx_map.get(priority_labels[0], None)
    data_idx = idx_map.get(priority_labels[1], None)

    bkg_idxs = [i for i in range(len(labels)) if i not in {mc_idx, data_idx}]
    bkg_idxs.reverse()

    new_order = []
    if mc_idx is not None:
        new_order.append(mc_idx)
    if data_idx is not None:
        new_order.append(data_idx)
    new_order.extend(bkg_idxs)

    new_handles = [handles[i] for i in new_order]
    new_labels  = [labels[i]  for i in new_order]
    ax.legend(new_handles, new_labels, loc="best", fontsize=fontsize)

def plot_stack(plotter, region, variable):
    """
    Draws:
      - Top pad: stacked MC histograms + data points + MC stat band
      - Bottom pad: Data/MC ratio with Poisson-based uncertainty band
    """
    bkg_stack = plotter.stack_list
    bkg_labels = plotter.stack_labels
    bkg_colors = plotter.stack_colors

    tot = sum(plotter.stack_list)
    data = sum(plotter.data_hist)
    edges = tot.axes[0].edges

    # -- Top pad styling --
    hep.style.use("CMS")
    fig, (ax, rax) = plt.subplots(
        2, 1, 
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), 
        sharex=True
    )

    hep.histplot(
        bkg_stack, 
        stack=True, 
        label=bkg_labels, 
        color=bkg_colors, 
        histtype='fill', 
        ax=ax
    )
    hep.histplot(
        data, 
        label="Data", 
        color='k', 
        histtype='errorbar', 
        ax=ax
    )

    # MC stat uncertainty band on top pad
    errps = {'hatch':'////', 'facecolor':'none', 'lw': 0, 'edgecolor': 'k', 'alpha': 0.5}
    hep.histplot(tot, histtype='band', ax=ax, **errps, label='MC stat. unc.')

    # -- Compute ratio for bottom pad --
    data_vals = data.values()
    tot_vals  = tot.values()
    ratio     = np.zeros_like(data_vals, dtype=float)
    ratio_err = np.zeros_like(data_vals, dtype=float)
    low       = np.zeros_like(data_vals, dtype=float)
    high      = np.zeros_like(data_vals, dtype=float)

    mask = tot_vals > 0
    if np.any(mask):
        ratio[mask]     = data_vals[mask] / tot_vals[mask]
        ratio_err[mask] = np.sqrt(data_vals[mask]) / tot_vals[mask]
        low[mask], high[mask] = ratio_uncertainty(data_vals[mask], tot_vals[mask], "poisson-ratio")

    # -- Bottom pad: data points with error bars --
    hep.histplot(
        ratio,
        edges,
        yerr=ratio_err,
        ax=rax,
        histtype="errorbar",
        color="k",
        capsize=4,
        label="Data",
    )

    # Draw MC uncertainty band only where MC > 0
    band_low  = np.ones_like(data_vals)
    band_high = np.ones_like(data_vals)
    band_low[mask]  = 1 - low[mask]
    band_high[mask] = 1 + high[mask]
    rax.stairs(band_low, edges, baseline=band_high, **errps)

    # -- Axes formatting --
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylim(*plotter.ylim)

    ax.text(
        0.05, 0.96, region.tlatex_alias, 
        transform=ax.transAxes, 
        fontsize=FONT_SIZE_TITLE, 
        verticalalignment='top'
    )
    hep.cms.label(
        loc=0, 
        ax=ax, 
        data=region.unblind_data, 
        label="Work in Progress", 
        lumi=f"{plotter.lumi:.2f}", 
        fontsize=FONT_SIZE_LABEL
    )

    # Ratio pad labeling
    if variable.unit:
        xlabel = f"{variable.tlatex_alias} [{variable.unit}]"
    else:
        xlabel = variable.tlatex_alias
    rax.set_xlabel(xlabel)
    ax.set_xlabel("") # no x-label on top pad
    rax.set_ylabel("Data/Sim.")
    rax.set_ylim(0.7, 1.3)
    rax.set_yticks([0.8, 1.0, 1.2])
    rax.axhline(1, ls='--', color='k')
    ax.set_xlim(*plotter.xlim)

    set_y_label(ax, tot, variable)

    reorder_legend(ax)
    return fig

def main():
    args = parse_args()
    setup_logging()

    # Verify that the YAML config exists
    if not Path(args.plot_config).is_file():
        logging.error(f"Plot‐config YAML not found: {args.plot_config}")
        sys.exit(1)

    # Load YAML configurations
    plot_settings = load_plot_settings(args.plot_config)

    # Build & configure the Plotter
    plotter = setup_plotter(args)
    plotter.plot_settings = plot_settings

    # Validate YAML entries for each region
    missing_regions = [r.name for r in plotter.regions_to_draw if r.name not in plot_settings]
    if missing_regions:
        logging.error(f"Missing YAML entries for regions: {missing_regions}")
        sys.exit(1)

    input_dir = Path(plotter.input_directory)

    # e.g. WR_EE_Resolved_DYCR
    for region in plotter.regions_to_draw:
        logging.info(f"Processing region '{region.name}'")
        # e.g. mass_fourobject
        for variable in plotter.variables_to_draw: # e.g. Lepton_0_Pt
            logging.info(f"  Variable '{variable.name}'")

            # Check YAML entry for this region/variable
            if variable.name not in plot_settings[region.name]:
                logging.error(f"Missing '{variable.name}' under '{region.name}' in YAML")
                continue

            cfg = plotter.plot_settings[region.name][variable.name]
            rebin = cfg['rebin']
            xmin, xmax = map(float, cfg['xlim'])
            ymin, ymax = map(float, cfg['ylim'])

            plotter.configure_axes(nrebin = rebin, xlim = (xmin,  xmax), ylim = (ymin,  ymax),)
            plotter.reset_stack()
            plotter.reset_data()

            hist_key = f"{region.name}/{variable.name}_{region.name}"

            # Loop over each sample group and combine histograms from all samples in that group 
            # E.g. WJets and SingleTop to Nonprompt
            for sample_group in plotter.sample_groups:
                combined = None
                is_data_group = sample_group.name in DATA_GROUPS

                # Loop over each sample in this group and build the sum E.g. WJets and SingleTop to Nonprompt
                for sample in sample_group.samples:
                    hist_obj = load_and_rebin(input_dir, sample, hist_key, plotter)
                    if hist_obj is None:
                        continue

                    # If MC, scale; if data, rebin is enough
                    if not is_data_group and plotter.scale:
                        hist_obj = plotter.scale_hist(hist_obj)

                    combined = hist_obj if combined is None else (combined + hist_obj)

                # If nothing was found, skip
                if combined is None:
                    logging.warning(f"    No histograms found for group '{sample_group.name}'")
                    continue

                if is_data_group:
                    plotter.store_data(combined)
                else:
                    plotter.accumulate_histogram(
                        combined,
                        sample_group.color,
                        sample_group.tlatex_alias
                    )

            # Plot the specific variable in the specific region
            fig = plot_stack(plotter, region, variable)
            outpath = f"{plotter.output_directory}/{region.name}/{variable.name}_{region.name}.pdf"
            try:
                save_figure(fig, outpath)
                logging.info(f"    Saved: {outpath}")
            except Exception as e:
                logging.error(f"    Failed to save {outpath}: {e}")
            finally:
                plt.close(fig)

if __name__ == '__main__':
    main()
