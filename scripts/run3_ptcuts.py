#!/usr/bin/env python3

import sys
from pathlib import Path

# Get the repository root (assuming scripts/ is one level below repo root)
repo_root = Path(__file__).resolve().parent.parent

# Add the repository root so that "python" and "src" are both available as packages
sys.path.insert(0, str(repo_root))

# Now you can import modules from python and src packages
from python.plotter import SampleGroup, Variable, Region, Plotter
from python import predefined_samples as ps

import argparse
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hist.intervals import ratio_uncertainty
import mplhep as hep
import numpy as np
import uproot
import hist
import matplotlib.ticker as mticker

# Import custom formatter from src
from src import custom_log_formatter, save_figure, set_y_label

def parse_args():
    parser = argparse.ArgumentParser(description='CR plot commands')
    parser.add_argument('--era', dest='era', type=str,
                        choices=["RunIISummer20UL18", "Run3Summer22"],
                        required=True, help='Specify the era (e.g. RunIISumer20UL18)')
    parser.add_argument('--dir', type=str, default="",
                        help='Save histograms to a new directory')
    parser.add_argument('--name', type=str, default="",
                        help='Append a suffix to the filenames')
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(level=logging.INFO)

def setup_plotter(args) -> Plotter:
    working_dir = Path('/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter')

    era_mapping = {
            "RunIISummer20UL18": {"run": "RunII", "year": "2018", "lumi": 59.832422397},
            "Run3Summer22": {"run": "Run3", "year": "2022", "lumi":  7.9804},
    }
    mapping = era_mapping.get(args.era)
    if mapping is None:
        raise ValueError(f"Unsupported era: {args.era}")

    plotter = Plotter()

    plotter.run  = mapping["run"] 
    plotter.year = mapping["year"]
    plotter.era = args.era

    plotter.scale = True
    plotter.lumi = mapping["lumi"]

    # File templates for binning, xaxis, and yaxis files
    binning_file = working_dir / 'data' / plotter.run / plotter.year / plotter.era / 'ZPeak_rebins.txt'
    xaxis_file = working_dir / 'data' / plotter.run / plotter.year / plotter.era / 'ZPeak_xaxis.txt'
    yaxis_file = working_dir / 'data' / plotter.run / plotter.year / plotter.era / 'ZPeak_yaxis.txt'
     
    # Set directories
    if args.dir:
        plotter.input_directory = str(working_dir / 'rootfiles' / plotter.run / plotter.year / plotter.era / args.dir)
        plotter.output_directory = Path(f"/eos/user/w/wijackso/{plotter.run}/{plotter.year}/{plotter.era}/{args.dir}")
    else:
        plotter.input_directory = str(working_dir / 'rootfiles' / plotter.run / plotter.year / plotter.era)
        plotter.output_directory = f"/eos/user/w/wijackso/{plotter.run}/{plotter.year}/{plotter.era}"

    if args.name:
        plotter.suffix = args.name

    try:
        plotter.set_binning_filepath(str(binning_file), str(xaxis_file), str(yaxis_file))
    except FileNotFoundError as e:
        logging.error(f"File path error: {e}")
        raise

    sample_group_names = [
        f'SampleGroup_{args.era}_DY',
        f'SampleGroup_{args.era}_EGamma',
        f'SampleGroup_{args.era}_Muon',
    ]
    plotter.regions_to_draw = [
        Region('MuMu_DYCR_PT20',    'Muon', unblind_data = True, tlatex_alias='$\mu\mu$\n$p_{T}^{\ell} > 20\mathrm{~GeV}$'),
        Region('MuMu_DYCR_PT30', 'Muon', unblind_data = True, tlatex_alias='$\mu\mu$\n$p_{T}^{\ell} > 30\mathrm{~GeV}$'),
        Region('MuMu_DYCR_PT40', 'Muon', unblind_data = True, tlatex_alias='$\mu\mu$\n$p_{T}^{\ell} > 40\mathrm{~GeV}$'),
        Region('MuMu_DYCR_PT50', 'Muon', unblind_data = True, tlatex_alias='$\mu\mu$\n$p_{T}^{\ell} > 50\mathrm{~GeV}$'),
        Region('MuMu_DYCR_PT60', 'Muon', unblind_data = True, tlatex_alias='$\mu\mu$\n$p_{T}^{\ell} > 60\mathrm{~GeV}$'),
        Region('EE_DYCR_PT20',    'EGamma', unblind_data = True, tlatex_alias='ee\n$p_{T}^{\ell} > 20\mathrm{~GeV}$'),
        Region('EE_DYCR_PT30', 'EGamma', unblind_data = True, tlatex_alias='ee\n$p_{T}^{\ell} > 30\mathrm{~GeV}$'),
        Region('EE_DYCR_PT40', 'EGamma', unblind_data = True, tlatex_alias='ee\n$p_{T}^{\ell} > 40\mathrm{~GeV}$'),
        Region('EE_DYCR_PT50', 'EGamma', unblind_data = True, tlatex_alias='ee\n$p_{T}^{\ell} > 50\mathrm{~GeV}$'),
        Region('EE_DYCR_PT60', 'EGamma', unblind_data = True, tlatex_alias='ee\n$p_{T}^{\ell} > 60\mathrm{~GeV}$'),
    ]

    plotter.sample_groups = [getattr(ps, name) for name in sample_group_names]

    # Print for debugging purposes
    plotter.print_samples()

    plotter.print_regions()

    # Define Variables
    plotter.variables_to_draw = [
        Variable('ZCand_Mass', r'$m_{ll}$', 'GeV'),
        Variable('Lepton_0_Pt', r'$p_{T}$ of the leading lepton', 'GeV'),
        Variable('Lepton_1_Pt', r'$p_{T}$ of the subleading lepton', 'GeV'),
    ]
    plotter.print_variables()

    return plotter

def plot_stack(plotter, region, variable):

    bkg_stack = plotter.stack_list
    bkg_labels = plotter.stack_labels
    bkg_colors = plotter.stack_colors

    tot = sum(plotter.stack_list)
    data = sum(plotter.data_hist)

    # Styling
    hep.style.use("CMS")
    fig, (ax, rax) = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), sharex=True)

    # Main Plot
    hep.histplot(bkg_stack, stack=True, label=bkg_labels, color=bkg_colors, histtype='fill', ax=ax)
    hep.histplot(data, label="Data", color='k', histtype='errorbar', ax=ax)

    # MC Stat uncert. band
    errps = {'hatch':'////', 'facecolor':'none', 'lw': 0, 'edgecolor': 'k', 'alpha': 0.5}
    hep.histplot(tot, histtype='band', ax=ax, **errps, label='MC stat. unc.')

    # Ratio Plot
    num = data.values()
    tot_vals = np.maximum(tot.values(), 0) # If bin is negative, set it = 0

    # Use tot_vals for plotting the ratio and error bars
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = data.values() / tot_vals
        ratio_err = np.sqrt(data.values()) / tot_vals

    hep.histplot(ratio, tot.axes[0].edges, yerr=ratio_err, ax=rax, histtype='errorbar', color='k', capsize=4, label="Data")

    yerr = ratio_uncertainty(data.values(), tot_vals, 'poisson-ratio')
    rax.stairs(1 + yerr[1], edges=tot.axes[0].edges, baseline=1 - yerr[0], **errps)

    # Set axis limits and labels for the ratio plot
    ax.set_xlim(*plotter.xlim)
    rax.set_xlabel(f"{variable.tlatex_alias} {f'[{variable.unit}]' if variable.unit else f'{variable.unit}'}")
    ax.set_xlabel("")

    rax.set_ylim(0.3, 1.7)
    rax.axhline(1, ls='--', color='k')
    rax.set_ylabel("Data/Sim.")

    # Set y-axis scale
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
    ax.set_ylim(*plotter.ylim)

    # Plot region information and CMS label
    ax.text(0.05, 0.96, region.tlatex_alias, transform=ax.transAxes, fontsize=22, verticalalignment='top')
    hep.cms.label(loc=0, ax=ax, data=region.unblind_data, label="Work in Progress", lumi=f"{plotter.lumi:.2f}", fontsize=20)

    set_y_label(ax, tot, variable)

    # Add the legend
    handles, labels = ax.get_legend_handles_labels()
    # Reorder: MC stat. unc. first, Data second, then reverse order of backgrounds
    mc_unc_idx = labels.index("MC stat. unc.")
    data_idx = labels.index("Data")
    # Reverse background order
    bkg_indices = [i for i in range(len(labels)) if i not in [mc_unc_idx, data_idx]]
    bkg_indices.reverse()
    # New order: MC stat. unc. → Data → Reversed backgrounds
    new_order = [mc_unc_idx, data_idx] + bkg_indices
    handles = [handles[i] for i in new_order]
    labels = [labels[i] for i in new_order]

    ax.legend(handles, labels, loc="best", fontsize=20)

    return fig

def main():
    args = parse_args()
    setup_logging()
    plotter = setup_plotter(args)
    for region in plotter.regions_to_draw:
        logging.info(f"Processing region '{region.name}'") # e.g. WR_EE_Resolved_DYCR
        rebins, xaxis_ranges, yaxis_ranges = plotter.read_binning_info(region.name)
        for variable in plotter.variables_to_draw: # e.g. Lepton_0_Pt
            logging.info(f"Processing variable '{variable.name}' in region '{region.name}'")

            hist_key = f"{region.name}/{variable.name}_{region.name}"

            # Configure axes using binning information
            plotter.configure_axes(rebins[variable.name], xaxis_ranges[variable.name], yaxis_ranges[variable.name])
            plotter.reset_stack()
            plotter.reset_data()

            # Loop over each sample group and combine histograms from all samples in that group (e.g. WJets and SingleTop to Nonprompt)
            for sample_group in plotter.sample_groups: # e.g. tt+tW
                combined_hist = None
                combined_data_hist = None
                if sample_group.name in ("EGamma", "SingleMuon", "Muon"):
                    if sample_group.name == region.primary_dataset:
                        for sample in sample_group.samples:
                            # TO-DO: Make this flexible with argparse
                            file_path = Path(plotter.input_directory) / f"WRAnalyzer_{sample}.root"
                            logging.info(f"Reading histogram from: {file_path}")
                            try:
                                with uproot.open(file_path) as file:
                                    hist_obj = file[hist_key].to_hist()
                            except Exception as e:
                                logging.error(f"Failed to read histogram from {file_path}: {e}")
                                continue

                            rebinned_data_hist = plotter.rebin_hist(hist_obj)

                            combined_data_hist = rebinned_data_hist.copy() if combined_data_hist is None else combined_data_hist + rebinned_data_hist

                        plotter.store_data(combined_data_hist)

                else:
                    for sample in sample_group.samples: # e.g. TTbar, tW,
                        file_path = Path(plotter.input_directory) / f"WRAnalyzer_{sample}.root"
                        logging.info(f"Reading histogram from: {file_path}")
                        try:
                            with uproot.open(file_path) as file:
                                hist_obj = file[hist_key].to_hist()
                        except Exception as e:
                            logging.error(f"Failed to read histogram from {file_path}: {e}")
                            continue

                        # Rebin every histogram before adding them
                        rebinned_hist = plotter.rebin_hist(hist_obj)

                        # Scale every histogram to the luminosit
                        if plotter.scale:
                            rebinned_hist = plotter.scale_hist(rebinned_hist)

                        combined_hist = rebinned_hist.copy() if combined_hist is None else combined_hist + rebinned_hist

                    # Append the tt+tW combined hit, the tt+tW sample group color, and the latex label
                    plotter.accumulate_histogram(combined_hist, sample_group.color, sample_group.tlatex_alias)

            # Plot the specific variable in the specific region
            figure = plot_stack(plotter, region, variable)
            save_figure(figure, f"{plotter.output_directory}/{region.name}/{variable.name}_{region.name}.pdf")
            plt.close(figure)

if __name__ == '__main__':
    main()
