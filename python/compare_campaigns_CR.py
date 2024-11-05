import argparse
import os
from Plotter import SampleGroup, Variable, Region, Plotter
import ROOT 
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib.ticker as mticker
import numpy as np
import mylib
import subprocess
import tempfile
import copy
from pathlib import Path

def load_histogram(file, region_name, variable_name, n_rebin, lum=None):
    """Load and rebin a histogram, with optional luminosity scaling."""
    hist_path = f"{region_name}/{variable_name}_{region_name}"
    hist = file.Get(hist_path)

    if not hist:
        logging.error(f"Histogram '{hist_path}' not found in '{file.GetName()}'")
        return None

    if lum is not None:
        hist.Scale(lum * 1000)

    if variable_name=='WRCand_Mass':
        return mylib.RebinWRMass(hist)
    elif variable_name=='Jet_0_Pt':
        return mylib.RebinJetPt(hist)
    elif n_rebin > 0:
        hist.Rebin(n_rebin)
    return hist

def draw_histogram(run2_hist, run3_hist, ratio_hist, sample_run2, sample_run3,  region, variable, xlim, ylim):
    """Draw comparison and ratio histograms between Run2 and Run3 data."""
    fig, (ax, ax_ratio) = plt.subplots(
            nrows=2, sharex = 'col',
            gridspec_kw= {"height_ratios": [5, 1], "hspace": 0.07},
            figsize=(10, 10)
    )

    run2_content, run2_errors, x_bins = mylib.get_data(run2_hist)
    run3_content, run3_errors, _ = mylib.get_data(run3_hist)
    hep.histplot(run2_content, bins=x_bins, yerr=run2_errors, label=sample_run2.tlatex_alias, color=sample_run2.color, histtype='step', ax=ax,  linewidth=1)
    hep.histplot(run3_content, bins=x_bins, yerr=run3_errors, label=sample_run3.tlatex_alias, color=sample_run3.color, histtype='step', ax=ax, linewidth=1)

    # Plot the ratio
    bin_centers = 0.5 * (x_bins[1:] + x_bins[:-1])
    ratio_contents, ratio_errors, _ = mylib.get_data(ratio_hist)
    nonzero_mask = ratio_contents != 0
    ax_ratio.errorbar(
        bin_centers[nonzero_mask], ratio_contents[nonzero_mask],
        yerr=ratio_errors[nonzero_mask], fmt='o', linewidth=2, capsize=2, color='black'
    )

    # Set ratio limits and labels
    ax_ratio.set_xlim(*xlim)
    ax_ratio.set_xlabel(f"{variable.tlatex_alias} {f'[{variable.unit}]' if variable.unit else f'{variable.unit}'}")
    ax_ratio.set_ylabel(r"$\frac{Run 2}{Run 3}$")
    ax_ratio.set_yticks([0, 0.5, 1, 1.5, 2])
    ax_ratio.set_ylim(0, 2)
    ax_ratio.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2g'))
    ax_ratio.axhline(1.5, color='black', linestyle=':')
    ax_ratio.axhline(1, color='black', linestyle=':')
    ax_ratio.axhline(0.5, color='black', linestyle=':')


    # Set y-axis scale and labels for the main plot
    ax.set_yscale("log" if region.logy > 0 else "linear")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(mylib.custom_log_formatter))
    ax.set_ylim(*ylim)
    bin_widths = [round(x_bins[i + 1] - x_bins[i], 1) for i in range(len(x_bins) - 1)]
    if all(width == bin_widths[0] for width in bin_widths):
        bin_width = bin_widths[0]
        formatted_bin_width = int(bin_width) if bin_width.is_integer() else f"{bin_width:.1f}"
        ax.set_ylabel(f"Events / {formatted_bin_width} {variable.unit}")
    else:
        ax.set_ylabel(f"Events / X {variable.unit}")

    # Plot region information and CMS label
    ax.text(0.05, 0.96, region.tlatex_alias, transform=ax.transAxes,fontsize=20, verticalalignment='top')
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress", fontsize=22)
    ax.legend(fontsize=20)

    # Save and upload plot
    eos_path = f"/eos/user/w/wijackso/plots/compare_campaigns/{sample_run2.mc_campaign}_{sample_run3.mc_campaign}/{sample_run2.name}/{region.name}/{variable.name}_{region.name}.pdf"
    mylib.save_and_upload_plot(fig, eos_path)
    plt.close(fig)

def main():
    plotter = Plotter()

    plotter.sample_groups = [
        SampleGroup('DY', 'Run2UltraLegacy', '2018', '#5790fc', 'Z+Jets (RunIISummer20UL18)'),
        SampleGroup('DY', 'Run3', '2022', '#f89c20', 'Z+Jets (Run3Summer22)')
    ]
    plotter.print_samples()

    plotter.regions_to_draw = [
        Region('WR_EE_Resolved_DYCR', 'EGamma', unblind_data=True, logy=1, tlatex_alias='ee\nResolved DY CR'),
        Region('WR_MuMu_Resolved_DYCR', 'SingleMuon', unblind_data=True, logy=1, tlatex_alias='$\mu\mu$\nResolved DY CR'),
    ]
    plotter.print_regions()

    plotter.variables_to_draw = [
        Variable('WRCand_Mass', r'$m_{lljj}$', 'GeV'),
        Variable('WRCand_Pt', r'$p^{T}_{lljj}$', 'GeV'),
        Variable('NCand_Lepton_0_Mass', r'$m_{l_{Lead}jj}$', 'GeV'),
        Variable('NCand_Lepton_0_Pt', r'$p^{T}_{l_{Lead}jj}$', 'GeV'),
        Variable('NCand_Lepton_1_Mass', r'$m_{l_{Sublead}jj}$', 'GeV'),
        Variable('NCand_Lepton_1_Pt', r'$p^{T}_{l_{Sublead}jj}$', 'GeV'),
        Variable('Lepton_0_Pt', r'$p_{T}$ of the leading lepton', 'GeV'),
        Variable('Lepton_0_Eta', r'$\eta$ of the leading lepton', ''),
        Variable('Lepton_1_Pt', r'$p_{T}$ of the subleading lepton', 'GeV'),
        Variable('Lepton_1_Eta', r'$\eta$ of the subleading lepton', ''),
        Variable('Jet_0_Pt', r'$p_{T}$ of the leading jet', 'GeV'),
        Variable('Jet_1_Pt', r'$p_{T}$ of the subleading jet', 'GeV'),
    ]
    plotter.print_variables()

    rebin_filepath = Path('/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter/data/CompareRuns/CR_rebins.txt') 
    xaxis_filepath = Path('/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter/data/CompareRuns/CR_xaxis.txt')
    yaxis_filepath = Path('/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter/data/CompareRuns/CR_yaxis.txt')

    try:
        plotter.set_binning_filepath(str(rebin_filepath), str(xaxis_filepath), str(yaxis_filepath))
    except FileNotFoundError as e:
        logging.error(f"File path error: {e}")

    for region in plotter.regions_to_draw:
        rebins, xaxis_ranges, yaxis_ranges = plotter.read_binning_info(region.name)
        for variable in plotter.variables_to_draw:
            hists = {}
            n_rebin, xlim, ylim = rebins[variable.name], xaxis_ranges[variable.name], yaxis_ranges[variable.name]
            for sample in plotter.sample_groups:
                file_path = Path(f"/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter/rootfiles/{sample.mc_campaign}/Regions/{sample.year}/WRAnalyzer_SkimTree_LRSMHighPt_DYJets.root")
                if not file_path.exists():
                    logging.warning(f"File {file_path} does not exist.")
                    continue

                with ROOT.TFile.Open(str(file_path)) as file_run:
                    hist = load_histogram(file_run, region.name, variable.name, n_rebin)
                    if hist:
                        hists[sample.year] = copy.deepcopy(hist.Clone(f"{variable.name}_{region.name}_clone"))

            hist_run2, hist_run3 = hists['2018'], hists['2022']
            hist_ratio = mylib.divide_histograms(hist_run2, hist_run3)
            draw_histogram(hist_run2, hist_run3, hist_ratio, plotter.sample_groups[0], plotter.sample_groups[1], region, variable, xlim, ylim)

if __name__ == "__main__":
    main()
