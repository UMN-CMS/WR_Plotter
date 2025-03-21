import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python')))

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

parser = argparse.ArgumentParser(description="Run2 vs Run3 Histogram Comparison Script")
parser.add_argument('--umn', action='store_true', default=False,
                    help="Enable UMN-specific paths and configurations. Default: False")
args = parser.parse_args()

def load_histogram(file, region_name, variable_name, n_rebin, lum=None):
    """Load and rebin a histogram, with optional luminosity scaling."""
    hist_path = f"{region_name}/{variable_name}_{region_name}"
    hist = file.Get(hist_path)

    if not hist:
        logging.error(f"Histogram '{hist_path}' not found in '{file.GetName()}'")
        return None

    if lum is not None:
        hist.Scale(lum * 1000)

    integral = hist.Integral()
    if integral > 0: hist.Scale(1.0 / integral)

    if variable_name=='WRCand_Mass':
        return mylib.RebinWRMass(hist)
    elif variable_name=='Jet_0_Pt':
        return mylib.RebinJetPt(hist)
    elif n_rebin > 0:
        hist.Rebin(n_rebin)
    return hist

def draw_histogram(run2_hist, run3_hist, ratio_hist, sample, process, region, variable, xlim, ylim):
    """Draw comparison and ratio histograms between Run2 and Run3 data."""
    fig, (ax, ax_ratio) = plt.subplots(
            nrows=2, sharex = 'col',
            gridspec_kw= {"height_ratios": [5, 1], "hspace": 0.07},
            figsize=(10, 10)
    )

    run2_content, run2_errors, x_bins = mylib.get_data(run2_hist)
    run3_content, run3_errors, _ = mylib.get_data(run3_hist)

    hep.histplot(run2_content, bins=x_bins, yerr=run2_errors, label=sample.tlatex_alias[0], color=sample.color[0], histtype='step', ax=ax,  linewidth=1)
    hep.histplot(run3_content, bins=x_bins, yerr=run3_errors, label=sample.tlatex_alias[1], color=sample.color[1], histtype='step', ax=ax, linewidth=1)

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
    ax_ratio.set_ylabel(r"$\frac{N800}{N3000}$")
    ax_ratio.set_yticks([0, 1, 2, 3])
    ax_ratio.set_ylim(0, 3)
    ax_ratio.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2g'))
#    ax_ratio.axhline(1.5, color='black', linestyle=':')
#    ax_ratio.axhline(1, color='black', linestyle=':')
#    ax_ratio.axhline(0.5, color='black', linestyle=':')


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
    if not args.umn:
        file_path = f"/eos/user/w/wijackso/plots/N3000_vs_N800/{sample.name}/{process}/{region.name}/{variable.name}_{region.name}.pdf"
    else:
        file_path = f"plots/{process}/{region.name}/{variable.name}_{region.name}.pdf"
    mylib.save_and_upload_plot(fig, file_path, args.umn)
    plt.close(fig)

def main():
    plotter = Plotter()

    plotter.sample_groups = [
        SampleGroup('signals', 'Run2Legacy', ['2018','2018_3jets'], ['#5790fc', '#f89c20'], [r'$(m_{W_{R}}, m_{N}) = (3200\mathrm{~GeV}, 3000\mathrm{~GeV})$', r'$(m_{W_{R}}, m_{N}) = (3200\mathrm{~GeV}, 800\mathrm{~GeV})$'], 'WR3200_N3000',),
#        SampleGroup('DYJets', ['Run2UltraLegacy', 'Run3'], ['2018','2022'], ['#5790fc', '#f89c20'], [r'$DY+Jets$ (Run2 UL18)', r'$DY+Jets$ (Run3 22)'], ['DYJets'],),
    ]
    plotter.print_samples()

    plotter.regions_to_draw = [
        Region('WR_EE_Resolved_SR', 'EGamma', unblind_data=True, logy=1, tlatex_alias='ee\nResolved SR'),
        Region('WR_MuMu_Resolved_SR', 'SingleMuon', unblind_data=True, logy=1, tlatex_alias='$\mu\mu$\nResolved SR'),
    ]

    plotter.print_regions()

    plotter.variables_to_draw = [
        Variable('Lepton_0_Pt', r'$p_{T}$ of the leading lepton', 'GeV'),
        Variable('Lepton_0_Eta', r'$\eta$ of the leading lepton', ''),
        Variable('Lepton_0_Phi', r'$\phi$ of the leading lepton', ''),
        Variable('Lepton_1_Pt', r'$p_{T}$ of the subleading lepton', 'GeV'),
        Variable('Lepton_1_Eta', r'$\eta$ of the subleading lepton', ''),
        Variable('Lepton_1_Phi', r'$\phi$ of the subleading lepton', ''),
        Variable('Jet_0_Pt', r'$p_{T}$ of the leading jet', 'GeV'),
        Variable('Jet_0_Eta', r'$\eta$ of the leading jet', ''),
        Variable('Jet_0_Phi', r'$\phi$ of the leading jet', ''),
        Variable('Jet_1_Pt', r'$p_{T}$ of the subleading jet', 'GeV'),
        Variable('Jet_1_Eta', r'$\eta$ of the subleading jet', ''),
        Variable('Jet_1_Phi', r'$\phi$ of the subleading jet', ''),
        Variable('ZCand_Mass', r'$m_{ll}$', 'GeV'),
        Variable('ZCand_Pt', r'$p^{T}_{ll}$', 'GeV'),
        Variable('Dijet_Mass', r'$m_{jj}$', 'GeV'),
        Variable('Dijet_Pt', r'$p^{T}_{jj}$', 'GeV'),
        Variable('NCand_Lepton_0_Mass', r'$m_{l_{Lead}jj}$', 'GeV'),
        Variable('NCand_Lepton_0_Pt', r'$p^{T}_{l_{Lead}jj}$', 'GeV'),
        Variable('NCand_Lepton_1_Mass', r'$m_{l_{Sublead}jj}$', 'GeV'),
        Variable('NCand_Lepton_1_Pt', r'$p^{T}_{l_{Sublead}jj}$', 'GeV'),
        Variable('WRCand_Mass', r'$m_{lljj}$', 'GeV'),
        Variable('WRCand_Pt', r'$p^{T}_{lljj}$', 'GeV'),
    ]

    plotter.print_variables()

    rebin_filepath = Path('data/241215_N3000_vs_N800/SR_rebins.txt') 
    xaxis_filepath = Path('data/241215_N3000_vs_N800/SR_xaxis.txt')
    yaxis_filepath = Path('data/241215_N3000_vs_N800/SR_yaxis.txt')

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
                for year in sample.year:
                    process=sample.samples
                    file_path = Path(f"rootfiles/{sample.mc_campaign}/Regions/{year}/WRAnalyzer_SkimTree_LRSMHighPt_signal_{process}.root")
                    if not file_path.exists():
                        logging.warning(f"File {file_path} does not exist.")
                        continue
                    with ROOT.TFile.Open(str(file_path)) as file_run:
                        lumi = 59.74
                        hist = load_histogram(file_run, region.name, variable.name, n_rebin, lumi)
                        if hist:
                            hists[year] = copy.deepcopy(hist.Clone(f"{variable.name}_{region.name}_clone"))

                hist_N3000, hist_N800 = hists['WR3200_N3000'], hists['WR3200_N800']
                hist_ratio = mylib.divide_histograms(hist_N800, hist_N3000)
                draw_histogram(hist_N3000, hist_N800, hist_ratio, sample, process, region, variable, xlim, ylim)

if __name__ == "__main__":
    main()
