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

def load_histogram(self, file, region_name, variable_name, n_rebin, xlim, lum=None):
    hist_path = f"{region_name}/{variable_name}_{region_name}"
    hist = file.Get(hist_path)

    if not hist:
        print(f"Error: Histogram '{hist_path}' not found in '{filepath}'")
        file.Close()
        return None

    # FIX
#        hist = mylib.make_overflow_bin(hist, xlim)

    # Scale to the luminosity
    if lum is not None:
        lumi = lum*1000
        hist.Scale(lumi)

    # Rebin mlljj and lead jet pt
    if variable_name=='WRCand_Mass':
        return mylib.RebinWRMass(hist)
    elif variable_name=='Jet_0_Pt':
        return mylib.RebinJetPt(hist)
    elif n_rebin > 0:
        hist.Rebin(n_rebin)
    return hist

def get_data(hist):
    n_bins = hist.GetNbinsX()
    y_bins, y_errors, x_bins = [], [], []

    for i in range(1, n_bins + 1):
        bin_low_edge = hist.GetBinLowEdge(i)
        bin_up_edge = hist.GetBinLowEdge(i + 1)
        content, error = hist.GetBinContent(i), hist.GetBinError(i)

        y_bins.append(content)
        y_errors.append(error)

        if i == 1 or bin_low_edge != x_bins[-1]:  # Figure out exactly what this does
            x_bins.append(bin_low_edge)

    x_bins.append(hist.GetBinLowEdge(n_bins + 1)) # Add the final upper edge of the last bin
    return np.array(y_bins), np.array(y_errors), np.array(x_bins)

            
def draw_histogram(run2_hist, run3_hist, ratio_hist, sample_run2, sample_run3,  region, variable):

    # Create plot with main and ratio axes
    fig, (ax, ax_ratio) = plt.subplots(
            nrows=2,
            sharex = 'col',
            gridspec_kw= {"height_ratios": [5, 1], "hspace": 0.07},
            figsize=(10, 10)
    )

    # Get the Run 2 content and error
    run2_content, run2_errors, x_bins = get_data(run2_hist)

    # Plot Run 2
    hep.histplot(run2_content, bins=x_bins, yerr=run2_errors, label=sample_run2.tlatex_alias, color=sample_run2.color, histtype='step', ax=ax,  linewidth=1)

    # Get the Run 3 content and error
    run3_content, run3_errors, x_bins = get_data(run3_hist)

    # Plot Run 3
    hep.histplot(run3_content, bins=x_bins, yerr=run3_errors, label=sample_run3.tlatex_alias, color=sample_run3.color, histtype='step', ax=ax, linewidth=1)

    # Calculate bin centers
    bin_centers = 0.5 * (x_bins[1:] + x_bins[:-1])

    # Get ratio data and apply mask to zero points
    ratio_contents, ratio_errors, _ = get_data(ratio_hist)
    nonzero_mask = ratio_contents != 0

    # Plot ratio points
    ax_ratio.errorbar(
        bin_centers[nonzero_mask],
        ratio_contents[nonzero_mask],
        yerr=ratio_errors[nonzero_mask],
        fmt='o',
        linewidth=2,
        capsize=2,
        color='black',
    )

    # Set axis limits and labels for the ratio plot
#    ax_ratio.set_xlim(*xlims)
    ax_ratio.set_xlabel(f"{variable.tlatex_alias} {f'[{variable.unit}]' if variable.unit else f'{variable.unit}'}")
    ax_ratio.set_ylabel(r"$\frac{Run 2}{Run 3}$")
    ax_ratio.set_yticks([0, 0.5, 1, 1.5, 2])
    ax_ratio.set_ylim(0, 2)
    ax_ratio.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2g'))
    ax_ratio.axhline(1.5, color='black', linestyle=':')
    ax_ratio.axhline(1, color='black', linestyle=':')
    ax_ratio.axhline(0.5, color='black', linestyle=':')

    # Function for changing the y-axis format
    def custom_log_formatter(y, pos):
        if y == 1:
            return '1'
        elif y == 10:
            return '10'
        else:
            return f"$10^{{{int(np.log10(y))}}}$"

    # Set y-axis scale
    ax.set_yscale("log" if region.logy > 0 else "linear")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_formatter))
#    y_min, y_max = ylims
#    ax.set_ylim(*ylims)

    # Format y-label
    bin_widths = [round(x_bins[i + 1] - x_bins[i], 1) for i in range(len(x_bins) - 1)]
    if all(width == bin_widths[0] for width in bin_widths):
        bin_width = bin_widths[0]
        formatted_bin_width = int(bin_width) if bin_width.is_integer() else f"{bin_width:.1f}"
        ax.set_ylabel(f"Events / {formatted_bin_width} {variable.unit}")
    else:
        ax.set_ylabel(f"Events / X {variable.unit}")

    # Plot region information and CMS label
    ax.text(0.05, 0.96, region.tlatex_alias, transform=ax.transAxes,fontsize=22, verticalalignment='top')
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress")

    # Plot legend
    ax.legend()

    # Save and upload plot
    eos_path = f"/eos/user/w/wijackso/plots/compare_run2_run3/{region.name}/{variable.name}_{region.name}.pdf"
    save_and_upload_plot(fig, eos_path)

def save_and_upload_plot(fig, eos_path):
    # Ensure EOS directory exists
    eos_dir = os.path.dirname(eos_path)
    try:
        subprocess.run(["xrdfs", "eosuser.cern.ch", "mkdir", "-p", eos_dir], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to create directory on EOS: {e}")
        return  # Exit function if directory creation fails

    # Save plot to a temporary PDF file and upload it to EOS
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
        fig.savefig(tmp_file.name, format="pdf")  # Save the plot as a PDF
        tmp_file.flush()  # Ensure all data is written to disk

        # Upload the temporary PDF file to EOS
        try:
            subprocess.run(["xrdcp", "-f", tmp_file.name, f"root://eosuser.cern.ch/{eos_path}"], check=True)
            print(f"File uploaded successfully to EOS at {eos_path}.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to upload file to EOS: {e}")

    # Close the plot figure
    plt.close(fig)

# Paths to the ROOT files
file_run3 = "/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter/rootfiles/Run3/Regions/2022/WRAnalyzer_SkimTree_LRSMHighPt_DYJets.root"
file_run2 = "/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter/rootfiles/Run2UltraLegacy/Regions/2018/WRAnalyzer_SkimTree_LRSMHighPt_DYJets.root"

SampleGroup_DY_2018 = SampleGroup(
    name='DY+Jets', mc_campaign='Run2UltraLegacy', year = '2018', color='#5790fc', tlatex_alias='Z+jets (RunIISummer20UL18)'
)

SampleGroup_DY_2022 = SampleGroup(
    name='DY+Jets', mc_campaign='Run3', year = '2022', color='#f89c20', tlatex_alias='Z+jets (Run3Summer22)'
)

sample_groups = [SampleGroup_DY_2018, SampleGroup_DY_2022]

regions = [
    Region('WR_EE_Resolved_DYCR', 'EGamma', unblind_data=True, logy=1, tlatex_alias='ee\nResolved DY CR'),
    Region('WR_MuMu_Resolved_DYCR', 'SingleMuon', unblind_data=True, logy=1, tlatex_alias='$\mu\mu$\nResolved DY CR'),
]

variables = [
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

# Open the ROOT files
file_run3 = ROOT.TFile.Open(file_run3)
file_run2 = ROOT.TFile.Open(file_run2)

for region in regions:
    print(f"Drawing region {region.name}")
    for variable in variables:
        hists = {}
        print(f"\tDrawing variable {variable.name}")
        for sample in sample_groups:
            file_path = f"/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter/rootfiles/{sample.mc_campaign}/Regions/{sample.year}/WRAnalyzer_SkimTree_LRSMHighPt_DYJets.root"
            hist_name = f"{variable.name}_{region.name}"

            # Open ROOT file if it exists, else skip
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} does not exist.")
                continue
            file_run = ROOT.TFile.Open(file_path)
            if not file_run or file_run.IsZombie():
                print(f"Error: Could not open file {file_path}")
                continue

            # Retrieve and clone histogram if both directory and histogram exist
            directory = file_run.Get(region.name)
            hist = directory.Get(hist_name) if directory else None
            if hist:
                cloned_hist = copy.deepcopy(hist.Clone(f"{hist_name}_clone"))
                hists[sample.year] = cloned_hist
            else:
                print(f"Error: Histogram {hist_name} not found in {file_path}")

            file_run.Close()

        # Perform histogram division if both years' histograms are present
        if '2018' in hists and '2022' in hists:
            try:
                hist_run2, hist_run3 = hists['2018'], hists['2022']
                hist_ratio = mylib.divide_histograms(hist_run2, hist_run3)
            except AttributeError as e:
                print("Error while dividing histograms:", e)
        else:
            print("Error: Expected both Run2 and Run3 histograms but got fewer. Skipping this variable.")

        draw_histogram(hist_run2, hist_run3, hist_ratio, SampleGroup_DY_2018, SampleGroup_DY_2022, region, variable)

