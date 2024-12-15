import os
import ROOT
#import uproot
import numpy as np
import mylib
from array import array
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib.ticker as mticker
import logging
import subprocess
import tempfile
from hist import Hist

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
hep.style.use("CMS")

class SampleGroup:
    def __init__(self, name, mc_campaign, year, color, tlatex_alias, samples=""):
        self.name = name
        self.mc_campaign = mc_campaign
        self.year = year
        self.color = color
        self.tlatex_alias = tlatex_alias
        self.samples = samples

    def print(self):
        logging.info(f'Sample group name = {self.name}')
        logging.info(f'  MC Campaign = {self.mc_campaign}')
        logging.info(f'  Year = {self.year}')
        logging.info(f'  Color = {self.color}')
        logging.info(f'  tLatex alias = {self.tlatex_alias}')

class Variable:
    def __init__(self, name, tlatex_alias, unit):
        self.name = name
        self.tlatex_alias = tlatex_alias
        self.unit = unit

    def print(self):
        logging.info(f"{self.name}, {self.tlatex_alias}, {self.unit}")

class Region:
    def __init__(self, name, primary_dataset, unblind_data=True, logy=-1, tlatex_alias=""):
        self.name = name
        self.primary_dataset = primary_dataset
        self.unblind_data = unblind_data
        self.logy = logy
        self.tlatex_alias = tlatex_alias
        self.draw_data = True
        self.draw_ratio = True

    def print(self):
        logging.info(f"{self.name}, {self.primary_dataset}, unblind_data={self.unblind_data}, logy={self.logy}, {self.tlatex_alias}")

class Plotter:
    def __init__(self):
        self.DoDebug = False
        self.data_year = 2018
        self.data_directory = "2018"
        self.sample_groups = []
        self.regions_to_draw = []
        self.variables_to_draw = []
        self.input_directory = ""
        self.filename_prefix = ""
        self.filename_suffix = ""
        self.filename_skim = ""
        self.output_directory = ""
        self.ScaleMC = False

    def print_border(self):
        logging.info('--------------------------------------------------------------------------')

    def print_samples(self):
        self.print_border()
        logging.info('[Plotter.print_samples()] Printing samples')
        for sample_group in self.sample_groups:
            sample_group.print()
        self.print_border()

    def print_regions(self):
        self.print_border()
        logging.info('[Plotter.print_regions()] Printing regions to be drawn')
        for region in self.regions_to_draw:
            region.print()
            self.print_border()

    def print_variables(self):
        self.print_border()
        logging.info('[Plotter.print_variables()] Printing variables to be drawn')
        for variable in self.variables_to_draw:
            variable.print()
        self.print_border()

    def set_binning_filepath(self, rebin_filepath, xaxis_filepath, yaxis_filepath):
        self.rebin_filepath = rebin_filepath
        self.xaxis_filepath = xaxis_filepath
        self.yaxis_filepath = yaxis_filepath

    def read_binning_info(self, region):
        rebins = {}
        with open(self.rebin_filepath) as f:
            for line in f:
                words = line.split()
                if region != words[0]:
                    continue
                rebins[words[1]] = int(words[2])

        xaxis_ranges = {}
        with open(self.xaxis_filepath) as f:
            for line in f:
                words = line.split()
                if region != words[0]:
                    continue
                xaxis_ranges[words[1]] = [float(words[2]), float(words[3])]

        yaxis_ranges = {}
        with open(self.yaxis_filepath) as f:
            for line in f:
                words = line.split()
                if region != words[0]:
                    continue
                yaxis_ranges[words[1]] = [float(words[2]), float(words[3])]

        return rebins, xaxis_ranges, yaxis_ranges

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

    def get_data(self, hist):
        n_bins = hist.GetNbinsX()
        y_bins, y_errors, x_bins = [], [], []

        for i in range(1, n_bins + 1):
            bin_low_edge = hist.GetBinLowEdge(i)
            bin_up_edge = hist.GetBinLowEdge(i + 1)
            content, error = hist.GetBinContent(i), hist.GetBinError(i)

            y_bins.append(content)
            y_errors.append(error)

            if self.DoDebug: print(f'[{binLowEdge:.5f}, {binUpEdge:.5f}] : {content:.5f} ± {error:.5f}')

            if i == 1 or bin_low_edge != x_bins[-1]:  # Figure out exactly what this does
                x_bins.append(bin_low_edge)

        x_bins.append(hist.GetBinLowEdge(n_bins + 1)) # Add the final upper edge of the last bin
        return np.array(y_bins), np.array(y_errors), np.array(x_bins)


    def divide_histograms(self, data_hist, bkgd_hist):
        ratio_hist = data_hist.Clone("ratio_hist")
        ratio_hist.Divide(bkgd_hist)
        return ratio_hist

    def create_histograms(self):
        for region in self.regions_to_draw:
            logging.info(f'Drawing {region.name}')
            rebins, xaxis_ranges, yaxis_ranges = self.read_binning_info(region.name)
#            data_path = f"{self.input_directory}/{self.data_directory}/{self.filename_prefix}{self.filename_skim}_data_{region.primary_dataset}{self.filename_suffix}.root"
            data_path = f"{self.input_directory}/{self.data_directory}/{self.filename_prefix}{self.filename_skim}_data_{region.primary_dataset}{self.filename_suffix}_test.root"
            if not os.path.exists(data_path):
                logging.error(f"Data file '{data_path}' not found")
                continue
            data_file = ROOT.TFile(data_path)

            for variable in self.variables_to_draw:
                n_rebin = rebins[variable.name]
                xlim = xaxis_ranges[variable.name]
                ylim = yaxis_ranges[variable.name]
                data_hist = self.load_histogram(data_file, region.name, variable.name, n_rebin, xlim)
                if not data_hist:
                    continue

                stack_data, labels, colors = [], [], []
                bkgd_combined_hist = None

                for sample_group in self.sample_groups: # e.g. Nonprompt
                    lumi = mylib.total_lumi(sample_group.year)
                    for sample in sample_group.samples: # e.g. tt_semi, singletop, WJets
                        sample_path = f"{self.input_directory}/{sample_group.year}/{self.filename_prefix}{self.filename_skim}_{sample}{self.filename_suffix}.root"
                        if not os.path.exists(sample_path):
                            logging.warning(f"Sample file '{sample_path}' not found")
                            continue
                        sample_file = ROOT.TFile(sample_path)

                        sample_hist = self.load_histogram(sample_file, region.name, variable.name, n_rebin, xlim, lumi)
                        if not sample_hist:
                            sample_file.Close()
                            continue

                        bkgd_combined_hist = mylib.add_histograms(bkgd_combined_hist, sample_hist)

                        sample_contents, _, _ = self.get_data(sample_hist)

                        stack_data.append(sample_contents)
                        labels.append(sample_group.tlatex_alias)
                        colors.append(sample_group.color)

                        sample_file.Close()

                ratio_hist = mylib.divide_histograms(data_hist, bkgd_combined_hist)

                self.draw_histogram(data_hist, bkgd_combined_hist, ratio_hist, stack_data, lumi, xlim, ylim, labels, colors, region, variable)

    def draw_histogram(self, data_hist, bkgd_combined_hist, ratio_hist, stack_data, lumi, xlims, ylims,  labels, colors, region, variable):
        # Create plot with main and ratio axes
        fig, (ax, ax_ratio) = plt.subplots(
                nrows=2,
                sharex = 'col',
                gridspec_kw= {"height_ratios": [5, 1], "hspace": 0.07},
                figsize=(10, 10)
        )

        # Get combined background data (sum of stack_data)
        background_contents, background_errors, x_bins = self.get_data(bkgd_combined_hist)

        # Plot stacked backgrounds
        hep.histplot(stack_data, bins=x_bins, stack=True, label=labels, color=colors, histtype='fill', ax=ax, edgecolor="none", linewidth=1)

        # Get data
        data_contents, data_errors, x_bins = self.get_data(data_hist)

        # Plot data
        data_label = 'Data'
        hep.histplot(
            data_contents, bins=x_bins, xerr=True,  yerr=data_errors, label=data_label,
            color='black', marker='o', markersize=4, histtype='errorbar', ax=ax
        )

        # Calculate bin centers and extend background for plotting
        bin_centers = 0.5 * (x_bins[1:] + x_bins[:-1])
        background_contents = np.append(background_contents, background_contents[-1])
        background_errors = np.append(background_errors, background_errors[-1])

        # Plot MC uncertainty in the main plot with hatched band
        uncert_label = 'Stat. uncert.'
        ax.fill_between(
                x_bins, 
                background_contents - background_errors, 
                background_contents + background_errors,
                color="none", edgecolor='gray', hatch='////', step='post', label=uncert_label)

        # Get ratio data and apply mask to zero points
        ratio_contents, ratio_errors, _ = self.get_data(ratio_hist)
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

        # Plot the relative MC uncertainty in ratio plot
        relative_background_uncert = np.zeros_like(background_contents)
        nonzero_mask = background_contents != 0
        relative_background_uncert[nonzero_mask] = background_errors[nonzero_mask] / background_contents[nonzero_mask]

        ax_ratio.fill_between(
            x_bins[nonzero_mask],
            1 - relative_background_uncert[nonzero_mask],
            1 + relative_background_uncert[nonzero_mask],
            color="none", edgecolor="gray", hatch="////", step="post"
        )

        # Set axis limits and labels for the ratio plot
        ax_ratio.set_xlim(*xlims)
        ax_ratio.set_xlabel(f"{variable.tlatex_alias} {f'[{variable.unit}]' if variable.unit else f'{variable.unit}'}")
        ax_ratio.set_ylabel(r"$\frac{Data}{Bkg}$")
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
        y_min, y_max = ylims
        ax.set_ylim(*ylims)

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
        hep.cms.label(loc=0, ax=ax, data=True, label="Work in Progress", lumi=lumi)

        # Format legend
        handles, all_labels = ax.get_legend_handles_labels()
        desired_order = [uncert_label, data_label] + labels[::-1]
        label_order_map = {label: index for index, label in enumerate(desired_order)}
        sorted_legend = sorted(zip(handles, all_labels), key=lambda x: label_order_map.get(x[1], float('inf')))
        ordered_handles, ordered_labels = zip(*sorted_legend)

        # Plot legend
        ax.legend(ordered_handles, ordered_labels, fontsize=20)

        # Save and upload plot
        eos_path = f"/eos/user/w/wijackso/{self.output_directory}/{region.name}/{variable.name}_{region.name}_test.pdf"
        self.save_and_upload_plot(fig, eos_path)

    def save_and_upload_plot(self, fig, eos_path):
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
