import os
import numpy as np
from array import array
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib.ticker as mticker
import logging
import subprocess
import tempfile
from hist import Hist

# Import generic utilities from src
from src import rebin_histogram, scale_histogram

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
hep.style.use("CMS")

class Plotter:
    def __init__(self):
        # Basic info
        self.run = ""
        self.year = ""
        self.era = ""
        self.input_directory = ""
        self.dir = ""
        self.suffix = ""

        # Samples/Processes, analysis regions, variables
        self.sample_groups = []
        self.regions_to_draw = []
        self.variables_to_draw = []

        # New attributes for accumulating histogram stacks
        self.stack_list = []
        self.stack_colors = []
        self.stack_labels = []
        self.data_hist = []

        # These will be set via configure_axes()
        self.scale = False
        self.input_lumis = []

        # These will be set via configure_axes()
        self.n_rebin = None
        self.xlim = None
        self.ylim = None

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

    def configure_axes(self, nrebin, xlim, ylim):
        """Configure axis and rebinning parameters for plotting."""
        self.n_rebin = nrebin
        self.xlim = xlim
        self.ylim = ylim 

    def reset_stack(self):
        """Reset the histogram stack lists."""
        self.stack_list = []
        self.stack_colors = []
        self.stack_labels = []

    def reset_data(self):
        """Reset the histogram data list."""
        self.data_hist = []

    def store_data(self, dataHist):
        """Accumulate a combined histogram along with its color and label."""
        if dataHist is not None:
            self.data_hist.append(dataHist)
        else:
            logging.warning("Attempted to accumulate a None histogram.")

    def rebin_hist(self, hist_obj, spec=None):
        """
        spec: int (merge‐N) or list of edges [e0,e1,…]
        """
        spec = spec if spec is not None else self.n_rebin
        return rebin_histogram(hist_obj, spec)

    def scale_hist(self, hist):
        scale_factor = 1
        return scale_histogram(hist, scale_factor)

    def accumulate_histogram(self, combined_hist, color, label):
        """Accumulate a combined histogram along with its color and label."""
        if combined_hist is not None:
            self.stack_list.append(combined_hist)
            self.stack_colors.append(color)
            self.stack_labels.append(label)
        else:
            logging.warning("Attempted to accumulate a None histogram.")

 
