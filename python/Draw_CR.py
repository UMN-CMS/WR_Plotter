import argparse
import os
from Plotter import SampleGroup, Variable, Region, Plotter
from PredefinedSamples import *

# Define constants
WORKING_DIR = '/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter'
DATASET = 'Run2UltraLegacy'
ENV_PLOT_PATH = os.path.join(WORKING_DIR, 'output')
BINNING_FILE_TEMPLATE = os.path.join(WORKING_DIR, 'data', DATASET, '{year}', 'CR_rebins.txt')
XAXIS_FILE_TEMPLATE = os.path.join(WORKING_DIR, 'data', DATASET, '{year}', 'CR_xaxis.txt')
YAXIS_FILE_TEMPLATE = os.path.join(WORKING_DIR, 'data', DATASET, '{year}', 'CR_yaxis.txt')

# Argument parsing
parser = argparse.ArgumentParser(description='CR plot commands')
parser.add_argument('-y', dest='Year', type=int, required=True, help='Specify the data year')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--ScaleMC', action='store_true', help='Enable MC scaling')
args = parser.parse_args()

# Validate Year
if not (2016 <= args.Year <= 2022):
    raise ValueError("Invalid Year: Please specify a year between 2016 and 2022.")

# Initialize Plotter
plotter = Plotter()
plotter.DoDebug = args.debug
plotter.data_year = args.Year
plotter.ScaleMC = args.ScaleMC

# Set directories and file paths
year_str = str(args.Year)
plotter.input_directory = os.path.join(WORKING_DIR, 'rootfiles', DATASET, 'Regions')
plotter.filename_prefix = "WRAnalyzer"
plotter.data_directory = year_str
plotter.output_directory = os.path.join('plots', DATASET, 'CR', year_str)
plotter.filename_suffix = ""
plotter.filename_skim = "_SkimTree_LRSMHighPt"

# Set binning file paths with error handling
try:
    plotter.set_binning_filepath(
        BINNING_FILE_TEMPLATE.format(year=year_str),
        XAXIS_FILE_TEMPLATE.format(year=year_str),
        YAXIS_FILE_TEMPLATE.format(year=year_str),
    )
except FileNotFoundError as e:
    raise FileNotFoundError(f"File path error: {e}")

# Define Sample Groups
if args.Year > 0:
    plotter.sample_groups = [
        eval(f'SampleGroup_Others_{args.Year}'),
        eval(f'SampleGroup_NonPrompt_{args.Year}'),
        eval(f'SampleGroup_TT_TW_{args.Year}'),
        eval(f'SampleGroup_DY_{args.Year}')
    ]

plotter.print_samples()

# Define Regions
plotter.regions_to_draw = [
    Region('WR_EE_Resolved_DYCR', 'EGamma', unblind_data=True, logy=1, tlatex_alias='ee\nResolved DY CR'),
    Region('WR_MuMu_Resolved_DYCR', 'SingleMuon', unblind_data=True, logy=1, tlatex_alias='$\mu\mu$\nResolved DY CR'),
]
plotter.print_regions()

# Define Variables (removed duplicates)
plotter.variables_to_draw = [
    Variable('Jet_0_Pt', r'$p_{T}$ of the leading jet', 'GeV'),
    Variable('Jet_1_Pt', r'$p_{T}$ of the subleading jet', 'GeV'),
    Variable('Lepton_0_Pt', r'$p_{T}$ of the leading lepton', 'GeV'),
    Variable('Lepton_0_Eta', r'$\eta$ of the leading lepton', ''),
    Variable('Lepton_1_Eta', r'$\eta$ of the subleading lepton', ''),
    Variable('Lepton_1_Pt', r'$p_{T}$ of the subleading lepton', 'GeV'),
    Variable('WRCand_Mass', r'$m_{lljj}$', 'GeV'),
]
plotter.print_variables()

# Draw Plots
plotter.create_histograms()
