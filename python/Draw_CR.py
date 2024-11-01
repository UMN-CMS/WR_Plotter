from Plotter import SampleGroup, Variable, Region, Systematic
from Plotter import Plotter
import argparse

## Arguments

parser = argparse.ArgumentParser(description='CR plot commands')
parser.add_argument('-y', dest='Year', type=int) ## Year : 2016/2017/2018/-1 (-1=YearCombined)
parser.add_argument('--debug',action='store_true')
parser.add_argument('--ScaleMC', action='store_true')
args = parser.parse_args()

## Enviroment

WORKING_DIR =  '/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter'
dataset = 'Run2UltraLegacy'
ENV_PLOT_PATH = '/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter/plots'

m = Plotter()

m.DoDebug = args.debug

#### In/Out
m.DataYear = args.Year
str_Year = str(args.Year)
m.InputDirectory = WORKING_DIR+'/rootfiles/'+dataset+'/Regions'
m.Filename_prefix = "WRAnalyzer"
m.DataDirectory = str_Year
m.Filename_suffix = ""
m.Filename_skim = "_SkimTree_LRSMHighPt"

m.OutputDirectory = ENV_PLOT_PATH+"/"+dataset+"/CR"+str_Year

#### Category
m.ScaleMC = args.ScaleMC

#### Systematic
m.Systematics = [Systematic(Name="Central", Direction=0, Year=-1)]

#### Binning infos
m.SetBinningFilepath(
  WORKING_DIR+'/data/'+dataset+'/'+str_Year+'/CR_rebins.txt',
  WORKING_DIR+'/data/'+dataset+'/'+str_Year+'/CR_xaxis.txt',
  WORKING_DIR+'/data/'+dataset+'/'+str_Year+'/CR_yaxis.txt',
)

#### Predef samples
from PredefinedSamples import *

###############
#### DY CR ####
###############

#### Define Samples
if args.Year>0:
    exec('m.SampleGroups = [SampleGroup_Others_%s, SampleGroup_NonPrompt_%s, SampleGroup_TT_TW_%s, SampleGroup_DY_%s]'%(args.Year,args.Year,args.Year,args.Year))

#### Signals
#### Print
m.PrintSamples()

#### Define regions
m.RegionsToDraw = [
    ## 60<mll<150
    Region('WR_EE_Resolved_DYCR', 'EGamma', UnblindData=True, Logy=1, TLatexAlias='#splitline{ee}{Resolved DY CR}'),
    Region('WR_MuMu_Resolved_DYCR', 'SingleMuon', UnblindData=True, Logy=1, TLatexAlias='#splitline{#mu#mu}{Resolved DY CR}'),
]
m.PrintRegions()

#### Define Variables
m.VariablesToDraw = [
    Variable('Jet_0_Pt', 'p_{T} of the leading jet', 'GeV'),
    Variable('Jet_1_Pt', 'p_{T} of the subleading jet', 'GeV'),
    Variable('Lepton_0_Eta', '#eta of the leading lepton', ''),
    Variable('Lepton_1_Pt', 'p_{T} of the subleading lepton', 'GeV'),
    Variable('Lepton_1_Eta', '#eta of the subleading lepton', ''),
    Variable('Lepton_1_Pt', 'p_{T} of the subleading lepton', 'GeV'),
    Variable('WRCand_Mass', 'm_{W_{R}} (GeV)', 'GeV'),
]
m.PrintVariables()

#### Draw
m.Draw()
