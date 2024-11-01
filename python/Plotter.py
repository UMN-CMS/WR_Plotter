import os, ROOT
import numpy as np
import mylib
from array import array
import matplotlib.pyplot as plt
import mplhep as hep

## SampleGroup ##
class SampleGroup:
  def __init__(self, Name, Type, Samples, Year, Color=0, Style=1, TLatexAlias="", LatexAlias=""):
    self.Name = Name
    self.Type = Type
    self.Samples = Samples
    self.Year = Year
    self.Color = Color
    self.Style = Style
    self.TLatexAlias = TLatexAlias
    self.LatexAlias = LatexAlias

  def Print(self):
    print('Sample group name = '+self.Name)
    print('  Type = '+self.Type)
    print('  Samples = ')
    print(self.Samples)
    print( '  Year = '+str(self.Year))
    print( '  Color = '+str(self.Color))
    print( '  TLatexAlias = '+str(self.TLatexAlias))
    print( '  LatexAlias = '+str(self.LatexAlias))

## Variable ##
class Variable:
    def __init__(self, Name, TLatexAlias, Unit):
        self.Name = Name
        self.TLatexAlias = TLatexAlias
        self.Unit = Unit
    def Print(self):
        print(f"{self.Name}, {self.TLatexAlias}, {self.Unit}")

## Region ##
class Region:
    def __init__(self, Name, PrimaryDataset, UnblindData=True, Logy=-1, TLatexAlias=""):
        self.Name = Name
        self.PrimaryDataset = PrimaryDataset
        self.UnblindData = UnblindData
        self.Logy = Logy
        self.TLatexAlias = TLatexAlias

        self.DrawData = True
        self.DrawRatio = True

    def Print(self):
        print(f"{self.Name}, {self.PrimaryDataset}, UnblindData={self.UnblindData}, Logy={self.Logy}, {self.TLatexAlias}")

## Systematic ##
class Systematic:
  def __init__(self, Name, Direction, Year):
    self.Name = Name
    self.Direction = Direction
    self.Year = Year ## if <0, it's correalted
  def FullName(self):
    if self.Year>0:
      return 'Run'+str(self.Year)+'_'+self.Name
    else:
      return self.Name
  def Print(self):
    str_Direction = 'Up' if self.Direction>0 else 'Down'
    if self.Direction==0:
      str_Direction = "Central"
    print('(%s, %s, %d)'%(self.Name, str_Direction, self.Year))

## Plotter ##
class Plotter:

    def __init__(self):

        self.DoDebug = False

        self.DataYear = 2016
        self.DataDirectory = "2016"

        self.SampleGroups = []
        self.RegionsToDraw = []
        self.VariablesToDraw = []
        self.SignalsToDraw = []

        self.Systematics = []
        self.InputDirectory = ""
        self.Filename_prefix = ""
        self.Filename_suffix = ""
        self.Filename_skim = ""
        self.OutputDirectory = ""

        self.ScaleMC = False

        self.ExtraLines = ""

        self.ErrorFromShape = False
        self.AddErrorLinear = False

        self.NoErrorBand = False

        self.FixedBinWidth = -1 # TeV

    def PrintBorder(self):
        print('--------------------------------------------------------------------------')

    def PrintSamples(self):
        self.PrintBorder()
        print('[Plotter.PrintSamples()] Printing samples')
        for s in self.SampleGroups:
            s.Print()
        self.PrintBorder()

    def PrintRegions(self):
        self.PrintBorder()
        print('[Plotter.PrintRegions()] Printing regions to be drawn')
        for s in self.RegionsToDraw:
            s.Print()
            self.PrintBorder()

    def PrintVariables(self):
        self.PrintBorder()
        print('[Plotter.PrintVariables()] Printing variables to be drawn')
        for s in self.VariablesToDraw:
            s.Print()
        self.PrintBorder()

    ## Binning related

    def SetBinningFilepath(self, RebinFilepath, XaxisFilepath, YaxisFilepath):
        self.RebinFilepath = RebinFilepath
        self.XaxisFilepath = XaxisFilepath
        self.YaxisFilepath = YaxisFilepath

    def ReadBinningInfo(self, Region):
        ## Rebin
        Rebins = dict()
        for line in open(self.RebinFilepath).readlines():
            words = line.split()
            if Region!=words[0]:
                continue
            Rebins[words[1]] = int(words[2])
        ## xaxis
        XaxisRanges = dict()
        for line in open(self.XaxisFilepath).readlines():
            words = line.split()
            if Region!=words[0]:
                continue
            XaxisRanges[words[1]] = [float(words[2]), float(words[3])]
        return Rebins, XaxisRanges


    def load_histogram(self, file, region_name, variable_name, nRebin):

        hist_path = f"{region_name}/{variable_name}_{region_name}"
        hist = file.Get(hist_path)
        
        if not hist:
            print(f"Error: Histogram '{hist_path}' not found in '{filepath}'")
            file.Close()
            return None

        if variable_name=='WRCand_Mass':
            return mylib.RebinWRMass(hist, region, self.DataYear)
        elif variable_name=='ToBeCorrected_Jet_Pt':
            return mylib.RebinJetPt(hist, region, self.DataYear)
        else:
            if nRebin>0:
                hist.Rebin(nRebin)
                return hist
            else:
                return hist

    def get_data(self, hist):
        """
        Extracts bin contents, errors, and edges from a ROOT histogram using GetBinContent and GetBinError.

        Parameters:
        - hist: The ROOT histogram (TH1 object).

        Returns:
        - yBins: Array of bin contents.
        - yErrors: Array of bin errors.
        - xBins: Array of bin edges.
        """
        nBins = hist.GetNbinsX()  # Total number of bins along the x-axis

        # Initialize lists to collect bin contents, errors, and edges
        yBins = []
        yErrors = []
        xBins = []

        # Loop through the bins to collect data
        for i in range(1, nBins + 1):  # ROOT bins are 1-indexed
            binLowEdge = hist.GetBinLowEdge(i)
            binUpEdge = hist.GetBinLowEdge(i + 1)

            # Collect bin content and error
            content = hist.GetBinContent(i)
            error = hist.GetBinError(i)

            # Append data to lists
            yBins.append(content)
            yErrors.append(error)

            if self.DoDebug: print(f'[{binLowEdge:.5f}, {binUpEdge:.5f}] : {content:.5f} ± {error:.5f}')

            # Append the bin lower edge; we'll add the last upper edge after the loop
            if i == 1 or binLowEdge != xBins[-1]:  # Avoid duplicates in edges
                xBins.append(binLowEdge)

        # Add the final upper edge of the last bin
        xBins.append(hist.GetBinLowEdge(nBins + 1))

        # Convert lists to numpy arrays for consistency
        return np.array(yBins), np.array(yErrors), np.array(xBins)

    def Draw(self):
        for Region in self.RegionsToDraw:
            print('# Drawing '+Region.Name)
            Rebins, XaxisRanges = self.ReadBinningInfo(Region.Name)

            Indir = self.InputDirectory
            Outdir = self.OutputDirectory+'/'+Region.Name+'/'
            os.system('mkdir -p '+Outdir)

            ## Data file
            data_path = f"{Indir}/{self.DataDirectory}/{self.Filename_prefix}{self.Filename_skim}_data_{Region.PrimaryDataset}{self.Filename_suffix}.root"
            f_Data = ROOT.TFile(data_path)

            # Loop over variables
            for Variable in self.VariablesToDraw:
                if self.DoDebug: print(f"[DEBUG] Trying to draw variable = {Variable.Name}")
                print(f"[DEBUG] Trying to draw variable = {Variable.Name}")
                nRebin = Rebins[Variable.Name]
                xMin, xMax = XaxisRanges[Variable.Name]
                xtitle = Variable.TLatexAlias
                HistsToDraw = {}

                # Get data first
                if self.DoDebug: print(f"[DEBUG] Trying to get data histogram {self.Filename_prefix}{self.Filename_skim}_data_{Region.PrimaryDataset}{self.Filename_suffix}.root")

                h_Data = self.load_histogram(f_Data, Region.Name, Variable.Name, nRebin)
                y_values, y_errs, x_bins = self.get_data(h_Data) 

                # Prepare background stack
                stack_data = []
                labels = []
                colors = []
                total_bkgd_yValues = np.zeros_like(y_values)

                # Prepare background stack
                stack_Bkgd = ROOT.THStack("stack_Bkgd", "")
                h_Bkgd = 0

                for Syst in self.Systematics:
                    if self.DoDebug: print(f"[DEBUG] Trying to make a histogram for Syst = {Syst.Print()}")
                    print(f"[DEBUG] Trying to make a histogram for Syst = {Syst.Print()}")
                    dirName = Region.Name
                    for SampleGroup in self.SampleGroups:
                        Color = SampleGroup.Color
                        for Sample in SampleGroup.Samples:
                            if self.DoDebug: print(f"[DEBUG] Trying to make histogram for Sample = {Sample}")
                            print(f"[DEBUG] Trying to make histogram for Sample = {Sample}")
                            sample_path = f"{Indir}/{SampleGroup.Year}/{self.Filename_prefix}{self.Filename_skim}_{Sample}{self.Filename_suffix}.root"
                            f_Sample = ROOT.TFile(sample_path)
                            h_Sample = self.load_histogram(f_Sample, Region.Name, Variable.Name, nRebin)

                            sample_yValues, sample_yErrors, _ = self.get_data(h_Sample)
                            sample_yValues = sample_yValues * 59.74*1000
                            sample_yErrors = sample_yErrors * 59.74*1000

                            total_bkgd_yValues += sample_yValues

                            stack_data.append(sample_yValues)
                            labels.append(SampleGroup.TLatexAlias)
                            colors.append(Color)

                            ## AddError option
                            AddErrorOption = ''

                            ## If central, add to h_Bkgd
                            stack_Bkgd.Add(h_Sample)
                            if not h_Bkgd:
                                h_Bkgd = h_Sample.Clone()
                            else:
                                h_Bkgd = mylib.AddHistograms(h_Bkgd, h_Sample, AddErrorOption)

                            HistsToDraw[Sample] = h_Sample.Clone()

                            ## Close file
                            f_Sample.Close()

#                combined_yValues, combined_yErrors, _ = self.get_data(h_Bkgd)

                hep.style.use("CMS")
                fig, ax = plt.subplots()

                hep.cms.label(
                    loc=0,
                    ax=ax,
                    data=True,
                    label="Work in Progress",
                    lumi=59.74,
                )

                hep.histplot(y_values, bins=x_bins, xerr=True,  yerr=y_errs, label='Data', color='black', marker='o', markersize=4, histtype='errorbar')
                hep.histplot(stack_data, bins=x_bins, xerr=True, stack=True, label=labels, color=colors, histtype='fill')

                ax.set_xlabel(xtitle)
                ax.set_ylabel("Events / bin")
                ax.set_yscale("log" if Region.Logy > 0 else "linear")
                ax.set_ylim(1e0, 2.5e4)
                ax.set_xlim(xMin, xMax)
                plt.legend()

                plt.savefig(f"{Outdir}/{Variable.Name}_{Region.Name}.png")
                print(f"Saved {Outdir}/{Variable.Name}_{Region.Name}.png")
                plt.close()

                    
