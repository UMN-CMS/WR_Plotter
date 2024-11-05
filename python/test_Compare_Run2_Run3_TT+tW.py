import ROOT
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

# Set up the hep style for plotting
plt.style.use(hep.style.ROOT)

# Define paths to the files
file_path_run2 = "/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter/root_files/Run2Summer20UL18/WRAnalyzer_SkimTree_LRSMHighPt_tt+tW.root"
file_path_run3 = "/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter/root_files/Run3Summer22/WRAnalyzer_SkimTree_LRSMHighPt_tt+tW.root"

# Define the folders and histograms in each folder
folders = [
    "WR_EE_Resolved_DYCR",
    "WR_MuMu_Resolved_DYCR",
    "WR_EE_Resolved_DYSR",
    "WR_MuMu_Resolved_DYSR",
    "WR_EMu_Resolved_Sideband"
]

histograms = [
    "Jet_0_Pt_",
    "Lepton_0_Pt_",
    "WRCand_Mass_"
]

# Define rebinning settings
bin_widths = {
    "Jet_0_Pt_": 20,
    "Lepton_0_Pt_": 20,
    "WRCand_Mass_": [800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 8000]
}

# Open both files using ROOT
file_run2 = ROOT.TFile.Open(file_path_run2)
file_run3 = ROOT.TFile.Open(file_path_run3)

# Define the text mappings based on folder names
text_mapping = {
    "WR_EE_Resolved_DYCR": ("ee", "DY CR"),
    "WR_MuMu_Resolved_DYCR": ("$\mu\mu$", "DY CR"),
    "WR_EE_Resolved_DYSR": ("ee", "DY SR"),
    "WR_MuMu_Resolved_DYSR": ("$\mu\mu$", "DY SR"),
    "WR_EMu_Resolved_Sideband": ("$e\mu$", "Sideband")
}

# Loop through each folder and histogram, plot them overlayed
for folder in folders:
    for hist_prefix in histograms:
        # Construct the histogram name
        hist_name = f"{hist_prefix}{folder}"
        print(hist_name)

        # Access the histogram from both files
        hist_run2 = file_run2.Get(f"{folder}/{hist_name}")
        hist_run3 = file_run3.Get(f"{folder}/{hist_name}")

        # Rebin according to the desired bin width
        if hist_prefix == "Jet_0_Pt_":
            rebin_factor = bin_widths["Jet_0_Pt_"] // 10  # Original bin width is 10 GeV
            hist_run2.Rebin(rebin_factor)
            hist_run3.Rebin(rebin_factor)
        elif hist_prefix == "Lepton_0_Pt_":
            rebin_factor = bin_widths["Lepton_0_Pt_"] // 10  # Original bin width is 10 GeV
            hist_run2.Rebin(rebin_factor)
            hist_run3.Rebin(rebin_factor)

        if hist_prefix == "WRCand_Mass_":
            # Use variable binning
            new_bins = np.array(bin_widths["WRCand_Mass_"], dtype="d")
            hist_run2 = hist_run2.Rebin(len(new_bins) - 1, f"{hist_name}_rebin", new_bins)
            hist_run3 = hist_run3.Rebin(len(new_bins) - 1, f"{hist_name}_rebin", new_bins)
        
            # Get the bin widths
            bin_widths_array = np.diff(new_bins)

            # Convert ROOT histograms to numpy arrays
            values_run2 = np.array([hist_run2.GetBinContent(i)/bin_widths_array[i-1] for i in range(1, hist_run2.GetNbinsX() + 1)])
            edges_run2 = np.array([hist_run2.GetBinLowEdge(i) for i in range(1, hist_run2.GetNbinsX() + 2)])

            values_run3 = np.array([hist_run3.GetBinContent(i)/bin_widths_array[i-1] for i in range(1, hist_run3.GetNbinsX() + 1)])
            edges_run3 = np.array([hist_run3.GetBinLowEdge(i) for i in range(1, hist_run3.GetNbinsX() + 2)])
        else:
            # Convert ROOT histograms to numpy arrays
            values_run2 = np.array([hist_run2.GetBinContent(i) for i in range(1, hist_run2.GetNbinsX() + 1)])
            edges_run2 = np.array([hist_run2.GetBinLowEdge(i) for i in range(1, hist_run2.GetNbinsX() + 2)])

            values_run3 = np.array([hist_run3.GetBinContent(i) for i in range(1, hist_run3.GetNbinsX() + 1)])
            edges_run3 = np.array([hist_run3.GetBinLowEdge(i) for i in range(1, hist_run3.GetNbinsX() + 2)])

        # Get annotation text for the current folder
        annotation_top, annotation_bottom = text_mapping[folder]

        # Plot unnormalized histograms
        fig, ax = plt.subplots()
        hep.histplot(values_run2, edges_run2, label="tt+tW (Run2Summer20UL18)", histtype="step", color="blue")
        hep.histplot(values_run3, edges_run3, label="tt+tW (Run3Summer22)", histtype="step", color="red")

        # Add text annotations
        ax.text(0.05, 0.96, annotation_top, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', horizontalalignment='left')
        ax.text(0.05, 0.91, annotation_bottom, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', horizontalalignment='left')

        # Add titles and labels for unnormalized plot
        if hist_prefix == "Jet_0_Pt_":
            plt.xlabel("$p_{T}$ of the leading jet [GeV]")
            plt.ylabel("Events / 20 GeV")
            if "DYCR" in hist_name:
                plt.ylim(1e-5,1e-0)
            elif "DYSR" in hist_name:
                plt.ylim(1e-6,1e-2)
            elif "Sideband" in hist_name:
                plt.ylim(1e-5,1e-1)
        elif hist_prefix == "Lepton_0_Pt_":
            plt.xlabel("$p_{T}$ of the leading lepton [GeV]")
            plt.ylabel("Events / 20 GeV")
            if "DYCR" in hist_name:
                plt.ylim(1e-5,1e0)
            elif "DYSR" in hist_name:
                plt.ylim(1e-6,1e-2)
            elif "Sideband" in hist_name:
                plt.ylim(1e-5,1e0)
        elif hist_prefix == "WRCand_Mass_":
            plt.xlabel("$m_{lljj}$ [GeV]")
            plt.ylabel("Events / X GeV")
            plt.xlim(800, 8000)
            if "DYCR" in hist_name:
                plt.ylim(1e-7,1e-2)
            elif "DYSR" in hist_name:
                plt.ylim(1e-9,1e-3)
            elif "Sideband" in hist_name:
                plt.ylim(1e-7,1e-2)

        ax.set_yscale('log')
        plt.legend()

        # Save unnormalized plot
        plt.savefig(f"{hist_name}_unnormalized.png")
        plt.close()

        # Normalize the histograms by area using ROOT's Integral method
        integral_run2 = hist_run2.Integral()
        integral_run3 = hist_run3.Integral()

        # Avoid division by zero in case of empty histograms
        values_run2_norm = values_run2 / integral_run2 if integral_run2 != 0 else values_run2
        values_run3_norm = values_run3 / integral_run3 if integral_run3 != 0 else values_run3

        # Plot normalized histograms
        fig, ax = plt.subplots()
        hep.histplot(values_run2_norm, edges_run2, label="tt+tW (Run2Summer20UL18)", histtype="step", color="blue")
        hep.histplot(values_run3_norm, edges_run3, label="tt+tW (Run3Summer22)", histtype="step", color="red")

        ax.text(0.05, 0.96, annotation_top, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', horizontalalignment='left')
        ax.text(0.05, 0.91, annotation_bottom, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', horizontalalignment='left')

        # Add titles and labels for normalized plot
        if hist_prefix == "Jet_0_Pt_":
            plt.xlabel("$p_{T}$ of the leading jet [GeV]")
            plt.ylabel("Normalized Events / 20 GeV")
            if "DYCR" in hist_name:
                plt.ylim(1e-4,1e-0)
            elif "DYSR" in hist_name:
                plt.ylim(1e-4,1e-0)
            elif "Sideband" in hist_name:
                plt.ylim(1e-5,1e-1)
        elif hist_prefix == "Lepton_0_Pt_":
            plt.xlabel("$p_{T}$ of the leading lepton [GeV]")
            plt.ylabel("Normalized Events / 20 GeV")
            if "DYCR" in hist_name:
                plt.ylim(1e-4,1e-0)
            elif "DYSR" in hist_name:
                plt.ylim(1e-4,1e-0)
            elif "Sideband" in hist_name:
                plt.ylim(1e-5,1e-1)
        elif hist_prefix == "WRCand_Mass_":
            plt.xlabel("$m_{lljj}$ [GeV]")
            plt.ylabel("Normalized Events / X GeV")
            plt.xlim(800, 8000)
            if "DYCR" in hist_name:
                plt.ylim(1e-7,1e-1)
            elif "DYSR" in hist_name:
                plt.ylim(1e-7,1e-1)
            elif "Sideband" in hist_name:
                plt.ylim(1e-7,1e-2)

        ax.set_yscale('log')
        plt.legend()

        # Save normalized plot
        plt.savefig(f"{hist_name}_normalized.png")
        plt.close()

print("Both normalized and unnormalized histograms have been plotted and saved.")

# Close ROOT files
file_run2.Close()
file_run3.Close()
