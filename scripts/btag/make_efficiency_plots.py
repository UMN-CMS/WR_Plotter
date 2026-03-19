import os
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT.gROOT.SetBatch(True)

# =============================
# Configuration
# =============================
base_path_template = "rootfiles/RunII/2018/RunIISummer20UL18/{btag}/"
signal_file_template = "WRAnalyzer_signal_WR3200_N{Nmass}.root"
dy_file = "WRAnalyzer_DYJets.root"
ttbar_file = "WRAnalyzer_TTbar.root"

variable_path_2d = "wr_mumu_resolved_sr/WRMass_DeltaRbb_wr_mumu_resolved_sr"
lum = 59.74  # fb^-1
rebin_factor = 2

btag_categories = ["btagloose", "btagmedium", "btagtight"]
colors = {"btagloose": "blue", "btagmedium": "orange", "btagtight": "green"}
markers = {"btagloose": "o", "btagmedium": "s", "btagtight": "^"}

neutrino_masses = [800, 1200, 1600, 2400, 3000]

outdir = Path("plots/fom_1d")
outdir.mkdir(parents=True, exist_ok=True)

scale_factor = lum * 1000

# =============================
# Loop over neutrino masses
# =============================
for Nmass in neutrino_masses:
    print(f"\n===== Processing Neutrino Mass N={Nmass} =====")

    # Load signal from loose for fit
    signal_file = signal_file_template.format(Nmass=Nmass)
    signal_file_root = ROOT.TFile(base_path_template.format(btag="btagloose") + signal_file)
    h_signal_fit = signal_file_root.Get(variable_path_2d)
    h_signal_fit.RebinY(rebin_factor)

    # Project X for m_lljj
    h_mlljj = h_signal_fit.ProjectionX(f"h_mlljj_N{Nmass}")

    # Asymmetric Gaussian fit
    asym_gaus = ROOT.TF1(
        f"asym_gaus_N{Nmass}",
        """
        [0] * (
            (x < [1]) * exp(-0.5*((x-[1])/[2])^2) +
            (x >= [1]) * exp(-0.5*((x-[1])/[3])^2)
        )
        """,
        h_mlljj.GetXaxis().GetXmin(),
        h_mlljj.GetXaxis().GetXmax()
    )

    # Initial guesses
    peak_bin = h_mlljj.GetMaximumBin()
    peak_pos = h_mlljj.GetBinCenter(peak_bin)
    peak_val = h_mlljj.GetMaximum()
    asym_gaus.SetParameters(peak_val, peak_pos, 200, 50)
    h_mlljj.Fit(asym_gaus)

    mu = asym_gaus.GetParameter(1)
    sigmaL = abs(asym_gaus.GetParameter(2))
    sigmaR = abs(asym_gaus.GetParameter(3))
    low_edge = mu - 2*sigmaL
    high_edge = mu + 2*sigmaR

    print(f"N={Nmass}: mu={mu:.1f}, sigmaL={sigmaL:.1f}, sigmaR={sigmaR:.1f}, 2σ=[{low_edge:.1f},{high_edge:.1f}]")

    # Save fit validation plot
    c = ROOT.TCanvas(f"c_fit_N{Nmass}", f"Asymmetric Gaussian Fit N{Nmass}", 800, 600)
    h_mlljj.SetLineColor(ROOT.kBlack)
    h_mlljj.SetMarkerStyle(20)
    h_mlljj.Draw("E")
    asym_gaus.SetLineColor(ROOT.kRed)
    asym_gaus.SetLineWidth(2)
    asym_gaus.Draw("same")

    line_low = ROOT.TLine(low_edge, 0, low_edge, h_mlljj.GetMaximum()*1.1)
    line_high = ROOT.TLine(high_edge, 0, high_edge, h_mlljj.GetMaximum()*1.1)
    line_mu = ROOT.TLine(mu, 0, mu, h_mlljj.GetMaximum()*1.1)
    for l in [line_low, line_high]:
        l.SetLineColor(ROOT.kBlue)
        l.SetLineStyle(2)
        l.Draw()
    line_mu.SetLineColor(ROOT.kGreen+2)
    line_mu.SetLineStyle(3)
    line_mu.Draw()

    leg = ROOT.TLegend(0.6, 0.7, 0.88, 0.88)
    leg.AddEntry(h_mlljj, "Signal Projection", "lep")
    leg.AddEntry(asym_gaus, "Asymmetric Gaussian Fit", "l")
    leg.AddEntry(line_mu, "Fit Mean (#mu)", "l")
    leg.AddEntry(line_low, "2#sigma Window", "l")
    leg.Draw()
    c.SetGrid()
    c.SaveAs(str(outdir / f"mlljj_asym_fit_N{Nmass}.pdf"))

    # =============================
    # Compute FOM for all b-tags
    # =============================
    fom_dict = {}
    signal_dict = {}
    background_dict = {}
    ttbar_dict = {}
    deltar_dict = {}

    for btag in btag_categories:
        base_path = base_path_template.format(btag=btag)
        f_signal = ROOT.TFile(base_path + signal_file)
        f_dy = ROOT.TFile(base_path + dy_file)
        f_ttbar = ROOT.TFile(base_path + ttbar_file)

        h_signal = f_signal.Get(variable_path_2d)
        h_dy = f_dy.Get(variable_path_2d)
        h_ttbar = f_ttbar.Get(variable_path_2d)

        h_signal.Scale(scale_factor)
        h_dy.Scale(scale_factor)
        h_ttbar.Scale(scale_factor)

        h_signal.RebinY(rebin_factor)
        h_dy.RebinY(rebin_factor)
        h_ttbar.RebinY(rebin_factor)

        ny = h_signal.GetNbinsY()
        nx = h_signal.GetNbinsX()

        fom_vals = []
        s_vals = []
        b_vals = []
        tt_vals = []
        deltar_centers = []

        for iy in range(1, ny+1):
            S = 0; B = 0; tt = 0
            for ix in range(1, nx+1):
                x = h_signal.GetXaxis().GetBinCenter(ix)
                if low_edge <= x <= high_edge:
                    S += h_signal.GetBinContent(ix, iy)
                    B += h_dy.GetBinContent(ix, iy) + h_ttbar.GetBinContent(ix, iy)
                    tt += h_ttbar.GetBinContent(ix, iy)
            fom_vals.append(S/np.sqrt(B) if B>0 else 0)
            s_vals.append(S)
            b_vals.append(B)
            tt_vals.append(tt)
            deltar_centers.append(h_signal.GetYaxis().GetBinCenter(iy))

        # Remove first bin
        fom_dict[btag] = np.array(fom_vals[1:])
        signal_dict[btag] = np.array(s_vals[1:])
        background_dict[btag] = np.array(b_vals[1:])
        ttbar_dict[btag] = np.array(tt_vals[1:])
        deltar_dict[btag] = np.array(deltar_centers[1:])

    # =============================
    # Plot comparison
    # =============================
    def plot_comparison(data_dict, y_label, title, filename):
        fig, ax = plt.subplots(figsize=(8,6))
        for btag in btag_categories:
            ax.plot(deltar_dict[btag], data_dict[btag],
                    marker=markers[btag], color=colors[btag],
                    linestyle='-', label=btag)
        ax.set_xlabel("Delta R_bb")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(filename)
        print("Saved", filename)

    plot_comparison(fom_dict, "S / sqrt(B)", 
                    f"FOM vs Delta R (2σ m_lljj window) N={Nmass}", 
                    outdir / f"fom_vs_deltar_2sigma_allBtag_N{Nmass}.pdf")

    plot_comparison(signal_dict, "Signal", 
                    f"Signal vs Delta R (2σ m_lljj window) N={Nmass}", 
                    outdir / f"signal_vs_deltar_2sigma_allBtag_N{Nmass}.pdf")

    plot_comparison(background_dict, "Background", 
                    f"Background vs Delta R (2σ m_lljj window) N={Nmass}", 
                    outdir / f"background_vs_deltar_2sigma_allBtag_N{Nmass}.pdf")

    plot_comparison(ttbar_dict, "TTbar", 
                    f"TTbar vs Delta R (2σ m_lljj window) N={Nmass}", 
                    outdir / f"ttbar_vs_deltar_2sigma_allBtag_N{Nmass}.pdf")