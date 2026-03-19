import ROOT
from pathlib import Path

ROOT.gROOT.SetBatch(True)

# =============================
# Configuration
# =============================

base_path_template = "rootfiles/RunII/2018/RunIISummer20UL18/{btag}/"

signal_file_template = "WRAnalyzer_signal_WR3200_N{Nmass}.root"
dy_file = "WRAnalyzer_DYJets.root"
ttbar_file = "WRAnalyzer_TTbar.root"

hist_path = "wr_mumu_resolved_sr/bjet_multiplicity_wr_mumu_resolved_sr"

lum = 59.74
scale_factor = lum * 1000

btag_categories = ["btagloose"]
neutrino_masses = [800, 1200, 1600, 2400, 3000]

outdir = Path("plots/bjet_multiplicity_stack")
outdir.mkdir(parents=True, exist_ok=True)

# =============================
# Function to label bins
# =============================
def label_bins(hist):
    hist.GetXaxis().SetBinLabel(1, "0")
    hist.GetXaxis().SetBinLabel(2, "1")
    hist.GetXaxis().SetBinLabel(3, "#geq2")
    hist.GetXaxis().SetTitle("b-jet multiplicity")
    hist.GetYaxis().SetTitle("Events")
    return hist

def normalize_per_bin(hists):
    """
    Convert list of histograms into per-bin fractions.
    Modifies histograms in-place.
    """
    nbins = hists[0].GetNbinsX()

    for i in range(1, nbins + 1):

        total = sum(h.GetBinContent(i) for h in hists)

        if total > 0:
            for h in hists:
                value = h.GetBinContent(i)
                h.SetBinContent(i, value / total)
        else:
            for h in hists:
                h.SetBinContent(i, 0)


# =============================
# Loop over neutrino masses
# =============================
for Nmass in neutrino_masses:

    print(f"\nProcessing N = {Nmass}")

    signal_file = signal_file_template.format(Nmass=Nmass)

    for btag in btag_categories:

        base_path = base_path_template.format(btag=btag)

        f_signal = ROOT.TFile(base_path + signal_file)
        f_dy = ROOT.TFile(base_path + dy_file)
        f_ttbar = ROOT.TFile(base_path + ttbar_file)

        h_signal = f_signal.Get(hist_path).Clone()
        h_dy = f_dy.Get(hist_path).Clone()
        h_ttbar = f_ttbar.Get(hist_path).Clone()

        # Scale to luminosity
        h_signal.Scale(scale_factor)
        h_dy.Scale(scale_factor)
        h_ttbar.Scale(scale_factor)

        # Normalize per bin (fractions)
        # normalize_per_bin([h_signal, h_ttbar, h_dy])
        
        # Label bins
        label_bins(h_signal)
        label_bins(h_dy)
        label_bins(h_ttbar)

        # =============================
        # Style histograms
        # =============================
        h_signal.SetFillColor(ROOT.kOrange-3)
        h_signal.SetLineColor(ROOT.kOrange+1)

        h_ttbar.SetFillColor(ROOT.kGreen+2)
        h_ttbar.SetLineColor(ROOT.kGreen+3)

        h_dy.SetFillColor(ROOT.kAzure-9)
        h_dy.SetLineColor(ROOT.kAzure-3)

        # =============================
        # Stack (Signal → TTbar → DY)
        # =============================
        stack = ROOT.THStack(
            f"stack_{btag}_{Nmass}",
            f"b-jet multiplicity ({btag})  N={Nmass};b-jet multiplicity;Events"
        )

        stack.Add(h_signal)  # bottom
        stack.Add(h_ttbar)
        stack.Add(h_dy)      # top

        # =============================
        # Canvas
        # =============================
        c = ROOT.TCanvas(f"c_{btag}_{Nmass}", "", 800, 700)
        c.SetLogy()  # log scale

        stack.Draw("HIST")

        # Important for log scale
        stack.SetMinimum(0.1)
        stack.SetMaximum(stack.GetMaximum() * 10)

        # =============================
        # Legend
        # =============================
        leg = ROOT.TLegend(0.65, 0.65, 0.88, 0.88)
        leg.AddEntry(h_signal, "Signal", "f")
        leg.AddEntry(h_ttbar, "t#bar{t}", "f")
        leg.AddEntry(h_dy, "DY", "f")
        leg.Draw()

        # =============================
        # CMS style text
        # =============================
        text = ROOT.TLatex()
        text.SetNDC()
        text.SetTextSize(0.04)
        text.DrawLatex(0.12, 0.92, "#bf{CMS} Simulation")
        text.DrawLatex(0.65, 0.92, "59.7 fb^{-1} (13 TeV)")

        c.SetGrid()

        # =============================
        # Save plot
        # =============================
        outfile = outdir / f"bjet_multiplicity_stack_{btag}_N{Nmass}.pdf"
        c.SaveAs(str(outfile))

        print("Saved", outfile)


# =============================
# SIGNAL PT + RATIO PLOTS
# =============================
for Nmass in neutrino_masses:

    print(f"\nProcessing N = {Nmass} for pt plots")

    signal_file = signal_file_template.format(Nmass=Nmass)

    for btag in btag_categories:

        base_path = base_path_template.format(btag=btag)
        f_signal = ROOT.TFile(base_path + signal_file)

        for jet_idx, jet_name in enumerate(["jet0", "jet1"]):

            h_btagged = f_signal.Get("wr_mumu_resolved_sr/b" + jet_name + '_pt_wr_mumu_resolved_sr').Clone()
            h_vetoed = f_signal.Get("wr_mumu_resolved_sr/b" + jet_name + "veto_pt_wr_mumu_resolved_sr").Clone()

            if not h_btagged or not h_vetoed:
                print(f"WARNING: {jet_name} histograms not found for N={Nmass}")
                continue

            h_btagged.Scale(scale_factor)
            h_vetoed.Scale(scale_factor)

            xmax = max(h_btagged.GetXaxis().GetXmax(), h_vetoed.GetXaxis().GetXmax())
            h_btagged.GetXaxis().SetRangeUser(40, xmax)
            h_vetoed.GetXaxis().SetRangeUser(40, xmax)

            h_btagged.GetXaxis().SetLabelSize(0)
            h_btagged.GetXaxis().SetTitleSize(0)

            h_btagged.SetLineColor(ROOT.kOrange+1)
            h_vetoed.SetLineColor(ROOT.kAzure+3)

            h_btagged.SetLineWidth(2)
            h_vetoed.SetLineWidth(2)

            h_btagged.SetStats(0)
            h_vetoed.SetStats(0)

            c = ROOT.TCanvas(f"c_{jet_name}_ratio_{btag}_N{Nmass}", "", 800, 800)
            pad1 = ROOT.TPad("pad1", "pad1", 0, 0.3, 1, 1)
            pad2 = ROOT.TPad("pad2", "pad2", 0, 0, 1, 0.3)

            pad1.SetBottomMargin(0.02)
            pad2.SetTopMargin(0.02)
            pad2.SetBottomMargin(0.3)

            pad1.Draw()
            pad2.Draw()

            pad1.cd()
            h_btagged.Draw("HIST")
            h_vetoed.Draw("HIST SAME")

            h_btagged.SetMaximum(max(h_btagged.GetMaximum(), h_vetoed.GetMaximum()) * 1.5)
            h_btagged.SetTitle(f"{jet_name} ({btag}) Signal N={Nmass}")
            h_btagged.GetYaxis().SetTitle("Events")

            leg = ROOT.TLegend(0.65, 0.7, 0.88, 0.88)
            leg.AddEntry(h_btagged, "b-tagged", "l")
            leg.AddEntry(h_vetoed, "b-tag veto", "l")
            leg.Draw()

            text = ROOT.TLatex()
            text.SetNDC()
            text.DrawLatex(0.12, 0.92, "#bf{CMS} Simulation")
            text.DrawLatex(0.65, 0.92, "59.7 fb^{-1} (13 TeV)")

            pad2.cd()
            h_ratio = h_vetoed.Clone("ratio")
            h_ratio.Divide(h_btagged)

            h_ratio.SetLineColor(ROOT.kBlack)
            h_ratio.SetLineWidth(2)
            h_ratio.GetYaxis().SetTitle("Veto / Tagged")
            h_ratio.GetXaxis().SetTitle("p_{T} [GeV]")

            h_ratio.SetMinimum(0)
            h_ratio.SetMaximum(4)

            h_ratio.Draw("HIST")

            pad1.SetGrid()
            pad2.SetGrid()

            outfile = outdir / f"{jet_name}_signal_lines_{btag}_N{Nmass}.pdf"
            c.SaveAs(str(outfile))

            print("Saved", outfile)