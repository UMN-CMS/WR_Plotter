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

nonprompt_files = [
    "WRAnalyzer_WJets.root",
    "WRAnalyzer_TTbarSemileptonic.root",
    "WRAnalyzer_SingleTop.root",
]

other_samples_files = [
    "WRAnalyzer_Diboson.root",
    "WRAnalyzer_Triboson.root",
]

hist_path_btag   = "wr_mumu_resolved_sr/jetidxbtag_wr_mumu_resolved_sr"
hist_path_nobtag = "wr_mumu_resolved_sr/jetidxnobtag_wr_mumu_resolved_sr"

lum = 59.74
scale_factor = lum * 1000

btag_categories = ["btagloose"]
neutrino_masses = [800, 1200, 1600, 2400, 3000]

signal_scale = 1

outdir = Path("plots/jetidxbtag")
outdir.mkdir(parents=True, exist_ok=True)

# =============================
# Helpers
# =============================

def get_both(f):
    return (
        f.Get(hist_path_btag).Clone(),
        f.Get(hist_path_nobtag).Clone()
    )

# 🔥 Interleave with spacing (key function)
def interleave_hist_with_spacing(h_btag, h_nobtag, name):

    nbins = h_btag.GetNbinsX()

    # Each original bin → 3 bins: [no b, b, gap]
    h_new = ROOT.TH1F(name, "", nbins * 3, 0, nbins * 3)

    for i in range(1, nbins + 1):

        base = 3 * (i - 1)

        # no-btag (left)
        h_new.SetBinContent(base + 1, h_nobtag.GetBinContent(i))
        h_new.SetBinError(base + 1, h_nobtag.GetBinError(i))

        # btag (right)
        h_new.SetBinContent(base + 2, h_btag.GetBinContent(i))
        h_new.SetBinError(base + 2, h_btag.GetBinError(i))

        # base+3 = gap (left empty)

    return h_new

def label_split_bins(hist):
    labels = ["0", "1", "2"]

    for i, lab in enumerate(labels, start=1):
        base = 3 * (i - 1)
        hist.GetXaxis().SetBinLabel(base + 1, f"{lab} (no b)")
        hist.GetXaxis().SetBinLabel(base + 2, f"{lab} (b)")
        hist.GetXaxis().SetBinLabel(base + 3, "")  # gap

    hist.GetXaxis().SetTitle("Jet index")
    hist.GetYaxis().SetTitle("Events")

# =============================
# Loop
# =============================

for Nmass in neutrino_masses:

    print(f"\nProcessing N = {Nmass}")
    signal_file = signal_file_template.format(Nmass=Nmass)

    for btag in btag_categories:

        base_path = base_path_template.format(btag=btag)

        # Open files
        f_signal = ROOT.TFile(base_path + signal_file)
        f_dy = ROOT.TFile(base_path + dy_file)
        f_ttbar = ROOT.TFile(base_path + ttbar_file)
        f_nonprompt = [ROOT.TFile(base_path + fname) for fname in nonprompt_files]
        f_other = [ROOT.TFile(base_path + fname) for fname in other_samples_files]

        # =============================
        # Get histograms
        # =============================

        h_signal_b, h_signal_nb = get_both(f_signal)
        h_dy_b, h_dy_nb = get_both(f_dy)
        h_ttbar_b, h_ttbar_nb = get_both(f_ttbar)

        h_nonprompt_b, h_nonprompt_nb = get_both(f_nonprompt[0])
        h_other_b, h_other_nb = get_both(f_other[0])

        # Add remaining files
        for f in f_nonprompt[1:]:
            b, nb = get_both(f)
            h_nonprompt_b.Add(b)
            h_nonprompt_nb.Add(nb)

        for f in f_other[1:]:
            b, nb = get_both(f)
            h_other_b.Add(b)
            h_other_nb.Add(nb)

        # =============================
        # Scale
        # =============================

        for h in [h_signal_b, h_signal_nb]:
            h.Scale(scale_factor * signal_scale)

        for h in [h_dy_b, h_dy_nb,
                  h_ttbar_b, h_ttbar_nb,
                  h_nonprompt_b, h_nonprompt_nb,
                  h_other_b, h_other_nb]:
            h.Scale(scale_factor)

        # =============================
        # Interleave with spacing
        # =============================

        h_signal = interleave_hist_with_spacing(h_signal_b, h_signal_nb, "signal")
        h_dy = interleave_hist_with_spacing(h_dy_b, h_dy_nb, "dy")
        h_ttbar = interleave_hist_with_spacing(h_ttbar_b, h_ttbar_nb, "ttbar")
        h_nonprompt = interleave_hist_with_spacing(h_nonprompt_b, h_nonprompt_nb, "nonprompt")
        h_other = interleave_hist_with_spacing(h_other_b, h_other_nb, "other")

        # Labels
        for h in [h_signal, h_dy, h_ttbar, h_nonprompt, h_other]:
            label_split_bins(h)

        # =============================
        # Style (single color per process)
        # =============================

        h_ttbar.SetFillColor(ROOT.kGreen+2)
        h_ttbar.SetLineColor(ROOT.kGreen+3)

        h_dy.SetFillColor(ROOT.kAzure-9)
        h_dy.SetLineColor(ROOT.kAzure-3)

        h_nonprompt.SetFillColor(ROOT.kMagenta-9)
        h_nonprompt.SetLineColor(ROOT.kMagenta-3)

        h_other.SetFillColor(ROOT.kCyan-9)
        h_other.SetLineColor(ROOT.kCyan-3)

        h_signal.SetLineColor(ROOT.kRed+1)
        h_signal.SetLineWidth(3)
        h_signal.SetFillStyle(0)

        # =============================
        # Stack
        # =============================

        stack = ROOT.THStack(
            f"stack_{btag}_{Nmass}",
            f"JetIdx split ({btag})  N={Nmass};Jet index;Events"
        )

        stack.Add(h_ttbar)
        stack.Add(h_dy)
        stack.Add(h_nonprompt)
        stack.Add(h_other)

        # =============================
        # Canvas
        # =============================

        c = ROOT.TCanvas(f"c_{btag}_{Nmass}", "", 900, 700)

        stack.Draw("HIST")
        h_signal.Draw("HIST SAME")

        max_val = max(stack.GetMaximum(), h_signal.GetMaximum())
        stack.SetMinimum(1.1)
        stack.SetMaximum(max_val * 1.2)

        # =============================
        # Vertical separators (between bins only)
        # =============================

        line = ROOT.TLine()
        line.SetLineStyle(2)

        for x in [3, 6]:
            line.DrawLine(x, 0, x, max_val * 1.2)

        # =============================
        # Legend
        # =============================

        leg = ROOT.TLegend(0.68, 0.70, 0.88, 0.86)
        leg.SetTextSize(0.03)
        leg.SetBorderSize(0)

        leg.AddEntry(h_ttbar, "t#bar{t}", "f")
        leg.AddEntry(h_dy, "DY", "f")
        leg.AddEntry(h_nonprompt, "Non-prompt", "f")
        leg.AddEntry(h_other, "Other", "f")
        leg.AddEntry(h_signal, "Signal", "l")
        leg.Draw()

        # =============================
        # CMS text
        # =============================

        text = ROOT.TLatex()
        text.SetNDC()
        text.SetTextSize(0.04)
        text.DrawLatex(0.12, 0.92, "#bf{CMS} Simulation")
        text.DrawLatex(0.65, 0.92, "59.7 fb^{-1} (13 TeV)")

        c.SetGrid()

        # =============================
        # Save
        # =============================

        outfile = outdir / f"jetidx_inset_{btag}_N{Nmass}.pdf"
        c.SaveAs(str(outfile))

        print("Saved", outfile)