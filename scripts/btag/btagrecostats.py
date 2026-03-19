import ROOT
from pathlib import Path

ROOT.gROOT.SetBatch(True)

# =============================
# Configuration
# =============================

base_path = "rootfiles/RunII/2018/RunIISummer20UL18/btagloose/"

dy_file = "WRAnalyzer_DYJets.root"
ttbar_file = "WRAnalyzer_TTbar.root"
signal_file_template = "WRAnalyzer_signal_{mass}.root"

region = "wr_mumu_resolved_sr"

hist_names = {
    "btag_reco": "btag_reco_pt",
    "nobtag_reco": "nobtag_reco_pt",
    "btag_noreco": "btag_noreco_pt",
    "nobtag_noreco": "nobtag_noreco_pt",
}

lum = 59.74
scale_factor = lum * 1000

outdir = Path("plots/btag_truth_validation")
outdir.mkdir(parents=True, exist_ok=True)

whichone = "WR3200_N3000"

# =============================
# Load ROOT files
# =============================

f_dy = ROOT.TFile(base_path + dy_file)
f_tt = ROOT.TFile(base_path + ttbar_file)
f_signal = ROOT.TFile(base_path + signal_file_template.format(mass=whichone))

# =============================
# Load histograms
# =============================

hists = {}

for key,name in hist_names.items():

    h_dy = f_dy.Get(f"{region}/{name}_{region}").Clone()
    h_tt = f_tt.Get(f"{region}/{name}_{region}").Clone()
    h_signal = f_signal.Get(f"{region}/{name}_{region}").Clone()

    # h = h_tt.Clone(name+"_combined")
    # h.Add(h_dy)

    h = h_signal.Clone(name+"_signal")

    h.Scale(scale_factor)
    h.SetStats(0)

    # Axis limits
    h.GetXaxis().SetRangeUser(40,1500)

    hists[key] = h

# ==========================================================
# 1) Plot pT histograms in confusion matrix layout
# ==========================================================

c = ROOT.TCanvas("c_pt_matrix","",1000,1000)
c.Divide(2,2)

layout = [
    ("nobtag_reco","True b + Not tagged"),
    ("btag_reco","True b + Tagged"),
    ("nobtag_noreco","Non-b + Not tagged"),
    ("btag_noreco","Non-b + Tagged"),
]

for i,(key,title) in enumerate(layout,1):

    c.cd(i)

    ROOT.gPad.SetLogy()

    h = hists[key]

    h.SetTitle(title)
    h.GetXaxis().SetTitle("Jet p_{T} [GeV]")
    h.GetYaxis().SetTitle("Events")

    h.Draw("HIST")

    ROOT.gPad.SetGrid()

outfile = outdir / f"btag_pt_confusion_layout_{whichone}.pdf"
c.SaveAs(str(outfile))
print("Saved",outfile)


# ==========================================================
# 2) Build confusion matrix
# ==========================================================

N_true_tag = hists["btag_reco"].Integral()
N_true_notag = hists["nobtag_reco"].Integral()
N_false_tag = hists["btag_noreco"].Integral()
N_false_notag = hists["nobtag_noreco"].Integral()

conf = ROOT.TH2F(
    "conf",
    f"b-tagging(loose) {whichone};;",
    2,0,2,
    2,0,2
)

# Fill bins (True,True top-right)
conf.SetBinContent(2,2,N_true_tag)
conf.SetBinContent(1,2,N_true_notag)
conf.SetBinContent(2,1,N_false_tag)
conf.SetBinContent(1,1,N_false_notag)

# Axis labels
conf.GetXaxis().SetBinLabel(1,"Not tagged")
conf.GetXaxis().SetBinLabel(2,"Tagged")

conf.GetYaxis().SetBinLabel(1,"Non-b jet")
conf.GetYaxis().SetBinLabel(2,"True b jet")

# =============================
# Draw matrix
# =============================

c2 = ROOT.TCanvas("c_confusion","",700,600)

ROOT.gStyle.SetPaintTextFormat(".0f")

conf.SetStats(0)
conf.Draw("COLZ TEXT")

text = ROOT.TLatex()
text.SetNDC()
text.SetTextSize(0.04)
text.DrawLatex(0.12,0.92,"#bf{CMS} Simulation")
text.DrawLatex(0.65,0.92,"59.7 fb^{-1} (13 TeV)")

outfile = outdir / f"btag_confusion_matrix_{whichone}.pdf"
c2.SaveAs(str(outfile))
print("Saved",outfile)


# ==========================================================
# 3) Print performance metrics
# ==========================================================

eff_b = N_true_tag/(N_true_tag + N_true_notag)
mistag = N_false_tag/(N_false_tag + N_false_notag)

print("\n=== B-tag performance ===")
print("b-tag efficiency =",eff_b)
print("mistag rate =",mistag)