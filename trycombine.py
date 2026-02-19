import ROOT
import numpy as np
import matplotlib.pyplot as plt
ROOT.gROOT.SetBatch(True)

def shift_histogram(hist, shift):
    # Shift histogram along x-axis by 'shift' units.
    shifted = hist.Clone(hist.GetName() + f"_shifted_{shift}")
    shifted.Reset()

    nbins = hist.GetNbinsX()
    for i in range(1, nbins + 1):
        x = hist.GetBinCenter(i)
        y = hist.GetBinContent(i)
        new_x = x + shift
        new_bin = shifted.FindBin(new_x)
        if 1 <= new_bin <= nbins:
            shifted.SetBinContent(new_bin, shifted.GetBinContent(new_bin) + y)
    return shifted

WRmasses=[1200,1600,2000,2400,2800,3200]
Nmasses=[[200,400,600,800,1100],[400,600,800,1200,1500],[400,800,1000,1400,1900],[600,800,1200,1800,2300],[600,1000,1400,2000,2700],[800,1200,1600,2400,3000]]

fs_1200=ROOT.TFile("WRAnalyzer_signal_WR1200_N1100.root","r")
sr_1200=fs_1200.Get("wr_mumu_resolved_sr/mass_fourobject_wr_mumu_resolved_sr")
fs_1600=ROOT.TFile("WRAnalyzer_signal_WR1600_N1200.root","r")
sr_1600=fs_1600.Get("wr_mumu_resolved_sr/mass_fourobject_wr_mumu_resolved_sr")
fs_2000=ROOT.TFile("WRAnalyzer_signal_WR2000_N1000.root","r")
sr_2000=fs_2000.Get("wr_mumu_resolved_sr/mass_fourobject_wr_mumu_resolved_sr")
fs_2800=ROOT.TFile("WRAnalyzer_signal_WR2800_N1000.root","r")
sr_2800=fs_2800.Get("wr_mumu_resolved_sr/mass_fourobject_wr_mumu_resolved_sr")
fs_3200=ROOT.TFile("WRAnalyzer_signal_WR3200_N1200.root","r")
sr_3200=fs_3200.Get("wr_mumu_resolved_sr/mass_fourobject_wr_mumu_resolved_sr")

fs_2400= ROOT.TFile("WRAnalyzer_signal_WR2400_N1200.root","r")
fb= ROOT.TFile("WRAnalyzer_DYJets.root","r")
ftt=ROOT.TFile("WRAnalyzer_TTbar.root","r")

s_sr = fs_2400.Get("wr_mumu_resolved_sr/mass_fourobject_wr_mumu_resolved_sr")
sr_2400 = s_sr.Clone("mass_fourobject_wr_mumu_resolved_sr")
b_sr = fb.Get("wr_mumu_resolved_sr/mass_fourobject_wr_mumu_resolved_sr")
tt_sr = ftt.Get("wr_mumu_resolved_sr/mass_fourobject_wr_mumu_resolved_sr")

s_cr = fs_2400.Get("wr_mumu_resolved_dy_cr/mass_fourobject_wr_mumu_resolved_dy_cr")
b_cr = fb.Get("wr_mumu_resolved_dy_cr/mass_fourobject_wr_mumu_resolved_dy_cr")
tt_cr = ftt.Get("wr_mumu_resolved_dy_cr/mass_fourobject_wr_mumu_resolved_dy_cr")

s_fcr = fs_2400.Get("wr_resolved_flavor_cr/mass_fourobject_wr_resolved_flavor_cr")
b_fcr = fb.Get("wr_resolved_flavor_cr/mass_fourobject_wr_resolved_flavor_cr")
tt_fcr = ftt.Get("wr_resolved_flavor_cr/mass_fourobject_wr_resolved_flavor_cr")

for h in (sr_1200, sr_1600, sr_2000, sr_2400, sr_2800, sr_3200):
    integral = h.Integral()
    if integral > 0:
        h.Scale(1.0 / integral)

frac=5
lum=59000
b_cr.Add(tt_cr)
b_cr.Add(s_cr,frac/100)
b_cr.Scale(lum)

b_sr.Add(tt_sr)
b_sr.Add(s_sr,frac/100)
b_sr.Scale(lum)

b_fcr.Add(tt_fcr)
b_fcr.Add(s_fcr,frac/100)
b_fcr.Scale(lum)


def getmasspoint(massp):
    lowpoint=massp//400 * 400
    highpoint=min([(massp//400+1)* 400,3200.0])
    shift=massp-lowpoint
    ratio=shift/400
    shiftedlow=shift_histogram(globals()[f'sr_{int(lowpoint)}'],shift)
    shiftedhigh=shift_histogram(globals()[f'sr_{int(highpoint)}'],shift-400)
    sr_masspoint = shiftedlow.Clone("mass_fourobject_wr_mumu_resolved_sr")
    sr_masspoint.Reset()
    sr_masspoint.Add(shiftedlow, shiftedhigh, 1-ratio, ratio)
    return sr_masspoint

mass = ROOT.RooRealVar("m_lljj", "m_lljj", 2400, 1000, 4000)
mc_sr = ROOT.RooDataHist("sr","sr", ROOT.RooArgSet(mass),b_sr)
mc_cr = ROOT.RooDataHist("cr","cr", ROOT.RooArgSet(mass),b_cr)
mc_fcr= ROOT.RooDataHist("fcr","fcr", ROOT.RooArgSet(mass),b_fcr)

n_bins = 100
binning = ROOT.RooFit.Binning(n_bins,1000,4000)
mass.setRange("loSB", 1000, 1750 )
mass.setRange("hiSB", 2750, 4000 )
mass.setRange("full", 1000, 4000 )
fit_range = "loSB,hiSB"
# fit_range = "full"

alpha1 = ROOT.RooRealVar("alpha1", "alpha1", -0.0038, -0.2, 0 )
alpha2 = ROOT.RooRealVar("alpha2", "alpha2", -0.0038, -0.2, 0 )
a = ROOT.RooRealVar("A", "Coeff TT", 0.005*lum, 0, lum)
d = ROOT.RooRealVar("D", "Delta", 0, -15*lum/59000, 15*lum/59000)
a_scaled = ROOT.RooFormulaVar("A_scaled", "2*@0+@1", ROOT.RooArgList(a,d))
b = ROOT.RooRealVar("B", "Coeff DY", 0.01*lum, 0, lum)

exp1 = ROOT.RooExponential("exp1", "exp1", mass, alpha1)
exp2 = ROOT.RooExponential("exp2", "exp2", mass, alpha2)
exp1_ext  = ROOT.RooExtendPdf("ext_exp1", "A*exp1", exp1, a)
exp2_ext  = ROOT.RooExtendPdf("ext_exp2", "B*exp2", exp2, b)

model_sr = ROOT.RooAddPdf("model_sr_1", "A * exp(-alpha1 * m_lljj) + B * exp(-alpha2 * m_lljj)", ROOT.RooArgList(exp1_ext,exp2_ext))
model_fcr= ROOT.RooExtendPdf("ext_exp1_scaled", "2*A*exp(-alpha1 * m_lljj)", exp1, a_scaled)

sample = ROOT.RooCategory("sample", "sample")
sample.defineType("SR")
sample.defineType("FCR")

combData = ROOT.RooDataHist("combData", "combined", ROOT.RooArgSet(mass), ROOT.RooFit.Index(sample), ROOT.RooFit.Import("SR", mc_sr), ROOT.RooFit.Import("FCR", mc_fcr))#, ROOT.RooFit.Import("CR", mc_cr)
simPdf = ROOT.RooSimultaneous("simPdf", "simultaneous pdf", sample)
simPdf.addPdf(model_sr, "SR")
simPdf.addPdf(model_fcr, "FCR")

fitResult=simPdf.fitTo(combData, ROOT.RooFit.Range(fit_range), ROOT.RooFit.Slice(sample, "SR"), ROOT.RooFit.ProjWData(ROOT.RooArgSet(sample), combData), ROOT.RooFit.Save())

can = ROOT.TCanvas()
plot = mass.frame()
mc_sr.plotOn(plot, binning, ROOT.RooFit.MarkerColor(0), ROOT.RooFit.LineColor(0))
model_sr.plotOn( plot, ROOT.RooFit.NormRange(fit_range), ROOT.RooFit.Range("full"), ROOT.RooFit.LineColor(2))
mc_sr.plotOn(plot, ROOT.RooFit.CutRange(fit_range), binning )
plot.Draw()
can.Update()
can.Draw()
can.SaveAs(f"background_sr.pdf")

can = ROOT.TCanvas()
plot = mass.frame()
mc_fcr.plotOn(plot, binning, ROOT.RooFit.MarkerColor(0), ROOT.RooFit.LineColor(0))
model_fcr.plotOn( plot, ROOT.RooFit.NormRange("full"), ROOT.RooFit.Range("full"), ROOT.RooFit.LineColor(2))
mc_fcr.plotOn(plot, ROOT.RooFit.CutRange("full"), binning )
plot.Draw()
can.Update()
can.Draw()
can.SaveAs(f"background_fcr.pdf")

alpha1.setConstant(True)
alpha1.setError(0)
alpha2.setConstant(True)
alpha2.setError(0)
a.setConstant(True)
a.setError(0)
b.setConstant(True)
b.setError(0)
fit_range="full"

mpoints=np.linspace(1200,3200,num=100,endpoint=False)
cs=np.zeros_like(mpoints)
cov=np.zeros_like(mpoints)
errs=np.zeros_like(mpoints)
i=-1
for mpoint in mpoints:
    i+=1
    s_sr_mass=getmasspoint(mpoint)
    mc_s_sr = ROOT.RooDataHist("s_sr","s_sr", ROOT.RooArgSet(mass),s_sr_mass)
    pdf_s_sr= ROOT.RooHistPdf("signal", "signal", ROOT.RooArgSet(mass), mc_s_sr)

    c = ROOT.RooRealVar("C", "Coeff Signal", 0.00158*lum, -10*lum, 10*lum)
    signal_ext= ROOT.RooExtendPdf("ext_signal", "C*signal", pdf_s_sr, c)

    model_sr = ROOT.RooAddPdf("model_sr_1", "A * exp(-alpha1 * m_lljj) + B * exp(-alpha2 * m_lljj) + C * signal", ROOT.RooArgList(exp1_ext,exp2_ext,signal_ext))
    fres=model_sr.fitTo(mc_sr, ROOT.RooFit.Save(True))

    cs[i]=c.getVal()
    cov[i]=fres.covarianceMatrix()[0][0]
    errs[i]=c.getError()

plt.plot(mpoints,cs)
maxc=np.max(cs)
maxx=mpoints[np.argmax(cs)]
plt.plot([maxx,maxx],[0,maxc],'--',label=f'{maxx}')
plt.xlabel("WR Mass")
plt.ylabel("Fit Const")
plt.legend()
plt.savefig("coeffs.pdf", format="pdf")
plt.close()

plt.plot(mpoints,errs)
plt.xlabel("WR Mass")
plt.ylabel("Fitting Error")
plt.savefig("errors.pdf", format="pdf")
plt.close()

pererr=np.abs(np.divide(errs,cs))
plt.plot(mpoints,pererr)
minerr=np.min(pererr)
minx=mpoints[np.argmin(pererr)]
plt.plot([minx,minx],[0,minerr],'--',label=f'{minx}')
plt.xlabel("WR Mass")
plt.ylabel("Fitting Error/Fit Const")
plt.ylim([0,2])
plt.legend()
plt.savefig("pererrors.pdf", format="pdf")
plt.close()


plt.plot(mpoints,cov)
plt.xlabel("WR Mass")
plt.ylabel("Covariance Element")
plt.savefig("covr.pdf", format="pdf")
plt.close()

fitResult.floatParsFinal().Print("v")
fitResult.covarianceMatrix().Print()
fitResult.correlationMatrix().Print()