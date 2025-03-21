import argparse
import os
import sys
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python')))

from Plotter import SampleGroup, Variable, Region, Plotter
import ROOT 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mplhep as hep
import matplotlib.ticker as mticker
import numpy as np
import mylib
import subprocess
import tempfile
import copy
from pathlib import Path
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser(description="Run2 vs Run3 Histogram Comparison Script")
parser.add_argument('--umn', action='store_true', default=False,help="Enable UMN-specific paths and configurations. Default: False")
parser.add_argument("--exc", action="store_true", help="Exclusively 2 & 3 jets")
args = parser.parse_args()

def load_histogram(file, region_name, variable_name, n_rebin, lum=None):
    """Load and rebin a histogram, with optional luminosity scaling."""
    hist_path = f"{region_name}/{variable_name}_{region_name}"
    hist = file.Get(hist_path)

    if not hist:
        logging.error(f"Histogram '{hist_path}' not found in '{file.GetName()}'")
        return None

    if lum is not None:
        hist.Scale(lum * 1000)

    integral = hist.Integral()
    if integral > 0: hist.Scale(1.0 / integral)

    if variable_name=='WRCand_Mass':
        return mylib.RebinWRMass(hist)
    elif variable_name=='Jet_0_Pt':
        return mylib.RebinJetPt(hist)
    elif n_rebin > 0:
        hist.Rebin(n_rebin)
    return hist
    
def gauss(x, A, x0, sigma1, sigma2): 
    return A * np.exp(-(x - x0) ** 2 / (2 * np.where((x-x0)>0,sigma1,sigma2) ** 2))

def draw_histogram(run2_hist, run3_hist, ratio_hist, sample, process, region, variable, xlim, ylim, dofit):
    """Draw comparison and ratio histograms between Run2 and Run3 data."""
    fig, (ax, ax_ratio) = plt.subplots(
            nrows=2, sharex = 'col',
            gridspec_kw= {"height_ratios": [5, 1], "hspace": 0.07},
            figsize=(10, 10)
    )

    run2_content, run2_errors, x_bins = mylib.get_data(run2_hist)
    run3_content, run3_errors, _ = mylib.get_data(run3_hist)
    
    hep.histplot(run2_content, bins=x_bins, yerr=run2_errors, label=sample.tlatex_alias[0], color=sample.color[0], histtype='step', ax=ax,  linewidth=1)
    hep.histplot(run3_content, bins=x_bins, yerr=run3_errors, label=sample.tlatex_alias[1], color=sample.color[1], histtype='step', ax=ax, linewidth=1)

    # Plot the ratio
    bin_centers = 0.5 * (x_bins[1:] + x_bins[:-1])
    ratio_contents, ratio_errors, _ = mylib.get_data(ratio_hist)
    nonzero_mask = ratio_contents != 0
    ax_ratio.errorbar(
        bin_centers[nonzero_mask], ratio_contents[nonzero_mask],
        yerr=ratio_errors[nonzero_mask], fmt='o', linewidth=2, capsize=2, color='black'
    )

    # Set ratio limits and labels
    ax_ratio.set_xlim(*xlim)
    ax_ratio.set_xlabel(f"{variable.tlatex_alias} {f'[{variable.unit}]' if variable.unit else f'{variable.unit}'}")
    ax_ratio.set_ylabel(r"$\frac{3jet}{2jet}$")
    ax_ratio.set_yticks([0, 1, 2, 3])
    ax_ratio.set_ylim(0, 3)
    ax_ratio.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2g'))


    # Set y-axis scale and labels for the main plot
    ax.set_yscale("log" if region.logy > 0 else "linear")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(mylib.custom_log_formatter))
    ax.set_ylim(*ylim)
    bin_widths = [round(x_bins[i + 1] - x_bins[i], 1) for i in range(len(x_bins) - 1)]
    if all(width == bin_widths[0] for width in bin_widths):
        bin_width = bin_widths[0]
        formatted_bin_width = int(bin_width) if bin_width.is_integer() else f"{bin_width:.1f}"
        ax.set_ylabel(f"Events / {formatted_bin_width} {variable.unit}")
    else:
        ax.set_ylabel(f"Events / X {variable.unit}")
    
    if dofit:
        par2,_= curve_fit(gauss, bin_centers, run2_content,[0.1,3000,200,200])
        par3,_= curve_fit(gauss, bin_centers, run3_content,[0.1,3000,200,200])
        
        xf=np.linspace(bin_centers[0],bin_centers[-1],1000)
        yf1= gauss(xf,par2[0],par2[1],par2[2],par2[3])
        yf2= gauss(xf,par3[0],par3[1],par3[2],par3[3])
        
        ax.plot(xf,yf1,alpha=0.5,label=r'$\sigma_L={},\sigma_R={}$'.format(np.round(par2[3]*10)/10,np.round(par2[2]*10)/10))
        ax.plot(xf,yf2,alpha=0.5,label=r'$\sigma_L={},\sigma_R={}$'.format(np.round(par3[3]*10)/10,np.round(par3[2]*10)/10))
    
    # Plot region information and CMS label
    ax.text(0.05, 0.96, region.tlatex_alias, transform=ax.transAxes,fontsize=20, verticalalignment='top')
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress", fontsize=22)
    ax.legend(fontsize=20)
    
    exclu=""
    if args.exc:
        exclu="_exclusive"
    
    # Save and upload plot
    if not args.umn:
        file_path = f"/eos/user/w/wijackso/plots/N3000_vs_N800/{sample.name}/{process}/{region.name}{exclu}/{variable.name}_{region.name}.pdf"
    else:
        file_path = f"plots/{process}/{region.name}{exclu}/{variable.name}_{region.name}.pdf"
    mylib.save_and_upload_plot(fig, file_path, args.umn)
    plt.close(fig)

def main():
    plotter = Plotter()

    plotter.sample_groups = [
#        SampleGroup('signals', 'Run2Legacy', ['2018','2018_3jets'], ['#5790fc', '#f89c20'], [r'2 jet', r'3 jet'], 'WR3200_N3000',),
        SampleGroup('DYJets', ['Run2UltraLegacy'], ['2018','2018_3jets'], ['#5790fc', '#f89c20'], [r'$DY+Jets$ (2 Jet)', r'$DY+Jets$ (3 Jet)'], ['DYJets'],),
    ]
    plotter.print_samples()

    plotter.regions_to_draw = [
        Region('WR_EE_Resolved_SR', 'EGamma', unblind_data=True, logy=1, tlatex_alias='ee\nResolved SR'),
        Region('WR_MuMu_Resolved_SR', 'SingleMuon', unblind_data=True, logy=1, tlatex_alias='$\mu\mu$\nResolved SR'),
    ]

    plotter.print_regions()

    plotter.variables_to_draw = [
        #Variable('Lepton_0_Pt', r'$p_{T}$ of the leading lepton', 'GeV'),
        #Variable('Lepton_0_Eta', r'$\eta$ of the leading lepton', ''),
        #Variable('Lepton_0_Phi', r'$\phi$ of the leading lepton', ''),
        #Variable('Lepton_1_Pt', r'$p_{T}$ of the subleading lepton', 'GeV'),
        #Variable('Lepton_1_Eta', r'$\eta$ of the subleading lepton', ''),
        #Variable('Lepton_1_Phi', r'$\phi$ of the subleading lepton', ''),
        #Variable('Jet_0_Pt', r'$p_{T}$ of the leading jet', 'GeV'),
        #Variable('Jet_0_Eta', r'$\eta$ of the leading jet', ''),
        #Variable('Jet_0_Phi', r'$\phi$ of the leading jet', ''),
        #Variable('Jet_1_Pt', r'$p_{T}$ of the subleading jet', 'GeV'),
        #Variable('Jet_1_Eta', r'$\eta$ of the subleading jet', ''),
        #Variable('Jet_1_Phi', r'$\phi$ of the subleading jet', ''),
        #Variable('ZCand_Mass', r'$m_{ll}$', 'GeV'),where
        #Variable('ZCand_Pt', r'$p^{T}_{ll}$', 'GeV'),
        #Variable('Dijet_Mass', r'$m_{jj}$', 'GeV'),
        #Variable('Dijet_Pt', r'$p^{T}_{jj}$', 'GeV'),
        Variable('NCand_Lepton_0_Mass', r'$m_{l_{Lead}jj}$', 'GeV'),
        #Variable('NCand_Lepton_0_Pt', r'$p^{T}_{l_{Lead}jj}$', 'GeV'),
        #Variable('NCand_Lepton_1_Mass', r'$m_{l_{Sublead}jj}$', 'GeV'),
        #Variable('NCand_Lepton_1_Pt', r'$p^{T}_{l_{Sublead}jj}$', 'GeV'),
        #Variable('WRCand_Mass', r'$m_{lljj}$', 'GeV'),
        #Variable('WRCand_Pt', r'$p^{T}_{lljj}$', 'GeV'),
    ]

    plotter.print_variables()

    rebin_filepath = Path('data/241215_N3000_vs_N800/SR_rebins.txt') 
    xaxis_filepath = Path('data/241215_N3000_vs_N800/SR_xaxis.txt')
    yaxis_filepath = Path('data/241215_N3000_vs_N800/SR_yaxis.txt')

    try:
        plotter.set_binning_filepath(str(rebin_filepath), str(xaxis_filepath), str(yaxis_filepath))
    except FileNotFoundError as e:
        logging.error(f"File path error: {e}")
    
    exclu=""
    if args.exc:
        exclu="_exclusive"

    for region in plotter.regions_to_draw:
        rebins, xaxis_ranges, yaxis_ranges = plotter.read_binning_info(region.name)
        """for variable in plotter.variables_to_draw:
            hists = {}
            n_rebin, xlim, ylim = rebins[variable.name], xaxis_ranges[variable.name], yaxis_ranges[variable.name]
            for sample in plotter.sample_groups:
                for year in sample.year:
                    process='DYJets'
                    file_path = Path(f"rootfiles/{sample.mc_campaign}/Regions/{year}{exclu}/WRAnalyzer_SkimTree_LRSMHighPt_{process}.root")
                    if not file_path.exists():
                        logging.warning(f"File {file_path} does not exist.")
                        continue
                    with ROOT.TFile.Open(str(file_path)) as file_run:
                        lumi = 59.74
                        hist = load_histogram(file_run, region.name, variable.name, n_rebin, lumi)
                        if hist:
                            hists[year] = copy.deepcopy(hist.Clone(f"{variable.name}_{region.name}_clone"))

                hist_N3000, hist_N800 = hists['2018'], hists['2018_3jets']
                hist_ratio = mylib.divide_histograms(hist_N800, hist_N3000)
                draw_histogram(hist_N3000, hist_N800, hist_ratio, sample, process, region, variable, xlim, ylim, variable.name=="NCand_Lepton_0_Mass")"""
        process='DYJets'
        file_path = Path(f"rootfiles/Run2UltraLegacy/Regions/2018_3jets{exclu}/WRAnalyzer_SkimTree_LRSMHighPt_DYJets.root")
            
        with ROOT.TFile.Open(str(file_path)) as file_run:
            lumi = 59.74
            hist = load_histogram(file_run, region.name , 'Delta_r20', -1, lumi)
            run2_content, run2_errors, x_bins = mylib.get_data(hist)
            fig, ax = plt.subplots()
            hep.histplot(run2_content, bins=x_bins, yerr=run2_errors, color='#5790fc', histtype='step', ax=ax,  linewidth=1)
            ax.set_yscale("log" if region.logy > 0 else "linear")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(mylib.custom_log_formatter))
            #ax.set_ylim(*ylim)
            ax.set_ylabel(f"Events / 0.0875")
            ax.set_xlabel(r"Delta_R02")
            
            ax.text(0.05, 0.96, region.tlatex_alias, transform=ax.transAxes,fontsize=20, verticalalignment='top')
            hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress", fontsize=22)
            #ax.legend(fontsize=20)
            mylib.save_and_upload_plot(fig, f"plots/{process}/{region.name}{exclu}/Del_R02_{region.name}.pdf", args.umn)
            plt.close(fig)
            
        with ROOT.TFile.Open(str(file_path)) as file_run:
            lumi = 59.74
            hist = load_histogram(file_run, region.name , 'Delta_r21', -1, lumi)
            run2_content, run2_errors, x_bins = mylib.get_data(hist)
            fig, ax = plt.subplots()
            hep.histplot(run2_content, bins=x_bins, yerr=run2_errors, color='#5790fc', histtype='step', ax=ax,  linewidth=1)
            ax.set_yscale("log" if region.logy > 0 else "linear")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(mylib.custom_log_formatter))
            #ax.set_ylim(*ylim)
            ax.set_ylabel(f"Events / 0.0875")
            ax.set_xlabel(r"Delta_R12$")
            
            ax.text(0.05, 0.96, region.tlatex_alias, transform=ax.transAxes,fontsize=20, verticalalignment='top')
            hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress", fontsize=22)
            #ax.legend(fontsize=20)
            mylib.save_and_upload_plot(fig, f"plots/{process}/{region.name}{exclu}/Del_R12_{region.name}.pdf", args.umn)
            plt.close(fig)
        
        n4_values=np.array([])
        n4_errors=np.array([])
        with ROOT.TFile.Open(str(file_path)) as file_run:
            lumi = 59.74
            hist = load_histogram(file_run, region.name , 'WRMass4_DeltaR', -1, lumi)
            nxbin=hist.GetNbinsX()
            nybin=hist.GetNbinsY()
            y_bins,  x_bins = np.zeros(nybin+1), np.zeros(nxbin+1)
            n_values = np.zeros((nxbin,nybin))
            n4_values= np.zeros((nxbin,nybin))
            n4_errors= np.zeros((nxbin,nybin))
            
            for i in range(1,nxbin+1):
                for j in range(1,nybin+1):
                    content = hist.GetBinContent(i,j)
                    n_values[i-1,j-1]= np.log10(np.where(content>1e-4,content,1e-4))
                    n4_values[i-1,j-1]= content
                    n4_errors[i-1,j-1]= hist.GetBinError(i,j)
                    
                    if i==1:
                        yax = hist.GetYaxis()
                        if j==1:
                            y_bins[0]=yax.GetBinLowEdge(j)
                        y_bins[j]=yax.GetBinLowEdge(j+1)
                xax = hist.GetXaxis()
                if i == 1:
                    x_bins[0]=xax.GetBinLowEdge(i)
                x_bins[i]=xax.GetBinLowEdge(i+1)
            fig, ax = plt.subplots()
            hep.hist2dplot(n_values,xbins=x_bins,ybins=y_bins,ax=ax)
            ax.set_ylim(0, 5)
            ax.set_xlim(0, 7000)
            ax.set_xlabel(r"$m_{lljj}$ [GeV]")
            ax.set_ylabel(r"$\Delta R_{min}$")
            ax.text(0.05, 0.96, region.tlatex_alias, transform=ax.transAxes,fontsize=20, verticalalignment='top',color='white')
            hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress", fontsize=22)
            mylib.save_and_upload_plot(fig, f"plots/{process}/{region.name}{exclu}/2d4obj_{region.name}.pdf", args.umn)
            plt.close(fig)
            
        with ROOT.TFile.Open(str(file_path)) as file_run:
            lumi = 59.74
            hist = load_histogram(file_run, region.name , 'WRMass5_DeltaR', -1, lumi)
            nxbin=hist.GetNbinsX()
            nybin=hist.GetNbinsY()
            y_bins,  x_bins, sl, sr, s4l, s4r = np.zeros(nybin+1), np.zeros(nxbin+1), np.zeros(nybin+1), np.zeros(nybin+1), np.zeros(nybin+1), np.zeros(nybin+1)
            sle, sre, s4le, s4re = np.zeros(nybin+1), np.zeros(nybin+1), np.zeros(nybin+1), np.zeros(nybin+1)
            n_values = np.zeros((nxbin,nybin))
            n_errors = np.zeros((nxbin,nybin))
            
            for j in range(1,nybin+1):
                fitsuccess = False
                fitsuccess4= False
                
                yax = hist.GetYaxis()
                if j == 1:
                    y_bins[0]=yax.GetBinLowEdge(j)
                y_bins[j]=yax.GetBinLowEdge(j+1)
                
                for i in range(1,nxbin+1):
                    content = hist.GetBinContent(i,j)
                    n_values[i-1,j-1]= content
                    n_errors[i-1,j-1]= hist.GetBinError(i,j)
                    
                    if j==1:
                        xax = hist.GetXaxis()
                        if i==1:
                            x_bins[0]=xax.GetBinLowEdge(i)
                        x_bins[i]=xax.GetBinLowEdge(i+1)
                
                x_val=(x_bins[1:]+x_bins[:-1])/2
                
                np.seterr(divide='raise',invalid='raise')
                
                n_errors=np.where(n_errors==0,np.inf,n_errors)
                n4_errors=np.where(n4_errors==0,np.inf,n4_errors)
                
                try:
                    A=np.max(n_values[:,j-1])
                    argx0=np.argmax(n_values[:,j-1])
                    x0=x_val[argx0]
                    sigL=np.sqrt(np.sum(np.multiply(n_values[:argx0,j-1],(x_val[:argx0]-x0)**2))/np.sum(n_values[:argx0,j-1]))
                    sigR=np.sqrt(np.sum(np.multiply(n_values[argx0:,j-1],(x_val[argx0:]-x0)**2))/np.sum(n_values[argx0:,j-1]))
                    
                    par,pcov = curve_fit(gauss, x_val, n_values[:,j-1],[A,x0,sigR,sigL],sigma=n_errors[:,j-1])
                    sr[j-1], sl[j-1]= np.abs(par[2]), np.abs(par[3])
                    errors = np.sqrt(np.diag(pcov))
                    sre[j-1], sle[j-1]= errors[2], errors[3]
                    fitsuccess = True
                except:
                    sr[j-1]=sl[j-1]=0
                    fitsuccess = False
                
                try:
                    A=np.max(n4_values[:,j-1])
                    argx0=np.argmax(n4_values[:,j-1])
                    x0=x_val[argx0]
                    sigL=np.sqrt(np.sum(np.multiply(n4_values[:argx0,j-1],(x_val[:argx0]-x0)**2))/np.sum(n4_values[:argx0,j-1]))
                    sigR=np.sqrt(np.sum(np.multiply(n4_values[argx0:,j-1],(x_val[argx0:]-x0)**2))/np.sum(n4_values[argx0:,j-1]))
                    
                    par4,p4cov = curve_fit(gauss, x_val, n4_values[:,j-1],[A,x0,sigR,sigL],sigma=n4_errors[:,j-1])
                    s4r[j-1], s4l[j-1]= np.abs(par4[2]), np.abs(par4[3])
                    errors4 = np.sqrt(np.diag(p4cov))
                    s4re[j-1], s4le[j-1]= errors4[2], errors4[3]
                    fitsuccess4 = True
                except:
                    s4r[j-1]=s4l[j-1]=0
                    fitsuccess4 = False
                
                np.seterr(divide='warn',invalid='warn')
            
            for i in range(0,nxbin):
                for j in range(0,nybin):
                    n_values[i,j]=np.log10(np.where(n_values[i,j]>1e-4,n_values[i,j],1e-4))
            
            fig, ax = plt.subplots()
            hep.hist2dplot(n_values,xbins=x_bins,ybins=y_bins,ax=ax)
            ax.set_ylim(0, 5)
            ax.set_xlim(0, 7000)
            ax.set_xlabel(r"$m_{lljjj}$ [GeV]")
            ax.set_ylabel(r"$\Delta R_{min}$")
            ax.text(0.05, 0.96, region.tlatex_alias, transform=ax.transAxes,fontsize=20, verticalalignment='top',color='white')
            hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress", fontsize=22)
            mylib.save_and_upload_plot(fig, f"plots/{process}/{region.name}{exclu}/2d5obj_{region.name}.pdf", args.umn)
            plt.close(fig)
            
            fig, ax = plt.subplots()
            ax.set_ylim(0.3,3.3)
            ax.set_xlim(-1250,1250)
            ax.set_xlabel(r"$\sigma_L$ and $\sigma_R$ [GeV]")
            ax.set_ylabel(r"$\Delta R_{min}$")
            
            for i in range(0,nybin):
                bin_width = y_bins[i+1]-y_bins[i]
                insets=0.1
                insets4=0.01
                
                x_cor_l, y_cor_l, w_l, h_l=-sl[i], y_bins[i]+insets*bin_width, sl[i], (0.5-insets-insets4)*bin_width
                x_cor_r, y_cor_r, w_r, h_r= 0, y_bins[i]+insets*bin_width, sr[i], (0.5-insets-insets4)*bin_width
                x4_cor_l, y4_cor_l, w4_l, h4_l=-s4l[i], y_bins[i]+(0.5+insets4)*bin_width, s4l[i], (0.5-insets-insets4)*bin_width
                x4_cor_r, y4_cor_r, w4_r, h4_r= 0, y_bins[i]+(0.5+insets4)*bin_width, s4r[i], (0.5-insets-insets4)*bin_width
                
                rectl= patches.Rectangle((x_cor_l,y_cor_l),w_l,h_l,facecolor='blue',label=r'$\sigma_L$ 5 object' if i==0 else '')
                rectr= patches.Rectangle((x_cor_r,y_cor_r),w_r,h_r,facecolor='red',label=r'$\sigma_R$ 5 object' if i==0 else '')
                rect4l= patches.Rectangle((x4_cor_l,y4_cor_l),w4_l,h4_l,facecolor='orange',label=r'$\sigma_L$ 4 object' if i==0 else '')
                rect4r= patches.Rectangle((x4_cor_r,y4_cor_r),w4_r,h4_r,facecolor='green',label=r'$\sigma_R$ 4 object' if i==0 else '')
                
                ax.add_patch(rectl)
                ax.add_patch(rectr)
                ax.add_patch(rect4l)
                ax.add_patch(rect4r)
            
            ax.errorbar(-sl,y_bins+(0.25+(insets-insets4)/2)*bin_width,xerr=sle,linestyle='none',color='black',capsize=5)
            ax.errorbar(sr,y_bins+(0.25+(insets-insets4)/2)*bin_width,xerr=sre,linestyle='none',color='black',capsize=5)
            ax.errorbar(-s4l,y_bins+(0.75+(insets4-insets)/2)*bin_width,xerr=s4le,linestyle='none',color='black',capsize=5)
            ax.errorbar(s4r,y_bins+(0.75+(insets4-insets)/2)*bin_width,xerr=s4re,linestyle='none',color='black',capsize=5)
            
            ax.axvline(x=0, color='k')
            
            xticks=np.arange(-1200,1201,300)
            xlabel=[f'{x}' for x in np.abs(xticks)]
            ax.set_xticks(xticks, labels=xlabel)
            ax.text(0.05, 0.96, region.tlatex_alias, transform=ax.transAxes,fontsize=20, verticalalignment='top')
            ax.legend(fontsize=20)
            hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress", fontsize=22)
            mylib.save_and_upload_plot(fig, f"plots/{process}/{region.name}{exclu}/sigma_{region.name}.pdf", args.umn)
            plt.close(fig)
        
if __name__ == "__main__":
    main()
