import argparse
import os
import sys
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../python')))

from Plotter import SampleGroup, Variable, Region, Plotter
import ROOT
import matplotlib
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
from scipy.interpolate import pchip_interpolate

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



    #if variable_name=='WRCand_Mass' and n_rebin > 0:
    #    return mylib.RebinWRMass(hist)
    #elif variable_name=='Jet_0_Pt' and n_rebin > 0:
    #    return mylib.RebinJetPt(hist)
    if n_rebin > 0:
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
    
    mylib.save_and_upload_plot(fig, file_path, args.umn)
    plt.close(fig)

def make2Dplot(n_values,x_bins,y_bins,xlabel,ylabel,plotname,mass,region):
    logvals = np.log10(np.where(n_values>1e-5,n_values,1e-5))   
    fig, ax = plt.subplots()
    hep.hist2dplot(logvals,xbins=x_bins,ybins=y_bins,ax=ax)
    ax.set_ylim(np.min(y_bins), np.max(y_bins))
    ax.set_xlim(np.min(x_bins), np.max(x_bins))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(0.05, 0.96, region.tlatex_alias, transform=ax.transAxes,fontsize=20, verticalalignment='top',color='white')
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress", fontsize=22)
    
    exclu=""
    if args.exc:
        exclu="_exclusive"
    
    mylib.save_and_upload_plot(fig, f"plots/{mass}/{region.name}{exclu}/{plotname}_{region.name}.pdf", args.umn)
    plt.close(fig)

def makeSigmaFits(n_values,n_err,x_bins):

    x_val=(x_bins[1:]+x_bins[:-1])/2
    nybin=n_values.shape[1]
    sl, sr  = np.zeros(nybin+1), np.zeros(nybin+1)
    sle, sre= np.zeros(nybin+1), np.zeros(nybin+1)
    
    np.seterr(divide='raise',invalid='raise')
    n_errors=np.where(n_err==0,np.inf,n_err)
    for j in range(1,nybin+1):
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
    np.seterr(divide='warn',invalid='warn')
    
    return sl,sr,sle,sre
    
def makeSigmaPlots(n_non_values,n_non_errors,n4_non_values,n4_non_errors,x_bins,y_bins,xlim,ylim,xlabel,ylabel,varlab1,varlab2,plotname,mass,region):
    
    n_values,n_errors=IntegralDist(n_non_values,n_non_errors)
    n4_values,n4_errors=IntegralDist(n4_non_values,n4_non_errors)
    
    nxbin=n_values.shape[0]
    nybin=n_values.shape[1]
    sl, sr, sle, sre     = makeSigmaFits(n_values, n_errors, x_bins)
    s4l, s4r, s4le, s4re = makeSigmaFits(n4_values,n4_errors,x_bins)
                    
    fig, ax = plt.subplots()
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
            
    for i in range(0,nybin):
        bin_width = y_bins[i+1]-y_bins[i]
        insets=0.1
        insets4=0.01
                
        x_cor_l, y_cor_l, w_l, h_l=-sl[i], y_bins[i]+insets*bin_width, sl[i], (0.5-insets-insets4)*bin_width
        x_cor_r, y_cor_r, w_r, h_r= 0, y_bins[i]+insets*bin_width, sr[i], (0.5-insets-insets4)*bin_width
        x4_cor_l, y4_cor_l, w4_l, h4_l=-s4l[i], y_bins[i]+(0.5+insets4)*bin_width, s4l[i], (0.5-insets-insets4)*bin_width
        x4_cor_r, y4_cor_r, w4_r, h4_r= 0, y_bins[i]+(0.5+insets4)*bin_width, s4r[i], (0.5-insets-insets4)*bin_width
                
        rectl= patches.Rectangle((x_cor_l,y_cor_l),w_l,h_l,facecolor='red',label=varlab1 if i==0 else '')
        rectr= patches.Rectangle((x_cor_r,y_cor_r),w_r,h_r,facecolor='red',label='')
        rect4l= patches.Rectangle((x4_cor_l,y4_cor_l),w4_l,h4_l,facecolor='blue',label=varlab2 if i==0 else '')
        rect4r= patches.Rectangle((x4_cor_r,y4_cor_r),w4_r,h4_r,facecolor='blue',label='')
                
        ax.add_patch(rectl)
        ax.add_patch(rectr)
        ax.add_patch(rect4l)
        ax.add_patch(rect4r)
            
    ax.errorbar(-sl,y_bins+(0.25+(insets-insets4)/2)*bin_width,xerr=sle,linestyle='none',color='black',capsize=5)
    ax.errorbar(sr,y_bins+(0.25+(insets-insets4)/2)*bin_width,xerr=sre,linestyle='none',color='black',capsize=5)
    ax.errorbar(-s4l,y_bins+(0.75+(insets4-insets)/2)*bin_width,xerr=s4le,linestyle='none',color='black',capsize=5)
    ax.errorbar(s4r,y_bins+(0.75+(insets4-insets)/2)*bin_width,xerr=s4re,linestyle='none',color='black',capsize=5)
    
    xticks,_=plt.xticks()
    xticks=xticks[1:-1]
    ax.axvline(x=0, color='k')
    xlabel=[f'{x}' for x in np.abs(xticks)]
    ax.set_xticks(xticks, labels=xlabel)
    ax.text(0.05, 0.96, region.tlatex_alias, transform=ax.transAxes,fontsize=20, verticalalignment='top')
    ax.legend(fontsize=20)
    hep.cms.label(loc=0, ax=ax, data=False, label="Work in Progress", fontsize=22)
    
    exclu=""
    if args.exc:
        exclu="_exclusive"
    
    mylib.save_and_upload_plot(fig, f"plots/{mass}/{region.name}{exclu}/{plotname}_{region.name}.pdf", args.umn)
    plt.close(fig)
    return sl, sr, sle, sre, s4l, s4r, s4le, s4re

def IntegralDist(n_values,n_errors):
    n_int,nerr_int=np.copy(n_values),np.copy(n_errors)
    nybin=n_values.shape[1]
    for j in range(2,nybin+1):
        n_int[:,j-1]+=n_int[:,j-2]
        nerr_int[:,j-1]=np.sqrt(nerr_int[:,j-2]**2+nerr_int[:,j-1]**2)
    return n_int,nerr_int

def comparesigmas(sl, sr, sle, sre, s4l, s4r, s4le, s4re, y_bins, mass, region, plotname):
    
    allgood = np.where((sl>0) & (sr>0) & (s4l>0) & (s4r>0))
    ratio=(sl[allgood]+sr[allgood])/(s4l[allgood]+s4r[allgood])
    fig, ax = plt.subplots()
    ymid=(y_bins[1:]+y_bins[:-1])/2
    ax.plot(ymid[allgood],ratio)
    
    exclu=""
    if args.exc:
        exclu="_exclusive"
    
    mylib.save_and_upload_plot(fig, f"plots/{mass}/{region.name}{exclu}/{plotname}_{region.name}.pdf", args.umn)
    plt.close(fig)

def main():
    matplotlib.use('Agg')
    
    plotter = Plotter()

    mass_options = ["WR1200_N200", "WR1200_N400", "WR1200_N600", "WR1200_N800", "WR1200_N1100",
                    "WR1600_N400", "WR1600_N600", "WR1600_N800", "WR1600_N1200", "WR1600_N1500",
                    "WR2000_N400", "WR2000_N800", "WR2000_N1000", "WR2000_N1400", "WR2000_N1900",
                    "WR2400_N600", "WR2400_N800", "WR2400_N1200", "WR2400_N1800", "WR2400_N2300",
                    "WR2800_N600", "WR2800_N1000", "WR2800_N1400", "WR2800_N2000", "WR2800_N2700",
                    "WR3200_N800", "WR3200_N1200", "WR3200_N1600", "WR3200_N2400", "WR3200_N3000"]
    
    valueat1=[]
    valueat1mean=[]
    WRmasses=[1200,1600,2000,2400,2800,3200]
    Nmasses=[200,400,600,800,1100,400,600,800,1200,1500,400,800,1000,1400,1900,600,800,1200,1800,2300,600,1000,1400,2000,2700,800,1200,1600,2400,3000]

    plotter.regions_to_draw = [
        Region('WR_EE_Resolved_SR', 'EGamma', unblind_data=True, logy=1, tlatex_alias='ee\nResolved SR'),
        Region('WR_MuMu_Resolved_SR', 'SingleMuon', unblind_data=True, logy=1, tlatex_alias='$\mu\mu$\nResolved SR'),
    ]

    plotter.print_regions()

    plotter.variables_to_draw = [
        Variable('NCand_Lepton_0_Mass', r'$m_{l_{Lead}jj}$', 'GeV'),
        Variable('NCand_Lepton_0_Pt', r'$p^{T}_{l_{Lead}jj}$', 'GeV'),
        Variable('WRCand_Mass', r'$m_{lljj}$', 'GeV'),
        Variable('WRCand_Pt', r'$p^{T}_{lljj}$', 'GeV'),
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
        
        valueat1.clear()
        valueat1mean.clear()
        
        for mass in mass_options:
            
            file_path = Path(f"rootfiles/RunII/2018/RunIISummer20UL18/3jets/WRAnalyzer_signal_{mass}.root")
            file_path2= Path(f"rootfiles/RunII/2018/RunIISummer20UL18/exclusive/WRAnalyzer_signal_{mass}.root")
            
            with ROOT.TFile.Open(str(file_path)) as file_run, ROOT.TFile.Open(str(file_path2)) as file_run2:
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
                mylib.save_and_upload_plot(fig, f"plots/{mass}/{region.name}{exclu}/Del_R02_{region.name}.pdf", args.umn)
                plt.close(fig)
                
                
                
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
                mylib.save_and_upload_plot(fig, f"plots/{mass}/{region.name}{exclu}/Del_R12_{region.name}.pdf", args.umn)
                plt.close(fig)
                
                
                
                hist = load_histogram(file_run, region.name , 'WRMass4_DeltaR', -1, lumi)
                hist2 = load_histogram(file_run2, region.name , 'WRCand_Mass', 8, lumi)
                n4_values, n4_errors, x_bins, y_bins= mylib.get_2Ddata(hist,False)
                zroadd , zroadd_err , _ = mylib.get_data(hist2,False)
                n4_values[:,0] += zroadd
                n4_errors[:,0] = np.sqrt(n4_errors[:,0]**2 + zroadd_err**2)
                make2Dplot(n4_values,x_bins,y_bins,r"$m_{lljj}$ [GeV]",r"$\Delta R_{min}$",'2d4obj',mass,region)
                
                
                
                hist = load_histogram(file_run, region.name , 'WRMass4_pT', -1, lumi)
                n4_values_pT, n4_errors_pT, x_bins_pT, y_bins_pT= mylib.get_2Ddata(hist,False)
                n4_values_pT[:,0] += zroadd
                n4_errors_pT[:,0] = np.sqrt(n4_errors_pT[:,0]**2 + zroadd_err**2)
                make2Dplot(n4_values_pT,x_bins_pT,y_bins_pT,r"$m_{lljj}$ [GeV]",r"$pT_{min}$",'2d4obj_pT',mass,region)
                
                
                hist = load_histogram(file_run, region.name , 'WRMass4_sin', -1, lumi)
                n4_values_sin, n4_errors_sin, x_bins_sin, y_bins_sin= mylib.get_2Ddata(hist,False)
                n4_values_sin[:,0] += zroadd
                n4_errors_sin[:,0] = np.sqrt(n4_errors_sin[:,0]**2 + zroadd_err**2)
                make2Dplot(n4_values_sin,x_bins_sin,y_bins_sin,r"$m_{lljj}$ [GeV]",r"$sin_{min}$",'2d4obj_sin',mass,region)
                
                
                hist = load_histogram(file_run, region.name , 'WRMass5_DeltaR', -1, lumi)
                n_values, n_errors, x_bins, y_bins= mylib.get_2Ddata(hist,False)
                make2Dplot(n_values,x_bins,y_bins,r"$m_{lljjj}$ [GeV]",r"$\Delta R_{min}$",'2d5obj',mass,region)
                
                
                
                hist = load_histogram(file_run, region.name , 'WRMass5_pT', -1, lumi)
                n_values_pT, n_errors_pT, x_bins_pT, y_bins_pT= mylib.get_2Ddata(hist,False)
                make2Dplot(n_values_pT,x_bins_pT,y_bins_pT,r"$m_{lljjj}$ [GeV]",r"$pT_{min}$",'2d5obj_pT',mass,region)
                
                
                hist = load_histogram(file_run, region.name , 'WRMass5_sin', -1, lumi)
                n_values_sin, n_errors_sin, x_bins_sin, y_bins_sin= mylib.get_2Ddata(hist,False)
                make2Dplot(n_values_sin, x_bins_sin, y_bins_sin, r"$m_{lljjj}$ [GeV]", r"$sin_{min}$",'2d5obj_sin', mass, region)
            
            
            sl, sr, sle, sre, s4l, s4r, s4le, s4re = makeSigmaPlots(n_values,n_errors,n4_values,n4_errors,x_bins,y_bins,(-1000,1000),(0.3,3.3),r"$\sigma_L$ and $\sigma_R$ [GeV]",r"$\Delta R_{min}$",
            r'$\sigma$ 5 object',r'$\sigma$ 4 object','sigma_deltaR',mass,region)
            comparesigmas(sl, sr, sle, sre, s4l, s4r, s4le, s4re, y_bins, mass, region, 'CompareDelR')
            
            sl, sr, sle, sre, s4l, s4r, s4le, s4re = makeSigmaPlots(n_values_pT,n_errors_pT,n4_values_pT,n4_errors_pT,x_bins_pT,y_bins_pT,(-1000,1000),(0,150),r"$\sigma_L$ and $\sigma_R$ [GeV]",
            r"$pT_{min}$",r'$\sigma$ 5 object',r'$\sigma$ 4 object','sigma_pT',mass,region)
            comparesigmas(sl, sr, sle, sre, s4l, s4r, s4le, s4re, y_bins_pT, mass, region, 'ComparePT')
            
            sl, sr, sle, sre, s4l, s4r, s4le, s4re = makeSigmaPlots(n_values_sin, n_errors_sin, n4_values_sin, n4_errors_sin, x_bins_sin, y_bins_sin, (-1000,1000), (0,1), r"$\sigma_L$ and $\sigma_R$ [GeV]",
            r"$sin_{min}$", r'$\sigma$ 5 object', r'$\sigma$ 4 object', 'sigma_sin', mass, region)
            comparesigmas(sl, sr, sle, sre, s4l, s4r, s4le, s4re, y_bins_sin, mass, region, 'CompareSine')
            
if __name__ == "__main__":
    main()
