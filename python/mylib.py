import os,ROOT
import math
from array import array
import subprocess
import tempfile
import numpy as np

def add_histograms(bkgd_combined_hist, sample_hist):
    if bkgd_combined_hist is None:
        bkgd_combined_hist = sample_hist.Clone("bkgd_combined")
        bkgd_combined_hist.SetDirectory(0)  # Keep in memory
    else:
        bkgd_combined_hist.Add(sample_hist)  # Add to combined background histogram
    return bkgd_combined_hist

def divide_histograms(data_hist, bkgd_hist):
    ratio_hist = data_hist.Clone("ratio_hist")
    ratio_hist.Divide(bkgd_hist)
    return ratio_hist

def total_lumi(data_year):
  if data_year==2016:
    return 35.92
  if data_year==2017:
    return 41.53
  if data_year==2018:
    return 59.74
  if data_year<0:
    return 138
  else:
    print(f"[mylib.py, TotalLumi()] Wrong DataYear : {DataYear}")
    return None

# TO-DO
def make_overflow_bin(hist, xlim):
    # Apply x-axis range according to xlim
    hist.GetXaxis().SetRangeUser(xlim[0], xlim[1])

    # Original bin info
    n_bin_origin = hist.GetXaxis().GetNbins()
    bin_first = hist.GetXaxis

    # Get the edges of the first and last bins within the specified range
    x_first_lowedge = hist.GetXaxis().GetBinLowEdge(bin_first)
    x_last_upedge = hist.GetXaxis().GetBinUpEdge(bin_last)

    # Calculate all underflow contents and errors
    all_underflows = hist.Integral(0, bin_first - 1)
    all_underflows_error = sum(hist.GetBinError(i)**2 for i in range(0, bin_first))**0.5

    # Calculate all overflow contents and errors
    all_overflows = hist.Integral(bin_last + 1, n_bin_origin + 1)
    all_overflows_error = sum(hist.GetBinError(i)**2 for i in range(bin_last + 1, n_bin_origin + 2))**0.5

    # Define new x-axis bin edges within the specified range
    x_bins = [hist.GetXaxis().GetBinLowEdge(i) for i in range(bin_first, bin_last + 1)]
    x_bins.append(x_last_upedge)

    # Create a new histogram with the adjusted x-axis bins
    hist_out = ROOT.TH1D(hist.GetName() + "_with_overflow", hist.GetTitle(), len(x_bins) - 1, array("d", x_bins))

    # Fill the new histogram with original bin contents and errors
    for i in range(1, len(x_bins)):
        content = hist.GetBinContent(bin_first - 1 + i)
        error = hist.GetBinError(bin_first - 1 + i)

        # Add underflow to the first bin
        if i == 1:
            content += all_underflows
            error = math.sqrt(error**2 + all_underflows_error**2)

        # Add overflow to the last bin
        if i == len(x_bins) - 1:
            content += all_overflows
            error = math.sqrt(error**2 + all_overflows_error**2)

        hist_out.SetBinContent(i, content)
        hist_out.SetBinError(i, error)

    # Rename the histogram to match the original name (optional)
    hist_out.SetName(hist.GetName())
    hist_out.SetDirectory(0)  # Detach from any file to keep it in memory

    return hist_out

def RebinWRMass(hist):
    variable_bins = [800, 1000, 1200, 1400, 1600, 2000, 2400, 2800, 3200, 8000]
    hist = hist.Rebin(len(variable_bins)-1, hist.GetName(), array("d", variable_bins))

    # Divide bin contents and errors by bin width
    for bin_idx in range(1, hist.GetNbinsX() + 1):  # ROOT bins are 1-indexed
        bin_content = hist.GetBinContent(bin_idx)
        bin_error = hist.GetBinError(bin_idx)
        bin_width = hist.GetBinWidth(bin_idx)
        
        # Update bin content and error
        hist.SetBinContent(bin_idx, bin_content / bin_width)
        hist.SetBinError(bin_idx, bin_error / bin_width)

    return hist
#   return ChangeGeVToTeVXaxis(hist):

def RebinJetPt(hist):
    variable_bins = [0, 40, 100, 200, 400, 600, 800, 1000, 1500, 2000]
    hist = hist.Rebin(len(variable_bins)-1, hist.GetName(), array("d", variable_bins))

    # Divide bin contents and errors by bin width
    for bin_idx in range(1, hist.GetNbinsX() + 1):
        bin_content = hist.GetBinContent(bin_idx)
        bin_error = hist.GetBinError(bin_idx)
        bin_width = hist.GetBinWidth(bin_idx)

        # Update bin content and error
        hist.SetBinContent(bin_idx, bin_content / bin_width)
        hist.SetBinError(bin_idx, bin_error / bin_width)

    return hist

# TO-DO
def ChangeGeVToTeVXaxis(hist):
    # Convert bin edges from GeV to TeV
    x_new = [hist.GetXaxis().GetBinLowEdge(1) / 1000.0]  # Initial low edge in TeV
    x_new += [hist.GetXaxis().GetBinUpEdge(i) / 1000.0 for i in range(1, hist.GetXaxis().GetNbins() + 1)]

    # Create a new histogram with updated TeV bin edges
    hist_new_name = hist.GetName() + "_TeV"
    hist_new = ROOT.TH1D(hist_new_name, hist.GetTitle(), len(x_new) - 1, array("d", x_new))

    # Copy bin contents and errors to the new histogram
    for i in range(1, hist.GetNbinsX() + 1):
        hist_new.SetBinContent(i, hist.GetBinContent(i))
        hist_new.SetBinError(i, hist.GetBinError(i))

    return hist_new

def get_data(hist):
    n_bins = hist.GetNbinsX()
    y_bins, y_errors, x_bins = [], [], []

    for i in range(1, n_bins + 1):
        bin_low_edge = hist.GetBinLowEdge(i)
        bin_up_edge = hist.GetBinLowEdge(i + 1)
        content, error = hist.GetBinContent(i), hist.GetBinError(i)

        y_bins.append(content)
        y_errors.append(error)

        if i == 1 or bin_low_edge != x_bins[-1]:  # Figure out exactly what this does
            x_bins.append(bin_low_edge)

    x_bins.append(hist.GetBinLowEdge(n_bins + 1)) # Add the final upper edge of the last bin
    return np.array(y_bins), np.array(y_errors), np.array(x_bins)

def custom_log_formatter(y, pos):
    if y == 1:
        return '1'
    elif y == 10:
        return '10'
    else:
        return f"$10^{{{int(np.log10(y))}}}$"

def save_and_upload_plot(fig, eos_path):
    # Ensure EOS directory exists
    eos_dir = os.path.dirname(eos_path)
    try:
        subprocess.run(["xrdfs", "eosuser.cern.ch", "mkdir", "-p", eos_dir], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to create directory on EOS: {e}")
        return  # Exit function if directory creation fails

    # Save plot to a temporary PDF file and upload it to EOS
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp_file:
        fig.savefig(tmp_file.name, format="pdf")  # Save the plot as a PDF
        tmp_file.flush()  # Ensure all data is written to disk

        # Upload the temporary PDF file to EOS
        try:
            subprocess.run(["xrdcp", "-f", tmp_file.name, f"root://eosuser.cern.ch/{eos_path}"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to upload file to EOS: {e}")
