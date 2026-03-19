# import os
# import ROOT
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# from pathlib import Path
# ROOT.gROOT.SetBatch(True)
# # import sys

# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# base_path = "rootfiles/RunII/2018/RunIISummer20UL18/"
# signal_file = "WRAnalyzer_signal_WR3200_N3000.root"
# dy_file = "WRAnalyzer_DYJets.root"
# ttbar_file = "WRAnalyzer_TTbar.root"
# variable_path_mumu = "wr_mumu_resolved_sr/mass_fourobject_wr_mumu_resolved_sr"
# variable_path_ee = "wr_ee_resolved_sr/mass_fourobject_wr_ee_resolved_sr"

# no_btag_signal_file= ROOT.TFile(base_path + signal_file,"r")
# no_btag_signal=no_btag_signal_file.Get(variable_path_mumu)
# btag_loose_signal_file= ROOT.TFile(base_path + "btagloose/" + signal_file,"r")
# btag_loose_signal=btag_loose_signal_file.Get(variable_path_mumu)
# btag_medium_signal_file= ROOT.TFile(base_path + "btagmedium/" + signal_file,"r")
# btag_medium_signal=btag_medium_signal_file.Get(variable_path_mumu)
# btag_tight_signal_file= ROOT.TFile(base_path + "btagtight/" + signal_file,"r")
# btag_tight_signal=btag_tight_signal_file.Get(variable_path_mumu)

# no_btag_dy_file= ROOT.TFile(base_path + dy_file,"r")
# no_btag_dy=no_btag_dy_file.Get(variable_path_mumu)
# btag_loose_dy_file= ROOT.TFile(base_path + "btagloose/" + dy_file,"r")
# btag_loose_dy=btag_loose_dy_file.Get(variable_path_mumu)
# btag_medium_dy_file= ROOT.TFile(base_path + "btagmedium/" + dy_file,"r")
# btag_medium_dy=btag_medium_dy_file.Get(variable_path_mumu)
# btag_tight_dy_file= ROOT.TFile(base_path + "btagtight/" + dy_file,"r")
# btag_tight_dy=btag_tight_dy_file.Get(variable_path_mumu)

# no_btag_ttbar_file= ROOT.TFile(base_path + ttbar_file,"r")
# no_btag_ttbar=no_btag_ttbar_file.Get(variable_path_mumu)
# btag_loose_ttbar_file= ROOT.TFile(base_path + "btagloose/" + ttbar_file,"r")
# btag_loose_ttbar=btag_loose_ttbar_file.Get(variable_path_mumu)
# btag_medium_ttbar_file= ROOT.TFile(base_path + "btagmedium/" + ttbar_file,"r")
# btag_medium_ttbar=btag_medium_ttbar_file.Get(variable_path_mumu)
# btag_tight_ttbar_file= ROOT.TFile(base_path + "btagtight/" + ttbar_file,"r")
# btag_tight_ttbar=btag_tight_ttbar_file.Get(variable_path_mumu)

# lum=59
# btag_loose_signal.Scale(lum*1000)
# btag_medium_signal.Scale(lum*1000)
# btag_tight_signal.Scale(lum*1000)
# btag_loose_dy.Scale(lum*1000)
# btag_medium_dy.Scale(lum*1000)
# btag_tight_dy.Scale(lum*1000)
# btag_loose_ttbar.Scale(lum*1000)
# btag_medium_ttbar.Scale(lum*1000)
# btag_tight_ttbar.Scale(lum*1000)
# no_btag_signal.Scale(lum*1000)
# no_btag_dy.Scale(lum*1000)
# no_btag_ttbar.Scale(lum*1000)



# def _root_hist_to_arrays(h):
# 	nbins = int(h.GetNbinsX())
# 	content = np.array([h.GetBinContent(i + 1) for i in range(nbins)])
# 	errors = np.array([h.GetBinError(i + 1) for i in range(nbins)])
# 	edges = np.zeros(nbins + 1)
# 	for i in range(nbins):
# 		edges[i] = h.GetBinLowEdge(i + 1)
# 	# last edge = lowEdge(last) + width(last)
# 	edges[-1] = edges[-2] + h.GetBinWidth(nbins)
# 	return content, errors, edges


# def rebin_arrays(content, errors, edges, target_width=100.0):
# 	"""Rebin arrays so that new bin width is approximately `target_width` (GeV).

# 	Uses the original bin width to compute an integer factor to merge bins.
# 	"""
# 	if len(edges) < 2:
# 		return content, errors, edges
# 	orig_width = edges[1] - edges[0]
# 	if orig_width <= 0:
# 		return content, errors, edges
# 	factor = max(1, int(round(target_width / orig_width)))
# 	# if factor == 1, nothing to do
# 	if factor == 1:
# 		return content, errors, edges
# 	n = len(content)
# 	m = (n // factor) * factor
# 	if m == 0:
# 		return content, errors, edges
# 	c = content[:m].reshape(-1, factor).sum(axis=1)
# 	e = np.sqrt((errors[:m].reshape(-1, factor) ** 2).sum(axis=1))
# 	# edges: take every factor-th low edge and append final edge
# 	new_edges = np.concatenate([edges[0:m:factor], edges[m:m+1]])
# 	return c, e, new_edges


# def sum_hists(h1, h2, name=None):
# 	"""Return a new histogram that is the sum of h1 and h2 (handles None)."""
# 	if h1 is None and h2 is None:
# 		return None
# 	if h1 is None:
# 		out = h2.Clone(name or (h2.GetName() + "_sum"))
# 		out.SetDirectory(0)
# 		return out
# 	if h2 is None:
# 		out = h1.Clone(name or (h1.GetName() + "_sum"))
# 		out.SetDirectory(0)
# 		return out
# 	out = h1.Clone(name or (h1.GetName() + "_" + h2.GetName() + "_sum"))
# 	out.SetDirectory(0)
# 	out.Add(h2)
# 	return out


# def combine_hist_arrays(h1, h2, target_width=100.0):
# 	"""Return (content, errors, edges) arrays that are the binwise sum of h1 and h2.

# 	If one of the histograms is None, returns the rebinned arrays of the other.
# 	Returns None if neither provides usable edges.
# 	"""
# 	if h1 is None and h2 is None:
# 		return None
# 	if h1 is not None:
# 		c1, e1, edges1 = _root_hist_to_arrays(h1)
# 		c1, e1, edges1 = rebin_arrays(c1, e1, edges1, target_width=target_width)
# 	else:
# 		c1 = np.array([]); e1 = np.array([]); edges1 = np.array([])
# 	if h2 is not None:
# 		c2, e2, edges2 = _root_hist_to_arrays(h2)
# 		c2, e2, edges2 = rebin_arrays(c2, e2, edges2, target_width=target_width)
# 	else:
# 		c2 = np.array([]); e2 = np.array([]); edges2 = np.array([])

# 	edges = edges1 if edges1.size else edges2
# 	if edges.size == 0:
# 		return None
# 	n = edges.size - 1
# 	c1p = np.zeros(n)
# 	c2p = np.zeros(n)
# 	if c1.size:
# 		c1p[:min(len(c1), n)] = c1[:min(len(c1), n)]
# 	if c2.size:
# 		c2p[:min(len(c2), n)] = c2[:min(len(c2), n)]
# 	csum = c1p + c2p
# 	# we don't need errors for plotting (user requested no errorbars)
# 	esum = np.zeros_like(csum)
# 	return csum, esum, edges


# def plot_btag_comparison():
# 	"""Make three plots (signal, DY, TTbar) showing no-btag/loose/medium/tight.

# 	Saves PDFs to `plots/btag_manual/<category>/`.
# 	"""
# 	os.makedirs('plots/btag_manual/signal', exist_ok=True)
# 	os.makedirs('plots/btag_manual/dy', exist_ok=True)
# 	os.makedirs('plots/btag_manual/ttbar', exist_ok=True)

# 	# create combined DY+TTbar histograms
# 	# create combined DY+TTbar arrays (rebinned to 100 GeV)
# 	no_btag_dytt = combine_hist_arrays(no_btag_dy, no_btag_ttbar, target_width=100.0)
# 	btag_loose_dytt = combine_hist_arrays(btag_loose_dy, btag_loose_ttbar, target_width=100.0)
# 	btag_medium_dytt = combine_hist_arrays(btag_medium_dy, btag_medium_ttbar, target_width=100.0)
# 	btag_tight_dytt = combine_hist_arrays(btag_tight_dy, btag_tight_ttbar, target_width=100.0)

# 	mapping = [
# 		('signal', no_btag_signal, btag_loose_signal, btag_medium_signal, btag_tight_signal),
# 		('dy', no_btag_dy, btag_loose_dy, btag_medium_dy, btag_tight_dy),
# 		('ttbar', no_btag_ttbar, btag_loose_ttbar, btag_medium_ttbar, btag_tight_ttbar),
# 		('dy_ttbar', no_btag_dytt, btag_loose_dytt, btag_medium_dytt, btag_tight_dytt),
# 	]

# 	colors = {'nobtag': '#1f77b4', 'loose': '#2ca02c', 'medium': '#ff7f0e', 'tight': '#d62728'}
# 	labels = ['nobtag', 'loose', 'medium', 'tight']

# 	for cat, nob, loose, medium, tight in mapping:
# 		# create main plot + ratio subplot
# 		fig, (ax, axr) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, sharex=True, figsize=(10, 8))
# 		# determine rebin factor from nobtag to get ~100 GeV bins and rebin all hists
# 		factor_target_width = 100.0
# 		if nob is not None:
# 			# nob may be a ROOT histogram or a precomputed (content, errors, edges) tuple
# 			if isinstance(nob, (tuple, list)):
# 				_bcontent, _berrors, _bedges = nob
# 			else:
# 				_bcontent, _berrors, _bedges = _root_hist_to_arrays(nob)
# 			binw = _bedges[1] - _bedges[0] if len(_bedges) > 1 else 1.0
# 			factor = max(1, int(round(factor_target_width / binw)))
# 		else:
# 			factor = 1

# 		for hist, label in [(nob, 'nobtag'), (loose, 'loose'), (medium, 'medium'), (tight, 'tight')]:
# 			if hist is None:
# 				continue
# 			# hist may be a ROOT histogram or a precomputed (content, errors, edges) tuple
# 			if isinstance(hist, (tuple, list)):
# 				content, errors, edges = hist
# 			else:
# 				content, errors, edges = _root_hist_to_arrays(hist)
# 				content, errors, edges = rebin_arrays(content, errors, edges, target_width=factor_target_width)
# 			centers = (edges[:-1] + edges[1:]) / 2.0
# 			ax.step(edges[:-1], content, where='post', label=label, color=colors[label])

# 		ax.set_yscale('log')
# 		# determine displayed bin width (GeV) from rebinned nobtag or last edges
# 		ref_edges = None
# 		if isinstance(nob, (tuple, list)):
# 			ref_edges = nob[2]
# 		elif nob is not None:
# 			try:
# 				_rc, _re, _re_edges = _root_hist_to_arrays(nob)
# 				_rc, _re, ref_edges = rebin_arrays(_rc, _re, _re_edges, target_width=factor_target_width)
# 			except Exception:
# 				ref_edges = None
# 		# fallback: use edges from last plotted hist if available
# 		try:
# 			_last_bin_width = abs(edges[1] - edges[0]) if 'edges' in locals() and len(edges) > 1 else None
# 		except Exception:
# 			_last_bin_width = None
# 		if ref_edges is not None and len(ref_edges) > 1:
# 			binw = abs(ref_edges[1] - ref_edges[0])
# 		elif _last_bin_width is not None:
# 			binw = _last_bin_width
# 		else:
# 			binw = factor_target_width
# 		# format bin width string
# 		if abs(binw - round(binw)) < 1e-6:
# 			binw_str = f"{int(round(binw))}"
# 		else:
# 			binw_str = f"{binw:.1f}"
# 		ax.set_ylabel(f"Events / {binw_str} GeV")
# 		ax.legend()
# 		# add left-aligned label above the main plot; special-case dy_ttbar
# 		if cat == 'dy_ttbar':
# 			label_txt = 'dy+ttbar'
# 		else:
# 			label_txt = cat.replace('_', ' ').capitalize()
# 		ax.text(0.01, 1.02, label_txt, transform=ax.transAxes, ha='left', va='bottom', fontsize=12)

# 		# build ratio plots (loose/medium/tight) relative to nobtag
# 		base_hist = nob
# 		if base_hist is None:
# 			# nothing to ratio to
# 			axr.text(0.5, 0.5, 'No base (nobtag) histogram', ha='center', va='center')
# 		else:
# 			# base_hist may be precomputed arrays for combined category
# 			if isinstance(base_hist, (tuple, list)):
# 				base_c, base_e, base_edges = base_hist
# 			else:
# 				base_c, base_e, base_edges = _root_hist_to_arrays(base_hist)
# 				base_c, base_e, base_edges = rebin_arrays(base_c, base_e, base_edges, target_width=100.0)
# 			ratio_mins = []
# 			ratio_maxs = []
# 			for hist, label in [(loose, 'loose'), (medium, 'medium'), (tight, 'tight')]:
# 				if hist is None:
# 					continue
# 				if isinstance(hist, (tuple, list)):
# 					c, e, edges = hist
# 				else:
# 					c, e, edges = _root_hist_to_arrays(hist)
# 					c, e, edges = rebin_arrays(c, e, edges, target_width=100.0)
# 				# propagate errors for r = c / c0
# 				c0 = base_c
# 				e0 = base_e
# 				# avoid division by zero
# 				with np.errstate(divide='ignore', invalid='ignore'):
# 					r = np.true_divide(c, c0)
# 					# variance: (err^2 / c0^2) + (c^2 * err0^2 / c0^4)
# 					var = np.true_divide(e**2, c0**2) + np.true_divide(c**2 * e0**2, c0**4)
# 					r_err = np.sqrt(var)
# 					# mask where c0 == 0
# 					r = np.where(c0 <= 0, np.nan, r)
# 					r_err = np.where(c0 <= 0, np.nan, r_err)
# 				centers = (edges[:-1] + edges[1:]) / 2.0
# 				axr.step(edges[:-1], r, where='post', label=label, color=colors[label])
# 				ratio_mins.append(np.nanmin(r))
# 				ratio_maxs.append(np.nanmax(r))

# 			# format ratio axis
# 			axr.axhline(1.0, color='gray', linestyle='--', linewidth=1)
# 			axr.set_ylabel('Ratio')
# 			axr.set_xlabel('m_lljj [GeV]')

# 			# set x-axis range: 1 TeV to 7 TeV (1000-7000 GeV)
# 			ax.set_xlim(1000, 7000)
# 			axr.set_xlim(1000, 7000)
# 			# x-axis ticks: major every 1 TeV (1000 GeV), minor every 100 GeV
# 			ax.xaxis.set_major_locator(mticker.MultipleLocator(1000))
# 			ax.xaxis.set_minor_locator(mticker.MultipleLocator(100))
# 			axr.xaxis.set_major_locator(mticker.MultipleLocator(1000))
# 			axr.xaxis.set_minor_locator(mticker.MultipleLocator(100))
# 			# label majors in TeV (e.g., '1 TeV') for readability
# 			ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x/1000)} TeV"))
# 			axr.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, pos: f"{int(x/1000)} TeV"))
# 			# set reasonable ylim if ratios present
# 			if ratio_mins and ratio_maxs:
# 				vmin = max(0.0, min(ratio_mins) * 0.9)
# 				vmax = max(ratio_maxs) * 1.1
# 				if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
# 					axr.set_ylim(vmin, vmax)

# 		# save (ensure output directory exists)
# 		# finalize layout
# 		fig.tight_layout()
# 		outdir = Path(f'plots/btag_manual/{cat}')
# 		outdir.mkdir(parents=True, exist_ok=True)
# 		out = outdir / f"{variable_path_mumu.split('/')[1]}_{cat}_btag_compare.pdf"
# 		fig.savefig(str(out))
# 		plt.close(fig)
# 		print('Saved', out)


# if __name__ == '__main__':
# 	# create the comparison plots
# 	plot_btag_comparison()










import os
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT.gROOT.SetBatch(True)

# =============================
# Configuration (same as yours)
# =============================

base_path = "rootfiles/RunII/2018/RunIISummer20UL18/btagloose/"
signal_file = "WRAnalyzer_signal_WR3200_N3000.root"
dy_file = "WRAnalyzer_DYJets.root"
ttbar_file = "WRAnalyzer_TTbar.root"

variable_path_2d = "wr_mumu_resolved_sr/WRMass_DeltaRbb_wr_mumu_resolved_sr" 

lum = 59  # fb^-1

# =============================
# Helper: Convert ROOT TH2 to numpy arrays
# =============================

def root_th2_to_arrays(h):
    nx = h.GetNbinsX()
    ny = h.GetNbinsY()

    content = np.zeros((nx, ny))

    for ix in range(nx):
        for iy in range(ny):
            content[ix, iy] = h.GetBinContent(ix+1, iy+1)

    x_edges = np.array([h.GetXaxis().GetBinLowEdge(i+1) for i in range(nx)])
    x_edges = np.append(x_edges, h.GetXaxis().GetBinUpEdge(nx))

    y_edges = np.array([h.GetYaxis().GetBinLowEdge(i+1) for i in range(ny)])
    y_edges = np.append(y_edges, h.GetYaxis().GetBinUpEdge(ny))

    return content, x_edges, y_edges


# =============================
# Load histograms
# =============================

signal_file_root = ROOT.TFile(base_path + signal_file)
dy_file_root = ROOT.TFile(base_path + dy_file)
ttbar_file_root = ROOT.TFile(base_path + ttbar_file)

h_signal = signal_file_root.Get(variable_path_2d)
h_dy = dy_file_root.Get(variable_path_2d)
h_ttbar = ttbar_file_root.Get(variable_path_2d)

# Scale by luminosity (same as your script)
scale_factor = lum * 1000
h_signal.Scale(scale_factor)
h_dy.Scale(scale_factor)
h_ttbar.Scale(scale_factor)

# =============================
# Convert to arrays
# =============================

S, x_edges, y_edges = root_th2_to_arrays(h_signal)
DY, _, _ = root_th2_to_arrays(h_dy)
TT, _, _ = root_th2_to_arrays(h_ttbar)

B = DY + TT

# =============================
# Compute FOM
# =============================

with np.errstate(divide='ignore', invalid='ignore'):
    FOM = np.where(B > 0,S / (np.sqrt(B)),0)

# =============================
# Plot
# =============================

fig, ax = plt.subplots(figsize=(8, 7))

mesh = ax.pcolormesh(
    x_edges,
    y_edges,
    FOM.T,
    shading='auto'
)

cbar = plt.colorbar(mesh, ax=ax)
cbar.set_label("S / sqrt(B)")

ax.set_xlabel("m_lljj (GeV)")
ax.set_ylabel("Delta R")
ax.set_title("Figure of Merit (2D)")

# =============================
# Save
# =============================

outdir = Path("plots/fom_2d")
outdir.mkdir(parents=True, exist_ok=True)

out = outdir / "fom_2d.pdf"
fig.tight_layout()
fig.savefig(out)

print("Saved", out)