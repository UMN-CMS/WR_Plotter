# import os
# import ROOT
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path

# ROOT.gROOT.SetBatch(True)

# # =============================
# # Configuration
# # =============================

# base_path = "rootfiles/RunII/2018/RunIISummer20UL18/btagtightmed/"
# signal_file = "WRAnalyzer_signal_WR3200_N3000.root"
# dy_file = "WRAnalyzer_DYJets.root"
# ttbar_file = "WRAnalyzer_TTbar.root"

# variable_path_2d = "wr_mumu_resolved_sr/WRMass_DeltaRbb_wr_mumu_resolved_sr"  # <-- change if needed

# lum = 59
# scale_factor = lum * 1000


# # =============================
# # Utilities
# # =============================

# def get_hist(file, path):
#     h = file.Get(path)
#     if not h:
#         raise RuntimeError(f"Histogram '{path}' not found in {file.GetName()}")
#     h.SetDirectory(0)
#     return h


# def root_th2_to_arrays(h):
#     nx = h.GetNbinsX()
#     ny = h.GetNbinsY()

#     content = np.zeros((nx, ny))

#     for ix in range(nx):
#         for iy in range(ny):
#             content[ix, iy] = h.GetBinContent(ix+1, iy+1)

#     x_edges = np.array([h.GetXaxis().GetBinLowEdge(i+1) for i in range(nx)])
#     x_edges = np.append(x_edges, h.GetXaxis().GetBinUpEdge(nx))

#     y_edges = np.array([h.GetYaxis().GetBinLowEdge(i+1) for i in range(ny)])
#     y_edges = np.append(y_edges, h.GetYaxis().GetBinUpEdge(ny))

#     return content, x_edges, y_edges


# # =============================
# # Load histograms
# # =============================

# f_sig = ROOT.TFile(base_path + signal_file)
# f_dy = ROOT.TFile(base_path + dy_file)
# f_tt = ROOT.TFile(base_path + ttbar_file)

# h_sig = get_hist(f_sig, variable_path_2d)
# h_dy = get_hist(f_dy, variable_path_2d)
# h_tt = get_hist(f_tt, variable_path_2d)

# h_sig.Scale(scale_factor)
# h_dy.Scale(scale_factor)
# h_tt.Scale(scale_factor)

# S, mass_edges, dR_edges = root_th2_to_arrays(h_sig)
# DY, _, _ = root_th2_to_arrays(h_dy)
# TT, _, _ = root_th2_to_arrays(h_tt)

# B = DY + TT

# # =============================
# # Cumulative integration in ΔR
# # =============================

# # axis 1 = deltaR axis
# S_cum = np.cumsum(S, axis=1)
# B_cum = np.cumsum(B, axis=1)

# # =============================
# # Compute FOM
# # =============================

# FOM = np.where(B_cum > 0, S_cum / (np.sqrt(B_cum)), 0)

# # =============================
# # Plot FOM vs ΔR cut
# # =============================

# # Option 1: take maximum FOM over mass for each ΔR cut
# FOM_max_over_mass = np.max(FOM, axis=0)

# dR_centers = 0.5 * (dR_edges[:-1] + dR_edges[1:])

# plt.figure(figsize=(8,6))
# plt.plot(dR_centers, FOM_max_over_mass)

# plt.xlabel("ΔR cut (ΔR < cut)")
# plt.ylabel("Max S / sqrt(B)")
# plt.title("Cumulative ΔR Optimization")

# plt.grid(True)

# outdir = Path("plots/fom_deltaR_scan")
# outdir.mkdir(parents=True, exist_ok=True)

# outfile = outdir / "fom_vs_deltaR_tightmed.pdf"
# plt.tight_layout()
# plt.savefig(outfile)
# plt.close()

# print("Saved", outfile)



import ROOT
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT.gROOT.SetBatch(True)

# =============================
# Configuration (same structure as your script)
# =============================

base_path = "rootfiles/RunII/2018/RunIISummer20UL18/btagloose/"
signal_file = "WRAnalyzer_signal_WR3200_N3000.root"
dy_file = "WRAnalyzer_DYJets.root"
ttbar_file = "WRAnalyzer_TTbar.root"

variable_path_2d = "wr_mumu_resolved_sr/WRMass_DeltaRbb_wr_mumu_resolved_sr"  # <-- adjust if needed

lum = 59
scale_factor = lum * 1000


# =============================
# Utilities
# =============================

def get_hist(file, path):
    h = file.Get(path)
    if not h:
        raise RuntimeError(f"Histogram '{path}' not found in {file.GetName()}")
    h.SetDirectory(0)
    return h


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

f_sig = ROOT.TFile(base_path + signal_file)
f_dy  = ROOT.TFile(base_path + dy_file)
f_tt  = ROOT.TFile(base_path + ttbar_file)

h_sig = get_hist(f_sig, variable_path_2d)
h_dy  = get_hist(f_dy, variable_path_2d)
h_tt  = get_hist(f_tt, variable_path_2d)

h_sig.Scale(scale_factor)
h_dy.Scale(scale_factor)
h_tt.Scale(scale_factor)

S, mass_edges, dR_edges = root_th2_to_arrays(h_sig)
DY, _, _ = root_th2_to_arrays(h_dy)
TT, _, _ = root_th2_to_arrays(h_tt)

B = DY + TT


# =============================
# Cumulative in ΔR
# =============================

# ΔR is axis=1
S_cum = np.cumsum(S, axis=1)
B_cum = np.cumsum(B, axis=1)


# =============================
# Compute FOM (no B>0 protection needed)
# =============================

FOM = np.where(B_cum > 0, S_cum / np.sqrt(B_cum), 0)


# =============================
# Plot 2D heatmap
# =============================

fig, ax = plt.subplots(figsize=(8, 7))

mesh = ax.pcolormesh(
    mass_edges,
    dR_edges,
    FOM.T,  # transpose because pcolormesh expects (Y,X)
    shading='auto'
)

cbar = plt.colorbar(mesh, ax=ax)
cbar.set_label("S / sqrt(B)")

ax.set_xlabel("m_lljj [GeV]")
ax.set_ylabel("ΔR cut (ΔR < value)")
ax.set_title("Cumulative ΔR Scan FOM")

# optional: restrict mass window
ax.set_xlim(0, 8000)

fig.tight_layout()

outdir = Path("plots/fom_2d")
outdir.mkdir(parents=True, exist_ok=True)

outfile = outdir / "fom_2d_cumulative.pdf"
fig.savefig(outfile)
plt.close()

print("Saved", outfile)