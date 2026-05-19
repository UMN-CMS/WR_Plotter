import ROOT
from pathlib import Path
import math

ROOT.gROOT.SetBatch(True)

# =============================
# Configuration
# =============================

base_path_template = "rootfiles/RunII/2018/RunIISummer20UL18/{btag}/"
signal_file_template = "WRAnalyzer_signal_WR3200_N{Nmass}.root"

hist_path_mlljj  = "wr_mumu_resolved_sr/case_vs_j3_vs_mlljj_wr_mumu_resolved_sr"
hist_path_mlljjj = "wr_mumu_resolved_sr/case_vs_j3_vs_mlljjj_wr_mumu_resolved_sr"

lum = 59.74
scale_factor = lum * 1000

btag_categories = ["btagloose"]
neutrino_masses = [800, 1200, 1600, 2400, 3000]

signal_scale = 1

case_labels_short = {0: "00", 1: "01", 2: "10", 3: "11"}

# =============================
# Helpers
# =============================

def get_hist(f, path):
    h = f.Get(path)
    if not h:
        raise RuntimeError(f"Histogram {path} not found in {f.GetName()}")
    return h.Clone()

# -----------------------------
# Replace one (case, j3) cell
# -----------------------------
def replace_cell(sig_nominal, sig_alt, ix, iy):
    h_new = sig_nominal.Clone()
    nz = h_new.GetNbinsZ()
    
    for iz in range(1, nz+1):
        val = sig_alt.GetBinContent(ix, iy, iz)
        h_new.SetBinContent(ix, iy, iz, val)

    return h_new

# -----------------------------
# Projection
# -----------------------------
def project_mass(hist3d):
    return hist3d.Project3D("Z")

# -----------------------------
# Asymmetric Gaussian (PyROOT-safe)
# -----------------------------
def asym_gaus(x, p):
    A   = p[0]
    mu  = p[1]
    sigL = p[2]
    sigR = p[3]

    if sigL <= 0 or sigR <= 0:
        return 0

    if x[0] < mu:
        return A * math.exp(-0.5*((x[0]-mu)/sigL)**2)
    else:
        return A * math.exp(-0.5*((x[0]-mu)/sigR)**2)

# -----------------------------
# Fit function (stable)
# -----------------------------
def fit_asym_gaussian(hist):

    if hist.Integral() < 10:
        return 0.0, 0.0

    func = lambda x, p: asym_gaus(x, p)

    f = ROOT.TF1(f"asymGaus_{id(hist)}", func, 0, 8000, 4)
    f._func = func   # 🔥 keep alive

    mean = hist.GetMean()
    rms  = hist.GetRMS()

    f.SetParameters(
        hist.GetMaximum(),
        mean,
        max(rms, 1e-3),
        max(rms, 1e-3),
    )

    fit_min = mean - 2*rms
    fit_max = mean + 2*rms

    hist.Fit(f, "Q", "", fit_min, fit_max)

    sigL = abs(f.GetParameter(2))
    sigR = abs(f.GetParameter(3))

    return sigL, sigR

# -----------------------------
# LaTeX tables
# -----------------------------
def make_latex_width_table(matrix, caption):
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{c|ccc}")
    lines.append("\\hline")
    lines.append("Case & No J3 & J3 no btag & J3 btag \\\\")
    lines.append("\\hline")

    for case in range(4):
        row = matrix[case]
        lines.append(
            f"{case_labels_short[case]} & "
            f"{row[0]:.2f} & {row[1]:.2f} & {row[2]:.2f} \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)

def make_latex_width_percent_table(matrix, caption):
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{c|ccc}")
    lines.append("\\hline")
    lines.append("Case & No J3 & J3 no btag & J3 btag \\\\")
    lines.append("\\hline")

    for case in range(4):
        row = matrix[case]
        row_str = [case_labels_short[case]]

        for pct in row:
            if pct < 0:
                intensity = min(abs(pct)/20, 1.0)
                color = f"\\cellcolor{{green!{int(intensity*100)}}}"
            else:
                intensity = min(abs(pct)/20, 1.0)
                color = f"\\cellcolor{{red!{int(intensity*100)}}}"

            row_str.append(f"{color}{pct:.2f}\\%")

        lines.append(" & ".join(row_str) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)

# =============================
# Main
# =============================

for btag in btag_categories:

    base_path = base_path_template.format(btag=btag)

    for Nmass in neutrino_masses:

        print("\n" + "="*70)
        print(f"Processing WR3200, N={Nmass}, {btag}")
        print("="*70)

        f_sig = ROOT.TFile(base_path + signal_file_template.format(Nmass=Nmass))

        h_sig_mlljj  = get_hist(f_sig, hist_path_mlljj)
        h_sig_mlljjj = get_hist(f_sig, hist_path_mlljjj)

        h_sig_mlljj.Scale(scale_factor * signal_scale)
        h_sig_mlljjj.Scale(scale_factor * signal_scale)

        # -----------------------------
        # Baseline
        # -----------------------------
        h_base_proj = project_mass(h_sig_mlljj)

        sigL0, sigR0 = fit_asym_gaussian(h_base_proj)
        base_width = sigL0 + sigR0

        print(f"Baseline width: {base_width:.2f}")

        # -----------------------------
        # Loop over cells
        # -----------------------------
        width_matrix = [[0.0]*3 for _ in range(4)]
        percent_matrix = [[0.0]*3 for _ in range(4)]

        for ix in range(1, 5):
            for iy in range(1, 4):

                h_mod = replace_cell(h_sig_mlljj, h_sig_mlljjj, ix, iy)

                proj = project_mass(h_mod)
                sigL, sigR = fit_asym_gaussian(proj)

                width = sigL + sigR

                width_matrix[ix-1][iy-1] = width

                if base_width > 0:
                    percent_matrix[ix-1][iy-1] = 100*(width - base_width)/base_width

        # -----------------------------
        # Save tables
        # -----------------------------
        latex_width = make_latex_width_table(
            width_matrix,
            caption=f"Signal width (σ_L + σ_R) for WR3200, N={Nmass}, {btag}"
        )

        latex_percent = make_latex_width_percent_table(
            percent_matrix,
            caption=f"Percentage change in width for WR3200, N={Nmass}, {btag}"
        )

        out1 = Path(f"width_table_N{Nmass}_{btag}.tex")
        out2 = Path(f"width_percent_table_N{Nmass}_{btag}.tex")

        with open(out1, "w") as f:
            f.write(latex_width)

        with open(out2, "w") as f:
            f.write(latex_percent)

        print(f"Saved: {out1}")
        print(f"Saved: {out2}")