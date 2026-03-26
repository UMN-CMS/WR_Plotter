import ROOT
from pathlib import Path
import math

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

hist_path_3d = "wr_mumu_resolved_sr/case_vs_j3_vs_mlljj_wr_mumu_resolved_sr"

lum = 59.74
scale_factor = lum * 1000

btag_categories = ["btagloose"]
neutrino_masses = [800, 1200, 1600, 2400, 3000]

signal_scale = 1

# =============================
# Labels
# =============================

case_labels_short = {
    0: "00",
    1: "01",
    2: "10",
    3: "11",
}

# =============================
# Helpers
# =============================

def get_hist(f, path):
    h = f.Get(path)
    if not h:
        raise RuntimeError(f"Histogram {path} not found in {f.GetName()}")
    return h.Clone()

def build_sum_hist(files, path):
    h_sum = get_hist(files[0], path)
    for f in files[1:]:
        h_sum.Add(get_hist(f, path))
    return h_sum

def get_global_mlljj_window(sig3d, nsigma=2):
    proj = sig3d.Project3D("Z")  # sum over case,J3
    median = proj.GetMean()       # approximate median
    sigma = proj.GetRMS()
    low = median - nsigma*sigma
    high = median + nsigma*sigma
    return low, high

def sum_in_window_2d(hist3d, low, high):
    nx = hist3d.GetNbinsX()
    ny = hist3d.GetNbinsY()
    nz = hist3d.GetNbinsZ()
    result = [[0.0]*ny for _ in range(nx)]
    
    for ix in range(1, nx+1):
        for iy in range(1, ny+1):
            s = 0.0
            for iz in range(1, nz+1):
                z_center = hist3d.GetZaxis().GetBinCenter(iz)
                if low <= z_center <= high:
                    s += hist3d.GetBinContent(ix, iy, iz)
            result[ix-1][iy-1] = s
    return result

def compute_leave_one_out_significance(sig_matrix, bkg_matrix):
    S_total = sum(sum(row) for row in sig_matrix)
    B_total = sum(sum(row) for row in bkg_matrix)
    
    signif_matrix = [[0.0]*3 for _ in range(4)]
    for ix in range(4):
        for iy in range(3):
            S_remain = S_total - sig_matrix[ix][iy]
            B_remain = B_total - bkg_matrix[ix][iy]
            signif_matrix[ix][iy] = S_remain / math.sqrt(B_remain) if B_remain>0 else 0.0
    
    initial_signif = S_total / math.sqrt(B_total) if B_total>0 else 0.0
    return initial_signif, signif_matrix

def print_significance_table(matrix, title):
    print("\n" + "="*70)
    print(title)
    print("="*70)
    header = ["Case \\ J3", "No J3", "J3 no btag", "J3 btag"]
    print("{:<12} {:>15} {:>15} {:>15}".format(*header))
    for case in range(4):
        row = matrix[case]
        print("{:<12} {:>15.4f} {:>15.4f} {:>15.4f}".format(
            case_labels_short[case], row[0], row[1], row[2]
        ))

def print_significance_diff_matrix(matrix, initial, title):
    """
    Print leave-one-cell-out S/sqrt(B) as percentage difference from initial
    """
    print("\n" + "="*70)
    print(title)
    print("="*70)
    header = ["Case \\ J3", "No J3", "J3 no btag", "J3 btag"]
    print("{:<12} {:>15} {:>15} {:>15}".format(*header))

    for case in range(4):
        row = matrix[case]
        percent_diff = [100*(v - initial)/initial if initial>0 else 0.0 for v in row]
        print("{:<12} {:>15.2f}% {:>15.2f}% {:>15.2f}%".format(
            case_labels_short[case],
            percent_diff[0], percent_diff[1], percent_diff[2]
        ))

def make_latex_significance_table_asym(matrix, initial, caption, pos_max=2, neg_max=20):
    """
    LaTeX table with colored cells.
    - Positive % differences: green, max intensity at pos_max%
    - Negative % differences: red, max intensity at -neg_max%
    """
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
        for v in row:
            pct = 100*(v - initial)/initial if initial>0 else 0.0

            if pct > 0:
                # Clamp and scale for green
                intensity = min(pct / pos_max, 1.0)
                color = f"\\cellcolor{{green!{int(intensity*100)}}}"
            elif pct < 0:
                # Clamp and scale for red
                intensity = min(abs(pct) / neg_max, 1.0)
                color = f"\\cellcolor{{red!{int(intensity*100)}}}"
            else:
                color = ""

            row_str.append(f"{color}{pct:.2f}\\%")
        lines.append(" & ".join(row_str) + " \\\\")
    
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\end{table}")
    lines.append("\n")
    return "\n".join(lines)
# =============================
# Main
# =============================

for btag in btag_categories:

    base_path = base_path_template.format(btag=btag)
    
    # -----------------------------
    # Load backgrounds
    # -----------------------------
    f_dy = ROOT.TFile(base_path + dy_file)
    f_ttbar = ROOT.TFile(base_path + ttbar_file)
    f_nonprompt = [ROOT.TFile(base_path + f) for f in nonprompt_files]
    f_other = [ROOT.TFile(base_path + f) for f in other_samples_files]

    h_dy = get_hist(f_dy, hist_path_3d)
    h_ttbar = get_hist(f_ttbar, hist_path_3d)
    h_nonprompt = build_sum_hist(f_nonprompt, hist_path_3d)
    h_other = build_sum_hist(f_other, hist_path_3d)

    for h in [h_dy, h_ttbar, h_nonprompt, h_other]:
        h.Scale(scale_factor)

    total_hist = h_dy.Clone()
    total_hist.Add(h_ttbar)
    total_hist.Add(h_nonprompt)
    total_hist.Add(h_other)

    # -----------------------------
    # Loop over signal masses
    # -----------------------------
    for Nmass in neutrino_masses:

        f_sig = ROOT.TFile(base_path + signal_file_template.format(Nmass=Nmass))
        h_sig3d = get_hist(f_sig, hist_path_3d)
        h_sig3d.Scale(scale_factor * signal_scale)

        # Determine global ±2σ m_lljj window from signal
        low, high = get_global_mlljj_window(h_sig3d, nsigma=2)
        print(f"\nGlobal ±2σ m_lljj window for N={Nmass}: [{low:.2f}, {high:.2f}] GeV")

        # Sum signal and background in this window per (case,J3)
        sig_matrix = sum_in_window_2d(h_sig3d, low, high)
        bkg_matrix = sum_in_window_2d(total_hist, low, high)

        # Compute initial and leave-one-out significances
        initial_signif, signif_matrix = compute_leave_one_out_significance(sig_matrix, bkg_matrix)

        # Print
        print("\n" + "#"*70)
        print(f"Signal: WR3200, N={Nmass}, category={btag}")
        print(f"Initial S/sqrt(B) in global ±2σ m_lljj window = {initial_signif:.4f}")
        print("#"*70)

        print_significance_diff_matrix(
            signif_matrix,
            initial_signif,
            "Leave-one-cell-out S/sqrt(B) relative change (%)"
        )

        # Generate LaTeX table
        latex_table = make_latex_significance_table_asym(
            signif_matrix,
            initial_signif,
            caption=f"Leave-one-cell-out S/sqrt(B) % difference for WR3200, N={Nmass}, {btag}"
        )

        # Save LaTeX table
        outpath = Path(f"leave_one_out_percent_diff_N{Nmass}_{btag}.tex")
        with open(outpath, "w") as f:
            f.write(latex_table)

        print(f"LaTeX table saved to {outpath}")