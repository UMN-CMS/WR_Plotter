from pathlib import Path
import ROOT
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

hist_path = "wr_mumu_resolved_sr/case_vs_j3_wr_mumu_resolved_sr"

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

def get_hist(f):
    h = f.Get(hist_path)
    if not h:
        raise RuntimeError(f"Histogram {hist_path} not found in {f.GetName()}")
    return h.Clone()

def build_sum_hist(files):
    h_sum = get_hist(files[0])
    for f in files[1:]:
        h_sum.Add(get_hist(f))
    return h_sum

def compute_total_significance(sig_hist, bkg_hist):
    S = sig_hist.Integral()
    B = bkg_hist.Integral()
    if B > 0:
        return S / math.sqrt(B)
    return 0.0

def compute_significance_map(sig_hist, bkg_hist):
    nx = sig_hist.GetNbinsX()
    ny = sig_hist.GetNbinsY()

    S_total = sig_hist.Integral()
    B_total = bkg_hist.Integral()

    result = [[0.0]*ny for _ in range(nx)]

    for ix in range(1, nx + 1):
        for iy in range(1, ny + 1):
            s_bin = sig_hist.GetBinContent(ix, iy)
            b_bin = bkg_hist.GetBinContent(ix, iy)

            S_remain = S_total - s_bin
            B_remain = B_total - b_bin

            if B_remain > 0:
                signif = S_remain / math.sqrt(B_remain)
            else:
                signif = 0.0

            # Percentage difference relative to initial significance
            initial_sig = S_total / math.sqrt(B_total) if B_total>0 else 0
            if initial_sig != 0:
                percent_diff = (signif - initial_sig) / initial_sig * 100
            else:
                percent_diff = 0.0

            result[ix-1][iy-1] = percent_diff

    return result

def print_significance_table(matrix, title):
    print("\n" + "="*70)
    print(title)
    print("="*70)

    header = ["Case \\ J3", "No J3", "J3 no btag", "J3 btag"]
    print("{:<12} {:>15} {:>15} {:>15}".format(*header))

    for case in range(4):
        row = matrix[case]
        print("{:<12} {:>15.2f}% {:>15.2f}% {:>15.2f}%".format(
            case_labels_short[case],
            row[0], row[1], row[2]
        ))

def make_latex_table(matrix, title, filename):
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{c|ccc}")
    lines.append("\\hline")
    lines.append("Case & No J3 & J3 no btag & J3 btag \\\\")
    lines.append("\\hline")

    for case in range(4):
        row = matrix[case]
        line = f"{case_labels_short[case]} "
        for val in row:
            # Color scaling
            if val < 0:
                intensity = min(abs(val)/20, 1.0)  # max -20%
                color = f"\\cellcolor{{red!{int(intensity*100)}}}"
            elif val > 0:
                intensity = min(val/2, 1.0)  # max +2%
                color = f"\\cellcolor{{green!{int(intensity*100)}}}"
            else:
                color = "\\cellcolor{white}"

            line += f"& {color}{val:.2f}\\% "
        line += "\\\\"
        lines.append(line)

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{title}}}")
    lines.append("\\end{table}")
    lines.append("\n")

    Path(filename).write_text("\n".join(lines))

# =============================
# Main
# =============================

for btag in btag_categories:

    base_path = base_path_template.format(btag=btag)

    # Backgrounds
    f_dy = ROOT.TFile(base_path + dy_file)
    f_ttbar = ROOT.TFile(base_path + ttbar_file)
    f_nonprompt = [ROOT.TFile(base_path + f) for f in nonprompt_files]
    f_other = [ROOT.TFile(base_path + f) for f in other_samples_files]

    h_dy = get_hist(f_dy)
    h_ttbar = get_hist(f_ttbar)
    h_nonprompt = build_sum_hist(f_nonprompt)
    h_other = build_sum_hist(f_other)

    for h in [h_dy, h_ttbar, h_nonprompt, h_other]:
        h.Scale(scale_factor)

    total_hist = h_dy.Clone()
    total_hist.Add(h_ttbar)
    total_hist.Add(h_nonprompt)
    total_hist.Add(h_other)

    # Signals
    for Nmass in neutrino_masses:
        signal_file = signal_file_template.format(Nmass=Nmass)
        f_sig = ROOT.TFile(base_path + signal_file)

        h_sig = get_hist(f_sig)
        h_sig.Scale(scale_factor * signal_scale)

        initial_sig = compute_total_significance(h_sig, total_hist)
        print("\n" + "#"*70)
        print(f"Signal: WR3200, N={Nmass}, category={btag}")
        print(f"Initial S/sqrt(B) = {initial_sig:.4f}")
        print("#"*70)

        signif_matrix = compute_significance_map(h_sig, total_hist)
        print_significance_table(signif_matrix, f"Percentage difference from initial S/sqrt(B)")

        # Save LaTeX table
        latex_file = f"significance_table_N{Nmass}_{btag}.tex"
        make_latex_table(signif_matrix, f"Percentage difference S/sqrt(B) for N={Nmass}, {btag}", latex_file)

    print("\nDone.\n")