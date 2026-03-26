import ROOT
from pathlib import Path

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

def make_table(matrix, title):
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
    lines.append(f"\\caption{{{title}}}")
    lines.append("\\end{table}")
    lines.append("\n")

    return "\n".join(lines)

def build_matrix(hist):
    nx = hist.GetNbinsX()
    ny = hist.GetNbinsY()

    matrix = [[0.0]*3 for _ in range(4)]

    for ix in range(1, nx + 1):
        for iy in range(1, ny + 1):
            case = ix - 1
            j3 = iy - 1
            matrix[case][j3] = hist.GetBinContent(ix, iy)

    return matrix

# =============================
# Main
# =============================

for btag in btag_categories:

    base_path = base_path_template.format(btag=btag)

    latex_content = []

    # =============================
    # Backgrounds
    # =============================

    f_dy = ROOT.TFile(base_path + dy_file)
    f_ttbar = ROOT.TFile(base_path + ttbar_file)
    f_nonprompt = [ROOT.TFile(base_path + f) for f in nonprompt_files]
    f_other = [ROOT.TFile(base_path + f) for f in other_samples_files]

    h_dy = get_hist(f_dy)
    h_ttbar = get_hist(f_ttbar)
    h_nonprompt = build_sum_hist(f_nonprompt)
    h_other = build_sum_hist(f_other)

    # Scale backgrounds
    for h in [h_dy, h_ttbar, h_nonprompt, h_other]:
        h.Scale(scale_factor)

    # Build matrices
    dy_matrix = build_matrix(h_dy)
    tt_matrix = build_matrix(h_ttbar)

    total_hist = h_dy.Clone()
    total_hist.Add(h_ttbar)
    total_hist.Add(h_nonprompt)
    total_hist.Add(h_other)
    total_matrix = build_matrix(total_hist)

    # Add tables
    latex_content.append(make_table(dy_matrix, "DY yields"))
    latex_content.append(make_table(tt_matrix, "TTbar yields"))
    latex_content.append(make_table(total_matrix, "Total background yields"))

    # =============================
    # Signals
    # =============================

    for Nmass in neutrino_masses:

        signal_file = signal_file_template.format(Nmass=Nmass)
        f_sig = ROOT.TFile(base_path + signal_file)

        h_sig = get_hist(f_sig)

        # Scale signal
        h_sig.Scale(scale_factor * signal_scale)

        sig_matrix = build_matrix(h_sig)

        latex_content.append(
            make_table(sig_matrix, f"Signal yields (WR3200, N={Nmass})")
        )

    # =============================
    # Save file
    # =============================

    outpath = Path(f"case_vs_j3_tables_{btag}.tex")
    with open(outpath, "w") as f:
        f.write("\n\n".join(latex_content))

    print(f"Saved LaTeX tables (background + signal) to {outpath}")