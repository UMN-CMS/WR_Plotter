import ROOT
from pathlib import Path

ROOT.gROOT.SetBatch(True)

# Prevent ROOT ownership problems
ROOT.TH1.AddDirectory(False)

# =========================================================
# Configuration
# =========================================================

base_path_template = "rootfiles/RunII/2018/RunIISummer20UL18/"

signal_file_template = "WRAnalyzer_signal_WR{WRmass}_N{Nmass}.root"

regions = [
    "wr_ee_resolved_sr",
    "wr_mumu_resolved_sr"
]

wr_masses = [1200, 1600, 2000, 2400, 2800, 3200]

neutrino_masses = {
    1200: [200, 400, 600, 800, 1100],
    1600: [400, 600, 800, 1200, 1500],
    2000: [400, 800, 1000, 1400, 1900],
    2400: [600, 800, 1200, 1800, 2300],
    2800: [600, 800, 1400, 2000, 2700],
    3200: [800, 1200, 1600, 2400, 3000]
}

signal_scale = 1.0

# Rebin factor
rebin_factor = 10

outdir = Path("plots/JEC/m_lljj_comparison")
outdir.mkdir(parents=True, exist_ok=True)

# =========================================================
# Histogram Helper
# =========================================================

def get_histogram(root_file, hist_path, unique_name=""):
    """
    Retrieve histogram safely from ROOT file.
    Returns detached clone.
    """

    hist = root_file.Get(hist_path)

    if not hist:
        print(f"WARNING: Missing histogram: {hist_path}")
        return None

    cloned = hist.Clone(f"{hist.GetName()}_{unique_name}")

    # VERY IMPORTANT
    cloned.SetDirectory(0)

    return cloned


# =========================================================
# Main Loop
# =========================================================

for region in regions:

    print(f"\n{'=' * 70}")
    print(f"Processing region: {region}")
    print(f"{'=' * 70}")

    for wr_mass in wr_masses:

        print(f"\nWR mass: {wr_mass} GeV")

        for nmass in neutrino_masses[wr_mass]:

            print(f"  Processing N{nmass}")

            signal_file = signal_file_template.format(
                WRmass=wr_mass,
                Nmass=nmass
            )

            signal_path = base_path_template + signal_file

            try:

                # =========================================================
                # Open file
                # =========================================================

                f_signal = ROOT.TFile.Open(signal_path)

                if not f_signal or f_signal.IsZombie():
                    print(f"    ERROR: Could not open {signal_path}")
                    continue

                # =========================================================
                # Histogram paths
                # =========================================================

                corrected_path = (
                    f"{region}/mass_lljj_corrected_{region}"
                )

                original_path = (
                    f"{region}/mass_fourobject_{region}"
                )

                # =========================================================
                # Retrieve histograms
                # =========================================================

                h_corrected = get_histogram(
                    f_signal,
                    corrected_path,
                    unique_name=f"corr_WR{wr_mass}_N{nmass}_{region}"
                )

                h_original = get_histogram(
                    f_signal,
                    original_path,
                    unique_name=f"orig_WR{wr_mass}_N{nmass}_{region}"
                )

                # Close file immediately after cloning
                f_signal.Close()

                if not h_corrected:
                    print("    ERROR: Corrected histogram missing")
                    continue

                if not h_original:
                    print("    ERROR: Original histogram missing")
                    continue

                # =========================================================
                # Rebin
                # =========================================================

                h_corrected.Rebin(rebin_factor)
                h_original.Rebin(rebin_factor)

                # =========================================================
                # Scale
                # =========================================================

                h_corrected.Scale(signal_scale)
                h_original.Scale(signal_scale)

                # =========================================================
                # Styling
                # =========================================================

                # Corrected mass
                h_corrected.SetLineColor(ROOT.kRed + 1)
                h_corrected.SetLineWidth(1)
                h_corrected.SetFillStyle(0)

                # Original mass
                h_original.SetLineColor(ROOT.kBlue + 1)
                h_original.SetLineWidth(1)
                h_original.SetFillStyle(0)

                # =========================================================
                # Canvas
                # =========================================================

                canvas_name = (
                    f"c_{region}_WR{wr_mass}_N{nmass}"
                )

                c = ROOT.TCanvas(
                    canvas_name,
                    canvas_name,
                    1000,
                    700
                )

                c.SetLeftMargin(0.12)
                c.SetRightMargin(0.05)
                c.SetBottomMargin(0.12)
                c.SetTopMargin(0.08)

                # Log-scale Y axis
                c.SetLogy()

                # =========================================================
                # Determine y-axis range
                # =========================================================

                max_y = max(
                    h_corrected.GetMaximum(),
                    h_original.GetMaximum()
                )

                h_corrected.SetMaximum(max_y * 20)
                h_corrected.SetMinimum(1e-4)

                # =========================================================
                # Axis titles
                # =========================================================

                h_corrected.SetTitle(
                    f"WR{wr_mass} GeV, N{nmass} GeV - {region}"
                )

                h_corrected.GetXaxis().SetTitle(
                    "m_{#ell#ell jj} [GeV]"
                )

                h_corrected.GetYaxis().SetTitle("Events")

                h_corrected.GetXaxis().SetTitleSize(0.045)
                h_corrected.GetYaxis().SetTitleSize(0.045)

                h_corrected.GetXaxis().SetLabelSize(0.04)
                h_corrected.GetYaxis().SetLabelSize(0.04)

                # =========================================================
                # Draw histograms
                # =========================================================

                h_corrected.Draw("HIST")
                h_original.Draw("HIST SAME")

                # =========================================================
                # Legend (TOP LEFT)
                # =========================================================

                legend = ROOT.TLegend(
                    0.15,
                    0.75,
                    0.40,
                    0.90
                )

                legend.SetBorderSize(0)
                legend.SetFillStyle(0)
                legend.SetTextSize(0.035)

                legend.AddEntry(
                    h_corrected,
                    "Corrected mass",
                    "l"
                )

                legend.AddEntry(
                    h_original,
                    "Original four-object mass",
                    "l"
                )

                legend.Draw()

                # =========================================================
                # Force redraw
                # =========================================================

                c.Modified()
                c.Update()

                # =========================================================
                # Save ONLY PDF
                # =========================================================

                output_pdf = (
                    outdir /
                    f"mass_comparison_{region}_WR{wr_mass}_N{nmass}.pdf"
                )

                c.SaveAs(str(output_pdf))

                print(f"    Saved: {output_pdf}")

                # =========================================================
                # Cleanup
                # =========================================================

                c.Close()

                del h_corrected
                del h_original
                del legend
                del c

            except Exception as e:

                print(f"    ERROR: {e}")
                print("    Skipping this mass point.")

                continue

# =========================================================
# Done
# =========================================================

print(f"\n{'=' * 70}")
print(f"All plots saved to: {outdir}")
print(f"{'=' * 70}")