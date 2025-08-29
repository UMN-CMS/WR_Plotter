#!/usr/bin/env python3
"""
compare_bg_2d.py

Plot a single background at a time (DYJets, then TTbar) for the 2D histogram
'met_vs_mll_1p5TeV' (MET vs m_ll), reading ROOT files from the 'met2d' directory.
If era='2022', it combines Run3Summer22 and Run3Summer22EE using their respective
luminosities. Outputs one PDF per region & sample under EOS with a 'met2d' path.

Only bins with m_ll > 150 GeV are drawn.
Now includes a small 2D rebin (default 2x2).
"""

import argparse
import logging
from pathlib import Path
import sys

import uproot
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

# allow imports of Plotter & Region
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
from python.plotter import Plotter, Region
from src import save_figure

plt.rcParams.update({"savefig.dpi": 300})

# where your ROOT files live
WORKING_DIR = Path('/uscms/home/bjackson/nobackup/WrCoffea/WR_Plotter')

# EOS output layout anchors
RUN  = "Run3"
YEAR = "2022"

# era → run/year/lumi
ERA_CONFIG = {
    "Run3Summer22":   {"run": "Run3", "year": "2022", "lumi":  7.9804},
    "Run3Summer22EE": {"run": "Run3", "year": "2022", "lumi": 26.6717},
    "2022":           {"run": "Run3", "year": "2022", "lumi":  7.9804 + 26.6717},
}

# explicit filenames for samples we draw
SAMPLE_TO_FILE = {
    "DYJets": "WRAnalyzer_DYJets.root",
    "TTbar":  "WRAnalyzer_TTbar.root",
}

# fixed subdirectory and settings
INPUT_SUBDIR   = "met2d"
MLL_MIN_GEV    = 150.0      # only plot m_ll > 150 GeV
REBIN_X, REBIN_Y = 1,1    # light rebin (x=MET, y=m_ll)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def parse_args():
    p = argparse.ArgumentParser(description="Plot single 2D hist (met_vs_mll_1p5TeV) per background with mll > 150 GeV")
    p.add_argument(
        "--era", "-e",
        dest="era",
        type=str,
        choices=list(ERA_CONFIG.keys()),
        default="2022",
        help="Era to process (default: 2022). Options: Run3Summer22, Run3Summer22EE, or 2022",
    )
    return p.parse_args()

def load_histogram(path: Path, hist_key: str):
    with uproot.open(path) as f:
        return f[hist_key].to_hist()

def load_and_combine_sample_2d(plotter: Plotter, sample: str, hist_key: str):
    """
    For each input_directory (possibly multiple eras when era == '2022'),
    load WRAnalyzer_<sample>.root directly in that directory (no subdirs),
    scale to the sub-lumi, and sum. No rebinning yet (we rebin arrays later).
    """
    filename = SAMPLE_TO_FILE[sample]
    combined  = None
    orig_lumi = plotter.lumi

    for indir, sublumi in zip(plotter.input_directory, plotter.input_lumis):
        fp = indir / filename
        try:
            raw = load_histogram(fp, hist_key)
        except Exception as exc:
            logging.warning(f"Missing or unreadable file for {sample}: {fp} ({exc}); continuing.")
            continue
        plotter.lumi = sublumi  # scale each part to its own lumi
        h = plotter.scale_hist(raw)
        combined = h if combined is None else (combined + h)

    plotter.lumi = orig_lumi
    return combined

def rebin2d_arrays(vals, xedges, yedges, rx=1, ry=1):
    """
    Rebin a 2D array by integer factors (sum of contents).
    vals shape: (nx, ny); xedges length nx+1; yedges length ny+1.
    Returns (vals_rb, xedges_rb, yedges_rb).
    """
    nx, ny = vals.shape
    # trim overflow if not divisible (shouldn't happen for 200 bins, but safe)
    nx_trim = (nx // rx) * rx
    ny_trim = (ny // ry) * ry
    vals = vals[:nx_trim, :ny_trim]

    # reshape and sum
    vals_rb = vals.reshape(nx_trim // rx, rx, ny_trim // ry, ry).sum(axis=(1, 3))

    # build rebinned edges (keep last edge exact)
    xedges_rb = xedges[::rx]
    if xedges_rb[-1] != xedges[-1]:
        xedges_rb = np.concatenate([xedges_rb, xedges[-1:]])
    yedges_rb = yedges[::ry]
    if yedges_rb[-1] != yedges[-1]:
        yedges_rb = np.concatenate([yedges_rb, yedges[-1:]])

    return vals_rb, xedges_rb, yedges_rb

def plot_heatmap(plotter: Plotter, region: Region, h2, sample_name, mll_min=MLL_MIN_GEV, rx=REBIN_X, ry=REBIN_Y,):
    """
    Draw a 2D heatmap after integer rebin (rx, ry),
    keeping only m_ll > mll_min, and **flipping axes**:
        x-axis = m_ll,   y-axis = MET
    """
    hep.style.use("CMS")

    # (optional) keep your smaller fonts if you added them earlier:
    # plt.rcParams.update({...})

    fig, ax = plt.subplots(1, 1)

    # original edges and values: axis0 = MET (x), axis1 = m_ll (y)
    met_edges = h2.axes[0].edges
    mll_edges = h2.axes[1].edges
    vals      = h2.values()  # shape (n_met, n_mll)

    # light integer rebin
    vals_rb, met_edges_rb, mll_edges_rb = rebin2d_arrays(vals, met_edges, mll_edges, rx=rx, ry=ry)

    # cut on m_ll > threshold using **rebinned** m_ll edges (lower edge >= mll_min)
    ny_rb = vals_rb.shape[1]
    j_start = int(np.searchsorted(mll_edges_rb[:-1], mll_min, side="left"))
    if j_start >= ny_rb:
        logging.error(f"No m_ll bins above {mll_min} GeV after rebin; nothing to draw.")
        plt.close(fig)
        return None

    # ----- FLIP AXES -----
    # x-axis := m_ll (after cut); y-axis := MET
    xedges = mll_edges_rb[j_start:]   # m_ll edges (selected)
    yedges = met_edges_rb             # MET edges
    Z = vals_rb[:, j_start:]          # shape (n_met_rb, n_mll_sel) matches (len(yedges)-1, len(xedges)-1)

    mesh = ax.pcolormesh(xedges, yedges, Z, shading="auto")

#    mesh = ax.pcolormesh(
#        xedges, yedges, Z,
#        shading="nearest",        # no interpolated seams
#        antialiased=False,        # don't draw anti-aliased cell edges
#        linewidth=0.0,            # no borders
#        edgecolors="none",
#        rasterized=True,          # render as an image inside the PDF
#    )
    cbar = fig.colorbar(mesh, ax=ax)
#    cbar.set_label("Events")

    ax.set_xlabel(r"$m_{\ell\ell}$ [GeV]")
    ax.set_ylabel(r"$E_{T}^{\text{miss}}$ [GeV]")
    ax.set_xlim(max(mll_min, xedges[0]), xedges[-1])
    ax.set_ylim(0, 500)
    ax.set_xlim(150, 1000)
    # Top-right label instead of a title

    corner_text = "\n".join([
        region.tlatex_alias,
        rf"$m_{{\ell\ell}} > {mll_min:.0f}$ GeV",
    ])

    ax.text(
        0.02, 0.98, corner_text,
        transform=ax.transAxes,
        ha="left", va="top",
        color="white", fontsize=20,
        # bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.35, edgecolor="none"),
    )

    sample_label = {"DYJets": "DY+Jets", "TTbar": r"$t\bar{t}$"}.get(sample_name, sample_name)
    ax.text(
        0.98, 0.98, sample_label,
        transform=ax.transAxes, ha="right", va="top",
        color="white", fontsize=20,
    )

#    ax.set_title(f"{region.tlatex_alias}\n$m_{{\\ell\\ell}} > {mll_min:.0f}$ GeV  •  rebin {rx}×{ry}")
    hep.cms.label(
        loc=0,
        ax=ax,
        data=True,
        label="Work in Progress",
        lumi=f"{plotter.lumi:.1f}",
        com=13.6,
        fontsize=20,
    )
    return fig


def main():
    args = parse_args()
    setup_logging()

    # initialize Plotter
    plotter      = Plotter()
    cfg0         = ERA_CONFIG[args.era]
    plotter.run  = cfg0["run"]
    plotter.year = cfg0["year"]
    plotter.era  = args.era
    plotter.lumi = cfg0["lumi"]
    plotter.scale = True

    # define Regions (same as your SRs)
    plotter.regions_to_draw = [
        Region('wr_mumu_resolved_sr', 'Muon',   unblind_data=True,
               tlatex_alias=f"$\\mu\\mu$\nResolved SR\n{plotter.era}"),
        Region('wr_ee_resolved_sr',   'EGamma', unblind_data=True,
               tlatex_alias=f"ee\nResolved SR\n{plotter.era}"),
    ]

    # input directories (+ lumis)
    if args.era == "2022":
        d1 = WORKING_DIR / 'rootfiles' / plotter.run / plotter.year / 'Run3Summer22'   / INPUT_SUBDIR
        d2 = WORKING_DIR / 'rootfiles' / plotter.run / plotter.year / 'Run3Summer22EE' / INPUT_SUBDIR
        plotter.input_directory = [d1, d2]
        plotter.input_lumis     = [
            ERA_CONFIG["Run3Summer22"]["lumi"],
            ERA_CONFIG["Run3Summer22EE"]["lumi"],
        ]
    else:
        base = WORKING_DIR / 'rootfiles' / plotter.run / plotter.year / args.era / INPUT_SUBDIR
        plotter.input_directory = [base]
        plotter.input_lumis     = [cfg0["lumi"]]

    var2d_name = "met_vs_mll_1p5TeV"

    # iterate samples one at a time (DYJets first, then TTbar)
    for sample in ["DYJets", "TTbar"]:
        for region in plotter.regions_to_draw:
            logging.info(f"[{sample}] Region: {region.name}")

            hist_key = f"{region.name}/{var2d_name}_{region.name}"
            h2 = load_and_combine_sample_2d(plotter, sample, hist_key)

            if h2 is None:
                logging.error(f"{var2d_name}: missing {sample}; skipping.")
                continue

            fig = plot_heatmap(plotter, region, h2, sample, mll_min=MLL_MIN_GEV, rx=REBIN_X, ry=REBIN_Y)
            if fig is None:
                continue

            outbase = (Path(f"/eos/user/w/wijackso/{RUN}/{YEAR}/{plotter.era}")
                       / INPUT_SUBDIR
                       / f"{region.name}_{region.primary_dataset}"
                       / f"{sample}_{var2d_name}_reb{REBIN_X}x{REBIN_Y}_mllgt{int(MLL_MIN_GEV)}_{region.name}")

            save_figure(fig, outbase)  # writes + uploads outbase.pdf AND outbase.jpg

            plt.close(fig)

if __name__ == "__main__":
    main()
