"""
Plot jet veto maps from correctionlib JSON files.
One figure per era, in the CMS style of the plotter: the vetoed (eta, phi) cells in one colour, the
label above the frame (the map fills it), into WR_Plotter/plots/jet_veto_maps/<era>.pdf.
"""

import gzip
import sys
from pathlib import Path

import correctionlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

# Base path to the veto map files (relative to repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "WR_Plotter"))
from wrplotter.config import era_label  # noqa: E402
from wrplotter.plotting_helpers import CMS_EXTRA_TEXT, FONT_SIZE_LABEL, FONT_SIZE_LEGEND  # noqa: E402

VETO_COLOR = "#440154"   # the dark end of viridis (CAT: viridis for 2D), legible in greyscale

# Era -> (json path, correction name)
VETO_MAPS = {
    "RunIISummer20UL18": (
        "data/jsonpog/JME/RunII/RunIISummer20UL18/jetvetomaps.json.gz",
        "Summer19UL18_V1",
    ),
    "Run3Summer22": (
        "data/jsonpog/JME/Run3/Run3Summer22/jetvetomaps.json.gz",
        "Summer22_23Sep2023_RunCD_V1",
    ),
    "Run3Summer22EE": (
        "data/jsonpog/JME/Run3/Run3Summer22EE/jetvetomaps.json.gz",
        "Summer22EE_23Sep2023_RunEFG_V1",
    ),
    "Run3Summer23": (
        "data/jsonpog/JME/Run3/Run3Summer23/jetvetomaps.json.gz",
        "Summer23Prompt23_RunC_V1",
    ),
    "Run3Summer23BPix": (
        "data/jsonpog/JME/Run3/Run3Summer23BPix/jetvetomaps.json.gz",
        "Summer23BPixPrompt23_RunD_V1",
    ),
    "RunIII2024Summer24": (
        "data/jsonpog/JME/Run3/RunIII2024Summer24/jetvetomaps.json.gz",
        "Summer24Prompt24_RunBCDEFGHI_V1",
    ),
}

# Grid resolution
N_ETA = 500
N_PHI = 500


def load_veto_map(json_path, correction_name):
    """Load a correctionlib veto map and evaluate on an (eta, phi) grid."""
    with gzip.open(json_path, "rt") as f:
        cset = correctionlib.CorrectionSet.from_string(f.read())

    corr = cset[correction_name]

    eta_bins = np.linspace(-5.0, 5.0, N_ETA)
    phi_bins = np.linspace(-np.pi, np.pi, N_PHI)
    eta_grid, phi_grid = np.meshgrid(eta_bins, phi_bins)

    veto_vals = corr.evaluate(
        "jetvetomap",
        eta_grid.flatten().astype(np.float64),
        phi_grid.flatten().astype(np.float64),
    )

    return eta_bins, phi_bins, veto_vals.reshape(eta_grid.shape)


def main():
    hep.style.use("CMS")
    out_dir = REPO_ROOT / "WR_Plotter" / "plots" / "jet_veto_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmap = ListedColormap(["white", VETO_COLOR])

    for era, (json_rel, corr_name) in VETO_MAPS.items():
        json_path = REPO_ROOT / json_rel
        if not json_path.exists():
            print(f"Skipping {era}: {json_path} not found")
            continue

        print(f"Loading {era}...")
        eta_bins, phi_bins, veto_map_2d = load_veto_map(json_path, corr_name)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.pcolormesh(eta_bins, phi_bins, (veto_map_2d > 0).astype(float), cmap=cmap,
                      norm=BoundaryNorm([0, 0.5, 1], cmap.N), shading="auto", rasterized=True)
        ax.set_xlim(eta_bins[0], eta_bins[-1])
        ax.set_ylim(phi_bins[0], phi_bins[-1])
        ax.set_xlabel(r"Jet $\eta$")
        ax.set_ylabel(r"Jet $\phi$")
        com = 13 if era.startswith("RunIISummer") else 13.6
        hep.cms.label(CMS_EXTRA_TEXT, loc=0, ax=ax, data=True, fontsize=FONT_SIZE_LABEL,
                      rlabel=f"{era_label(era)} ({com:g} TeV)")
        ax.legend(handles=[Patch(facecolor=VETO_COLOR, label=f"Vetoed ({corr_name})")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.11), frameon=False, fontsize=FONT_SIZE_LEGEND)
        fig.subplots_adjust(bottom=0.2)

        outpath = out_dir / f"{era}.pdf"
        fig.savefig(outpath)
        plt.close(fig)
        print(f"Saved to {outpath}")


if __name__ == "__main__":
    main()
