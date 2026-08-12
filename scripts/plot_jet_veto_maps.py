"""
Plot jet veto maps from correctionlib JSON files.
Produces a multi-panel figure showing vetoed (eta, phi) regions for each era.
"""

import gzip
from pathlib import Path

import correctionlib
import matplotlib.pyplot as plt
import numpy as np

# Base path to the veto map files (relative to repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]

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
    n_eras = len(VETO_MAPS)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (era, (json_rel, corr_name)) in enumerate(VETO_MAPS.items()):
        json_path = REPO_ROOT / json_rel
        if not json_path.exists():
            print(f"Skipping {era}: {json_path} not found")
            axes[idx].set_title(f"{era} (not found)")
            continue

        print(f"Loading {era}...")
        eta_bins, phi_bins, veto_map_2d = load_veto_map(json_path, corr_name)

        ax = axes[idx]
        im = ax.pcolormesh(eta_bins, phi_bins, veto_map_2d, cmap="Reds", shading="auto")
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$\phi$")
        ax.set_title(era)
        fig.colorbar(im, ax=ax, label="Veto flag")

    # Hide unused subplots
    for idx in range(n_eras, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Jet Veto Maps", fontsize=16)
    plt.tight_layout()

    outpath = REPO_ROOT / "WR_Plotter" / "plots" / "jet_veto_maps.png"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print(f"Saved to {outpath}")
    plt.show()


if __name__ == "__main__":
    main()
