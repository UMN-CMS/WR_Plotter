#!/usr/bin/env python3
"""Plot the new muon-corrected four-object mass histogram.

This script compares the original resolved four-object mass histogram
(`mass_fourobject`) with the muon-corrected mass histogram
(`mass_fourobj_muon_corr`) from a WR_Plotter ROOT file.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

hep.style.use("CMS")


def _hist_root_key(stem, region):
    folder = f"wr_{region}_resolved_sr"
    name = f"{stem}_wr_{region}_resolved_sr"
    return f"{folder}/{name}"


def _read_histogram(root, key):
    hist_obj = root[key]
    counts, edges = hist_obj.to_numpy()
    return np.asarray(counts, dtype=float), np.asarray(edges, dtype=float)


def _is_signal_region(region):
    region_lower = region.lower()
    return all(token not in region_lower for token in ["cr", "background", "bkg"])


def _rebin_histogram(counts, edges, factor=10):
    if factor <= 1 or counts.size <= 1:
        return counts, edges

    bin_count = counts.size
    if bin_count % factor == 0:
        rebinned_counts = np.add.reduceat(counts, np.arange(0, bin_count, factor))
        rebinned_edges = edges[::factor]
        if rebinned_edges.size == rebinned_counts.size:
            rebinned_edges = np.append(rebinned_edges, edges[-1])
    else:
        indices = np.arange(0, bin_count, factor)
        rebinned_counts = np.add.reduceat(counts, indices)
        rebinned_edges = edges[indices]
        if rebinned_edges[-1] != edges[-1]:
            rebinned_edges = np.append(rebinned_edges, edges[-1])

    return rebinned_counts, rebinned_edges


def _make_plot_title(mass_point, plot_name):
    return f"{mass_point}_{plot_name}"


def _plot_histograms(counts_orig, edges_orig, counts_corr, edges_corr, output_path, normalize=False, whichone="Generic", xlabel=r"$m_{\ell\ell jj}$ [GeV]", ylabel="Events"):
    if normalize:
        if counts_orig.sum() > 0:
            counts_orig = counts_orig / counts_orig.sum()
        if counts_corr.sum() > 0:
            counts_corr = counts_corr / counts_corr.sum()

    fig, (ax_main, ax_ratio) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        figsize=(10, 8),
    )

    ax_main.step(edges_orig[:-1], counts_orig, where="post", label="Original", linewidth=1.5)
    ax_main.step(edges_corr[:-1], counts_corr, where="post", label="Muon-corrected", linewidth=1.5)
    ax_main.set_ylabel(ylabel)
    ax_main.set_title(whichone)
    ax_main.legend(loc="best", fontsize=10)
    ax_main.grid(True, alpha=0.3)

    ratio = np.full_like(counts_orig, np.nan)
    mask = counts_orig > 0
    if mask.any():
        ratio[mask] = counts_corr[mask] / counts_orig[mask]

    ax_ratio.step(edges_orig[:-1], ratio, where="post", color="black")
    ax_ratio.set_xlabel(xlabel)
    ax_ratio.set_ylabel("Corr / Orig")
    ax_ratio.set_ylim(0.0, 2.0)
    ax_ratio.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_single_histogram(counts, edges, output_path, normalize=False, whichone="Generic", xlabel="Value", ylabel="Events"):
    if normalize and counts.sum() > 0:
        counts = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.step(edges[:-1], counts, where="post", linewidth=1.5, color="C0")
    ax.set_title(whichone)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare the original four-object histograms with the new muon-corrected variants "
            "for all WRAnalyzer ROOT files in a WR_Plotter rootfiles directory."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="rootfiles/Run3/2024/RunIII2024Summer24",
        help="Path to the WR_Plotter rootfiles directory containing WRAnalyzer_*.root files.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save comparison plots (default: ./output_muon).",
    )
    parser.add_argument(
        "--regions",
        nargs="*",
        default=["mumu"],
        help="List of signal regions to plot (default: mumu).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize histograms to unit area.",
    )
    args = parser.parse_args()

    base_dir = Path(args.input_dir)
    if not base_dir.exists() or not base_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {base_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd() / "output_muon"
    output_dir.mkdir(parents=True, exist_ok=True)

    root_files = sorted(base_dir.glob("WRAnalyzer_*.root"))
    if not root_files:
        raise FileNotFoundError(f"No WRAnalyzer_*.root files found under {base_dir}")

    for root_file in root_files:
        with uproot.open(str(root_file)) as root:
            for region in args.regions:
                if not _is_signal_region(region):
                    print(f"Skipping non-signal region: {region}")
                    continue

                comparisons = [
                    ("mass_fourobject", "mass_fourobj_muon_corr", r"$m_{\ell\ell jj}$ [GeV]"),
                    ("mass_fourobject", "mass_fourobj_muon_corr_exact4", r"$m_{\ell\ell jj}$ [GeV]"),
                    ("pt_fourobject", "pt_total_fourobj_muon_corr", r"$p_{T}^{\mathrm{tot}}$ [GeV]"),
                    ("pt_fourobject", "pt_total_fourobj_muon_corr_exact4", r"$p_{T}^{\mathrm{tot}}$ [GeV]"),
                ]

                for orig_stem, corr_stem, xlabel in comparisons:
                    orig_key = _hist_root_key(orig_stem, region)
                    corr_key = _hist_root_key(corr_stem, region)

                    if orig_key not in root:
                        print(f"Skipping {root_file.name} region={region}: missing {orig_key}")
                        continue
                    if corr_key not in root:
                        print(f"Skipping {root_file.name} region={region}: missing {corr_key}")
                        continue

                    counts_orig, edges_orig = _read_histogram(root, orig_key)
                    counts_corr, edges_corr = _read_histogram(root, corr_key)

                    counts_orig, edges_orig = _rebin_histogram(counts_orig, edges_orig, factor=20)
                    counts_corr, edges_corr = _rebin_histogram(counts_corr, edges_corr, factor=20)

                    outpath = output_dir / f"{root_file.stem}_{region}_resolved_{orig_stem}_vs_{corr_stem}.pdf"
                    plot_name = "fourobjmass" if orig_stem.startswith("mass") else "totalpt"
                    if corr_stem.endswith("exact4"):
                        plot_name = f"{plot_name}_exact4"
                    _plot_histograms(
                        counts_orig,
                        edges_orig,
                        counts_corr,
                        edges_corr,
                        outpath,
                        normalize=args.normalize,
                        whichone=_make_plot_title(root_file.stem[11:], plot_name),
                        xlabel=xlabel,
                        ylabel="Events",
                    )
                    print(f"Saved comparison plot to: {outpath}")

                for param, label in [("s1", r"$s_{l1}$"), ("s2", r"$s_{l2}$"), ("s3", r"$s_{j1}$"), ("s4", r"$s_{j2}$")]:
                    corr_key = _hist_root_key(f"{param}_correction", region)
                    if corr_key not in root:
                        print(f"Skipping {root_file.name} region={region}: missing {corr_key}")
                        continue

                    counts_corr, edges_corr = _read_histogram(root, corr_key)
                    counts_corr, edges_corr = _rebin_histogram(counts_corr, edges_corr, factor=20)

                    outpath = output_dir / f"{root_file.stem}_{region}_resolved_{param}_correction.pdf"
                    _plot_single_histogram(
                        counts_corr,
                        edges_corr,
                        outpath,
                        normalize=args.normalize,
                        whichone=_make_plot_title(root_file.stem[11:], label[1:-1].replace("_{", "_").replace("}", "")),
                        xlabel=rf"{label} correction factor",
                        ylabel="Events",
                    )
                    print(f"Saved correction-factor plot to: {outpath}")


if __name__ == "__main__":
    main()
