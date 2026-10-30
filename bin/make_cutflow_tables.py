#!/usr/bin/env python3
"""
Generate region-based cutflow LaTeX tables combining all MC samples (2024)
and Data samples (2025).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import uproot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mc-dir",
        type=Path,
        default=Path("rootfiles/Run3/2024/RunIII2024Summer24"),
        help="Path to 2024 MC ROOT files directory",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("rootfiles/Run3/2025/RunIII2025/unskimmed"),
        help="Path to 2025 Data ROOT files directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cutflows/2025_Data_vs_2024_MC"),
        help="Output directory for generated LaTeX files",
    )
    return parser.parse_args()


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)


def get_hist_data(root_file, path: str):
    """Retrieve cut labels and values array for a given histogram path."""
    try:
        hist = root_file[path]
        labels = [str(lbl) for lbl in hist.axis(0).labels()]
        values = hist.values(flow=False)
        return labels, np.array(values, dtype=float)
    except (KeyError, uproot.KeyInFileError):
        return None, None


def read_and_sum_histograms(file_paths: list[Path], hist_subpath: str):
    """
    Sum histogram bin contents across multiple ROOT files.
    Returns (labels, total_values_array, dict_of_individual_contributions).
    """
    total_values = None
    common_labels = None
    individual_contribs = {}

    for filepath in file_paths:
        if not filepath.exists():
            continue
        
        sample_name = filepath.stem.replace("WRAnalyzer_", "")
        with uproot.open(filepath) as f:
            labels, values = get_hist_data(f, hist_subpath)
            if values is None:
                continue

            individual_contribs[sample_name] = values
            if total_values is None:
                total_values = np.zeros_like(values)
                common_labels = labels
            
            total_values += values

    return common_labels, total_values, individual_contribs


def make_latex_table(region_name: str, labels: list[str], mc_contribs: dict, mc_total: np.ndarray, data_total: np.ndarray, mode: str) -> str:
    """Format individual MC processes, total MC, and Data into a LaTeX table."""
    mc_samples = sorted(mc_contribs.keys())
    
    # Headers
    headers = ["Cut"] + ["Total MC (2024)", "Data (2025)"]
    col_align = "l" + "r" * (len(headers) - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Cutflow for {latex_escape(region_name)} ({mode.capitalize()})}}",
        rf"\begin{{tabular}}{{{col_align}}}",
        r"\hline",
        " & ".join(headers) + r" \\",
        r"\hline",
    ]

    for i, label in enumerate(labels):
        row_vals = []
        # for s in mc_samples:
        #     row_vals.append(f"{mc_contribs[s][i]:.6g}")
        row_vals.append(f"{mc_total[i]:.6g}")
        
        data_val_str = f"{data_total[i]:.6g}" if data_total is not None else "N/A"
        row_vals.append(data_val_str)

        row_str = f"{latex_escape(label)} & " + " & ".join(row_vals) + r" \\"
        lines.append(row_str)

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])

    return "\n".join(lines)


def process_topology(mc_files: list[Path], data_files: list[Path], topology: str, output_dir: Path):
    """Process resolved or boosted topology and produce LaTeX document."""
    
    # Discover available regions from any valid MC file
    sample_mc = next((p for p in mc_files if p.exists()), None)
    if not sample_mc:
        print(f"No MC ROOT files found for topology: {topology}")
        return

    by_region_path = f"cutflow_{topology}/by_region"
    with uproot.open(sample_mc) as f:
        if by_region_path not in f:
            print(f"Path '{by_region_path}' not found in {sample_mc.name}")
            return
        regions = [k.split(";")[0] for k in f[by_region_path].keys(cycle=False)]

    for mode in ["cumulative", "cumulative_unweighted"]:
        tables = []
        for region in sorted(set(regions)):
            hist_subpath = f"{by_region_path}/{region}/{mode}"

            labels, mc_total, mc_contribs = read_and_sum_histograms(mc_files, hist_subpath)
            _, data_total, _ = read_and_sum_histograms(data_files, hist_subpath)

            if labels is None or mc_total is None:
                continue

            table_tex = make_latex_table(
                region_name=region,
                labels=labels,
                mc_contribs=mc_contribs,
                mc_total=mc_total,
                data_total=data_total,
                mode="weighted" if mode == "cumulative" else "unweighted",
            )
            tables.append(table_tex)

        if tables:
            mode_suffix = "weighted" if mode == "cumulative" else "unweighted"
            output_file = output_dir / f"cutflow_{topology}_{mode_suffix}_unskimmed.tex"
            
            tex_doc = [
                r"\documentclass{article}",
                r"\usepackage[margin=0.5in]{geometry}",
                r"\usepackage{booktabs}",
                r"\begin{document}",
                f"\\section*{{Cutflow Tables - {topology.capitalize()} ({mode_suffix.capitalize()})}}",
                "",
                *tables,
                r"\end{document}",
            ]
            output_file.write_text("\n".join(tex_doc), encoding="utf-8")
            print(f"Generated: {output_file}")


def main():
    args = parse_args()

    # Define file paths based on directory structure
    mc_files = [
        args.mc_dir / "WRAnalyzer_DYJets.root",
        args.mc_dir / "WRAnalyzer_Nonprompt.root",
        args.mc_dir / "WRAnalyzer_Other.root",
        args.mc_dir / "WRAnalyzer_tt_tW.root",
    ]

    data_files = [
        args.data_dir / "WRAnalyzer_EGamma.root",
        args.data_dir / "WRAnalyzer_Muon.root",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate tables for both boosted and resolved
    for topology in ["resolved", "boosted"]:
        process_topology(mc_files, data_files, topology, args.output_dir)


if __name__ == "__main__":
    main()