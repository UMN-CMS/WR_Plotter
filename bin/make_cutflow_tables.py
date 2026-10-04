#!/usr/bin/env python3
"""Write analyzer cutflow histograms comparing EGamma vs Muon as LaTeX tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import uproot


ERA_DETAILS = {
    "RunIISummer20UL18": ("RunII", "2018"),
    "Run3Summer22": ("Run3", "2022"),
    "Run3Summer22EE": ("Run3", "2022"),
    "Run3Summer23": ("Run3", "2023"),
    "Run3Summer23BPix": ("Run3", "2023"),
    "RunIII2024Summer24": ("Run3", "2024"),
    "RunIII2025": ("Run3", "2025"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era", required=True, choices=sorted(ERA_DETAILS))
    parser.add_argument(
        "--dir",
        default="",
        help="Read ROOT files from an additional era subdirectory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for LaTeX files (default: WR_Plotter/cutflows/<era path>).",
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


def cut_labels(histogram) -> list[str]:
    axis = histogram.axis(0)
    return [str(label) for label in axis.labels()]


def histogram_values(root_file, path: str):
    try:
        histogram = root_file[path]
    except (KeyError, uproot.KeyInFileError):
        return None
    return histogram, cut_labels(histogram), histogram.values(flow=False)


def table_for_comparison(
    title: str,
    egamma_data,
    muon_data,
    value_type: str,
) -> str:
    _, labels, egamma_values = egamma_data
    _, _, muon_values = muon_data
    
    lines = [
        rf"\begin{{table}}[htbp]",
        rf"\centering",
        rf"\caption{{{latex_escape(title)} ({value_type.capitalize()})}}",
        rf"\begin{{tabular}}{{lrr}}",
        rf"\hline",
        rf"Cut & EGamma & Muon \\",
        rf"\hline",
    ]
    for label, val_eg, val_mu in zip(labels, egamma_values, muon_values):
        lines.append(
            f"{latex_escape(label)} & {val_eg:.6g} & {val_mu:.6g} \\\\"
        )
    lines.extend([
        rf"\hline",
        rf"\end{{tabular}}",
        rf"\end{{table}}",
        "",
    ])
    return "\n".join(lines)


def process_stream_pair(
    egamma_root_path: Path,
    muon_root_path: Path,
    output_dir: Path,
    sample_tag: str,
) -> bool:
    weighted_sections = []
    unweighted_sections = []

    with uproot.open(egamma_root_path) as file_eg, uproot.open(muon_root_path) as file_mu:
        for topology in ("resolved", "boosted"):
            prefix = f"cutflow_{topology}"
            if prefix not in file_eg or prefix not in file_mu:
                continue

            for flavor in ("ee", "mumu", "em"):
                base = f"{prefix}/{flavor}"

                # 1. Weighted Comparison
                weighted_eg = histogram_values(file_eg, f"{base}/cumulative")
                weighted_mu = histogram_values(file_mu, f"{base}/cumulative")
                if weighted_eg is not None and weighted_mu is not None:
                    weighted_sections.append(
                        table_for_comparison(
                            f"{sample_tag}: {topology} {flavor}",
                            weighted_eg,
                            weighted_mu,
                            "weighted",
                        )
                    )

                # 2. Unweighted Comparison
                unweighted_eg = histogram_values(file_eg, f"{base}/cumulative_unweighted")
                unweighted_mu = histogram_values(file_mu, f"{base}/cumulative_unweighted")
                if unweighted_eg is not None and unweighted_mu is not None:
                    unweighted_sections.append(
                        table_for_comparison(
                            f"{sample_tag}: {topology} {flavor}",
                            unweighted_eg,
                            unweighted_mu,
                            "unweighted",
                        )
                    )

    if not weighted_sections and not unweighted_sections:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write separate output files for weighted and unweighted tables
    for mode, sections in [("weighted", weighted_sections), ("unweighted", unweighted_sections)]:
        if not sections:
            continue
        output_path = output_dir / f"{sample_tag}_EGamma_vs_Muon_cutflows_{mode}.tex"
        document = [
            r"\documentclass{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\begin{document}",
            "",
            *sections,
            r"\end{document}",
            "",
        ]
        output_path.write_text("\n".join(document), encoding="utf-8")

    return True


def run_for_directory(input_dir: Path, output_dir: Path) -> int:
    """Matches corresponding EGamma and Muon files in an input directory."""
    egamma_files = sorted(input_dir.glob("WRAnalyzer_EGamma*.root"))
    if not egamma_files:
        return 0

    written = 0
    for eg_path in egamma_files:
        # Match EGamma file with its corresponding Muon dataset counter-part
        muon_filename = eg_path.name.replace("EGamma", "Muon", 1)
        if muon_filename == eg_path.name:
            muon_filename = eg_path.name.replace("EGamma", "SingleMuon", 1)
            
        mu_path = input_dir / muon_filename
        if not mu_path.exists():
            print(f"Skipping {eg_path.name}: matching Muon file {muon_filename} not found in {input_dir}")
            continue

        sample_tag = eg_path.stem.replace("EGamma_", "").replace("WRAnalyzer_", "")
        if process_stream_pair(eg_path, mu_path, output_dir, sample_tag):
            print(f"Wrote EGamma vs Muon cutflow tables for {sample_tag} ({output_dir.name})")
            written += 1

    return written


def main() -> None:
    args = parse_args()
    run, year = ERA_DETAILS[args.era]
    plotter_dir = Path(__file__).resolve().parents[1]

    base_input_dir = plotter_dir / "rootfiles" / run / year / args.era
    if args.dir:
        base_input_dir /= args.dir

    base_output_dir = args.output_dir or (
        plotter_dir / "cutflows" / run / year / args.era / (args.dir or "")
    )

    total_written = 0

    # 1. Process Skimmed directory -> outputs to 'skimmed' folder
    skimmed_input = base_input_dir
    skimmed_output = base_output_dir / "skimmed"
    total_written += run_for_directory(skimmed_input, skimmed_output)

    # 2. Process Unskimmed directory -> outputs to 'unskimmed' folder
    unskimmed_input = base_input_dir / "unskimmed"
    if unskimmed_input.exists():
        unskimmed_output = base_output_dir / "unskimmed"
        total_written += run_for_directory(unskimmed_input, unskimmed_output)

    if not total_written:
        raise SystemExit("No cutflow tables were produced")


if __name__ == "__main__":
    main()