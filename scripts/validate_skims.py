#!/usr/bin/env python3
"""
compare_skimmed_unskimmed.py

Compare histogram integrals between skimmed and unskimmed ROOT files to
validate that the skim does not alter physics distributions.

For each sample the script compares the integral (sum of bin values) of
every region histogram present in both files.

Usage (run from the WR_Plotter directory):
    # Using subdirectory names under rootfiles/<run>/<year>/<era>/:
    python scripts/compare_skimmed_unskimmed.py \
        --skimmed   20260212_SKIMMED \
        --unskimmed 20260212_UNSKIMMED

    # Using explicit absolute paths:
    python scripts/compare_skimmed_unskimmed.py \
        --skimmed   /full/path/to/SKIMMED \
        --unskimmed /full/path/to/UNSKIMMED

    # Specify a different era (default RunIII2024Summer24):
    python scripts/compare_skimmed_unskimmed.py \
        --era RunIII2024Summer24 \
        --skimmed 20260212_SKIMMED --unskimmed 20260212_UNSKIMMED
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wrplotter.paths import repo_root
from wrplotter.config import load_lumi, list_eras
from wrplotter.cli_utils import setup_logging

logger = logging.getLogger(__name__)

RTOL = 1e-6

# ── helpers ──────────────────────────────────────────────────────────────────

def discover_samples(directory):
    """Return dict of sample_name -> Path for WRAnalyzer_*.root files."""
    samples = {}
    for p in sorted(directory.glob("WRAnalyzer_*.root")):
        name = p.stem.replace("WRAnalyzer_", "")
        samples[name] = p
    return samples


def list_hist_keys(root_file):
    """Return all TH1D keys inside a ROOT file."""
    with uproot.open(root_file) as f:
        return sorted(
            k for k, cls in f.classnames().items()
            if cls in ("TH1D", "TH1F")
        )


def read_values(root_file, key):
    """Read histogram values as a numpy array."""
    with uproot.open(root_file) as f:
        h = f[key]
        return h.values(flow=False)


# ── comparison logic ─────────────────────────────────────────────────────────

def compare_histograms(skim_path, unskim_path, sample_name="", verbose=False):
    """
    Compare all region histograms between two ROOT files by integral.

    Returns (n_compared, n_match, n_mismatch, details_list, worst_diff).
    worst_diff is a tuple (abs_delta, sample, key, integral_s, integral_u) or None.
    """
    skim_keys = set(list_hist_keys(skim_path))
    unskim_keys = set(list_hist_keys(unskim_path))

    # Exclude cutflow histograms — only compare region histograms
    region_keys_skim = {k for k in skim_keys if not k.startswith("cutflow")}
    region_keys_unskim = {k for k in unskim_keys if not k.startswith("cutflow")}

    common = sorted(region_keys_skim & region_keys_unskim)
    only_skim = sorted(region_keys_skim - region_keys_unskim)
    only_unskim = sorted(region_keys_unskim - region_keys_skim)

    details = []
    if only_skim:
        details.append(f"  Keys only in SKIMMED: {only_skim}")
    if only_unskim:
        details.append(f"  Keys only in UNSKIMMED: {only_unskim}")

    n_compared = 0
    n_match = 0
    n_mismatch = 0
    worst_diff = None  # (abs_delta, sample, key, integral_s, integral_u)

    for key in common:
        vals_s = read_values(skim_path, key)
        vals_u = read_values(unskim_path, key)
        n_compared += 1

        integral_s = float(np.sum(vals_s))
        integral_u = float(np.sum(vals_u))
        abs_delta = abs(integral_s - integral_u)

        if np.isclose(integral_s, integral_u, rtol=RTOL, atol=0):
            n_match += 1
        else:
            n_mismatch += 1
            details.append(
                f"  MISMATCH {key}: "
                f"skimmed={integral_s:.4f}  unskimmed={integral_u:.4f}  "
                f"delta={integral_s - integral_u:+.4f}"
            )

        if worst_diff is None or abs_delta > worst_diff[0]:
            worst_diff = (abs_delta, sample_name, key, integral_s, integral_u)

    return n_compared, n_match, n_mismatch, details, worst_diff


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare skimmed vs unskimmed ROOT files by histogram integral.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skimmed", required=True,
        help="Path or subdirectory name for the skimmed ROOT files.",
    )
    parser.add_argument(
        "--unskimmed", required=True,
        help="Path or subdirectory name for the unskimmed ROOT files.",
    )
    parser.add_argument(
        "--era", default="RunIII2024Summer24", choices=list_eras(),
        help="Era (used to resolve relative subdirectory names).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show keys only in one file.",
    )
    args = parser.parse_args()

    setup_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve paths
    era_info = load_lumi(args.era)
    run = era_info["run"]
    year = str(era_info["year"])

    def resolve_dir(path_str):
        p = Path(path_str)
        if p.is_dir():
            return p
        candidate = (
            repo_root() / "rootfiles" / run / year / args.era / path_str
        )
        if candidate.is_dir():
            return candidate
        logger.error("Directory not found: %s (tried %s)", path_str, candidate)
        sys.exit(1)

    skim_dir = resolve_dir(args.skimmed)
    unskim_dir = resolve_dir(args.unskimmed)

    skim_samples = discover_samples(skim_dir)
    unskim_samples = discover_samples(unskim_dir)

    common_samples = sorted(set(skim_samples) & set(unskim_samples))
    only_skim = sorted(set(skim_samples) - set(unskim_samples))
    only_unskim = sorted(set(unskim_samples) - set(skim_samples))

    print(f"SKIMMED directory:   {skim_dir}")
    print(f"UNSKIMMED directory: {unskim_dir}")
    print(f"Common samples:      {len(common_samples)}")
    if only_skim:
        print(f"Only in SKIMMED:     {only_skim}")
    if only_unskim:
        print(f"Only in UNSKIMMED:   {only_unskim}")
    print()

    # Accumulators
    total_compared = 0
    total_match = 0
    total_mismatch = 0
    sample_summaries = []
    global_worst = None  # (abs_delta, sample, key, integral_s, integral_u)

    for sample in common_samples:
        skim_path = skim_samples[sample]
        unskim_path = unskim_samples[sample]

        n_comp, n_match, n_mismatch, details, worst = compare_histograms(
            skim_path, unskim_path, sample_name=sample, verbose=args.verbose,
        )
        total_compared += n_comp
        total_match += n_match
        total_mismatch += n_mismatch

        if worst is not None and (global_worst is None or worst[0] > global_worst[0]):
            global_worst = worst

        sample_summaries.append((sample, n_comp, n_match, n_mismatch, details))

    # ── Summary table ──
    sep = "=" * 80
    print(sep)
    print("SUMMARY")
    print(sep)
    print(f"Pass criterion: relative tolerance = {RTOL:.0e}  (|skim - unskim| <= {RTOL:.0e} * |unskim|)")
    print()
    header = f"{'Sample':<35s} {'Compared':>9s} {'Match':>6s} {'Mismatch':>9s}"
    print(header)
    print("-" * len(header))
    for sample, nc, nm, nmm, details in sample_summaries:
        flag = " <-- MISMATCH" if nmm > 0 else ""
        print(f"{sample:<35s} {nc:>9d} {nm:>6d} {nmm:>9d}{flag}")
        if args.verbose:
            for line in details:
                print(line)
    print("-" * len(header))
    print(f"{'TOTAL':<35s} {total_compared:>9d} {total_match:>6d} {total_mismatch:>9d}")
    print()

    if global_worst is not None:
        abs_d, w_sample, w_key, w_s, w_u = global_worst
        print(
            f"Largest difference: {abs_d:.6f}  "
            f"(sample={w_sample}, hist={w_key}, "
            f"skimmed={w_s:.4f}, unskimmed={w_u:.4f})"
        )
        print()

    all_pass = total_mismatch == 0
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
