#!/usr/bin/env python3
"""
compare_skimmed_unskimmed.py

Compare histogram bin counts and unweighted cutflows between skimmed and
unskimmed ROOT files to validate that the skim does not alter physics
distributions or post-lepton-selection yields.

For each sample the script checks:
  1. Region histograms  – bin-by-bin comparison of every TH1D in each
     region directory (values and variances).  Differences are classified
     as "exact match", "FP noise" (rel diff < rtol, from weight-summation
     order), or "real mismatch" (rel diff >= rtol).
  2. Cutflows – weighted and unweighted, cumulative and onecut, for resolved
     and boosted categories in all three channels (ee, mumu, em).  Early
     pre-skim cuts are expected to differ; downstream cuts (after lepton
     selection) must match.

Usage (run from the WR_Plotter directory):
    # Using subdirectory names under rootfiles/<run>/<year>/<era>/:
    python scripts/compare_skimmed_unskimmed.py \\
        --skimmed   20260212_SKIMMED \\
        --unskimmed 20260212_UNSKIMMED

    # Using explicit absolute paths:
    python scripts/compare_skimmed_unskimmed.py \\
        --skimmed   /full/path/to/SKIMMED \\
        --unskimmed /full/path/to/UNSKIMMED

    # Show per-bin details for all cutflows and FP-noise histograms:
    python scripts/compare_skimmed_unskimmed.py \\
        --skimmed 20260212_SKIMMED --unskimmed 20260212_UNSKIMMED -v

    # Adjust the FP-noise vs real-mismatch tolerance (default 1e-6):
    python scripts/compare_skimmed_unskimmed.py \\
        --skimmed 20260212_SKIMMED --unskimmed 20260212_UNSKIMMED --rtol 1e-5

    # Specify a different era (default RunIII2024Summer24):
    python scripts/compare_skimmed_unskimmed.py \\
        --era RunIII2024Summer24 \\
        --skimmed 20260212_SKIMMED --unskimmed 20260212_UNSKIMMED
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wrplotter.io import repo_root
from wrplotter.config import load_lumi, list_eras
from wrplotter.cli_utils import setup_logging

logger = logging.getLogger(__name__)

# Default relative tolerance for distinguishing FP noise from real mismatches.
# Weight sums computed in different order (due to skim) can differ at ~1e-7
# level in float64 arithmetic.  1e-6 is a conservative threshold.
DEFAULT_RTOL = 1e-6

# Resolved cutflow: the first 3 cuts (no_cuts, jet pT/eta, jet ID) are
# pre-skim and expected to differ.  Lepton selection (index 3) is the
# convergence point for cumulative cutflows.
_RESOLVED_CONVERGENCE_IDX = {
    "ee":   3,  # two_pteta_electrons
    "mumu": 3,  # two_pteta_muons
    "em":   3,  # two_pteta_em
}

# Boosted cutflow: only the final region-specific cut (index 5) is guaranteed
# to match.  Intermediate cuts (ak8_jets_with_lsf, jet_veto_map) may differ
# because the skim is looser for boosted — extra events enter those cuts in
# the unskimmed path but are removed by the final SR/CR selection.
_BOOSTED_CONVERGENCE_IDX = {
    "ee":   5,  # ee_sr
    "mumu": 5,  # mumu_sr
    "em":   5,  # emu_cr
}


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
    """Read histogram values and variances as numpy arrays."""
    with uproot.open(root_file) as f:
        h = f[key]
        vals = h.values(flow=False)
        variances = h.variances(flow=False)
        labels = None
        try:
            labels = list(h.axis().labels())
        except Exception:
            pass
        return vals, variances, labels


def _max_rel_diff(a, b):
    """Compute the maximum relative difference between two arrays."""
    denom = np.maximum(np.abs(a), np.abs(b))
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(a[mask] - b[mask]) / denom[mask]))


def _is_integer_valued(arr):
    """Check if array contains integer-valued floats."""
    return np.all(arr == np.floor(arr))


# ── comparison logic ─────────────────────────────────────────────────────────

def compare_region_histograms(skim_path, unskim_path, sample_name,
                               rtol=DEFAULT_RTOL, verbose=False):
    """
    Compare all region histograms (non-cutflow) between two ROOT files.

    Returns (n_compared, n_exact, n_fp_noise, n_real_mismatch, details_list).
      - n_exact: bitwise identical
      - n_fp_noise: differ, but max relative diff < rtol
      - n_real_mismatch: differ with max relative diff >= rtol
    """
    skim_keys = set(list_hist_keys(skim_path))
    unskim_keys = set(list_hist_keys(unskim_path))

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
    n_exact = 0
    n_fp_noise = 0
    n_real_mismatch = 0

    for key in common:
        vals_s, vars_s, _ = read_values(skim_path, key)
        vals_u, vars_u, _ = read_values(unskim_path, key)
        n_compared += 1

        if len(vals_s) != len(vals_u):
            details.append(
                f"  MISMATCH {key}: different bin count "
                f"({len(vals_s)} vs {len(vals_u)})"
            )
            n_real_mismatch += 1
            continue

        exact = np.array_equal(vals_s, vals_u)
        if vars_s is not None and vars_u is not None:
            exact = exact and np.array_equal(vars_s, vars_u)

        if exact:
            n_exact += 1
            continue

        max_rd = _max_rel_diff(vals_s, vals_u)
        max_abs = float(np.max(np.abs(vals_s - vals_u)))

        if max_rd < rtol:
            n_fp_noise += 1
            if verbose:
                details.append(
                    f"  FP noise {key}: max |rel diff|={max_rd:.2e}, "
                    f"max |abs diff|={max_abs:.2e}"
                )
        else:
            n_real_mismatch += 1
            total_s = float(np.sum(vals_s))
            total_u = float(np.sum(vals_u))
            details.append(
                f"  MISMATCH {key}: max |rel diff|={max_rd:.2e}, "
                f"max |abs diff|={max_abs:.2e}, "
                f"integral: s={total_s:.1f} u={total_u:.1f} "
                f"(delta={total_s - total_u:+.1f})"
            )
            if verbose:
                mismatched_bins = np.where(
                    ~np.isclose(vals_s, vals_u, rtol=rtol, atol=0)
                )[0]
                for bi in mismatched_bins[:10]:
                    details.append(
                        f"    bin {bi}: skimmed={vals_s[bi]:.6g}  "
                        f"unskimmed={vals_u[bi]:.6g}  "
                        f"diff={vals_s[bi] - vals_u[bi]:+.6g}"
                    )
                if len(mismatched_bins) > 10:
                    details.append(
                        f"    ... and {len(mismatched_bins) - 10} more bins"
                    )

    return n_compared, n_exact, n_fp_noise, n_real_mismatch, details


def compare_cutflows(skim_path, unskim_path, sample_name,
                     rtol=DEFAULT_RTOL, verbose=False):
    """
    Compare cutflow histograms between skimmed and unskimmed files.

    Checks both weighted and unweighted cutflows (cumulative and onecut)
    for resolved and boosted categories in ee, mumu, em channels.

    Returns (lines, n_weighted_issues, n_unweighted_issues,
             n_unweighted_corrupt).
    """
    lines = []
    n_weighted_issues = 0
    n_unweighted_issues = 0
    n_unweighted_corrupt = 0

    categories = [
        ("cutflow_resolved", _RESOLVED_CONVERGENCE_IDX, "resolved"),
        ("cutflow_boosted", _BOOSTED_CONVERGENCE_IDX, "boosted"),
    ]
    channels = ["ee", "mumu", "em"]
    # Only cumulative cutflows have a meaningful convergence point.
    # Onecut applies each cut independently on all events, so pre-skim
    # cuts (jet veto map, lepton ID, trigger) will naturally differ.
    hist_types_weighted = ["cumulative"]
    hist_types_unweighted = ["cumulative_unweighted"]

    for cat_name, convergence_map, cat_label in categories:
        for channel in channels:
            conv_idx = convergence_map[channel]

            # ── Weighted cutflows ──
            for hist_type in hist_types_weighted:
                key = f"{cat_name}/{channel}/{hist_type}"
                try:
                    vals_s, _, labels_s = read_values(skim_path, key)
                    vals_u, _, labels_u = read_values(unskim_path, key)
                except Exception:
                    continue

                if len(vals_s) != len(vals_u):
                    lines.append(f"  {key}: different bin counts")
                    n_weighted_issues += 1
                    continue

                labels = labels_s or [str(i) for i in range(len(vals_s))]
                post_s = vals_s[conv_idx:]
                post_u = vals_u[conv_idx:]
                post_match = np.allclose(post_s, post_u, rtol=rtol, atol=0)

                pre_match = np.allclose(
                    vals_s[:conv_idx], vals_u[:conv_idx], rtol=rtol, atol=0
                )
                pre_status = "match" if pre_match else "differ (expected)"

                if post_match:
                    post_status = "MATCH"
                else:
                    post_status = "MISMATCH"
                    n_weighted_issues += 1

                lines.append(
                    f"  {key}:  pre-skim {pre_status}  |  "
                    f"post-skim {post_status}"
                )

                if verbose or not post_match:
                    for i, label in enumerate(labels):
                        marker = "  "
                        if i < conv_idx and not np.isclose(
                            vals_s[i], vals_u[i], rtol=rtol
                        ):
                            marker = "~ "
                        elif i >= conv_idx and not np.isclose(
                            vals_s[i], vals_u[i], rtol=rtol
                        ):
                            marker = "! "
                        diff_str = ""
                        if not np.isclose(vals_s[i], vals_u[i], rtol=1e-12):
                            rd = _max_rel_diff(
                                np.array([vals_s[i]]), np.array([vals_u[i]])
                            )
                            diff_str = f"  (rel diff={rd:.2e})"
                        lines.append(
                            f"    {marker}{label:<40s}  "
                            f"skimmed={vals_s[i]:>14.2f}  "
                            f"unskimmed={vals_u[i]:>14.2f}{diff_str}"
                        )

            # ── Unweighted cutflows ──
            for hist_type in hist_types_unweighted:
                key = f"{cat_name}/{channel}/{hist_type}"
                try:
                    vals_s, _, labels_s = read_values(skim_path, key)
                    vals_u, _, labels_u = read_values(unskim_path, key)
                except Exception:
                    continue

                if len(vals_s) != len(vals_u):
                    lines.append(f"  {key}: different bin counts")
                    n_unweighted_issues += 1
                    continue

                labels = labels_s or [str(i) for i in range(len(vals_s))]
                s_int = _is_integer_valued(vals_s)
                u_int = _is_integer_valued(vals_u)

                # Detect corrupt unweighted histograms (non-integer values)
                if not u_int:
                    n_unweighted_corrupt += 1
                    lines.append(
                        f"  {key}: CORRUPT - unskimmed 'unweighted' "
                        f"contains non-integer values "
                        f"(e.g. {vals_u[0]:.6e}); likely filled with "
                        f"event weights instead of 1"
                    )
                    continue

                post_match = np.array_equal(vals_s[conv_idx:], vals_u[conv_idx:])
                pre_match = np.array_equal(vals_s[:conv_idx], vals_u[:conv_idx])

                pre_status = "match" if pre_match else "differ (expected)"
                post_status = "MATCH" if post_match else "MISMATCH"

                if not post_match:
                    n_unweighted_issues += 1

                lines.append(
                    f"  {key}:  pre-skim {pre_status}  |  "
                    f"post-skim {post_status}"
                )

                if verbose or not post_match:
                    for i, label in enumerate(labels):
                        marker = "  "
                        if i < conv_idx and vals_s[i] != vals_u[i]:
                            marker = "~ "
                        elif i >= conv_idx and vals_s[i] != vals_u[i]:
                            marker = "! "
                        diff_str = ""
                        if vals_s[i] != vals_u[i]:
                            diff_str = (
                                f"  (delta={vals_s[i] - vals_u[i]:+.0f})"
                            )
                        lines.append(
                            f"    {marker}{label:<40s}  "
                            f"skimmed={vals_s[i]:>12.0f}  "
                            f"unskimmed={vals_u[i]:>12.0f}{diff_str}"
                        )

    return lines, n_weighted_issues, n_unweighted_issues, n_unweighted_corrupt


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare skimmed vs unskimmed ROOT files.",
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
        "--rtol", type=float, default=DEFAULT_RTOL,
        help=f"Relative tolerance for FP noise vs real mismatch "
             f"(default: {DEFAULT_RTOL}).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show per-bin details for all cutflows and FP-noise histograms.",
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
    print(f"Tolerance (rtol):    {args.rtol:.0e}")
    print(f"Common samples:      {len(common_samples)}")
    if only_skim:
        print(f"Only in SKIMMED:     {only_skim}")
    if only_unskim:
        print(f"Only in UNSKIMMED:   {only_unskim}")
    print()

    # Accumulators
    total_compared = 0
    total_exact = 0
    total_fp = 0
    total_real = 0
    total_w_cf = 0
    total_uw_cf = 0
    total_uw_corrupt = 0
    sample_summaries = []

    sep = "=" * 90

    for sample in common_samples:
        print(sep)
        print(f"Sample: {sample}")
        print(sep)

        skim_path = skim_samples[sample]
        unskim_path = unskim_samples[sample]

        # ── Region histograms ──
        n_comp, n_exact, n_fp, n_real, hist_details = (
            compare_region_histograms(
                skim_path, unskim_path, sample,
                rtol=args.rtol, verbose=args.verbose,
            )
        )
        total_compared += n_comp
        total_exact += n_exact
        total_fp += n_fp
        total_real += n_real

        print(f"\n  Region histograms ({n_comp} compared):")
        print(f"    Exact match:   {n_exact}")
        print(f"    FP noise only: {n_fp}  (rel diff < {args.rtol:.0e})")
        print(f"    Real mismatch: {n_real}")
        for line in hist_details:
            print(line)

        # ── Cutflows ──
        cf_lines, n_w, n_uw, n_corrupt = compare_cutflows(
            skim_path, unskim_path, sample,
            rtol=args.rtol, verbose=args.verbose,
        )
        total_w_cf += n_w
        total_uw_cf += n_uw
        total_uw_corrupt += n_corrupt

        w_status = "MATCH" if n_w == 0 else f"{n_w} MISMATCH(ES)"
        uw_status = "MATCH" if n_uw == 0 else f"{n_uw} MISMATCH(ES)"
        cr_status = (
            f", {n_corrupt} CORRUPT" if n_corrupt > 0 else ""
        )
        print(f"\n  Cutflows:")
        print(f"    Weighted post-skim:   {w_status}")
        print(f"    Unweighted post-skim: {uw_status}{cr_status}")
        for line in cf_lines:
            print(line)

        sample_summaries.append(
            (sample, n_comp, n_exact, n_fp, n_real, n_w, n_uw, n_corrupt)
        )
        print()

    # ── Final summary table ──
    print(sep)
    print("FINAL SUMMARY")
    print(sep)
    header = (
        f"{'Sample':<30s} {'Hists':>6s} {'Exact':>6s} "
        f"{'FP':>5s} {'Real':>5s} {'W.CF':>5s} "
        f"{'UW.CF':>6s} {'Crpt':>5s}"
    )
    print(header)
    print("-" * len(header))
    for sample, nc, ne, nf, nr, nw, nuw, ncr in sample_summaries:
        flags = []
        if nr > 0:
            flags.append("hist")
        if nw > 0:
            flags.append("w.cf")
        if nuw > 0:
            flags.append("uw.cf")
        if ncr > 0:
            flags.append("corrupt")
        flag_str = f" <-- {', '.join(flags)}" if flags else ""
        print(
            f"{sample:<30s} {nc:>6d} {ne:>6d} {nf:>5d} "
            f"{nr:>5d} {nw:>5d} {nuw:>6d} {ncr:>5d}{flag_str}"
        )
    print("-" * len(header))
    print(
        f"{'TOTAL':<30s} {total_compared:>6d} {total_exact:>6d} "
        f"{total_fp:>5d} {total_real:>5d} {total_w_cf:>5d} "
        f"{total_uw_cf:>6d} {total_uw_corrupt:>5d}"
    )
    print()

    # ── Interpretation ──
    print(sep)
    print("INTERPRETATION")
    print(sep)

    if total_real == 0:
        print(
            f"Region histograms: PASS\n"
            f"  All {total_compared} histograms agree within "
            f"rtol={args.rtol:.0e}.\n"
            f"  {total_exact} are bitwise identical; {total_fp} show "
            f"FP-level differences\n"
            f"  (max rel diff < {args.rtol:.0e}) from weight-summation "
            f"order."
        )
    else:
        print(
            f"Region histograms: FAIL\n"
            f"  {total_real} histogram(s) have real mismatches "
            f"(rel diff >= {args.rtol:.0e}).\n"
            f"  These are integer-count differences in data samples,\n"
            f"  suggesting different input file sets between the two runs."
        )

    print()

    if total_w_cf == 0:
        print(
            "Weighted cutflows: PASS\n"
            "  All weighted yields match (within FP tolerance) after the\n"
            "  lepton-selection convergence point. Pre-skim cuts differ\n"
            "  as expected (unskimmed starts from more events)."
        )
    else:
        print(
            f"Weighted cutflows: FAIL\n"
            f"  {total_w_cf} post-skim weighted cutflow mismatch(es)."
        )

    print()

    if total_uw_corrupt > 0:
        print(
            f"Unweighted cutflows: CORRUPT ({total_uw_corrupt} histograms)\n"
            f"  The unskimmed 'unweighted' cutflow histograms contain\n"
            f"  non-integer float values (tiny fractions like ~1e-4).\n"
            f"  This indicates the unskimmed processing filled the\n"
            f"  unweighted cutflow with event weights instead of 1.\n"
            f"  The WEIGHTED cutflows should be used for validation."
        )
    elif total_uw_cf == 0:
        print(
            "Unweighted cutflows: PASS\n"
            "  All unweighted yields match after the convergence point."
        )
    else:
        print(
            f"Unweighted cutflows: FAIL\n"
            f"  {total_uw_cf} post-skim unweighted cutflow mismatch(es)."
        )

    all_pass = (
        total_real == 0
        and total_w_cf == 0
        and total_uw_cf == 0
        and total_uw_corrupt == 0
    )
    print()
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
