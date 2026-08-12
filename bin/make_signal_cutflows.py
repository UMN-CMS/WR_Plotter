#!/usr/bin/env python3
"""
Generate signal-only cutflow tables from ROOT files.

Two comparison modes:
  --mode fixed-wr   (default) One table per WR mass, columns = N mass points
  --mode fixed-n    One table per N mass, columns = WR mass points
  --mode wr-compare One table comparing WR masses (picks one N per WR)

Examples:
    # All WR masses, one table each (default)
    python bin/make_signal_cutflows.py --era RunIII2024Summer24 --dir 20260317_lo_dy --channel ee

    # Only WR=4000
    python bin/make_signal_cutflows.py --era RunIII2024Summer24 --dir 20260317_lo_dy --channel mumu --wr-mass 4000

    # Compare across WR masses
    python bin/make_signal_cutflows.py --era RunIII2024Summer24 --dir 20260317_lo_dy --channel ee --mode wr-compare

    # Terminal output
    python bin/make_signal_cutflows.py --era RunIII2024Summer24 --dir 20260317_lo_dy --channel ee --format terminal

    # Boosted region
    python bin/make_signal_cutflows.py --era RunIII2024Summer24 --dir 20260317_lo_dy --channel ee --boosted
"""
import argparse
import logging
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wrplotter.config import load_lumi
from wrplotter.paths import input_dirs_for_era
from wrplotter.cli_utils import setup_logging, add_era_args

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Cut sequences (must match the analyzer output)
# -------------------------------------------------------------------------
_CUT_SEQUENCES = {
    "mumu": [
        ("no_cuts",                  "No cuts"),
        ("min_two_ak4_jets_pteta",   r"$\geq$2 AK4 jets ($p_T > 40$, $|\eta| < 2.4$)"),
        ("min_two_ak4_jets_id",      r"$\geq$2 AK4 jets (tight w/ lepton veto)"),
        ("two_pteta_muons",          r"2 muons ($p_T > 53$, $|\eta| < 2.4$)"),
        ("two_id_muons",             r"2 muons (high-$p_T$ ID)"),
        ("mu_trigger",               "Muon trigger"),
        ("dr_all_pairs_gt0p4",       r"$\Delta R > 0.4$ (all pairs)"),
        ("mlljj_gt800",              r"$m(\ell\ell jj) > 800$ GeV"),
        ("mll_gt200",                r"$m(\ell\ell) > 200$ GeV"),
        ("mll_gt400",                r"$m(\ell\ell) > 400$ GeV"),
    ],
    "ee": [
        ("no_cuts",                  "No cuts"),
        ("min_two_ak4_jets_pteta",   r"$\geq$2 AK4 jets ($p_T > 40$, $|\eta| < 2.4$)"),
        ("min_two_ak4_jets_id",      r"$\geq$2 AK4 jets (tight w/ lepton veto)"),
        ("two_pteta_electrons",      r"2 electrons ($p_T > 53$, $|\eta| < 2.4$)"),
        ("two_id_electrons",         "2 electrons (HEEP ID)"),
        ("e_trigger",                "Electron trigger"),
        ("dr_all_pairs_gt0p4",       r"$\Delta R > 0.4$ (all pairs)"),
        ("mlljj_gt800",              r"$m(\ell\ell jj) > 800$ GeV"),
        ("mll_gt200",                r"$m(\ell\ell) > 200$ GeV"),
        ("mll_gt400",                r"$m(\ell\ell) > 400$ GeV"),
    ],
}

_CUT_SEQUENCES_BOOSTED = {
    "mumu": [
        ("no_cuts",                          "No cuts"),
        ("boosted_tag",                      "Boosted tag"),
        ("lead_is_muon",                     r"Leading lepton is $\mu$"),
        ("lead_tight_lepton_pt60_boosted",   r"Leading tight lepton $p_T > 60$ GeV"),
        ("mu_trigger",                       r"$\mu$ trigger fired"),
        ("no_dy_pair",                       r"No $60 < m(\ell\ell) < 150$ GeV"),
        ("ak8_jets_with_lsf",               r"Merged AK8 jet with $\Delta\phi > 2.0$"),
        ("no_extra_tight_sr",                "No extra tight lepton"),
        ("sf_lepton_in_ak8",                r"Loose SF lepton in merged jet"),
        ("no_of_lepton_in_ak8",             r"No loose OF lepton in merged jet"),
        ("mll_gt200_boosted",               r"$m(\ell\ell) > 200$ GeV"),
        ("mlj_gt800_boosted",               r"$m(\ell J) > 800$ GeV"),
    ],
    "ee": [
        ("no_cuts",                          "No cuts"),
        ("boosted_tag",                      "Boosted tag"),
        ("lead_is_electron",                 r"Leading lepton is $e$"),
        ("lead_tight_lepton_pt60_boosted",   r"Leading tight lepton $p_T > 60$ GeV"),
        ("e_trigger",                        r"$e$ trigger fired"),
        ("no_dy_pair",                       r"No $60 < m(\ell\ell) < 150$ GeV"),
        ("ak8_jets_with_lsf",               r"Merged AK8 jet with $\Delta\phi > 2.0$"),
        ("no_extra_tight_sr",                "No extra tight lepton"),
        ("sf_lepton_in_ak8",                r"Loose SF lepton in merged jet"),
        ("no_of_lepton_in_ak8",             r"No loose OF lepton in merged jet"),
        ("mll_gt200_boosted",               r"$m(\ell\ell) > 200$ GeV"),
        ("mlj_gt800_boosted",               r"$m(\ell J) > 800$ GeV"),
    ],
}

_CHANNEL_PRETTY = {"ee": "dielectron", "mumu": "dimuon", "em": r"electron--muon",
                    "ee_mumu": "dielectron and dimuon"}

# Generic labels used when ee and mumu are shown side-by-side
_CUT_LABELS_GENERIC = {
    False: [  # resolved
        "No cuts",
        r"$\geq$2 AK4 jets ($p_T > 40$, $|\eta| < 2.4$)",
        r"$\geq$2 AK4 jets (tight w/ lepton veto)",
        r"2 leptons ($p_T > 53$, $|\eta| < 2.4$)",
        "2 leptons (ID)",
        "Trigger",
        r"$\Delta R > 0.4$ (all pairs)",
        r"$m(\ell\ell jj) > 800$ GeV",
        r"$m(\ell\ell) > 200$ GeV",
        r"$m(\ell\ell) > 400$ GeV",
    ],
    True: [  # boosted
        "No cuts",
        "Boosted tag",
        "Leading lepton flavor",
        r"Leading tight lepton $p_T > 60$ GeV",
        "Trigger",
        r"No $60 < m(\ell\ell) < 150$ GeV",
        r"Merged AK8 jet with $\Delta\phi > 2.0$",
        "No extra tight lepton",
        r"Loose SF lepton in merged jet",
        r"No loose OF lepton in merged jet",
        r"$m(\ell\ell) > 200$ GeV",
        r"$m(\ell J) > 800$ GeV",
    ],
}
_CHANNELS = ["ee", "mumu"]

_PREVIEW_TEMPLATE = r"""\documentclass[12pt]{{article}}
\usepackage[landscape,margin=0.3in,paperwidth=24in,paperheight=16in]{{geometry}}
\usepackage{{amsmath}}
\usepackage{{float}}

%% CMS style stubs
\usepackage{{caption}}
\captionsetup{{font=LARGE}}
\newcommand{{\topcaption}}[1]{{\caption{{#1}}}}
\newcommand{{\cmsTable}}[1]{{\Large #1}}
\setlength{{\floatsep}}{{4pt}}
\setlength{{\textfloatsep}}{{4pt}}
\setlength{{\intextsep}}{{4pt}}

\renewcommand{{\topfraction}}{{1.0}}
\renewcommand{{\bottomfraction}}{{1.0}}
\renewcommand{{\textfraction}}{{0.0}}
\renewcommand{{\floatpagefraction}}{{0.99}}
\setcounter{{totalnumber}}{{10}}
\setcounter{{topnumber}}{{10}}
\setcounter{{bottomnumber}}{{10}}

\begin{{document}}

{body}

\end{{document}}
"""


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def _parse_mass_point(name):
    """Parse 'WR2000_N1100' -> (2000, 1100)."""
    m = re.match(r"WR(\d+)_N(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _read_cutflow(root_path, channel, hist_type="cumulative", expected_keys=None,
                  boosted=False):
    """Read a cutflow histogram from a ROOT file. Returns np.ndarray or None."""
    prefixes = ["cutflow_boosted"] if boosted else ["cutflow_resolved", "cutflow"]
    candidates = [f"{p}/{channel}/{hist_type}" for p in prefixes]
    try:
        with uproot.open(root_path) as f:
            key = next((k for k in candidates if k in f), None)
            if key is None:
                logger.debug("No cutflow key found in %s", root_path)
                return None
            h = f[key]
            vals = h.values(flow=False)
            if expected_keys is not None and len(vals) != len(expected_keys):
                logger.warning("Skipping %s: expected %d bins, got %d",
                               root_path.name, len(expected_keys), len(vals))
                return None
            return vals
    except FileNotFoundError:
        return None


def _discover_signals(input_dirs):
    """Find all signal ROOT files. Returns dict: mass_point_str -> yields accumulator path list."""
    signals = {}
    for d in input_dirs:
        for p in sorted(d.glob("WRAnalyzer_signal_*.root")):
            name = p.stem.replace("WRAnalyzer_signal_", "")
            signals.setdefault(name, []).append(p)
    return signals


def _load_signal_yields(signals, channel, boosted, cut_keys, hist_type="cumulative"):
    """Load cutflow yields for all signal samples.

    Returns dict: mass_point_str -> np.ndarray of yields.
    """
    yields = {}
    n_cuts = len(cut_keys)
    for mass_str, paths in signals.items():
        vals = None
        for p in paths:
            sub = _read_cutflow(p, channel, hist_type=hist_type,
                                expected_keys=cut_keys, boosted=boosted)
            if sub is None:
                continue
            vals = sub if vals is None else vals + sub
        if vals is not None:
            yields[mass_str] = vals
        else:
            logger.warning("No cutflow data for %s", mass_str)
    return yields


# -------------------------------------------------------------------------
# Formatting
# -------------------------------------------------------------------------
def _fmt(x):
    """Format a yield value for terminal display."""
    if x == 0:
        return "0"
    if abs(x) < 0.01:
        return f"{x:.2e}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def _fmt_eff(x):
    """Format efficiency as a fraction (0.000–1.000)."""
    if x == 0:
        return "0"
    if abs(x) < 0.001:
        return f"{x:.2e}"
    return f"{x:.3f}"


def _running_eff(yields, i):
    return yields[i] / yields[0] if yields[0] > 0 else 0.0


def _step_eff(yields, i):
    if i == 0:
        return 1.0
    return yields[i] / yields[i - 1] if yields[i - 1] > 0 else 0.0


def _fmt_latex_yield(x):
    """Format yield for LaTeX. Plain numbers unless < 0.01."""
    import math
    if x == 0:
        return "0"
    if abs(x) < 0.01:
        exp = int(math.floor(math.log10(abs(x))))
        mant = x / (10 ** exp)
        if round(mant, 2) >= 10:
            exp += 1
            mant /= 10
        return f"${mant:.2f} \\times 10^{{{exp}}}$"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


# -------------------------------------------------------------------------
# Table builders
# -------------------------------------------------------------------------
def build_terminal_table(cut_labels, columns, title, efficiency=None,
                         col_group_labels=None, col_group_sizes=None):
    """Build a terminal-format table.

    columns: list of (header_label, yields_array)
    col_group_labels: optional list of group names (e.g. ["ee", "mumu"])
    col_group_sizes: optional list of ints matching col_group_labels
    """
    n_cuts = len(cut_labels)
    headers = ["Cut"] + [label for label, _ in columns]

    rows = []
    for i in range(n_cuts):
        row = [cut_labels[i]]
        for _, y in columns:
            if efficiency == "running":
                row.append(_fmt_eff(_running_eff(y, i)))
            elif efficiency == "relative":
                row.append(_fmt_eff(_step_eff(y, i)))
            else:
                row.append(_fmt(y[i]))
        rows.append(row)

    # If showing yields, append efficiency sections
    if efficiency is None:
        rows.append(["--- Running efficiency (%) ---"] + [""] * len(columns))
        for i in range(n_cuts):
            row = [cut_labels[i]]
            for _, y in columns:
                row.append(_fmt_eff(_running_eff(y, i)))
            rows.append(row)

        rows.append(["--- Relative efficiency (%) ---"] + [""] * len(columns))
        for i in range(n_cuts):
            row = [cut_labels[i]]
            for _, y in columns:
                row.append(_fmt_eff(_step_eff(y, i)))
            rows.append(row)

    all_rows = [headers] + rows
    widths = [max(len(str(r[j])) for r in all_rows) for j in range(len(headers))]

    lines = [title, "=" * sum(widths + [3 * len(widths)])]

    def fmt_row(r):
        return " | ".join(str(r[j]).ljust(widths[j]) for j in range(len(headers)))

    # Group header row (e.g. "              | ee               | mumu             ")
    if col_group_labels and col_group_sizes:
        group_row = " " * widths[0] + " | "
        col_idx = 1  # skip "Cut" column
        parts = []
        for glabel, gsize in zip(col_group_labels, col_group_sizes):
            span = sum(widths[col_idx:col_idx + gsize]) + 3 * (gsize - 1)
            parts.append(glabel.center(span))
            col_idx += gsize
        group_row += " | ".join(parts)
        lines.append(group_row)

    lines.append(fmt_row(headers))
    lines.append("-+-".join("-" * w for w in widths))

    sep_indices = {i for i, r in enumerate(rows) if r[0].startswith("---")}
    for idx, row in enumerate(rows):
        if idx in sep_indices:
            lines += ["", row[0], "-+-".join("-" * w for w in widths)]
            continue
        lines.append(fmt_row(row))

    return "\n".join(lines)


def build_latex_table(cut_labels, columns, channel, lumi, year, era,
                      region="resolved", title_extra="", efficiency=None,
                      onecut=False, col_group_sizes=None, col_group_labels=None,
                      wr_mass=None):
    """Build a CMS-style LaTeX signal cutflow table.

    col_group_sizes: optional list of ints for column grouping with vertical
                     separators (e.g. [3, 3] for two groups of 3 columns).
    col_group_labels: optional list of group names for a spanning header row.
    """
    n_cuts = len(cut_labels)
    chan_pretty = _CHANNEL_PRETTY.get(channel, channel)
    col_headers = [label for label, _ in columns]
    if col_group_sizes:
        groups = []
        idx = 0
        for size in col_group_sizes:
            groups.append(" ".join(["c"] * size))
            idx += size
        col_spec = "l | " + " | ".join(groups)
    else:
        col_spec = "l | " + " ".join(["c"] * len(col_headers))

    # Caption description — matches make_cutflow_tables.py style
    if efficiency == "running" and onecut:
        cut_description = (
            "Each cell shows the single-cut efficiency: the fraction of events "
            "passing only that cut, applied independently of all other cuts. "
        )
    elif efficiency == "running":
        cut_description = (
            "Each cell shows the running efficiency "
            "relative to the first selection row. "
        )
    elif efficiency == "relative":
        cut_description = (
            "Each cell shows the relative efficiency: the fraction of events "
            "passing the previous cut that also pass the current cut. "
        )
    else:
        cut_description = (
            "Each row shows the yield after applying that cut and all preceding cuts. "
        )

    # Determine hist label for LaTeX label
    if efficiency:
        hist_label = f"{efficiency}_eff"
    elif onecut:
        hist_label = "onecut"
    else:
        hist_label = "cumulative"

    lines = []
    lines.append(r"\begin{table}[htp]")
    if wr_mass is not None:
        wr_str = f"$m_{{W_R}} = {wr_mass}$ GeV"
        lines.append(
            f"  \\topcaption{{{era}: expected signal event yields for "
            f"{wr_str} in the {region}, "
            f"{chan_pretty} channel. "
            f"{cut_description}"
            f"All yields are normalized to ${lumi:.1f}\\,\\mathrm{{fb}}^{{-1}}$ ({year}).}}"
        )
    else:
        lines.append(
            f"  \\topcaption{{{era}: expected signal event yields for the {region}, "
            f"{chan_pretty} channel. "
            f"{cut_description}"
            f"{title_extra}"
            f"All yields are normalized to ${lumi:.1f}\\,\\mathrm{{fb}}^{{-1}}$ ({year}).}}"
        )
    safe_extra = re.sub(r"[^a-zA-Z0-9_]", "", title_extra.split(",")[0] if title_extra else "")
    lines.append(
        f"  \\label{{tab:signal_cutflow_{hist_label}_{region}_{channel}_{safe_extra}_{year}}}"
    )
    lines.append(r"  \centering")
    lines.append(r"  \cmsTable{")
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"      \hline")
    if col_group_sizes and col_group_labels:
        # Spanning header row: Selection | ee (multicolumn) | mumu (multicolumn)
        # First group gets "c" alignment, subsequent groups get "|c" to preserve vertical line
        mcols = [""]
        for i, (glabel, gsize) in enumerate(zip(col_group_labels, col_group_sizes)):
            align = "c" if i == 0 else "|c"
            mcols.append(f"\\multicolumn{{{gsize}}}{{{align}}}{{{glabel}}}")
        lines.append("      " + " & ".join(mcols) + r" \\")
    lines.append("      " + " & ".join(["Selection"] + col_headers) + r" \\")
    lines.append(r"      \hline")

    for i in range(n_cuts):
        cols = [cut_labels[i]]
        for _, y in columns:
            if efficiency == "running":
                cols.append(f"{_running_eff(y, i):.3f}")
            elif efficiency == "relative":
                cols.append(f"{_step_eff(y, i):.3f}")
            else:
                cols.append(_fmt_latex_yield(y[i]))
        lines.append("      " + " & ".join(cols) + r" \\")

    lines.append(r"      \hline")
    lines.append(r"    \end{tabular}")
    lines.append(r"  }")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Signal-only cutflow tables: compare N masses within a WR mass, "
                    "or compare across WR masses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_era_args(parser)
    parser.add_argument("--channel", choices=["ee", "mumu"], default=None,
                        help="Lepton channel (omit to generate all channels + preview PDF)")
    parser.add_argument("--mode", choices=["fixed-wr", "wr-compare", "channel-compare"],
                        default="fixed-wr",
                        help="fixed-wr: one table per WR mass with N-mass columns (default). "
                             "wr-compare: one table comparing WR masses (highest N each). "
                             "channel-compare: ee and mumu side-by-side per WR mass.")
    parser.add_argument("--wr-mass", type=int, nargs="*", default=None,
                        help="WR masses to include (default: all found)")
    parser.add_argument("--boosted", action="store_true")
    parser.add_argument("--onecut", action="store_true",
                        help="Each cut applied independently")
    parser.add_argument("--efficiency", choices=["running", "relative"], default=None,
                        help="Show efficiency instead of yields")
    parser.add_argument("--format", choices=["terminal", "latex"], default="terminal")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Override output directory")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    era_info = load_lumi(args.era)
    run, year = era_info["run"], str(era_info["year"])
    lumi = float(era_info["lumi"])
    region = "boosted" if args.boosted else "resolved"

    repo_dir = Path(__file__).resolve().parent.parent
    input_dirs, _ = input_dirs_for_era(args.era, repo_dir, args.dir)
    valid_dirs = [d for d in input_dirs if d.is_dir()]
    if not valid_dirs:
        logger.error("No input directories found for era '%s'", args.era)
        sys.exit(1)

    # Discover and load signals
    signals = _discover_signals(valid_dirs)
    if not signals:
        logger.error("No signal ROOT files found")
        sys.exit(1)

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = repo_dir / "tables" / run / year / args.era / args.dir / "signal_cutflows"
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = _CHANNELS if args.channel is None else [args.channel]
    modes = ["fixed-wr", "wr-compare"] if args.channel is None else [args.mode]
    batch_mode = args.channel is None and args.mode != "channel-compare"
    fmt = "latex" if batch_mode else args.format

    all_tables = []

    # channel-compare mode: load ee + mumu together
    if args.mode == "channel-compare":
        all_tables.extend(_generate_channel_compare(
            signals, args.boosted, args.onecut, args.wr_mass,
            lumi, year, args.era, region, fmt, args.efficiency, out_dir,
        ))
    else:
        for channel in channels:
            cut_seqs = _CUT_SEQUENCES_BOOSTED if args.boosted else _CUT_SEQUENCES
            cut_seq = cut_seqs[channel]
            cut_keys = [k for k, _ in cut_seq]
            cut_labels = [label for _, label in cut_seq]
            hist_type = "onecut" if args.onecut else "cumulative"

            channel_yields = _load_signal_yields(signals, channel, args.boosted,
                                                 cut_keys, hist_type=hist_type)
            if not channel_yields:
                logger.warning("No signal cutflow data for %s", channel)
                continue

            # Group by WR mass
            by_wr = defaultdict(dict)
            for mass_str, y in channel_yields.items():
                parsed = _parse_mass_point(mass_str)
                if parsed:
                    wr, n = parsed
                    by_wr[wr][n] = (mass_str, y)

            if args.wr_mass:
                by_wr = {wr: ns for wr, ns in by_wr.items() if wr in args.wr_mass}
            if not by_wr:
                logger.warning("No signal data for requested WR masses (%s)", channel)
                continue

            for mode in modes:
                tables = _generate_tables(
                    mode, by_wr, cut_labels, channel, lumi, year, args.era,
                    region, fmt, args.efficiency, args.onecut, out_dir,
                )
                all_tables.extend(tables)

    if all_tables:
        print(f"\nWrote {len(all_tables)} table(s) to {out_dir}")

    # Compile preview PDF when we have latex tables
    if all_tables and fmt == "latex":
        _compile_preview(all_tables, out_dir, region)


def _generate_tables(mode, by_wr, cut_labels, channel, lumi, year, era,
                     region, fmt, efficiency, onecut, out_dir):
    """Generate tables for one channel/mode combination. Returns list of written Paths."""
    tables = []
    eff_tag = f"_{efficiency}_eff" if efficiency else ""
    onecut_tag = "_onecut" if onecut else ""

    if mode == "fixed-wr":
        for wr in sorted(by_wr.keys()):
            ns = by_wr[wr]
            columns = []
            for n in sorted(ns.keys()):
                mass_str, y = ns[n]
                columns.append((f"$N_{{{n}}}$" if fmt == "latex" else f"N={n}", y))

            title = f"WR={wr} GeV signal cutflow — {channel} {region}"
            title_extra = f"$m_{{W_R}} = {wr}$ GeV. "

            if fmt == "terminal":
                print(build_terminal_table(cut_labels, columns, title,
                                           efficiency=efficiency))
                print()
            else:
                table = build_latex_table(cut_labels, columns, channel,
                                          lumi, year, era, region,
                                          title_extra=title_extra,
                                          efficiency=efficiency, onecut=onecut,
                                          wr_mass=wr)
                fname = f"signal_WR{wr}_{region}_{channel}{eff_tag}{onecut_tag}.tex"
                path = out_dir / fname
                path.write_text(table + "\n")
                tables.append(path)
                logger.info("Wrote %s", path)

    elif mode == "wr-compare":
        columns = []
        for wr in sorted(by_wr.keys()):
            ns = by_wr[wr]
            max_n = max(ns.keys())
            mass_str, y = ns[max_n]
            label = (f"$W_R({wr}),\\,N({max_n})$" if fmt == "latex"
                     else f"WR{wr}_N{max_n}")
            columns.append((label, y))

        title = f"WR mass comparison — {channel} {region} (highest N each)"

        if fmt == "terminal":
            print(build_terminal_table(cut_labels, columns, title,
                                       efficiency=efficiency))
        else:
            table = build_latex_table(cut_labels, columns, channel,
                                      lumi, year, era, region,
                                      title_extra="Comparing $W_R$ masses (highest $N$ each). ",
                                      efficiency=efficiency, onecut=onecut)
            fname = f"signal_wr_compare_{region}_{channel}{eff_tag}{onecut_tag}.tex"
            path = out_dir / fname
            path.write_text(table + "\n")
            tables.append(path)
            logger.info("Wrote %s", path)

    return tables


def _generate_channel_compare(signals, boosted, onecut, wr_mass_filter,
                              lumi, year, era, region, fmt, efficiency, out_dir):
    """Generate side-by-side ee vs mumu tables for each WR mass."""
    tables = []
    cut_seqs = _CUT_SEQUENCES_BOOSTED if boosted else _CUT_SEQUENCES
    hist_type = "onecut" if onecut else "cumulative"
    cut_labels = _CUT_LABELS_GENERIC[boosted]
    eff_tag = f"_{efficiency}_eff" if efficiency else ""
    onecut_tag = "_onecut" if onecut else ""

    # Load yields for both channels and both hist types
    def _load_by_wr(hist_type):
        by_wr = {}
        for channel in ["ee", "mumu"]:
            cut_seq = cut_seqs[channel]
            cut_keys = [k for k, _ in cut_seq]
            ch_yields = _load_signal_yields(signals, channel, boosted,
                                            cut_keys, hist_type=hist_type)
            for mass_str, y in ch_yields.items():
                parsed = _parse_mass_point(mass_str)
                if parsed:
                    wr, n = parsed
                    by_wr.setdefault(wr, {}).setdefault(channel, {})[n] = (mass_str, y)
        if wr_mass_filter:
            by_wr = {wr: chs for wr, chs in by_wr.items() if wr in wr_mass_filter}
        return by_wr

    by_wr_cumul = _load_by_wr(hist_type)
    by_wr_onecut = _load_by_wr("onecut") if not onecut else None

    # Build variants list
    # (efficiency_mode, file_suffix, use_onecut_data)
    variants = [(efficiency, eff_tag, False)]
    if efficiency is None:
        variants.append(("running", "_running_eff", False))
        variants.append(("running", "_onecut", True))
        variants.append(("relative", "_relative_eff", False))

    for wr in sorted(by_wr_cumul.keys()):
        for eff_variant, eff_variant_tag, use_onecut in variants:
            src = by_wr_onecut if use_onecut and by_wr_onecut else by_wr_cumul
            chs = src.get(wr, {})
            ee_ns = chs.get("ee", {})
            mumu_ns = chs.get("mumu", {})
            all_ns = sorted(set(ee_ns.keys()) | set(mumu_ns.keys()))
            if not all_ns:
                continue

            columns = []
            for n in all_ns:
                if n in ee_ns:
                    _, y = ee_ns[n]
                    label = f"$N_{{{n}}}$" if fmt == "latex" else f"N={n}"
                    columns.append((label, y))
            n_ee = len([n for n in all_ns if n in ee_ns])
            for n in all_ns:
                if n in mumu_ns:
                    _, y = mumu_ns[n]
                    label = f"$N_{{{n}}}$" if fmt == "latex" else f"N={n}"
                    columns.append((label, y))
            n_mumu = len([n for n in all_ns if n in mumu_ns])

            group_sizes = [n_ee, n_mumu]
            group_labels_term = ["ee", "mumu"]
            group_labels_latex = ["$ee$", r"$\mu\mu$"]

            title = f"WR={wr} GeV — ee vs mumu, {region}"
            title_extra = f"$m_{{W_R}} = {wr}$ GeV, dielectron vs.\\ dimuon. "

            if fmt == "terminal":
                if use_onecut:
                    suffix = " [one-cut]"
                elif eff_variant:
                    suffix = f" [{eff_variant} efficiency]"
                else:
                    suffix = ""
                print(build_terminal_table(cut_labels, columns, title + suffix,
                                           efficiency=eff_variant,
                                           col_group_labels=group_labels_term,
                                           col_group_sizes=group_sizes))
                print()
            else:
                table = build_latex_table(
                    cut_labels, columns, "ee_mumu", lumi, year, era, region,
                    title_extra=title_extra, efficiency=eff_variant,
                    onecut=use_onecut,
                    col_group_sizes=group_sizes,
                    col_group_labels=group_labels_latex,
                    wr_mass=wr,
                )
                fname = f"signal_WR{wr}_{region}_ee_vs_mumu{eff_variant_tag}.tex"
                path = out_dir / fname
                path.write_text(table + "\n")
                tables.append(path)
                logger.info("Wrote %s", path)

    return tables


def _compile_preview(table_paths, out_dir, region):
    """Write a preview.tex that includes all generated tables, then compile to PDF."""
    # Group tables for readable sections
    wr_tables = [p for p in table_paths if not p.name.startswith("signal_wr_compare")]
    compare_tables = [p for p in table_paths if p.name.startswith("signal_wr_compare")]

    body_lines = []
    if wr_tables:
        body_lines.append(r"{\LARGE \bfseries Signal cutflows by $W_R$ mass}" + "\n" + r"\bigskip")
        for p in wr_tables:
            body_lines.append(f"\\input{{{p.name}}}")
            body_lines.append(r"\vspace{3em}")
    if compare_tables:
        body_lines.append(r"\newpage")
        body_lines.append(r"{\LARGE \bfseries $W_R$ mass comparison}" + "\n" + r"\bigskip")
        for p in compare_tables:
            body_lines.append(f"\\input{{{p.name}}}")
            body_lines.append(r"\vspace{3em}")

    preview_tex = _PREVIEW_TEMPLATE.format(body="\n".join(body_lines))
    preview_path = out_dir / "preview.tex"
    preview_path.write_text(preview_tex)
    print(f"Wrote {preview_path}")

    print("Compiling preview PDF ...", end=" ", flush=True)
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "preview.tex"],
        cwd=str(out_dir), capture_output=True, text=True,
    )
    pdf_path = out_dir / "preview.pdf"
    if pdf_path.exists():
        print("ok")
        print(f"PDF: {pdf_path}")
    else:
        print("FAILED")
        for line in result.stdout.splitlines()[-10:]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
