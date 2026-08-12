#!/usr/bin/env python3
"""
Generate cutflow tables from ROOT files produced by the WrCoffea analyzer.

Default behaviour (no --channel) generates all 24 LaTeX tables
(4 variants x 2 regions x 3 channels) and compiles a preview PDF.

With --channel (and other optional flags), generates a single table.

Examples:
    # All tables + preview PDF
    python bin/make_cutflow_tables.py --era RunIII2024Summer24 --dir 20260214_unskimmed

    # Single table
    python bin/make_cutflow_tables.py --era RunIII2024Summer24 --dir 20260214_unskimmed --channel mumu
    python bin/make_cutflow_tables.py --era RunIII2024Summer24 --dir 20260214_unskimmed --channel ee --onecut --boosted
    python bin/make_cutflow_tables.py --era RunIII2024Summer24 --dir 20260214_unskimmed --channel mumu --efficiency running

    # Other formats
    python bin/make_cutflow_tables.py --era RunIII2024Summer24 --dir 20260214_unskimmed --channel mumu --format terminal
    python bin/make_cutflow_tables.py --era RunIII2024Summer24 --dir 20260214_unskimmed --channel ee --format latex-single --sample DYJets

    # Re-render the preview PDF (after editing preview.tex or individual tables)
    cd tables/Run3/2024/RunIII2024Summer24/20260214_unskimmed/cutflows && pdflatex -interaction=nonstopmode preview.tex
"""
import argparse
import logging
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wrplotter.config import load_lumi, list_eras, default_signals
from wrplotter.paths import input_dirs_for_era
from wrplotter.sample_groups import load_sample_groups
from wrplotter.cli_utils import setup_logging, add_era_args

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Cut sequences per channel
# -------------------------------------------------------------------------
# Maps channel -> list of (ROOT key suffix, display label).
# The first entry is always "no_cuts" (pre-selection total).

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
    "em": [
        ("no_cuts",                  "No cuts"),
        ("min_two_ak4_jets_pteta",   r"$\geq$2 AK4 jets ($p_T > 40$, $|\eta| < 2.4$)"),
        ("min_two_ak4_jets_id",      r"$\geq$2 AK4 jets (tight w/ lepton veto)"),
        ("two_pteta_em",             r"1 electron + 1 muon ($p_T > 53$, $|\eta| < 2.4$)"),
        ("two_id_em",                "Electron and muon passing ID"),
        ("emu_trigger",              r"$e\mu$ trigger"),
        ("dr_all_pairs_gt0p4",       r"$\Delta R > 0.4$ (all pairs)"),
        ("mlljj_gt800",              r"$m(\ell\ell jj) > 800$ GeV"),
        ("mll_gt200",                r"$m(\ell\ell) > 200$ GeV"),
        ("mll_gt400",                r"$m(\ell\ell) > 400$ GeV"),
    ],
}

# Boosted cut sequences (expanded SR progression for ee/mumu, flavor CR for em)
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
    "em": [
        ("no_cuts",                          "No cuts"),
        ("boosted_tag",                      "Boosted tag"),
        ("lead_is_electron",                 r"Leading lepton is $e$"),
        ("lead_tight_lepton_pt60_boosted",   r"Leading tight lepton $p_T > 60$ GeV"),
        ("e_trigger",                        r"$e$ trigger fired"),
        ("no_dy_pair",                       r"No $60 < m(\ell\ell) < 150$ GeV"),
        ("ak8_jets_with_lsf",               r"Merged AK8 jet with $\Delta\phi > 2.0$"),
        ("no_extra_tight_cr",                "No extra tight lepton"),
        ("no_sf_lepton_in_ak8",             r"No loose SF lepton in merged jet"),
        ("of_lepton_in_ak8",                r"Loose OF lepton in merged jet"),
        ("mll_gt200_boosted",               r"$m(\ell\ell) > 200$ GeV"),
        ("mlj_gt800_boosted",               r"$m(\ell J) > 800$ GeV"),
    ],
}

# Backward-compatibility aliases for historical boosted label spellings.
_BOOSTED_KEY_ALIASES = {
    "boostedtag": "boosted_tag",
    "lead_tight_pt60_boosted": "lead_tight_lepton_pt60_boosted",
    "ak8jets_with_lsf": "ak8_jets_with_lsf",
}

# Preferred background column order for LaTeX cutflow tables
_LATEX_BKG_ORDER = ["dy", "ttbar", "nonprompt", "other"]

# Human-readable channel names for table captions
_CHANNEL_PRETTY_NAMES = {"ee": "dielectron", "mumu": "dimuon", "em": r"electron--muon"}

# Batch-mode definitions
_VARIANTS = [
    ("cumulative", {"onecut": False, "efficiency": None}),
    ("onecut",     {"onecut": True,  "efficiency": None}),
    ("running_eff",  {"onecut": False, "efficiency": "running"}),
    ("relative_eff", {"onecut": False, "efficiency": "relative"}),
]
_REGIONS = [
    ("resolved", False),
    ("boosted",  True),
]
_CHANNELS = ["mumu", "ee", "em"]

_PREVIEW_TEMPLATE = r"""\documentclass[12pt]{article}
\usepackage[landscape,margin=0.3in,paperwidth=24in,paperheight=16in]{geometry}
\usepackage{amsmath}
\usepackage{float}

%% CMS style stubs
\usepackage{caption}
\captionsetup{font=LARGE}
\newcommand{\topcaption}[1]{\caption{#1}}
\newcommand{\cmsTable}[1]{\Large #1}
\setlength{\floatsep}{4pt}
\setlength{\textfloatsep}{4pt}
\setlength{\intextsep}{4pt}

%% Allow many floats per page
\renewcommand{\topfraction}{1.0}
\renewcommand{\bottomfraction}{1.0}
\renewcommand{\textfraction}{0.0}
\renewcommand{\floatpagefraction}{0.99}
\setcounter{totalnumber}{10}
\setcounter{topnumber}{10}
\setcounter{bottomnumber}{10}

\begin{document}

{\LARGE \bfseries Cumulative cutflows}\bigskip
\input{cutflow_cumulative_resolved_mumu.tex}
\vspace{3em}
\input{cutflow_cumulative_resolved_ee.tex}
\vspace{3em}
\input{cutflow_cumulative_boosted_mumu.tex}
\vspace{3em}
\input{cutflow_cumulative_boosted_ee.tex}

\newpage
{\LARGE \bfseries One-cut cutflows}\bigskip
\input{cutflow_onecut_resolved_mumu.tex}
\vspace{3em}
\input{cutflow_onecut_resolved_ee.tex}
\vspace{3em}
\input{cutflow_onecut_boosted_mumu.tex}
\vspace{3em}
\input{cutflow_onecut_boosted_ee.tex}

\newpage
{\LARGE \bfseries Running efficiency}\bigskip
\input{cutflow_running_eff_resolved_mumu.tex}
\vspace{3em}
\input{cutflow_running_eff_resolved_ee.tex}
\vspace{3em}
\input{cutflow_running_eff_boosted_mumu.tex}
\vspace{3em}
\input{cutflow_running_eff_boosted_ee.tex}

\newpage
{\LARGE \bfseries Relative efficiency}\bigskip
\input{cutflow_relative_eff_resolved_mumu.tex}
\vspace{3em}
\input{cutflow_relative_eff_resolved_ee.tex}
\vspace{3em}
\input{cutflow_relative_eff_boosted_mumu.tex}
\vspace{3em}
\input{cutflow_relative_eff_boosted_ee.tex}

\end{document}
"""


# -------------------------------------------------------------------------
# Reading helpers
# -------------------------------------------------------------------------
def _read_cutflow(root_path, channel, unweighted=False, hist_type="cumulative",
                   expected_keys=None, boosted=False):
    """
    Read a cutflow histogram for one ROOT file.

    Args:
        root_path: Path to the ROOT file.
        channel: Lepton channel (ee, mumu, em).
        unweighted: If True, read the unweighted variant.
        hist_type: "cumulative" (sequential cuts) or "onecut" (each cut independently).
        expected_keys: Optional list of expected bin label names (from _CUT_SEQUENCES).
                       If the histogram has StrCategory labels, validates they match.
        boosted: If True, read boosted cutflow (cutflow_boosted/...).

    Returns:
        np.ndarray of yields (one per cut), or None if the file is missing.
    """
    suffix = "_unweighted" if unweighted else ""
    cutflow_prefixes = ["cutflow_boosted"] if boosted else ["cutflow_resolved", "cutflow"]
    candidate_keys = [f"{prefix}/{channel}/{hist_type}{suffix}" for prefix in cutflow_prefixes]

    def _normalize_keys(keys):
        return [_BOOSTED_KEY_ALIASES.get(k, k) for k in keys]

    try:
        with uproot.open(root_path) as f:
            key = next((k for k in candidate_keys if k in f), None)
            if key is None:
                available = sorted(k for k in f.keys(recursive=True, cycle=False)
                                   if "cutflow" in k.lower())
                key_hint = " or ".join(f"'{k}'" for k in candidate_keys)
                if available:
                    logger.warning(
                        "Cutflow key %s not found in %s. Available cutflow keys include: %s",
                        key_hint, root_path, ", ".join(available[:8]) + (" ..." if len(available) > 8 else ""),
                    )
                else:
                    logger.warning("Cutflow key %s not found in %s", key_hint, root_path)
                return None

            h = f[key]
            # Validate bin labels if available (StrCategory axis)
            ax_labels = h.axis().labels()
            if ax_labels is not None and expected_keys is not None:
                labels_norm = _normalize_keys(list(ax_labels))
                expected_norm = _normalize_keys(list(expected_keys))
                if labels_norm != expected_norm:
                    logger.warning(
                        "Bin labels in %s:%s don't match expected cut sequence.\n"
                        "  ROOT file: %s\n  Expected:  %s",
                        root_path, key, list(ax_labels), list(expected_keys),
                    )
            return h.values(flow=False)
    except FileNotFoundError:
        logger.debug("File not found: %s", root_path)
        return None


def _discover_samples(input_dir):
    """
    Auto-discover sample names from WRAnalyzer_*.root files in a directory.

    Returns:
        dict mapping sample_name -> Path to ROOT file
    """
    samples = {}
    for p in sorted(input_dir.glob("WRAnalyzer_*.root")):
        name = p.stem.replace("WRAnalyzer_", "")
        samples[name] = p
    return samples


# -------------------------------------------------------------------------
# Formatting
# -------------------------------------------------------------------------
def _fmt_yield(x, sci=False):
    """Format a yield value."""
    if sci:
        if x == 0:
            return "0"
        exp = int(math.floor(math.log10(abs(x))))
        mant = x / (10 ** exp)
        return f"{mant:.2f}e{exp}"
    if x >= 1e6:
        return f"{x:.3e}"
    if x >= 10:
        return f"{x:.1f}"
    if x >= 0.01:
        return f"{x:.2f}"
    return f"{x:.2e}"


def _fmt_eff(x):
    """Format efficiency as percent."""
    if x >= 10:
        return f"{x:.1f}"
    if x >= 0.1:
        return f"{x:.2f}"
    return f"{x:.3f}"


def _fmt_latex_yield(x):
    """Format for LaTeX: e.g. $1.04 \\times 10^{7}$."""
    if x == 0:
        return "0"
    if abs(x) < 10:
        return f"{x:.2f}"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / (10 ** exp)
    # Renormalize when rounding pushes mantissa to 10.00 (e.g. 9.997 → 10.00)
    if round(mant, 2) >= 10:
        exp += 1
        mant /= 10
    return f"${mant:.2f} \\times 10^{{{exp}}}$"


_fmt_latex_eff = _fmt_latex_yield


def _running_eff(yields, i):
    """Running efficiency: yields[i] / yields[0], as a percentage."""
    return 100.0 * yields[i] / yields[0] if yields[0] > 0 else 0.0


def _step_eff(yields, i):
    """Step (relative) efficiency: yields[i] / yields[i-1], as a percentage.
    Returns 100.0 for i == 0 (first cut is always 100% of itself).
    """
    if i == 0:
        return 100.0
    return 100.0 * yields[i] / yields[i - 1] if yields[i - 1] > 0 else 0.0


def _signal_display_label(mass_point):
    """Convert WR2000_N1100 to (2000,1100) for LaTeX column headers."""
    m = re.match(r'WR(\d+)_N(\d+)', mass_point)
    if m:
        return f"({m.group(1)},{m.group(2)})"
    return mass_point


# -------------------------------------------------------------------------
# Table builders
# -------------------------------------------------------------------------
def build_terminal_table(cut_labels, sample_columns, mc_total, data_col, channel,
                         onecut=False, blind_indices=frozenset()):
    """
    Build a formatted terminal table string.

    Args:
        cut_labels: list of cut display names
        sample_columns: list of (label, yields_array) for each MC sample
        mc_total: np.ndarray of total MC yields
        data_col: np.ndarray of data yields (or None)
        channel: channel name for the header
        onecut: if True, label efficiency as single-cut efficiency
    """
    n_cuts = len(cut_labels)

    has_mc = mc_total.sum() > 0

    # Column headers
    headers = ["Cut"]
    for label, _ in sample_columns:
        headers.append(label)
    if has_mc:
        headers.append("Total MC")
    if data_col is not None:
        headers.append("Data")
        if has_mc:
            headers.append("Data/MC")

    # Build rows
    rows = []
    for i in range(n_cuts):
        row = [cut_labels[i]]
        for _, yields in sample_columns:
            row.append(_fmt_yield(yields[i]))
        if has_mc:
            row.append(_fmt_yield(mc_total[i]))
        if data_col is not None:
            if i in blind_indices:
                row.append("BLINDED")
                if has_mc:
                    row.append("-")
            else:
                row.append(_fmt_yield(data_col[i]))
                if has_mc:
                    ratio = data_col[i] / mc_total[i] if mc_total[i] > 0 else 0
                    row.append(f"{ratio:.3f}")
        rows.append(row)

    # Running efficiency (relative to first cut)
    eff_label = ("--- Single-cut efficiency (%) ---" if onecut
                 else "--- Running efficiency (%) ---")
    rows.append([eff_label] + [""] * (len(headers) - 1))
    for i in range(n_cuts):
        row = [cut_labels[i]]
        for _, yields in sample_columns:
            row.append(_fmt_eff(_running_eff(yields, i)))
        if has_mc:
            row.append(_fmt_eff(_running_eff(mc_total, i)))
        if data_col is not None:
            if i in blind_indices:
                row.append("BLINDED")
            else:
                row.append(_fmt_eff(_running_eff(data_col, i)))
            if has_mc:
                row.append("")
        rows.append(row)

    # Relative efficiency (relative to previous cut)
    if not onecut:
        rows.append(["--- Relative efficiency (%) ---"] + [""] * (len(headers) - 1))
        for i in range(n_cuts):
            row = [cut_labels[i]]
            for _, yields in sample_columns:
                row.append(_fmt_eff(_step_eff(yields, i)))
            if has_mc:
                row.append(_fmt_eff(_step_eff(mc_total, i)))
            if data_col is not None:
                if i in blind_indices or (i > 0 and (i - 1) in blind_indices):
                    row.append("BLINDED")
                else:
                    row.append(_fmt_eff(_step_eff(data_col, i)))
                if has_mc:
                    row.append("")
            rows.append(row)

    # Compute column widths
    all_rows = [headers] + rows
    widths = [max(len(str(row[j])) for row in all_rows) for j in range(len(headers))]

    # Format
    lines = []
    mode_str = "onecut (each cut independently)" if onecut else "cumulative (sequential)"
    lines.append(f"Cutflow: {channel} channel [{mode_str}]")
    lines.append("=" * sum(widths + [3 * len(widths)]))

    def fmt_row(row):
        return " | ".join(str(row[j]).ljust(widths[j]) for j in range(len(headers)))

    lines.append(fmt_row(headers))
    lines.append("-+-".join("-" * w for w in widths))

    # Separator rows are the ones with "---" labels
    separator_indices = {i for i, row in enumerate(rows) if row[0].startswith("---")}
    for idx, row in enumerate(rows):
        if idx in separator_indices:
            lines.append("")
            lines.append(row[0])
            lines.append("-+-".join("-" * w for w in widths))
            continue
        lines.append(fmt_row(row))

    return "\n".join(lines)


def build_latex_table(cut_labels, sample_columns, channel, lumi, year,
                      region="resolved", n_bkg=None, onecut=False, era="",
                      efficiency=""):
    """Build a CMS-style LaTeX cutflow table.

    Background and signal columns are separated by a vertical line.
    When *efficiency* is set ("running" or "relative"), cell values are
    percentages instead of raw yields.
    """
    n_cuts = len(cut_labels)
    chan_pretty = _CHANNEL_PRETTY_NAMES.get(channel, channel)

    # Determine table variant label for filenames / LaTeX labels
    if efficiency:
        hist_label = f"{efficiency}_eff"
    elif onecut:
        hist_label = "onecut"
    else:
        hist_label = "cumulative"

    col_headers = [label for label, _ in sample_columns]
    n_cols = len(col_headers)

    # Column spec: backgrounds | signals
    if n_bkg is not None and n_bkg < n_cols:
        col_spec = "l | " + " ".join(["c"] * n_bkg) + " | " + " ".join(["c"] * (n_cols - n_bkg))
    else:
        col_spec = "l | " + " ".join(["c"] * n_cols)

    # Caption description
    if efficiency == "running":
        cut_description = (
            "Each cell shows the running efficiency (\\%) "
            "relative to the first selection row. "
        )
    elif efficiency == "relative":
        cut_description = (
            "Each cell shows the relative efficiency (\\%): the fraction of events "
            "passing the previous cut that also pass the current cut. "
        )
    elif onecut:
        cut_description = (
            "Each row shows the yield after applying only that single cut, "
            "independent of all other cuts. "
        )
    else:
        cut_description = (
            "Each row shows the yield after applying that cut and all preceding cuts. "
        )

    lines = []
    lines.append(r"\begin{table}[htp]")
    lines.append(
        f"  \\topcaption{{{era}: expected event yields for the {region}, {chan_pretty} channel. "
        f"{cut_description}"
        f"Background estimates are from simulation; "
        f"signal columns show $(m_{{W_R}},\\,m_{{N}})$ mass hypotheses in GeV. "
        f"All yields are normalized to ${lumi:.1f}\\,\\mathrm{{fb}}^{{-1}}$ ({year}).}}"
    )
    lines.append(f"  \\label{{tab:cutflow_{hist_label}_{region}_{channel}_{year}}}")
    lines.append(r"  \centering")
    lines.append(r"  \cmsTable{")
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"      \hline")
    header_str = " & ".join(["Selection"] + col_headers) + r" \\"
    lines.append(f"      {header_str}")
    lines.append(r"      \hline")

    for i in range(n_cuts):
        cols = [cut_labels[i]]
        for _, yields in sample_columns:
            if efficiency == "running":
                cols.append(_fmt_latex_eff(_running_eff(yields, i)))
            elif efficiency == "relative":
                cols.append(_fmt_latex_eff(_step_eff(yields, i)))
            else:
                cols.append(_fmt_latex_yield(yields[i]))
        lines.append("      " + " & ".join(cols) + r" \\")

    lines.append(r"      \hline")
    lines.append(r"    \end{tabular}")
    lines.append(r"  }")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def build_latex_single_table(cut_labels, yields, channel, lumi, year,
                             region="resolved", sample_label=""):
    """
    Build a CMS AN-style LaTeX cutflow table for a single sample.

    Columns: Selection | Weighted cumulative events | Running eff [%] | Relative eff [%]
    """
    n_cuts = len(cut_labels)
    chan_pretty = _CHANNEL_PRETTY_NAMES.get(channel, channel)

    lines = []
    lines.append(r"\begin{table}[htp]")
    caption_sample = f"Predicted {sample_label} yields" if sample_label else "Predicted yields"
    lines.append(
        f"  \\topcaption{{{caption_sample} for the {year} data-taking period, "
        f"corresponding to ${lumi:.2f}\\,\\mathrm{{fb}}^{{-1}}$,\n"
        f"  after the application of each selection requirement for the "
        f"{region}, {chan_pretty} channel.}}"
    )
    label_sample = sample_label.lower().replace(" ", "_") if sample_label else "sample"
    lines.append(f"  \\label{{tab:cutflow_{region}_{chan_pretty}_{year}_{label_sample}}}")
    lines.append(r"  \centering")
    lines.append(r"  \cmsTable{")
    lines.append(r"    \begin{tabular}{l | c c c}")
    lines.append(r"      \hline")
    lines.append(
        r"      Selection & Weighted cumulative events & "
        r"Running efficiency [\%] & Relative efficiency [\%] \\"
    )
    lines.append(r"      \hline")

    for i in range(n_cuts):
        cols = [
            cut_labels[i],
            _fmt_latex_yield(yields[i]),
            f"{_running_eff(yields, i):.2f}",
            f"{_step_eff(yields, i):.2f}",
        ]
        lines.append("      " + " & ".join(cols) + r" \\")

    lines.append(r"      \hline")
    lines.append(r"    \end{tabular}")
    lines.append(r"  }")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# -------------------------------------------------------------------------
# Core table generation
# -------------------------------------------------------------------------
def generate_table(era, dir_name, channel, boosted=False, onecut=False,
                   efficiency=None, unweighted=False, sample_filter=None,
                   signal_filter=None, fmt="latex", region=None, output=None,
                   verbose=False):
    """Generate a single cutflow table and write it to disk.

    Returns the output Path on success, or None on failure.
    """
    if region is None:
        region = "boosted" if boosted else "resolved"

    era_info = load_lumi(era)
    run = era_info["run"]
    year = str(era_info["year"])
    lumi = float(era_info["lumi"])

    # Default signal points from lumi.yaml
    signal_points = signal_filter
    if not signal_points:
        signal_points = default_signals(era_info, boosted=boosted, for_cutflow=True)
        logger.info("Default signal points: %s", ", ".join(signal_points))

    # Resolve input directories (handles combined eras like "2022", "2023")
    repo_dir = Path(__file__).resolve().parent.parent
    input_dirs, _ = input_dirs_for_era(era, repo_dir, dir_name)
    valid_dirs = [d for d in input_dirs if d.is_dir()]
    if not valid_dirs:
        logger.error("No input directories found for era '%s': %s",
                     era, [str(d) for d in input_dirs])
        return None

    # Discover ROOT files across all sub-era directories
    # Maps sample_name -> list of ROOT file paths (one per sub-era)
    file_map: dict[str, list[Path]] = {}
    for input_dir in valid_dirs:
        for name, path in _discover_samples(input_dir).items():
            file_map.setdefault(name, []).append(path)
    if not file_map:
        logger.error("No WRAnalyzer_*.root files found in %s",
                     [str(d) for d in valid_dirs])
        return None

    # Apply sample filter
    if sample_filter:
        file_map = {k: v for k, v in file_map.items() if k in sample_filter}
        if not file_map:
            logger.error("No ROOT files match --sample %s", sample_filter)
            return None

    logger.info("Found %d distinct samples across %d directory(ies)",
                len(file_map), len(valid_dirs))

    # Categorize samples using sample_groups config
    groups, order = load_sample_groups()
    sample_to_group = {}
    for gkey, grp in groups.items():
        for s in grp.samples:
            sample_to_group[s] = gkey

    # Cut sequence
    cut_seqs = _CUT_SEQUENCES_BOOSTED if boosted else _CUT_SEQUENCES
    cut_seq = cut_seqs[channel]
    cut_keys = [key for key, _ in cut_seq]
    cut_labels = [label for _, label in cut_seq]
    n_cuts = len(cut_seq)

    mc_yields = {}
    data_yields = None
    signal_yields = {}
    hist_type = "onecut" if onecut else "cumulative"

    for sample_name, root_paths in file_map.items():
        # Accumulate cutflow yields across sub-era ROOT files
        vals = None
        for root_path in root_paths:
            sub_vals = _read_cutflow(root_path, channel, unweighted=unweighted,
                                     hist_type=hist_type, expected_keys=cut_keys,
                                     boosted=boosted)
            if sub_vals is None:
                continue
            if len(sub_vals) != n_cuts:
                logger.warning("Skipping %s in %s: expected %d bins, got %d",
                               sample_name, root_path.parent.name, n_cuts, len(sub_vals))
                continue
            vals = sub_vals if vals is None else vals + sub_vals
        if vals is None:
            logger.warning("Skipping %s (no cutflow found)", sample_name)
            continue

        if sample_name.startswith("signal_"):
            mass_point = sample_name.replace("signal_", "")
            if not signal_points or mass_point in signal_points:
                signal_yields[mass_point] = vals
            continue

        gkey = sample_to_group.get(sample_name)
        if gkey is None:
            logger.warning("Sample '%s' not in any sample group, skipping", sample_name)
            continue

        grp = groups[gkey]
        if grp.kind == "data":
            if data_yields is None:
                data_yields = np.zeros(n_cuts)
            data_yields += vals
        else:
            if gkey not in mc_yields:
                mc_yields[gkey] = np.zeros(n_cuts)
            mc_yields[gkey] += vals

    if not mc_yields and data_yields is None:
        logger.error("No cutflow data loaded for any sample.")
        return None

    # Build ordered MC columns
    is_latex_multi = fmt == "latex"
    if is_latex_multi:
        mc_order = [k for k in _LATEX_BKG_ORDER if k in mc_yields]
        mc_order += [k for k in order if k in mc_yields and k not in mc_order]
        # Warn if any loaded MC group is missing from _LATEX_BKG_ORDER
        extra = [k for k in mc_yields if k not in _LATEX_BKG_ORDER]
        if extra:
            logger.warning(
                "MC group(s) %s not in _LATEX_BKG_ORDER — appended at end. "
                "Consider adding them to the list.",
                extra,
            )
    else:
        mc_order = [k for k in order if k in mc_yields]
    sample_columns = []
    for gkey in mc_order:
        label = groups[gkey].tlatex_alias
        sample_columns.append((label, mc_yields[gkey]))

    n_bkg = len(sample_columns)

    # Add signal columns
    for mass_point in sorted(signal_yields.keys()):
        label = _signal_display_label(mass_point) if is_latex_multi else mass_point
        sample_columns.append((label, signal_yields[mass_point]))

    # Total MC
    mc_total = np.zeros(n_cuts)
    for gkey in mc_order:
        mc_total += mc_yields[gkey]

    # Blind data in SR cuts for Run3
    blind_indices = frozenset()
    if run != "RunII":
        blind_indices = frozenset(
            i for i, key in enumerate(cut_keys) if "_sr" in key
        )

    # Build table
    if fmt == "terminal":
        table = build_terminal_table(cut_labels, sample_columns, mc_total, data_yields,
                                     channel, onecut=onecut,
                                     blind_indices=blind_indices)
    elif fmt == "latex-single":
        combined = np.zeros(n_cuts)
        for _, y in sample_columns:
            combined += y
        if data_yields is not None:
            combined += data_yields
        sample_label = ", ".join(s for s, _ in sample_columns)
        if data_yields is not None and not sample_columns:
            sample_label = "Data"
        table = build_latex_single_table(
            cut_labels, combined, channel, lumi, year,
            region, sample_label=sample_label,
        )
    else:
        table = build_latex_table(
            cut_labels, sample_columns, channel, lumi, year,
            region, n_bkg=n_bkg, onecut=onecut, era=era,
            efficiency=efficiency or "",
        )

    # Determine output path
    if output:
        out_path = Path(output)
    else:
        out_dir = repo_dir / "tables" / run / year / era / dir_name / "cutflows"
        if efficiency:
            hist_label = f"{efficiency}_eff"
        elif onecut:
            hist_label = "onecut"
        else:
            hist_label = "cumulative"
        filename = f"cutflow_{hist_label}_{region}_{channel}.tex"
        out_path = out_dir / filename

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table + "\n")
    logger.info("Table written to %s", out_path)
    return out_path


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate cutflow tables from ROOT files. "
        "Without --channel, generates all 24 tables and a preview PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_era_args(parser)
    parser.add_argument("--channel", choices=["ee", "mumu", "em"], default=None,
                        help="Lepton channel (omit to generate all tables)")
    parser.add_argument("--unweighted", action="store_true",
                        help="Show unweighted (raw event count) cutflow")
    parser.add_argument("--onecut", action="store_true",
                        help="Show onecut mode: each cut applied independently")
    parser.add_argument("--efficiency", choices=["running", "relative"],
                        default=None,
                        help="Show efficiency instead of yields: "
                        "'running' = yield[i]/yield[0], "
                        "'relative' = yield[i]/yield[i-1]")
    parser.add_argument("--boosted", action="store_true",
                        help="Show boosted cutflow instead of resolved")
    parser.add_argument("--sample", nargs="*", default=[],
                        help="Only include these samples (e.g. DYJets Muon)")
    parser.add_argument("--signal", nargs="*", default=[],
                        help="Signal mass points to include (e.g. WR2000_N100)")
    parser.add_argument("--format", choices=["terminal", "latex", "latex-single"],
                        default="latex",
                        help="Output format (default: latex)")
    parser.add_argument("--region", default=None,
                        help="Region name for LaTeX label/caption")
    parser.add_argument("-o", "--output", default=None,
                        help="Override default output path")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    # Single-table mode: --channel was provided
    if args.channel is not None:
        result = generate_table(
            era=args.era, dir_name=args.dir, channel=args.channel,
            boosted=args.boosted, onecut=args.onecut, efficiency=args.efficiency,
            unweighted=args.unweighted, sample_filter=args.sample or None,
            signal_filter=args.signal or None, fmt=args.format,
            region=args.region, output=args.output, verbose=args.verbose,
        )
        if result is None:
            sys.exit(1)
        return

    # Batch mode: generate all 16 tables + preview PDF
    era_info = load_lumi(args.era)
    run, year = era_info["run"], str(era_info["year"])
    repo_dir = Path(__file__).resolve().parent.parent
    out_dir = repo_dir / "tables" / run / year / args.era / args.dir / "cutflows"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_total = len(_VARIANTS) * len(_REGIONS) * len(_CHANNELS)
    for label, variant_opts in _VARIANTS:
        for region_name, is_boosted in _REGIONS:
            for channel in _CHANNELS:
                print(f"  {label} / {region_name} / {channel} ...", end=" ", flush=True)
                result = generate_table(
                    era=args.era, dir_name=args.dir, channel=channel,
                    boosted=is_boosted, onecut=variant_opts["onecut"],
                    efficiency=variant_opts["efficiency"],
                )
                if result is not None:
                    print("ok")
                    n_ok += 1
                else:
                    print("FAILED")

    print(f"\nGenerated {n_ok}/{n_total} tables in {out_dir}")

    # Write preview.tex
    preview_path = out_dir / "preview.tex"
    preview_path.write_text(_PREVIEW_TEMPLATE)
    print(f"Wrote {preview_path}")

    # Compile PDF
    print("Compiling preview PDF ...", end=" ", flush=True)
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "preview.tex"],
        cwd=str(out_dir), capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("ok")
        print(f"PDF: {out_dir / 'preview.pdf'}")
    else:
        print("FAILED")
        for line in result.stdout.splitlines()[-10:]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
