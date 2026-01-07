#!/usr/bin/env python3
import argparse
import math

import numpy as np
import uproot


# ----------------------------------------------------------------------
# Helper formatting
# ----------------------------------------------------------------------
def format_sci(x, sigfigs=3):
    """
    Format a positive number in LaTeX scientific notation, e.g.
      10400000 -> '1.04\\times 10^{7}'
    If x == 0, returns '0'.
    """
    if x == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / (10 ** exp)
    fmt = f"{{:.{sigfigs}g}}"
    mant_str = fmt.format(mant)
    return f"{mant_str}\\times 10^{exp}"


def format_eff(x, digits=3):
    """
    Format an efficiency in percent with a reasonable number of digits.
    Example: 100.0, 95.0, 0.00632
    """
    fmt = f"{{:.{digits}g}}"
    return fmt.format(x)


# ----------------------------------------------------------------------
# Cutflow configuration
# ----------------------------------------------------------------------
def get_cut_sequence(channel):
    """
    Return ordered list of (full_path, label) for the given channel.
    full_path is the key within the ROOT file.
    """
    base = "cutflow"

    if channel == "ee":
        chan_dir = f"{base}/ee"
        return [
            (f"{base}/no_cuts",
             "No cuts"),
            (f"{chan_dir}/min_two_ak4_jets_pteta",
             "At least 2 AK4 jets w/ $p_T > 40$ and $|\\eta| < 2.4$"),
            (f"{chan_dir}/min_two_ak4_jets_id",
             "At least 2 AK4 jets w/ tight lepton veto"),
            (f"{chan_dir}/min_two_pteta_electrons",
             "Two electrons w/ $p_T > 53$ and $|\\eta| < 2.4$"),
            (f"{chan_dir}/min_two_id_electrons",
             "Two electrons w/ HEEP ID"),
            (f"{chan_dir}/e_trigger",
             "Electron trigger fired"),
            (f"{chan_dir}/dr_all_pairs_gt0p4",
             "$\\Delta R > 0.4$"),
            (f"{chan_dir}/mlljj_gt800",
             "$m(\\ell\\ell jj) > 800~\\mathrm{GeV}$"),
            (f"{chan_dir}/mll_gt200",
             "$m(\\ell\\ell) > 200~\\mathrm{GeV}$"),
            (f"{chan_dir}/mll_gt400",
             "$m(\\ell\\ell) > 400~\\mathrm{GeV}$"),
        ]

    elif channel == "mumu":
        chan_dir = f"{base}/mumu"
        return [
            (f"{base}/no_cuts",
             "No cuts"),
            (f"{chan_dir}/min_two_ak4_jets_pteta",
             "At least 2 AK4 jets w/ $p_T > 40$ and $|\\eta| < 2.4$"),
            (f"{chan_dir}/min_two_ak4_jets_id",
             "At least 2 AK4 jets w/ tight lepton veto"),
            (f"{chan_dir}/min_two_pteta_muons",
             "Two muons w/ $p_T > 53$ and $|\\eta| < 2.4$"),
            (f"{chan_dir}/min_two_id_muons",
             "Two muons w/ tight ID"),
            (f"{chan_dir}/mu_trigger",
             "Muon trigger fired"),
            (f"{chan_dir}/dr_all_pairs_gt0p4",
             "$\\Delta R > 0.4$"),
            (f"{chan_dir}/mlljj_gt800",
             "$m(\\ell\\ell jj) > 800~\\mathrm{GeV}$"),
            (f"{chan_dir}/mll_gt200",
             "$m(\\ell\\ell) > 200~\\mathrm{GeV}$"),
            (f"{chan_dir}/mll_gt400",
             "$m(\\ell\\ell) > 400~\\mathrm{GeV}$"),
        ]

    elif channel == "em":
        chan_dir = f"{base}/em"
        return [
            (f"{base}/no_cuts",
             "No cuts"),
            (f"{chan_dir}/min_two_ak4_jets_pteta",
             "At least 2 AK4 jets w/ $p_T > 40$ and $|\\eta| < 2.4$"),
            (f"{chan_dir}/min_two_ak4_jets_id",
             "At least 2 AK4 jets w/ tight lepton veto"),
            (f"{chan_dir}/min_two_pteta_em",
             "One electron + one muon w/ $p_T > 53$ and $|\\eta| < 2.4$"),
            (f"{chan_dir}/min_two_id_em",
             "Electron and muon passing ID"),
            (f"{chan_dir}/emu_trigger",
             "e$\\mu$ trigger fired"),
            (f"{chan_dir}/dr_all_pairs_gt0p4",
             "$\\Delta R > 0.4$"),
            (f"{chan_dir}/mlljj_gt800",
             "$m(\\ell\\ell jj) > 800~\\mathrm{GeV}$"),
            (f"{chan_dir}/mll_gt200",
             "$m(\\ell\\ell) > 200~\\mathrm{GeV}$"),
            (f"{chan_dir}/mll_gt400",
             "$m(\\ell\\ell) > 400~\\mathrm{GeV}$"),
        ]

    else:
        raise ValueError(f"Unknown channel: {channel}. Use ee, mumu, or em.")


# ----------------------------------------------------------------------
# Core logic
# ----------------------------------------------------------------------
def read_yields(root_path, cut_sequence):
    """
    Given a list of (key, label), return:
       labels: list of labels (strings)
       yields: np.array of yields (floats)
    """
    with uproot.open(root_path) as f:
        vals = []
        labels = []
        for key, label in cut_sequence:
            try:
                h = f[key]
            except KeyError:
                raise KeyError(f"Could not find histogram '{key}' in file '{root_path}'")

            # For a single-bin TH1F, this is just that bin.
            # More generally, sum over all non-flow bins.
            try:
                values = np.array(h.values(flow=False))
            except Exception:
                values, _ = h.to_numpy(flow=False)
            total = float(values.sum())
            vals.append(total)
            labels.append(label)

    return labels, np.array(vals, dtype=float)


def make_latex_table(labels, yields, process, lumi_fb, year, channel, region="resolved"):
    """
    Build and return a LaTeX table string (no overall-efficiency summary row).
    """
    if len(yields) == 0:
        raise ValueError("No yields to tabulate")

    # Efficiencies
    running_eff = 100.0 * yields / yields[0]
    rel_eff = np.empty_like(running_eff)
    rel_eff[0] = 100.0
    for i in range(1, len(yields)):
        rel_eff[i] = 100.0 * yields[i] / yields[i - 1] if yields[i - 1] > 0 else 0.0

    # Channel pretty name
    chan_name = {
        "ee": "dielectron",
        "mumu": "dimuon",
        "em": "electron--muon",
    }.get(channel, channel)

    # Caption & label
    caption = (
        f"Predicted {process} yields for the {year} data-taking period, "
        f"corresponding to ${lumi_fb}\\,\\mathrm{{fb}}^{{-1}}$, "
        f"after the application of each selection requirement for the {region}, {chan_name} channel."
    )
    label = f"tab:cutflow_{region}_{chan_name.replace('--', '').replace(' ', '_')}_{year}"

    lines = []
    lines.append("\\begin{table}[htp]")
    lines.append(f"  \\topcaption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append("  \\centering")
    lines.append("  \\cmsTable{")
    lines.append("    \\begin{tabular}{l | c c c}")
    lines.append("      \\hline")
    lines.append("      Selection & Weighted cumulative events & Running efficiency [\\%] & Relative efficiency [\\%] \\\\")
    lines.append("      \\hline")

    for label_txt, y, re, rle in zip(labels, yields, running_eff, rel_eff):
        y_tex = f"${format_sci(y)}$"
        run_tex = format_eff(re)
        rel_tex = format_eff(rle)
        lines.append(
            f"      {label_txt} & {y_tex} & {run_tex}  & {rel_tex} \\\\"
        )

    lines.append("      \\hline")
    lines.append("    \\end{tabular}")
    lines.append("  }")
    lines.append("\\end{table}")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Make cutflow LaTeX table from ROOT file.")
    parser.add_argument("input", help="Input ROOT file (path allowed)")
    parser.add_argument("--channel", choices=["ee", "mumu", "em"], required=True,
                        help="Lepton channel (ee, mumu, em)")
    parser.add_argument("--process", default="Drell--Yan",
                        help="Process name to use in the caption (default: Drell--Yan)")
    parser.add_argument("--lumi", type=float, default=109.08,
                        help="Integrated luminosity in fb^-1 (default: 109.08)")
    parser.add_argument("--year", type=int, default=2024,
                        help="Year for the caption/label (default: 2024)")
    parser.add_argument("--region", default="resolved",
                        help="Region name for the caption/label (default: resolved)")
    parser.add_argument("--output", "-o", default=None,
                        help="If set, write LaTeX table to this file instead of stdout")
    args = parser.parse_args()

    cut_sequence = get_cut_sequence(args.channel)
    labels, yields = read_yields(args.input, cut_sequence)
    table = make_latex_table(labels, yields, args.process, args.lumi, args.year, args.channel, args.region)

    if args.output:
        with open(args.output, "w") as f:
            f.write(table + "\n")
    else:
        print(table)


if __name__ == "__main__":
    main()
