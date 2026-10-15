#!/usr/bin/env python3
"""Draw the DY jet-pT reshape of an era (AN-19-083 Fig. 24) and, given a dy-cr-split study,
its DY CR1 -> CR2 closure (AN-19-083 Figs. 27-30).

Outputs under <plots>/<Run>/<Year>/<Era>/[<dir>/]dy_reshape/:

    ratio_{resolved,boosted}[_<tag>].pdf
        the payload: (data - non-DY MC) / DY MC vs the leading AK4 jet pT (resolved DY CRs) or
        the DY-candidate AK8 jet pT (boosted DY CRs).  Points: ee + mumu combined with their
        statistical error; band: the statistical and ee-vs-mumu (channel) uncertainties in
        quadrature; lines: the ee and mumu ratios.
    cr2_closure/<region>/<variable>_<region>_{before,after}.pdf
        with --after: the stacks of the DY CR2 halves (100 < m(ll) < 150 GeV) of a production
        made with ``run_analysis.py --study dy-cr-split`` (--dir; DY with --dy-reweight zpt), before
        and after the reshape derived in DY CR1 (--after DIR: DY rerun with --dy-reweight
        zpt+reshape and the CR1 payload, scripts/derive_dy_reshape.py --region-suffix _mll60to100).

The payload defaults to the one config.yaml (dy_reshape) maps for the era; --payload draws
another one, --tag names its outputs.

Examples:
    python bin/plot_dy_reshape.py --era RunIII2024Summer24 --dir 20261013_skims_v2 --local-plots
    python bin/plot_dy_reshape.py --era RunIII2024Summer24 --dir 20261013_dycr_split \\
        --after 20261013_dycr_split_cr1reshape --payload <cr1 payload.json> --tag cr1 --local-plots
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wrplotter.cli_utils import add_era_args, add_output_args, setup_logging
from wrplotter.config import (configured_variables, era_label, index_plot_settings, load_lumi,
                              load_plot_settings)
from wrplotter.dy_zpt import C_REFERENCE, C_VARIATIONS
from wrplotter.inputs import make_inputs
from wrplotter.paths import plots_base, read_yaml, repo_root, save_figure
from wrplotter.plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header, plot_stack
from wrplotter.regions import Region, regions_by_name
from wrplotter.sample_groups import load_sample_groups
from wrplotter.stack import axis_config, build_stack
from wrplotter.variables import build_variables

XLABEL = {"pt_leading_jet": r"$p_{T}$ of the leading AK4 jet [GeV]",
          "pt_leading_AK8Jets": r"$p_{T}$ of the leading AK8 jet [GeV]"}
TOPOLOGY = {"resolved": "Resolved", "boosted": "Boosted"}
CR2_SUFFIX = "_mll100to150"
CR2_TEXT = r"$100 < m_{\ell\ell} < 150$ GeV"
# Per topology: the fit observable (AN-19-083 Figs. 27-30) and the variable the reshape acts on.
CR2_VARIABLES = {"resolved": ("mass_fourobject", "pt_leading_jet"),
                 "boosted": ("mass_twoobject", "pt_leading_AK8Jets")}


def analyzer_root() -> Path:
    return repo_root().parent


def default_payload(era: str) -> Path:
    cfg = read_yaml(analyzer_root() / "wrcoffea/config.yaml")
    path = (cfg.get("dy_reshape") or {}).get(era)
    if not path:
        raise SystemExit(f"config.yaml maps no dy_reshape payload for {era}; pass --payload")
    return analyzer_root() / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    add_era_args(parser, required=True)
    add_output_args(parser)
    parser.add_argument("--payload", default=None, help="Reshape payload JSON (default: config.yaml's for the era).")
    parser.add_argument("--tag", default="", help="Suffix of the ratio plots' names (ratio_<kind>_<tag>.pdf).")
    parser.add_argument("--after", default=None, metavar="DIR",
                        help="Directory of the DY rerun with the CR1 reshape: draw the CR2 closure stacks.")
    return parser.parse_args()


def save(fig, path: Path) -> None:
    try:
        save_figure(fig, path)
        logging.info("  %s", path)
    finally:
        plt.close(fig)


def plot_ratio(block: dict, *, lumi: float, com: float, text: str, year: str | None = None):
    """One topology's reshape ratio: combined points with stat errors, the stat (+) channel band,
    and the ee and mumu ratios as lines; bins without DY MC (below the jet threshold) are left out."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    e = np.asarray(block["edges"], dtype=float)
    keep = np.asarray(block["yields"]["dy_mc"], dtype=float) > 0
    lo, hi = e[:-1][keep], e[1:][keep]
    edges = np.append(lo, hi[-1])
    r, stat = np.asarray(block["ratio"])[keep], np.asarray(block["stat_err"])[keep]
    total = np.hypot(stat, np.asarray(block["channel_err"])[keep])
    ax.stairs(r + total, edges, baseline=r - total, fill=True, color=C_REFERENCE, alpha=0.4, lw=0,
              label=r"Stat. $\oplus$ ee vs $\mu\mu$")
    ax.stairs(np.asarray(block["ee_ratio"])[keep], edges, color=C_VARIATIONS[0], lw=2, label="ee")
    ax.stairs(np.asarray(block["mumu_ratio"])[keep], edges, color=C_VARIATIONS[1], lw=2, ls="--", label=r"$\mu\mu$")
    ax.errorbar(0.5 * (lo + hi), r, xerr=0.5 * (hi - lo), yerr=stat, fmt="o", color="k", ms=8,
                elinewidth=1.5, label=r"ee + $\mu\mu$ (stat. unc.)")
    ax.axhline(1.0, color="0.5", lw=1, ls=":")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0.0, 2.5)
    ax.set_xlabel(XLABEL.get(block["variable"], block["variable"]))
    ax.set_ylabel(r"(Data $-$ non-DY) / DY")
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    cms_label(ax, lumi, com, data=True, text=text, year=year)
    fit_header(ax)
    return fig


def cr2_region(parent: Region) -> Region:
    """The DY CR2 half of a DY CR, with the parent's data stream and a label naming the m(ll) window."""
    sep = "\\n" if "\\n" in parent.tlatex_alias else "\n"
    lines = parent.tlatex_alias.split(sep)
    lines[1] = lines[1].replace("DY CR", "DY CR2")
    lines.insert(2, CR2_TEXT)
    return Region(parent.name + CR2_SUFFIX, parent.primary_dataset, sep.join(lines), parent.year)


def main():
    args = parse_args()
    setup_logging(args.verbose)
    info = load_lumi(args.era)
    run, year, lumi, com = info["run"], info["year"], info["lumi"], float(info.get("com", 13.6))
    base = plots_base(run, year, args.era, args.dir or None, local=args.local_plots) / "dy_reshape"

    payload_path = Path(args.payload) if args.payload else default_payload(args.era)
    payload = json.loads(payload_path.read_text())
    regions = ((payload.get("provenance") or {}).get("regions") or {}).get("resolved", {})
    cr1 = any(r.endswith("_mll60to100") for r in regions.values())
    logging.info("Era %s: payload %s -> %s", args.era, payload_path, base)
    name = "_".join(filter(None, ["ratio", "{kind}", args.tag])) + ".pdf"
    for kind in ("resolved", "boosted"):
        where = [f"{TOPOLOGY[kind]} DY CR1", r"$60 < m_{\ell\ell} < 100$ GeV"] if cr1 else [f"{TOPOLOGY[kind]} DY CRs"]
        lines = [*where, r"DY: LO HT-binned, Z $p_{T}$ reweight"]     # the year goes beside the luminosity
        save(plot_ratio(payload[kind], lumi=lumi, com=com, text="\n".join(lines), year=era_label(args.era)),
             base / name.format(kind=kind))

    if not args.after:
        return
    if not args.dir:
        raise SystemExit("--after needs --dir (the dy-cr-split production)")
    try:
        before = make_inputs(args.era, args.dir, ["lo_ht_zpt"])[0]
        after = make_inputs(args.era, args.dir, [f"lo_ht_zpt_reshape={args.after}"])[0]
    except (FileNotFoundError, ValueError) as exc:
        logging.error(str(exc))
        sys.exit(2)
    groups, order = load_sample_groups()
    groups = [groups[k] for k in order]
    region_cfgs, common_vars = index_plot_settings(load_plot_settings(args.era))
    variables = {v.name: v for v in build_variables()}
    parents = regions_by_name(args.era)
    for kind, var_names in CR2_VARIABLES.items():
        for ch in ("ee", "mumu"):
            parent = parents[f"wr_{ch}_{kind}_dy_cr"][0]
            region = cr2_region(parent)
            for var in var_names:
                (variable, vcfg), = configured_variables(region_cfgs, common_vars, parent.name, [variables[var]])
                axes = axis_config(vcfg, variable_rebin=False)
                for when, inputs, extra in (("before", before, None), ("after", after, "Reshape from DY CR1")):
                    stack = build_stack(region, var, inputs=inputs, groups=groups, binning=axes.binning)
                    if stack.empty:
                        logging.warning("  no histograms for %s / %s (%s), skipped", region.name, var, when)
                        continue
                    label = "\n".join([f"DY: {inputs.dy.label}".replace(", ", ",\n")] + ([extra] if extra else []))
                    fig = plot_stack(region, variable,
                                     stack_list=[h for _, h in stack.mc],
                                     stack_colors=[g.color for g, _ in stack.mc],
                                     stack_labels=[g.tlatex_alias for g, _ in stack.mc],
                                     data_hist=[stack.data] if stack.data is not None else [],
                                     xlim=axes.xlim, ylim=axes.ylim, ratio_ylim=axes.ratio_ylim,
                                     lumi=lumi, com=com, show_data=not stack.data_blinded,
                                     extra_label=label, per_unit=axes.binning.per_gev)
                    save(fig, base / "cr2_closure" / region.name / f"{var}_{region.name}_{when}.pdf")


if __name__ == "__main__":
    main()
