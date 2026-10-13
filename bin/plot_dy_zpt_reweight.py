#!/usr/bin/env python3
"""Draw the DY Z-pT reweighting of an era: the payload the analyzer applies and, given an
analysis set, what it did to the DY.

Outputs under <plots>/<Run>/<Year>/<Era>/[<dir>/]dy_zpt_reweight/:

    qcd_ratio.pdf             QCD NLO / LO vs generator Z pT up to 1 TeV, with the scale, PDF
                              and alpha_S variations and the Run 2 ratio (AN-19-083 Fig. 14)
    qcd_ratio_full_range.pdf  the same over the payload's whole range (log x)
    ew_correction.pdf         the EW factor with its three uncertainties
    weight.pdf                QCD x EW, the weight the analyzer gives a DY event, with the total
                              uncertainty and the two factors
    spectra.pdf               the inclusive LO and NLO generator spectra the ratio came from
    applied/<region>.pdf      with --dir: the reweighted over the unreweighted DY in the reco
                              pT(ll) of each DY control region, against the payload
                              (--dy before,after; default lo_ht,lo_ht_zpt)

The before/after pT(ll) stack plots of AN-19-083 Figs. 15-18 are make_stackplots.py's:
    python bin/make_stackplots.py --era E --dir D --dy lo_ht,lo_ht_zpt -r resolved_dy_cr,boosted_dy_cr -v pt_dilepton --local-plots

The payload defaults to the one config.yaml (dy_zpt_reweight) maps for the era, the Run 2
reference to the converted Run 2 payload the EW correction was copied from.

Examples:
    python bin/plot_dy_zpt_reweight.py --era RunIII2024Summer24 --local-plots
    python bin/plot_dy_zpt_reweight.py --era RunIII2024Summer24 --dir 20261012_skims_v2 --dy lo_ht,lo_ht_zpt --local-plots
    python bin/plot_dy_zpt_reweight.py --era RunIISummer20UL18 --payload ../data/reweights/RunII/2018/RunIISummer20UL18/dy_zpt_reweight_ul_20261011.json --reference none --local-plots
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wrplotter.cli_utils import (add_dy_arg, add_era_args, add_output_args, dy_specs, select_regions,
                                 select_variables, setup_logging)
from wrplotter.config import (configured_variables, era_label, index_plot_settings, load_lumi,
                              load_plot_settings)
from wrplotter.dy_zpt import (load_zpt_payload, plot_applied_ratio, plot_ew_correction,
                              plot_qcd_ratio, plot_spectra, plot_weight)
from wrplotter.inputs import DY_SAMPLE, make_inputs
from wrplotter.paths import plots_base, read_yaml, repo_root, save_figure
from wrplotter.stack import HistLoader, axis_config, hist_key

VARIABLE = "pt_dilepton"


def analyzer_root() -> Path:
    """The analyzer repository, which holds the payloads and config.yaml (the plotter is its submodule)."""
    return repo_root().parent


def default_payload(era: str) -> Path:
    cfg = read_yaml(analyzer_root() / "wrcoffea" / "config.yaml")
    mapped = (cfg.get("dy_zpt_reweight") or {}).get(era)
    if not mapped:
        raise SystemExit(f"config.yaml maps no dy_zpt_reweight payload for {era}; give --payload.")
    return analyzer_root() / mapped


def default_reference(payload) -> Path | None:
    """The converted Run 2 payload named in the payload's ew_note, if it exists."""
    m = re.search(r"copied unchanged from (\S+\.json)", str(payload.provenance.get("ew_note", "")))
    if not m:
        return None
    found = sorted((analyzer_root() / "data" / "reweights").rglob(m[1]))
    return found[0] if found else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    add_era_args(parser)
    parser.add_argument("--payload", type=Path, default=None,
                        help="Z-pT payload JSON (default: the era's in config.yaml dy_zpt_reweight).")
    parser.add_argument("--reference", default=None,
                        help="Run 2 payload to overlay on the QCD ratio, or 'none' (default: the one the "
                             "EW correction was copied from).")
    add_dy_arg(parser, multiple=True)
    add_output_args(parser)
    args = parser.parse_args()
    args.dy = dy_specs(args.dy) or ["lo_ht", "lo_ht_zpt"]
    if len(args.dy) != 2:
        parser.error("--dy takes the unreweighted and the reweighted DY sample, in that order")
    return args


def save(fig, path: Path) -> None:
    try:
        save_figure(fig, path)
        logging.info("  %s", path)
    finally:
        plt.close(fig)


def main():
    args = parse_args()
    setup_logging(args.verbose)
    info = load_lumi(args.era)
    run, year, com = info["run"], info["year"], float(info.get("com", 13.6))

    payload = load_zpt_payload(args.payload or default_payload(args.era))
    reference = None
    if args.reference != "none":
        ref_path = Path(args.reference) if args.reference else default_reference(payload)
        if ref_path is not None:
            reference = load_zpt_payload(ref_path)
        elif args.reference:
            raise SystemExit(f"reference payload not found: {args.reference}")
        else:
            logging.info("No Run 2 reference payload found for the overlay (--reference).")
    base = plots_base(run, year, args.era, args.dir or None, local=args.local_plots) / "dy_zpt_reweight"
    logging.info("Era %s: payload %s (derived %s), reference %s -> %s", args.era, payload.era,
                 payload.provenance.get("derived_on", "?"), reference.era if reference else "none", base)

    head = f"DY Z $p_{{T}}$ reweighting\n{era_label(args.era)}"
    save(plot_qcd_ratio(payload, reference=reference, com=com, text=f"{head}\nInclusive NLO over inclusive LO"),
         base / "qcd_ratio.pdf")
    save(plot_qcd_ratio(payload, reference=reference, com=com, xlim=(1.0, payload.qcd_edges[-1]), logx=True,
                        text=f"{head}\nInclusive NLO over inclusive LO"), base / "qcd_ratio_full_range.pdf")
    save(plot_ew_correction(payload, com=com,
                            text=f"{head}\nEW correction (Run 2 payload)\nBelow 30 GeV: the 30-40 GeV value"),
         base / "ew_correction.pdf")
    save(plot_weight(payload, com=com, text=f"{head}\nApplied to the LO HT-binned DY"), base / "weight.pdf")
    try:
        save(plot_spectra(payload, com=com, text=f"{head}\nGenerator-level Z, $m_Z > 50$ GeV"), base / "spectra.pdf")
    except ValueError as exc:
        logging.warning("spectra.pdf skipped: %s", exc)

    if not args.dir:
        logging.info("No --dir: the applied/ plots need an analysis set with the DY samples %s.", args.dy)
        return
    try:
        before, after = make_inputs(args.era, args.dir, args.dy)
    except (FileNotFoundError, ValueError) as exc:
        logging.error(str(exc))
        sys.exit(2)
    region_cfgs, common_vars = index_plot_settings(load_plot_settings(args.era))
    variables = select_variables([VARIABLE])
    for region in select_regions(args.era, None):
        if not region.name.endswith("_dy_cr"):
            continue
        configured = list(configured_variables(region_cfgs, common_vars, region.name, variables))
        if not configured:
            logging.warning("  %s: no %s in the plot settings, skipped", region.name, VARIABLE)
            continue
        variable, vcfg = configured[0]
        axes = axis_config(vcfg, variable_rebin=False)
        key = hist_key(region.name, VARIABLE)
        hists = [HistLoader(inp, axes.binning)(DY_SAMPLE, key) for inp in (before, after)]
        if any(h is None for h in hists):
            logging.warning("  %s: no %s for one of the DY samples, skipped", region.name, key)
            continue
        xlabel = f"{variable.tlatex_alias} [{variable.unit}]" if variable.unit else variable.tlatex_alias
        text = f"{region.tlatex_alias}\nDY: {after.dy.label}\nover {before.dy.label}"
        save(plot_applied_ratio(payload, hists[0], hists[1], xlabel=xlabel, xlim=axes.xlim, com=com, text=text),
             base / "applied" / f"{region.name}.pdf")


if __name__ == "__main__":
    main()
