#!/usr/bin/env python3
"""Compare the signal samples of several eras: efficiencies and shapes.

Give two or more --era ERA[=DIR], each an era or a combination of data/lumi.yaml
(2022 = Run3Summer22 + Run3Summer22EE, Run3, ...) read from DIR, else from --dir.
The first is the reference of every ratio.  Only the signal files are read
(WRAnalyzer_signal_WR<m>_N<m>.root with their manifests), so a signal-only
directory works.  Definitions: wrplotter/signal_compare.py.

    <plots>/<Run>/<Year>/<Era>/<dir>/signal_compare/<era>_vs_<era>.../   (reference era and dir)
        efficiency/<topology>_<channel>_WR<m>.pdf
            acceptance x efficiency vs m_N at each m_WR, for the resolved and
            boosted SRs and their sum (combined), with the ratio to the reference;
        resolution/<topology>_<channel>_WR<m>.pdf
            position (median / m_WR) and width (sigma_eff / median) of the
            signal-region mass peak vs m_N;
        mass/<region>/<point>.pdf
            signal-region mass of each point common to all sets, unit area;
        cutflow/<topology>_<channel>_<point>.pdf      (--points)
            cumulative efficiency of the channel's flavour after each cut (base weight);
        kinematics/<region>/<variable>_<point>.pdf    (--points)
            the other signal-region variables, unit area.
    tables/<Run>/<Year>/<Era>/<dir>/signal_compare/<era>_vs_<era>.../
        efficiency.csv     per set, point, topology and channel: efficiency, its factors
                           (no_cuts, selection, scale factors; skim and filters) and the mass peak;
        scale_factors.csv  mean of each weight factor (lepton SFs, pileup) in each signal region;
        cutflow.csv        cutflow efficiencies per set, point, topology, channel and cut;
        summary.txt        each set against the reference over the points both have, and the mean
                           weight factors of every set.

A set read from another directory than the reference's appears as <era>-<dir>
in the folder name.  --points picks the points of the cutflow and kinematics
plots (default: the cutflow_signals of data/lumi.yaml that every set has; "all"
= every common point).  -o DIR writes the plots and tables there instead.

Examples:
    python bin/compare_signal_eras.py --era RunIII2024Summer24 --era 2022 --era 2023 --dir 20261008_signals --local-plots
    python bin/compare_signal_eras.py --era RunIISummer20UL18 --era 2022 --era 2023 --era 2024 --dir 20261008_signals --local-plots
    python bin/compare_signal_eras.py --era 2024=20261008_signals --era 2024=20261008_skims_v2 --points WR4000_N2100 --local-plots
"""

import argparse
import csv
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wrplotter.cli_utils import add_output_args, parse_multi, setup_logging
from wrplotter.config import default_signals, list_eras, load_lumi
from wrplotter.inputs import mass_point
from wrplotter.paths import plots_base, repo_root, save_figure
from wrplotter.signal_compare import (CHANNEL_LABELS, CHANNELS, COMBINED, MASS_LABELS, MASS_VARIABLES,
                                      NAN, TOPOLOGIES, TOPOLOGY_LABELS, EraSet, Factors, Produced, Shape,
                                      Value, common_points, cutflow_efficiency, efficiency,
                                      efficiency_factors, load_point_hist, make_era_sets,
                                      no_cuts_breakdown, plot_cutflow_efficiency, plot_resolution,
                                      plot_shape_overlay, plot_vs_mn, produced_events, region_variables,
                                      set_points, shape_summary, sr_region, weight_factor_means)
from wrplotter.stack import hist_key
from wrplotter.variables import build_variables

EFFICIENCY_TOPOLOGIES = TOPOLOGIES + (COMBINED,)


@dataclass
class SetResults:
    """What one era set gives; keys are mass points or (point, topology, channel)."""
    produced: dict[str, Produced] = field(default_factory=dict)
    efficiency: dict[tuple, Value] = field(default_factory=dict)
    factors: dict[tuple, Factors] = field(default_factory=dict)
    shapes: dict[tuple, Shape] = field(default_factory=dict)
    skim_filters: dict[str, tuple[float, float]] = field(default_factory=dict)
    weight_factors: dict[tuple, dict[str, float]] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("\n\n", 1)[1])
    parser.add_argument("--era", dest="eras", action="append", required=True, metavar="ERA[=DIR]",
                        help="Era or combination (data/lumi.yaml), optionally with its directory. "
                             "Repeat or comma-separate; the first is the reference.")
    parser.add_argument("--dir", dest="dir", default="",
                        help="Directory of the eras given without =DIR.")
    parser.add_argument("--points", action="append", default=None,
                        help="Mass points of the cutflow and kinematics plots (WR4000_N2100, ...), or "
                             "'all'. Repeat or comma-separate.")
    parser.add_argument("--no-kinematics", action="store_true",
                        help="Skip the kinematics plots.")
    parser.add_argument("--analyzer-root", type=Path, default=None,
                        help="WrCoffea checkout whose data/configs (cross sections) and data/filesets "
                             "(skim counts) are read (default: the repository above WR_Plotter).")
    parser.add_argument("--fileset", dest="filesets", action="append", default=None, metavar="ERA=PATH",
                        help="The signal fileset the analyzer ran with for a single era (analyzer "
                             "--fileset): its cross sections and skim counts replace the era's "
                             "data/configs and data/filesets. Repeatable.")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Write the plots and tables here instead (tables at the top, plots in "
                             "their subdirectories).")
    add_output_args(parser)
    parser.add_argument("--verbose", action="store_true", help="DEBUG-level logging.")
    args = parser.parse_args()

    known = set(list_eras())
    specs = []
    for item in parse_multi(args.eras) or []:
        era, _, directory = item.partition("=")
        era, directory = era.strip(), (directory.strip() or args.dir)
        if era not in known:
            parser.error(f"unknown era {era!r}; known: {', '.join(sorted(known))}")
        specs.append((era, directory))
    if len(specs) < 2:
        parser.error("give at least two --era")
    if len(set(specs)) < len(specs):
        parser.error("an (era, directory) pair is given twice")
    args.specs = specs
    args.points = parse_multi(args.points)
    filesets = {}
    for item in args.filesets or []:
        era, _, path = item.partition("=")
        era, path = era.strip(), Path(path.strip()).expanduser()
        if era not in known or "sub_eras" in load_lumi(era):
            parser.error(f"--fileset needs a single era of data/lumi.yaml, not {era!r}")
        if not path.is_file():
            parser.error(f"--fileset {era}: no file {path}")
        filesets[era] = path
    args.filesets = filesets
    return args


def save(fig, path: Path) -> bool:
    try:
        save_figure(fig, path)
        return True
    except Exception as exc:  # EOS upload errors must not stop the batch
        logging.error("  failed to save %s: %s", path, exc)
        return False
    finally:
        plt.close(fig)


def fmt(x: float | None, digits: int = 4) -> str:
    return "" if x is None or not math.isfinite(x) else f"{x:.{digits}g}"


def mean(values) -> float:
    values = [v for v in values if math.isfinite(v)]
    return sum(values) / len(values) if values else math.nan


def default_points(sets: list[EraSet], common: list[str]) -> list[str]:
    """The cutflow_signals of data/lumi.yaml of any set that every set has (else first, middle, last)."""
    wanted = []
    for s in sets:
        for boosted in (False, True):
            wanted += default_signals(load_lumi(s.era), boosted=boosted, for_cutflow=True)
    points = [p for p in dict.fromkeys(wanted) if p in common]
    if points or not common:
        return points
    return list(dict.fromkeys([common[0], common[len(common) // 2], common[-1]]))


def label_text(topology: str, channel: str, point: str | None = None, m_wr: int | None = None) -> str:
    lines = [f"{CHANNEL_LABELS[channel]}, {TOPOLOGY_LABELS[topology]}"]
    if point:
        wr, n = mass_point(point)
        lines.append(rf"$m_{{W_R}}$ = {wr} GeV, $m_N$ = {n} GeV")
    elif m_wr:
        lines.append(rf"$m_{{W_R}}$ = {m_wr} GeV")
    return "\n".join(lines)


def collect(era_set: EraSet, analyzer_root, filesets: dict[str, Path]) -> SetResults:
    out = SetResults()
    for point in set_points(era_set):
        try:
            produced = produced_events(era_set, point, analyzer_root=analyzer_root, filesets=filesets)
        except ValueError as exc:
            logging.warning("%s: %s left out: %s", era_set.label, point, exc)
            continue
        out.produced[point] = produced
        split = no_cuts_breakdown(era_set, point, produced, analyzer_root=analyzer_root, filesets=filesets)
        if split is not None:
            out.skim_filters[point] = split
        for channel in CHANNELS:
            if produced.channels[channel] <= 0:
                continue          # no signal of this flavour (e.g. an electron-only point)
            for topology in EFFICIENCY_TOPOLOGIES:
                value = efficiency(era_set, point, topology, channel, produced)
                if value is not None:
                    out.efficiency[(point, topology, channel)] = value
            for topology in TOPOLOGIES:
                factors = efficiency_factors(era_set, point, topology, channel, produced)
                if factors is not None:
                    out.factors[(point, topology, channel)] = factors
                weight_factors = weight_factor_means(era_set, point, topology, channel)
                if weight_factors:
                    out.weight_factors[(point, topology, channel)] = weight_factors
                h = load_point_hist(era_set, point, hist_key(sr_region(topology, channel),
                                                             MASS_VARIABLES[topology]))
                if h is not None and h.values(flow=True).sum() > 0:
                    out.shapes[(point, topology, channel)] = shape_summary(h)
    if out.produced and not out.skim_filters:
        logging.warning("%s: no skimmed signal fileset counts, so no skim / filter split of no_cuts.",
                        era_set.label)
    logging.info("%s: %d point(s) from %s", era_set.label, len(out.produced), era_set.dir_name)
    return out


def write_efficiency_csv(path: Path, sets: list[EraSet], results: list[SetResults]) -> None:
    ref = results[0]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "era", "dir", "point", "m_wr", "m_n", "topology", "channel", "produced_events",
                    "efficiency", "efficiency_err", "ratio_to_ref", "ratio_err", "skim", "filters", "no_cuts",
                    "selection", "scale_factors", "n_eff", "median", "median_err", "q16", "q84", "sigma_eff",
                    "sigma_eff_err", "median_over_mwr", "sigma_eff_over_median", "width_ratio_to_ref"])
        for s, res in zip(sets, results):
            for key, value in res.efficiency.items():
                point, topology, channel = key
                wr, n = mass_point(point)
                ratio = value / ref.efficiency[key] if key in ref.efficiency else NAN
                skim, filters = res.skim_filters.get(point, (math.nan, math.nan))
                fa = res.factors.get(key)
                row = [s.label, s.era, s.dir_name, point, wr, n, topology, channel,
                       fmt(res.produced[point].channels[channel], 6), fmt(value.value, 6), fmt(value.error, 3),
                       fmt(ratio.value), fmt(ratio.error, 3), fmt(skim), fmt(filters)]
                row += [fmt(fa.no_cuts), fmt(fa.selection), fmt(fa.scale_factors)] if fa else [""] * 3
                shape = res.shapes.get(key)
                if shape is None:
                    row += [""] * 10
                else:
                    width = shape.sigma_eff / shape.median
                    ref_shape = ref.shapes.get(key)
                    width_ratio = width / (ref_shape.sigma_eff / ref_shape.median) if ref_shape else NAN
                    row += [fmt(shape.n_eff, 5), fmt(shape.median.value, 6), fmt(shape.median.error, 3),
                            fmt(shape.q16, 6), fmt(shape.q84, 6), fmt(shape.sigma_eff.value, 5),
                            fmt(shape.sigma_eff.error, 3), fmt(shape.median.value / wr, 5),
                            fmt(width.value), fmt(width_ratio.value)]
                w.writerow(row)


def weight_factor_lines(sets: list[EraSet], results: list[SetResults], common: list[str]) -> list[str]:
    """Mean of each weight factor per set over the common points, one table per signal region."""
    lines = []
    for topology in TOPOLOGIES:
        for channel in CHANNELS:
            # the common points every set has this region's factors for (a set without the
            # histograms shows n/a)
            have = {i for i, r in enumerate(results) if any(k[1:] == (topology, channel) for k in r.weight_factors)}
            keys = [(p, topology, channel) for p in common
                    if all((p, topology, channel) in results[i].weight_factors for i in have)]
            if not keys:
                continue
            rows = []
            for i, (s, res) in enumerate(zip(sets, results)):
                if i in have:
                    names = set.intersection(*(set(res.weight_factors[k]) for k in keys))
                    rows.append((s, {n: mean(res.weight_factors[k][n] for k in keys) for n in names}))
                else:
                    rows.append((s, None))
            # scale factors first, then pileup and their product (corrections)
            names = sorted(set().union(*(m for _, m in rows if m)),
                           key=lambda n: (n == "corrections", n == "pileup", n))
            widths = [max(len(n), 6) for n in names]
            lines.append(f"  {sr_region(topology, channel)} ({len(keys)} points)")
            lines.append(f"    {'set':<48}" + "".join(f" {n:>{w}}" for n, w in zip(names, widths)))
            for s, m in rows:
                cells = "".join(f" {fmt(m.get(n)) if m else 'n/a':>{w}}" for n, w in zip(names, widths))
                lines.append(f"    {f'{s.label} ({s.dir_name})':<48}{cells}")
    if lines:
        lines = (["Mean weight factors in the signal regions, over the common points every set has the region "
                  "for (analyzer event_weights components,",
                  "  each event counted once, to about 0.003): pileup is part of the base weight (the cutflow, so "
                  "no_cuts and select.), the lepton SFs",
                  "  make the SFs factor, corrections is the product of all; factors equal to 1 for every event "
                  "are left out."]
                 + lines + [""])
    return lines


def summary_text(sets: list[EraSet], results: list[SetResults], common: list[str]) -> str:
    """Skim and filter efficiencies over the common points, then each set against the reference over
    the points both have: efficiency ratio (mean, min, max) and the ratios of its factors, peak shift
    and width ratio."""
    ref, ref_res = sets[0], results[0]
    lines = [f"Reference: {ref.label} ({ref.dir_name})", ""]
    if common and any(r.skim_filters for r in results):
        lines.append(f"Events kept before the selection, mean over the {len(common)} common point(s):")
        lines.append(f"  {'set':<48} {'skim':>6} {'filters':>8}")
        for s, res in zip(sets, results):
            name = f"{s.label} ({s.dir_name})"
            if all(p in res.skim_filters for p in common):
                lines.append(f"  {name:<48} {mean(res.skim_filters[p][0] for p in common):6.3f} "
                             f"{mean(res.skim_filters[p][1] for p in common):8.3f}")
            else:
                lines.append(f"  {name:<48} {'n/a':>6} {'n/a':>8}   (no skimmed-fileset counts)")
        lines.append("  skim = nevts / genEventCount of the skimmed fileset; filters = unweighted no_cuts / nevts "
                     "(MET filters, jet veto map).")
        lines.append("")
    for s, res in zip(sets[1:], results[1:]):
        lines.append(f"{s.label} ({s.dir_name}) / reference, mean over the points both have:")
        lines.append(f"  {'topology':<9} {'channel':<7} {'points':>6}  {'eff.':>6} {'min':>6} {'max':>6}  |"
                     f" {'no_cuts':>7} {'select.':>7} {'SFs':>6}  | {'peak shift':>10} {'width':>6}")
        for topology in EFFICIENCY_TOPOLOGIES:
            for channel in CHANNELS:
                keys = [k for k in res.efficiency if k[1:] == (topology, channel) and k in ref_res.efficiency]
                ratios = [r for k in keys if math.isfinite(r := (res.efficiency[k] / ref_res.efficiency[k]).value)]
                if not ratios:
                    continue
                line = (f"  {topology:<9} {channel:<7} {len(ratios):>6}  {mean(ratios):6.3f} {min(ratios):6.3f} "
                        f"{max(ratios):6.3f}  |")
                pairs = [(res.factors[k], ref_res.factors[k]) for k in keys
                         if k in res.factors and k in ref_res.factors]
                if pairs:
                    line += (f" {mean(a.no_cuts / b.no_cuts for a, b in pairs):7.3f}"
                             f" {mean(a.selection / b.selection for a, b in pairs):7.3f}"
                             f" {mean(a.scale_factors / b.scale_factors for a, b in pairs):6.3f}  |")
                else:
                    line += f" {'':>7} {'':>7} {'':>6}  |"
                shapes = [(res.shapes[k], ref_res.shapes[k]) for k in keys if k in res.shapes and k in ref_res.shapes]
                if shapes:
                    shift = mean((a.median.value / b.median.value - 1) * 100 for a, b in shapes)
                    width = mean((a.sigma_eff.value / a.median.value) / (b.sigma_eff.value / b.median.value)
                                 for a, b in shapes)
                    line += f" {shift:+9.2f}% {width:6.3f}"
                lines.append(line)
        lines.append("")
    lines += weight_factor_lines(sets, results, common)
    lines += ["eff.: acceptance x efficiency over the reference's (mean, min, max over the points).",
              "no_cuts, select., SFs: the same ratio for each factor of the efficiency (per point, their product",
              "  is eff.): cutflow row 0 / sigma x L (skim x event filters), last row / row 0 (selection, base",
              "  weight), region yield / last row (mean lepton scale-factor weight).",
              "peak shift: median of the SR mass over the reference's, minus 1; width: sigma_eff / median over the "
              "reference's."]
    return "\n".join(lines)


def main():
    args = parse_args()
    setup_logging(args.verbose)
    try:
        sets = make_era_sets(args.specs)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logging.error(str(exc))
        sys.exit(2)

    ref = sets[0]
    info = load_lumi(ref.era)
    # <era>_vs_<era>...; a set read from another directory than the reference's names it too
    tag = "_vs_".join(s.era if s.dir_name == ref.dir_name else f"{s.era}-{s.dir_name}" for s in sets)
    if args.output_dir:
        base = tables = args.output_dir
    else:
        base = plots_base(info["run"], info["year"], ref.era, ref.dir_name or None,
                          local=args.local_plots) / "signal_compare" / tag
        tables = repo_root() / "tables" / info["run"] / info["year"] / ref.era
        tables = (tables / ref.dir_name if ref.dir_name else tables) / "signal_compare" / tag
    tables.mkdir(parents=True, exist_ok=True)

    results = [collect(s, args.analyzer_root, args.filesets) for s in sets]
    common = [p for p in common_points(sets) if all(p in r.produced for r in results)]
    if args.points in (None, []):
        points = default_points(sets, common)
    elif args.points == ["all"]:
        points = common
    else:
        unknown = [p for p in args.points if p not in common]
        if unknown:
            logging.error("Point(s) not in every set: %s. Common points: %s", unknown, ", ".join(common))
            sys.exit(2)
        points = args.points
    logging.info("Reference %s; %d common point(s); cutflow/kinematics for %s -> %s",
                 ref.label, len(common), ", ".join(points) or "none", base)

    n_saved = n_failed = 0

    def out(fig, path):
        nonlocal n_saved, n_failed
        if save(fig, path):
            n_saved += 1
        else:
            n_failed += 1

    # Efficiency and mass-peak summaries vs m_N, at every m_WR that two or more sets have
    wr_masses = sorted({mass_point(p)[0] for r in results for p in r.produced})
    for topology in EFFICIENCY_TOPOLOGIES:
        for channel in CHANNELS:
            for m_wr in wr_masses:
                def at_wr(values: dict) -> list:
                    return sorted(((mass_point(p)[1], v) for (p, t, c), v in values.items()
                                   if t == topology and c == channel and mass_point(p)[0] == m_wr),
                                  key=lambda x: x[0])
                series = [(s, at_wr(r.efficiency)) for s, r in zip(sets, results)]
                if sum(bool(pts) for _, pts in series) < 2:
                    continue
                out(plot_vs_mn(series, ylabel=r"Acceptance $\times$ efficiency [%]", percent=True,
                               text=label_text(topology, channel, m_wr=m_wr)),
                    base / "efficiency" / f"{topology}_{channel}_WR{m_wr}.pdf")
                if topology == COMBINED:
                    continue
                res = [(s, [(m, v) for m, v in at_wr(r.shapes)
                            if math.isfinite(v.median.value) and math.isfinite(v.sigma_eff.value)])
                       for s, r in zip(sets, results)]
                if sum(bool(pts) for _, pts in res) >= 2:
                    out(plot_resolution(res, m_wr=m_wr, text=label_text(topology, channel, m_wr=m_wr)),
                        base / "resolution" / f"{topology}_{channel}_WR{m_wr}.pdf")

    # Signal-region mass of every common point
    for point in common:
        for topology in TOPOLOGIES:
            for channel in CHANNELS:
                region = sr_region(topology, channel)
                key = hist_key(region, MASS_VARIABLES[topology])
                entries = [(s, load_point_hist(s, point, key)) for s in sets]
                if any(h is None or h.values(flow=True).sum() <= 0 for _, h in entries):
                    continue
                out(plot_shape_overlay(entries, xlabel=f"{MASS_LABELS[topology]} [GeV]",
                                       text=label_text(topology, channel, point)),
                    base / "mass" / region / f"{point}.pdf")

    # Cutflow efficiencies and kinematics of the chosen points
    labels = {v.name: (f"{v.tlatex_alias} [{v.unit}]" if v.unit else v.tlatex_alias) for v in build_variables()}
    cutflow_rows = []
    for point in points:
        for topology in TOPOLOGIES:
            for channel in CHANNELS:
                series = []
                for s, r in zip(sets, results):
                    cf = cutflow_efficiency(s, point, topology, channel, r.produced[point])
                    if cf is not None:
                        series.append((s, cf))
                        cutflow_rows += [(s, point, topology, channel, cut, v, e)
                                         for cut, v, e in zip(cf.cuts, cf.values, cf.errors)]
                if len(series) == len(sets):
                    out(plot_cutflow_efficiency(series, text=label_text(topology, channel, point)),
                        base / "cutflow" / f"{topology}_{channel}_{point}.pdf")
                if args.no_kinematics:
                    continue
                region = sr_region(topology, channel)
                for variable in region_variables(sets[0], point, region):
                    if variable == MASS_VARIABLES[topology]:
                        continue
                    entries = [(s, load_point_hist(s, point, hist_key(region, variable))) for s in sets]
                    if any(h is None or h.values(flow=True).sum() <= 0 for _, h in entries):
                        continue
                    out(plot_shape_overlay(entries, xlabel=labels.get(variable, variable),
                                           text=label_text(topology, channel, point)),
                        base / "kinematics" / region / f"{variable}_{point}.pdf")

    write_efficiency_csv(tables / "efficiency.csv", sets, results)
    with open(tables / "scale_factors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "era", "dir", "point", "topology", "channel", "factor", "mean"])
        for s, r in zip(sets, results):
            for (point, topology, channel), means in r.weight_factors.items():
                w.writerows([s.label, s.era, s.dir_name, point, topology, channel, name, fmt(value, 6)]
                            for name, value in means.items())
    with open(tables / "cutflow.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "era", "dir", "point", "topology", "channel", "cut", "efficiency", "efficiency_err"])
        for s, point, topology, channel, cut, v, e in cutflow_rows:
            w.writerow([s.label, s.era, s.dir_name, point, topology, channel, cut, fmt(v, 6), fmt(e, 3)])
    summary = summary_text(sets, results, common)
    (tables / "summary.txt").write_text(summary + "\n")
    print(summary)
    logging.info("Done: %d plot(s) saved, %d failed. Plots: %s  Tables: %s", n_saved, n_failed, base, tables)
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
