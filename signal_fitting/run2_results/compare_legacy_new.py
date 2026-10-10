#!/usr/bin/env python3
"""Legacy (HNWRAnalyzer, 2018) vs new-framework (WrCoffea) histograms, every region.

Reads the old-format files in this directory directly (so all of their variables
are available, not just the converted masses) and WrCoffea sets of any era under
rootfiles/<Run>/<Year>/<Era>/<set>/, and compares

  * yields per region and component (MC groups and both data streams), every set
    scaled to the legacy luminosity so that other eras compare per fb^-1,
  * data/MC closure per region in each framework,
  * the matched resolved cutflow stages (2 tight leptons + 2 jets + dR, then the
    mass cuts), which locate where a yield difference first appears,
  * new/legacy data ratios (per fb^-1) against the lepton, jet and mass variables,
  * data/MC against the same variables in every framework,
  * the jet eta-phi occupancy of the first set (legacy has no such histogram).

Run from WR_Plotter/:
    python signal_fitting/run2_results/compare_legacy_new.py
    -> prints the tables, writes framework_comparison/{yields,closure,cutflow}.csv
       and framework_comparison/{data_ratio_resolved_dy_cr,closure_resolved_dy_cr,jet_eta_phi}.pdf

A set is ``label=[<era>/]<subdir>[:<subdir holding the DYJets file>]``; the era
defaults to RunIISummer20UL18.  The default sets are the September UL18
production without and with the DY reshape (the reshape is what carries the DY
normalization in WrCoffea, since its UL18 HT-binned cross sections are bare LO),
the two March UL18 DY variants, and the 2024 skims_v2 production.

Blinding is built in and cannot be switched off here:
  * Run 3 data never enters a signal region (``wr_*_sr``) or the ee/mumu data
    cutflow rows from ``mll_gt200`` on, whatever the file holds (2024 sets made
    before analyzer-side blinding have them filled);
  * Run 2 data enters them only when the file's manifest says it was made with
    ``--unblind`` or the file predates the analyzer's blinding record (the
    September UL18 sets); a file recorded as blinded has no SR data anyway.
Masked entries are reported as "blinded" and written as empty CSV cells.

Sibling of 4_background_fits/smoothing/compare_frameworks.py (ee resolved SR only,
adds expo slopes and N_eff).
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import convert_run2_backgrounds as C                                     # noqa: E402
from wrplotter.cli_utils import setup_logging                            # noqa: E402
from wrplotter.config import load_lumi                                   # noqa: E402
from wrplotter.inputs import blinding_recorded, sr_data_unblinded       # noqa: E402
from wrplotter.paths import repo_root                                    # noqa: E402

logger = logging.getLogger("compare_legacy_new")

# dyreshape_20260907 is the production the UL18 DY reshape was DERIVED from (its
# DY is LO-HT, no reshape); dyreshape_20260907_closure holds the same DY with the
# reshape applied.  20261008_skims_v2 is the 2024 production (Run 3: blinded).
DEFAULT_SETS = ["sep_noreshape=dyreshape_20260907",
                "sep_reshaped=dyreshape_20260907:dyreshape_20260907_closure",
                "mar_loinc=20260313_lo_inc_dy", "mar_nlo=20260313_nlo_dy",
                "y24=RunIII2024Summer24/20261008_skims_v2"]
DEFAULT_ERA = "RunIISummer20UL18"
DEFAULT_PLOT_SETS = ["sep_reshaped", "y24"]
# Run 2 HNWRAnalyzer luminosity constant (pb^-1 -> fb^-1); legacy MC is scaled to it.
LEGACY_LUMI = 59.735969
MC = ["tt_tW", "DYJets", "Nonprompt", "Other"]
DATA = ["EGamma", "Muon"]
# legacy DY variants next to the one the converter uses (FILE_MAP["DYJets"])
LEGACY_EXTRA = {"DYJets_reweighted": "DYJets_MG_HT_ReweightedQCDErrorEWCorr",
                "DYJets_raw": "DYJets_MG_HT"}
# first data cutflow row the analyzer blinds in ee/mumu (em is a control channel)
BLINDED_CUTFLOW_FROM = "mll_gt200"
BLINDED_CUTFLOW_CHANNELS = ("ee", "mumu")

# new variable -> (legacy variable, coarse edges)
VARIABLES = {
    "pt_leading_lepton":     ("Lepton_0_Pt",  [60, 70, 80, 100, 150, 200, 300, 500, 1000]),
    "pt_subleading_lepton":  ("Lepton_1_Pt",  [53, 60, 70, 80, 100, 150, 200, 300]),
    "eta_leading_lepton":    ("Lepton_0_Eta", [-2.4, -2.0, -1.6, -1.4, -0.8, 0, 0.8, 1.4, 1.6, 2.0, 2.4]),
    "eta_subleading_lepton": ("Lepton_1_Eta", [-2.4, -2.0, -1.6, -1.4, -0.8, 0, 0.8, 1.4, 1.6, 2.0, 2.4]),
    "pt_leading_jet":        ("Jet_0_Pt",     [40, 60, 80, 100, 150, 200, 300, 500, 1000]),
    "pt_subleading_jet":     ("Jet_1_Pt",     [40, 60, 80, 100, 150, 200, 300, 500]),
    "mass_dilepton":         ("ZCand_Mass",   [60, 80, 86, 90, 92, 96, 100, 110, 150]),
    "mass_fourobject":       ("WRCand_Mass",  [800, 1000, 1200, 1500, 2000, 2500, 3000, 4000]),
}
XLABEL = {"pt_leading_lepton": r"leading lepton $p_T$ [GeV]",
          "pt_subleading_lepton": r"subleading lepton $p_T$ [GeV]",
          "eta_leading_lepton": r"leading lepton $\eta$",
          "eta_subleading_lepton": r"subleading lepton $\eta$",
          "pt_leading_jet": r"leading jet $p_T$ [GeV]",
          "pt_subleading_jet": r"subleading jet $p_T$ [GeV]",
          "mass_dilepton": r"$m_{ll}$ [GeV]",
          "mass_fourobject": r"$m_{lljj}$ [GeV]"}
LOGX = {"pt_leading_lepton", "pt_subleading_lepton", "pt_leading_jet",
        "pt_subleading_jet", "mass_fourobject"}

# legacy CutFlow stage -> new cumulative-cutflow label; both orderings reach the
# same cumulative selection at these three points.
CUTFLOW_STAGES = [("dRSeparation", "dr_all_pairs_gt0p4", "2 tight l + 2 jets + dR>0.4"),
                  ("mWRGT800", "mll_gt200", "+ mlljj>800, mll>200"),
                  ("mWRGT800_mll400", "mll_gt400", "+ mll>400")]
CUTFLOW_CHANNELS = {"ee": "SingleElectron", "mumu": "SingleMuon", "em": "EMu"}

# one colour + marker per plotted set (slots 1-4 of the plotting palette), legacy in black
SET_STYLES = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^"), ("#eda100", "v")]

_files: dict[Path, uproot.ReadOnlyDirectory] = {}


@dataclass(frozen=True)
class NewSet:
    label: str
    era: str
    run: str
    base: Path            # rootfiles/<Run>/<Year>/<Era>
    sub: str
    dy_sub: str
    lumi: float           # fb^-1

    @property
    def scale(self) -> float:
        """Factor that brings this set to the legacy luminosity."""
        return LEGACY_LUMI / self.lumi

    @property
    def is_run2(self) -> bool:
        return self.run == "RunII"


def parse_set(spec: str) -> NewSet:
    """``label=[<era>/]<subdir>[:<dy subdir>]`` -> NewSet."""
    if "=" not in spec:
        raise ValueError(f"set '{spec}' is not label=[<era>/]<subdir>[:<dy subdir>]")
    label, rest = spec.split("=", 1)
    path, _, dy_sub = rest.partition(":")
    era, sep, sub = path.partition("/")
    if not sep:
        era, sub = DEFAULT_ERA, path
    info = load_lumi(era)
    base = repo_root() / "rootfiles" / info["run"] / str(info["year"]) / era
    return NewSet(label=label, era=era, run=info["run"], base=base, sub=sub,
                  dy_sub=dy_sub or sub, lumi=float(info["lumi"]))


def _open(path: Path):
    if path not in _files:
        _files[path] = uproot.open(path)
    return _files[path]


def is_signal_region(region: str) -> bool:
    return region.endswith("_sr") or "_sr_" in region


def sr_data_allowed(s: NewSet, comp: str) -> bool:
    """Whether data of *comp* in *s* may enter signal regions and blinded cutflow rows."""
    if comp not in DATA:
        return True
    if not s.is_run2:
        return False
    path = new_path(s, comp)
    return sr_data_unblinded(path) or not blinding_recorded(path)


def legacy_file(comp: str):
    suffix = LEGACY_EXTRA.get(comp) or C.FILE_MAP[comp]
    return _open(HERE / f"{C.PREFIX}{suffix}.root")


def legacy_hist(comp: str, region: str, old_var: str):
    old_region = C.REGION_MAP[region][0]
    return legacy_file(comp)[f"{old_region}/{old_var}_{old_region}"]


def new_path(s: NewSet, comp: str) -> Path:
    return s.base / (s.dy_sub if comp == "DYJets" else s.sub) / f"WRAnalyzer_{comp}.root"


def new_file(s: NewSet, comp: str):
    return _open(new_path(s, comp))


def new_hist(s: NewSet, comp: str, region: str, var: str):
    """The histogram, or None when it is missing or blinded."""
    if is_signal_region(region) and not sr_data_allowed(s, comp):
        return None
    try:
        return new_file(s, comp)[f"{region}/{var}_{region}"]
    except (uproot.KeyInFileError, FileNotFoundError):
        return None


def blinded(s: NewSet, comp: str, region: str) -> bool:
    return is_signal_region(region) and not sr_data_allowed(s, comp)


def total(h) -> float:
    return float(h.values(flow=True).sum()) if h is not None else float("nan")


def coarse(h, edges):
    """Sum a fine 1D histogram into `edges` -> (values, variances)."""
    e, v, var = h.axes[0].edges(), h.values(), h.variances()
    c = 0.5 * (e[:-1] + e[1:])
    idx = [(c >= lo) & (c < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    return (np.array([v[m].sum() for m in idx]), np.array([var[m].sum() for m in idx]))


def streams_for(region: str) -> list[str]:
    if "_ee_" in region or "_emu_" in region:
        return ["EGamma"]
    if "_mumu_" in region or "_mue_" in region:
        return ["Muon"]
    return DATA


def _fmt(x: float, width: int, prec: int = 1) -> str:
    return f"{x:{width}.{prec}f}" if np.isfinite(x) else f"{'-':>{width}s}"


def _csv(x: float, nd: int):
    return round(x, nd) if np.isfinite(x) else ""


def yield_tables(sets: dict[str, NewSet], outdir: Path):
    rows, closure = [], []
    for region, (_, mass_var) in C.REGION_MAP.items():
        print(f"\n== {region}   (new sets scaled to {LEGACY_LUMI:.2f} fb^-1)")
        print(f"{'component':18s} {'legacy':>10s} "
              + " ".join(f"{s:>13s} {'/leg':>6s}" for s in sets))
        tot = {"legacy": {}, **{s: {} for s in sets}}
        for comp in MC + list(LEGACY_EXTRA) + DATA:
            leg = total(legacy_hist(comp, region, C.OLD_VAR))
            tot["legacy"][comp] = leg
            row = {"region": region, "component": comp, "legacy": _csv(leg, 2)}
            line = f"{comp:18s} {leg:10.1f} "
            for label, s in sets.items():
                if comp in LEGACY_EXTRA:
                    n, note = float("nan"), ""
                elif comp in DATA and blinded(s, comp, region):
                    n, note = float("nan"), "blinded"
                else:
                    n, note = total(new_hist(s, comp, region, mass_var)) * s.scale, ""
                tot[label][comp] = n
                r = n / leg if leg > 0 else float("nan")
                row[label] = note or _csv(n, 2)
                row[f"{label}_over_legacy"] = "" if note else _csv(r, 3)
                line += (f"{'blinded':>13s} {'':>6s} " if note else f"{_fmt(n, 13)} {_fmt(r, 6, 3)} ")
            print(line)
            rows.append(row)
        for stream in streams_for(region):
            for label, t in tot.items():
                mc = np.nansum([t[c] for c in MC])
                nondy = mc - (t["DYJets"] if np.isfinite(t["DYJets"]) else 0.0)
                data = t[stream]
                closure.append({
                    "region": region, "stream": stream, "set": label,
                    "data": "blinded" if not np.isfinite(data) and label != "legacy" else _csv(data, 1),
                    "mc": round(mc, 1),
                    "data_over_mc": _csv(data / mc, 3) if np.isfinite(data) else "",
                    # DY scale that would close the region, other groups fixed
                    "dy_scale_to_close": _csv((data - nondy) / t["DYJets"], 2)
                    if t["DYJets"] > 0 and np.isfinite(data) else ""})
    print("\n== data/MC closure   (DY sf = DY scale that closes the region)")
    print(f"{'region':26s} {'stream':7s} "
          + " ".join(f"{s:>13s} {'DY sf':>6s}" for s in ["legacy", *sets]))
    for i in range(0, len(closure), len(sets) + 1):
        grp = closure[i:i + len(sets) + 1]

        def cell(g):
            if g["data_over_mc"] == "":
                return f"{'blinded':>13s} {'':>6s}"
            sf = g["dy_scale_to_close"]
            return f"{g['data_over_mc']:13.2f} {sf if sf == '' else f'{sf:6.2f}':>6}"
        print(f"{grp[0]['region']:26s} {grp[0]['stream']:7s} " + " ".join(cell(g) for g in grp))
    for name, table in (("yields", rows), ("closure", closure)):
        with open(outdir / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)


def _cutflow_value(s: NewSet, comp: str, ch: str, stage: str) -> float:
    """Cumulative cutflow value scaled to the legacy luminosity, NaN when blinded or absent."""
    if (comp in DATA and ch in BLINDED_CUTFLOW_CHANNELS and not sr_data_allowed(s, comp)):
        order = [n for _, n, _ in CUTFLOW_STAGES]
        if order.index(stage) >= order.index(BLINDED_CUTFLOW_FROM):
            return float("nan")
    try:
        h = new_file(s, comp)[f"cutflow_resolved/{ch}/cumulative"]
    except (uproot.KeyInFileError, FileNotFoundError):
        return float("nan")
    labels = [str(x) for x in h.member("fXaxis").member("fLabels")]
    if stage not in labels:          # analyzer-blinded files stop before mll_gt200
        return float("nan")
    return float(h.values()[labels.index(stage)]) * s.scale


def cutflow_table(sets: dict[str, NewSet], outdir: Path):
    rows = []
    for label, s in sets.items():
        print(f"\n== matched resolved cutflow stages, new ({label}: {s.era}/{s.sub}, "
              f"scaled to {LEGACY_LUMI:.2f} fb^-1) / legacy")
        print(f"{'sample':10s} {'ch':5s} " + " ".join(f"{d:>28s}" for _, _, d in CUTFLOW_STAGES))
        for comp in ["Muon", "EGamma", "tt_tW", "DYJets"]:
            fold = legacy_file(comp)
            for ch, old_ch in CUTFLOW_CHANNELS.items():
                line = f"{comp:10s} {ch:5s} "
                for old_stage, new_stage, desc in CUTFLOW_STAGES:
                    n = _cutflow_value(s, comp, ch, new_stage)
                    o = total(fold[f"CutFlow/NTightLeptonIsTwo_{old_ch}_{old_stage}_HNWR"])
                    if np.isfinite(n):
                        line += f"{n:10.0f}/{o:9.0f} = {n / o:5.3f} "
                    else:
                        line += f"{'blinded':>10s}/{o:9.0f}         "
                    rows.append({"set": label, "sample": comp, "channel": ch, "stage": desc,
                                 "legacy": round(o, 1), "new": _csv(n, 1),
                                 "ratio": _csv(n / o, 3) if np.isfinite(n) else ""})
                print(line)
    with open(outdir / "cutflow.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _centers(var, edges):
    return np.sqrt(edges[:-1] * edges[1:]) if var in LOGX else 0.5 * (edges[:-1] + edges[1:])


def _ratio(num, numvar, den, denvar):
    ok = (num > 0) & (den > 0)
    r = np.where(ok, num / np.where(ok, den, 1), np.nan)
    err = r * np.sqrt(np.where(ok, numvar / np.where(ok, num, 1) ** 2
                               + denvar / np.where(ok, den, 1) ** 2, 0))
    return r, err


DY_CR_SERIES = [("wr_ee_resolved_dy_cr", "EGamma", "ee (EGamma)"),
                ("wr_mumu_resolved_dy_cr", "Muon", r"$\mu\mu$ (Muon)")]


def plot_data_ratio(sets: dict[str, NewSet], plot_sets: list[str], outdir: Path):
    """new/legacy data ratio (per fb^-1) per variable in the two resolved DY CRs."""
    hep.style.use("CMS")
    fig, axes = plt.subplots(2, 8, figsize=(44, 12), sharey=True)
    for row, (region, stream, title) in enumerate(DY_CR_SERIES):
        for ax, (var, (old_var, edges)) in zip(axes[row], VARIABLES.items()):
            edges = np.asarray(edges, dtype=float)
            cen = _centers(var, edges)
            o, ovar = coarse(legacy_hist(stream, region, old_var), edges)
            for (color, marker), label in zip(SET_STYLES, plot_sets):
                s = sets[label]
                h = new_hist(s, stream, region, var)
                if h is None:
                    continue
                n, nvar = coarse(h, edges)
                n, nvar = n * s.scale, nvar * s.scale ** 2
                # the two selections share most events in 2018, so this overstates the error there
                r, err = _ratio(n, nvar, o, ovar)
                ax.errorbar(cen, r, yerr=err, xerr=[cen - edges[:-1], edges[1:] - cen], fmt=marker,
                            color=color, markersize=8, linewidth=2, label=f"{label} ({s.era})")
                ax.axhline(n.sum() / o.sum(), color=color, linestyle=":", linewidth=1.5)
            ax.axhline(1, color="black", linestyle="--", linewidth=1)
            if var in LOGX:
                ax.set_xscale("log")
            ax.set_xlabel(XLABEL[var], fontsize=22)
            ax.set_ylim(0.4, 1.6)
            ax.grid(axis="y", alpha=0.3)
        axes[row, 0].set_ylabel(f"{title}\ndata: new / legacy", fontsize=22)
    axes[0, 0].legend(fontsize=16, loc="lower left", title="resolved DY CR", title_fontsize=16)
    fig.suptitle(f"Data per fb$^{{-1}}$, WrCoffea over HNWRAnalyzer 2018 ({LEGACY_LUMI:.2f} fb$^{{-1}}$); "
                 "dotted = ratio of totals", fontsize=24)
    fig.tight_layout()
    fig.savefig(outdir / "data_ratio_resolved_dy_cr.pdf")
    plt.close(fig)


def _mc_coarse(src, region, var, edges, old_var=None):
    v = np.zeros(len(edges) - 1)
    w = v.copy()
    for comp in MC:
        h = legacy_hist(comp, region, old_var) if src == "legacy" else new_hist(src, comp, region, var)
        if h is None:
            continue
        a, b = coarse(h, edges)
        v += a
        w += b
    return v, w


def plot_closure(sets: dict[str, NewSet], plot_sets: list[str], outdir: Path):
    """data/MC per variable in the two resolved DY CRs, legacy and every plotted set."""
    hep.style.use("CMS")
    fig, axes = plt.subplots(2, 8, figsize=(44, 12), sharey=True)
    for row, (region, stream, title) in enumerate(DY_CR_SERIES):
        for ax, (var, (old_var, edges)) in zip(axes[row], VARIABLES.items()):
            edges = np.asarray(edges, dtype=float)
            cen = _centers(var, edges)
            d, dvar = coarse(legacy_hist(stream, region, old_var), edges)
            m, _ = _mc_coarse("legacy", region, var, edges, old_var)
            ok = (m > 0) & (d > 0)
            ax.errorbar(cen, np.where(ok, d / np.where(ok, m, 1), np.nan),
                        yerr=np.where(ok, np.sqrt(dvar) / np.where(ok, m, 1), 0), fmt="D", color="black",
                        markersize=8, linewidth=2, label="legacy HNWR 2018")
            for (color, marker), label in zip(SET_STYLES, plot_sets):
                s = sets[label]
                h = new_hist(s, stream, region, var)
                if h is None:
                    continue
                d, dvar = coarse(h, edges)
                m, _ = _mc_coarse(s, region, var, edges)
                ok = (m > 0) & (d > 0)
                ax.errorbar(cen, np.where(ok, d / np.where(ok, m, 1), np.nan),
                            yerr=np.where(ok, np.sqrt(dvar) / np.where(ok, m, 1), 0), fmt=marker, color=color,
                            markersize=8, linewidth=2, label=f"{label} ({s.era})")
            ax.axhline(1, color="0.4", linestyle="--", linewidth=1)
            if var in LOGX:
                ax.set_xscale("log")
            ax.set_xlabel(XLABEL[var], fontsize=22)
            ax.set_ylim(0.4, 2.0)
            ax.grid(axis="y", alpha=0.3)
        axes[row, 0].set_ylabel(f"{title}\ndata / MC", fontsize=22)
    axes[0, 0].legend(fontsize=16, loc="upper left", title="resolved DY CR", title_fontsize=16)
    fig.suptitle("Data/MC in each framework (resolved DY CRs)", fontsize=24)
    fig.tight_layout()
    fig.savefig(outdir / "closure_resolved_dy_cr.pdf")
    plt.close(fig)


def plot_jet_eta_phi(s: NewSet, outdir: Path):
    """Leading+subleading jet eta-phi occupancy of the new set (data and DY MC)."""
    region = "wr_mumu_resolved_dy_cr"
    hep.style.use("CMS")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, comp, title in zip(axes, ["Muon", "DYJets"], ["data (Muon)", "DY+jets MC"]):
        hs = [new_hist(s, comp, region, f"etavsPhi_{j}Jet") for j in ("leading", "subleading")]
        v = hs[0].values() + hs[1].values()
        ex, ey = hs[0].axes[0].edges(), hs[0].axes[1].edges()
        # 0.2 x 0.2 cells so that the empty patches are not lost in the noise
        v = v.reshape(v.shape[0] // 2, 2, v.shape[1] // 2, 2).sum(axis=(1, 3))
        mesh = ax.pcolormesh(ex[::2], ey[::2], v.T, cmap="Blues", vmin=0)
        fig.colorbar(mesh, ax=ax, label="jets / (0.2 x 0.2)")
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_xlabel(r"jet $\eta$")
        ax.set_ylabel(r"jet $\phi$")
        ax.set_title(f"{title}, {region}", fontsize=20)
    fig.suptitle(f"WrCoffea {s.era}/{s.sub}: the two selected jets", fontsize=24)
    fig.tight_layout()
    fig.savefig(outdir / "jet_eta_phi.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sets", nargs="+", default=DEFAULT_SETS,
                   help="label=[<era>/]<subdir>[:<subdir for DYJets>] (era default "
                        f"{DEFAULT_ERA}); the first one is used for the eta-phi plot")
    p.add_argument("--plot-sets", nargs="+", default=None,
                   help=f"labels overlaid in the per-variable plots (default {DEFAULT_PLOT_SETS} "
                        "when present, else the first set; at most 4)")
    p.add_argument("--output-dir", type=Path, default=HERE / "framework_comparison")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)
    sets = {s.label: s for s in map(parse_set, args.sets)}
    plot_sets = args.plot_sets or [x for x in DEFAULT_PLOT_SETS if x in sets] or [next(iter(sets))]
    unknown = [x for x in plot_sets if x not in sets]
    if unknown or len(plot_sets) > len(SET_STYLES):
        p.error(f"--plot-sets must name at most {len(SET_STYLES)} of {list(sets)}; got {plot_sets}")
    for s in sets.values():
        if not s.is_run2:
            logger.info("%s (%s) is Run 3: signal-region data and blinded cutflow rows are masked",
                        s.label, s.era)
        elif any(not sr_data_allowed(s, c) for c in DATA):
            logger.info("%s (%s): data recorded as blinded, signal-region data absent", s.label, s.era)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    yield_tables(sets, args.output_dir)
    cutflow_table(sets, args.output_dir)
    plot_data_ratio(sets, plot_sets, args.output_dir)
    plot_closure(sets, plot_sets, args.output_dir)
    plot_jet_eta_phi(next(iter(sets.values())), args.output_dir)
    logger.info("wrote %s", args.output_dir)


if __name__ == "__main__":
    main()
