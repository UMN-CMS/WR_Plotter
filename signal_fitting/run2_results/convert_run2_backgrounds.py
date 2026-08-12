#!/usr/bin/env python3
"""Convert the Run2 HNWRAnalyzer files into WrCoffea-format plotter inputs.

The old (2018 analysis) files in this directory use
    HNWR_{SingleElectron,SingleMuon,EMu}_{Resolved,Boosted}_<REGION>/
        WRCand_Mass_HNWR_..._<REGION>
while the current pipeline reads
    wr_{ee,mumu}_{resolved,boosted}_{sr,dy_cr}/ and wr_*_flavor_cr/
        {mass_fourobject|mass_twoobject}_<region>
Both bin WRCand mass identically (800 bins, [0, 8000] GeV, 10 GeV), so this is
a pure key-renaming copy (values + sumw2 + flow preserved).

Region semantics (checked against wrcoffea/analyzer.py):
  * emu  = electron-triggered, lead e + loose mu  -> old SingleElectron_EMu_*
  * mue  = muon-triggered,     lead mu + loose e  -> old SingleMuon_EMu_*
  * dy_cr maps to the old unnumbered DYCR (DYCR1/2/3 sub-variants unmapped)
  * each old data STREAM file becomes one new dataset file (EGamma / Muon);
    old cross-stream fills (e.g. SingleMuon regions inside the SingleElectron
    stream file) are intentionally not used.

The DY background is the LO MG HT sample, ReweightedQCDErrorEWCorr_Reshaped
variant. All old MC histograms are already scaled to the 2018 luminosity
(59.83 fb^-1), matching the pre-lumi-scaled convention of 20260317_lo_dy.

Writes rootfiles/RunII/2018/RunIISummer20UL18/20260714_run2_bkgs/WRAnalyzer_*.root
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import hist
import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from wrplotter.cli_utils import setup_logging  # noqa: E402

logger = logging.getLogger("convert_run2_backgrounds")

PREFIX = "HNWRAnalyzer_SkimTree_LRSMHighPt_"

# new sample name -> old file suffix
FILE_MAP = {
    "DYJets":    "DYJets_MG_HT_ReweightedQCDErrorEWCorr_Reshaped",
    "tt_tW":     "TT_TW",
    "Nonprompt": "NonPrompt",
    "Other":     "Others",
    "EGamma":    "data_SingleElectron",
    "Muon":      "data_SingleMuon",
}

# new region -> (old region, new mass variable)
REGION_MAP = {
    "wr_ee_resolved_sr":        ("HNWR_SingleElectron_Resolved_SR",      "mass_fourobject"),
    "wr_mumu_resolved_sr":      ("HNWR_SingleMuon_Resolved_SR",          "mass_fourobject"),
    "wr_ee_resolved_dy_cr":     ("HNWR_SingleElectron_Resolved_DYCR",    "mass_fourobject"),
    "wr_mumu_resolved_dy_cr":   ("HNWR_SingleMuon_Resolved_DYCR",        "mass_fourobject"),
    "wr_resolved_flavor_cr":    ("HNWR_EMu_Resolved_SR",                 "mass_fourobject"),
    "wr_ee_boosted_sr":         ("HNWR_SingleElectron_Boosted_SR",       "mass_twoobject"),
    "wr_mumu_boosted_sr":       ("HNWR_SingleMuon_Boosted_SR",           "mass_twoobject"),
    "wr_ee_boosted_dy_cr":      ("HNWR_SingleElectron_Boosted_DYCR",     "mass_twoobject"),
    "wr_mumu_boosted_dy_cr":    ("HNWR_SingleMuon_Boosted_DYCR",         "mass_twoobject"),
    "wr_emu_boosted_flavor_cr": ("HNWR_SingleElectron_EMu_Boosted_CR",   "mass_twoobject"),
    "wr_mue_boosted_flavor_cr": ("HNWR_SingleMuon_EMu_Boosted_CR",       "mass_twoobject"),
}

OLD_VAR = "WRCand_Mass"


def convert_hist(h_old) -> hist.Hist:
    """uproot TH1 -> hist.Hist with Weight storage, flow preserved."""
    edges = h_old.axes[0].edges()
    h = hist.Hist(hist.axis.Variable(edges, name="mass"),
                  storage=hist.storage.Weight())
    view = h.view(flow=True)
    view["value"] = h_old.values(flow=True)
    view["variance"] = h_old.variances(flow=True)
    return h


def convert_file(src: Path, dst: Path) -> None:
    n_copied, n_missing = 0, 0
    with uproot.open(src) as fin, uproot.recreate(dst) as fout:
        keys = set(fin.keys(cycle=False))
        for new_region, (old_region, new_var) in REGION_MAP.items():
            old_key = f"{old_region}/{OLD_VAR}_{old_region}"
            if old_key not in keys:
                logger.warning("  %s: missing %s", src.name, old_key)
                n_missing += 1
                continue
            h_old = fin[old_key]
            fout[f"{new_region}/{new_var}_{new_region}"] = convert_hist(h_old)
            n_copied += 1
            logger.debug("  %-26s <- %-42s integral=%.2f",
                         new_region, old_region, float(h_old.values().sum()))
    logger.info("%s -> %s: %d regions copied, %d missing",
                src.name, dst.name, n_copied, n_missing)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=HERE)
    p.add_argument("--output-dir", type=Path,
                   default=HERE.parents[1] / "rootfiles" / "RunII" / "2018"
                   / "RunIISummer20UL18" / "20260714_run2_bkgs")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    setup_logging(args.verbose)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for new_sample, old_suffix in FILE_MAP.items():
        src = args.input_dir / f"{PREFIX}{old_suffix}.root"
        if not src.exists():
            raise SystemExit(f"ERROR: missing input {src}")
        convert_file(src, args.output_dir / f"WRAnalyzer_{new_sample}.root")
    logger.info("Done -> %s", args.output_dir)


if __name__ == "__main__":
    main()
