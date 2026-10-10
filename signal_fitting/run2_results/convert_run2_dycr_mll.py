#!/usr/bin/env python3
"""Old-framework (HNWRAnalyzer) ZCand_Mass in the resolved DY CR -> WrCoffea plotter format.

Companion to convert_run2_backgrounds.py, which only maps WRCand_Mass. Same FILE_MAP
and the same stream rule (the EGamma file takes only the SingleElectron region, the
Muon file only SingleMuon). The old 1 GeV [0, 2000] histogram is padded to the
plotter's native 1 GeV [0, 5000] mass_dilepton axis; values, variances and flow bins
are copied unchanged.

Run from WR_Plotter/:
    python signal_fitting/run2_results/convert_run2_dycr_mll.py
    python bin/make_stackplots.py --era RunIISummer20UL18 --dir 20260926_run2_bkgs_mll \
        -r resolved_dy_cr -v mass_dilepton --local-plots

Made 2026-09-26 for the HEP-seminar DY-CR m_ll slide
(research-slides/talks/20261007_hep_seminar).
"""
from pathlib import Path

import hist
import uproot

SRC = Path("signal_fitting/run2_results")
PREFIX = "HNWRAnalyzer_SkimTree_LRSMHighPt_"
OUT = Path("rootfiles/RunII/2018/RunIISummer20UL18/20260926_run2_bkgs_mll")
FILE_MAP = {
    "DYJets":    "DYJets_MG_HT_ReweightedQCDErrorEWCorr_Reshaped",
    "tt_tW":     "TT_TW",
    "Nonprompt": "NonPrompt",
    "Other":     "Others",
    "EGamma":    "data_SingleElectron",
    "Muon":      "data_SingleMuon",
}
REGIONS = {
    "wr_ee_resolved_dy_cr":   "HNWR_SingleElectron_Resolved_DYCR",
    "wr_mumu_resolved_dy_cr": "HNWR_SingleMuon_Resolved_DYCR",
}
STREAM_ONLY = {"EGamma": "wr_ee_resolved_dy_cr", "Muon": "wr_mumu_resolved_dy_cr"}
NB = 5000          # plotter-native mass_dilepton axis: 1 GeV bins, [0, 5000]
NB_OLD = 2000      # old ZCand_Mass axis: 1 GeV bins, [0, 2000]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for new, old in FILE_MAP.items():
        with uproot.open(SRC / f"{PREFIX}{old}.root") as fin, \
             uproot.recreate(OUT / f"WRAnalyzer_{new}.root") as fout:
            for region, old_region in REGIONS.items():
                if new in STREAM_ONLY and STREAM_ONLY[new] != region:
                    continue
                h_old = fin[f"{old_region}/ZCand_Mass_{old_region}"]
                v, var = h_old.values(flow=True), h_old.variances(flow=True)
                h = hist.Hist(hist.axis.Regular(NB, 0, NB, name="mass"),
                              storage=hist.storage.Weight())
                view = h.view(flow=True)                      # [under, NB bins, over]
                view["value"][0], view["variance"][0] = v[0], var[0]
                view["value"][1:NB_OLD + 1] = v[1:-1]
                view["variance"][1:NB_OLD + 1] = var[1:-1]
                view["value"][-1], view["variance"][-1] = v[-1], var[-1]
                fout[f"{region}/mass_dilepton_{region}"] = h
                print(f"{new:10s} {region:24s} integral {float(v[1:-1].sum()):9.1f}")
    print("->", OUT)


if __name__ == "__main__":
    main()
