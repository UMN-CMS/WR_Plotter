#!/usr/bin/env python3
"""Run-2 DY CR m_lljj with the UNSHAPED Drell-Yan, in plotter format.

Same conversion as convert_run2_backgrounds.py, but the DY sample is the fully raw
LO MG-HT file: no QCD/EW reweighting and no DY reshape (the corrections derived in
the DY CR). Writes only the resolved DY CR histograms (m_lljj and the leading jet pT) to
rootfiles/RunII/2018/RunIISummer20UL18/20260929_run2_bkgs_unshaped/.

Run from WR_Plotter/:
    python signal_fitting/run2_results/convert_run2_dycr_unshaped.py
    python bin/make_stackplots.py --era RunIISummer20UL18 --dir 20260929_run2_bkgs_unshaped \
        -r resolved_dy_cr -v mass_fourobject,pt_leading_jet --local-plots

Made 2026-09-29 for the HEP-seminar slide on DY CR corrections
(research-slides/talks/20261007_hep_seminar).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import convert_run2_backgrounds as C  # noqa: E402

C.FILE_MAP["DYJets"] = "DYJets_MG_HT"
C.REGION_MAP = {k: v for k, v in C.REGION_MAP.items() if k.endswith("resolved_dy_cr")}
OUT = HERE.parents[1] / "rootfiles" / "RunII" / "2018" / "RunIISummer20UL18" / "20260929_run2_bkgs_unshaped"

# extra variables: old name -> (plotter name, rebin factor to the plotter's native axis)
EXTRA_VARS = {"Jet_0_Pt": ("pt_leading_jet", 10)}     # 1 GeV [0,2000] -> 10 GeV [0,2000]


def add_extra_vars(src, dst):
    """Append the EXTRA_VARS histograms (rebinned to the native axis) to dst."""
    import hist
    import uproot
    with uproot.open(src) as fin, uproot.update(dst) as fout:
        keys = set(fin.keys(cycle=False))
        for new_region, (old_region, _) in C.REGION_MAP.items():
            for old_var, (new_var, rb) in EXTRA_VARS.items():
                key = f"{old_region}/{old_var}_{old_region}"
                if key not in keys:
                    continue
                h_old = fin[key]
                v, var = h_old.values(flow=True), h_old.variances(flow=True)
                edges = h_old.axes[0].edges()[::rb]
                h = hist.Hist(hist.axis.Variable(edges, name="x"), storage=hist.storage.Weight())
                view = h.view(flow=True)
                view["value"][1:-1] = v[1:-1].reshape(-1, rb).sum(axis=1)
                view["variance"][1:-1] = var[1:-1].reshape(-1, rb).sum(axis=1)
                view["value"][0], view["value"][-1] = v[0], v[-1]
                view["variance"][0], view["variance"][-1] = var[0], var[-1]
                fout[f"{new_region}/{new_var}_{new_region}"] = h


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for new_sample, old_suffix in C.FILE_MAP.items():
        src = HERE / f"{C.PREFIX}{old_suffix}.root"
        dst = OUT / f"WRAnalyzer_{new_sample}.root"
        C.convert_file(src, dst)
        add_extra_vars(src, dst)
    print("->", OUT)
