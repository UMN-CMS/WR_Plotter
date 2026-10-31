#!/usr/bin/env python3
"""Event-level signal plots of one era, the analogues of the AN-19-083 signal figures that the
analyzer's histograms cannot give (they need generator particles or objects before a cut):

    AN Figs. 3-5     generator-level pT, eta and mass of W_R, the primary lepton, N, the secondary
                     lepton and the two quarks (unit area), from the unskimmed samples;
    AN Fig. 130      generator-level W_R mass with the on-shell fraction;
    AN Fig. 10       energy of the reconstructed AK8 jet matched to a merged N over the generator
                     N energy, with and without the jet energy corrections;
    AN Fig. 11       soft-drop mass of the leading AK8 jet, the JME recipe the analyzer applies and
                     the NanoAOD value, for several mass points;
    AN Fig. 12       LSF_3 of the AK8 jet after the full boosted signal-region selection except the
                     LSF requirement (signal only; data and backgrounds need an analyzer variant);
    AN Figs. 83-87,  reconstruction efficiency of the tight leptons and of the AK8 jet of a merged
    Table 42         N as a function of the generator pT, and the LSF > 0.75 efficiency of the
                     N-matched jets.

Two steps.  ``fill`` runs one mass point and lepton flavour through the analyzer's own corrections
and object selection (noise filters, JEC/JER and the soft-drop recipe, jet veto map, electron and
muon scale/smearing, the tight/loose leptons and the boosted selection of wrcoffea.analyzer) on
its skimmed file, reads the generator particles of the unskimmed file, and writes the histograms
to ``<out>/hists/<point>_<flavour>.pkl``.  ``plot`` draws every point found there.

    ../.venv/bin/python bin/signal_event_plots.py fill --era RunIII2024Summer24 --point WR4000_N2100 --flavour El --out DIR
    ../.venv/bin/python bin/signal_event_plots.py plot --out DIR --era RunIII2024Summer24 --dir 20261013_skims_v2 --local-plots

Runs with the analyzer's environment: ``wrcoffea`` is imported from the checkout the plotter is a submodule of.

With --era/--dir the plots go next to that production's other signal outputs,
WR_Plotter/<plots>/<Run>/<Year>/<Era>/<dir>/signal_outputs/event_level/ (summary.txt under tables/ likewise);
without, into <out>/plots.

Generator objects (GenPart): the produced W_R is the last copy of pdg 34 whose parent is not a
heavy neutrino (the virtual W_R of the N decay is left out); N is the last copy of pdg 9900012/14/16;
the secondary lepton is the hard-process lepton whose distinct parent is N, the primary the other
one; the quarks are the two outgoing hard-process quarks.  A merged N
has its lepton and both quarks within dR < 0.8 of its direction; the N jet is the AK8 jet of the
analyzer's baseline selection (pT > 200, |eta| < 2.4, m_SD > 40, tight ID) within dR < 0.8 of it.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import sys
from pathlib import Path

import awkward as ak
import numpy as np
from hist import Hist
import hist

PLOTTER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PLOTTER))
sys.path.insert(0, str(PLOTTER.parent))   # the analyzer checkout: wrcoffea

logger = logging.getLogger("signal_event_plots")

N_PDG = (9900012, 9900014, 9900016)
WR_PDG = 34
FLAVOURS = {"El": ("WRtoNEltoElElJJ", "ee"), "Mu": ("WRtoNMutoMuMuJJ", "mumu")}
PT_EDGES = np.array([0, 53, 80, 120, 180, 260, 380, 550, 800, 1200, 1700, 2500, 4000], dtype=float)
ETA_EDGES = np.linspace(-2.5, 2.5, 11)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def era_paths(era: str) -> tuple[Path, Path, Path]:
    """(dataset config, skimmed fileset, unskimmed fileset) of the era's signal, in the analyzer checkout."""
    from wrplotter.signal_compare import signal_config_path, signal_fileset_path
    return (signal_config_path(era), signal_fileset_path(era, skimmed=True), signal_fileset_path(era, skimmed=False))


def dataset_name(config: dict, point: str, flavour: str) -> str:
    wr, n = point.removeprefix("WR").split("_N")
    prefix = f"{FLAVOURS[flavour][0]}_MWR{wr}_N{n}_"
    names = [k for k in config if k.startswith(prefix)]
    if len(names) != 1:
        raise KeyError(f"{point} {flavour}: {len(names)} datasets start with {prefix}")
    return names[0]


def nanoevents(url: str, metadata: dict):
    from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
    NanoAODSchema.warn_missing_crossrefs = False
    NanoAODSchema.error_missing_event_ids = False
    return NanoEventsFactory.from_root({url: "Events"}, schemaclass=NanoAODSchema, mode="virtual",
                                       metadata=metadata).events()


# ---------------------------------------------------------------------------
# Generator objects
# ---------------------------------------------------------------------------

def distinct_parent_pdg(gp, objs, steps: int = 12):
    """pdgId of the first ancestor of each object with a different pdgId (0 when there is none).

    A pure awkward walk over ``genPartIdxMother`` (coffea's ``distinctParent`` needs numba, which
    fails to compile on this machine)."""
    pdg_all, mother_all = gp.pdgId, gp.genPartIdxMother
    pdg, m = objs.pdgId, objs.genPartIdxMother
    result = ak.zeros_like(pdg)
    done = m < 0
    for _ in range(steps):
        m_safe = ak.where(m < 0, 0, m)
        mp, mm = pdg_all[m_safe], mother_all[m_safe]
        different = (mp != pdg) & (m >= 0) & ~done
        result = ak.where(different, mp, result)
        done = done | different | (m < 0)
        m = ak.where(done, -1, mm)
        if not ak.any(~done):
            break
    return result


def gen_objects(gp):
    """Per-event (W_R, N, l1, l2, q1, q2) from GenPart; None entries where an object is missing.

    The produced W_R is l1 + N: MadGraph writes no W_R entry when the W_R is far off-shell (most
    events of the heaviest points), so the pdg 34 entry alone would keep only the on-shell events.
    """
    last = gp[gp.hasFlags(["isLastCopy"])]
    pdg = abs(last.pdgId)
    is_n = ak.zeros_like(pdg, dtype=bool)
    for p in N_PDG:
        is_n = is_n | (pdg == p)
    n = ak.firsts(last[is_n])
    leptons = last[((pdg == 11) | (pdg == 13)) & last.hasFlags(["fromHardProcess"])]
    lparent = abs(distinct_parent_pdg(gp, leptons))
    from_n = ak.zeros_like(lparent, dtype=bool)
    for p in N_PDG:
        from_n = from_n | (lparent == p)
    l2 = ak.firsts(leptons[from_n])
    others = leptons[~from_n]
    l1 = ak.firsts(others[ak.argsort(others.pt, axis=1, ascending=False)])
    wr = l1 + n
    quarks = gp[(abs(gp.pdgId) <= 6) & gp.hasFlags(["isHardProcess"]) & (gp.status == 23)]
    quarks = quarks[ak.argsort(quarks.pt, axis=1, ascending=False)]
    q1 = ak.firsts(quarks)
    q2 = ak.firsts(quarks[:, 1:])
    return wr, n, l1, l2, q1, q2


def merged(n, l2, q1, q2, radius: float = 0.8):
    """Events whose N decay products all lie within *radius* of the N direction."""
    ok = (n.deltaR(l2) < radius) & (n.deltaR(q1) < radius) & (n.deltaR(q2) < radius)
    return ak.fill_none(ok, False)


# ---------------------------------------------------------------------------
# Histogram booking
# ---------------------------------------------------------------------------

def _h(axis) -> Hist:
    return Hist(axis, storage=hist.storage.Weight())


def book(m_wr: int) -> dict[str, Hist]:
    hi_pt = max(1.2 * m_wr, 1000.0)
    out = {}
    for obj in ("wr", "l1", "n", "l2", "q1", "q2"):
        out[f"gen_pt_{obj}"] = _h(hist.axis.Regular(100, 0, hi_pt, name="pt"))
        out[f"gen_eta_{obj}"] = _h(hist.axis.Regular(60, -6, 6, name="eta"))
    out["gen_mass_wr"] = _h(hist.axis.Regular(200, 0, 1.4 * m_wr, name="mass"))
    out["gen_mass_wr_fine"] = _h(hist.axis.Regular(400, 0.5 * m_wr, 1.25 * m_wr, name="mass"))
    out["gen_mass_n"] = _h(hist.axis.Regular(200, 0, 1.1 * m_wr, name="mass"))
    out["gen_mass_qq"] = _h(hist.axis.Regular(200, 0, 1.1 * m_wr, name="mass"))
    out["gen_mass_l2qq"] = _h(hist.axis.Regular(200, 0, 1.1 * m_wr, name="mass"))
    out["gen_dr_n_l2"] = _h(hist.axis.Regular(60, 0, 6, name="dr"))
    out["gen_dr_n_qmax"] = _h(hist.axis.Regular(60, 0, 6, name="dr"))
    # reco, after the analyzer's corrections and object selection
    for tag in ("recipe", "nano"):
        out[f"msd_lead_ak8_{tag}"] = _h(hist.axis.Regular(150, 0, 1500, name="msd"))
        out[f"msd_njet_{tag}"] = _h(hist.axis.Regular(150, 0, 1500, name="msd"))
    for tag in ("jec", "raw"):
        out[f"eresp_njet_{tag}"] = _h(hist.axis.Regular(100, 0, 2, name="ratio"))
        out[f"e_njet_{tag}"] = _h(hist.axis.Regular(100, 0, hi_pt * 2, name="energy"))
    out["e_gen_n"] = _h(hist.axis.Regular(100, 0, hi_pt * 2, name="energy"))
    out["lsf_nolsf_sr"] = _h(hist.axis.Regular(55, 0, 1.1, name="lsf"))
    out["lsf_njet"] = _h(hist.axis.Regular(55, 0, 1.1, name="lsf"))
    for lep in ("l1", "l2"):
        for ax, edges in (("pt", PT_EDGES), ("eta", ETA_EDGES)):
            out[f"eff_{lep}_{ax}_den"] = _h(hist.axis.Variable(edges, name=ax))
            out[f"eff_{lep}_{ax}_num"] = _h(hist.axis.Variable(edges, name=ax))
    for what in ("njet", "njet_lsf"):
        out[f"eff_{what}_pt_den"] = _h(hist.axis.Variable(PT_EDGES, name="pt"))
        out[f"eff_{what}_pt_num"] = _h(hist.axis.Variable(PT_EDGES, name="pt"))
    return out


def fill1(h: Hist, values, weight=None) -> None:
    v = ak.to_numpy(ak.fill_none(values, np.nan)).astype(float)
    ok = np.isfinite(v)
    if weight is None:
        h.fill(v[ok])
    else:
        w = ak.to_numpy(weight).astype(float)
        h.fill(v[ok], weight=w[ok])


# ---------------------------------------------------------------------------
# fill
# ---------------------------------------------------------------------------

def fill(era: str, point: str, flavour: str, out: Path) -> Path:
    from wrcoffea.analysis_config import (
        CUTS, SEL_ATLEAST1AK8_DPHI_GT2, SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_NO_DY_PAIR,
    )
    from wrcoffea.analyzer import WrAnalysis
    from wrcoffea.scale_factors import (
        apply_electron_scale_smearing, apply_jet_corrections, apply_muon_scale_smearing, jet_veto_event_mask,
    )

    config_path, skim_path, unskim_path = era_paths(era)
    config = json.load(open(config_path))
    dataset = dataset_name(config, point, flavour)
    skim = json.load(open(skim_path))[dataset]
    unskim = json.load(open(unskim_path))[dataset]
    m_wr = int(point.removeprefix("WR").split("_N")[0])
    hists = book(m_wr)
    counters: dict[str, float] = {}
    metadata = dict(skim["metadata"])
    metadata.setdefault("era", era)
    metadata.setdefault("physics_group", "Signal")
    metadata.setdefault("sample", dataset)
    metadata.setdefault("datatype", "mc")

    # ---- generator level, unskimmed (every produced event)
    n_gen = 0
    for url in unskim["files"]:
        ev = nanoevents(url, metadata)
        wr, n, l1, l2, q1, q2 = gen_objects(ev.GenPart)
        n_gen += len(ev)
        for name, obj in (("wr", wr), ("n", n), ("l1", l1), ("l2", l2), ("q1", q1), ("q2", q2)):
            fill1(hists[f"gen_pt_{name}"], obj.pt)
            fill1(hists[f"gen_eta_{name}"], obj.eta)
            counters[f"gen_missing_{name}"] = counters.get(f"gen_missing_{name}", 0) + int(ak.sum(ak.is_none(obj)))
        fill1(hists["gen_mass_wr"], wr.mass)
        fill1(hists["gen_mass_wr_fine"], wr.mass)
        fill1(hists["gen_mass_n"], n.mass)
        fill1(hists["gen_mass_qq"], (q1 + q2).mass)
        fill1(hists["gen_mass_l2qq"], (l2 + q1 + q2).mass)
        fill1(hists["gen_dr_n_l2"], n.deltaR(l2))
        fill1(hists["gen_dr_n_qmax"], np.maximum(ak.fill_none(n.deltaR(q1), np.nan), ak.fill_none(n.deltaR(q2), np.nan)))
        is_merged = merged(n, l2, q1, q2)
        counters["gen_merged"] = counters.get("gen_merged", 0) + int(ak.sum(is_merged))
        mass = ak.fill_none(wr.mass, np.nan)
        counters["gen_wr_within10pct"] = counters.get("gen_wr_within10pct", 0) + int(ak.sum(abs(mass - m_wr) < 0.1 * m_wr))
        counters["gen_wr_below90pct"] = counters.get("gen_wr_below90pct", 0) + int(ak.sum(mass < 0.9 * m_wr))
    counters["gen_events"] = n_gen

    # ---- reconstruction level, skimmed, through the analyzer's pipeline (wrcoffea.analyzer.process)
    ana = WrAnalysis(mass_point=dataset, region="both")
    n_reco = 0
    for url in skim["files"]:
        events = nanoevents(url, metadata)
        events = ana.apply_noise_filter(events, era)
        events["FatJet", "pt_nano"] = events.FatJet.pt
        events["FatJet", "mass_nano"] = events.FatJet.mass
        events["FatJet", "msoftdrop_nano"] = events.FatJet.msoftdrop
        events = apply_jet_corrections(events, era, True)
        ak4_id_mask = ana.jetid_mask_ak4puppi(events.Jet, era, jetid_name="AK4PUPPI_TightLeptonVeto")
        events = events[jet_veto_event_mask(events, ak4_id_mask, era)]
        corrected = apply_electron_scale_smearing(events, era, True)
        if corrected:
            events["Electron", "pt"] = corrected["pt"]
            events["Electron", "energyErr"] = corrected["energyErr"]
        corrected = apply_muon_scale_smearing(events, era, True)
        if corrected:
            events["Muon", "pt"] = corrected["pt"]
        n_reco += len(events)

        tight_leptons, _, _ = ana.select_leptons(events)
        _, ak8_baseline, _ = ana.select_jets(events, era)      # pT, eta, m_SD only (no jet ID)
        triggers = ana.build_trigger_masks(events, era)
        sel, tight_lep, ak8_cand_dy, _, _, _, _ = ana.boosted_selections(events, era, triggers=triggers)
        wr, n, l1, l2, q1, q2 = gen_objects(events.GenPart)
        is_merged = merged(n, l2, q1, q2)

        # Fig. 11: soft-drop mass of the leading AK8 jet (pT, eta, tight ID; no mass cut)
        ak8_id = ana.jetid_mask_ak4puppi(events.FatJet, era, jetid_name="AK8PUPPI_Tight")
        ak8_kin = events.FatJet[(events.FatJet.pt > CUTS["ak8_pt_min"]) & (abs(events.FatJet.eta) < CUTS["ak8_eta_max"]) & ak8_id]
        ak8_kin = ak8_kin[ak.argsort(ak8_kin.pt, axis=1, ascending=False)]
        lead = ak.firsts(ak8_kin)
        fill1(hists["msd_lead_ak8_recipe"], lead.msoftdrop)
        fill1(hists["msd_lead_ak8_nano"], lead.msoftdrop_nano)

        # N jet: the analyzer's baseline AK8 (with m_SD > 40 and tight ID) nearest a merged N
        ak8_sel = ana.selectAK8Jets(events, era)
        ak8_lsf = ana.selectAK8Jets_withLSF(events, era)
        n_dir = ak.mask(n, is_merged)
        near = ak8_sel[ak.fill_none(ak8_sel.deltaR(n_dir) < CUTS["dr_ak8_loose"], False)]
        njet = ak.firsts(near[ak.argsort(near.pt, axis=1, ascending=False)])
        has_njet = ~ak.is_none(njet)
        near_lsf = ak8_lsf[ak.fill_none(ak8_lsf.deltaR(n_dir) < CUTS["dr_ak8_loose"], False)]
        has_njet_lsf = ak.num(near_lsf) >= 1
        n_pt = ak.fill_none(n_dir.pt, np.nan)
        fill1(hists["eff_njet_pt_den"], n_pt)
        fill1(hists["eff_njet_pt_num"], ak.mask(n_pt, has_njet))
        fill1(hists["eff_njet_lsf_pt_den"], n_pt)
        fill1(hists["eff_njet_lsf_pt_num"], ak.mask(n_pt, has_njet_lsf))
        fill1(hists["lsf_njet"], njet.lsf3)
        counters["njet_matched"] = counters.get("njet_matched", 0) + int(ak.sum(has_njet))
        counters["njet_matched_lsf"] = counters.get("njet_matched_lsf", 0) + int(ak.sum(ak.fill_none(njet.lsf3 > CUTS["ak8_lsf3_min"], False)))
        fill1(hists["msd_njet_recipe"], njet.msoftdrop)
        fill1(hists["msd_njet_nano"], njet.msoftdrop_nano)
        # Fig. 10: jet energy over the N energy, corrected and raw
        raw = ak.zip({"pt": njet.pt_nano * (1 - njet.rawFactor), "eta": njet.eta, "phi": njet.phi,
                      "mass": njet.mass_nano * (1 - njet.rawFactor)}, with_name="PtEtaPhiMLorentzVector")
        e_gen = ak.fill_none(n_dir.energy, np.nan)
        fill1(hists["e_gen_n"], ak.mask(e_gen, has_njet))
        fill1(hists["e_njet_jec"], njet.energy)
        fill1(hists["e_njet_raw"], raw.energy)
        fill1(hists["eresp_njet_jec"], njet.energy / e_gen)
        fill1(hists["eresp_njet_raw"], raw.energy / e_gen)

        # Figs. 83-84: tight-lepton reconstruction efficiency vs generator pT and eta
        eta_max = CUTS["electron_eta_max"] if flavour == "El" else CUTS["muon_eta_max"]
        for name, lep in (("l1", l1), ("l2", l2)):
            in_acc = ak.mask(lep, ak.fill_none(abs(lep.eta) < eta_max, False))
            matched = ak.fill_none(ak.min(tight_leptons.deltaR(in_acc), axis=1) < 0.1, False)
            for ax in ("pt", "eta"):
                vals = ak.fill_none(getattr(in_acc, ax), np.nan)
                fill1(hists[f"eff_{name}_{ax}_den"], vals)
                fill1(hists[f"eff_{name}_{ax}_num"], ak.mask(vals, matched))

        # Fig. 12: LSF_3 of the AK8 jet after the boosted SR selection without the LSF requirement.
        # Mirrors wrcoffea.analyzer.boosted_selections with the no-LSF jet (ak8_cand_dy) in place of the LSF jet.
        loose_e = ana.selectLooseElectrons(events)
        loose_mu = ana.selectLooseMuons(events)
        loose_e = loose_e[ana.ElectronCut(loose_e.vidNestedWPBitmap, id_level=2)]
        loose_e = ak.with_field(loose_e, "electron", "flavor")
        loose_mu = ak.with_field(loose_mu, "muon", "flavor")

        def _cand(col):
            d_eta_sc = col.deltaEtaSC if hasattr(col, "deltaEtaSC") else ak.zeros_like(col.eta)
            return ak.zip({"pt": col.pt, "eta": col.eta, "phi": col.phi, "mass": col.mass, "charge": col.charge,
                           "pdgId": col.pdgId, "flavor": col.flavor, "deltaEtaSC": d_eta_sc},
                          with_name="PtEtaPhiMCandidate")
        loose = ak.concatenate([_cand(loose_e), _cand(loose_mu)], axis=1)
        loose = loose[ak.argsort(loose.pt, axis=1, ascending=False)]
        loose = ana.remove_lepton(loose, tight_lep)
        sf_loose = loose[abs(loose.pdgId) == abs(tight_lep.pdgId)]
        of_loose = loose[abs(loose.pdgId) != abs(tight_lep.pdgId)]
        sf_in = ak.firsts(sf_loose[ak.fill_none(ak8_cand_dy.deltaR(sf_loose) < CUTS["dr_ak8_loose"], False)])
        of_in = ak.firsts(of_loose[ak.fill_none(ak8_cand_dy.deltaR(of_loose) < CUTS["dr_ak8_loose"], False)])
        no_extra = ana._no_extra_tight_leptons(loose_mu, loose_e, tight_lep, sf_in)
        mll = ak.fill_none((tight_lep + sf_in).mass, -1.0)
        mlj = ak.fill_none((tight_lep + ak8_cand_dy).mass, -1.0)
        lead_pdg = ak.fill_none(abs(tight_lep.pdgId), 0)
        e_trig, mu_trig, _ = triggers
        trig = ak.where(lead_pdg == 11, e_trig, ak.where(lead_pdg == 13, mu_trig, False))
        want_pdg = 11 if flavour == "El" else 13
        mask = (sel.all(SEL_BOOSTEDTAG, SEL_LEAD_TIGHT_PT60_BOOSTED, SEL_NO_DY_PAIR, SEL_ATLEAST1AK8_DPHI_GT2)
                & (lead_pdg == want_pdg) & ak.to_numpy(trig) & ~ak.is_none(sf_in) & ak.is_none(of_in)
                & ak.to_numpy(no_extra) & (mll > CUTS["mll_sr_min"]) & (mlj > CUTS["mlljj_min"]))
        fill1(hists["lsf_nolsf_sr"], ak.mask(ak8_cand_dy.lsf3, mask))
        counters["nolsf_sr_events"] = counters.get("nolsf_sr_events", 0) + int(ak.sum(mask))
        counters["nolsf_sr_pass_lsf"] = counters.get("nolsf_sr_pass_lsf", 0) + int(
            ak.sum(ak.fill_none(ak.mask(ak8_cand_dy.lsf3, mask) > CUTS["ak8_lsf3_min"], False)))
    counters["reco_events"] = n_reco

    out = Path(out)
    (out / "hists").mkdir(parents=True, exist_ok=True)
    path = out / "hists" / f"{point}_{flavour}.pkl"
    with open(path, "wb") as f:
        pickle.dump({"era": era, "point": point, "flavour": flavour, "dataset": dataset, "m_wr": m_wr,
                     "hists": hists, "counters": counters}, f)
    logger.info("%s %s: %d generated, %d skimmed+filtered events -> %s", point, flavour, n_gen, n_reco, path)
    for k, v in sorted(counters.items()):
        logger.info("  %s = %s", k, v)
    return path


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------

GEN_LABELS = {"wr": r"$W_R$", "l1": r"$\ell_1$ (from $W_R$)", "n": r"$N$", "l2": r"$\ell_2$ (from $N$)",
              "q1": r"$q_{\mathrm{lead}}$", "q2": r"$q_{\mathrm{sublead}}$"}
# CMS colour scheme (CAT: Petroff's 10 colours); no black, which CMS keeps for data.  Overlays
# also change line style or marker, so the figures read in greyscale (PubGuidelines, figures).
COLORS = ["#3f90da", "#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70",
          "#717581", "#92dadd"]
LINESTYLES = ["-", "--", "-.", ":"]     # named styles only: mplhep splits a tuple per histogram
MARKERS = ["o", "s", "^", "v", "D", "P"]
LEGEND_FONTSIZE = 20                    # WR_Plotter FONT_SIZE_LEGEND: 5.7 pt printed at 0.45 of the AN text width


def point_label(point: str) -> str:
    wr, n = point.removeprefix("WR").split("_N")
    return rf"$m_{{W_R}}$ = {wr}, $m_N$ = {n} GeV"


def _load(out: Path) -> list[dict]:
    items = []
    for path in sorted((out / "hists").glob("*.pkl")):
        with open(path, "rb") as f:
            items.append(pickle.load(f))
    return items


def _sort_key(item: dict):
    wr, n = item["point"].removeprefix("WR").split("_N")
    return int(wr), int(n), item["flavour"]


def _figure():
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 9))
    return fig, ax


YEAR: str | None = None    # the era's plot label, set by main() from --era: shown beside the energy


def _label(ax, com: float, text: str, **kw):
    from wrplotter.plotting_helpers import cms_label, fit_header
    cms_label(ax, None, com, data=False, year=YEAR, text=text, **kw)
    fit_header(ax)


def _normalized(h: Hist):
    values = h.values(flow=True)
    total = float(values.sum())
    vals = h.values()
    errs = np.sqrt(np.clip(h.variances(), 0, None))
    scale = 1.0 / total if total > 0 else 0.0
    return h.axes[0].edges, vals * scale, errs * scale


def _auto_range(hists, *, target_bins: int = 50, coverage=(0.002, 0.995)) -> tuple[int, tuple[float, float]]:
    """(rebin factor, x range) showing the bulk of every histogram in about *target_bins* bins."""
    edges = hists[0].axes[0].edges
    lo, hi = edges[-1], edges[0]
    for h in hists:
        v = np.clip(h.values(), 0, None)
        total = v.sum()
        if total <= 0:
            continue
        cdf = np.cumsum(v) / total
        lo = min(lo, edges[max(int(np.searchsorted(cdf, coverage[0])) - 1, 0)])
        hi = max(hi, edges[min(int(np.searchsorted(cdf, coverage[1])) + 1, len(edges) - 1)])
    if hi <= lo:
        return 1, (float(edges[0]), float(edges[-1]))
    width = float(edges[1] - edges[0])
    factor = max(1, int(round((hi - lo) / width / target_bins)))
    step = factor * width
    lo = edges[0] + math.floor((lo - edges[0]) / step) * step
    hi = edges[0] + math.ceil((hi - edges[0]) / step) * step
    return factor, (float(max(lo, edges[0])), float(min(hi, edges[-1])))


def _overlay(ax, entries, key, *, rebin: int | None = None, xlim=None):
    """Unit-area overlay of one histogram of every entry; the range and binning follow the data unless given."""
    import mplhep as hep
    hists = [item["hists"][key] for _, item in entries]
    auto_factor, auto_xlim = _auto_range(hists)
    factor = rebin or auto_factor
    xlim = xlim or auto_xlim
    top = 0.0
    for i, (label, item) in enumerate(entries):
        h = item["hists"][key]
        if factor > 1:
            h = h[::hist.rebin(factor)]
        edges, vals, errs = _normalized(h)
        keep = (edges[:-1] >= xlim[0] - 1e-9) & (edges[1:] <= xlim[1] + 1e-9)
        top = max(top, float(vals[keep].max()) if keep.any() else 0.0)
        hep.histplot(vals, edges, yerr=errs, histtype="step", color=COLORS[i % len(COLORS)], linewidth=2,
                     linestyle=LINESTYLES[i % len(LINESTYLES)], label=label, ax=ax)
    ax.set_ylim(0, 1.5 * top if top > 0 else 1)
    ax.set_xlim(*xlim)
    ax.set_ylabel("Fraction of events / bin")


def _save(fig, path: Path):
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _families(items: list[dict]) -> dict[str, list[dict]]:
    """Groups of points to overlay: every m_N at each m_WR, and every m_WR at the lowest m_N."""
    fam: dict[str, list[dict]] = {}
    by_wr: dict[int, list[dict]] = {}
    for it in items:
        by_wr.setdefault(it["m_wr"], []).append(it)
    for wr, group in sorted(by_wr.items()):
        fam[f"WR{wr}"] = sorted(group, key=_sort_key)
    lowest = []
    for wr, group in sorted(by_wr.items()):
        lowest.append(min(group, key=_sort_key))
    if len(lowest) > 1:
        fam["lowest_N"] = lowest
    return fam


def _eff(num: Hist, den: Hist):
    n, d = num.values(), den.values()
    eff = np.divide(n, d, out=np.full_like(n, np.nan), where=d > 0)
    err = np.sqrt(np.clip(eff * (1 - eff), 0, None) / np.where(d > 0, d, 1))
    return num.axes[0].centers, num.axes[0].widths / 2, eff, err


def plot(out: Path, com: float = 13.6, *, plots_dir: Path | None = None, tables_dir: Path | None = None) -> None:
    """Draw every point filled under *out*: plots into *plots_dir* (default <out>/plots), summary.txt into
    *tables_dir* (default <out>)."""
    items = sorted(_load(out), key=_sort_key)
    if not items:
        raise SystemExit(f"no histograms under {out / 'hists'}")
    plots = Path(plots_dir) if plots_dir else out / "plots"
    for flavour, (_, channel) in FLAVOURS.items():
        these = [it for it in items if it["flavour"] == flavour]
        if not these:
            continue
        chan0 = {"ee": "ee", "mumu": r"$\mu\mu$"}[channel]
        for fam, group in _families(these).items():
            # One m_WR for the whole family: say it once in the description, and keep the legend to m_N,
            # short enough to stay at full size beside the CMS label.
            # The same for one m_N across the family (lowest_N).
            wrs = {it["m_wr"] for it in group}
            ns = {it["point"].split("_N")[1] for it in group}
            if len(wrs) == 1:
                chan = rf"{chan0}, $m_{{W_R}}$ = {wrs.pop()} GeV"
                entries = [(rf"$m_N$ = {it['point'].split('_N')[1]} GeV", it) for it in group]
            elif len(ns) == 1:
                chan = rf"{chan0}, $m_N$ = {ns.pop()} GeV"
                entries = [(rf"$m_{{W_R}}$ = {it['m_wr']} GeV", it) for it in group]
            else:
                chan = chan0
                entries = [(point_label(it["point"]), it) for it in group]
            # AN Figs. 3-5: generator-level kinematics
            for obj in GEN_LABELS:
                for ax_name, xlabel in (("pt", rf"$p_T$ of {GEN_LABELS[obj]} [GeV]"), ("eta", rf"$\eta$ of {GEN_LABELS[obj]}")):
                    fig, ax = _figure()
                    _overlay(ax, entries, f"gen_{ax_name}_{obj}")
                    ax.set_xlabel(xlabel)
                    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")
                    _label(ax, com, f"{chan}\ngenerator level\nall generated events")
                    _save(fig, plots / channel / fam / "gen" / f"{ax_name}_{obj}.pdf")
            for key, xlabel in (("gen_mass_wr", r"$m(W_R)$ [GeV]"), ("gen_mass_n", r"$m(N)$ [GeV]"),
                                ("gen_mass_qq", r"$m(qq)$ [GeV]"), ("gen_mass_l2qq", r"$m(\ell_2 qq)$ [GeV]"),
                                ("gen_dr_n_l2", r"$\Delta R(N, \ell_2)$"), ("gen_dr_n_qmax", r"max $\Delta R(N, q)$")):
                fig, ax = _figure()
                _overlay(ax, entries, key)
                ax.set_xlabel(xlabel)
                ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")
                _label(ax, com, f"{chan}\ngenerator level\nall generated events")
                _save(fig, plots / channel / fam / "gen" / f"{key.removeprefix('gen_')}.pdf")
            # AN Fig. 11: soft-drop mass of the leading AK8 jet
            for tag, what in (("recipe", "JME recipe (analysis)"), ("nano", "NanoAOD")):
                fig, ax = _figure()
                _overlay(ax, entries, f"msd_lead_ak8_{tag}")
                ax.set_xlabel(r"$m_{SD}$ of the leading AK8 jet [GeV]")
                ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")
                _label(ax, com, f"{chan}\nleading AK8 jet\n$p_T$ > 200 GeV, $|\\eta|$ < 2.4, tight ID\n{what}")
                _save(fig, plots / channel / fam / f"msd_lead_ak8_{tag}.pdf")
            fig, ax = _figure()
            _overlay(ax, entries, "msd_njet_recipe")
            ax.set_xlabel(r"$m_{SD}$ of the $N$ jet [GeV]")
            ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")
            _label(ax, com, f"{chan}\nAK8 jet matched to a merged $N$\nJME recipe (analysis)")
            _save(fig, plots / channel / fam / "msd_njet_recipe.pdf")
            # AN Fig. 12: LSF_3 without the LSF requirement, and of the N jet
            fig, ax = _figure()
            _overlay(ax, entries, "lsf_nolsf_sr", rebin=1, xlim=(0, 1.1))
            ax.axvline(0.75, color="grey", ls="--")
            ax.set_xlabel(r"$\mathrm{LSF}_3$ of the AK8 jet")
            ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper left")
            _label(ax, com, f"{chan}\nboosted SR selection\nwithout the LSF requirement")
            _save(fig, plots / channel / fam / "lsf3_boosted_sr_nolsf.pdf")
            fig, ax = _figure()
            _overlay(ax, entries, "lsf_njet", rebin=1, xlim=(0, 1.1))
            ax.axvline(0.75, color="grey", ls="--")
            ax.set_xlabel(r"$\mathrm{LSF}_3$ of the $N$ jet")
            ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper left")
            _label(ax, com, f"{chan}\nAK8 jet matched to a merged $N$")
            _save(fig, plots / channel / fam / "lsf3_njet.pdf")
            # AN Fig. 10: jet energy response
            for tag, what in (("jec", "with JEC (analysis)"), ("raw", "without JEC")):
                fig, ax = _figure()
                _overlay(ax, entries, f"eresp_njet_{tag}", rebin=2, xlim=(0, 2))
                ax.set_xlabel(r"$E(\mathrm{AK8\ jet}) / E(N)$")
                ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")
                _label(ax, com, f"{chan}\nAK8 jet matched to a merged $N$\n{what}")
                _save(fig, plots / channel / fam / f"energy_response_njet_{tag}.pdf")
            # Figs. 83-86 / Table 42: efficiencies
            for lep in ("l1", "l2"):
                for ax_name, xlabel in (("pt", rf"generator $p_T$ of {GEN_LABELS[lep]} [GeV]"), ("eta", rf"generator $\eta$ of {GEN_LABELS[lep]}")):
                    fig, ax = _figure()
                    for i, (label, it) in enumerate(entries):
                        x, xe, eff, err = _eff(it["hists"][f"eff_{lep}_{ax_name}_num"], it["hists"][f"eff_{lep}_{ax_name}_den"])
                        ax.errorbar(x, eff, xerr=xe, yerr=err, fmt=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)], label=label, markersize=6)
                    ax.set_ylim(0, 1.45)
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel("Tight-lepton efficiency")
                    if ax_name == "pt":
                        ax.set_xscale("log")
                    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")
                    _label(ax, com, f"{chan}\ntight lepton within $\\Delta R$ < 0.1\nof the generator lepton")
                    _save(fig, plots / channel / fam / "efficiency" / f"tight_{lep}_vs_{ax_name}.pdf")
            for what, ylabel, text in (("njet", "AK8 jet efficiency", "baseline AK8 jet\n$p_T$ > 200 GeV, $m_{SD}$ > 40 GeV, tight ID"),
                                       ("njet_lsf", "AK8 jet efficiency", r"AK8 jet with $\mathrm{LSF}_3$ > 0.75")):
                fig, ax = _figure()
                for i, (label, it) in enumerate(entries):
                    x, xe, eff, err = _eff(it["hists"][f"eff_{what}_pt_num"], it["hists"][f"eff_{what}_pt_den"])
                    ax.errorbar(x, eff, xerr=xe, yerr=err, fmt=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)], label=label, markersize=6)
                ax.set_ylim(0, 1.45)
                ax.set_xscale("log")
                ax.set_xlabel(r"generator $p_T$ of the merged $N$ [GeV]")
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")
                _label(ax, com, f"{chan}\n{text}\nwithin $\\Delta R$ < 0.8 of a merged $N$")
                _save(fig, plots / channel / fam / "efficiency" / f"{what}_vs_pt.pdf")
        # AN Fig. 130: W_R mass with the on-shell fraction, one plot per point
        chan = chan0
        for it in these:
            fig, ax = _figure()
            _overlay(ax, [(point_label(it["point"]), it)], "gen_mass_wr", rebin=2, xlim=(0, 1.4 * it["m_wr"]))
            c = it["counters"]
            n = max(c.get("gen_events", 0), 1)
            txt = (f"{100 * c.get('gen_wr_within10pct', 0) / n:.1f} % within 10 % of $m_{{W_R}}$\n"
                   f"{100 * c.get('gen_wr_below90pct', 0) / n:.1f} % below 0.9 $m_{{W_R}}$")
            ax.set_xlabel(r"generator $m(W_R)$ [GeV]")
            ax.set_yscale("log")
            vals = it["hists"]["gen_mass_wr"].values()
            ax.set_ylim(max(vals.min() / n, 1e-5) if vals.min() > 0 else 1e-5, 5)
            ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper left")
            _label(ax, com, f"{chan}\ngenerator level\n{txt}")
            _save(fig, plots / channel / "wr_mass" / f"{it['point']}.pdf")
    # Table 42 analogue and the W_R on-shell fractions
    lines = ["point flavour gen_events merged_frac njet_matched lsf_pass_frac nolsf_sr_events nolsf_sr_lsf_pass_frac wr_within10pct wr_below90pct"]
    for it in items:
        c = it["counters"]
        n = max(c.get("gen_events", 0), 1)
        nj = c.get("njet_matched", 0)
        ns = c.get("nolsf_sr_events", 0)
        lines.append(f"{it['point']} {it['flavour']} {c.get('gen_events', 0)} {c.get('gen_merged', 0) / n:.3f} {nj} "
                     f"{c.get('njet_matched_lsf', 0) / max(nj, 1):.3f} {ns} {c.get('nolsf_sr_pass_lsf', 0) / max(ns, 1):.3f} "
                     f"{c.get('gen_wr_within10pct', 0) / n:.3f} {c.get('gen_wr_below90pct', 0) / n:.3f}")
    summary = (Path(tables_dir) if tables_dir else out) / "summary.txt"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines) + "\n")
    logger.info("plots under %s, summary in %s", plots, summary)


# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fill", help="histogram one mass point and flavour")
    f.add_argument("--era", required=True)
    f.add_argument("--point", required=True, help="WR<m>_N<m>")
    f.add_argument("--flavour", required=True, choices=sorted(FLAVOURS))
    f.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("plot", help="draw every filled point")
    p.add_argument("--out", required=True, type=Path, help="the fill step's --out (the histograms)")
    p.add_argument("--era", default=None, help="with --dir: write into that production's signal_outputs/event_level")
    p.add_argument("--dir", default=None, help="production (--dir of run_analysis.py) whose signal outputs get the plots")
    p.add_argument("--local-plots", action="store_true", help="plots/ instead of EOS")
    p.add_argument("--com", type=float, default=None, help="centre-of-mass energy label (default: the era's)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.cmd == "fill":
        fill(args.era, args.point, args.flavour, args.out)
    else:
        import matplotlib
        matplotlib.use("Agg")
        plots_dir = tables_dir = None
        com = args.com or 13.6
        if args.era:
            from wrplotter.config import era_label
            global YEAR
            YEAR = era_label(args.era)
        if args.dir:
            if not args.era:
                parser.error("--dir needs --era")
            from wrplotter.config import load_lumi
            from wrplotter.signal_plots import signal_output_dirs
            plots_dir, tables_dir = signal_output_dirs(args.era, args.dir, "event_level", local_plots=args.local_plots)
            com = args.com or float(load_lumi(args.era).get("com", 13.6))
        plot(args.out, com, plots_dir=plots_dir, tables_dir=tables_dir)


if __name__ == "__main__":
    main()
