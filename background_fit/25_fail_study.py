#!/usr/bin/env python3
"""Why S+B fits fail: the background component or the signal piece.

Per channel and mass, toys of the raw MC over the full range (N_sig free). Every toy is fitted three ways:
  B     the background function alone (no signal term);
  S+B   the standard fit of the study (background seeded from the fit to the expectation, N_sig from 0);
  and, when S+B does not pass, again from other starting points (the toy's own B fit; N_sig seeded at -1 and +1
  sigma_fit) and with the signal fixed at a few values (the likelihood against N_sig).
Recorded: which criterion fails (valid minimum, covariance complete, falling background, pull within +-5), the
fitter's covariance status, the events under the peak, whether B alone passes on the same toy, whether another
start rescues the fit, and where the likelihood against N_sig has its minimum.

  cd WR_Plotter && FIT_TOPOLOGY=boosted FIT_DATASET=run3 FIT_SIGNAL_MAP=background_fit/signal_map_2024.json \
      python background_fit/25_fail_study.py [--ntoys 6000]
Writes results*/FAILS.md, fails.json, fails_plot_m<scan mass>.json (what the figures of the scan mass draw;
--figures-only redraws from it), fails_scan_m<scan mass>.json (the two example toys) and the figures
results*/<ch>/25_fails/.
--scan-only: the study of the scan-mass cell alone (the hole and likelihood figures of that mass and their two files);
FAILS.md and fails.json stay as they are.
--table: only FAIL_TABLE.md / fail_table.json, the share of fits per outcome and mass (no retries, no scans).
--examples: one toy per outcome that is not a pass (the first drawn of each) at --scan-mass, with its S+B fit
(fail_examples_m<mass>.json and the figures <ch>/25_fails/toy_<outcome>_m<mass>).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lib"))
import config as C                                                       # noqa: E402

PULL_RANGE = 5.0


def run_cell(job):
    import ROOT
    from fits import fit_poisson, th1
    from sbfit import SBModel, bkg_only, signal_pars, signal_shape
    from spectrum import centers, rebin
    from truth_sets import load_truth
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kFatal
    ch, m, cand, lo, hi, ntoys, seed = (job[k] for k in ("ch", "m", "cand", "lo", "hi", "ntoys", "seed"))
    e10, parts, _ = load_truth(C.RESULTS / ch / "3_truths_split9" / "lo_ht" / "truth_raw")
    e, t, var = rebin(e10, parts["sum"], np.maximum(parts["sum"], 0.0))
    c = centers(e)
    mu, sg = signal_pars(ch, m)
    shape = signal_shape(e, mu, sg)
    r1 = (c >= mu - sg) & (c <= mu + sg)
    r2 = (c >= mu - 2 * sg) & (c <= mu + 2 * sg)
    infit = (c >= lo) & (c <= hi)
    fb0 = bkg_only(cand, e, t, var, lo, hi, {})
    model = SBModel(cand, mu, sg, lo, hi)
    asim = model.fit(e, t, fb0["params"], 0.0, None)
    f, npar = model.f, model.npar
    sfit = asim["nsig_err"]

    def sb(toy, bkg_seed, nsig_seed, fix=None):
        """One S+B fit; fix = a value to hold N_sig at (the likelihood against N_sig)."""
        for i, sv in enumerate(list(bkg_seed)[:npar]):
            f.SetParameter(i, float(sv))
        if fix is None:
            f.SetParameter(npar, float(nsig_seed))
            f.SetParLimits(npar, 0.0, 0.0)
        else:
            f.FixParameter(npar, float(fix))
        h = th1(e, np.where(toy > 0, toy, 0.0), np.sqrt(np.maximum(toy, 0.0)))
        r = h.Fit(f, "LRNSQ").Get()
        if fix is not None:
            f.ReleaseParameter(npar)
        if not r or not math.isfinite(r.MinFcnValue()):
            return None
        par = np.array([r.Parameter(i) for i in range(npar + 1)])
        for i in range(npar):
            model.fb.SetParameter(i, float(par[i]))
        fg = np.array([model.fb.Eval(x) for x in model.grid])
        bk = np.array([model.fb.Eval(x) for x in c])
        tot = bk + par[npar] * shape
        err = float(r.ParError(npar))
        return {"valid": bool(r.IsValid()), "cov": int(r.CovMatrixStatus()), "status": int(r.Status()), "edm": float(r.Edm()),
                "nll": float(r.MinFcnValue()), "nsig": float(par[npar]), "err": err, "bkg": par[:npar],
                "mono": bool(np.all(fg[1:] <= fg[:-1] * (1.0 + 1e-6))),
                "min_tot": float(tot[infit & (shape > 1e-4)].min()) if (infit & (shape > 1e-4)).any() else float("nan")}

    def category(r):
        if r is None:
            return "no result"
        if not r["valid"]:
            return "minimum not valid"
        if r["cov"] != 3:
            return "covariance incomplete"
        if not r["mono"]:
            return "background not falling"
        if not (r["err"] > 0) or abs(r["nsig"] / r["err"]) > PULL_RANGE:
            return "pull beyond 5"
        return "pass"

    rng = np.random.default_rng(seed)
    out = []
    ex_pass, ex_fail = None, None                 # (toy, background seed): the example toys of the likelihood scan
    examples = {}                                 # job["examples"]: the first toy of each outcome that is not a pass
    for itoy in range(ntoys):
        toy = rng.poisson(np.maximum(t, 0.0)).astype(float)
        n1, n2 = float(toy[r1].sum()), float(toy[r2].sum())
        vt = np.maximum(toy, 0.0)
        if job.get("skip_b"):                          # the S+B fit only (the N_sig of every fit by its outcome)
            b, b_ok = None, None
        else:
            b = fit_poisson(cand, e, toy, np.where(vt > 0, vt, 1.0), lo, hi, C.PIVOT, [list(asim["bkg"])])
            b_ok = bool(b is not None and b["passed"])
        r = sb(toy, asim["bkg"], 0.0)
        cat = category(r)
        rec = {"n1": n1, "n2": n2, "ntot": float(toy[infit].sum()), "b_ok": b_ok, "cat": cat,
               "nsig": None if r is None else r["nsig"], "err": None if r is None else r["err"],
               "cov": None if r is None else r["cov"], "min_tot": None if r is None else r["min_tot"]}
        if job.get("scan") and cat == "pass" and ex_pass is None and r["err"] > 0 and abs(r["nsig"] / r["err"]) <= 1.0:
            ex_pass = (toy.copy(), list(b["params"]) if b is not None else list(asim["bkg"]), r["nsig"], r["err"])
        if r is not None:
            rec.update({"valid": r["valid"], "mono": r["mono"], "status": r["status"], "edm": r["edm"]})
        if job.get("examples") and r is not None and cat in job["examples"] and cat not in examples:
            for i, pv in enumerate(r["bkg"]):
                model.fb.SetParameter(i, float(pv))
            bk = np.array([model.fb.Eval(x) for x in c])
            examples[cat] = {"toy": toy.tolist(), "total": (bk + r["nsig"] * shape).tolist(), "nsig": r["nsig"], "err": r["err"],
                             "valid": r["valid"], "cov": r["cov"], "status": r["status"], "edm": r["edm"], "n1": n1, "n2": n2,
                             "index": itoy + 1}
            if len(examples) == len(job["examples"]):
                out.append(rec)
                break
        if cat != "pass" and job.get("alts"):
            # the same toy from other starting points, the outcome of each: is the failure type a property of the toy?
            starts = [("own", list(r["bkg"]), r["nsig"])] if r is not None else []          # continue from where it stopped
            if b is not None:
                starts.append(("toy_b", list(b["params"]), 0.0))
            starts += [("minus", list(asim["bkg"]), -sfit), ("plus", list(asim["bkg"]), +sfit)]
            rec["alt"] = []
            for name, bs, ns in starts:
                rr = sb(toy, bs, ns)
                rec["alt"].append([name, category(rr), None if rr is None else rr["nsig"], None if rr is None else rr["err"]])
        if cat != "pass" and not job.get("categories_only"):
            # other starting points
            starts = []
            if b is not None:
                starts.append(("toy_b", list(b["params"]), 0.0))
            starts += [("minus", list(asim["bkg"]), -sfit), ("plus", list(asim["bkg"]), +sfit)]
            rescued = None
            for name, bs, ns in starts:
                rr = sb(toy, bs, ns)
                if category(rr) == "pass":
                    rescued = {"by": name, "nsig": rr["nsig"], "err": rr["err"]}
                    break
            rec["rescued"] = rescued
            # the likelihood against N_sig: the background refitted with N_sig held at each value
            bs = list(b["params"]) if b is not None else list(asim["bkg"])
            for i, pv in enumerate(bs):
                model.fb.SetParameter(i, float(pv))
            bk = np.array([model.fb.Eval(x) for x in c])
            sel = infit & (shape > 1e-4)
            edge = -float(np.min(bk[sel] / shape[sel]))                   # where S+B first reaches zero
            grid = [1.5 * edge, 1.2 * edge, 1.05 * edge, 1.0 * edge, 0.95 * edge, 0.8 * edge, 0.5 * edge, 0.0, sfit, 2 * sfit]
            prof = []
            for g in grid:
                rr = sb(toy, bs, g, fix=g)
                prof.append(None if rr is None else rr["nll"])
            ok = [(g, p) for g, p in zip(grid, prof) if p is not None]
            if ok:
                gmin = min(ok, key=lambda x: x[1])[0]
                rec["edge"] = edge
                rec["prof_min"] = gmin / edge if edge != 0 else None          # 1 = at the zero edge, > 1 beyond it, 0 = at N_sig 0
                rec["prof_drop"] = (dict(ok).get(0.0, float("nan")) - min(p for _, p in ok))
                if job.get("scan") and ex_fail is None and cat == "pull beyond 5" and n1 == 0:
                    ex_fail = (toy.copy(), bs, edge, r["nsig"], r["err"])
        out.append(rec)
    scan = None
    if job.get("scan") and ex_pass is not None and ex_fail is not None:
        # the likelihood against N_sig for the two example toys: N_sig held at each value, the background refitted
        edge = ex_fail[2]
        grid = sorted(set(np.linspace(2.0 * edge, 3.0 * sfit, 46).tolist() + [0.0]))
        def curve(toy, bs):
            return [(lambda rr: None if rr is None else rr["nll"])(sb(toy, bs, g, fix=g)) for g in grid]
        scan = {"grid": grid, "pass": curve(ex_pass[0], ex_pass[1]), "fail": curve(ex_fail[0], ex_fail[1]), "edge": edge,
                "pass_fit": [ex_pass[2], ex_pass[3]], "fail_fit": [ex_fail[3], ex_fail[4]], "sfit": sfit,
                "pass_n1": float(ex_pass[0][r1].sum()), "fail_n1": float(ex_fail[0][r1].sum())}
    return {"ch": ch, "m": m, "mu": mu, "sigma": sg, "b_core": float(t[r2].sum()), "b_1s": float(t[r1].sum()), "sfit": sfit,
            "toys": out, "scan": scan, "examples": examples, "lo": lo, "hi": hi}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masses", nargs="+", type=float, default=[1800.0, 2000.0, 2200.0, 2400.0, 2600.0, 2800.0, 3000.0, 3200.0])
    ap.add_argument("--scan-mass", type=float, default=2200.0, help="the mass of the likelihood scan and of the hole figure")
    ap.add_argument("--channels", nargs="+", default=C.CHANNELS)
    ap.add_argument("--cand", default="atlasemu")
    ap.add_argument("--lo", type=float, default=800.0)
    ap.add_argument("--ntoys", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--figures-only", action="store_true", help="redraw the figures from the stored fails.json and fails_plot.json")
    ap.add_argument("--table", action="store_true",
                    help="only the table of outcomes per mass (FAIL_TABLE.md, fail_table.json): every fit under the first "
                         "criterion it fails; no retries, no likelihood scans, the other outputs untouched")
    ap.add_argument("--overlap", action="store_true",
                    help="do the failure types share one cause (FAIL_OVERLAP.md, fail_overlap.json): every criterion on its "
                         "own per fit, what the failing fits have in common, and the same toys refitted from other starts")
    ap.add_argument("--nsig", action="store_true",
                    help="the fitted N_sig of every fit at --scan-mass, by outcome (fail_nsig.npz and the figure "
                         "<ch>/25_fails/nsig_outcomes_m<mass>): --ntoys toys per channel, the S+B fit only")
    ap.add_argument("--scan-only", action="store_true",
                    help="only the cell of --scan-mass: fails_plot_m<mass>.json, fails_scan_m<mass>.json and the figures "
                         "holes_m<mass>, scan_m<mass>; FAILS.md and fails.json untouched")
    ap.add_argument("--examples", action="store_true",
                    help="one toy per outcome that is not a pass at --scan-mass, the first drawn of each among --ntoys, "
                         "with its S+B fit (fail_examples_m<mass>.json, <ch>/25_fails/toy_<outcome>_m<mass>)")
    args = ap.parse_args()
    if args.examples:
        return examples(args)
    if args.table:
        return table(args)
    if args.overlap:
        return overlap(args)
    if args.nsig:
        return nsig_by_outcome(args)
    plot_file = C.RESULTS / f"fails_plot_m{args.scan_mass:.0f}.json"
    if args.figures_only:
        figures(args, json.loads(plot_file.read_text()), json.loads((C.RESULTS / "fails.json").read_text()))
        report_figures(args)
        example_figures(args)
        return
    if args.scan_only:
        args.masses = [args.scan_mass]
    jobs = [{"ch": ch, "m": m, "cand": args.cand, "lo": args.lo, "hi": 8000.0, "ntoys": args.ntoys, "seed": 7000 + int(m) + (0 if ch == "ee" else 1),
             "scan": m == args.scan_mass}
            for ch in args.channels for m in args.masses]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        cells = pool.map(run_cell, jobs)
    CATS = ["pass", "pull beyond 5", "covariance incomplete", "minimum not valid", "background not falling", "no result"]
    L = [f"# Why S+B fits fail: {C.ERA} {C.REGION_NAME}, raw MC toys, {args.cand}, full range from {args.lo:.0f} GeV, N_sig free", "",
         f"{args.ntoys} toys per cell. B = the background function alone on the same toy.", ""]
    summary = []
    for cell in ([] if args.scan_only else cells):
        toys = cell["toys"]
        n = len(toys)
        fails = [x for x in toys if x["cat"] != "pass"]
        row = {"ch": cell["ch"], "m": cell["m"], "b_core": cell["b_core"], "b_1s": cell["b_1s"], "n": n,
               "cats": {c_: sum(x["cat"] == c_ for x in toys) / n for c_ in CATS},
               "b_ok_all": sum(x["b_ok"] for x in toys) / n,
               "b_ok_fail": (sum(x["b_ok"] for x in fails) / len(fails)) if fails else None,
               "n1": {c_: float(np.mean([x["n1"] for x in toys if x["cat"] == c_])) for c_ in CATS if any(x["cat"] == c_ for x in toys)},
               "n2": {c_: float(np.mean([x["n2"] for x in toys if x["cat"] == c_])) for c_ in CATS if any(x["cat"] == c_ for x in toys)},
               "empty1": {c_: float(np.mean([x["n1"] == 0 for x in toys if x["cat"] == c_])) for c_ in CATS if any(x["cat"] == c_ for x in toys)},
               "rescued": {c_: float(np.mean([x.get("rescued") is not None for x in toys if x["cat"] == c_]))
                           for c_ in CATS[1:] if any(x["cat"] == c_ for x in toys)},
               "prof_at_edge": {c_: float(np.mean([x.get("prof_min") is not None and x["prof_min"] >= 0.95 for x in toys if x["cat"] == c_]))
                                for c_ in CATS[1:] if any(x["cat"] == c_ for x in toys)},
               "prof_drop": {c_: float(np.median([x["prof_drop"] for x in toys if x["cat"] == c_ and x.get("prof_drop") is not None]))
                             for c_ in CATS[1:] if any(x["cat"] == c_ and x.get("prof_drop") is not None for x in toys)},
               "neg": {c_: float(np.mean([x["nsig"] is not None and x["nsig"] < 0 for x in toys if x["cat"] == c_]))
                       for c_ in CATS[1:] if any(x["cat"] == c_ for x in toys)},
               "cov_codes": {str(k): sum(x["cov"] == k for x in fails) for k in (0, 1, 2, 3)},
               # where the likelihood against N_sig is lowest, in units of the zero edge (1 = S+B just reaches zero;
               # > 1 beyond it; 0 = at N_sig = 0), over the failing fits
               "prof_min_dist": {f"{v:g}": sum(x.get("prof_min") is not None and abs(x["prof_min"] - v) < 1e-6 for x in fails) / max(len(fails), 1)
                                 for v in (1.5, 1.2, 1.05, 1.0, 0.95, 0.8, 0.5, 0.0)},
               "edge_events": float(np.median([x["edge"] for x in fails if x.get("edge") is not None])) if fails else None,
               "fail_nsig_over_edge": float(np.median([x["nsig"] / x["edge"] for x in fails if x.get("edge") and x["nsig"] is not None])) if fails else None,
               # toys with nothing within +-1 sigma of the peak: how they end
               "by_n1": {str(k): (sum(x["cat"] != "pass" and x["n1"] == k for x in toys) / max(sum(x["n1"] == k for x in toys), 1))
                         for k in range(0, 6) if sum(x["n1"] == k for x in toys) >= 30},
               "empty_share": sum(x["n1"] == 0 for x in toys) / n,
               "empty_outcome": {c_: sum(x["cat"] == c_ and x["n1"] == 0 for x in toys) / max(sum(x["n1"] == 0 for x in toys), 1) for c_ in CATS}}
        summary.append(row)
        L += [f"## {cell['ch']} {cell['m']:.0f} GeV: expected background {cell['b_core']:.1f} within +-2 sigma, {cell['b_1s']:.1f} within +-1 sigma", "",
              f"B alone passes on {100 * row['b_ok_all']:.1f} % of all toys" + (f", on {100 * row['b_ok_fail']:.1f} % of the toys whose S+B fit fails." if fails else "."), "",
              "| S+B outcome | share of toys | events within +-1 sigma (mean) | none within +-1 sigma | N_sig < 0 | passes from another start | likelihood lowest at or beyond the zero edge | likelihood drop from N_sig = 0 |",
              "|---|---|---|---|---|---|---|---|"]
        for c_ in CATS:
            if row["cats"][c_] == 0:
                continue
            L.append(f"| {c_} | {100 * row['cats'][c_]:.2f} % | {row['n1'][c_]:.2f} | {100 * row['empty1'][c_]:.0f} % | "
                     + ("-" if c_ == "pass" else f"{100 * row['neg'][c_]:.0f} %") + " | "
                     + ("-" if c_ == "pass" else f"{100 * row['rescued'][c_]:.0f} %") + " | "
                     + ("-" if c_ == "pass" else f"{100 * row['prof_at_edge'][c_]:.0f} %") + " | "
                     + ("-" if c_ == "pass" or c_ not in row["prof_drop"] else f"{row['prof_drop'][c_]:.2f}") + " |")
        if fails:
            L += ["", f"Failing fits: the zero edge sits at N_sig = {row['edge_events']:.1f} (median); their fitted N_sig is {row['fail_nsig_over_edge']:.2f} of it (median). "
                  "Where the likelihood against N_sig is lowest, in units of the edge: "
                  + ", ".join(f"{k}: {100 * v:.0f} %" for k, v in row["prof_min_dist"].items() if v > 0) + ".",
                  f"Toys with nothing within +-1 sigma ({100 * row['empty_share']:.1f} % of toys) end as: "
                  + ", ".join(f"{c_} {100 * v:.0f} %" for c_, v in row["empty_outcome"].items() if v > 0) + "."]
        L.append("")
    if args.scan_only:
        summary = json.loads((C.RESULTS / "fails.json").read_text())
    else:
        (C.RESULTS / "FAILS.md").write_text("\n".join(L) + "\n")
        (C.RESULTS / "fails.json").write_text(json.dumps(summary, indent=1))
        print("\n".join(L))
    # what the figures of the scan mass draw: the failing share per number of events under the peak, and the scan
    record = {}
    for cell in cells:
        if cell["m"] != args.scan_mass:
            continue
        holes = []
        for k in range(0, 8):
            sel = [x for x in cell["toys"] if x["n1"] == k]
            if len(sel) >= 30:
                holes.append([k, sum(x["cat"] != "pass" for x in sel), len(sel)])
        record[cell["ch"]] = {"holes": holes, "scan": cell.get("scan"), "n": len(cell["toys"]), "b_1s": cell["b_1s"],
                              "empty_share": sum(x["n1"] == 0 for x in cell["toys"]) / len(cell["toys"])}
        print(f"{cell['ch']} {cell['m']:.0f} GeV: {len(cell['toys'])} toys, {cell['b_1s']:.2f} background events within +-1 sigma; "
              f"toy events within +-1 sigma -> [k, fits not passing, toys]: {holes}")
    plot_file.write_text(json.dumps(record))
    figures(args, record, summary)


TABLE_COLS = (("pass", "pass"), ("minimum not valid", "no valid minimum"), ("covariance incomplete", "covariance incomplete"),
              ("background not falling", "background not falling"), ("pull beyond 5", "pull beyond +-5"))


def table(args):
    """The outcome of the S+B fit per channel and mass, every fit under the first criterion it fails, in the order of
    the criteria: valid minimum, complete covariance, falling background, |pull| <= 5."""
    jobs = [{"ch": ch, "m": m, "cand": args.cand, "lo": args.lo, "hi": 8000.0, "ntoys": args.ntoys, "seed": 7000 + int(m) + (0 if ch == "ee" else 1),
             "scan": False, "categories_only": True}
            for ch in args.channels for m in args.masses]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        cells = pool.map(run_cell, jobs)
    rows = []
    L = [f"# The outcome of the S+B fit: {C.ERA} {C.REGION_NAME}, raw MC toys, {args.cand}, full range from {args.lo:.0f} GeV, N_sig free", "",
         f"{args.ntoys} toys per mass, no signal injected. Every fit is counted under the first criterion it fails, in this order: "
         "valid minimum, complete covariance, falling background, |pull| <= 5. Shares of all toys, in %.", ""]
    for ch in args.channels:
        L += [f"## {ch}", "", "| m_WR [GeV] | background within +-2 sigma | " + " | ".join(lab for _, lab in TABLE_COLS) + " | background alone passes |",
              "|---|---|" + "---|" * (len(TABLE_COLS) + 1)]
        for cell in sorted((c_ for c_ in cells if c_["ch"] == ch), key=lambda c_: c_["m"]):
            toys = cell["toys"]
            n = len(toys)
            share = {key: sum(x["cat"] == key for x in toys) / n for key, _ in TABLE_COLS}
            share["no result"] = sum(x["cat"] == "no result" for x in toys) / n
            fail5 = [x for x in toys if x["cat"] == "pull beyond 5"]
            row = {"ch": ch, "m": cell["m"], "n": n, "b_core": cell["b_core"], "b_1s": cell["b_1s"], "share": share,
                   "b_ok": sum(x["b_ok"] for x in toys) / n,
                   "pull_below": (sum(x["nsig"] < 0 for x in fail5) / len(fail5)) if fail5 else None,
                   "neg": {key: (sum(x["nsig"] is not None and x["nsig"] < 0 for x in toys if x["cat"] == key) / max(sum(x["cat"] == key for x in toys), 1))
                           for key, _ in TABLE_COLS[1:]},
                   "cov_codes": {str(k): sum(x["cov"] == k for x in toys if x["cat"] == "covariance incomplete") / n for k in (0, 1, 2)}}
            rows.append(row)
            L.append(f"| {cell['m']:.0f} | {cell['b_core']:.1f} | " + " | ".join(f"{100 * share[key]:.2f}" for key, _ in TABLE_COLS)
                     + f" | {100 * row['b_ok']:.2f} |")
        L.append("")
    (C.RESULTS / "FAIL_TABLE.md").write_text("\n".join(L) + "\n")
    (C.RESULTS / "fail_table.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(L))
    report_figures(args)


OUTCOMES = ("pass", "minimum not valid", "covariance incomplete", "pull beyond 5")       # the order of the criteria
OUTCOME_LABEL = {"pass": "pass", "minimum not valid": "no valid minimum", "covariance incomplete": "covariance incomplete",
                 "pull beyond 5": r"pull beyond $\pm 5$"}
OUTCOME_STYLE = {"pass": ("k", "o"), "minimum not valid": ("#5790fc", "s"), "covariance incomplete": ("#f89c20", "^"),
                 "pull beyond 5": ("#e42536", "D")}


def nsig_by_outcome(args):
    """The fitted N_sig of every S+B fit at the scan mass, by outcome (every fit under the first criterion it does not
    meet). The toys are split into chunks with their own seeds; fail_nsig.npz holds, per channel, N_sig, its error and
    the outcome index of every toy."""
    m = args.scan_mass
    nchunk = max(args.workers, 1)
    per = int(math.ceil(args.ntoys / nchunk))
    jobs = [{"ch": ch, "m": m, "cand": args.cand, "lo": args.lo, "hi": 8000.0, "ntoys": per,
             "seed": 7000 + int(m) + (0 if ch == "ee" else 1) + 100000 * (k + 1), "scan": False, "categories_only": True, "skip_b": True}
            for ch in args.channels for k in range(nchunk)]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        cells = pool.map(run_cell, jobs)
    store = {}
    for ch in args.channels:
        toys = [x for c_ in cells if c_["ch"] == ch for x in c_["toys"] if x["nsig"] is not None and x["cat"] in OUTCOMES]
        store[f"{ch}_nsig"] = np.array([x["nsig"] for x in toys])
        store[f"{ch}_err"] = np.array([x["err"] for x in toys])
        store[f"{ch}_cat"] = np.array([OUTCOMES.index(x["cat"]) for x in toys])
        store[f"{ch}_n1"] = np.array([x["n1"] for x in toys])
    np.savez_compressed(C.RESULTS / f"fail_nsig_m{m:.0f}.npz", **store)
    for ch in args.channels:
        n, cat = store[f"{ch}_nsig"], store[f"{ch}_cat"]
        print(f"{ch} {m:.0f} GeV, {len(n)} fits")
        for i, key in enumerate(OUTCOMES):
            v = n[cat == i]
            if len(v):
                q = np.percentile(v, [0, 1, 16, 50, 84, 99, 100])
                print(f"  {key:22s} {len(v):6d} ({100 * len(v) / len(n):5.2f} %)  N_sig < 0: {100 * (v < 0).mean():6.2f} %   min {q[0]:.1f}  1% {q[1]:.1f}  16% {q[2]:.2f}  "
                      f"median {q[3]:.2f}  84% {q[4]:.2f}  99% {q[5]:.1f}  max {q[6]:.1f}")
    report_figures(args)


def report_figures(args):
    """Per channel, results*/<ch>/25_fails/, from the stored files:
      reports                  the share of fits per outcome against the mass (fail_table.json; --table)
      nsig_outcomes_m<mass>    the fitted N_sig of every fit at the scan mass, one histogram per outcome
                               (fail_nsig_m<mass>.npz; --nsig)"""
    import plots as P
    from matplotlib.lines import Line2D
    form = {"atlasemu": "running power law", "powexp": r"power $\times$ exponential"}.get(args.cand, args.cand)
    tab = C.RESULTS / "fail_table.json"
    npz = C.RESULTS / f"fail_nsig_m{args.scan_mass:.0f}.npz"
    NSIG_JSON = C.RESULTS / f"fail_nsig_m{args.scan_mass:.0f}.json"
    for ch in args.channels:
        out = C.RESULTS / ch / "25_fails"
        out.mkdir(parents=True, exist_ok=True)
        if tab.exists():
            rows = sorted((r for r in json.loads(tab.read_text()) if r["ch"] == ch), key=lambda r: r["m"])
            ms = np.array([r["m"] for r in rows])
            fig, ax = P.scan_figure()
            handles = []
            for key in OUTCOMES:
                col, mk = OUTCOME_STYLE[key]
                ax.plot(ms, [100.0 * r["share"][key] for r in rows], "-", marker=mk, ms=7, lw=1.8, color=col)
                handles.append(Line2D([], [], color=col, lw=1.8, marker=mk, ms=7, label=OUTCOME_LABEL[key]))
            P.finish_scan(ax, channel=ch, extra=f"{form}\nfull range\nfrom {args.lo:.0f} GeV", ylabel="Fits [%]", ylim=(0.0, 150.0),
                          handles=handles, ncol=1)
            ax.set_yticks(np.arange(0.0, 101.0, 25.0))
            P.save(fig, out / "reports")
        if npz.exists():
            z = np.load(npz)
            n, cat = z[f"{ch}_nsig"], z[f"{ch}_cat"]
            summ = json.loads(NSIG_JSON.read_text()) if NSIG_JSON.exists() else {}
            summ[ch] = {"mass": args.scan_mass, "n": int(len(n)),
                        "outcomes": {key: {"n": int((cat == i).sum()), "neg": float((n[cat == i] < 0).mean()), "median": float(np.median(n[cat == i])),
                                           "max": float(n[cat == i].max()), "min": float(n[cat == i].min()),
                                           "zero": float((n[cat == i] == 0).mean()),       # fits that never left the start
                                           "outside": float(((n[cat == i] < NSIG_RANGE[0]) | (n[cat == i] > NSIG_RANGE[1])).mean())}
                                     for i, key in enumerate(OUTCOMES) if (cat == i).any()}}
            NSIG_JSON.write_text(json.dumps(summ, indent=1))
            bins = np.arange(NSIG_RANGE[0], NSIG_RANGE[1] + 1e-9, 0.5)
            fig, ax = P.scan_figure()
            handles, top = [], 1.0
            for i, key in enumerate(OUTCOMES):
                col, _ = OUTCOME_STYLE[key]
                h, _e, _p = ax.hist(n[cat == i], bins=bins, histtype="step", lw=2.0, color=col)
                top = max(top, h.max())
                # the number of fits of each outcome in the legend ("Pass (95,560), ...")
                handles.append(Line2D([], [], color=col, lw=2.0, label=f"{OUTCOME_LABEL[key]} ({int((cat == i).sum()):,})"))
            ax.plot([0.0, 0.0], [0.5, top * 3.0], color="k", lw=1.0)       # it ends below the legend
            handles.append(Line2D([], [], color="k", lw=1.0, label="injected: 0"))
            ax.set_yscale("log")
            P.finish_scan(ax, channel=ch, extra=f"{form}\nat {args.scan_mass / 1000:g} TeV", ylabel="toys / 0.5 events",
                          xlabel=r"fitted $N_{\rm sig}$ [events]", handles=handles, ncol=1)
            ax.set_xlim(*NSIG_RANGE)
            ax.set_ylim(0.5, top * 4e3)                           # room above the histograms for the labels
            P.save(fig, out / f"nsig_outcomes_m{args.scan_mass:.0f}")


NSIG_RANGE = (-20.0, 15.0)                  # the N_sig axis of the per-outcome histograms, in events
EXAMPLE_FILE = {"minimum not valid": "no_valid_minimum", "covariance incomplete": "covariance_incomplete", "pull beyond 5": "pull_beyond_5"}
COV_TEXT = {0: "not calculated", 1: "approximate", 2: "made positive-definite", 3: "accurate"}


def examples(args):
    """One toy per outcome that is not a pass (no valid minimum, covariance incomplete,
    pull beyond +-5): the first toy of each outcome among the toys of the scan mass, the
    seed of the table. fail_examples_m<mass>.json holds each toy, its fitted S+B per bin and what the fitter reported."""
    m = args.scan_mass
    jobs = [{"ch": ch, "m": m, "cand": args.cand, "lo": args.lo, "hi": 8000.0, "ntoys": args.ntoys, "seed": 7000 + int(m) + (0 if ch == "ee" else 1),
             "scan": False, "categories_only": True, "skip_b": True, "examples": list(OUTCOMES[1:])}
            for ch in args.channels]
    with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
        cells = pool.map(run_cell, jobs)
    path = C.RESULTS / f"fail_examples_m{m:.0f}.json"
    store = json.loads(path.read_text()) if path.exists() else {}
    for cell in cells:
        store[cell["ch"]] = {"mass": m, "cand": args.cand, "mu": cell["mu"], "sigma": cell["sigma"], "lo": cell["lo"], "hi": cell["hi"],
                             "toys_drawn": len(cell["toys"]), "examples": cell["examples"]}
        print(f"{cell['ch']} {m:.0f} GeV: Gaussian mu = {cell['mu']:.0f}, sigma = {cell['sigma']:.0f} GeV; {len(cell['toys'])} toys drawn")
        for key in OUTCOMES[1:]:
            x = cell["examples"].get(key)
            if x is None:
                print(f"  {key}: none among the toys")
                continue
            pull = x["nsig"] / x["err"] if x["err"] > 0 else float("nan")
            print(f"  {key:22s} toy #{x['index']}: N_sig = {x['nsig']:+.2f} +- {x['err']:.3f} (pull {pull:+.1f}); valid minimum {x['valid']}; "
                  f"covariance status {x['cov']} ({COV_TEXT[x['cov']]}); fitter status {x['status']}; edm {x['edm']:.2g}; "
                  f"toy events within +-1 sigma {x['n1']:.0f}, +-2 sigma {x['n2']:.0f}; lowest S+B {min(x['total']):+.3f}")
    path.write_text(json.dumps(store))
    example_figures(args)


def example_figures(args):
    """Per channel and outcome, results*/<ch>/25_fails/toy_<outcome>_m<mass>: the toy with its S+B fit, drawn as the
    single-toy figure of 11_bias_explainer.py (the expectation, the pseudo-data, the S+B fit; below, one pull per bin)."""
    path = C.RESULTS / f"fail_examples_m{args.scan_mass:.0f}.json"
    if not path.exists():
        return
    import plots as P
    from spectrum import centers, rebin
    from truth_sets import load_truth
    store = json.loads(path.read_text())
    form = {"atlasemu": "Running power law", "powexp": r"Power $\times$ expo"}.get(args.cand, args.cand)
    for ch in args.channels:
        if ch not in store:
            continue
        rec = store[ch]
        e10, parts, _ = load_truth(C.RESULTS / ch / "3_truths_split9" / "lo_ht" / "truth_raw")
        e, t, var = rebin(e10, parts["sum"], np.maximum(parts["sum"], 0.0))
        c = centers(e)
        sel = c >= C.THRESHOLD
        infit = (c >= rec["lo"]) & (c <= rec["hi"])
        out = C.RESULTS / ch / "25_fails"
        out.mkdir(parents=True, exist_ok=True)
        for key, x in rec["examples"].items():
            toy, total = np.array(x["toy"]), np.array(x["total"])
            fig, (ax, axr) = P.spectrum_figure(12)
            P.step_hist(ax, e, t, var, "0.55", "expected (raw MC)", errors=False)
            ax.errorbar(c[sel], toy[sel], yerr=np.sqrt(np.maximum(toy[sel], 0.0)), fmt="o", ms=4, color="k", label="pseudo-data (Poisson toy)")
            ax.plot(c[infit], np.maximum(total[infit], P.YMIN), color="#e42536", lw=2.2, label="S+B fit")
            ratio = np.where(total > 0, (toy - total) / np.sqrt(np.where(total > 0, total, 1.0)), np.nan)
            axr.plot(c[infit], ratio[infit], "o", ms=4, color="k")
            for yv in (-2, 2):
                axr.axhline(yv, color="0.5", lw=0.6, ls="--")
            P.finish(ax, axr, channel=ch, xmax=rec["hi"], top=float(np.nanmax(toy[sel])), log=True, ymin=1e-2, extra="Raw MC Bkg",
                     ratio_label=r"(data $-$ fit) / $\sqrt{\rm fit}$", ratio_lim=(-4, 4), ratio_ref=0.0,
                     legend_title=(f"{form} at {rec['mass'] / 1000:g} TeV\n"
                                   rf"Gaussian $\mu$ = {rec['mu']:.0f}, $\sigma$ = {rec['sigma']:.0f} GeV" + "\n"
                                   "injected: none\n"
                                   rf"fitted: $N_{{\rm sig}}$ = {x['nsig']:+.1f} $\pm$ {x['err']:.2f}" + "\n"
                                   f"outcome: {OUTCOME_LABEL[key]}"))
            P.save(fig, out / f"toy_{EXAMPLE_FILE[key]}_m{rec['mass']:.0f}")


def overlap(args):
    """Are the failure types one situation? Per channel and mass:
      A  every criterion on its own (a fit can fail several): the eight combinations of valid minimum / complete
         covariance / |pull| <= 5;
      B  per first failed criterion: N_sig < 0, no toy events within +-1 sigma, S+B <= 0 in a bin under the peak;
      C  the same toys refitted from other starting points: the outcome of each."""
    FAILS = ("minimum not valid", "covariance incomplete", "pull beyond 5")
    jobs = [{"ch": ch, "m": m, "cand": args.cand, "lo": args.lo, "hi": 8000.0, "ntoys": args.ntoys, "seed": 7000 + int(m) + (0 if ch == "ee" else 1),
             "scan": False, "categories_only": True, "alts": True}
            for ch in args.channels for m in args.masses]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        cells = pool.map(run_cell, jobs)
    pc = lambda a, b: f"{100.0 * a / b:.1f}" if b else "-"                                    # noqa: E731
    L = [f"# Do the failure types share one cause? {C.ERA} {C.REGION_NAME}, raw MC toys, {args.cand}, full range from {args.lo:.0f} GeV, N_sig free", "",
         f"{args.ntoys} toys per mass, no signal injected.", ""]
    rows = []
    for cell in sorted(cells, key=lambda c_: (c_["ch"], c_["m"])):
        toys = [x for x in cell["toys"] if x.get("valid") is not None]
        n = len(toys)
        for x in toys:
            x["V"], x["K"] = bool(x["valid"]), x["cov"] == 3
            x["P"] = bool(x["err"] and x["err"] > 0 and abs(x["nsig"] / x["err"]) <= PULL_RANGE)
        row = {"ch": cell["ch"], "m": cell["m"], "n": n, "not_falling": sum(not x["mono"] for x in toys)}
        L += [f"## {cell['ch']} {cell['m']:.0f} GeV ({cell['b_core']:.1f} background events within +-2 sigma)", "",
              "A. Every criterion on its own, % of all toys (the background falls in every fit: "
              + ("yes" if row["not_falling"] == 0 else f"no, {row['not_falling']} fits") + "):", "",
              "| valid minimum | covariance complete | pull within +-5 | % of toys |", "|---|---|---|---|"]
        row["patterns"] = {}
        for v in (True, False):
            for k in (True, False):
                for q in (True, False):
                    c_ = sum(x["V"] == v and x["K"] == k and x["P"] == q for x in toys)
                    row["patterns"][f"{int(v)}{int(k)}{int(q)}"] = c_ / n
                    if c_:
                        L.append(f"| {'yes' if v else 'no'} | {'yes' if k else 'no'} | {'yes' if q else 'no'} | {pc(c_, n)} |")
        L += ["", "B. What the failing fits have in common, per first failed criterion (% of the fits of that type):", "",
              "| first failed criterion | % of toys | N_sig < 0 | no toy events within +-1 sigma | S+B <= 0 in a bin under the peak |", "|---|---|---|---|---|"]
        row["common"] = {}
        for key in ("pass",) + FAILS:
            sel = [x for x in toys if x["cat"] == key]
            d = {"share": len(sel) / n, "neg": sum(x["nsig"] < 0 for x in sel), "empty": sum(x["n1"] == 0 for x in sel),
                 "zero": sum(x["min_tot"] is not None and x["min_tot"] <= 1e-6 for x in sel), "n": len(sel)}
            row["common"][key] = d
            L.append(f"| {key} | {pc(len(sel), n)} | {pc(d['neg'], len(sel))} | {pc(d['empty'], len(sel))} | {pc(d['zero'], len(sel))} |")
        L += ["", "C. The same toys from other starting points (% of the fits of that type; own = continued from where the fit stopped, "
              "toy_b = the background from the toy's own background-only fit, minus / plus = N_sig seeded at -/+ one sigma_fit):", "",
              "| first failed criterion | start | pass | minimum not valid | covariance incomplete | pull beyond 5 |", "|---|---|---|---|---|---|"]
        row["alt"] = {}
        for key in FAILS:
            sel = [x for x in toys if x["cat"] == key and x.get("alt")]
            row["alt"][key] = {}
            for name in ("own", "toy_b", "minus", "plus"):
                outs = [a[1] for x in sel for a in x["alt"] if a[0] == name]
                if not outs:
                    continue
                d = {c_: outs.count(c_) / len(outs) for c_ in ("pass",) + FAILS}
                row["alt"][key][name] = d
                L.append(f"| {key} | {name} | " + " | ".join(f"{100 * d[c_]:.1f}" for c_ in ("pass",) + FAILS) + " |")
            # over all starts: does the type change, does any start pass
            if sel:
                other = sum(any(a[1] in FAILS and a[1] != key for a in x["alt"]) for x in sel)
                anyp = sum(any(a[1] == "pass" for a in x["alt"]) for x in sel)
                row["alt"][key]["any_other_type"], row["alt"][key]["any_pass"] = other / len(sel), anyp / len(sel)
                L.append(f"| {key} | any start | passes from at least one: {pc(anyp, len(sel))} | another failure type from at least one: {pc(other, len(sel))} | | |")
        # the fits without a valid minimum that end at N_sig >= 0
        pos = [x for x in toys if x["cat"] == "minimum not valid" and x["nsig"] >= 0]
        if pos:
            st = {}
            for x in pos:
                st[x["status"]] = st.get(x["status"], 0) + 1
            row["invalid_pos"] = {"share": len(pos) / n, "empty": sum(x["n1"] == 0 for x in pos) / len(pos),
                                  "nsig_median": float(np.median([x["nsig"] for x in pos])), "status": {str(k): v for k, v in st.items()},
                                  "alt_pass": sum(any(a[1] == "pass" for a in x.get("alt", [])) for x in pos) / len(pos)}
            L += ["", f"Fits without a valid minimum that end at N_sig >= 0: {pc(len(pos), n)} % of toys; no toy events within +-1 sigma in "
                  f"{100 * row['invalid_pos']['empty']:.0f} %; median N_sig {row['invalid_pos']['nsig_median']:.2f}; fitter status "
                  + ", ".join(f"{k}: {v}" for k, v in sorted(st.items())) + f"; pass from another start {100 * row['invalid_pos']['alt_pass']:.0f} %."]
        L.append("")
        rows.append(row)
    (C.RESULTS / "FAIL_OVERLAP.md").write_text("\n".join(L) + "\n")
    (C.RESULTS / "fail_overlap.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(L))


def figures(args, record, summary):
    """Per channel, results*/<ch>/25_fails/:
      outcomes        the share of fits that pass against the mass: the background alone, and signal + background
      holes_m<mass>   the share of S+B fits failing against the number of toy events within +-1 sigma of the peak
      scan_m<mass>    -2 Delta ln L against N_sig (held fixed, background refitted) for a passing and a failing toy"""
    import plots as P
    from matplotlib.lines import Line2D
    form = {"atlasemu": "running power law", "powexp": r"power $\times$ exponential"}.get(args.cand, args.cand)
    extra = f"{form}\nfull range\nfrom {args.lo:.0f} GeV"
    for ch in args.channels:
        out = C.RESULTS / ch / "25_fails"
        out.mkdir(parents=True, exist_ok=True)
        rows = sorted((r for r in summary if r["ch"] == ch), key=lambda r: r["m"])
        ms = np.array([r["m"] for r in rows])
        # --- the fits that pass against the mass ---
        fig, ax = P.scan_figure()
        ax.plot(ms, [100.0 * r["b_ok_all"] for r in rows], "-o", ms=7, lw=1.8, color="k")
        ax.plot(ms, [100.0 * r["cats"]["pass"] for r in rows], "-s", ms=7, lw=1.8, color="#e42536")
        handles = [Line2D([], [], color="k", lw=1.8, marker="o", ms=7, label="background alone"),
                   Line2D([], [], color="#e42536", lw=1.8, marker="s", ms=7, label="signal + background")]
        P.finish_scan(ax, channel=ch, extra=extra, ylabel="Fits passing [%]", ylim=(0.0, 145.0), handles=handles, ncol=1)
        ax.set_yticks(np.arange(0.0, 101.0, 25.0))              # a share: no tick above 100 %
        P.save(fig, out / "outcomes")
        # --- the hole under the peak ---
        cell = record.get(ch)
        if cell is not None:
            xs, ys, es = [], [], []
            for k, nfail, nall in cell["holes"]:
                frac = nfail / nall
                xs.append(k); ys.append(100.0 * frac); es.append(100.0 * math.sqrt(max(frac * (1.0 - frac), 0.0) / nall))
            fig, ax = P.scan_figure()
            ax.errorbar(xs, ys, yerr=es, fmt="o", ms=7, lw=1.6, color="#e42536")
            ax.set_xlim(-0.5, max(xs) + 0.5)
            ax.set_xticks(xs)                                   # a count: whole numbers only
            ax.minorticks_off()
            handles = [Line2D([], [], color="#e42536", lw=0, marker="o", ms=7, label="signal + background fits")]
            P.finish_scan(ax, channel=ch, extra=f"{form}\nat {args.scan_mass / 1000:g} TeV", ylabel="Fits not passing [%]", ylim=(0.0, 145.0),
                          xlabel=r"Toy events within $\pm 1\sigma$ of the peak", handles=handles, ncol=1)
            ax.set_yticks(np.arange(0.0, 101.0, 20.0))
            P.save(fig, out / f"holes_m{args.scan_mass:.0f}")
            # --- the likelihood against N_sig ---
            sc = cell.get("scan")
            if sc:
                g = np.array(sc["grid"])
                i0 = int(np.argmin(np.abs(g)))
                fig, ax = P.scan_figure()
                YTOP = 12.0                                    # a curve leaves the frame where S+B <= 0 meets a bin with events
                ybot = -6.0
                for key, col, ls, lab in (("pass", "k", "-", "a toy whose fit passes"), ("fail", "#e42536", "-", "a toy with a pull beyond 5")):
                    y = np.array([np.nan if v is None else v for v in sc[key]], float)
                    d = 2.0 * (y - y[i0])
                    ax.plot(g, np.where(d <= YTOP, d, np.nan), color=col, lw=2.0, ls=ls)
                    ybot = min(ybot, 2.0 * math.floor((np.nanmin(d) - 1.5) / 2.0))     # the failing toy's minimum stays in the frame
                ax.axvline(sc["edge"], color="#e42536", lw=1.4, ls=(0, (2, 2)))
                ax.axhline(0.0, color="0.5", lw=0.8)
                ax.set_xlim(g[0], g[-1])
                handles = [Line2D([], [], color="k", lw=2.0, label="a toy whose fit passes"),
                           Line2D([], [], color="#e42536", lw=2.0, label=r"a toy with a pull beyond $\pm 5$"),
                           Line2D([], [], color="#e42536", lw=1.4, ls=(0, (2, 2)), label="S+B reaches zero")]
                P.finish_scan(ax, channel=ch, extra=f"{form}\nat {args.scan_mass / 1000:g} TeV", ylabel=r"$-2\,\Delta\ln L$ from $N_{\rm sig} = 0$",
                              ylim=(ybot, 14.0), xlabel=r"$N_{\rm sig}$ held fixed [events]", handles=handles, ncol=1)
                P.save(fig, out / f"scan_m{args.scan_mass:.0f}")
    # the scan's example toys, for the notes
    ex = {ch: c_["scan"] for ch, c_ in record.items() if c_.get("scan")}
    (C.RESULTS / f"fails_scan_m{args.scan_mass:.0f}.json").write_text(json.dumps({ch: {"mass": args.scan_mass, **{k: v for k, v in sc.items() if k not in ("grid", "pass", "fail")}}
                                                               for ch, sc in ex.items()}, indent=1))


if __name__ == "__main__":
    main()
