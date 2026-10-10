"""Total systematic envelope of a summed-MC histogram from the analyzer's
`syst_<name><up|down>_<region>` variations (data/systematics.yaml lists the
names). Used by the seminar explainer plots to draw a stat (+) syst band and
by prepare_fcr to vary the transfer factor R.

    edges, nom, up, dn = total_syst_envelope(dirs, region, var, samples)

`up`/`dn` are per-bin absolute uncertainties on the sum over `samples`: for
each source the larger upward (downward) deviation of its two variations,
summed over samples, rebinned, then the sources added in quadrature. MC
statistics (sumw2) can be folded in with `with_stat=True`.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import uproot

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))              # repo root (wrplotter)
from wrplotter.config import load_systematics          # noqa: E402

logger = logging.getLogger("syst_band")

# analyzer placeholders (see the *_systematics.json notes): not real
# uncertainties yet, so never part of the envelope
PLACEHOLDER = {"muscale", "musmear"}
# Composite sources: one band/nuisance from the ENVELOPE (not the quadrature
# sum) of several correlated analyzer variations.  The QCD scale uncertainty
# is the envelope of the muR, muF and muR=muF variations.
ENVELOPES = {"scale": ("scalemur", "scalemuf", "scalemurf")}


def _sum_hist(dirs, sample, key):
    """Summed (edges, values, sumw2) over sub-era dirs, or None if absent."""
    edges = vals = w2 = None
    for d in dirs:
        fp = Path(d) / f"WRAnalyzer_{sample}.root"
        if not fp.exists():
            continue
        f = uproot.open(fp)
        if key not in f:
            return None
        h = f[key]
        e, v, var = h.axes[0].edges(), h.values(), h.variances()
        if edges is None:
            edges, vals, w2 = e, v.copy(), var.copy()
        else:
            vals += v
            w2 += var
    if edges is None:
        return None
    vals = np.where(np.isfinite(vals) & (vals > 0), vals, 0.0)
    return edges, vals, w2


def _rebin(edges, vals, rebin):
    if rebin <= 1:
        return edges, vals
    n = (len(vals) // rebin) * rebin
    return edges[0:n + 1:rebin], vals[:n].reshape(-1, rebin).sum(axis=1)


def _syst_key(src, dirn, region, variable):
    return f"syst_{src}{dirn}_{region}/{variable}_syst_{src}{dirn}_{region}"


def variation_keys(src, region, variable):
    """[(label, key)] of every analyzer histogram that belongs to `src`:
    two (up/down) for a plain source, 2 x members for an envelope source."""
    members = ENVELOPES.get(src, (src,))
    return [(f"{m}{d}" if src in ENVELOPES else d, _syst_key(m, d, region, variable))
            for m in members for d in ("up", "down")]


def variation_hists(dirs, sample, src, region, variable):
    """[values] of the variations of `src` that exist for `sample` (any
    length; the caller takes the envelope over them)."""
    out = []
    for _, key in variation_keys(src, region, variable):
        r = _sum_hist(dirs, sample, key)
        if r is not None:
            out.append(r[1])
    return out


def source_deviations(dirs, region, variable, samples, sources=None,
                      exclude=PLACEHOLDER):
    """(edges, nominal, stat2, {source: (d_up, d_down)}) at native binning;
    d_up/d_down are the summed-over-samples upward/downward deviations."""
    sources = [s for s in (sources or load_systematics()) if s not in exclude]
    nom_key = f"{region}/{variable}_{region}"
    noms = {}
    edges = nom = stat2 = None
    for s in samples:
        r = _sum_hist(dirs, s, nom_key)
        if r is None:
            continue
        edges, v, w2 = r
        noms[s] = v
        nom = v.copy() if nom is None else nom + v
        stat2 = w2.copy() if stat2 is None else stat2 + w2
    if nom is None:
        raise RuntimeError(f"no nominal histograms for {samples} in {region}")
    devs = {}
    for src in sources:
        d_up = np.zeros_like(nom)
        d_dn = np.zeros_like(nom)
        found = False
        for s, v_nom in noms.items():
            pair = [v - v_nom for v in variation_hists(dirs, s, src, region, variable)]
            if not pair:
                continue
            found = True
            pair = np.array(pair)      # envelope over all variations of the source
            d_up += np.maximum(pair.max(axis=0), 0.0)
            d_dn += np.maximum((-pair).max(axis=0), 0.0)
        if found:
            devs[src] = (d_up, d_dn)
    logger.info("syst sources for %s/%s: %s", region, variable, sorted(devs))
    return edges, nom, stat2, devs


def total_syst_envelope(dirs, region, variable, samples, sources=None,
                        rebin=1, with_stat=False, exclude=PLACEHOLDER):
    """(edges, nominal, up, down) for the summed MC of `samples`."""
    edges, nom, stat2, devs = source_deviations(dirs, region, variable,
                                                samples, sources, exclude)
    e_r, nom_r = _rebin(edges, nom, rebin)
    up2 = np.zeros_like(nom_r)
    dn2 = np.zeros_like(nom_r)
    for d_up, d_dn in devs.values():
        _, du = _rebin(edges, d_up, rebin)
        _, dd = _rebin(edges, d_dn, rebin)
        up2 += du ** 2
        dn2 += dd ** 2
    if with_stat:
        _, st = _rebin(edges, stat2, rebin)
        up2 += st
        dn2 += st
    return e_r, nom_r, np.sqrt(up2), np.sqrt(dn2)


def window_yield_variations(dirs, region, variable, samples, lo, hi,
                            sources=None, exclude=PLACEHOLDER):
    """Per-source (up, down) yields of the summed MC with bin centres in
    [lo, hi]; returns (nominal, {source: (up, down)})."""
    sources = [s for s in (sources or load_systematics()) if s not in exclude]
    nom_key = f"{region}/{variable}_{region}"

    def ysum(key):
        tot, ok = 0.0, False
        for s in samples:
            r = _sum_hist(dirs, s, key)
            if r is None:
                continue
            e, v, _ = r
            c = 0.5 * (e[:-1] + e[1:])
            tot += float(v[(c >= lo) & (c <= hi)].sum())
            ok = True
        return tot if ok else None

    nominal = ysum(nom_key)
    out = {}
    for src in sources:
        ys = [ysum(key) for _, key in variation_keys(src, region, variable)]
        ys = [y for y in ys if y is not None]
        if len(ys) < 2:
            continue
        # (up, down) = the envelope: largest and smallest varied yield
        out[src] = (max(ys), min(ys))
    return nominal, out
