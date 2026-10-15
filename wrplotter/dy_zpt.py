"""The Drell-Yan Z-pT reweighting payload, drawn.

The analyzer (wrcoffea/dy_reweight.py) reweights the LO HT-binned DY by
QCD(pT) x EW(pT) at the generator-level Z pT, from a JSON payload (wrcoffea
``data/reweights/<...>/dy_zpt_reweight_*.json``; ``config.yaml: dy_zpt_reweight``
maps one per era).  This module reads a payload and draws:

- :func:`plot_qcd_ratio`: the QCD NLO / LO ratio with its scale, PDF and
  alpha_S variations, as AN-19-083 Fig. 14 showed the Run 2 correction, with
  the Run 2 ratio overlaid when given;
- :func:`plot_ew_correction`: the electroweak factor with its three uncertainties;
- :func:`plot_weight`: the product the analyzer applies, with the total
  uncertainty, next to its two factors;
- :func:`plot_spectra`: the generator-level LO and NLO Z pT spectra the ratio
  was taken from;
- :func:`plot_applied_ratio`: the reweighted over the unreweighted DY in a
  reconstructed pT(ll) distribution, against the payload.

The bin lookups mirror ``wrcoffea.dy_reweight.evaluate``: the QCD ratio is read
at min(pT, pt_max); the EW factor at ``pt_min`` below 30 GeV and at ``pt_max``
from 6500 GeV on (the Run 2 HNWRAnalyzer rule).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from hist import Hist
from matplotlib.figure import Figure

from .plotting_helpers import FONT_SIZE_LEGEND, cms_label, fit_header, region_text

# CMS (Petroff) palette: the three QCD variations / the three EW uncertainties, the Run 2 reference
C_VARIATIONS = ("#5790fc", "#f89c20", "#964a8b")
C_REFERENCE = "#9c9ca1"
C_LO, C_NLO = "#5790fc", "#e42536"

# wrcoffea.dy_reweight.evaluate (HNWRAnalyzer): below EW_PT_LOW the factor at pt_min, from EW_PT_HIGH on
# the factor at pt_max.
EW_PT_LOW, EW_PT_HIGH = 30.0, 6500.0

QCD_KEYS = ("central", "pdf_err", "scale_up", "scale_down", "alphas_up", "alphas_down")


@dataclass(frozen=True)
class ZptPayload:
    era: str
    provenance: dict
    qcd_edges: np.ndarray
    qcd: dict                   # QCD_KEYS and stat_err (zeros when the payload has none), per bin
    qcd_pt_max: float
    lo_pb: np.ndarray | None    # the LO and NLO cross sections per bin [pb] the ratio came from
    nlo_pb: np.ndarray | None
    ew_edges: np.ndarray
    ew: np.ndarray              # the EW factor per bin
    ew_err: np.ndarray          # (3, bins): its three uncertainties
    ew_pt_min: float
    ew_pt_max: float

    @property
    def qcd_centres(self) -> np.ndarray:
        return 0.5 * (self.qcd_edges[1:] + self.qcd_edges[:-1])


def load_zpt_payload(path: str | Path) -> ZptPayload:
    """Read a Z-pT payload (ours, or a converted Run 2 one, which has no stat_err and no spectra)."""
    path = Path(path)
    raw = json.loads(path.read_text())
    q, e = raw["qcd"], raw["ew"]
    qcd_edges = np.asarray(q["edges"], dtype=float)
    n = len(qcd_edges) - 1
    qcd = {k: np.asarray(q[k], dtype=float) for k in QCD_KEYS}
    qcd["stat_err"] = np.asarray(q.get("stat_err", np.zeros(n)), dtype=float)
    bad = [k for k, v in qcd.items() if v.shape != (n,)]
    if bad:
        raise ValueError(f"{path}: qcd {bad} do not have one value per bin ({n} bins)")
    ew_edges = np.asarray(e["edges"], dtype=float)
    m = len(ew_edges) - 1
    ew = np.asarray(e["central"], dtype=float)
    ew_err = np.asarray([e["e1_err"], e["e2_err"], e["e3_err"]], dtype=float)
    if ew.shape != (m,) or ew_err.shape != (3, m):
        raise ValueError(f"{path}: ew central/e1_err/e2_err/e3_err do not have one value per bin ({m} bins)")
    prov = raw.get("provenance") or {}
    spectra = [np.asarray(q[k], dtype=float) if k in q else None for k in ("lo_pb_per_bin", "nlo_pb_per_bin")]
    return ZptPayload(
        era=str(prov.get("era") or prov.get("year") or path.stem),
        provenance=prov,
        qcd_edges=qcd_edges, qcd=qcd, qcd_pt_max=float(q["pt_max"]),
        lo_pb=spectra[0], nlo_pb=spectra[1],
        ew_edges=ew_edges, ew=ew, ew_err=ew_err,
        ew_pt_min=float(e["pt_min"]), ew_pt_max=float(e["pt_max"]),
    )


# ---------------------------------------------------------------------------
# Evaluation, as the analyzer does it
# ---------------------------------------------------------------------------

def _bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """ROOT ``FindBin`` on the in-range axis: bin i covers [e_i, e_{i+1})."""
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)


def qcd_at(p: ZptPayload, pt: np.ndarray) -> dict:
    """The QCD ratio and its variations at generator Z *pt* (pT above pt_max reads the last bin)."""
    i = _bin_index(np.minimum(np.asarray(pt, dtype=float), p.qcd_pt_max), p.qcd_edges)
    return {k: v[i] for k, v in p.qcd.items()}


def ew_at(p: ZptPayload, pt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The EW factor and its (3, N) uncertainties at generator Z *pt*."""
    pt = np.asarray(pt, dtype=float)
    pt = np.where(pt < EW_PT_LOW, p.ew_pt_min, pt)
    pt = np.where(pt >= EW_PT_HIGH, p.ew_pt_max, pt)
    i = _bin_index(pt, p.ew_edges)
    return p.ew[i], p.ew_err[:, i]


def weight_at(p: ZptPayload, pt: np.ndarray) -> dict:
    """The applied weight QCD x EW at generator Z *pt*, with its total up/down from the six
    variations added in quadrature (scale and alpha_S asymmetric), and the two factors."""
    q = qcd_at(p, pt)
    e, err = ew_at(p, pt)
    c = q["central"]
    nominal = c * e
    common = (q["pdf_err"] / c) ** 2 + np.sum((err / e) ** 2, axis=0)
    up = np.sqrt(((q["scale_up"] - c) / c) ** 2 + ((q["alphas_up"] - c) / c) ** 2 + common)
    down = np.sqrt(((q["scale_down"] - c) / c) ** 2 + ((q["alphas_down"] - c) / c) ** 2 + common)
    return {"nominal": nominal, "up": nominal * (1 + up), "down": nominal * (1 - down), "qcd": c, "ew": e}


def weight_edges(p: ZptPayload, xmax: float | None = None) -> np.ndarray:
    """Bin edges on which QCD x EW is a step function: both payloads' edges merged, from 0 to *xmax*
    (default: the QCD range)."""
    hi = float(xmax if xmax is not None else p.qcd_edges[-1])
    edges = np.unique(np.concatenate([p.qcd_edges, p.ew_edges, [0.0, hi]]))
    return edges[(edges >= 0.0) & (edges <= hi)]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

X_LABEL = r"$p_{T}$ of the generator-level Z [GeV]"


def _finish(fig: Figure, ax, *, com: float, text: str, year: str | None = None,
            legend_fontsize: int = FONT_SIZE_LEGEND) -> Figure:
    ax.legend(loc="upper right", fontsize=legend_fontsize)
    cms_label(ax, None, com, data=False, year=year, text=text)
    fit_header(ax)
    return fig


def plot_qcd_ratio(p: ZptPayload, *, reference: ZptPayload | None = None,
                   xlim: tuple[float, float] = (0.0, 1000.0), logx: bool = False,
                   com: float = 13.6, text: str = "", year: str | None = None) -> Figure:
    """The QCD NLO / LO ratio vs generator Z pT with its scale (envelope), PDF and alpha_S
    variations as bands and its statistical error as bars (AN-19-083 Fig. 14), with the
    Run 2 *reference* ratio dashed on top."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    e, q = p.qcd_edges, p.qcd
    c = q["central"]
    bands = [(q["scale_down"], q["scale_up"], r"$\mu_R$, $\mu_F$ scale (envelope)"),
             (c - q["pdf_err"], c + q["pdf_err"], "PDF (Hessian)"),
             (q["alphas_down"], q["alphas_up"], r"$\alpha_S$")]
    for (lo, hi, label), color in zip(bands, C_VARIATIONS):
        ax.stairs(hi, e, baseline=lo, fill=True, color=color, alpha=0.35, lw=0, label=label)
    ax.stairs(c, e, color="k", lw=2, label="NLO / LO")
    if np.any(q["stat_err"] > 0):
        ax.errorbar(p.qcd_centres, c, yerr=q["stat_err"], fmt="none", ecolor="k", elinewidth=1.5)
    if reference is not None:
        ax.stairs(reference.qcd["central"], reference.qcd_edges, color=C_REFERENCE, lw=2, ls="--",
                  label=f"Run 2 ({reference.era})")
    ax.axhline(1.0, color="0.5", lw=1, ls=":")
    if logx:
        ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.8)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("QCD NLO / LO")
    return _finish(fig, ax, com=com, text=text, year=year)


def plot_ew_correction(p: ZptPayload, *, xlim: tuple[float, float] | None = None,
                       com: float = 13.6, text: str = "", year: str | None = None) -> Figure:
    """The EW factor vs generator Z pT with its three uncertainties as bands (log x)."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    e, c = p.ew_edges, p.ew
    for i, color in enumerate(C_VARIATIONS):
        ax.stairs(c + p.ew_err[i], e, baseline=c - p.ew_err[i], fill=True, color=color, alpha=0.35, lw=0,
                  label=f"EW{i + 1} uncertainty")
    ax.stairs(c, e, color="k", lw=2, label="EW correction")
    ax.axhline(1.0, color="0.5", lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(*(xlim or (e[0], e[-1])))
    ax.set_ylim(0.4, 1.4)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("EW correction")
    return _finish(fig, ax, com=com, text=text, year=year)


def plot_weight(p: ZptPayload, *, xlim: tuple[float, float] = (1.0, 8000.0),
                com: float = 13.6, text: str = "", year: str | None = None) -> Figure:
    """QCD x EW, the weight the analyzer gives a DY event, vs generator Z pT (log x), with the
    total uncertainty band and the two factors dashed."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    edges = weight_edges(p, xlim[1])
    w = weight_at(p, 0.5 * (edges[1:] + edges[:-1]))
    ax.stairs(w["up"], edges, baseline=w["down"], fill=True, color="0.6", alpha=0.4, lw=0,
              label="Total uncertainty")
    ax.stairs(w["qcd"], edges, color=C_VARIATIONS[0], lw=1.5, ls="--", label="QCD NLO / LO")
    ax.stairs(w["ew"], edges, color=C_VARIATIONS[1], lw=1.5, ls=":", label="EW correction")
    ax.stairs(w["nominal"], edges, color="k", lw=2, label=r"Weight (QCD $\times$ EW)")
    ax.axhline(1.0, color="0.5", lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.8)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(r"DY Z $p_{T}$ weight")
    return _finish(fig, ax, com=com, text=text, year=year)


def plot_spectra(p: ZptPayload, *, xlim: tuple[float, float] = (1.0, 8000.0),
                 com: float = 13.6, text: str = "", year: str | None = None) -> Figure:
    """The generator-level Z pT spectra of the inclusive LO and NLO samples the ratio was taken
    from, as cross sections per GeV (log-log), with NLO / LO below."""
    if p.lo_pb is None or p.nlo_pb is None:
        raise ValueError(f"payload of {p.era} holds no LO/NLO spectra (lo_pb_per_bin, nlo_pb_per_bin)")
    hep.style.use("CMS")
    fig, (ax, rax) = plt.subplots(2, 1, gridspec_kw=dict(height_ratios=[3, 1], hspace=0.1), sharex=True)
    e = p.qcd_edges
    widths = np.diff(e)
    ax.stairs(p.lo_pb / widths, e, color=C_LO, lw=2, label="LO madgraphMLM, inclusive")
    ax.stairs(p.nlo_pb / widths, e, color=C_NLO, lw=2, label="NLO amcatnloFXFX, inclusive")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    positive = np.concatenate([p.lo_pb / widths, p.nlo_pb / widths])
    positive = positive[positive > 0]
    ax.set_ylim(positive.min() * 0.5, positive.max() * 50)
    ax.set_ylabel(r"$\mathrm{d}\sigma / \mathrm{d}p_{T}$ [pb / GeV]")
    rax.stairs(p.qcd["central"], e, color="k", lw=2)
    rax.errorbar(p.qcd_centres, p.qcd["central"], yerr=p.qcd["stat_err"], fmt="none", ecolor="k",
                 elinewidth=1.5)
    rax.axhline(1.0, color="k", lw=1, ls="--")
    rax.set_ylim(0.5, 1.5)
    rax.set_ylabel("NLO / LO")
    rax.set_xlabel(X_LABEL)
    ax.set_xlabel("")
    integrated = (p.provenance.get("integrated") or {}).get("nlo_over_lo")
    if integrated is not None:
        text = f"{text}\nIntegrated NLO / LO = {integrated:.4f}" if text else f"Integrated NLO / LO = {integrated:.4f}"
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)
    cms_label(ax, None, com, data=False, year=year)
    # The spectra peak under the label block, so the description goes to the empty lower left.
    region_text(ax, text, x=0.05, y=0.30)
    fit_header(ax)
    return fig


def plot_applied_ratio(p: ZptPayload, before: Hist, after: Hist, *, xlabel: str,
                       xlim: tuple[float, float] = (0.0, 1000.0), com: float = 13.6, text: str = "",
                       year: str | None = None) -> Figure:
    """*after* / *before* per bin (the reweighted over the unreweighted DY in a reconstructed
    pT(ll) distribution: the mean weight of the events in the bin, so no error bars) against the
    payload's weight vs generator Z pT."""
    hep.style.use("CMS")
    fig, ax = plt.subplots()
    edges = np.asarray(before.axes[0].edges)
    b, a = before.values(), after.values()
    ratio = np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b > 0)
    centres = 0.5 * (edges[1:] + edges[:-1])
    wedges = weight_edges(p, xlim[1])
    w = weight_at(p, 0.5 * (wedges[1:] + wedges[:-1]))
    ax.stairs(w["nominal"], wedges, color=C_VARIATIONS[0], lw=2, label=r"Payload at generator Z $p_{T}$")
    ax.plot(centres, ratio, "o", color="k", ms=6, label="DY reweighted / not reweighted")
    ax.axhline(1.0, color="0.5", lw=1, ls=":")
    ax.set_xlim(*xlim)
    ax.set_ylim(0.4, 1.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"DY Z $p_{T}$ weight")
    return _finish(fig, ax, com=com, text=text, year=year)
