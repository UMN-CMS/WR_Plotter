"""PDF and alpha_s uncertainties from the analyzer's PDF members.

With ``--systs pdf`` the analyzer writes no Up/Down variations for the PDF.
It stores, per (region, PDF set), the fit mass of every LHEPdfWeight member
summed over the datasets that use the set:
``pdfmembers_<region>/<variable>_lha<id>``, a TH2 (x: the variable at the
analysis binning, y: member index, 0 = the set's central member).  The
manifest's ``pdf_sets`` gives each set's layout (WrCoffea ``wrcoffea/pdf_weights.py``).

The members must be rebinned BEFORE they are combined: each eigenvector keeps
the total yield and moves events between bins, so a quadrature formed on fine
bins and summed afterwards overstates the uncertainty of every wider bin (by a
few times in the wide last fit bins, on the 2024 signal).
Here every member is rebinned exactly like the plot (or the datacard window),
then, per bin:

- ``pdf``: sqrt(sum_i (h_i - h_0)^2) over the Hessian eigenvectors (standard
  deviation for replica sets);
- ``alphas``: (h_up - h_down) / 2 of the alpha_s members times the layout's
  ``alphas_scale`` (to delta alpha_s = 0.0015), signed.

Members of one set are summed over the sample's sub-era files before the
quadrature (the eigenvectors are the same).  Different sets in one sample
(e.g. 4FS single top next to 5FS W+jets in Nonprompt) are added linearly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from hist import Hist

from .histogram_cache import has_key, read_hist
from .inputs import read_manifest

logger = logging.getLogger(__name__)

PDF_SOURCES = ("pdf", "alphas")
_SHIFT_NAME = {"pdf": "PDF", "alphas": "AlphaS"}


def member_key(region: str, variable: str, lhaid: str | int) -> str:
    return f"pdfmembers_{region}/{variable}_lha{lhaid}"


def file_pdf_sets(path: str | Path) -> dict[str, dict]:
    """{lhaid: layout} recorded in the file's manifest (empty without PDF members)."""
    return dict((read_manifest(path) or {}).get("pdf_sets") or {})


def hessian_shifts(members: np.ndarray, layout: dict) -> dict[str, np.ndarray]:
    """Per-bin shifts {"PDF": delta >= 0, "AlphaS": signed} of (n_members, n_bins) members."""
    m = np.asarray(members, dtype=np.float64)
    h0 = m[layout.get("central", 0)]
    first, last = layout["eigenvectors"]
    eig = m[first:last + 1]
    if layout["error_type"] == "symmhessian":
        delta = np.sqrt(np.sum((eig - h0) ** 2, axis=0))
    elif layout["error_type"] == "replicas":
        delta = np.std(eig, axis=0, ddof=1)
    else:
        raise ValueError(f"Unsupported PDF error type {layout['error_type']!r}")
    out = {"PDF": delta}
    if layout.get("alphas_members"):
        a = layout["alphas_members"]
        out["AlphaS"] = float(layout["alphas_scale"]) * (m[a["up"]] - m[a["down"]]) / 2.0
    return out


def _member_hists(path: Path, key: str, n_members: int) -> list[Hist] | None:
    """The members of one TH2 as 1D histograms (x flow kept), or None if absent."""
    h2 = read_hist(path, key)
    if h2 is None:
        return None
    if h2.axes[1].size != n_members:
        raise ValueError(f"{path}:{key} has {h2.axes[1].size} members, the manifest says {n_members}")
    return [h2[:, i] for i in range(n_members)]


def sample_members(files: Sequence[str | Path], region: str, variable: str,
                   transform: Callable[[Hist], np.ndarray]) -> dict[str, tuple[np.ndarray, dict]]:
    """``{lhaid: (members (n_members, n_bins), layout)}`` summed over *files*.

    *transform* maps one member's 1D histogram to the values at the wanted
    binning (the plot's rebinning, or a datacard window).
    """
    out: dict[str, tuple[np.ndarray, dict]] = {}
    for f in files:
        path = Path(f)
        for lhaid, layout in file_pdf_sets(path).items():
            key = member_key(region, variable, lhaid)
            if not has_key(path, key):
                continue
            hists = _member_hists(path, key, int(layout["n_members"]))
            arr = np.stack([np.asarray(transform(h), dtype=np.float64) for h in hists])
            if lhaid in out:
                prev, prev_layout = out[lhaid]
                if prev_layout != layout:
                    raise ValueError(f"PDF set {lhaid} has different layouts across files of one sample")
                arr = prev + arr
            out[lhaid] = (arr, layout)
    return out


def total_shifts(members: dict[str, tuple[np.ndarray, dict]]) -> dict[str, np.ndarray]:
    """{"PDF": delta, "AlphaS": signed} summed linearly over the sample's PDF sets."""
    out: dict[str, np.ndarray] = {}
    for arr, layout in members.values():
        for name, shift in hessian_shifts(arr, layout).items():
            out[name] = shift if name not in out else out[name] + shift
    return out


def _with_values(template: Hist, values: np.ndarray) -> Hist:
    h = template.copy()
    view = h.view()
    if hasattr(view, "value"):
        view.value[...] = values
    else:
        view[...] = values
    return h


def pdf_variations(files: Sequence[str | Path], region: str, variable: str, nominal: Hist,
                   rebin: Callable[[Hist], Hist], wanted: Sequence[str] = PDF_SOURCES
                   ) -> dict[str, dict[str, Hist]]:
    """``{"pdf"|"alphas": {"up": Hist, "down": Hist}}`` of one sample, at the plot binning.

    *nominal* is the sample's nominal summed over *files* and rebinned with
    *rebin*; samples without PDF members (or variables without them) give {}.
    """
    members = sample_members(files, region, variable, lambda h: rebin(h).values())
    if not members:
        return {}
    shifts = total_shifts(members)
    nom = nominal.values()
    out = {}
    for syst in wanted:
        shift = shifts.get(_SHIFT_NAME[syst])
        if shift is None:
            continue
        if shift.shape != nom.shape:
            raise ValueError(f"PDF members of {region}/{variable} rebin to {shift.shape}, nominal {nom.shape}")
        out[syst] = {"up": _with_values(nominal, nom + shift), "down": _with_values(nominal, nom - shift)}
    return out
