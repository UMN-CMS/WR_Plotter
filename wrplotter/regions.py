# wrplotter/regions.py
import re
from dataclasses import dataclass
from typing import List, Dict
import logging
from collections import defaultdict

from .config import era_label, load_region_shorthands

_SR_PATTERN = re.compile(r"_sr(_|$)")


@dataclass(frozen=True)
class Region:
    name: str
    primary_dataset: str   # data stream of the region's trigger: "egamma" or "muon"
    tlatex_alias: str = ""
    year: str = ""         # the era's plot label (data/lumi.yaml), shown with the luminosity, not in the frame

    @property
    def is_signal_region(self) -> bool:
        """True for the SRs and the variants derived from them (e.g. ``_sr_nomass``)."""
        return _SR_PATTERN.search(self.name) is not None

    @property
    def is_boosted(self) -> bool:
        return "boosted" in self.name.lower()

    @property
    def topology(self) -> str:
        return "boosted" if self.is_boosted else "resolved"

    def print(self) -> None:
        logging.info(
            f"Region(name={self.name}, primary_dataset={self.primary_dataset}, "
            f"label={self.tlatex_alias})"
        )

def regions_for_era(era: str) -> List[Region]:
    """All supported regions for an era, each with the data stream of its trigger.

    The analyzer fills every region in both data files without overlap removal
    (WrCoffea analyzer.py): the electron trigger selects the ee regions and
    wr_emu_boosted_flavor_cr, the muon trigger the mumu regions,
    wr_resolved_flavor_cr (its e-mu trigger is the muon trigger) and
    wr_mue_boosted_flavor_cr.  The other stream would add a trigger-overlap subset.
    """
    era = era_label(era)    # the plots' year label: cms_label puts it beside the luminosity
    return [
        # Resolved DY CRs
        Region("wr_mumu_resolved_dy_cr", "muon",   rf"$\mu\mu$\nResolved DY CR", era),
        Region("wr_ee_resolved_dy_cr",   "egamma", rf"ee\nResolved DY CR", era),

        # Resolved flavor CR (muon trigger)
        Region("wr_resolved_flavor_cr", "muon", rf"$e\mu$\nResolved Flavor CR", era),

        # Resolved SRs
        Region("wr_mumu_resolved_sr", "muon",   rf"$\mu\mu$\nResolved SR", era),
        Region("wr_ee_resolved_sr",   "egamma", rf"ee\nResolved SR", era),

        # Boosted DY CRs
        Region("wr_mumu_boosted_dy_cr", "muon",   rf"$\mu\mu$\nBoosted DY CR", era),
        Region("wr_ee_boosted_dy_cr",   "egamma", rf"ee\nBoosted DY CR", era),

        # Boosted SRs
        Region("wr_mumu_boosted_sr", "muon",   rf"$\mu\mu$\nBoosted SR", era),
        Region("wr_ee_boosted_sr",   "egamma", rf"ee\nBoosted SR", era),

        # Boosted flavor CRs: leading electron (electron trigger) / leading muon (muon trigger)
        Region("wr_emu_boosted_flavor_cr", "egamma", rf"e ($\mu$-jet)\nBoosted Flavor CR", era),
        Region("wr_mue_boosted_flavor_cr", "muon",   rf"$\mu$ (e-jet)\nBoosted Flavor CR", era),
    ]

def regions_by_name(era: str) -> Dict[str, List[Region]]:
    """Map name -> Region variants (one per name since each region has one data stream)."""
    buckets = defaultdict(list)
    for r in regions_for_era(era):
        buckets[r.name].append(r)
    return dict(buckets)

def expand_region_requests(era: str, requested: List[str]) -> List[Region]:
    """
    Expand user requests into concrete Region objects.
    Supports:
      - "<name>"                 → that region
      - "<name>:muon"/":egamma"  → that region, if it uses that data stream
      - Shorthand: "resolved_dy_cr" → both ee and mumu DY CRs
      - Shorthand: "resolved_sr" → both ee and mumu SRs
      - Shorthand: "boosted_flavor_cr" → both boosted flavor CRs
    """
    shorthand_map = load_region_shorthands()

    # Expand shorthands first
    expanded_requested = []
    for tok in requested:
        if tok in shorthand_map:
            expanded_requested.extend(shorthand_map[tok])
        else:
            expanded_requested.append(tok)

    buckets = regions_by_name(era)
    valid_names = set(buckets.keys())
    valid_tokens = {f"{r.name}:{r.primary_dataset}" for rs in buckets.values() for r in rs}

    selected: List[Region] = []
    unknown: List[str] = []
    for tok in expanded_requested:
        if ":" in tok:
            name, ds = tok.split(":", 1)
            if name not in valid_names:
                unknown.append(tok); continue
            matched = [r for r in buckets[name] if r.primary_dataset == ds]
            if not matched:
                unknown.append(tok); continue
            selected.extend(matched)
        else:
            if tok not in valid_names:
                unknown.append(tok); continue
            selected.extend(buckets[tok])

    if unknown:
        raise ValueError(
            f"Unknown region token(s): {unknown}. "
            f"Valid region names: {sorted(valid_names)}. "
            f"Valid shorthands: {sorted(shorthand_map.keys())}. "
            f"Or target a dataset with ':muon' or ':egamma'. "
            f"Examples: {sorted(list(valid_tokens))[:2]} ..."
        )

    # A region requested twice (e.g. by name and by shorthand) is plotted once.
    return list(dict.fromkeys(selected))

__all__ = ["Region", "regions_for_era", "regions_by_name", "expand_region_requests"]
