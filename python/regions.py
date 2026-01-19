# python/regions.py
from dataclasses import dataclass
from typing import List, Dict
import logging
from collections import defaultdict

@dataclass(frozen=True)
class Region:
    name: str
    primary_dataset: str   # "egamma" or "muon"
    unblind_data: bool = False
    tlatex_alias: str = ""

    def print(self) -> None:
        logging.info(
            f"Region(name={self.name}, primary_dataset={self.primary_dataset}, "
            f"unblind={self.unblind_data}, label={self.tlatex_alias})"
        )

def regions_for_era(era: str) -> List[Region]:
    """All supported regions for an era."""
    return [
        # Resolved DY CRs
        Region("wr_mumu_resolved_dy_cr", "muon",   True, rf"$\mu\mu$\nResolved DY CR\n{era}"),
        Region("wr_ee_resolved_dy_cr",   "egamma", True, rf"ee\nResolved DY CR\n{era}"),

        # Resolved Flavor CR: SAME NAME, two datasets
        Region("wr_resolved_flavor_cr", "muon",   True, rf"$e\mu$\nResolved Flavor CR\n{era}"),
        Region("wr_resolved_flavor_cr", "egamma", True, rf"$e\mu$\nResolved Flavor CR\n{era}"),

        # Resolved SRs
        Region("wr_mumu_resolved_sr", "muon",   True, rf"$\mu\mu$\nResolved SR\n{era}"),
        Region("wr_ee_resolved_sr",   "egamma", True, rf"ee\nResolved SR\n{era}"),

        # Boosted DY CRs
        Region("wr_mumu_boosted_dy_cr", "muon",   True, rf"$\mu\mu$\nBoosted DY CR\n{era}"),
        Region("wr_ee_boosted_dy_cr",   "egamma", True, rf"ee\nBoosted DY CR\n{era}"),

        # Boosted Flavor CRs: separate e-mu and mu-e regions
        Region("wr_emu_boosted_flavor_cr", "egamma", True, rf"$e\mu$\nBoosted Flavor CR\n{era}"),
        Region("wr_mue_boosted_flavor_cr", "muon",   True, rf"$\mu e$\nBoosted Flavor CR\n{era}"),

        # Boosted SRs
        Region("wr_mumu_boosted_sr", "muon",   True, rf"$\mu\mu$\nBoosted SR\n{era}"),
        Region("wr_ee_boosted_sr",   "egamma", True, rf"ee\nBoosted SR\n{era}"),
    ]

def regions_by_name(era: str) -> Dict[str, List[Region]]:
    """Map name -> all Region variants (e.g. both muon/egamma for flavor CR)."""
    buckets = defaultdict(list)
    for r in regions_for_era(era):
        buckets[r.name].append(r)
    return dict(buckets)

def expand_region_requests(era: str, requested: List[str]) -> List[Region]:
    """
    Expand user requests into concrete Region objects.
    Supports:
      - "<name>"                 → all variants of that region
      - "<name>:muon"/":egamma"  → only that dataset
      - Shorthand: "resolved_dy_cr" → both ee and mumu DY CRs
      - Shorthand: "resolved_sr" → both ee and mumu SRs
      - Shorthand: "resolved_flavor_cr" → both ee and mumu flavor CRs
    """
    # Shorthand mappings
    shorthand_map = {
        "resolved_dy_cr": ["wr_ee_resolved_dy_cr", "wr_mumu_resolved_dy_cr"],
        "resolved_sr": ["wr_ee_resolved_sr", "wr_mumu_resolved_sr"],
        "resolved_flavor_cr": ["wr_resolved_flavor_cr"],
        "boosted_dy_cr": ["wr_ee_boosted_dy_cr", "wr_mumu_boosted_dy_cr"],
        "boosted_sr": ["wr_ee_boosted_sr", "wr_mumu_boosted_sr"],
        "boosted_flavor_cr": ["wr_emu_boosted_flavor_cr", "wr_mue_boosted_flavor_cr"],
    }
    
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

    return selected

__all__ = ["Region", "regions_for_era", "regions_by_name", "expand_region_requests"]
