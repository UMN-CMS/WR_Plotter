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
        # DY CRs
        Region("wr_mumu_resolved_dy_cr", "muon",   True, rf"$\mu\mu$\nResolved DY CR\n{era}"),
        Region("wr_ee_resolved_dy_cr",   "egamma", True, rf"ee\nResolved DY CR\n{era}"),

        # Flavor CR: SAME NAME, two datasets
        Region("wr_resolved_flavor_cr", "muon",   True, rf"$e\mu$\nResolved Flavor CR\n{era}"),
        Region("wr_resolved_flavor_cr", "egamma", True, rf"$e\mu$\nResolved Flavor CR\n{era}"),
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
    """
    buckets = regions_by_name(era)
    valid_names = set(buckets.keys())
    valid_tokens = {f"{r.name}:{r.primary_dataset}" for rs in buckets.values() for r in rs}

    selected: List[Region] = []
    unknown: List[str] = []
    for tok in requested:
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
            f"Or target a dataset with ':muon' or ':egamma'. "
            f"Examples: {sorted(list(valid_tokens))[:2]} ..."
        )

    return selected

__all__ = ["Region", "regions_for_era", "regions_by_name", "expand_region_requests"]
