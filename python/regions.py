# python/regions.py
from __future__ import annotations
from dataclasses import dataclass
import logging

@dataclass(frozen=True)
class Region:
    name: str
    # Must match a SampleGroup *key* (e.g. "muon", "egamma"), not the display label.
    primary_dataset: str
    unblind_data: bool = False
    tlatex_alias: str = ""

    # Keep a print() for compatibility with Plotter.print_regions()
    def print(self) -> None:
        logging.info(
            f"Region(name={self.name}, primary_dataset={self.primary_dataset}, "
            f"unblind={self.unblind_data}, label={self.tlatex_alias})"
        )


def build_regions(era: str, category: str) -> list[Region]:
    cat = category.lower()
    if cat == "dy_cr":
        return [
            Region("wr_mumu_resolved_dy_cr", "muon",   True, f"$\\mu\\mu$\nResolved DY CR\n{era}"),
            Region("wr_ee_resolved_dy_cr",   "egamma", True, f"ee\nResolved DY CR\n{era}"),
        ]
    if cat == "flavor_cr":
        return [
            Region("wr_resolved_flavor_cr", "muon",   True, f"$e\\mu$\nResolved Flavor CR\n{era}"),
            Region("wr_resolved_flavor_cr", "egamma", True, f"$e\\mu$\nResolved Flavor CR\n{era}"),
        ]
    raise ValueError(f"Unknown category: {category}")
