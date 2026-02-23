from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import logging

from .paths import data_path, read_yaml

@dataclass(frozen=True)
class Variable:
    name: str
    tlatex_alias: str
    unit: str = ""

    # default: regular binning (uniform)
    nbins: int | None = None
    xmin: float | None = None
    xmax: float | None = None

    def label(self) -> str:
        """Return formatted axis label: 'alias [unit]' or just 'alias' if no unit.

        Note: make_stackplots.py accesses tlatex_alias directly; this method is
        kept as a utility for downstream scripts that need a pre-formatted string.
        """
        return f"{self.tlatex_alias} [{self.unit}]" if self.unit else self.tlatex_alias

    def print(self) -> None:
        logging.info(
            f"Variable(name={self.name}, tlatex={self.tlatex_alias}, unit={self.unit}, "
            f"nbins={self.nbins}, range=({self.xmin},{self.xmax}))"
        )

@lru_cache(maxsize=None)
def build_variables() -> list[Variable]:
    """Load variable definitions from data/variables.yaml."""
    path = data_path("variables.yaml")
    data = read_yaml(path) or {}
    entries = data.get("variables", [])
    return [
        Variable(
            name=e["name"],
            tlatex_alias=e.get("label", e["name"]),
            unit=e.get("unit", ""),
            nbins=e.get("nbins"),
            xmin=e.get("xmin"),
            xmax=e.get("xmax"),
        )
        for e in entries
    ]
