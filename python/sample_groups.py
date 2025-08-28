# python/sample_groups.py
from dataclasses import dataclass
from typing import Literal, Dict, List, Any
from pathlib import Path
import logging, yaml

@dataclass(frozen=True)
class SampleGroup:
    key: str                        # stable identifier: "dy", "ttbar", "egamma", "muon"
    run: str
    year: int
    mc_campaign: str
    color: str                      # hex color like "#5790fc"
    tlatex_alias: str               # TLatex / legend label
    samples: List[str]              # underlying dataset nicknames
    kind: Literal["data","mc"] = "mc"

    def print(self) -> None:
        logging.info(f"  - key={self.key}")
        logging.info(f"    run/year={self.run}/{self.year}  mc_campaign={self.mc_campaign}")
        logging.info(f"    kind={self.kind}  color={self.color}")
        logging.info(f"    label={self.tlatex_alias}")
        logging.info(f"    samples={','.join(self.samples)}")

def _mk_group(key: str, g: Dict[str, Any], era_defaults: Dict[str, Any]) -> SampleGroup:
    """Build a SampleGroup allowing era-level defaults & safe fallbacks."""
    return SampleGroup(
        key          = key,
        run          = g.get("run", era_defaults.get("run", "Run3")),
        year         = int(g.get("year", era_defaults.get("year", 2022))),
        mc_campaign  = g.get("mc_campaign", era_defaults.get("mc_campaign", "")),
        color        = g["color"],
        tlatex_alias = g.get("tlatex_alias", g.get("label", key)),
        samples      = list(g.get("samples", [])),
        kind         = g.get("kind", "mc"),
    )

def load_sample_groups(era: str) -> tuple[Dict[str, SampleGroup], List[str]]:
    """
    Load sample groups for an era from data/sample_groups/<era>.yaml.

    Returns:
      (groups_by_key, display_order)
    """
    cfg_path = Path("data/sample_groups") / f"{era}.yaml"
    data = yaml.safe_load(cfg_path.read_text())

    era_defaults = {
        "run": data.get("run"),
        "year": data.get("year"),
        "mc_campaign": data.get("mc_campaign", ""),
    }

    groups = { key: _mk_group(key, g, era_defaults) for key, g in data["groups"].items() }
    order = data.get("order", list(groups.keys()))
    return groups, order
