# python/sample_groups.py
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Dict, List, Any
import logging

from .io import data_path, read_yaml

@dataclass(frozen=True)
class SampleGroup:
    key: str                        # stable identifier: "dy", "ttbar", "egamma", "muon"
    color: str                      # hex like "#5790fc"
    tlatex_alias: str               # TLatex / legend label
    samples: List[str]              # underlying dataset nicknames
    kind: Literal["data", "mc"] = "mc"
    stack_position: str | None = None  # "top" or "bottom"; None = use YAML order

    def print(self) -> None:
        logging.info(f"  - key={self.key}")
        logging.info(f"    kind={self.kind}  color={self.color}")
        logging.info(f"    label={self.tlatex_alias}")
        logging.info(f"    samples={','.join(self.samples)}")


# ---------------------------- internal helpers ---------------------------- #

_ALLOWED_KEYS = {"color", "tlatex_alias", "label", "samples", "kind", "stack_position"}

def _mk_group(key: str, g: Dict[str, Any]) -> SampleGroup:
    # warn on unknown keys (helps catch typos in YAML)
    unknown = set(g.keys()) - _ALLOWED_KEYS
    if unknown:
        logging.warning(f"[sample_groups] group '{key}' has unknown keys: {sorted(unknown)}")

    # required fields
    if "color" not in g:
        raise ValueError(f"[sample_groups] group '{key}' missing required field 'color'")
    if "samples" not in g or not g["samples"]:
        logging.warning(f"[sample_groups] group '{key}' has empty 'samples' list")

    label = g.get("tlatex_alias", g.get("label", key))

    raw_pos = g.get("stack_position")
    if raw_pos is not None and raw_pos not in ("top", "bottom"):
        logging.warning(
            "[sample_groups] group '%s' has unknown stack_position %r — ignored. "
            "Use 'top' or 'bottom'.",
            key, raw_pos,
        )
    stack_position = raw_pos if raw_pos in ("top", "bottom") else None

    return SampleGroup(
        key            = key,
        color          = str(g["color"]),
        tlatex_alias   = str(label),
        samples        = list(g.get("samples", [])),
        kind           = "data" if g.get("kind", "mc") == "data" else "mc",
        stack_position = stack_position,
    )


# ----------------------------- public loader ------------------------------ #

@lru_cache(maxsize=None)
def load_sample_groups() -> tuple[Dict[str, SampleGroup], List[str]]:
    """
    Load sample groups from data/sample_groups.yaml.

    Returns:
      (groups_by_key, display_order)
    """
    path = data_path("sample_groups.yaml")
    data: Dict[str, Any] = read_yaml(path) if path.exists() else {}

    groups_cfg = data.get("groups", {})
    if not isinstance(groups_cfg, dict) or not groups_cfg:
        raise ValueError(f"[sample_groups] no 'groups' mapping found in {path}")

    groups: Dict[str, SampleGroup] = {
        key: _mk_group(key, g) for key, g in groups_cfg.items()
    }

    order = data.get("order", list(groups.keys()))
    # drop any keys listed in order that aren't defined (typo guard)
    order = [k for k in order if k in groups]

    return groups, order
