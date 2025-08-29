# python/sample_groups.py
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Dict, List, Any, Mapping, MutableMapping
import logging

from .io import data_path, read_yaml

@dataclass(frozen=True)
class SampleGroup:
    key: str                        # stable identifier: "dy", "ttbar", "egamma", "muon"
    run: str
    year: int
    mc_campaign: str
    color: str                      # hex like "#5790fc"
    tlatex_alias: str               # TLatex / legend label
    samples: List[str]              # underlying dataset nicknames
    kind: Literal["data", "mc"] = "mc"

    def print(self) -> None:
        logging.info(f"  - key={self.key}")
        logging.info(f"    run/year={self.run}/{self.year}  mc_campaign={self.mc_campaign}")
        logging.info(f"    kind={self.kind}  color={self.color}")
        logging.info(f"    label={self.tlatex_alias}")
        logging.info(f"    samples={','.join(self.samples)}")


# ---------------------------- internal helpers ---------------------------- #

_ALLOWED_KEYS = {"run", "year", "mc_campaign", "color", "tlatex_alias", "label", "samples", "kind"}

def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> dict:
    out: dict = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(dict(out[k]), v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out

def _mk_group(key: str, g: Dict[str, Any], era_defaults: Dict[str, Any]) -> SampleGroup:
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

    return SampleGroup(
        key          = key,
        run          = g.get("run", era_defaults.get("run", "Run3")),
        year         = int(g.get("year", era_defaults.get("year", 2022))),
        mc_campaign  = g.get("mc_campaign", era_defaults.get("mc_campaign", "")),
        color        = str(g["color"]),
        tlatex_alias = str(label),
        samples      = list(g.get("samples", [])),
        kind         = "data" if g.get("kind", "mc") == "data" else "mc",
    )


# ----------------------------- public loader ------------------------------ #

@lru_cache(maxsize=None)
def load_sample_groups(era: str) -> tuple[Dict[str, SampleGroup], List[str]]:
    """
    Load sample groups for an era with optional base overrides:
      data/sample_groups/base.yaml  <-  data/sample_groups/<era>.yaml

    Returns:
      (groups_by_key, display_order)
    """
    base_path = data_path("sample_groups", "base.yaml")
    era_path  = data_path("sample_groups", f"{era}.yaml")

    base_cfg: Dict[str, Any] = read_yaml(base_path) if base_path.exists() else {}
    if not era_path.exists():
        logging.warning(f"[sample_groups] era file missing: {era_path} (using base only)")
        data = base_cfg
    else:
        era_cfg: Dict[str, Any] = read_yaml(era_path) or {}
        data = _deep_merge(base_cfg, era_cfg)

    # era-level defaults
    era_defaults = {
        "run": data.get("run"),
        "year": data.get("year"),
        "mc_campaign": data.get("mc_campaign", ""),
    }

    groups_cfg = data.get("groups", {})
    if not isinstance(groups_cfg, dict) or not groups_cfg:
        raise ValueError(f"[sample_groups] no 'groups' mapping found in {era_path}")

    groups: Dict[str, SampleGroup] = {
        key: _mk_group(key, g, era_defaults) for key, g in groups_cfg.items()
    }

    order = data.get("order", list(groups.keys()))
    # drop any keys listed in order that aren't defined (typo guard)
    order = [k for k in order if k in groups]

    return groups, order
