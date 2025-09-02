# python/config.py
from __future__ import annotations
from functools import lru_cache
from typing import Any, Dict, Mapping, MutableMapping
from pathlib import Path

from .io import data_path, read_yaml, read_json

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> dict:
    """
    Recursively merge mapping 'override' into 'base' (without mutating input).
    Later keys win. Lists are replaced (not concatenated) on purpose.
    """
    out: dict = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(dict(out[k]), v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


# -----------------------------------------------------------------------------
# Lumi / Eras
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _lumi_table() -> Dict[str, Any]:
    """
    Load lumi table once. Supports either JSON (current) or YAML (future).
    """
    # Prefer YAML if present; fall back to JSON for backward-compat.
    yaml_path = data_path("lumi.yaml")
    json_path = data_path("lumi.json")
    if yaml_path.exists():
        return read_yaml(yaml_path) or {}
    return read_json(json_path) or {}

def list_eras() -> list[str]:
    """Sorted list of known eras (keys of lumi table)."""
    return sorted(_lumi_table().keys())

def load_lumi(era: str) -> Dict[str, Any]:
    """
    Returns dict with keys: run, year, lumi, and optionally sub_eras, sub_lumis.
    Raises KeyError/ValueError with helpful messages if something is wrong.
    """
    table = _lumi_table()
    if era not in table:
        raise KeyError(
            f"Era '{era}' not found in lumi table. Known eras: {', '.join(sorted(table.keys()))}"
        )
    info = dict(table[era])  # copy to avoid accidental mutation

    for k in ("run", "year", "lumi"):
        if k not in info:
            raise ValueError(f"lumi entry for '{era}' missing required key '{k}'")

    # If era is a combination, verify lengths of sub_eras/sub_lumis
    if ("sub_eras" in info) or ("sub_lumis" in info):
        subs = info.get("sub_eras", [])
        lums = info.get("sub_lumis", [])
        if len(subs) != len(lums):
            raise ValueError(
                f"lumi entry for '{era}' has mismatched lengths: sub_eras={len(subs)} vs sub_lumis={len(lums)}"
            )
    return info


# -----------------------------------------------------------------------------
# Plot settings (YAML with base + era-specific overrides)
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_plot_settings(era_or_path: str) -> Dict[str, Any]:
    """
    Load plotting config.

    Accepts either:
      - an explicit YAML path (endswith .yaml/.yml), OR
      - an era key (e.g., 'Run3Summer22', '2022')

    Merge order when an era is given:
      base.yaml  <-  <era>.yaml
    """
    # If the caller passed a YAML path, just load it
    if era_or_path.endswith((".yaml", ".yml")):
        return read_yaml(era_or_path) or {}

    # Otherwise treat it as an era key and resolve files under data/plot_settings
    base_path = data_path("plot_settings", "base.yaml")
    era_path  = data_path("plot_settings", f"{era_or_path}.yaml")

    base_cfg: Dict[str, Any] = read_yaml(base_path) if base_path.exists() else {}
    if not era_path.exists():
        # It’s fine to have only base.yaml; warn at call sites if needed
        return base_cfg

    era_cfg: Dict[str, Any] = read_yaml(era_path) or {}
    return _deep_merge(base_cfg, era_cfg)


# -----------------------------------------------------------------------------
# Sample groups (kept here if you like, or in python/sample_groups.py)
# -----------------------------------------------------------------------------
# If you prefer to keep sample group loading in a separate module, delete this
# section and keep your existing python/sample_groups.py loader.

@lru_cache(maxsize=None)
def load_sample_groups_raw(era: str) -> Dict[str, Any]:
    """
    Raw load of sample group YAML for a given era, with optional base merge:
      data/sample_groups/base.yaml  <-  data/sample_groups/<era>.yaml
    """
    base_path = data_path("sample_groups", "base.yaml")
    era_path  = data_path("sample_groups", f"{era}.yaml")

    base_cfg: Dict[str, Any] = read_yaml(base_path) if base_path.exists() else {}
    if not era_path.exists():
        return base_cfg

    era_cfg: Dict[str, Any] = read_yaml(era_path) or {}
    return _deep_merge(base_cfg, era_cfg)


# -----------------------------------------------------------------------------
# K-factors
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_kfactors(path: Path | None = None) -> Dict[str, Any]:
    """
    Load k-factors YAML, defaulting to data/kfactors.yaml.
    Allows override via explicit 'path' for testing.
    """
    cfg_path = path or data_path("kfactors.yaml")
    if not Path(cfg_path).exists():
        return {}
    return read_yaml(cfg_path) or {}

def get_kfactor(scales: Mapping[str, Mapping[str, Any]],
                era: str, sample: str, default: float = 1.0) -> float:
    """
    Fetch a k-factor with precedence:
      era->sample → era->_default → ALL->sample → ALL->_default → default
    """
    era_map = scales.get(era, {})
    all_map = scales.get("ALL", {})

    if sample in era_map:        return float(era_map[sample])
    if "_default" in era_map:    return float(era_map["_default"])
    if sample in all_map:        return float(all_map[sample])
    if "_default" in all_map:    return float(all_map["_default"])
    return float(default)

def index_plot_settings(plot_settings: dict) -> tuple[dict, dict]:
    common = plot_settings.get("common_variables", {})
    regions = {k: v for k, v in plot_settings.items() if k != "common_variables"}
    return regions, common

def get_var_cfg(region_cfgs, common_vars, region_name, var_name):
    reg = region_cfgs.get(region_name, {})
    return reg.get(var_name, common_vars.get(var_name))
