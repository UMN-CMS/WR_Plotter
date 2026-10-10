from __future__ import annotations
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

from .paths import data_path, read_yaml

logger = logging.getLogger(__name__)

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
    """Load lumi table once from data/lumi.yaml."""
    yaml_path = data_path("lumi.yaml")
    return read_yaml(yaml_path) or {}

def list_eras() -> list[str]:
    """Sorted list of known eras (keys of lumi table, excluding internal anchor keys)."""
    return sorted(k for k in _lumi_table().keys() if not k.startswith("_"))

def load_lumi(era: str) -> Dict[str, Any]:
    """
    Returns dict with keys: run, year, lumi, and optionally sub_eras, sub_lumis.

    A combination (``sub_eras``) gets ``sub_lumis`` from its eras and ``lumi``
    as their sum; values written in the YAML must agree with them.  Its eras
    must be single eras with one centre-of-mass energy.
    Raises KeyError/ValueError with helpful messages if something is wrong.
    """
    table = _lumi_table()
    if era not in table:
        raise KeyError(
            f"Era '{era}' not found in lumi table. Known eras: {', '.join(sorted(table.keys()))}"
        )
    info = dict(table[era])  # copy to avoid accidental mutation

    if "sub_eras" in info or "sub_lumis" in info:
        subs = list(info.get("sub_eras") or [])
        if not subs:
            raise ValueError(f"lumi entry for '{era}' has sub_lumis but no sub_eras")
        sub_infos = []
        for se in subs:
            if se not in table:
                raise KeyError(
                    f"sub_era '{se}' referenced by '{era}' not found in lumi table. "
                    f"Known eras: {', '.join(sorted(table.keys()))}"
                )
            if "sub_eras" in table[se]:
                raise ValueError(f"'{era}' lists the combination '{se}' as a sub_era; list its eras instead")
            sub_infos.append(table[se])
        lumis = [float(s["lumi"]) for s in sub_infos]
        given = info.get("sub_lumis")
        if given is not None and (len(given) != len(lumis)
                                  or any(abs(float(g) - l) > 1e-6 * l for g, l in zip(given, lumis))):
            raise ValueError(f"lumi entry for '{era}': sub_lumis {given} disagree with its eras {lumis}")
        if "lumi" in info and abs(float(info["lumi"]) - sum(lumis)) > 1e-6 * sum(lumis):
            raise ValueError(f"lumi entry for '{era}': lumi {info['lumi']} is not the sum of its eras "
                             f"({sum(lumis):g})")
        coms = {float(s.get("com", 13.6)) for s in sub_infos}
        if len(coms) > 1:
            raise ValueError(f"'{era}' combines eras with different centre-of-mass energies: {sorted(coms)}")
        info["sub_lumis"] = lumis
        info["lumi"] = sum(lumis)
        info.setdefault("com", coms.pop())

    for k in ("run", "year", "lumi"):
        if k not in info:
            raise ValueError(f"lumi entry for '{era}' missing required key '{k}'")
    return info


def era_label(era: str) -> str:
    """Name of an era on plots: its ``label`` in data/lumi.yaml, else the era key."""
    entry = _lumi_table().get(era)
    return str(entry.get("label", era)) if isinstance(entry, dict) else era


# -----------------------------------------------------------------------------
# Plot settings (YAML with base + era-specific overrides)
# -----------------------------------------------------------------------------

PLOT_SETTINGS_DEFAULT = "default"


@lru_cache(maxsize=None)
def load_plot_settings(era_or_path: str) -> Dict[str, Any]:
    """
    Load plotting config: an explicit YAML path, or an era key.

    An era reads ``data/plot_settings/<era>.yaml``; without one, the file named by
    its ``plot_settings`` entry in data/lumi.yaml, else ``default.yaml``.  A name
    that is neither a file nor an era gives an empty config.  A file may start
    from another one with a top-level ``inherit: <name>`` and override any
    (region, variable, key) of it, so an era lists only what differs.
    """
    if era_or_path.endswith((".yaml", ".yml")):
        return _read_plot_settings(Path(era_or_path), ())
    path = _plot_settings_path(era_or_path)
    if path is None:
        logger.debug("No plot settings for %s; returning empty config.", era_or_path)
        return {}
    return _read_plot_settings(path, ())


def _plot_settings_path(name: str, seen: tuple = ()) -> Path | None:
    path = data_path("plot_settings", f"{name}.yaml")
    if path.exists():
        return path
    entry = _lumi_table().get(name)
    if not isinstance(entry, dict):
        return None
    alias = entry.get("plot_settings")
    if alias and alias not in seen + (name,):
        return _plot_settings_path(alias, seen + (name,))
    return data_path("plot_settings", f"{PLOT_SETTINGS_DEFAULT}.yaml")


def _read_plot_settings(path: Path, chain: tuple) -> Dict[str, Any]:
    if path in chain:
        raise ValueError("Plot settings inherit in a cycle: "
                         + " -> ".join(p.name for p in chain + (path,)))
    cfg = read_yaml(path) or {}
    parent = cfg.pop("inherit", None)
    if parent is None:
        return cfg
    parent_path = data_path("plot_settings", f"{parent}.yaml")
    if not parent_path.exists():
        raise FileNotFoundError(f"{path} inherits '{parent}', but {parent_path} does not exist")
    return _deep_merge(_read_plot_settings(parent_path, chain + (path,)), cfg)



def index_plot_settings(plot_settings: dict) -> tuple[dict, dict]:
    common = plot_settings.get("common_variables", {})
    regions = {k: v for k, v in plot_settings.items() if k != "common_variables"}
    return regions, common


# Valid keys for per-variable plot configuration entries
_VALID_VAR_CFG_KEYS = {"rebin", "rebin_variable", "xlim", "ylim", "ratio_ylim"}


def validate_var_cfg(var_name: str, cfg: dict, region_name: str = "") -> None:
    """Warn about unrecognized keys in a variable's plot config.

    This catches typos like 'rebin_varaible' that would otherwise be
    silently ignored.
    """
    unknown = set(cfg.keys()) - _VALID_VAR_CFG_KEYS
    if unknown:
        loc = f" in region '{region_name}'" if region_name else ""
        logger.warning(
            f"Unknown plot-setting key(s) for variable '{var_name}'{loc}: "
            f"{sorted(unknown)}. Valid keys: {sorted(_VALID_VAR_CFG_KEYS)}"
        )


def get_var_cfg(region_cfgs: dict, common_vars: dict,
                region_name: str, var_name: str) -> dict | None:
    """Plot settings of *var_name* in *region_name*, or None if it is not plotted there.

    The region's own entry, else (resolved regions only) the ``common_variables``
    entry: that block holds resolved-topology variables (AK4 jets, dijet, 3- and
    4-object), which the analyzer does not fill in boosted regions; the boosted
    blocks list every variable they plot.
    """
    reg = region_cfgs.get(region_name, {})
    cfg = reg.get(var_name)
    if cfg is None and "boosted" not in region_name.lower():   # same rule as Region.is_boosted
        cfg = common_vars.get(var_name)
    if cfg is not None:
        validate_var_cfg(var_name, cfg, region_name)
    return cfg

def configured_variables(region_cfgs: dict, common_vars: dict,
                         region_name: str, all_variables) -> list[tuple]:
    """Return (Variable, cfg_dict) pairs that have plot settings for this region."""
    result = []
    for v in all_variables:
        cfg = get_var_cfg(region_cfgs, common_vars, region_name, v.name)
        if cfg is not None:
            result.append((v, cfg))
    return result


# -----------------------------------------------------------------------------
# Default signal samples
# -----------------------------------------------------------------------------

def default_signals(era_info: dict, *, boosted: bool = False,
                    for_cutflow: bool = False) -> list[str]:
    """Return default signal sample names from era_info with fallbacks."""
    if for_cutflow:
        key = "cutflow_signals_boosted" if boosted else "cutflow_signals_resolved"
        fallback = (["WR2000_N100", "WR4000_N100", "WR6000_N100"] if boosted
                    else ["WR2000_N1100", "WR4000_N2100", "WR6000_N3100"])
    else:
        key = "default_signal_boosted" if boosted else "default_signal_resolved"
        fallback = ["signal_WR4000_N100"] if boosted else ["signal_WR4000_N2100"]
    result = era_info.get(key)
    if result is None:
        logger.warning(
            "Era info missing key %r; using hardcoded fallback %r. "
            "Add this key to lumi.yaml to suppress this warning.",
            key, fallback,
        )
        result = fallback
    return [result] if isinstance(result, str) else list(result)


# -----------------------------------------------------------------------------
# Region shorthands
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_region_shorthands() -> Dict[str, list[str]]:
    """Load region shorthand aliases from data/region_shorthands.yaml."""
    path = data_path("region_shorthands.yaml")
    if not path.exists():
        return {}
    return read_yaml(path) or {}


# -----------------------------------------------------------------------------
# Systematics
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_systematics() -> list[str]:
    """Load the list of systematic names from data/systematics.yaml."""
    path = data_path("systematics.yaml")
    if not path.exists():
        return []
    data = read_yaml(path) or {}
    return list(data.get("systematics", []))


# -----------------------------------------------------------------------------
# Drell-Yan samples and cutflow presentation
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_dy_samples() -> Dict[str, Any]:
    """Load data/dy_samples.yaml: {"default": tag, "samples": {tag: {label, color, datasets}}}."""
    data = read_yaml(data_path("dy_samples.yaml")) or {}
    samples = data.get("samples") or {}
    default = data.get("default")
    if default not in samples:
        raise ValueError(f"dy_samples.yaml: default {default!r} is not one of {sorted(samples)}")
    return {"default": default, "samples": samples}


@lru_cache(maxsize=None)
def load_cutflow_config() -> Dict[str, Any]:
    """Load data/cutflows.yaml: {"data_stream": {topology: {channel: stream}}, "labels": {...}}."""
    data = read_yaml(data_path("cutflows.yaml")) or {}
    return {"data_stream": data.get("data_stream") or {}, "labels": data.get("labels") or {}}
