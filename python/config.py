# python/config.py
from pathlib import Path
from typing import Dict, Any
import json

def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"

def load_lumi(era: str) -> Dict[str, Any]:
    """
    Returns dict with keys: run, year, lumi, and optionally sub_eras, sub_lumis.
    Raises KeyError/ValueError with helpful messages if something is wrong.
    """
    path = _data_dir() / "lumi.json"
    with open(path) as f:
        table = json.load(f)

    if era not in table:
        raise KeyError(f"Era '{era}' not found in {path}. Known eras: {', '.join(sorted(table.keys()))}")

    info = table[era]

    for k in ("run", "year", "lumi"):
        if k not in info:
            raise ValueError(f"data/lumi.json entry for '{era}' missing '{k}'")

    if ("sub_eras" in info) or ("sub_lumis" in info):
        subs = info.get("sub_eras", [])
        lums = info.get("sub_lumis", [])
        if len(subs) != len(lums):
            raise ValueError(f"data/lumi.json entry for '{era}' has mismatched sub_eras/sub_lumis lengths")

    return info
