# python/util.py
import os
from pathlib import Path
from typing import Optional
import subprocess

def resolve_eos_user() -> str:
    """
    Returns the EOS path segment like 'w/wijackso'.
    Priority:
      1) $EOSUSER_PATH (full 'w/wijackso' style)
      2) $EOSUSER (username like 'wijackso') -> first letter + username
      3) map local $USER -> explicit mapping dict
      4) fallback: local $USER (first letter + same name)
    """
    # 1) explicit full path segment
    if (p := os.environ.get("EOSUSER_PATH")):
        return p.strip("/")

    # 2) username only
    if (u := os.environ.get("EOSUSER")):
        return f"{u[0]}/{u}"

    # 3) site-specific mapping (add usernames here (LPC on left, CERN on right))
    mapping = {
        "bjackson": "w/wijackso",
    }
    user = os.environ.get("USER", "user")
    if user in mapping:
        return mapping[user]

    # 4) generic fallback
    return f"{user[0]}/{user}"

def build_output_path(run: str, year: str, era: str, subdir: Optional[str], eos_user: str) -> Path:
    base = Path(f"/eos/user/{eos_user}/{run}/{year}/{era}")
    return base / subdir if subdir else base

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]  # points to repo top
