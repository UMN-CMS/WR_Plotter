# python/io.py
from __future__ import annotations
import os, json, subprocess, tempfile
from pathlib import Path
from typing import Any, Optional

try:
    import yaml  # ensure pyyaml in requirements.txt
except Exception:
    yaml = None


# ── Repo & data ────────────────────────────────────────────────────────────────

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def data_path(*parts: str) -> Path:
    return repo_root() / "data" / Path(*parts)

def read_yaml(path: str | Path) -> Any:
    if yaml is None:
        raise RuntimeError("pyyaml not installed")
    with open(path, "r") as f:
        return yaml.safe_load(f)

def read_json(path: str | Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


# ── EOS utilities ──────────────────────────────────────────────────────────────

def eos_endpoint() -> str:
    """Hostname for xrdfs/xrdcp (override via $EOS_ENDPOINT)."""
    return os.environ.get("EOS_ENDPOINT", "eosuser.cern.ch")

def resolve_eos_user() -> str:
    """Return segment like 'w/wijackso' for /eos/user/<seg>/..."""
    if (p := os.environ.get("EOSUSER_PATH")):
        return p.strip("/")
    if (u := os.environ.get("EOSUSER")):
        return f"{u[0]}/{u}"
    mapping = {"bjackson": "w/wijackso"}
    user = os.environ.get("USER", "user")
    return mapping.get(user, f"{user[0]}/{user}")

def eos_base(run: str, year: str, era: str, subdir: Optional[str] = None) -> Path:
    """Build an EOS path prefix (can override root via $EOS_BASE)."""
    base_override = os.environ.get("EOS_BASE")  # e.g. /eos/user/w/wijackso
    root = Path(base_override) if base_override else Path(f"/eos/user/{resolve_eos_user()}")
    base = root / run / year / era
    return base / subdir if subdir else base

def is_eos_path(path: str | Path) -> bool:
    """Heuristic: treat paths under /eos as EOS destinations."""
    return str(path).startswith("/eos/")

def eos_mkdir_p(dir_path: str | Path) -> None:
    """
    Recursively create a directory on EOS using xrdfs.
    Equivalent to: xrdfs <host> mkdir -p <dir_path>
    """
    host = eos_endpoint()
    dir_path = str(dir_path)
    # xrdfs doesn't mkdir the final leaf if parents absent unless -p used
    subprocess.run(["xrdfs", host, "mkdir", "-p", dir_path], check=True)

def eos_upload(local_file: str | Path, eos_dest: str | Path) -> None:
    """
    Copy a local file to EOS with xrdcp.
    """
    host = eos_endpoint()
    local_file = str(local_file)
    eos_dest = str(eos_dest)
    subprocess.run(["xrdcp", "-f", local_file, f"root://{host}/{eos_dest}"], check=True)


# ── Output location (EOS vs local) ─────────────────────────────────────────────

def _eos_available() -> bool:
    if os.environ.get("FORCE_LOCAL"):
        return False
    if os.environ.get("FORCE_EOS"):
        return True
    base_override = os.environ.get("EOS_BASE")
    probe = Path(base_override) if base_override else Path("/eos")
    return probe.exists()

def output_dir(run: str, year: str, era: str, subdir: Optional[str] = None,
               *, prefer_local: bool = False) -> Path:
    """
    Choose an output root and ensure it exists:
      - EOS when mounted (or FORCE_EOS=1),
      - otherwise {repo}/outputs/...
    NOTE: For EOS, we don't mkdir with Path().mkdir(); we use xrdfs on demand in save_figure().
    """
    if prefer_local or not _eos_available():
        base = repo_root() / "outputs" / run / year / era
        out = base / subdir if subdir else base
        out.mkdir(parents=True, exist_ok=True)
        return out
    else:
        return eos_base(run, year, era, subdir)


# ── Save helpers ───────────────────────────────────────────────────────────────

def ensure_local_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def save_figure(fig, dest_path: str | Path, *, force_local: bool = False, fmt: str = "pdf") -> None:
    """
    Save a Matplotlib figure to either local filesystem or EOS, based on dest_path.
      - If path starts with /eos and not force_local: use xrdfs mkdir -p + xrdcp
      - Otherwise: save locally (mkdir -p)
    """
    dest_path = Path(dest_path)
    if is_eos_path(dest_path) and not force_local:
        # ensure EOS dir
        eos_dir = dest_path.parent
        eos_mkdir_p(eos_dir)
        # temp save then upload
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}") as tmp:
            fig.savefig(tmp.name, format=fmt)
            tmp.flush()
            eos_upload(tmp.name, dest_path)
    else:
        ensure_local_dir(dest_path.parent)
        fig.savefig(dest_path, format=fmt)
