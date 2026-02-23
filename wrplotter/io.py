# python/io.py
from __future__ import annotations
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

try:
    import yaml  # ensure pyyaml in requirements.txt
except ImportError:
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


# ── Input directories for era ──────────────────────────────────────────────────

def input_dirs_for_era(era: str, working_dir: str | Path,
                       subdir: str = "") -> tuple[list[Path], list[float]]:
    """Resolve rootfile input directories and per-directory luminosities for an era.

    For combined eras (with sub_eras in lumi config), returns one directory
    per sub-era. Otherwise returns a single directory for the era itself.

    Returns (input_dirs, input_lumis).
    """
    from .config import load_lumi

    working_dir = Path(working_dir)
    info = load_lumi(era)
    run = info["run"]
    year = info["year"]
    lumi = info["lumi"]

    if "sub_eras" in info:
        input_dirs = []
        for se in info["sub_eras"]:
            se_info = load_lumi(se)
            d = working_dir / "rootfiles" / se_info["run"] / se_info["year"] / se
            if subdir:
                d = d / subdir
            input_dirs.append(d)
        input_lumis = list(info["sub_lumis"])
    else:
        base = working_dir / "rootfiles" / run / year / era
        if subdir:
            base = base / subdir
        input_dirs = [base]
        input_lumis = [lumi]

    return input_dirs, input_lumis


# ── EOS utilities ──────────────────────────────────────────────────────────────
#
# Environment variables for EOS configuration:
#   EOS_ENDPOINT   – xrdfs/xrdcp hostname (default: eosuser.cern.ch)
#   EOSUSER_PATH   – full user segment, e.g. "w/wijackso"
#   EOSUSER        – CERN username if it differs from $USER, e.g. "wijackso"
#                    (builds segment as first-char/username)
#   EOS_BASE       – override the entire /eos/user/<seg> root
#   FORCE_EOS      – set to 1 to use EOS even if /eos is not mounted
#   FORCE_LOCAL    – set to 1 to always write locally instead of EOS

def eos_endpoint() -> str:
    """Hostname for xrdfs/xrdcp (override via $EOS_ENDPOINT)."""
    return os.environ.get("EOS_ENDPOINT", "eosuser.cern.ch")

def resolve_eos_user() -> str:
    """Return segment like 'w/wijackso' for /eos/user/<seg>/..."""
    if (p := os.environ.get("EOSUSER_PATH")):
        return p.strip("/")
    if (u := os.environ.get("EOSUSER")):
        return f"{u[0]}/{u}"
    user = os.environ.get("USER", "user")
    return f"{user[0]}/{user}"

def eos_base(run: str, year: str, era: str, subdir: Optional[str] = None) -> Path:
    """Build an EOS path prefix (can override root via $EOS_BASE)."""
    base_override = os.environ.get("EOS_BASE")  # e.g. /eos/user/w/wijackso
    root = Path(base_override) if base_override else Path(f"/eos/user/{resolve_eos_user()}")
    base = root / run / year / era
    return base / subdir if subdir else base

def is_eos_path(path: str | Path) -> bool:
    """Heuristic: treat paths under /eos as EOS destinations."""
    return str(path).startswith("/eos/")

def _eos_timeout() -> int:
    """Timeout in seconds for EOS subprocess calls (override via $EOS_TIMEOUT)."""
    return int(os.environ.get("EOS_TIMEOUT", "120"))


def eos_mkdir_p(dir_path: str | Path, *,
                timeout: int | None = None,
                _runner=subprocess.run) -> None:
    """
    Recursively create a directory on EOS using xrdfs.
    Equivalent to: xrdfs <host> mkdir -p <dir_path>

    Args:
        timeout: Seconds before the command is killed (default: $EOS_TIMEOUT or 120).
        _runner: Callable used to invoke the subprocess (injectable for testing).
    """
    host = eos_endpoint()
    dir_path = str(dir_path)
    _timeout = timeout if timeout is not None else _eos_timeout()
    try:
        # xrdfs doesn't mkdir the final leaf if parents absent unless -p used
        _runner(["xrdfs", host, "mkdir", "-p", dir_path], check=True, timeout=_timeout)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            raise RuntimeError(
                f"xrdfs mkdir timed out after {_timeout}s for {dir_path!r}. "
                "Check your grid proxy (voms-proxy-init) and network connectivity."
            ) from exc
        raise RuntimeError(
            f"xrdfs mkdir failed for {dir_path!r}: {exc}. "
            "Check your grid proxy (voms-proxy-init) and network connectivity."
        ) from exc


def eos_upload(local_file: str | Path, eos_dest: str | Path, *,
               timeout: int | None = None,
               _runner=subprocess.run) -> None:
    """
    Copy a local file to EOS with xrdcp.

    Args:
        timeout: Seconds before the command is killed (default: $EOS_TIMEOUT or 120).
        _runner: Callable used to invoke the subprocess (injectable for testing).
    """
    host = eos_endpoint()
    local_file = str(local_file)
    eos_dest = str(eos_dest)
    _timeout = timeout if timeout is not None else _eos_timeout()
    try:
        _runner(["xrdcp", "-f", local_file, f"root://{host}/{eos_dest}"],
                check=True, timeout=_timeout)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.TimeoutExpired):
            raise RuntimeError(
                f"xrdcp timed out after {_timeout}s uploading {local_file!r} to {eos_dest!r}. "
                "Check your grid proxy (voms-proxy-init) and network connectivity."
            ) from exc
        raise RuntimeError(
            f"xrdcp failed uploading {local_file!r} to {eos_dest!r}: {exc}. "
            "Check your grid proxy (voms-proxy-init) and network connectivity."
        ) from exc


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
