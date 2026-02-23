from __future__ import annotations
import logging

from .config import list_eras


def setup_logging(verbose: bool = False):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def add_era_args(parser, *, required: bool = True):
    """Add common --era, --dir, and --verbose arguments to an argparse parser."""
    parser.add_argument(
        "--era", dest="era", type=str, choices=list_eras(),
        required=required,
        help="Specify the era",
    )
    parser.add_argument(
        "--dir", dest="dir", type=str, default="",
        help="Optional subdirectory under the input & default EOS output paths",
    )
    parser.add_argument(
        "--verbose", dest="verbose", action="store_true", default=False,
        help="Enable DEBUG-level logging (shows skipped files, cache misses, etc.).",
    )


def parse_multi(opt):
    """
    Accepts a list of strings from argparse with action='append'.
    Each string may itself be a comma-separated list.
    Returns a deduped list preserving first-seen order, or None if empty.
    """
    if not opt:
        return None
    items = []
    for part in opt:
        items.extend([x.strip() for x in part.split(",") if x.strip()])
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out or None
