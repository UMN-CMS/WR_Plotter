from __future__ import annotations
import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
