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


def add_dy_arg(parser, *, multiple: bool = True):
    """Add --dy TAG[=DIR] (data/dy_samples.yaml); repeatable/comma-separated when *multiple*."""
    what = ("Drell-Yan sample(s) to use; repeat or comma-separate for several. "
            if multiple else "Drell-Yan sample to use. ")
    parser.add_argument(
        "--dy", dest="dy", action="append" if multiple else "store", default=None,
        metavar="TAG[=DIR]",
        help=what + "TAG is an analyzer --dy tag (lo_ht, lo_inc, nlo_inc, ...); the default "
             "sample is read from --dir and any other from <dir>_<TAG>_dy unless =DIR is given. "
             "Default: the analyzer's default sample (data/dy_samples.yaml).",
    )


def add_selection_args(parser):
    """Add -r/--region and -v/--variable filters."""
    parser.add_argument("--region", "-r", dest="regions", action="append", default=None,
                        help="Region name(s) or shorthand(s). Repeat or comma-separate.")
    parser.add_argument("--variable", "-v", dest="variables", action="append", default=None,
                        help="Variable name(s). Repeat or comma-separate.")


def add_output_args(parser):
    """Add --local-plots."""
    parser.add_argument("--local-plots", action="store_true",
                        help="Save plots under plots/ in the repository instead of EOS "
                             "(outputs/ when EOS is not available).")


def select_regions(era: str, requested):
    """Regions of *era*, restricted to the requested names/shorthands (exits on unknown)."""
    import sys
    from .regions import regions_for_era, expand_region_requests
    if not requested:
        return regions_for_era(era)
    try:
        return expand_region_requests(era, requested)
    except ValueError as exc:
        logging.error(str(exc))
        sys.exit(2)


def select_variables(requested):
    """Variables from data/variables.yaml, restricted to the requested names (exits on unknown)."""
    import sys
    from .variables import build_variables
    variables = build_variables()
    if not requested:
        return list(variables)
    by_name = {v.name: v for v in variables}
    unknown = [n for n in requested if n not in by_name]
    if unknown:
        logging.error("Unknown variable(s): %s. Valid: %s", unknown, sorted(by_name))
        sys.exit(2)
    return [by_name[n] for n in requested]


def dy_specs(values):
    """Flatten repeated/comma-separated --dy values (None when not given)."""
    if values is None:
        return None
    return parse_multi([values] if isinstance(values, str) else values)
