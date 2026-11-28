#!/usr/bin/env python3
"""
Make a data directory's files agree, without Kaldi.

The Python counterpart of `utils/fix_data_dir.sh`. All of the work is in
`BenNevis.utils.datadir.fix`, which needs nothing but the standard
library and is therefore tested in tier 1 -- the stage of the recipe
least likely to have a test and most likely to change a corpus silently.

Read `BenNevis/utils/datadir.py` before changing anything here: two of
Kaldi's rules are not guessable from the shell, and one of them is a
helper called with its arguments the other way round.

Usage
-----
    python3 run/fix_data_dir.py <data_dir>

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional


def _ensure_importable() -> None:
    """Make `BenNevis` importable when run as a script."""
    root = str(Path(__file__).resolve().parent.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Filter and sort a Kaldi data directory so its files "
        "agree about which utterances exist."
    )
    parser.add_argument("data_dir", type=Path, help="the data directory")
    args = parser.parse_args(argv)
    _ensure_importable()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    from BenNevis.utils.datadir import DataDirError, fix

    try:
        summary = fix(args.data_dir, report=logging.info)
    except DataDirError as failure:
        logging.error("%s", failure)
        return 1

    logging.info(
        "%s: %d utterances, %d speakers, %d dropped",
        args.data_dir,
        summary["kept"],
        summary["speakers"],
        summary["dropped"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
