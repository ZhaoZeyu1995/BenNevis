"""
Merging the per-rank files a distributed prediction leaves behind.

`run/predict.py` did this with `cat <glob> | awk '!seen[$N]++'`, which is
two shell dependencies and one latent bug. The bug is the glob: a shell
expands `ref.wrd.*.trn` in lexicographic order, so eleven ranks merge as
1, 10, 11, 2, 3, ... rather than 1, 2, 3, ... That has not bitten because
nothing downstream depends on the order -- both files are keyed by
utterance -- but it is wrong, and reproducing it in Python would be
reproducing a mistake.

The deduplication is kept exactly: first occurrence wins, later ones are
dropped. Two keys are needed, because the two files are not the same shape
-- `ref.wrd.trn` dedupes on the whole line and `output.scp` on its first
field, which is `awk '!seen[$0]++'` against `awk '!seen[$1]++'`.

Functions
---------
rank_sorted
    Per-rank paths in numeric rather than lexicographic order.
merge_unique
    Concatenate them, dropping repeats.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

__all__ = ["WHOLE_LINE", "FIRST_FIELD", "rank_sorted", "merge_unique"]

#: Dedupe on the entire line, as `awk '!seen[$0]++'` does. What
#: `ref.wrd.trn` needs: two ranks that transcribed the same utterance wrote
#: identical lines.
WHOLE_LINE: Callable[[str], str] = lambda line: line

#: Dedupe on the first whitespace-separated field, as `awk '!seen[$1]++'`
#: does. What `output.scp` needs: the utterance id is the key and the
#: archive offset after it differs between ranks.
FIRST_FIELD: Callable[[str], str] = lambda line: line.split(maxsplit=1)[0]

_NUMBER = re.compile(r"(\d+)")


def _numeric_key(path: Path):
    """Split a name into text and number runs, so `2` sorts before `10`."""
    parts = _NUMBER.split(path.name)
    return [int(part) if part.isdigit() else part for part in parts]


def rank_sorted(paths: Iterable[Path]) -> List[Path]:
    """
    Per-rank paths in numeric order.

    A shell glob gives lexicographic order, which for eleven ranks is
    1, 10, 11, 2, 3 ... This is the order a person would expect and the
    order the ranks actually ran in.

    Arguments
    ---------
    paths: Iterable[Path]
        The files.

    Returns
    -------
    List[Path]
        Sorted so that embedded numbers compare as numbers.
    """
    return sorted(paths, key=_numeric_key)


def merge_unique(
    paths: Sequence[Path],
    key: Optional[Callable[[str], str]] = None,
) -> str:
    """
    Concatenate files, keeping the first line per key.

    Arguments
    ---------
    paths: Sequence[Path]
        The files, which are read in the order given -- pass them through
        `rank_sorted` first.
    key: Optional[Callable[[str], str]]
        What makes a line a duplicate. Defaults to the whole line.

    Returns
    -------
    str
        The merged text, newline-terminated, or empty if there was nothing.
    """
    if key is None:
        key = WHOLE_LINE
    seen = set()
    kept: List[str] = []
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            identity = key(line)
            if identity in seen:
                continue
            seen.add(identity)
            kept.append(line)
    return "\n".join(kept) + "\n" if kept else ""
