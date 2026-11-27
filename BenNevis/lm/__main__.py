"""
A command line for the language-model and lexicon builders.

`python3 -m BenNevis.topos` already exists so a shell script need not know
how a topology is expanded; this is its counterpart for the lexicon, and it
exists for one reason: `prepare_topo.sh` can now write `k2/L.fst.txt` into
the language directory, which is what takes OpenFST off the **training**
path.

Before this, `Lang.load_lexicon` read the lexicon FST by shelling out to
`fstprint {lang_dir}/L.fst`, and `compile_training_graph` genuinely needs
the result -- so a first training run required OpenFST and Kaldi's compiled
`L.fst`, whatever else had been ported. The language directory already
carries `k2/topo.yaml` so that the topology need not be read back out of a
compiled FST; `k2/L.fst.txt` is the same move for the lexicon.

**Rebuilding it rather than converting it is safe because it is
byte-exact.** `build_lexicon_fst_text` reproduces Kaldi's `L.fst` for both
corpora -- 1,539,598 lines from mini-librispeech's 206,510-entry lexicon,
and yesno's four words -- and `tests/reference/test_corpora.py` gates that
by digest against the capture. So this is not an approximation of Kaldi's
output; it is Kaldi's output, computed in Python.

The settings have to match the `prepare_lang.sh` call that built the
directory, exactly as stage 10's do, and getting them wrong builds a
lexicon that is **wrong rather than broken**:
`tests/recipes/test_lexicon_settings.py` reads them out of each recipe and
checks them against each other.

Usage
-----
    python3 -m BenNevis.lm lexicon-fst <lexicon.txt> <words.txt> \\
        <phones.txt> [--sil-prob P] [--position-dependent-phones]

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional


def read_table(path: Path) -> Dict[str, int]:
    """
    An OpenFST symbol table as a mapping.

    Arguments
    ---------
    path: Path
        `words.txt` or `phones.txt`.

    Returns
    -------
    Dict[str, int]
        Symbol to integer id.
    """
    table: Dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2:
            table[fields[0]] = int(fields[1])
    return table


def optional_silence(lexicon: Path, override: Optional[str]) -> str:
    """
    Which phone is the optional silence.

    Read from `optional_silence.txt` beside the lexicon when it is there,
    because that is where `prepare_dict.sh` puts it and because the
    reference gate reads it the same way -- a recipe whose silence phone is
    not `SIL` would otherwise build a lexicon that composes and decodes and
    is quietly wrong.

    Arguments
    ---------
    lexicon: Path
        The lexicon, whose directory is searched.
    override: Optional[str]
        An explicit phone, which wins if given.

    Returns
    -------
    str
        The phone name.
    """
    if override:
        return override
    beside = lexicon.parent / "optional_silence.txt"
    if beside.is_file():
        content = beside.read_text(encoding="utf-8").split()
        if content:
            return content[0]
    return "SIL"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m BenNevis.lm",
        description="Build the artefacts BenNevis.lm can build, as text.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    lexicon_fst = commands.add_parser(
        "lexicon-fst",
        help="L.fst as OpenFST text, byte-identical to Kaldi's",
    )
    lexicon_fst.add_argument("lexicon", type=Path, help="the dictionary's lexicon.txt")
    lexicon_fst.add_argument("words", type=Path, help="words.txt")
    lexicon_fst.add_argument("phones", type=Path, help="phones.txt")
    lexicon_fst.add_argument(
        "--sil-prob",
        type=float,
        default=0.0,
        help="what prepare_lang.sh was given as --sil_prob. Its own default "
        "is 0.5, and this one is 0.0, so pass it explicitly: a mismatch "
        "builds a lexicon that is wrong rather than broken",
    )
    lexicon_fst.add_argument(
        "--position-dependent-phones",
        action="store_true",
        default=False,
        help="what prepare_lang.sh does unless told otherwise, so pass this "
        "unless the recipe passed --position-dependent-phones false",
    )
    lexicon_fst.add_argument(
        "--sil-phone",
        default=None,
        help="the optional-silence phone. Read from optional_silence.txt "
        "beside the lexicon when not given",
    )

    args = parser.parse_args(argv)
    if args.command != "lexicon-fst":  # pragma: no cover - argparse enforces
        parser.error(f"unknown command {args.command}")

    from BenNevis.lm.lexicon import build_lexicon_fst_text, read_lexicon

    text = build_lexicon_fst_text(
        read_lexicon(args.lexicon),
        read_table(args.words),
        read_table(args.phones),
        sil_prob=args.sil_prob,
        sil_phone=optional_silence(args.lexicon, args.sil_phone),
        position_dependent_phones=args.position_dependent_phones,
    )
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
