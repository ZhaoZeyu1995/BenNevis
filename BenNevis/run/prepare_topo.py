#!/usr/bin/env python3
"""
Prepare a language directory for a topology, without Kaldi.

This replaces `prepare_topo.sh`, which remains as a shim so the five
recipes that have not been reviewed keep working unchanged. The shell
version had two dependencies on Kaldi that had nothing to do with what it
produces:

- it sourced **Kaldi's `utils/parse_options.sh`** to read its own flags,
  and `utils/` is a symlink into `tools/kaldi`, so the script could not
  parse `--topo` without a Kaldi checkout;
- it ran `fstcompile` and `fstarcsort` unconditionally, for `T.fst` and
  `k2/T.fst`, and **nothing in the Python lane reads either**. `T.fst` is
  for Kaldi's `prepare_graph.sh`; `k2/T.fst` is read only by
  `Lang.load_topo`'s fallback, for a directory prepared before
  `k2/topo.yaml` existed.

So stage 3 failed without Kaldi over an artefact it did not need and an
argument parser. Everything the Python lane does read -- the symbol
tables, `k2/topo.yaml` and `k2/L.fst.txt` -- was already Python.

`--compile-fsts` decides what happens to the two FSTs: `auto` compiles
them when OpenFST is on PATH and says so when it is not, `true` requires
it, `false` never does. `prepare_graph.sh` passes `true` because it
composes against `T.fst`, and fails pointing at this if the file is
missing.

The flags take Kaldi's `true`/`false` spellings rather than argparse's
store_true, because the shim passes a recipe's existing command line
through untouched and `--position-dependent-phones true` is what those
recipes write.

Usage
-----
    python3 run/prepare_topo.py --topo ctc \\
        [--lexicon data/local/dict/lexicon.txt] \\
        [--sil-prob 0.5] [--position-dependent-phones true] \\
        [--compile-fsts auto] <src_lang_dir> <des_lang_dir>

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Optional

#: Kaldi's own spellings, since a recipe's command line is passed through.
BOOLEANS = ("true", "false")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _ensure_importable() -> None:
    """
    Make `BenNevis` importable when run as a script from a recipe.

    `env.sh` exports PYTHONPATH, but this is also run directly in tests and
    by anyone debugging a language directory, where it would otherwise fail
    on the first import rather than on a missing argument.
    """
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def parse(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a language directory for a topology.",
    )
    parser.add_argument("src", type=Path, help="the source language directory")
    parser.add_argument("des", type=Path, help="where to write the result")
    parser.add_argument("--topo", required=True, help="which topology")
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=None,
        help="the dictionary's lexicon.txt. Given, k2/L.fst.txt is written "
        "and training needs no OpenFST; omitted, Lang reads L.fst through "
        "fstprint once and warns",
    )
    parser.add_argument(
        "--sil-prob",
        type=float,
        default=0.0,
        help="what prepare_lang.sh was given as --sil_prob. Its own default "
        "is 0.5 and this one is 0.0, so pass it explicitly: a mismatch "
        "builds a lexicon that is wrong rather than broken",
    )
    parser.add_argument(
        "--position-dependent-phones",
        choices=BOOLEANS,
        default="false",
        help="what prepare_lang.sh did, which is true unless it was passed "
        "--position-dependent-phones false",
    )
    parser.add_argument(
        "--compile-fsts",
        choices=("auto",) + BOOLEANS,
        default="auto",
        help="whether to compile T.fst and k2/T.fst, which only Kaldi's "
        "prepare_graph.sh and Lang's legacy fallback read",
    )
    return parser.parse_args(argv)


def topology_exists(topo: str) -> bool:
    from BenNevis.topos import spec as topo_spec

    return topo in set(topo_spec.available())


def available_topologies() -> List[str]:
    from BenNevis.topos import spec as topo_spec

    return list(topo_spec.available())


def write_symbol_tables(topo: str, des: Path) -> None:
    """
    The tables, from the specification rather than from `grep` and `awk`.

    The shell version selected disambiguation symbols with
    `grep "#[^\\d]"`, which picks out `#0` only where `\\d` inside a bracket
    expression means the literal characters `\\` and `d` -- POSIX, but not
    what `ugrep` or `grep -P` do, under which the token table silently came
    out as `#0_0`, `#0_1`. These call the same functions
    `python3 -m BenNevis.topos` does, so the two cannot disagree.
    """
    from BenNevis.topos import build as topo_build
    from BenNevis.topos import spec as topo_spec

    topology = topo_spec.load(topo)
    phones = topo_build.read_phone_table(des / "phones.txt")
    k2 = des / "k2"

    def table(**kwargs) -> str:
        return topo_build.write_symbol_table(
            topo_build.build_tokens(topology, phones, **kwargs)
        )

    (des / "tokens_disambig.txt").write_text(table(), encoding="utf-8")
    (des / "tokens.txt").write_text(table(with_disambig=False), encoding="utf-8")
    (k2 / "tokens.txt").write_text(
        table(with_eps=False, with_disambig=False), encoding="utf-8"
    )
    (k2 / "phones.txt").write_text(
        topo_build.write_symbol_table(
            topo_build.build_phone_symbols(phones, with_disambig=False)
        ),
        encoding="utf-8",
    )
    ids = topo_build.symbol_ids(topo_build.build_tokens(topology, phones))
    (des / "disambig.int").write_text(
        "".join(f"{ids[phone]}\n" for phone in phones if topo_build.is_disambig(phone)),
        encoding="utf-8",
    )


def write_lexicon_text(args: argparse.Namespace, des: Path) -> None:
    """
    `k2/L.fst.txt`, which is what takes OpenFST off the training path.

    Byte-identical to Kaldi's `L.fst` -- 1,539,598 lines for
    mini-librispeech's 206,510-entry lexicon, matching the digest the
    capture recorded -- so this is Kaldi's lexicon FST computed in Python
    rather than an approximation of it.
    """
    from BenNevis.lm.__main__ import optional_silence, read_table
    from BenNevis.lm.lexicon import build_lexicon_fst_text, read_lexicon

    text = build_lexicon_fst_text(
        read_lexicon(args.lexicon),
        read_table(des / "words.txt"),
        read_table(des / "phones.txt"),
        sil_prob=args.sil_prob,
        sil_phone=optional_silence(args.lexicon, None),
        position_dependent_phones=args.position_dependent_phones == "true",
    )
    (des / "k2" / "L.fst.txt").write_text(text, encoding="utf-8")
    logging.info(
        "wrote %s (%d lines)",
        des / "k2" / "L.fst.txt",
        text.count("\n"),
    )


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _ensure_importable()
    args = parse(argv)

    # Imported here rather than at the top, like every other BenNevis
    # import in this file: the module is reachable only after
    # `_ensure_importable`.
    from BenNevis.utils.openfst import available as openfst_available
    from BenNevis.utils.openfst import compile_fst

    if not topology_exists(args.topo):
        logging.error("unknown topology %s", args.topo)
        logging.error("supported: %s", " ".join(available_topologies()))
        return 1
    if not args.src.is_dir():
        logging.error("%s is not a directory", args.src)
        return 1
    if args.lexicon is not None and not args.lexicon.is_file():
        logging.error("--lexicon %s does not exist", args.lexicon)
        return 1

    compile_fsts = args.compile_fsts
    if compile_fsts == "auto":
        if openfst_available():
            compile_fsts = "true"
        else:
            compile_fsts = "false"
            logging.info(
                "fstcompile is not on PATH, so T.fst and k2/T.fst will not be "
                "written. Nothing in the Python lane reads them; Kaldi's "
                "prepare_graph.sh does, and fails saying so."
            )
    elif compile_fsts == "true" and not openfst_available():
        logging.error("--compile-fsts true but OpenFST is not on PATH")
        return 1

    logging.info("using the %s topology", args.topo)
    des = args.des
    if des.exists():
        logging.info("%s already exists, removing it", des)
        shutil.rmtree(des)
    shutil.copytree(args.src, des, symlinks=True)
    (des / "k2").mkdir(exist_ok=True)

    write_symbol_tables(args.topo, des)

    # The specification itself, which is what `Lang` expands at training
    # time, so that a language directory stays reproducible when the
    # bundled specs change.
    from BenNevis.topos import build as topo_build
    from BenNevis.topos import spec as topo_spec

    (des / "k2" / "topo.yaml").write_text(
        topo_spec.load(args.topo).source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    if args.lexicon is not None:
        write_lexicon_text(args, des)
    else:
        logging.info(
            "no --lexicon, so %s was not written and training will read "
            "L.fst through fstprint once, needing OpenFST",
            des / "k2" / "L.fst.txt",
        )

    if compile_fsts == "true":
        loaded = topo_spec.load(args.topo)
        phones = topo_build.read_phone_table(des / "phones.txt")
        compile_fst(
            topo_build.build_openfst_text(loaded, "lang", phones),
            des / "T.fst",
            isymbols=des / "tokens_disambig.txt",
            osymbols=des / "phones.txt",
            keep_symbols=False,
        )
        k2_phones = topo_build.read_phone_table(des / "k2" / "phones.txt")
        compile_fst(
            topo_build.build_openfst_text(loaded, "k2", k2_phones),
            des / "k2" / "T.fst",
            isymbols=des / "k2" / "tokens.txt",
            osymbols=des / "k2" / "phones.txt",
            keep_symbols=True,
        )
        logging.info("wrote %s and %s", des / "T.fst", des / "k2" / "T.fst")
    return 0


if __name__ == "__main__":
    sys.exit(main())
