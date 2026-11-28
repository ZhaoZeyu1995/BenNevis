"""
Compiling OpenFST text to the binary form Kaldi's decoder reads.

Two entry points now need this -- `run/prepare_topo.py` for `T.fst` and
`python3 -m BenNevis.lm prepare-lang` for the two lexicon FSTs -- and
`prepare_topo.py` had it first, privately. Sharing it is deliberate for
the reason CLAUDE.md already records about the symbol tables: two
implementations of one thing are how the two come to disagree, and what
they would disagree about here is the **arc order** of a graph the
decoder loads, which composes either way and decodes slightly
differently.

**This is the boundary of the Pythonisation, not a gap in it.** Nothing
in BenNevis's training or decoding lane reads a binary FST: training
reads `k2/topo.yaml` and `k2/L.fst.txt`, and `run/decode_k2.py` builds
its graph in k2. The binary form exists for Kaldi's `prepare_graph.sh`
and `decode-faster`, which are the stages goal 2 deletes last and
together. So every caller treats OpenFST as **optional**: the text is
written always, the binary when `fstcompile` is there, and the stage
that genuinely needs it says so and fails.

The invocation is `prepare_lang.sh`'s and `prepare_topo.sh`'s, read off
the scripts rather than reconstructed:

    fstcompile --isymbols=... --osymbols=... \\
        --keep_isymbols=false --keep_osymbols=false | \\
        fstarcsort --sort_type=olabel

`--keep_isymbols=false` matters: with the tables attached the binary
carries them and is a different file, which Kaldi's own stages neither
write nor expect. `--sort_type=olabel` matters because `fsttablecompose`
requires its first argument sorted on the output side.

Functions
---------
available
    Whether both binaries can be found.
compile_fst
    `fstcompile | fstarcsort` over text.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

__all__ = ["BINARIES", "available", "compile_fst"]

#: What a compile needs on PATH. Both, because a tree with one and not the
#: other would fail halfway through and leave a truncated FST.
BINARIES = ("fstcompile", "fstarcsort")


def available() -> bool:
    """
    Whether OpenFST's two binaries are both on PATH.

    Returns
    -------
    bool
    """
    return all(shutil.which(name) for name in BINARIES)


def compile_fst(
    text: str,
    output: Path,
    *,
    isymbols: Optional[Path] = None,
    osymbols: Optional[Path] = None,
    keep_symbols: bool = False,
    sort_type: str = "olabel",
) -> None:
    """
    Compile OpenFST text and arc-sort it, writing the binary FST.

    Arguments
    ---------
    text: str
        The FST as OpenFST text. May carry numeric labels, in which case
        no symbol tables are needed -- `BenNevis.lm.lexicon` writes
        numeric text, byte-identical to `fstprint`'s output, so compiling
        it without tables gives the same binary as Kaldi's symbolic text
        compiled with them.
    output: Path
        Where to write the binary FST.
    isymbols: Optional[Path]
        The input symbol table, for symbolic text.
    osymbols: Optional[Path]
        The output symbol table, likewise.
    keep_symbols: bool
        Whether to leave the tables attached to the binary. False is what
        both of Kaldi's scripts pass, and a binary carrying them is a
        different file.
    sort_type: str
        `fstarcsort`'s sort type. `olabel` is what both scripts use and
        what `fsttablecompose` requires of its first argument.

    Raises
    ------
    subprocess.CalledProcessError
        If either binary fails. Two calls rather than a shell pipeline, so
        a failure in the first half cannot be hidden by the success of the
        second -- which is what `set -o pipefail` was for in the shell
        version, and the same trap as the nightly's `| tee`.
    FileNotFoundError
        If either binary is absent. Callers test `available()` first and
        decide whether that is fatal.
    """
    argv = ["fstcompile"]
    if isymbols is not None:
        argv.append(f"--isymbols={isymbols}")
    if osymbols is not None:
        argv.append(f"--osymbols={osymbols}")
    if (isymbols is not None or osymbols is not None) and not keep_symbols:
        argv += ["--keep_isymbols=false", "--keep_osymbols=false"]

    compiled = subprocess.run(
        argv, input=text.encode("utf-8"), capture_output=True, check=True
    )
    ordered = subprocess.run(
        ["fstarcsort", f"--sort_type={sort_type}"],
        input=compiled.stdout,
        capture_output=True,
        check=True,
    )
    output.write_bytes(ordered.stdout)
