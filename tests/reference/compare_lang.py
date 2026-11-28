#!/usr/bin/env python3
"""
Compare the Python language directory against Kaldi's, inside the recipe.

Stage 1 builds its language directory with
`python3 -m BenNevis.lm prepare-lang` where it used to run
`utils/prepare_lang.sh`. The committed reference cannot gate that any
more, for the reason `compare_features.py` exists: the capture's `lang/`
and `dict/` become **our** output the next time a capture is taken, and
`tests/lm/test_prepare_lang.py` would then compare the code with itself.
`Manifest.produced_by` records which side wrote each group and those
tests skip accordingly; this is where the live comparison went.

**The settings are read out of the recipe, not passed in.**
`prepare-lang` has to be given the same `--sil-prob` and
`--position-dependent-phones` as the lexicon it is compared against, and
getting them wrong produces a directory that is *wrong rather than
broken* -- the tables come out, the FSTs compose, every transcription is
quietly worse. A workflow holding its own copy of those flags is a copy
that drifts, so this parses the recipe's own stage-1 call: the comparison
is then against whatever the recipe actually does.

What is compared:

* **the tables**, byte for byte. There is no reason for a correct
  reimplementation to differ by a character, and nine of them were
  byte-exact first time.
* **the two lexicon FSTs**, as `fstprint` text against our `k2/*.fst.txt`.
  Byte-exact too, for both corpora -- which is not the usual standard for
  an FST, and holds here because `prepare_lang.sh` leaves both of the
  choices `G` makes arbitrary determined: states are allocated in
  `lexicon.txt`'s order and the arcs are left sorted by output label.

What is **not** compared is everything only GMM training reads --
`phones/sets.txt`, `roots.txt`, `extra_questions.txt`, `word_boundary.*`
and `topo`. BenNevis trains no GMM, and writing files nothing reads would
be four more things to keep correct.

Usage
-----
    python3 tests/reference/compare_lang.py --recipe egs/yesno/asr

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from BenNevis.lm.prepare import prepare_lang  # noqa: E402

#: The tables both sides write, compared as bytes. `phones/align_lexicon`
#: is here because it is the one file the *training* path reads, and it
#: was the gap in mini-librispeech's capture.
TABLES = (
    "phones.txt",
    "words.txt",
    "oov.txt",
    "oov.int",
    "phones/silence.int",
    "phones/nonsilence.int",
    "phones/disambig.int",
    "phones/optional_silence.int",
    "phones/wdisambig_phones.int",
    "phones/wdisambig_words.int",
    "phones/align_lexicon.txt",
    "phones/align_lexicon.int",
)

#: The lexicon FSTs, as Kaldi's binary against our text. The key is
#: Kaldi's name and the value ours.
FSTS = {"L.fst": "k2/L.fst.txt", "L_disambig.fst": "k2/L_disambig.fst.txt"}

#: Where `fstprint` lives inside the image's Kaldi tree, for the reason
#: `capture.py` has the same table: this is invoked directly, with none of
#: the PATH a recipe's `path.sh` builds.
FSTPRINT_LOCATIONS = ("tools/openfst/bin",)


def find_fstprint() -> Optional[str]:
    """
    Locate `fstprint`, on PATH or inside the Kaldi tree.

    Returns
    -------
    Optional[str]
        Its path, or None.
    """
    found = shutil.which("fstprint")
    if found:
        return found
    root = Path(os.environ.get("KALDI_ROOT", "/opt/kaldi"))
    for location in FSTPRINT_LOCATIONS:
        candidate = root / location / "fstprint"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def stage_one_call(recipe: Path) -> str:
    """
    The recipe's `prepare-lang` invocation, continuations folded away.

    Arguments
    ---------
    recipe: Path
        The recipe directory.

    Returns
    -------
    str
        The call, truncated at the shell's `||`, or an empty string if
        the recipe has not been converted.
    """
    text = (recipe / "run.sh").read_text(encoding="utf-8").replace("\\\n", " ")
    for line in text.splitlines():
        if "BenNevis.lm prepare-lang" in line and not line.strip().startswith("#"):
            tail = line.split("BenNevis.lm prepare-lang", 1)[1]
            for terminator in ("||", ";", "&&"):
                tail = tail.split(terminator, 1)[0]
            return tail
    return ""


def settings_from(call: str) -> Dict[str, object]:
    """
    What the recipe gives `prepare-lang`.

    Arguments
    ---------
    call: str
        From `stage_one_call`.

    Returns
    -------
    Dict[str, object]
        `dict_dir`, `oov_word`, `out_dir`, `sil_prob` and
        `position_dependent_phones`.

    Raises
    ------
    ValueError
        If the call does not state both settings, or does not have three
        positional arguments. Both are things
        `tests/recipes/test_stage1_wiring.py` pins, so reaching either
        here means the recipe changed shape.
    """
    sil = re.search(r"--sil-prob\s+(\S+)", call)
    position = re.search(r"--position-dependent-phones\s+(true|false)", call)
    if not sil or not position:
        raise ValueError(f"the call states no settings: {call!r}")

    valued = ("--sil-prob", "--position-dependent-phones", "--compile-fsts")
    positional: List[str] = []
    skip = False
    for token in call.split():
        if skip:
            skip = False
            continue
        if token in valued:
            skip = True
            continue
        if token.startswith("-"):
            continue
        positional.append(token)
    if len(positional) != 3:
        raise ValueError(f"expected three positional arguments, got {positional}")

    return {
        "dict_dir": positional[0],
        "oov_word": positional[1].strip('"').strip("'"),
        "out_dir": positional[2],
        "sil_prob": float(sil.group(1)),
        "position_dependent_phones": position.group(1) == "true",
    }


def kaldi_prepare_lang(recipe: Path, settings: Dict[str, object], out: Path) -> None:
    """
    Run `utils/prepare_lang.sh` with the same settings, into `out`.

    Arguments
    ---------
    recipe: Path
        The recipe directory, which `utils/` is a symlink inside.
    settings: Dict[str, object]
        From `settings_from`.
    out: Path
        Where Kaldi should write.

    Raises
    ------
    RuntimeError
        If the script fails, with its stderr.
    """
    argv = [
        "utils/prepare_lang.sh",
        "--sil-prob",
        str(settings["sil_prob"]),
        "--position-dependent-phones",
        "true" if settings["position_dependent_phones"] else "false",
        str(settings["dict_dir"]),
        str(settings["oov_word"]),
        str(out / "tmp"),
        str(out / "lang"),
    ]
    done = subprocess.run(argv, cwd=recipe, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(
            f"prepare_lang.sh exited {done.returncode}\n{done.stderr[-2000:]}"
        )


def fstprint(binary: str, path: Path) -> str:
    """
    Print a binary FST as text.

    Arguments
    ---------
    binary: str
        `fstprint`'s path.
    path: Path
        The FST.

    Returns
    -------
    str
        Its text, with numeric labels -- which is the form
        `BenNevis.lm.lexicon` writes.
    """
    done = subprocess.run([binary, str(path)], capture_output=True, check=True)
    return done.stdout.decode("utf-8")


def compare(recipe: Path) -> bool:
    """
    Compare the two language directories.

    Arguments
    ---------
    recipe: Path
        The recipe directory.

    Returns
    -------
    bool
        True if everything compared agreed.
    """
    call = stage_one_call(recipe)
    if not call:
        print(f"SKIP: {recipe} does not build its lang with prepare-lang")
        return True
    settings = settings_from(call)
    print(
        f"settings: sil_prob={settings['sil_prob']}, "
        f"position_dependent_phones={settings['position_dependent_phones']}, "
        f"dict={settings['dict_dir']}"
    )

    passed = True
    with tempfile.TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        kaldi_prepare_lang(recipe, settings, scratch)
        theirs = scratch / "lang"

        ours = scratch / "ours"
        prepare_lang(
            recipe / str(settings["dict_dir"]),
            str(settings["oov_word"]),
            ours,
            position_dependent_phones=bool(settings["position_dependent_phones"]),
            sil_prob=float(settings["sil_prob"]),
            compile_fsts="false",
        )

        compared = 0
        for name in TABLES:
            mine, their = ours / name, theirs / name
            if not their.is_file():
                print(f"note: Kaldi wrote no {name}")
                continue
            if not mine.is_file():
                print(f"FAIL: we wrote no {name}")
                passed = False
                continue
            compared += 1
            if mine.read_bytes() == their.read_bytes():
                print(f"OK: {name} is byte-identical")
            else:
                print(f"FAIL: {name} differs")
                passed = False
        if compared == 0:
            # A comparison over nothing passes, which is the defect this
            # repository has shipped three times.
            print("FAIL: no table was comparable")
            return False

        binary = find_fstprint()
        if binary is None:
            print("FAIL: fstprint is not available, so the FSTs are unchecked")
            return False
        for their_name, our_name in FSTS.items():
            their_fst = theirs / their_name
            our_fst = ours / our_name
            if not their_fst.is_file() or not our_fst.is_file():
                print(f"FAIL: {their_name} or {our_name} is missing")
                passed = False
                continue
            if fstprint(binary, their_fst) == our_fst.read_text(encoding="utf-8"):
                print(f"OK: {our_name} is byte-identical to {their_name}")
            else:
                print(f"FAIL: {our_name} differs from {their_name}")
                passed = False

    return passed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare BenNevis's language directory against Kaldi's."
    )
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args(argv)

    if shutil.which("bash") is None or os.name == "nt":
        print("FAIL: this needs a POSIX shell to run prepare_lang.sh")
        return 1

    recipe = args.recipe.resolve()
    if not (recipe / "utils" / "prepare_lang.sh").is_file():
        print(f"FAIL: {recipe}/utils/prepare_lang.sh is not there, so there is")
        print("FAIL: nothing to compare against. This needs the Kaldi image.")
        return 1

    try:
        passed = compare(recipe)
    except Exception as failure:  # pragma: no cover - needs the image
        print(f"FAIL: {failure!r}")
        passed = False
    print("the language directories agree" if passed else "they differ")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
