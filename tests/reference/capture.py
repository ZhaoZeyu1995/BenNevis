"""
Capture what Kaldi produced from a completed recipe run.

Run inside the Kaldi image, against a recipe that has finished, this writes
`tests/golden/reference/<corpus>/` and the manifest describing it. That
directory is then committed, and the Kaldi-removal work compares against the
committed bytes rather than against a Kaldi it has to keep working.

Why capture rather than compare live: recent Kaldi cannot fetch the OpenFST
version its own source needs, and the OpenBLAS it pins no longer compiles on
current hardware. The image that still works is pinned by digest, and its
output is worth recording while it can still be produced. A capture also
moves the comparison into the cheap test tiers, which is where the migration
work actually happens.

What is captured, and why each one:

- The symbol tables and lexicon FSTs from `prepare_lang.sh`. These are the
  part of the pipeline BenNevis does not yet build, and the part a Python
  lexicon has to reproduce. As text, compared as bytes.
- `G.fst` and `TLG.fst` as text, for the same reason.
- CMVN statistics in full: two rows per speaker, so there is no reason not to.
- A few feature matrices in full, and per-utterance statistics for the rest.
  Every matrix would be 12MB of float32 for yesno and buy nothing over a
  handful.
- The decoded hypotheses and the reference transcriptions, per topology.

Usage
-----
    python3 tests/reference/capture.py egs/yesno/asr tests/golden/reference

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from BenNevis.utils.reference import (  # noqa: E402
    FEATURE_TOLERANCE,
    MANIFEST_NAME,
    Manifest,
    array_stats,
    sha256_file,
)

#: How many feature matrices to keep whole. Enough for a genuine elementwise
#: comparison across more than one utterance; the rest are kept as statistics.
FULL_FEATURE_UTTERANCES = 5

#: Text artefacts copied verbatim from the language directory. These are what
#: a Python replacement for `prepare_lang.sh` has to reproduce.
LANG_TEXT = (
    "phones.txt",
    "words.txt",
    "oov.txt",
    "oov.int",
    "phones/align_lexicon.int",
    "phones/disambig.int",
    "phones/silence.int",
    "phones/optional_silence.int",
)

#: Binary FSTs, converted to text with fstprint so that a diff is readable
#: and a replacement can be compared without OpenFST.
LANG_FSTS = ("L.fst", "L_disambig.fst")
GRAPH_FSTS = ("G.fst", "TLG.fst", "LG.fst")


def find_fstprint() -> Optional[str]:
    """
    Locate `fstprint`, on PATH or inside the Kaldi tree.

    The Kaldi image has it, but only on the PATH that a recipe's `path.sh`
    builds. A capture invoked directly -- as the nightly workflow does -- has
    no such PATH, which is how the first captures came out without a single
    FST in them: `shutil.which` found nothing and the code quietly carried
    on. Looking under `KALDI_ROOT` as well makes the capture independent of
    how it was invoked.

    Returns
    -------
    Optional[str]
        The path to fstprint, or None if it is nowhere to be found.
    """
    found = shutil.which("fstprint")
    if found:
        return found

    kaldi_root = os.environ.get("KALDI_ROOT", "/opt/kaldi")
    candidate = Path(kaldi_root) / "tools" / "openfst" / "bin" / "fstprint"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def fstprint(binary: str, path: Path) -> Optional[bytes]:
    """
    Print a binary FST as text.

    Arguments
    ---------
    binary: str
        The fstprint executable, as `find_fstprint` located it.
    path: Path
        The compiled FST.

    Returns
    -------
    Optional[bytes]
        Its text form, or None if the call failed. A failure is reported on
        stderr rather than raised, so that one unreadable FST does not lose
        the rest of the capture -- but it is never silent.
    """
    try:
        completed = subprocess.run(
            [binary, str(path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        sys.stderr.write(f"capture: fstprint {path} failed: {error.stderr!r}\n")
        return None
    return completed.stdout


def read_archive(scp: Path) -> Dict[str, "object"]:
    """
    Read a Kaldi feature archive referenced by an scp file.

    Arguments
    ---------
    scp: Path
        An scp mapping utterance ids to archive offsets.

    Returns
    -------
    Dict[str, numpy.ndarray]
        Utterance id to feature matrix, in scp order.
    """
    import kaldiio

    return {key: value for key, value in kaldiio.load_scp_sequential(str(scp))}


def store(manifest: Manifest, destination: Path, relative: str, payload: bytes) -> None:
    """
    Write one exact artefact and record its digest.

    Arguments
    ---------
    manifest: Manifest
        The manifest to record it in.
    destination: Path
        The capture directory.
    relative: str
        Its path within that directory, which is also its manifest key.
    payload: bytes
        The content.
    """
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    manifest.exact[relative] = sha256_file(target)


def capture_language(
    manifest: Manifest,
    recipe: Path,
    destination: Path,
    topologies: List[str],
    printer: Optional[str],
) -> None:
    """
    Capture the language directories: symbol tables and lexicon FSTs.

    Arguments
    ---------
    manifest: Manifest
        The manifest to record into.
    recipe: Path
        The recipe directory.
    destination: Path
        The capture directory.
    topologies: List[str]
        The topologies the recipe built.
    printer: Optional[str]
        The fstprint executable, or None to skip the compiled FSTs.
    """
    base = recipe / "data" / "lang"
    for name in LANG_TEXT:
        source = base / name
        if source.is_file():
            store(manifest, destination, f"lang/{name}", source.read_bytes())

    if printer is None:
        return

    for name in LANG_FSTS:
        source = base / name
        if not source.is_file():
            continue
        text = fstprint(printer, source)
        if text is not None:
            store(manifest, destination, f"lang/{name}.txt", text)

    for topology in topologies:
        graph = recipe / "data" / f"lang_test_tg_{topology}"
        for name in GRAPH_FSTS:
            source = graph / name
            if not source.is_file():
                continue
            text = fstprint(printer, source)
            if text is not None:
                store(
                    manifest,
                    destination,
                    f"graph/{topology}/{name}.txt",
                    text,
                )


def capture_features(
    manifest: Manifest, recipe: Path, destination: Path, datasets: List[str]
) -> None:
    """
    Capture feature matrices and CMVN statistics.

    A few matrices whole, the rest as statistics, and all of the CMVN
    statistics, which are two rows per speaker.

    Arguments
    ---------
    manifest: Manifest
        The manifest to record into.
    recipe: Path
        The recipe directory.
    destination: Path
        The capture directory.
    datasets: List[str]
        The data directories to capture.
    """
    import numpy as np

    for dataset in datasets:
        data = recipe / "data" / dataset
        scp = data / "feats.cmvn.scp"
        if not scp.is_file():
            sys.stderr.write(f"capture: no {scp}, skipping {dataset}\n")
            continue

        features = read_archive(scp)
        keys = sorted(features)

        whole = keys[:FULL_FEATURE_UTTERANCES]
        if whole:
            arrays = {key: np.asarray(features[key]) for key in whole}
            relative = f"features/{dataset}.npz"
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **arrays)
            for key in whole:
                manifest.arrays[f"features/{dataset}/{key}"] = {
                    "path": relative,
                    "key": key,
                    "shape": list(arrays[key].shape),
                    "dtype": str(arrays[key].dtype),
                    "tolerance": {
                        "rtol": FEATURE_TOLERANCE.rtol,
                        "atol": FEATURE_TOLERANCE.atol,
                    },
                }

        for key in keys:
            manifest.stats[f"features/{dataset}/{key}"] = array_stats(features[key])

        cmvn = data / "cmvn.scp"
        if cmvn.is_file():
            stats = read_archive(cmvn)
            arrays = {key: np.asarray(value) for key, value in stats.items()}
            relative = f"cmvn/{dataset}.npz"
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **arrays)
            for key, value in arrays.items():
                manifest.arrays[f"cmvn/{dataset}/{key}"] = {
                    "path": relative,
                    "key": key,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "tolerance": {
                        "rtol": FEATURE_TOLERANCE.rtol,
                        "atol": FEATURE_TOLERANCE.atol,
                    },
                }


def capture_transcriptions(
    manifest: Manifest,
    recipe: Path,
    destination: Path,
    topologies: List[str],
    model: str,
    test_set: str,
    decode_dir: str,
) -> None:
    """
    Capture the reference and decoded transcriptions for each topology.

    Arguments
    ---------
    manifest: Manifest
        The manifest to record into.
    recipe: Path
        The recipe directory.
    destination: Path
        The capture directory.
    topologies: List[str]
        The topologies the recipe decoded.
    model: str
        The model name, as the experiment directories are prefixed.
    test_set: str
        The decoded data set.
    decode_dir: str
        The decoding parameter directory name.
    """
    for topology in topologies:
        experiment = recipe / "exp" / f"{model}-{topology}"
        reference = experiment / f"pred_{test_set}" / "ref.wrd.trn"
        if reference.is_file():
            store(
                manifest,
                destination,
                f"transcriptions/{topology}/ref.wrd.trn",
                reference.read_bytes(),
            )
        hypothesis = experiment / f"dec_{test_set}" / decode_dir / "hyp.wrd.trn"
        if hypothesis.is_file():
            store(
                manifest,
                destination,
                f"transcriptions/{topology}/hyp.wrd.trn",
                hypothesis.read_bytes(),
            )


def main(argv=None) -> int:
    """
    Capture a reference from a finished recipe run.

    Arguments
    ---------
    argv: Optional[List[str]]
        The arguments, defaulting to sys.argv[1:].

    Returns
    -------
    int
        0 if anything was captured, 1 if nothing was.
    """
    parser = argparse.ArgumentParser(
        description="Capture Kaldi's output from a finished recipe run.",
    )
    parser.add_argument("recipe", type=Path, help="the recipe directory")
    parser.add_argument(
        "destination",
        type=Path,
        help="where to write <corpus>/; usually tests/golden/reference",
    )
    parser.add_argument("--corpus", default="yesno")
    parser.add_argument("--model", default="rnnp")
    parser.add_argument("--test-set", default="test_yesno")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["train_yesno", "test_yesno"],
    )
    parser.add_argument("--decode-dir", default="aw_1.0-ma_5000-bm_32")
    parser.add_argument(
        "--topos",
        nargs="+",
        default=None,
        help="topologies to capture; by default whatever the recipe built",
    )
    parser.add_argument(
        "--allow-no-fstprint",
        action="store_true",
        help=(
            "capture everything else if fstprint cannot be found, instead of "
            "failing. The compiled FSTs are the point of the capture for the "
            "lexicon and graph work, so their absence is an error by default."
        ),
    )
    args = parser.parse_args(argv)

    recipe = args.recipe.resolve()
    if not (recipe / "data").is_dir():
        sys.stderr.write(f"capture: no data directory in {recipe}\n")
        return 1

    topologies = args.topos
    if topologies is None:
        topologies = sorted(
            path.name[len("lang_test_tg_") :]
            for path in (recipe / "data").glob("lang_test_tg_*")
            if path.is_dir()
        )
    if not topologies:
        sys.stderr.write("capture: no prepared graphs found\n")

    # Resolved before anything is written, so a capture that cannot include
    # the FSTs fails before producing a directory that looks complete.
    printer = find_fstprint()
    if printer is None:
        message = (
            "capture: fstprint was not found on PATH or under "
            f"{os.environ.get('KALDI_ROOT', '/opt/kaldi')}/tools/openfst/bin. "
            "The lexicon and graph FSTs are what a Python replacement for "
            "prepare_lang.sh and the graph compilation will be compared "
            "against, so a capture without them is not much use."
        )
        if not args.allow_no_fstprint:
            sys.stderr.write(message + " Pass --allow-no-fstprint to proceed.\n")
            return 1
        sys.stderr.write(message + " Continuing without them.\n")

    out = (args.destination / args.corpus).resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest = Manifest(
        corpus=args.corpus,
        captured=datetime.date.today().isoformat(),
        # Recorded by the workflow, which knows the digest it pulled. A
        # reference whose provenance is unknown is not a reference.
        image=os.environ.get("BENNEVIS_IMAGE", "unknown"),
        recipe=str(args.recipe),
        notes=[
            "Captured from a completed run of the recipe named above, inside "
            "the image named above.",
            "The `exact` entries are compared as bytes: there is no reason "
            "for a correct reimplementation to differ by a character.",
            "The `arrays` and `stats` entries carry a tolerance, because "
            "bit-exact features are not achievable across implementations "
            "(FFT and accumulation order differ).",
            f"{FULL_FEATURE_UTTERANCES} feature matrices per data set are "
            "stored whole; the rest are statistics, because every matrix "
            "would be megabytes and buy nothing over a handful.",
        ],
    )

    capture_language(manifest, recipe, out, topologies, printer)
    capture_features(manifest, recipe, out, args.datasets)
    capture_transcriptions(
        manifest,
        recipe,
        out,
        topologies,
        args.model,
        args.test_set,
        args.decode_dir,
    )

    manifest.write(out / MANIFEST_NAME)

    exact, arrays, stats = manifest.names()
    sys.stdout.write(
        f"capture: wrote {out}\n"
        f"  {len(exact)} exact artefact(s)\n"
        f"  {len(arrays)} array(s) stored whole\n"
        f"  {len(stats)} array(s) as statistics\n"
    )
    if manifest.is_empty():
        sys.stderr.write("capture: nothing was captured\n")
        return 1

    # Silence is not success: say which of the FSTs the capture exists for
    # are actually in it, so an incomplete one is visible in the log rather
    # than discovered when a migration PR has nothing to compare against.
    if printer is not None:
        expected = ["lang/L.fst.txt", "lang/L_disambig.fst.txt"] + [
            f"graph/{topology}/TLG.fst.txt" for topology in topologies
        ]
        absent = [name for name in expected if name not in manifest.exact]
        if absent:
            sys.stderr.write(
                f"capture: {len(absent)} expected FST(s) are missing, the "
                f"first being {absent[0]}; the recipe may not have built "
                f"them\n"
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
