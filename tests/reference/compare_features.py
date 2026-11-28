#!/usr/bin/env python3
"""
Compare BenNevis's stage 2 against Kaldi's, inside the recipe.

The recipes compute filterbank features, CMVN statistics and normalised
features in Python now. This is the measurement that licences that
switch, and it has to run inside a recipe run for the same reason
`compare_features`' graph counterpart does: both sides exist only there.
The committed reference cannot answer it any more, because the committed
reference *is* one of the two sides from this change on -- see
`Manifest.features_from`.

What is compared, on the same `wav.scp` and the same `conf/fbank.conf`:

* **the filterbank**, ours against `compute-fbank-feats`, both written
  compressed so the comparison is like for like. `make_fbank.sh` wrote
  compressed archives and the recipes still ask for compression, which
  is what keeps the WER ledger's measurements valid across the switch;
  8-bit quantisation is also the reason a *closeness* comparison here is
  weak, so the grid is reported as well as the difference.
* **the statistics**, ours against `compute-cmvn-stats`. These are
  double and are not expected to be bit-identical: Kaldi accumulates
  frame by frame in C++ while numpy pairwise-sums a matrix, which
  carries less error rather than more. What matters is the next line.
* **the normaliser those statistics imply**, as the float32 scale and
  offset `apply` derives from them. This is the number that decides
  whether the difference in the statistics can reach a feature at all,
  and it is the one to read first: if the scale and offset agree to the
  bit, the statistics' last-bit disagreement is invisible downstream.
* **the normalised features**, ours against `apply-cmvn`, each side
  driven by *its own* statistics -- the whole stage end to end, which is
  what the trainer actually reads.

Exit status is non-zero if any of the four exceeds its tolerance, so the
workflow step can gate rather than merely report. The tolerances are
stated in `TOLERANCES` with what each one means.

Usage
-----
    python3 tests/reference/compare_features.py --recipe egs/yesno/asr \\
        --data data/train_yesno [--data data/test_yesno]

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from BenNevis.utils.cmvn import apply, mean_and_variance  # noqa: E402
from BenNevis.utils.features import read_fbank_conf  # noqa: E402

#: Where each Kaldi binary lives inside the image's Kaldi tree, for the
#: same reason `capture.py` has this table: the nightly invokes these
#: scripts directly, with none of the PATH a recipe's `path.sh` builds.
TOOL_LOCATIONS = {
    "compute-fbank-feats": "src/featbin",
    "compute-cmvn-stats": "src/featbin",
    "apply-cmvn": "src/featbin",
    "copy-feats": "src/featbin",
}

#: What each comparison is allowed to differ by, and why.
TOLERANCES = {
    # Both sides are quantised to the same 8-bit grid, so agreement is
    # expected to be exact or one grid step. A step is about 0.05 in the
    # log-mel domain; anything past a step means the arithmetic differs
    # rather than the rounding.
    "fbank": 0.06,
    # Double sums over tens of thousands of frames, in different orders.
    # Relative, because a sum of squares over 80 dimensions of log-mel
    # energies is of order 1e6 and an absolute bound would say nothing.
    "stats_relative": 1e-12,
    # The float32 numbers that actually touch a feature. Exact: these are
    # rounded to float32, so a last-bit difference in the double
    # statistics almost always disappears here, and if it does not, that
    # is the thing to know.
    "normaliser": 0.0,
    # The stage end to end. Dominated by the filterbank's own difference
    # above, divided by the standard deviation.
    "features": 0.06,
}


def find_tool(name: str) -> Optional[str]:
    """
    Locate a Kaldi binary, on PATH or inside the Kaldi tree.

    Arguments
    ---------
    name: str
        One of `TOOL_LOCATIONS`.

    Returns
    -------
    Optional[str]
        Its path, or None.
    """
    found = shutil.which(name)
    if found:
        return found
    root = Path(os.environ.get("KALDI_ROOT", "/opt/kaldi"))
    candidate = root / TOOL_LOCATIONS[name] / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def discover(recipe: Path) -> List[Path]:
    """
    Every data directory in the recipe that stage 2 has written.

    Arguments
    ---------
    recipe: Path
        The recipe directory.

    Returns
    -------
    List[Path]
        Paths relative to the recipe, sorted.

    Notes
    -----
    Discovery rather than names passed in, because the two recipes spell
    their sets differently -- `train_yesno` and `test_yesno` against
    `train_clean_5` and `dev_clean_2` -- and a workflow holding both
    lists is a list that drifts from the recipe. A directory counts only
    if it has all four inputs this needs, so a `data/local` or a
    `data/lang_*` is skipped without a special case.
    """
    needed = (
        "wav.scp",
        "spk2utt",
        "utt2spk",
        "feats.scp",
        "cmvn.scp",
        "feats.cmvn.scp",
    )
    found = []
    for candidate in sorted((recipe / "data").glob("*")):
        if candidate.is_dir() and all((candidate / name).is_file() for name in needed):
            found.append(candidate.relative_to(recipe))
    return found


def run(command: List[str], cwd: Path) -> None:
    """
    Run a Kaldi binary, failing loudly.

    Arguments
    ---------
    command: List[str]
        The argument list.
    cwd: Path
        Where to run it -- the recipe directory, because `wav.scp` holds
        paths relative to it.

    Raises
    ------
    RuntimeError
        If it exits non-zero, with its stderr, which is where Kaldi puts
        everything worth reading.
    """
    done = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(
            f"{command[0]} exited {done.returncode}\n{done.stderr[-2000:]}"
        )


def kaldi_stage2(
    recipe: Path, data: Path, scratch: Path, conf: Path
) -> Dict[str, Path]:
    """
    Run Kaldi's stage 2 into a scratch directory.

    Deliberately **not** into the data directory: the recipe's own
    outputs are one side of the comparison and overwriting them would
    leave nothing to compare. Compressed, because the recipe's are.

    Arguments
    ---------
    recipe: Path
        The recipe directory, which the scp paths are relative to.
    data: Path
        The data directory, relative to the recipe.
    scratch: Path
        Where to put Kaldi's archives.
    conf: Path
        The recipe's `conf/fbank.conf`, read by Kaldi's extractor as well
        as ours so the two cannot be given different settings.

    Returns
    -------
    Dict[str, Path]
        The three scp files Kaldi wrote.
    """
    feats = scratch / "feats.scp"
    stats = scratch / "cmvn.scp"
    normalised = scratch / "feats.cmvn.scp"

    run(
        [
            find_tool("compute-fbank-feats"),
            f"--config={conf}",
            "--compress=true",
            f"scp:{data / 'wav.scp'}",
            f"ark,scp:{scratch / 'feats.ark'},{feats}",
        ],
        recipe,
    )
    run(
        [
            find_tool("compute-cmvn-stats"),
            f"--spk2utt=ark:{data / 'spk2utt'}",
            f"scp:{feats}",
            f"ark,scp:{scratch / 'cmvn.ark'},{stats}",
        ],
        recipe,
    )
    run(
        [
            find_tool("apply-cmvn"),
            "--norm-vars=true",
            f"--utt2spk=ark:{data / 'utt2spk'}",
            f"scp:{stats}",
            f"scp:{feats}",
            f"ark,scp:{scratch / 'feats.cmvn.ark'},{normalised}",
        ],
        recipe,
    )
    return {"feats": feats, "cmvn": stats, "feats.cmvn": normalised}


def worst_difference(mine, theirs, keys) -> Dict[str, float]:
    """
    The largest and mean absolute difference over a set of keys.

    Arguments
    ---------
    mine: Mapping
        Our archive, as a mapping from key to array.
    theirs: Mapping
        Kaldi's, likewise.
    keys: Iterable[str]
        The keys to compare, which must be in both.

    Returns
    -------
    Dict[str, float]
        `max`, `mean` and `compared`. `compared` is there because a
        comparison over nothing passes, which is this repository's
        most-repeated defect.
    """
    import numpy as np

    worst = 0.0
    total = 0.0
    elements = 0
    where = ""
    for key in keys:
        a = np.asarray(mine[key], dtype=np.float64)
        b = np.asarray(theirs[key], dtype=np.float64)
        rows = min(a.shape[0], b.shape[0])
        difference = np.abs(a[:rows] - b[:rows])
        if difference.size and difference.max() > worst:
            worst = float(difference.max())
            where = key
        total += float(difference.sum())
        elements += difference.size
    return {
        "max": worst,
        "mean": total / elements if elements else 0.0,
        "compared": elements,
        "worst_at": where,
    }


def report(name: str, measured: Dict[str, float], tolerance: float) -> bool:
    """
    Print one comparison's verdict and say whether it passed.

    Arguments
    ---------
    name: str
        What was compared.
    measured: Dict[str, float]
        From `worst_difference`.
    tolerance: float
        The bound from `TOLERANCES`.

    Returns
    -------
    bool
        True if it passed.
    """
    if not measured["compared"]:
        print(f"FAIL: {name} compared nothing")
        return False
    ok = measured["max"] <= tolerance
    print(
        f"{'OK' if ok else 'FAIL'}: {name} agrees to {measured['max']:.3e} "
        f"max, {measured['mean']:.3e} mean, over {measured['compared']} "
        f"values (tolerance {tolerance:g})"
        + (f", worst at {measured['worst_at']}" if measured["worst_at"] else "")
    )
    return ok


def compare(recipe: Path, data: Path) -> bool:
    """
    Compare one data set's stage-2 outputs against Kaldi's.

    Arguments
    ---------
    recipe: Path
        The recipe directory.
    data: Path
        The data directory, relative to the recipe.

    Returns
    -------
    bool
        True if every comparison was inside its tolerance.
    """
    import kaldiio
    import numpy as np

    absolute = recipe / data
    print(f"--- {data}")

    conf = recipe / "conf" / "fbank.conf"
    # Read rather than only passed to Kaldi, for two reasons: it refuses a
    # configuration the two extractors cannot both be given -- a `--dither`
    # above zero makes the comparison meaningless at any tolerance -- and
    # it puts the settings in the log, so a reader need not go and look.
    options = read_fbank_conf(conf)
    print(
        f"{options['num_mel_bins']} mel bins at {options['sample_frequency']:g} Hz, "
        f"dither {options['dither']:g}"
    )

    ours = {
        name: dict(kaldiio.load_scp(str(absolute / f"{name}.scp")))
        for name in ("feats", "cmvn", "feats.cmvn")
    }

    passed = True
    with tempfile.TemporaryDirectory() as temporary:
        scratch = Path(temporary)
        written = kaldi_stage2(recipe, data, scratch, conf.resolve())
        theirs = {
            name: dict(kaldiio.load_scp(str(path))) for name, path in written.items()
        }

        shared = sorted(set(ours["feats"]) & set(theirs["feats"]))
        if not shared:
            print("FAIL: the two sides have no utterance in common")
            return False
        if len(shared) != len(ours["feats"]):
            print(
                f"note: {len(ours['feats']) - len(shared)} of our "
                f"{len(ours['feats'])} utterances are not in Kaldi's archive"
            )

        passed &= report(
            "fbank",
            worst_difference(ours["feats"], theirs["feats"], shared),
            TOLERANCES["fbank"],
        )

        speakers = sorted(set(ours["cmvn"]) & set(theirs["cmvn"]))
        relative = {
            key: np.asarray(ours["cmvn"][key])
            / np.maximum(np.abs(np.asarray(theirs["cmvn"][key])), 1.0)
            for key in speakers
        }
        reference = {
            key: np.asarray(theirs["cmvn"][key])
            / np.maximum(np.abs(np.asarray(theirs["cmvn"][key])), 1.0)
            for key in speakers
        }
        passed &= report(
            "cmvn statistics (relative)",
            worst_difference(relative, reference, speakers),
            TOLERANCES["stats_relative"],
        )

        # The normaliser is the thing that decides whether the statistics'
        # disagreement can reach a feature. float32, as `apply` stores it.
        mine_norm = {}
        their_norm = {}
        for key in speakers:
            for store, stats in (
                (mine_norm, ours["cmvn"]),
                (their_norm, theirs["cmvn"]),
            ):
                mean, variance = mean_and_variance(stats[key])
                scale = (1.0 / np.sqrt(variance)).astype(np.float32)
                store[key] = np.stack(
                    [scale, (-(mean * (1.0 / np.sqrt(variance)))).astype(np.float32)]
                )
        passed &= report(
            "the float32 scale and offset",
            worst_difference(mine_norm, their_norm, speakers),
            TOLERANCES["normaliser"],
        )

        passed &= report(
            "normalised features",
            worst_difference(ours["feats.cmvn"], theirs["feats.cmvn"], shared),
            TOLERANCES["features"],
        )

        # How much of the filterbank difference is the shared grid rather
        # than the arithmetic. Reported, not gated: it is what makes the
        # 0.06 tolerance above readable.
        grid = max(
            len(np.unique(column))
            for key in shared[:5]
            for column in np.asarray(theirs["feats"][key]).T
        )
        print(f"note: Kaldi's archive holds at most {grid} distinct values a column")

        # And the one comparison that needs no Kaldi: our normalisation
        # of our features must be exactly what `apply` gives, or the
        # entry point has arithmetic of its own.
        drift = 0.0
        for key in sorted(ours["feats.cmvn"]):
            speaker = next(
                spk
                for spk, utts in (
                    (line.split()[0], line.split()[1:])
                    for line in (absolute / "spk2utt").read_text().splitlines()
                )
                if key in utts
            )
            expected = apply(
                ours["feats"][key],
                ours["cmvn"][speaker],
                **{"norm_means": True, "norm_vars": True},
            )
            drift = max(
                drift,
                float(
                    np.abs(expected.astype(np.float64) - ours["feats.cmvn"][key]).max()
                ),
            )
        print(
            f"{'OK' if drift == 0.0 else 'FAIL'}: our archive is exactly "
            f"BenNevis.utils.cmvn.apply of our own features ({drift:.3e})"
        )
        passed &= drift == 0.0

    return passed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare BenNevis's stage 2 against Kaldi's."
    )
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        action="append",
        default=None,
        help="a data directory relative to the recipe; repeatable. "
        "Without it, every data directory that stage 2 has written is "
        "discovered -- so a recipe's set names do not have to be "
        "repeated in the workflow, where they would drift",
    )
    args = parser.parse_args(argv)

    missing = [name for name in TOOL_LOCATIONS if find_tool(name) is None]
    if missing:
        print(f"FAIL: cannot find {', '.join(sorted(missing))}")
        return 1

    recipe = args.recipe.resolve()
    sets = args.data if args.data else discover(recipe)
    if not sets:
        # A comparison over no data set passes, which is the defect this
        # repository has shipped three times.
        print(f"FAIL: no data directory under {recipe / 'data'} has stage 2's output")
        return 1
    print(f"comparing {', '.join(str(path) for path in sets)}")

    passed = True
    for data in sets:
        try:
            passed &= compare(recipe, data)
        except Exception as failure:  # pragma: no cover - needs the image
            print(f"FAIL: {data}: {failure!r}")
            passed = False
    print("all comparisons passed" if passed else "at least one comparison failed")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
