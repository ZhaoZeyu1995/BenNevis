#!/usr/bin/env python3
"""
yesno's data directories, without Kaldi or Perl.

Replaces `prepare_data.sh` and the three Perl scripts beside it --
`create_yesno_waves_test_train.pl`, `create_yesno_wav_scp.pl` and
`create_yesno_txt.pl` -- plus Kaldi's `utils/utt2spk_to_spk2utt.pl`,
which was the last thing stage 1 needed from the Kaldi checkout.

**The three Perl scripts are kept deliberately, as the oracle.** They are
what every WER published in `../README.md` was measured through, and
`tests/recipes/test_yesno_prepare_data.py` drives them and requires this
to agree **byte for byte** on `wav.scp` and `text` -- a stronger standard
than restating the split in a constant and asserting the constant. They
have no caller and that is not an oversight; a test asserts they are
still there, with the reason, so removing them is a decision rather than
a tidy-up. `prepare_data.sh` itself is gone, because its own logic is
here and its two other outputs -- `data/local/waves_all.list` and a copy
of `task.arpabo` -- were read by nothing.

The corpus is sixty waves whose names are **eight binary digits**, one
per spoken word, so the transcription is the file name: `0` is NO, `1` is
YES and `_` is a space. That is the whole of the language model's input
and the whole of the reference transcription.

Three things here are decisions, and each is recorded because getting it
wrong would move a published WER.

**The split is 31 and 29, stated rather than computed.** The corpus
tarball holds `README` and `README~` beside the waves, and
`prepare_data.sh` used to list the directory unfiltered: under `LC_ALL=C`
digits sort before `R`, so both landed in the test half of a 62-entry
list, half of which is 31. Feature extraction then failed on them, Kaldi
warned *"Probably a serious error"*, `fix_data_dir.sh` dropped them, and
the test set was 29 while claiming 31. Filtering the list fixes that, but
halving 60 gives 30 and moves one utterance across -- which would make
the WERs published in `../README.md` no longer comparable. So the count
is explicit.

**Only `.wav` files are listed.** A real audio failure would have looked
exactly like the README one, which is why the filter is here rather than
left to `fix_data_dir` to clean up afterwards.

**The paths in `wav.scp` stay relative to the recipe.** Every stage of a
Kaldi recipe runs from the recipe directory, and
`create_yesno_wav_scp.pl` wrote `$waves_dir/$name` with `$waves_dir` as
it was given -- `waves_yesno`. Making them absolute here would be a
gratuitous difference from what every recorded run used, and
`run/make_fbank.py` reads them from the recipe directory, which is where
the recipe runs it.

Every utterance is the speaker `global`, which is Kaldi's own choice for
this corpus and the reason `run/apply_cmvn.sh` clamps its job count to
the speaker count.

Usage
-----
    local/prepare_data.py waves_yesno

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

#: How many waves go to training. See the module docstring: not half of
#: however many there are.
TRAIN_COUNT = 31

#: What the corpus has, so a tarball that changed is noticed here rather
#: than as a strange WER.
EXPECTED_WAVES = 60

#: The two data directories, in the order the recipe's loops name them.
TRAIN_SET = "train_yesno"
TEST_SET = "test_yesno"

#: Every utterance's speaker. Kaldi's own choice for this corpus.
SPEAKER = "global"


def _ensure_importable() -> None:
    """
    Make `BenNevis` importable when run as a script.

    The recipe's `env.sh` exports PYTHONPATH, and this is also run
    straight from a checkout. Four levels up from
    `egs/yesno/asr/local/`.
    """
    root = str(Path(__file__).resolve().parents[4])
    if root not in sys.path:
        sys.path.insert(0, root)


def list_waves(waves_dir: Path) -> List[str]:
    """
    The corpus's wave file names, byte-sorted.

    Arguments
    ---------
    waves_dir: Path
        The unpacked `waves_yesno` directory.

    Returns
    -------
    List[str]
        File names including the `.wav` suffix, sorted as bytes -- which
        is `ls -1` under the `LC_ALL=C` the recipe's `path.sh` exports.

    Raises
    ------
    FileNotFoundError
        If the directory is not there.
    ValueError
        If it holds no waves at all. An empty corpus would otherwise
        produce two empty data directories, which read downstream as a
        model that recognises nothing rather than as a missing corpus.
    """
    if not waves_dir.is_dir():
        raise FileNotFoundError(f"{waves_dir} is not a directory")
    names = sorted(
        (path.name for path in waves_dir.iterdir() if path.name.endswith(".wav")),
        key=lambda name: name.encode("utf-8"),
    )
    if not names:
        raise ValueError(f"{waves_dir} holds no .wav files")
    return names


def split(names: List[str], train_count: int = TRAIN_COUNT) -> Tuple[List[str], ...]:
    """
    The training and test halves.

    Arguments
    ---------
    names: List[str]
        Every wave, byte-sorted.
    train_count: int
        How many go to training. Stated rather than computed; see the
        module docstring.

    Returns
    -------
    Tuple[List[str], List[str]]
        The training names and the test names.

    Raises
    ------
    ValueError
        If there are fewer waves than the training count asks for, which
        would otherwise give an empty test set and a WER over nothing.
    """
    if train_count > len(names):
        raise ValueError(
            f"asked for {train_count} training waves but the corpus has "
            f"{len(names)}"
        )
    return names[:train_count], names[train_count:]


def transcribe(name: str) -> str:
    """
    One file name's transcription.

    Arguments
    ---------
    name: str
        A wave's name, with or without its suffix. Eight binary digits
        separated by underscores.

    Returns
    -------
    str
        The words, space separated: `0` is NO and `1` is YES.

    Notes
    -----
    `create_yesno_txt.pl` substituted in the order `0`, then `1`, then
    `_`, which is safe because neither NO nor YES contains a digit. Done
    here in one pass per character, which cannot depend on that.
    """
    stem = name[:-4] if name.endswith(".wav") else name
    words = {"0": "NO", "1": "YES"}
    return " ".join(words.get(field, field) for field in stem.split("_"))


def tables(waves_dir: Path, names: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """
    The four files one data directory needs.

    Arguments
    ---------
    waves_dir: Path
        As the recipe gave it, so the paths stay relative to the recipe.
    names: List[str]
        The waves in this half, byte-sorted.

    Returns
    -------
    Dict[str, List[Tuple[str, str]]]
        File name to its key-and-rest rows: `wav.scp`, `text`, `utt2spk`
        and `spk2utt`.
    """
    utterances = [name[:-4] for name in names]
    return {
        "wav.scp": [
            (utterance, f"{waves_dir}/{name}")
            for utterance, name in zip(utterances, names)
        ],
        "text": [(utterance, transcribe(utterance)) for utterance in utterances],
        "utt2spk": [(utterance, SPEAKER) for utterance in utterances],
        "spk2utt": [(SPEAKER, " ".join(utterances))],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write yesno's data directories from the unpacked corpus."
    )
    parser.add_argument(
        "waves_dir",
        type=Path,
        help="the unpacked waves_yesno directory, as the recipe names it: "
        "its value goes into wav.scp, so a relative path stays relative",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data"),
        help="where to write the data directories",
    )
    parser.add_argument(
        "--train-count",
        type=int,
        default=TRAIN_COUNT,
        help="how many waves go to training. The default is the split the "
        "published WERs were measured on and is not half of the corpus",
    )
    args = parser.parse_args(argv)
    _ensure_importable()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    from BenNevis.utils.datadir import write_table

    try:
        names = list_waves(args.waves_dir)
        training, testing = split(names, args.train_count)
    except (FileNotFoundError, ValueError) as failure:
        logging.error("%s", failure)
        return 1

    if len(names) != EXPECTED_WAVES:
        # Not fatal: a changed corpus is the author's business. But it
        # moves the split and so every published number, which is worth
        # saying rather than discovering from a WER.
        logging.warning(
            "the corpus has %d waves where this recipe's results were "
            "measured on %d; the %d/%d split is no longer what it was",
            len(names),
            EXPECTED_WAVES,
            len(training),
            len(testing),
        )

    for dataset, half in ((TRAIN_SET, training), (TEST_SET, testing)):
        directory = args.data / dataset
        directory.mkdir(parents=True, exist_ok=True)
        for name, rows in tables(args.waves_dir, half).items():
            write_table(directory / name, rows)
        logging.info("wrote %s: %d utterances", directory, len(half))

    logging.info(
        "%d waves split %d/%d between %s and %s",
        len(names),
        len(training),
        len(testing),
        TRAIN_SET,
        TEST_SET,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
