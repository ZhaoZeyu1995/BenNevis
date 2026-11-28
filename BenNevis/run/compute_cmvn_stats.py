#!/usr/bin/env python3
"""
CMVN statistics for a data directory, without Kaldi.

The Python counterpart of `steps/compute_cmvn_stats.sh`, which is a
wrapper round `compute-cmvn-stats`. The arithmetic is
`BenNevis.utils.cmvn.compute_stats` and has been in the tree since the
first stage of the Kaldi removal; what was missing was an entry point, so
the recipes went on calling Kaldi for a page of sums.

Three things about it are decisions rather than defaults.

**The archive is double, and that is not a preference.** Kaldi writes
these statistics as a double matrix and never compresses them, because
the normaliser rests on their precision: a mean is a sum over tens of
thousands of frames, and narrowing it to float32 before dividing would
throw away exactly the digits `apply` then rounds deliberately. The
capture reads them directly for the same reason, where it has Kaldi
rewrite the *feature* archives uncompressed first.

**Statistics are keyed by speaker.** `steps/compute_cmvn_stats.sh` passes
`--spk2utt`, so every utterance of a speaker is normalised by that
speaker's statistics rather than its own, and `--per-utt` is the escape
for a corpus with no speaker information worth the name. Note yesno
assigns every utterance the speaker `global`, so there the two differ by
a great deal and the recipe wants the default.

**Accumulation is not bit-identical to Kaldi's, and the difference is in
numpy's favour.** `compute-cmvn-stats` adds frame by frame in C++, while
`compute_stats` sums a matrix with numpy, which pairwise-sums and so
carries *less* error over a long utterance. The statistics therefore
differ in their last bits. What matters downstream is whether the float32
scale and offset `apply` derives from them come out the same, since those
are what touch the features, and that is measured rather than assumed:
`tests/reference/compare_features.py` reports it from a run that has both.

Usage
-----
    python3 run/compute_cmvn_stats.py <data_dir> [--output <dir>] \\
        [--per-utt]

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
    """
    Make `BenNevis` importable when run as a script.

    The same three lines as `run/make_fbank.py` and
    `run/prepare_topo.py`, and for the same reason: a recipe's `env.sh`
    exports PYTHONPATH, and this is also run straight from a checkout.
    """
    root = str(Path(__file__).resolve().parent.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


def read_spk2utt(path: Path) -> "dict":
    """
    Read a `spk2utt` file as a mapping from speaker to utterance list.

    Arguments
    ---------
    path: Path
        The file to read, one speaker per line followed by its utterances.

    Returns
    -------
    dict
        Speaker to list of utterance ids, in the file's own order -- which
        is the order the statistics are accumulated in, so it is part of
        the answer rather than incidental.

    Raises
    ------
    ValueError
        On a line naming a speaker and no utterances, which would
        otherwise produce a speaker whose statistics record no frames and
        a division by zero four stages later.
    """
    mapping = {}
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 2:
                raise ValueError(f"{path}:{number}: speaker with no utterances")
            mapping[fields[0]] = fields[1:]
    return mapping


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute CMVN statistics for a Kaldi data directory."
    )
    parser.add_argument("data_dir", type=Path, help="the data directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="where to write cmvn.ark, defaulting to the data directory. "
        "cmvn.scp is always written into the data directory, since that "
        "is where everything downstream looks for it",
    )
    parser.add_argument(
        "--per-utt",
        action="store_true",
        default=False,
        help="key the statistics by utterance rather than by speaker, "
        "for a corpus whose speaker labels mean nothing",
    )
    args = parser.parse_args(argv)
    _ensure_importable()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    import kaldiio

    from BenNevis.utils.cmvn import compute_stats

    data = args.data_dir
    feats = data / "feats.scp"
    if not feats.is_file():
        logging.error("%s does not exist; run make_fbank first", feats)
        return 1

    if args.per_utt:
        groups = None
    else:
        spk2utt = data / "spk2utt"
        if not spk2utt.is_file():
            logging.error(
                "%s does not exist, and statistics are per speaker unless "
                "--per-utt is given",
                spk2utt,
            )
            return 1
        try:
            groups = read_spk2utt(spk2utt)
        except ValueError as failure:
            logging.error("%s", failure)
            return 1

    # Lazy, so a five-hour corpus costs a seek per utterance rather than
    # a load of every feature matrix at once.
    loader = kaldiio.load_scp(str(feats))
    if groups is None:
        groups = {uttid: [uttid] for uttid in loader}

    out = args.output or data
    out.mkdir(parents=True, exist_ok=True)
    ark = (out / "cmvn.ark").resolve()
    scp = data / "cmvn.scp"

    written = 0
    missing = 0
    with kaldiio.WriteHelper(f"ark,scp:{ark},{scp}") as writer:
        for key, utterances in groups.items():
            matrices = []
            for uttid in utterances:
                if uttid not in loader:
                    # Kaldi warns and carries on here, and so does this:
                    # one utterance whose features failed should not cost
                    # the speaker, let alone the corpus. `fix_data_dir`
                    # is what removes it from the data directory.
                    missing += 1
                    logging.warning("%s has no features; skipping it", uttid)
                    continue
                matrices.append(loader[uttid])
            if not matrices:
                logging.warning("%s has no features at all; no statistics", key)
                continue
            writer(key, compute_stats(matrices))
            written += 1

    if written == 0:
        # An empty cmvn.scp would make apply_cmvn skip every utterance and
        # leave an empty feats.cmvn.scp, which trains on nothing and reads
        # as a bad model rather than as no data.
        logging.error("%s yielded no statistics", feats)
        return 1

    logging.info(
        "wrote statistics for %d %s to %s%s",
        written,
        "utterances" if args.per_utt else "speakers",
        scp,
        f", skipping {missing} utterances without features" if missing else "",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
