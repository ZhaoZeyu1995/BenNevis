#!/usr/bin/env python3
"""
Apply CMVN to a data directory, without Kaldi.

The Python counterpart of what `run/apply_cmvn.sh` used to do, which was
`utils/split_data.sh`, then `run.pl`, then `apply-cmvn` once per job, then
a `cat` of the per-job scp files. It writes `feats.cmvn.scp`, which is
the archive `BenNevis.core.Dataset` reads and the one
`BenNevis.utils.data.feature_dim` measures `idim` from.

The arithmetic is `BenNevis.utils.cmvn.apply`, which is **bit-exact with
`apply-cmvn` given identical inputs** -- it follows Kaldi's order of
operations rather than the obvious one, computing the moments in double,
rounding the scale and offset to float32 as it stores them, and applying
them as two float32 passes rather than one fused affine. Writing
`(x - mean) / sqrt(var)` in float64 instead agrees only to about 4e-6.

**There is no `--nj`, and that is measured rather than assumed.** The
shell split the data by speaker purely so that eight `apply-cmvn`
processes could run at once. In Python the whole of mini-librispeech's
`train_clean_5` -- 1,519 utterances, 1.79M frames, 80 dimensions -- takes
**1.2 seconds** to accumulate statistics over and **0.8 seconds** to
normalise and write, against the 3.9 seconds it takes merely to write the
compressed feature archive in the first place. A flag whose best case is
a second is a flag that can only go wrong, and dropping it also drops
`split_data.sh` and `run.pl`, which are Kaldi's.

**`--norm-vars` defaults to true, where Kaldi's own default is false.**
This reproduces the shell's default, which is what every recipe here has
always passed and so what every WER in the ledger was measured under.
`BenNevis.utils.cmvn.apply` takes it as a keyword for the same reason.

Usage
-----
    python3 run/apply_cmvn.py <data_dir> [--norm-vars true|false] \\
        [--output <dir>]

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


def boolean(text: str) -> bool:
    """
    Parse Kaldi's spelling of a boolean.

    Arguments
    ---------
    text: str
        `true` or `false`, which is what a Kaldi recipe writes.

    Returns
    -------
    bool

    Raises
    ------
    argparse.ArgumentTypeError
        On anything else. `read_fbank_conf` refuses `yes` for the same
        reason: a value that is neither is a mistake worth a message,
        and Python's `bool("false")` is True.
    """
    if text in ("true", "false"):
        return text == "true"
    raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")


def read_utt2spk(path: Path) -> "dict":
    """
    Read `utt2spk` as a mapping from utterance to speaker.

    Arguments
    ---------
    path: Path
        The file to read, one `utterance speaker` pair per line.

    Returns
    -------
    dict

    Raises
    ------
    ValueError
        On a line that is not a pair.
    """
    mapping = {}
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{number}: expected 'utterance speaker', got "
                    f"{len(fields)} fields"
                )
            mapping[fields[0]] = fields[1]
    return mapping


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply CMVN to a Kaldi data directory, writing " "feats.cmvn.scp."
    )
    parser.add_argument("data_dir", type=Path, help="the data directory")
    parser.add_argument(
        "--norm-vars",
        type=boolean,
        default=True,
        help="whether to divide by the standard deviation. True is what "
        "the recipes have always passed and so what the WER ledger was "
        "measured under; Kaldi's own default is false",
    )
    parser.add_argument(
        "--norm-means",
        type=boolean,
        default=True,
        help="whether to subtract the mean",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="where to write feats.cmvn.ark, defaulting to the data "
        "directory. feats.cmvn.scp is always written into the data "
        "directory, since that is where Dataset looks for it",
    )
    parser.add_argument("--report-every", type=int, default=500)
    args = parser.parse_args(argv)
    _ensure_importable()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    import kaldiio

    from BenNevis.utils.cmvn import apply

    data = args.data_dir
    feats = data / "feats.scp"
    cmvn = data / "cmvn.scp"
    utt2spk = data / "utt2spk"
    for path in (feats, cmvn, utt2spk):
        if not path.is_file():
            logging.error("%s does not exist", path)
            return 1

    try:
        speakers = read_utt2spk(utt2spk)
    except ValueError as failure:
        logging.error("%s", failure)
        return 1

    stats = dict(kaldiio.load_scp(str(cmvn)))
    loader = kaldiio.load_scp(str(feats))

    out = args.output or data
    out.mkdir(parents=True, exist_ok=True)
    ark = (out / "feats.cmvn.ark").resolve()
    scp = data / "feats.cmvn.scp"

    written = 0
    unmapped = 0
    with kaldiio.WriteHelper(f"ark,scp:{ark},{scp}") as writer:
        for uttid in loader:
            speaker = speakers.get(uttid)
            if speaker is None:
                # Not a warning to be shrugged at: an utterance with no
                # speaker cannot be normalised, so it would be trained on
                # unnormalised if it were written through. It is dropped
                # and counted, and a corpus losing all of them fails below.
                unmapped += 1
                logging.warning("%s is not in utt2spk; dropping it", uttid)
                continue
            if speaker not in stats:
                unmapped += 1
                logging.warning(
                    "%s has no statistics for speaker %s; dropping it",
                    uttid,
                    speaker,
                )
                continue
            writer(
                uttid,
                apply(
                    loader[uttid],
                    stats[speaker],
                    norm_means=args.norm_means,
                    norm_vars=args.norm_vars,
                ),
            )
            written += 1
            if args.report_every and written % args.report_every == 0:
                logging.info("%d utterances", written)

    if written == 0:
        logging.error("%s yielded no normalised features", feats)
        return 1

    logging.info(
        "wrote %d utterances to %s%s",
        written,
        scp,
        f", dropping {unmapped} without statistics" if unmapped else "",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
