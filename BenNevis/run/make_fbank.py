#!/usr/bin/env python3
"""
Filterbank features for a data directory, without Kaldi.

The Python counterpart of `steps/make_fbank.sh`. Added **beside** it
rather than in place of it, and the recipes still call Kaldi's: switching
would change the features, hence the CMVN statistics, hence every WER, and
that is a measured decision rather than a detail. What this provides is
the ability to make that measurement.

Two differences from `make_fbank.sh` worth knowing before comparing
anything:

**It writes uncompressed by default, and Kaldi's writes compressed.**
`CompressedMatrix` quantises each column to 8 bits under its default
`kSpeechFeature` scheme, which is a mean error of about 0.012 in the log
mel domain -- the whole of the "1.2% residual" CLAUDE.md chased through
six hypotheses. So an uncompressed archive is not the same data as
Kaldi's even when the arithmetic is identical, and `--compress` exists to
produce a like-for-like archive when that is what is wanted.

**It computes filterbank only, where `make_fbank_pitch.sh` appended three
pitch dimensions.** Pitch was dropped; `torchaudio` removed
`compute_kaldi_pitch` in 2.1 and nothing replaces it.

Usage
-----
    python3 run/make_fbank.py <data_dir> [--conf conf/fbank.conf] \\
        [--compress] [--write-utt2num-frames]

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

    A recipe's `env.sh` exports PYTHONPATH, but this is also run directly
    from a checkout and in tests, where it would otherwise fail on the
    first import rather than on a missing argument. Same as
    `run/prepare_topo.py`.
    """
    root = str(Path(__file__).resolve().parent.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute filterbank features for a Kaldi data directory."
    )
    parser.add_argument("data_dir", type=Path, help="the data directory")
    parser.add_argument(
        "--conf",
        type=Path,
        default=None,
        help="the recipe's conf/fbank.conf. Read rather than restated so "
        "that Kaldi's extractor and this one cannot be given different "
        "settings; without it, Kaldi's own defaults apply",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="where to write feats.ark and feats.scp, defaulting to the "
        "data directory",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        default=False,
        help="write a compressed archive, as make_fbank.sh does. That "
        "quantises each column to 8 bits, which is a mean error of about "
        "0.012 in the log mel domain -- only for a like-for-like "
        "comparison with Kaldi's own output",
    )
    parser.add_argument(
        "--write-utt2num-frames",
        action="store_true",
        default=False,
        help="also write utt2num_frames, which the sampler reads",
    )
    parser.add_argument("--report-every", type=int, default=500)
    args = parser.parse_args(argv)
    _ensure_importable()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    import kaldiio

    from BenNevis.utils.features import FBANK_DEFAULTS, fbank, read_fbank_conf

    data = args.data_dir
    scp = data / "wav.scp"
    if not scp.is_file():
        logging.error("%s does not exist", scp)
        return 1
    options = (
        read_fbank_conf(args.conf) if args.conf is not None else dict(FBANK_DEFAULTS)
    )
    logging.info(
        "%d mel bins at %g Hz, dither %g",
        options["num_mel_bins"],
        options["sample_frequency"],
        options["dither"],
    )

    segments = data / "segments"
    loader = kaldiio.load_scp(
        str(scp), segments=str(segments) if segments.is_file() else None
    )
    out = args.output or data
    out.mkdir(parents=True, exist_ok=True)

    frames = {}
    written = 0
    with kaldiio.WriteHelper(
        f"ark,scp:{out / 'feats.ark'},{out / 'feats.scp'}",
        compression_method=2 if args.compress else None,
    ) as writer:
        for uttid in loader:
            rate, samples = loader[uttid]
            features = fbank(samples, sample_rate=rate, **options)
            writer(uttid, features.numpy())
            frames[uttid] = features.shape[0]
            written += 1
            if args.report_every and written % args.report_every == 0:
                logging.info("%d utterances", written)

    if written == 0:
        # An empty archive would train on nothing and score as a run of
        # deletions, which reads as a bad model rather than as no data.
        logging.error("%s yielded no utterances", scp)
        return 1

    if args.write_utt2num_frames:
        (out / "utt2num_frames").write_text(
            "".join(f"{utt} {count}\n" for utt, count in frames.items()),
            encoding="utf-8",
        )
    logging.info(
        "wrote %d utterances, %d frames, to %s",
        written,
        sum(frames.values()),
        out / "feats.scp",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
