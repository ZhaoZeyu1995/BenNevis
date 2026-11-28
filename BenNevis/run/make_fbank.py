#!/usr/bin/env python3
"""
Filterbank features for a data directory, without Kaldi.

The Python counterpart of `steps/make_fbank.sh`, and **what every recipe
calls now**. It was added beside Kaldi's in #102 and left unused, because
switching changes the features, hence the CMVN statistics, hence every
WER. What made the switch safe is `--compress`: the recipes pass it, so
the archive stays on the same 8-bit grid Kaldi's was, which is the grid
every number in both WER ledgers was measured on. Writing uncompressed is
a separate measured decision and not a tidy-up.

Two differences from `make_fbank.sh` worth knowing before comparing
anything:

**It writes uncompressed by default, and Kaldi's writes compressed** --
so the recipes pass `--compress` and a bare invocation does not.
`CompressedMatrix` quantises each column to 8 bits under its default
`kSpeechFeature` scheme, which is a mean error of about 0.012 in the log
mel domain -- the whole of the "1.2% residual" CLAUDE.md chased through
six hypotheses. So an uncompressed archive is not the same data as
Kaldi's even when the arithmetic is identical, and `--compress` exists to
produce a like-for-like archive when that is what is wanted.

**It writes `utt2dur` unconditionally, and that is not optional.**
`make_fbank.sh` wrote it by default -- `write_utt2dur=true`, through
`compute-fbank-feats --write-utt2dur` -- and `BenNevis.core.Dataset`
**requires** it: it filters on `min_duration` and `max_duration`, sorts
by it, and the duration-based samplers in `BenNevis/samplers/` batch on
it. So a flag to turn it off would be a flag to stop training from
starting, which is what the first version of this script did by not
writing it at all.
It is the **waveform's** duration in seconds, as Kaldi's is: samples
divided by the sample rate, not `utt2num_frames` scaled. Kaldi has a
separate `utils/data/get_utt2dur.sh` that derives it from the frame
count with a 1.5-frame correction when there is no audio to measure,
and the two do not agree exactly -- so measuring the waveform is both
simpler and what the recorded runs used.

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
        "--ark-dir",
        type=Path,
        default=None,
        help="where to write feats.ark, defaulting beside feats.scp. The "
        "recipes keep the archive in data/<set>/fbank and the scp in "
        "data/<set>, which is what make_fbank.sh does",
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
    ark_dir = args.ark_dir or out
    ark_dir.mkdir(parents=True, exist_ok=True)
    # Absolute, because the scp is read by later stages that do not
    # necessarily run from the recipe directory -- which is the trap that
    # cost a five-hour capture, from the other side.
    ark = (ark_dir / "feats.ark").resolve()

    frames = {}
    durations = {}
    failed = []
    written = 0
    with kaldiio.WriteHelper(
        f"ark,scp:{ark},{out / 'feats.scp'}",
        compression_method=2 if args.compress else None,
    ) as writer:
        for uttid in loader:
            try:
                rate, samples = loader[uttid]
                features = fbank(samples, sample_rate=rate, **options)
                # The waveform's own duration, which is what
                # `compute-fbank-feats --write-utt2dur` records. `samples`
                # may be (1, n) from a single-channel archive.
                count = samples.shape[-1] if hasattr(samples, "shape") else len(samples)
                duration = float(count) / float(rate)
            except Exception as failure:
                # One unreadable utterance must not cost the corpus, and
                # this is how `make_fbank.sh` behaves: it logs the
                # failure, leaves the utterance out of feats.scp, and
                # `fix_data_dir` drops it from the rest of the directory
                # afterwards. Exactly the shape of the bug that cost run
                # 46100161743's capture, one stage along -- so the gap
                # between `wav.scp` and `feats.scp` is deliberate and is
                # what the next stage exists to reconcile.
                failed.append(uttid)
                logging.warning("cannot compute features for %s: %r", uttid, failure)
                continue
            writer(uttid, features.numpy())
            frames[uttid] = features.shape[0]
            durations[uttid] = duration
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

    # Unconditional: `Dataset` cannot be constructed without it. See the
    # module docstring.
    (out / "utt2dur").write_text(
        "".join(f"{utt} {duration:.6f}\n" for utt, duration in durations.items()),
        encoding="utf-8",
    )
    if failed:
        logging.warning(
            "%d of %d utterances could not be read: %s%s. Run "
            "run/fix_data_dir.py to drop them from the data directory",
            len(failed),
            written + len(failed),
            ", ".join(failed[:5]),
            " ..." if len(failed) > 5 else "",
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
