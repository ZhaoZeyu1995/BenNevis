"""
Stage 2 without Kaldi: features, CMVN statistics, fixing, normalisation.

Tier 1b, because the entry points need `kaldiio`, numpy and torchaudio,
all of which arrive with torch. They are run the way a recipe runs them
-- as subprocesses from a recipe directory -- rather than by importing
`main`, because the thing being tested is the stage, and two of the four
defects this stage has had were in how it was *invoked*: a relative
`wav.scp` read from the wrong directory, and an archive path that was not
absolute.

The arithmetic is not retested here; `tests/utils/test_cmvn.py` holds it
against the captured reference and `tests/utils/test_features.py` holds
the filterbank against the grid. What these pin is the wiring: that the
four steps compose, that the archive is the kind of archive Kaldi wrote,
and that one unreadable utterance costs that utterance and nothing more.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import os
import subprocess
import sys
import wave
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
kaldiio = pytest.importorskip("kaldiio")
pytest.importorskip("torch")
pytest.importorskip("torchaudio")

pytestmark = pytest.mark.torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUN = REPO_ROOT / "BenNevis" / "run"

#: Windows has no execute bit and the recipes are POSIX-only anyway, but
#: these are run through `sys.executable` with an argument list, so they
#: do work there. What does not is the piped `wav.scp` form.
posix_only = pytest.mark.skipif(
    os.name == "nt", reason="a piped wav.scp entry needs a POSIX shell"
)

SPEAKERS = ("spkA", "spkB")
UTTERANCES = tuple(f"{spk}-{n}" for spk in SPEAKERS for n in (1, 2))


def write_wave(path, seconds=4.0, rate=8000, seed=0):
    """
    A mono 16-bit wave of noise, which is all the filterbank needs.

    Four seconds rather than one, and the reason is the compression test:
    an 8-bit grid has 256 levels, so an utterance of fewer than 256
    frames cannot show one. At 10ms a frame this gives 398.
    """
    rng = np.random.default_rng(seed)
    samples = (rng.standard_normal(int(rate * seconds)) * 3000).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


@pytest.fixture
def recipe(tmp_path):
    """
    A recipe directory the stage can be run from.

    `wav.scp` holds **relative** paths, as a real recipe's does: every
    stage of a Kaldi recipe runs from the recipe directory, and reading
    one from anywhere else is what cost run 46100161743's capture.
    """
    root = tmp_path / "recipe"
    (root / "conf").mkdir(parents=True)
    (root / "corpus").mkdir()
    data = root / "data" / "train"
    data.mkdir(parents=True)

    (root / "conf" / "fbank.conf").write_text(
        "--num-mel-bins=80\n--sample-frequency=8000\n--dither=0\n", "utf-8"
    )
    for index, utt in enumerate(UTTERANCES):
        write_wave(root / "corpus" / f"{utt}.wav", seed=index)

    (data / "wav.scp").write_text(
        "".join(f"{utt} ./corpus/{utt}.wav\n" for utt in UTTERANCES), "utf-8"
    )
    (data / "text").write_text(
        "".join(f"{utt} HELLO WORLD\n" for utt in UTTERANCES), "utf-8"
    )
    (data / "utt2spk").write_text(
        "".join(f"{utt} {utt.split('-')[0]}\n" for utt in UTTERANCES), "utf-8"
    )
    (data / "spk2utt").write_text(
        "".join(f"{spk} {spk}-1 {spk}-2\n" for spk in SPEAKERS), "utf-8"
    )
    return root


def step(recipe, script, *arguments, expect=0):
    """
    Run one of the entry points from the recipe directory.

    `sys.executable` with an argument list rather than the shebang, so
    this works wherever pytest does, and `check` is explicit because two
    of these tests are about a non-zero status.
    """
    done = subprocess.run(
        [sys.executable, str(RUN / script), *map(str, arguments)],
        cwd=recipe,
        capture_output=True,
        text=True,
    )
    assert done.returncode == expect, (
        f"{script} exited {done.returncode}, expected {expect}\n"
        f"--- stderr\n{done.stderr}"
    )
    return done


def whole_stage(recipe, *, compress=True):
    """The four steps a recipe's stage 2 runs, in its order."""
    arguments = [
        "--conf",
        "conf/fbank.conf",
        "--write-utt2num-frames",
        "--ark-dir",
        "data/train/fbank",
    ]
    if compress:
        arguments.append("--compress")
    step(recipe, "make_fbank.py", *arguments, "data/train")
    step(recipe, "compute_cmvn_stats.py", "--output", "data/train/cmvn", "data/train")
    step(recipe, "fix_data_dir.py", "data/train")
    step(recipe, "apply_cmvn.py", "data/train")
    return recipe / "data" / "train"


class TestTheWholeStage:
    def test_it_produces_the_archive_the_dataset_reads(self, recipe):
        data = whole_stage(recipe)
        features = dict(kaldiio.load_scp(str(data / "feats.cmvn.scp")))
        assert sorted(features) == sorted(UTTERANCES)
        assert all(matrix.shape[1] == 80 for matrix in features.values())
        assert all(matrix.dtype == np.float32 for matrix in features.values())

    def test_normalisation_is_per_speaker(self, recipe):
        """
        The reason statistics are computed separately from being applied.
        Each speaker's own frames come out zero-mean and unit-variance;
        pooling them would not.
        """
        data = whole_stage(recipe)
        features = dict(kaldiio.load_scp(str(data / "feats.cmvn.scp")))
        for speaker in SPEAKERS:
            stacked = np.concatenate(
                [matrix for utt, matrix in features.items() if utt.startswith(speaker)]
            )
            assert np.abs(stacked.mean(axis=0)).max() < 1e-4, speaker
            assert np.abs(stacked.var(axis=0) - 1.0).max() < 1e-4, speaker

    def test_utt2num_frames_matches_the_archive(self, recipe):
        """
        The sampler batches on this file, so a disagreement with the
        features is a wrong batch rather than an error.
        """
        data = whole_stage(recipe)
        counts = dict(
            line.split() for line in (data / "utt2num_frames").read_text().splitlines()
        )
        features = dict(kaldiio.load_scp(str(data / "feats.scp")))
        assert {utt: str(matrix.shape[0]) for utt, matrix in features.items()} == counts

    def test_the_archive_path_is_absolute(self, recipe):
        """
        Later stages do not all run from the recipe directory, and a
        relative archive path in an scp is only readable from where it
        was written.

        Split from the **right**: an scp entry is `<path>:<offset>` and a
        Windows path carries a drive colon of its own, so splitting from
        the left gives `C` and reads as relative. The first version of
        this test did that and failed only on Windows -- the fourth time
        a path assumption has broken that tier and nothing else.
        """
        data = whole_stage(recipe)
        for name in ("feats.scp", "cmvn.scp", "feats.cmvn.scp"):
            for line in (data / name).read_text().splitlines():
                archive, _offset = line.split()[1].rsplit(":", 1)
                assert Path(archive).is_absolute(), f"{name}: {archive}"


class TestTheStatistics:
    def test_they_are_double_and_in_kaldis_layout(self, recipe):
        """
        Kaldi writes these as a double matrix and never compresses them:
        a mean is a sum over tens of thousands of frames, and narrowing it
        before dividing throws away the digits `apply` then rounds
        deliberately. The layout is `(2, dim + 1)` with the frame count in
        the first row's last column.
        """
        data = whole_stage(recipe)
        stats = dict(kaldiio.load_scp(str(data / "cmvn.scp")))
        assert sorted(stats) == sorted(SPEAKERS)
        for speaker, matrix in stats.items():
            assert matrix.dtype == np.float64, speaker
            assert matrix.shape == (2, 81), speaker
            assert matrix[1, 80] == 0.0

    def test_the_frame_count_is_the_sum_of_the_utterances(self, recipe):
        data = whole_stage(recipe)
        stats = dict(kaldiio.load_scp(str(data / "cmvn.scp")))
        counts = dict(
            (utt, int(count))
            for utt, count in (
                line.split()
                for line in (data / "utt2num_frames").read_text().splitlines()
            )
        )
        for speaker, matrix in stats.items():
            expected = sum(
                count for utt, count in counts.items() if utt.startswith(speaker)
            )
            assert matrix[0, 80] == expected, speaker

    def test_per_utt_keys_by_utterance_instead(self, recipe):
        step(
            recipe,
            "make_fbank.py",
            "--conf",
            "conf/fbank.conf",
            "--compress",
            "data/train",
        )
        step(recipe, "compute_cmvn_stats.py", "--per-utt", "data/train")
        stats = kaldiio.load_scp(str(recipe / "data" / "train" / "cmvn.scp"))
        assert sorted(stats) == sorted(UTTERANCES)

    def test_a_missing_spk2utt_is_refused_rather_than_assumed(self, recipe):
        """
        Per-utterance statistics are a different normaliser, not a
        fallback: on yesno, where every utterance is the speaker
        `global`, the two differ by a great deal.
        """
        step(
            recipe,
            "make_fbank.py",
            "--conf",
            "conf/fbank.conf",
            "data/train",
        )
        (recipe / "data" / "train" / "spk2utt").unlink()
        done = step(recipe, "compute_cmvn_stats.py", "data/train", expect=1)
        assert "--per-utt" in done.stderr


class TestTheNormalisationIsTheModulesAndNothingElse:
    def test_it_agrees_bit_for_bit_with_utils_cmvn(self, recipe):
        """
        The entry point must add no arithmetic of its own.
        `BenNevis.utils.cmvn.apply` is the piece gated against
        `apply-cmvn`, so anything the script does differently is
        ungated by construction.
        """
        from BenNevis.utils.cmvn import apply

        data = whole_stage(recipe)
        raw = dict(kaldiio.load_scp(str(data / "feats.scp")))
        stats = dict(kaldiio.load_scp(str(data / "cmvn.scp")))
        done = dict(kaldiio.load_scp(str(data / "feats.cmvn.scp")))
        for utt, matrix in done.items():
            expected = apply(raw[utt], stats[utt.split("-")[0]])
            assert np.array_equal(matrix, expected), utt

    def test_norm_vars_false_leaves_the_variance_alone(self, recipe):
        """
        True is the recipes' default and so what every WER in the ledger
        was measured under; Kaldi's own default is false, which is why the
        flag is spelled out at every call site.
        """
        data = whole_stage(recipe)
        step(recipe, "apply_cmvn.py", "--norm-vars", "false", "data/train")
        features = dict(kaldiio.load_scp(str(data / "feats.cmvn.scp")))
        stacked = np.concatenate(
            [m for utt, m in features.items() if utt.startswith("spkA")]
        )
        assert np.abs(stacked.mean(axis=0)).max() < 1e-3
        assert np.abs(stacked.var(axis=0) - 1.0).max() > 0.1

    def test_a_bad_boolean_is_refused(self, recipe):
        whole_stage(recipe)
        step(recipe, "apply_cmvn.py", "--norm-vars", "yes", "data/train", expect=2)


class TestCompression:
    def test_compress_puts_the_features_on_an_eight_bit_grid(self, recipe):
        """
        The recipes pass `--compress` deliberately. `make_fbank.sh` wrote
        compressed archives, so this keeps the data the trainer reads
        like for like across the switch -- the WER ledger was measured on
        8-bit features, and writing uncompressed would change CMVN's
        inputs and so every WER. Dropping the flag is a measured decision
        of its own, not a tidy-up.
        """
        data = whole_stage(recipe, compress=True)
        features = dict(kaldiio.load_scp(str(data / "feats.scp")))
        for utt, matrix in features.items():
            assert max(len(np.unique(column)) for column in matrix.T) <= 256, utt

    def test_without_it_the_features_are_not_quantised(self, recipe):
        data = whole_stage(recipe, compress=False)
        features = dict(kaldiio.load_scp(str(data / "feats.scp")))
        # 398 frames, so a column can hold more distinct values than an
        # 8-bit grid has levels -- which is the only way to tell the two
        # archives apart. See `write_wave`.
        widest = max(
            len(np.unique(column))
            for matrix in features.values()
            for column in matrix.T
        )
        assert widest > 256


class TestOneBadUtteranceCostsOneUtterance:
    def test_an_unreadable_wave_is_skipped_and_then_dropped(self, recipe):
        """
        How `make_fbank.sh` behaves, and the whole reason `fix_data_dir`
        runs after it: the utterance is left out of `feats.scp` and then
        removed from the rest of the directory. The first version of
        `make_fbank.py` died on it instead, which is the same defect that
        cost run 46100161743's capture, one stage along.
        """
        data = recipe / "data" / "train"
        with open(data / "wav.scp", "a", encoding="utf-8") as handle:
            handle.write("spkB-3 ./corpus/absent.wav\n")
        with open(data / "text", "a", encoding="utf-8") as handle:
            handle.write("spkB-3 GONE\n")
        with open(data / "utt2spk", "a", encoding="utf-8") as handle:
            handle.write("spkB-3 spkB\n")
        (data / "spk2utt").write_text(
            "spkA spkA-1 spkA-2\nspkB spkB-1 spkB-2 spkB-3\n", "utf-8"
        )

        whole_stage(recipe)

        for name in ("feats.scp", "text", "utt2spk", "wav.scp", "feats.cmvn.scp"):
            assert "spkB-3" not in (data / name).read_text(), name
        assert sorted(kaldiio.load_scp(str(data / "feats.cmvn.scp"))) == sorted(
            UTTERANCES
        )

    def test_losing_every_utterance_fails_rather_than_writing_nothing(self, recipe):
        """
        An empty archive trains on nothing and scores as a run of
        deletions, which reads as a bad model rather than as no data.
        """
        data = recipe / "data" / "train"
        (data / "wav.scp").write_text("spkA-1 ./corpus/absent.wav\n", "utf-8")
        step(
            recipe,
            "make_fbank.py",
            "--conf",
            "conf/fbank.conf",
            "data/train",
            expect=1,
        )

    @posix_only
    def test_a_piped_entry_is_read_relative_to_the_recipe(self, recipe):
        """
        mini-librispeech's entries are `flac -c -d -s ./corpus/... |`, so
        the path inside the command is relative to the recipe directory
        too -- which is exactly what the capture got wrong.
        """
        data = recipe / "data" / "train"
        (data / "wav.scp").write_text(
            "".join(f"{utt} cat ./corpus/{utt}.wav |\n" for utt in UTTERANCES), "utf-8"
        )
        whole_stage(recipe)
        assert sorted(kaldiio.load_scp(str(data / "feats.cmvn.scp"))) == sorted(
            UTTERANCES
        )
