"""
The stages compose: what stage 2 writes is what the trainer can read.

Tier 2, because `Dataset` builds a `Lang` and `BenNevis.core.lang`
imports k2. Tier 2 runs on every push, so this is cheap insurance in the
right place.

**This test exists because a regression shipped that every other test
missed.** `run/make_fbank.py` did not write `utt2dur`, which
`BenNevis.core.Dataset` requires -- it filters on `min_duration` and
`max_duration`, sorts by it, and the duration samplers batch on it -- so
for two commits on master the recipes could not start training while the
whole suite stayed green. `tests/recipes/test_stage2.py` checked the
archive, the frame counts, the normalisation and the compression; what
nothing did was **construct the thing that reads them**.

So this is deliberately not another test of any one stage. It runs the
real entry points over a synthetic three-utterance corpus -- prepare the
language directory, the topology, the features, the statistics, fix the
directory, normalise -- and then asks `Dataset` to load the result and
hand over a sample. Any file a stage forgets to write, or writes under
the wrong name, or writes in a form the reader cannot parse, fails here.

What it deliberately does **not** check is any value: the WERs, the
feature arithmetic and the table bytes are gated elsewhere and much more
precisely. This checks that the pipeline fits together.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import subprocess
import sys
import wave
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("torchaudio")
pytest.importorskip("kaldiio")
pytest.importorskip("k2")

pytestmark = pytest.mark.k2

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUN = REPO_ROOT / "BenNevis" / "run"

#: Three utterances over a two-word lexicon, which is enough for every
#: file in the chain to exist and be read.
UTTERANCES = {
    "spkA-001": "YES NO",
    "spkA-002": "NO NO",
    "spkB-001": "YES YES",
}


def write_wave(path, seconds=2.0, rate=8000, seed=0):
    rng = np.random.default_rng(seed)
    samples = (rng.standard_normal(int(rate * seconds)) * 3000).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())


def step(recipe, *argv, module=False):
    """Run one entry point from the recipe directory, as a recipe does."""
    command = [sys.executable]
    command += ["-m", *argv] if module else [str(RUN / argv[0]), *argv[1:]]
    done = subprocess.run(
        [str(part) for part in command],
        cwd=recipe,
        capture_output=True,
        text=True,
        env={**_environment()},
    )
    assert (
        done.returncode == 0
    ), f"{argv[0]} exited {done.returncode}\n--- stderr\n{done.stderr[-3000:]}"
    return done


def _environment():
    import os

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    return environment


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """
    A recipe directory taken through stages 1, 2 and 3.

    Module-scoped: the chain is six subprocesses and nothing below
    mutates it.
    """
    recipe = tmp_path_factory.mktemp("recipe")
    (recipe / "conf").mkdir()
    (recipe / "corpus").mkdir()
    data = recipe / "data" / "train"
    data.mkdir(parents=True)

    (recipe / "conf" / "fbank.conf").write_text(
        "--num-mel-bins=80\n--sample-frequency=8000\n--dither=0\n", "utf-8"
    )

    # Stage 1, by hand: the corpus-specific part is per recipe, so what
    # is reproduced here is its *output*, which is what stage 2 reads.
    for index, utterance in enumerate(sorted(UTTERANCES)):
        write_wave(recipe / "corpus" / f"{utterance}.wav", seed=index)
    (data / "wav.scp").write_text(
        "".join(f"{u} ./corpus/{u}.wav\n" for u in sorted(UTTERANCES)), "utf-8"
    )
    (data / "text").write_text(
        "".join(f"{u} {UTTERANCES[u]}\n" for u in sorted(UTTERANCES)), "utf-8"
    )
    (data / "utt2spk").write_text(
        "".join(f"{u} {u.split('-')[0]}\n" for u in sorted(UTTERANCES)), "utf-8"
    )
    (data / "spk2utt").write_text("spkA spkA-001 spkA-002\nspkB spkB-001\n", "utf-8")

    dictionary = recipe / "data" / "local" / "dict"
    dictionary.mkdir(parents=True)
    (dictionary / "lexicon.txt").write_text(
        "<UNK> SIL\nYES Y\nNO N\n", encoding="utf-8"
    )
    (dictionary / "silence_phones.txt").write_text("SIL\n", encoding="utf-8")
    (dictionary / "nonsilence_phones.txt").write_text("Y\nN\n", encoding="utf-8")
    (dictionary / "optional_silence.txt").write_text("SIL\n", encoding="utf-8")

    # The language directory, then the topology -- stage 1's tail and
    # stage 3, in the recipes' own order and with their own flags.
    step(
        recipe,
        "BenNevis.lm",
        "prepare-lang",
        "--position-dependent-phones",
        "false",
        "--sil-prob",
        "0.0",
        "--compile-fsts",
        "false",
        "data/local/dict",
        "<UNK>",
        "data/lang",
        module=True,
    )
    step(
        recipe,
        "prepare_topo.py",
        "--lexicon",
        "data/local/dict/lexicon.txt",
        "--sil-prob",
        "0.0",
        "--position-dependent-phones",
        "false",
        "--compile-fsts",
        "false",
        "--topo",
        "ctc",
        "data/lang",
        "data/lang_ctc",
    )

    # Stage 2, exactly as the recipes run it.
    step(
        recipe,
        "make_fbank.py",
        "--conf",
        "conf/fbank.conf",
        "--compress",
        "--write-utt2num-frames",
        "--ark-dir",
        "data/train/fbank",
        "data/train",
    )
    step(recipe, "compute_cmvn_stats.py", "--output", "data/train/cmvn", "data/train")
    step(recipe, "fix_data_dir.py", "data/train")
    step(recipe, "apply_cmvn.py", "data/train")
    return recipe


class TestTheDataDirectoryIsComplete:
    @pytest.mark.parametrize(
        "name",
        [
            "wav.scp",
            "text",
            "utt2spk",
            "spk2utt",
            "feats.scp",
            "feats.cmvn.scp",
            "cmvn.scp",
            "utt2num_frames",
            "utt2dur",
        ],
    )
    def test_the_file_exists_and_covers_every_utterance(self, prepared, name):
        """
        `utt2dur` is in this list because it was the one missing, and the
        whole directory is listed because the next omission will be a
        different file.
        """
        path = prepared / "data" / "train" / name
        assert path.is_file(), name
        keys = [line.split()[0] for line in path.read_text().splitlines()]
        expected = sorted(UTTERANCES) if name != "spk2utt" else ["spkA", "spkB"]
        if name == "cmvn.scp":
            expected = ["spkA", "spkB"]
        assert keys == expected, name


class TestTheTrainerCanReadIt:
    @pytest.fixture(scope="class")
    @staticmethod
    def dataset(prepared):
        """
        The reader the regression broke. `load_feats=True` because that
        is what every recipe but `wav2vec2` configures.
        """
        sys.path.insert(0, str(REPO_ROOT))
        from BenNevis.core.dataset import Dataset

        return Dataset(
            str(prepared / "data" / "train"),
            str(prepared / "data" / "lang_ctc"),
            load_feats=True,
        )

    def test_it_loads_every_utterance(self, dataset):
        assert sorted(dataset.uttids) == sorted(UTTERANCES)

    def test_the_durations_and_frame_counts_are_populated(self, dataset):
        """
        Both are read with a `mapping`, so a missing file raises and a
        malformed one raises differently. This is the assertion the
        `utt2dur` regression would have failed on.
        """
        for utterance in UTTERANCES:
            assert dataset.utt2dur[utterance] > 0.0
            assert dataset.utt2num_frames[utterance] > 0

    def test_the_duration_is_consistent_with_the_frame_count(self, dataset):
        """
        Not an arithmetic check -- the two are measured differently and
        deliberately so -- but they must describe the same utterance to
        within a frame or two, or one of them is being read from the
        wrong file.
        """
        for utterance in UTTERANCES:
            implied = dataset.utt2num_frames[utterance] * 0.01
            assert abs(dataset.utt2dur[utterance] - implied) < 0.05, utterance

    def test_a_sample_has_its_features_and_its_target(self, dataset):
        sample = dataset[0]
        assert sample["feats"].shape[1] == 80
        assert sample["feats_len"] == sample["feats"].shape[0]
        assert sample["target_length"] > 0
        assert sample["target"].numel() == sample["target_length"]
        assert sample["name"] in UTTERANCES
        assert sample["text"] == UTTERANCES[sample["name"]]

    def test_every_sample_loads(self, dataset):
        """
        Not just the first: the archive is read by offset, so a wrong
        offset shows on the second utterance rather than the first.
        """
        for index in range(len(dataset)):
            sample = dataset[index]
            assert sample["feats"].shape[0] > 0, sample["name"]

    def test_the_features_are_the_normalised_ones(self, dataset):
        """
        `Dataset` reads `feats.cmvn.scp`, not `feats.scp`. Reading the
        un-normalised archive would train on features whose mean is about
        12 -- a model that still trains, slightly worse, with no error.
        """
        import torch

        stacked = torch.cat([dataset[i]["feats"] for i in range(len(dataset))])
        assert abs(float(stacked.mean())) < 0.5, float(stacked.mean())

    def test_the_duration_filters_work(self, prepared):
        """
        The reason `utt2dur` is required rather than optional: these
        filters are what the recipes set, and with no durations they
        would silently keep everything.
        """
        from BenNevis.core.dataset import Dataset

        kept = Dataset(
            str(prepared / "data" / "train"),
            str(prepared / "data" / "lang_ctc"),
            load_feats=True,
            min_duration=1.0,
        )
        dropped = Dataset(
            str(prepared / "data" / "train"),
            str(prepared / "data" / "lang_ctc"),
            load_feats=True,
            min_duration=100.0,
        )
        assert len(kept) == len(UTTERANCES)
        assert len(dropped) == 0
