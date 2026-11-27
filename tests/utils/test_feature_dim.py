"""
Measuring a data directory's feature dimension, so `idim` need not be written.

Tier 1b: it needs `kaldiio` and numpy, which arrive with torch.

`odim` has always been derived -- `run/train.py` fills it in from the
language directory, so a topology change cannot desynchronise it -- while
`idim` was a number in each recipe's `conf/model/*.yaml`. Three recipes had
it wrong: `egs/librispeech/asr{1,2,3}` said 83 while running
`steps/make_fbank.sh`, which writes 80. That is a shape mismatch at the
first forward pass, and nothing caught it because those recipes have never
run in CI.

So the number is gone and the dimension is measured. The tests that matter
here are the two failure modes, because both would otherwise surface a long
way from the cause: an archive that is not there, which is what a run that
skipped feature extraction hits, and an archive that is empty, which would
give a model a zero-wide input layer and a run that trains on nothing.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import pytest

np = pytest.importorskip("numpy")
kaldiio = pytest.importorskip("kaldiio")

from BenNevis.utils.data import FEATURES_SCP, feature_dim  # noqa: E402

pytestmark = pytest.mark.torch


def write_features(directory, matrices, scp=FEATURES_SCP):
    directory.mkdir(parents=True, exist_ok=True)
    kaldiio.save_ark(
        str(directory / "feats.cmvn.ark"),
        matrices,
        scp=str(directory / scp),
    )
    return directory


class TestItMeasuresTheArchive:
    def test_eighty_dimensional_features_measure_eighty(self, tmp_path):
        data = write_features(
            tmp_path / "data",
            {"u1": np.zeros((7, 80), dtype=np.float32)},
        )
        assert feature_dim(str(data)) == 80

    @pytest.mark.parametrize("dim", [23, 40, 80, 83])
    def test_whatever_the_width_is(self, tmp_path, dim):
        data = write_features(
            tmp_path / f"data{dim}",
            {"u1": np.zeros((3, dim), dtype=np.float32)},
        )
        assert feature_dim(str(data)) == dim

    def test_it_reads_one_utterance_not_the_archive(self, tmp_path):
        """
        `kaldiio.load_scp_sequential` is lazy and this returns on the first
        entry, so a five-hour corpus costs a seek. Checked by making every
        utterance after the first unreadable: a pass over the archive would
        fail on them.
        """
        data = write_features(
            tmp_path / "data",
            {f"u{i}": np.zeros((3, 80), dtype=np.float32) for i in range(5)},
        )
        lines = (data / FEATURES_SCP).read_text(encoding="utf-8").splitlines()
        (data / FEATURES_SCP).write_text(
            "\n".join([lines[0]] + [f"u{i} /nonexistent:0" for i in range(1, 5)])
            + "\n",
            encoding="utf-8",
        )
        assert feature_dim(str(data)) == 80


class TestItFailsWhereTheCauseIs:
    def test_a_missing_archive_names_the_stage_that_writes_it(self, tmp_path):
        (tmp_path / "data").mkdir()
        with pytest.raises(FileNotFoundError, match="make_fbank"):
            feature_dim(str(tmp_path / "data"))

    def test_an_empty_archive_is_refused(self, tmp_path):
        data = tmp_path / "data"
        data.mkdir()
        (data / FEATURES_SCP).write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            feature_dim(str(data))

    def test_a_one_dimensional_entry_is_refused(self, tmp_path):
        """
        A vector where a matrix was expected would otherwise be indexed as
        `shape[1]` and raise an IndexError from inside this function.
        """
        data = write_features(tmp_path / "data", {"u1": np.zeros(80, dtype=np.float32)})
        with pytest.raises(ValueError, match="not a"):
            feature_dim(str(data))


class TestTheArchiveItMeasuresIsTheOneTrainingReads:
    def test_the_default_is_the_dataset_s_archive(self):
        """
        `Dataset` reads `feats.cmvn.scp` when `load_feats` is set. If this
        measured `feats.scp` instead it would report the dimension of the
        un-normalised features, which happens to be the same number today
        and would stop being so the moment normalisation changed the width.
        """
        import inspect

        from BenNevis.core import dataset

        assert FEATURES_SCP == "feats.cmvn.scp"
        assert FEATURES_SCP in inspect.getsource(dataset)


class TestTrainHonoursAndChecksAnExplicitValue:
    """
    An explicit `idim` is still allowed, and is **checked** against the data.

    Pinning a dimension stays possible -- it just cannot be silently
    wrong, which is the whole point. Read out of `run/train.py` rather than
    executed, because reaching that line needs a process group, a language
    directory and a model.
    """

    @staticmethod
    def source():
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent.parent
        return (root / "BenNevis" / "run" / "train.py").read_text(encoding="utf-8")

    def test_it_derives_when_the_configuration_says_null(self):
        text = self.source()
        assert 'feature_dim(cfg.data["train_ds"])' in text
        assert 'cfg.model["kwargs"]["idim"] = measured' in text

    def test_it_raises_on_a_disagreement_rather_than_trusting_either(self):
        text = self.source()
        assert "shape" in text and "mismatch" in text
        assert "int(configured) != measured" in text

    def test_it_only_measures_when_features_are_loaded(self):
        """
        `Wav2Vec2Model` takes the waveform, so there is no feature archive
        to measure and asking for one would fail a perfectly good run.
        """
        text = self.source()
        assert 'if cfg.data["load_feats"] and "idim" in cfg.model["kwargs"]:' in text
