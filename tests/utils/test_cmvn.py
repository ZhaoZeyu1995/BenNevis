"""
Tests for the Python cepstral mean and variance normalisation.

This is the first stage of the Kaldi removal, so the tests matter as much as
the code: they are the template for every stage that follows. Three kinds
here, in increasing strength:

1. Arithmetic on data whose answer is known by hand.
2. Properties that must hold for any input -- additivity of the statistics,
   what each flag does and does not change.
3. Two checks against the **captured Kaldi reference**, chosen because they
   are not circular. The frame count Kaldi recorded in its statistics is
   cross-checked against the per-utterance frame counts captured separately,
   which pins the one layout fact that would silently break everything if
   wrong; and Kaldi's own normalised output is checked to be normalised,
   which discriminates sharply because unnormalised features here have a
   mean near 10 and a standard deviation near 1.6.

What is **not** tested yet is equivalence: the capture holds only
post-CMVN features, so there is no input to drive this implementation with
and no Kaldi output to compare against element by element. That needs the
raw features, which the capture now records; until a run produces them, this
code is correct in isolation and unproven against Kaldi.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from BenNevis.utils import cmvn  # noqa: E402

pytestmark = pytest.mark.torch

REFERENCE = Path(__file__).resolve().parent.parent / "golden" / "reference" / "yesno"


class TestStatisticsLayout:
    def test_the_shape_is_two_by_dim_plus_one(self):
        stats = cmvn.compute_stats(np.zeros((7, 4)))
        assert stats.shape == (2, 5)

    def test_the_count_is_in_the_first_row_last_column(self):
        """
        Kaldi's layout, and the one fact that would quietly corrupt every
        normalisation if it were misread.
        """
        stats = cmvn.compute_stats(np.zeros((7, 4)))
        assert stats[0, 4] == 7

    def test_the_second_rows_last_column_is_unused(self):
        stats = cmvn.compute_stats(np.ones((7, 4)))
        assert stats[1, 4] == 0.0

    def test_the_sums_and_squares_are_what_they_say(self):
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        stats = cmvn.compute_stats(features)
        assert stats[0, 0] == 4.0 and stats[0, 1] == 6.0
        assert stats[1, 0] == 10.0 and stats[1, 1] == 20.0
        assert stats[0, 2] == 2

    def test_statistics_are_accumulated_in_double_precision(self):
        """
        Sums over tens of thousands of frames; float32 would lose precision
        where keeping it costs nothing.
        """
        assert cmvn.compute_stats(np.ones((3, 2), dtype=np.float32)).dtype == (
            np.float64
        )


class TestComputeStats:
    def test_several_utterances_equal_their_concatenation(self):
        rng = np.random.default_rng(1368)
        parts = [rng.standard_normal((5, 3)), rng.standard_normal((7, 3))]
        together = cmvn.compute_stats(parts)
        pooled = cmvn.compute_stats(np.concatenate(parts, axis=0))
        assert np.allclose(together, pooled)

    def test_a_single_matrix_is_accepted_without_wrapping(self):
        one = np.ones((4, 2))
        assert np.array_equal(cmvn.compute_stats(one), cmvn.compute_stats([one]))

    def test_nothing_to_accumulate_is_refused(self):
        with pytest.raises(ValueError, match="no features"):
            cmvn.compute_stats([])

    def test_disagreeing_dimensions_are_refused(self):
        with pytest.raises(ValueError, match="disagree about the dimension"):
            cmvn.compute_stats([np.zeros((2, 3)), np.zeros((2, 4))])

    def test_a_one_dimensional_input_is_refused(self):
        with pytest.raises(ValueError, match="frames, dim"):
            cmvn.compute_stats(np.zeros(5))


class TestAccumulate:
    def test_adding_statistics_matches_pooling_the_features(self):
        """
        The layout is additive, which is what lets a speaker's utterances be
        processed in pieces.
        """
        rng = np.random.default_rng(7)
        parts = [rng.standard_normal((6, 3)) for _ in range(3)]
        added = cmvn.accumulate([cmvn.compute_stats(p) for p in parts])
        pooled = cmvn.compute_stats(np.concatenate(parts, axis=0))
        assert np.allclose(added, pooled)

    def test_nothing_to_accumulate_is_refused(self):
        with pytest.raises(ValueError, match="no statistics"):
            cmvn.accumulate([])

    def test_disagreeing_shapes_are_refused(self):
        with pytest.raises(ValueError, match="shapes disagree"):
            cmvn.accumulate([np.zeros((2, 4)), np.zeros((2, 5))])


class TestMeanAndVariance:
    def test_the_normaliser_is_the_moments_of_the_data(self):
        rng = np.random.default_rng(99)
        features = rng.standard_normal((500, 4)) * 3.0 + 7.0
        mean, variance = cmvn.mean_and_variance(cmvn.compute_stats(features))
        assert np.allclose(mean, features.mean(axis=0))
        assert np.allclose(variance, features.var(axis=0))

    def test_no_frames_is_refused_rather_than_dividing_by_zero(self):
        """
        Otherwise the normaliser is NaN throughout and silently ruins every
        feature it touches.
        """
        stats = np.zeros((2, 5))
        with pytest.raises(ValueError, match="no mean to normalise"):
            cmvn.mean_and_variance(stats)

    def test_a_constant_dimension_is_floored_rather_than_infinite(self):
        features = np.concatenate(
            [np.random.default_rng(3).standard_normal((20, 1)), np.ones((20, 1))],
            axis=1,
        )
        _, variance = cmvn.mean_and_variance(cmvn.compute_stats(features))
        assert variance[1] == cmvn.VAR_FLOOR
        assert np.isfinite(variance).all()

    def test_the_wrong_shape_is_refused(self):
        with pytest.raises(ValueError, match=r"\(2, dim \+ 1\)"):
            cmvn.mean_and_variance(np.zeros((3, 5)))


class TestApply:
    def test_normalising_by_its_own_statistics_gives_zero_mean_unit_variance(
        self,
    ):
        rng = np.random.default_rng(11)
        features = rng.standard_normal((400, 5)) * 2.5 - 1.0
        out = cmvn.apply(features, cmvn.compute_stats(features))
        assert np.allclose(out.mean(axis=0), 0.0, atol=1e-5)
        assert np.allclose(out.std(axis=0), 1.0, atol=1e-5)

    def test_speaker_statistics_do_not_normalise_each_utterance_separately(
        self,
    ):
        """
        The point of computing statistics apart from applying them: an
        utterance is normalised by its speaker, so its own mean does not
        come out at zero.
        """
        rng = np.random.default_rng(13)
        a = rng.standard_normal((50, 3)) + 5.0
        b = rng.standard_normal((50, 3)) - 5.0
        speaker = cmvn.compute_stats([a, b])
        out = cmvn.apply(a, speaker)
        assert not np.allclose(out.mean(axis=0), 0.0, atol=0.5)

    def test_norm_vars_false_centres_without_scaling(self):
        rng = np.random.default_rng(17)
        features = rng.standard_normal((200, 3)) * 4.0 + 2.0
        stats = cmvn.compute_stats(features)
        out = cmvn.apply(features, stats, norm_vars=False)
        assert np.allclose(out.mean(axis=0), 0.0, atol=1e-4)
        assert np.allclose(out.std(axis=0), features.std(axis=0), rtol=1e-4)

    def test_norm_means_false_scales_without_centring(self):
        rng = np.random.default_rng(19)
        features = rng.standard_normal((200, 3)) * 4.0 + 2.0
        stats = cmvn.compute_stats(features)
        out = cmvn.apply(features, stats, norm_means=False)
        assert np.allclose(out.std(axis=0), 1.0, atol=1e-4)
        assert not np.allclose(out.mean(axis=0), 0.0, atol=0.1)

    def test_both_disabled_is_a_passthrough(self):
        features = np.random.default_rng(23).standard_normal((10, 3))
        out = cmvn.apply(
            features,
            cmvn.compute_stats(features),
            norm_means=False,
            norm_vars=False,
        )
        assert np.allclose(out, features.astype(np.float32))

    def test_the_output_is_float32(self):
        """
        Kaldi's archives hold float32. Returning float64 would make a byte
        comparison against them fail for a reason unrelated to arithmetic.
        """
        features = np.random.default_rng(29).standard_normal((10, 3))
        out = cmvn.apply(features, cmvn.compute_stats(features))
        assert out.dtype == np.float32

    def test_a_dimension_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="statistics describe"):
            cmvn.apply(np.zeros((4, 5)), cmvn.compute_stats(np.zeros((4, 3))))

    def test_a_single_frame_does_not_divide_by_zero(self):
        out = cmvn.apply(np.ones((1, 3)), cmvn.compute_stats(np.ones((1, 3))))
        assert np.isfinite(out).all()


@pytest.mark.skipif(
    not (REFERENCE / "manifest.json").is_file(),
    reason="no committed Kaldi reference",
)
class TestAgainstTheCapturedReference:
    """
    Checks against what Kaldi actually produced, chosen to be non-circular.

    Neither of these is an equivalence test -- that needs the raw features,
    which the capture did not hold when these were written. They pin the two
    things that could be wrong in a way unit tests on synthetic data cannot
    see: the layout, and which normalisations the recipe applies.
    """

    @pytest.fixture(scope="class")
    def manifest(self):
        return json.loads((REFERENCE / "manifest.json").read_text())

    @pytest.mark.parametrize("dataset", ["train_yesno", "test_yesno"])
    def test_the_frame_count_agrees_with_the_captured_utterances(
        self, manifest, dataset
    ):
        """
        Kaldi's statistics say how many frames went into them, and the
        per-utterance statistics were captured separately. Reading the count
        from the wrong cell would corrupt every mean, and this is the only
        check that would notice.
        """
        with np.load(REFERENCE / "cmvn" / f"{dataset}.npz") as bundle:
            stats = bundle["global"]
        dim = stats.shape[1] - 1
        frames = sum(
            entry["frames"]
            for name, entry in manifest["stats"].items()
            if name.startswith(f"features/{dataset}/")
        )
        assert stats[0, dim] == frames

    @pytest.mark.parametrize("dataset", ["train_yesno", "test_yesno"])
    def test_the_dimension_matches_the_features(self, manifest, dataset):
        with np.load(REFERENCE / "cmvn" / f"{dataset}.npz") as bundle:
            stats = bundle["global"]
        dims = {
            entry["dim"]
            for name, entry in manifest["stats"].items()
            if name.startswith(f"features/{dataset}/")
        }
        assert dims == {stats.shape[1] - 1}

    @pytest.mark.parametrize("dataset", ["train_yesno", "test_yesno"])
    def test_the_normaliser_is_usable(self, manifest, dataset):
        """
        Read Kaldi's own statistics through this implementation. A layout
        misunderstanding shows up here as a negative variance or a NaN.
        """
        with np.load(REFERENCE / "cmvn" / f"{dataset}.npz") as bundle:
            stats = bundle["global"]
        mean, variance = cmvn.mean_and_variance(stats)
        assert np.isfinite(mean).all()
        assert (variance > 0).all()
        assert variance.min() > cmvn.VAR_FLOOR, (
            "a variance reached the floor, so the floor is being exercised "
            "by real data and its value must be confirmed against Kaldi"
        )

    @pytest.mark.parametrize("dataset", ["train_yesno", "test_yesno"])
    def test_kaldis_output_was_both_centred_and_scaled(self, dataset):
        """
        That `--norm-vars=true` reaches the features, checked without a
        hardcoded threshold.

        The counterfactual comes from Kaldi's own statistics: had either
        normalisation been skipped, the output would show that speaker's
        mean and standard deviation instead. Those are far away -- a mean
        near 14 against an observed 0.2, and a standard deviation near 2.7
        against an observed 0.9 -- so the test asserts the observation is
        nearer to normalised than to not, which stays discriminating however
        many utterances the capture holds.

        Only a few utterances per set are stored whole, so a tight bound on
        the per-dimension maximum would be sampling noise; the mean across
        dimensions is the stable statistic.
        """
        with np.load(REFERENCE / "features" / f"{dataset}.npz") as bundle:
            pooled = np.concatenate(
                [bundle[key] for key in bundle.files], axis=0
            ).astype(np.float64)
        with np.load(REFERENCE / "cmvn" / f"{dataset}.npz") as bundle:
            mean, variance = cmvn.mean_and_variance(bundle["global"])

        observed_offset = np.abs(pooled.mean(axis=0)).mean()
        uncentred_offset = np.abs(mean).mean()
        assert observed_offset < uncentred_offset / 10, (
            f"mean offset {observed_offset:.3f} is not far enough below the "
            f"{uncentred_offset:.3f} of unnormalised features: the means may "
            f"not have been subtracted"
        )

        observed_scale = pooled.std(axis=0).mean()
        unscaled = np.sqrt(variance).mean()
        assert abs(observed_scale - 1.0) < abs(observed_scale - unscaled), (
            f"a scale of {observed_scale:.3f} is closer to the {unscaled:.3f} "
            f"of unnormalised features than to 1: --norm-vars may not have "
            f"been applied"
        )
