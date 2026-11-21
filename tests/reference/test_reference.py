"""
Tests for the reference manifest and the comparison it drives.

This is the instrument the Kaldi removal will be measured with, so it has to
be right before any measurement it produces means anything. Two things in
particular are worth pinning:

- A tolerance must actually reject. A comparison that passes whatever it is
  given would make every subsequent migration PR look correct.
- Counts are compared exactly even when the values around them are not. A
  different number of frames is a framing bug, not numerical noise, and a
  tolerance that absorbed it would hide the clearest signal available.

The array tests carry the `torch` marker because they need numpy, which tier
1 does not install. The manifest and digest tests need nothing and run
everywhere.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import json

import pytest

from BenNevis.utils.reference import (
    EXACT_TOLERANCE,
    FEATURE_TOLERANCE,
    MANIFEST_NAME,
    Difference,
    Manifest,
    Tolerance,
    array_stats,
    compare_array,
    compare_exact,
    compare_stats,
    sha256_bytes,
    sha256_file,
)


class TestTolerance:
    def test_the_feature_tolerance_is_the_documented_one(self):
        """
        rtol=1e-4 against Kaldi, per "Equivalence and tolerances" in
        CLAUDE.md. Tightening it here without measuring would turn correct
        migrations red.
        """
        assert FEATURE_TOLERANCE.rtol == 1e-4
        assert not FEATURE_TOLERANCE.is_exact

    def test_the_exact_tolerance_admits_nothing(self):
        assert EXACT_TOLERANCE.is_exact
        assert Tolerance(rtol=1e-12).is_exact is False


class TestDigests:
    def test_bytes_and_file_agree(self, tmp_path):
        payload = b"0 1 2 3 0.0\n1\n"
        path = tmp_path / "T.fst.txt"
        path.write_bytes(payload)
        assert sha256_file(path) == sha256_bytes(payload)

    def test_a_large_file_is_read_in_chunks(self, tmp_path):
        """The chunked read must give the same answer as one pass."""
        payload = bytes(range(256)) * 20000  # ~5MB, past the 1MB block
        path = tmp_path / "big.ark"
        path.write_bytes(payload)
        assert sha256_file(path) == sha256_bytes(payload)

    def test_empty_is_not_an_error(self, tmp_path):
        path = tmp_path / "empty"
        path.write_bytes(b"")
        assert sha256_file(path) == sha256_bytes(b"")


class TestCompareExact:
    def test_identical_content_matches(self):
        payload = b"<eps> 0\na 1\n"
        assert compare_exact("tokens", payload, sha256_bytes(payload)) is None

    def test_one_changed_character_is_caught(self):
        reference = sha256_bytes(b"<eps> 0\na 1\n")
        difference = compare_exact("tokens", b"<eps> 0\na 2\n", reference)
        assert difference is not None
        assert "content differs" in difference.detail

    def test_the_message_names_what_was_compared(self):
        difference = compare_exact("graph/ctc/TLG.fst.txt", b"x", "0" * 64)
        assert str(difference).startswith("graph/ctc/TLG.fst.txt: ")
        assert "1 bytes produced" in difference.detail


class TestManifest:
    def test_round_trips(self, tmp_path):
        manifest = Manifest(
            corpus="yesno",
            captured="2026-11-21",
            image="kaldiasr/kaldi@sha256:" + "a" * 64,
            recipe="egs/yesno/asr",
            notes=["captured from a finished run"],
            exact={"lang/phones.txt": "b" * 64},
            arrays={
                "features/test/u1": {
                    "path": "features/test.npz",
                    "key": "u1",
                    "shape": [4, 3],
                    "tolerance": {"rtol": 1e-4, "atol": 1e-5},
                }
            },
            stats={"features/test/u2": {"frames": 7, "dim": 3, "mean": 0.5}},
        )
        path = tmp_path / MANIFEST_NAME
        manifest.write(path)
        assert Manifest.read(path) == manifest

    def test_is_written_as_a_readable_diff(self, tmp_path):
        """
        Indented and key-sorted, so that a capture appears in review as
        something a person can read rather than one enormous line.
        """
        path = tmp_path / MANIFEST_NAME
        Manifest(corpus="yesno", exact={"b": "1", "a": "2"}).write(path)
        text = path.read_text()
        assert text.endswith("\n")
        assert "\n  " in text
        assert text.index('"a"') < text.index('"b"')

    def test_tolerance_comes_from_the_manifest(self, tmp_path):
        manifest = Manifest(arrays={"f": {"tolerance": {"rtol": 0.25, "atol": 0.5}}})
        assert manifest.tolerance_for("f") == Tolerance(rtol=0.25, atol=0.5)

    def test_an_unknown_array_is_exact_rather_than_lenient(self):
        """
        The safe default. A missing tolerance meaning "anything goes" would
        silently excuse whatever it was applied to.
        """
        assert Manifest().tolerance_for("absent").is_exact

    def test_an_unrecognised_key_is_refused(self, tmp_path):
        path = tmp_path / MANIFEST_NAME
        path.write_text(json.dumps({"corpus": "yesno", "sneaky": 1}))
        with pytest.raises(ValueError, match="unrecognised keys"):
            Manifest.read(path)

    def test_a_non_object_is_refused(self, tmp_path):
        path = tmp_path / MANIFEST_NAME
        path.write_text("[]")
        with pytest.raises(ValueError, match="expected a JSON object"):
            Manifest.read(path)

    def test_emptiness_is_reported(self):
        assert Manifest(corpus="yesno").is_empty()
        assert not Manifest(exact={"a": "b"}).is_empty()
        assert not Manifest(stats={"a": {}}).is_empty()

    def test_names_are_sorted(self):
        manifest = Manifest(
            exact={"b": "", "a": ""},
            arrays={"d": {}, "c": {}},
            stats={"f": {}, "e": {}},
        )
        assert manifest.names() == (["a", "b"], ["c", "d"], ["e", "f"])


@pytest.mark.torch
class TestArrayStats:
    def test_summarises_a_matrix(self):
        import numpy as np

        array = np.arange(12, dtype=np.float32).reshape(4, 3)
        stats = array_stats(array)
        assert stats["frames"] == 4
        assert stats["dim"] == 3
        assert stats["mean"] == pytest.approx(5.5)
        assert stats["min"] == pytest.approx(0.0)
        assert stats["max"] == pytest.approx(11.0)
        assert stats["first_row_sum"] == pytest.approx(3.0)

    def test_the_first_row_sum_catches_a_time_reversal(self):
        """
        Every whole-matrix moment is blind to reversing the frames, which is
        a real and easy mistake to make in a reimplementation. The first
        row's sum is not.
        """
        import numpy as np

        array = np.arange(12, dtype=np.float32).reshape(4, 3)
        forward = array_stats(array)
        reversed_ = array_stats(array[::-1])

        for key in ("frames", "dim", "mean", "std", "min", "max"):
            assert forward[key] == pytest.approx(reversed_[key]), key
        assert forward["first_row_sum"] != pytest.approx(reversed_["first_row_sum"])

    def test_a_one_dimensional_array_is_refused(self):
        import numpy as np

        with pytest.raises(ValueError, match="2-D"):
            array_stats(np.zeros(5))


@pytest.mark.torch
class TestCompareArray:
    def test_identical_arrays_agree(self):
        import numpy as np

        array = np.arange(6, dtype=np.float64).reshape(2, 3)
        assert compare_array("f", array, array, FEATURE_TOLERANCE) is None

    def test_floating_point_noise_is_allowed(self):
        import numpy as np

        reference = np.full((3, 4), 10.0)
        nudged = reference * (1 + 1e-7)
        assert compare_array("f", nudged, reference, FEATURE_TOLERANCE) is None

    def test_a_real_deviation_is_caught(self):
        """
        The important one: the tolerance has to reject something. A
        comparison that passed whatever it was given would make every
        migration look correct.
        """
        import numpy as np

        reference = np.full((3, 4), 10.0)
        wrong = reference.copy()
        wrong[1, 2] = 10.1
        difference = compare_array("f", wrong, reference, FEATURE_TOLERANCE)
        assert difference is not None
        assert "(1, 2)" in difference.detail
        assert "1 of 12 elements" in difference.detail

    def test_the_message_carries_the_measured_values(self):
        import numpy as np

        reference = np.zeros((1, 2))
        difference = compare_array(
            "f", np.array([[0.0, 5.0]]), reference, FEATURE_TOLERANCE
        )
        assert "5" in difference.detail
        assert "allowed" in difference.detail

    def test_a_shape_mismatch_is_reported_as_such(self):
        import numpy as np

        difference = compare_array(
            "f", np.zeros((2, 3)), np.zeros((4, 3)), FEATURE_TOLERANCE
        )
        assert difference is not None
        assert "shape" in difference.detail
        assert "(2, 3)" in difference.detail and "(4, 3)" in difference.detail

    def test_near_zero_is_governed_by_atol(self):
        """
        A relative tolerance is useless around zero, which is where
        normalised features spend much of their time.
        """
        import numpy as np

        reference = np.zeros((2, 2))
        assert (
            compare_array("f", np.full((2, 2), 1e-6), reference, FEATURE_TOLERANCE)
            is None
        )
        assert (
            compare_array("f", np.full((2, 2), 1e-3), reference, FEATURE_TOLERANCE)
            is not None
        )

    def test_empty_arrays_agree(self):
        import numpy as np

        empty = np.zeros((0, 3))
        assert compare_array("f", empty, empty, FEATURE_TOLERANCE) is None

    def test_an_exact_tolerance_rejects_any_difference(self):
        import numpy as np

        reference = np.ones((2, 2))
        nudged = reference * (1 + 1e-15)
        assert compare_array("f", reference, reference, EXACT_TOLERANCE) is None
        assert compare_array("f", nudged, reference, EXACT_TOLERANCE) is not None


@pytest.mark.torch
class TestCompareStats:
    def test_identical_statistics_agree(self):
        import numpy as np

        stats = array_stats(np.arange(12, dtype=np.float32).reshape(4, 3))
        assert compare_stats("f", stats, stats, FEATURE_TOLERANCE) == []

    def test_noise_in_the_moments_is_allowed(self):
        import numpy as np

        reference = array_stats(np.arange(12, dtype=np.float64).reshape(4, 3))
        nudged = dict(reference)
        nudged["mean"] = reference["mean"] * (1 + 1e-7)
        assert compare_stats("f", nudged, reference, FEATURE_TOLERANCE) == []

    def test_a_frame_count_difference_is_never_tolerated(self):
        """
        Counts are compared exactly whatever the tolerance: a different
        number of frames is a framing bug, not numerical noise, and letting
        a tolerance absorb it would hide the clearest signal there is.
        """
        import numpy as np

        reference = array_stats(np.zeros((100, 3)))
        got = array_stats(np.zeros((101, 3)))
        differences = compare_stats("f", got, reference, Tolerance(rtol=1.0, atol=1e6))
        assert [d.name for d in differences] == ["f.frames"]
        assert "compared exactly" in differences[0].detail

    def test_a_dimension_difference_is_never_tolerated(self):
        import numpy as np

        reference = array_stats(np.zeros((10, 83)))
        got = array_stats(np.zeros((10, 80)))
        differences = compare_stats("f", got, reference, Tolerance(rtol=1.0, atol=1e6))
        assert [d.name for d in differences] == ["f.dim"]

    def test_a_shifted_mean_is_caught(self):
        import numpy as np

        reference = array_stats(np.full((10, 3), 1.0))
        got = array_stats(np.full((10, 3), 1.01))
        names = [d.name for d in compare_stats("f", got, reference, FEATURE_TOLERANCE)]
        assert "f.mean" in names

    def test_a_missing_statistic_is_reported(self):
        differences = compare_stats(
            "f",
            {"frames": 4, "dim": 3},
            {"frames": 4, "dim": 3, "mean": 1.0},
            FEATURE_TOLERANCE,
        )
        assert [d.name for d in differences] == ["f.mean"]
        assert differences[0].detail == "not produced"

    def test_statistics_absent_from_the_reference_are_ignored(self):
        """
        An older capture may record fewer statistics. That is a reason to
        compare what is there, not to fail.
        """
        assert (
            compare_stats("f", array_stats_stub(), {"mean": 1.0}, FEATURE_TOLERANCE)
            == []
        )


def array_stats_stub():
    """Statistics with the right mean and nothing else of interest."""
    return {
        "frames": 1,
        "dim": 1,
        "mean": 1.0,
        "std": 0.0,
        "min": 1.0,
        "max": 1.0,
        "first_row_sum": 1.0,
    }


class TestDifference:
    def test_reads_as_a_sentence(self):
        assert str(Difference("a/b", "differs")) == "a/b: differs"
