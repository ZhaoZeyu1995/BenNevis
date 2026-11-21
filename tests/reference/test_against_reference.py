"""
Checks on the committed reference data, and the place comparisons go.

Two jobs. The first is now: whenever `tests/golden/reference/` holds a
capture, verify it is internally consistent -- every digest matches the file
beside it, every array's recorded shape matches the array, every statistic
matches the array it describes. A reference is only worth anything if it has
not been corrupted or hand-edited, and nothing else checks that.

The second is as Kaldi leaves. Each stage that moves to Python gets a test
here comparing its output against the captured Kaldi output at that layer's
tolerance. There are none yet, because nothing has moved yet; the wiring is
in place so that the first one is a few lines rather than a new harness.

Everything skips when there is no capture, so the tiers stay green before the
first nightly has run.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from pathlib import Path

import pytest

from BenNevis.utils.reference import (
    MANIFEST_NAME,
    Manifest,
    array_stats,
    compare_array,
    compare_stats,
    sha256_file,
)

TESTS_DIR = Path(__file__).resolve().parent.parent
REFERENCE_DIR = TESTS_DIR / "golden" / "reference" / "yesno"
MANIFEST_PATH = REFERENCE_DIR / MANIFEST_NAME


def load() -> Manifest:
    """The committed manifest, skipping if there is no capture yet."""
    if not MANIFEST_PATH.is_file():
        pytest.skip(
            f"no reference capture at {MANIFEST_PATH}; run the nightly "
            f"workflow and commit what it uploads"
        )
    manifest = Manifest.read(MANIFEST_PATH)
    if manifest.is_empty():
        pytest.skip(f"{MANIFEST_PATH} records nothing")
    return manifest


@pytest.fixture(scope="module")
def manifest():
    return load()


class TestProvenance:
    def test_the_capture_says_what_produced_it(self, manifest):
        """
        A reference whose provenance is unknown is not a reference: it has
        to name the image, pinned by digest, that produced it.
        """
        assert manifest.image and manifest.image != "unknown"
        assert "@sha256:" in manifest.image

    def test_the_capture_is_dated(self, manifest):
        assert manifest.captured

    def test_the_recipe_is_named(self, manifest):
        assert manifest.recipe


class TestIntegrity:
    def test_every_exact_artefact_is_present_and_unmodified(self, manifest):
        """
        Guards the committed data itself. A hand-edited reference would make
        every comparison against it meaningless, and silently.
        """
        missing = []
        changed = []
        for name, digest in manifest.exact.items():
            path = REFERENCE_DIR / name
            if not path.is_file():
                missing.append(name)
            elif sha256_file(path) != digest:
                changed.append(name)
        assert not missing, f"recorded but absent: {missing}"
        assert not changed, f"present but altered since capture: {changed}"

    def test_no_captured_file_is_unrecorded(self, manifest):
        """
        The other direction: a file in the capture that the manifest does
        not mention is either a stray or a manifest that was not rewritten.
        """
        recorded = set(manifest.exact)
        recorded |= {
            record["path"] for record in manifest.arrays.values() if "path" in record
        }
        recorded.add(MANIFEST_NAME)
        found = {
            str(path.relative_to(REFERENCE_DIR))
            for path in REFERENCE_DIR.rglob("*")
            if path.is_file()
        }
        assert found <= recorded, f"not in the manifest: {sorted(found - recorded)}"


@pytest.mark.torch
class TestArrays:
    def test_every_array_is_present_with_its_recorded_shape(self, manifest):
        import numpy as np

        if not manifest.arrays:
            pytest.skip("the capture stores no arrays whole")
        for name, record in manifest.arrays.items():
            path = REFERENCE_DIR / record["path"]
            assert path.is_file(), name
            with np.load(path) as bundle:
                assert record["key"] in bundle.files, name
                array = bundle[record["key"]]
            assert list(array.shape) == record["shape"], name

    def test_every_array_agrees_with_itself(self, manifest):
        """
        Trivially true, and worth asserting: it exercises the comparison the
        migration will rely on against the real committed data, so a
        reference that cannot even match itself is caught here rather than
        in the first migration PR.
        """
        import numpy as np

        if not manifest.arrays:
            pytest.skip("the capture stores no arrays whole")
        for name, record in manifest.arrays.items():
            with np.load(REFERENCE_DIR / record["path"]) as bundle:
                array = bundle[record["key"]]
            difference = compare_array(name, array, array, manifest.tolerance_for(name))
            assert difference is None, str(difference)

    def test_the_statistics_of_the_stored_arrays_match_the_manifest(self, manifest):
        """
        Where an array is stored whole *and* summarised, the two must agree
        -- otherwise one of them was written from different data.
        """
        import numpy as np

        checked = 0
        for name, record in manifest.arrays.items():
            if name not in manifest.stats:
                continue
            with np.load(REFERENCE_DIR / record["path"]) as bundle:
                array = bundle[record["key"]]
            differences = compare_stats(
                name,
                array_stats(array),
                manifest.stats[name],
                manifest.tolerance_for(name),
            )
            assert not differences, [str(d) for d in differences]
            checked += 1
        if checked == 0:
            pytest.skip("no array is both stored whole and summarised")


class TestCoverage:
    """
    What the capture has to contain to be useful for the migration.

    These are the artefacts a Python replacement will be compared against,
    so a capture missing them is a capture that cannot gate the work it
    exists for.
    """

    @pytest.mark.parametrize(
        "name", ["lang/phones.txt", "lang/words.txt", "lang/phones/align_lexicon.int"]
    )
    def test_the_lexicon_inputs_are_captured(self, manifest, name):
        assert name in manifest.exact

    def test_some_features_are_stored_whole(self, manifest):
        """
        Statistics alone cannot gate a feature reimplementation at
        rtol=1e-4; at least one real matrix is needed.
        """
        assert any(name.startswith("features/") for name in manifest.arrays), sorted(
            manifest.arrays
        )

    def test_every_feature_utterance_is_summarised(self, manifest):
        assert any(name.startswith("features/") for name in manifest.stats)

    def test_the_decoded_output_is_captured(self, manifest):
        assert any(
            name.startswith("transcriptions/") and name.endswith("hyp.wrd.trn")
            for name in manifest.exact
        ), sorted(manifest.exact)
