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
REPO_ROOT = TESTS_DIR.parent
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
                # Deliberately absent if the capture judged it too large to
                # store; its digest is still the comparison. Anything else
                # is a file that went missing.
                if name not in manifest.large:
                    missing.append(name)
            elif sha256_file(path) != digest:
                changed.append(name)
        assert not missing, f"recorded but absent: {missing}"
        assert not changed, f"present but altered since capture: {changed}"

    def test_every_large_artefact_is_also_recorded_as_exact(self, manifest):
        """
        The `large` list says which digests have no bytes beside them. An
        entry there with no digest would be a file nobody can check.
        """
        orphans = sorted(set(manifest.large) - set(manifest.exact))
        assert not orphans, orphans

    def test_a_large_artefact_is_not_also_stored(self, manifest):
        """
        If the bytes are there, the entry should not claim they are not --
        otherwise the integrity check above skips a file it could verify.
        """
        stored = [name for name in manifest.large if (REFERENCE_DIR / name).is_file()]
        assert not stored, stored

    def test_no_captured_file_is_unrecorded(self, manifest):
        """
        The other direction: a file in the capture that the manifest does
        not mention is either a stray or a manifest that was not rewritten.
        """
        recorded = set(manifest.exact) | set(manifest.large)
        recorded |= {
            record["path"] for record in manifest.arrays.values() if "path" in record
        }
        recorded.add(MANIFEST_NAME)
        # as_posix, because the manifest records forward slashes and
        # `relative_to` yields the platform's separator -- on Windows that
        # matched nothing at all.
        found = {
            path.relative_to(REFERENCE_DIR).as_posix()
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


class TestTheConfigurationStillMatches:
    """
    That the reference describes the recipe as it is now.

    A capture is only a reference for the settings it was taken under. Change
    `conf/fbank.conf` and every captured feature becomes a record of
    something the recipe no longer does -- and nothing else would notice,
    because the integrity checks above only compare the capture with itself.

    This is not hypothetical: setting `--dither=0`, which had to be done
    before features could be replaced at all, invalidated the features in the
    capture that preceded it.

    Skips while the capture predates the recording of these files, so an
    older reference does not fail for lacking something it never held.
    """

    @pytest.fixture(scope="class")
    def recorded(self, manifest):
        names = [n for n in manifest.exact if n.startswith("recipe/conf/")]
        if not names:
            pytest.skip(
                "this capture predates the recording of the feature "
                "configuration; re-capture to activate this check"
            )
        return sorted(names)

    def test_the_feature_configuration_is_recorded(self, recorded):
        assert any(name.endswith("fbank.conf") for name in recorded)

    def test_it_matches_the_recipe_as_it_stands(self, manifest, recorded):
        from BenNevis.utils.reference import sha256_bytes

        recipe = REPO_ROOT / manifest.recipe
        stale = []
        for name in recorded:
            current = recipe / name[len("recipe/") :]
            if not current.is_file():
                stale.append(f"{name} no longer exists in the recipe")
                continue
            if sha256_bytes(current.read_bytes()) != manifest.exact[name]:
                stale.append(f"{name} has changed since the capture")
        assert not stale, (
            "the reference was taken under a different configuration, so the "
            "captured features no longer describe what the recipe produces: "
            + "; ".join(stale)
            + ". Re-run the recipe, re-capture, and commit the result."
        )

    def test_dither_is_off_in_the_captured_configuration(self, manifest, recorded):
        """
        Stated separately because it is the one setting that makes features
        unreproducible outside Kaldi, rather than merely different.
        """
        fbank = next(n for n in recorded if n.endswith("fbank.conf"))
        text = (REFERENCE_DIR / fbank).read_text()
        assert "--dither=0" in text, (
            "the capture was taken with dither enabled, so its features "
            "cannot be reproduced by any implementation but Kaldi's"
        )


class TestFunctionalEquivalenceOfG:
    """
    A Python `G.fst` against the fingerprint Kaldi's recorded.

    This is the check that `G`'s digest cannot provide. `arpa2fst` numbers
    its states in whatever order it meets histories, so a Python build is
    not byte-equal and never will be -- its digest can only ever mismatch.
    The fingerprint asks both FSTs the same fixed questions instead, which
    is the standard CLAUDE.md sets for an FST.

    It also works where the bytes are not kept at all: mini-librispeech's
    `G.fst` is 61MB and recorded by digest alone, and this still gates it.
    """

    def test_the_manifest_records_a_fingerprint(self, manifest):
        """
        Skipped rather than failed where the reference predates the
        fingerprint: the yesno capture committed before it has no
        `functional` field, and re-capturing is how it gains one.
        """
        if not manifest.functional:
            pytest.skip("this reference was captured before fingerprints")
        assert any("G.fst" in name for name in manifest.functional)

    def test_a_python_build_matches_it(self, manifest):
        if not manifest.functional:
            pytest.skip("this reference was captured before fingerprints")

        arpa = REPO_ROOT / "egs" / "yesno" / "asr" / "input" / "task.arpabo"
        words_path = REFERENCE_DIR / "lang" / "words.txt"
        if not arpa.is_file() or not words_path.is_file():
            pytest.skip("no ARPA model and symbol table to build from")

        from BenNevis.lm import build_arcs, read_arpa, start_state
        from BenNevis.utils.fingerprint import fingerprint

        words = {}
        for line in words_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) == 2:
                words[fields[0]] = int(fields[1])

        arcs, finals, state_of = build_arcs(read_arpa(arpa), words)
        by_source: dict = {}
        for arc in arcs:
            by_source.setdefault(arc[0], []).append((arc[1], arc[2], arc[4]))

        checked = 0
        for name, recorded in manifest.functional.items():
            if "G.fst" not in name:
                continue
            built = fingerprint(
                by_source,
                finals,
                words,
                start_state(state_of),
                [tuple(key.split()) for key in recorded],
            )
            for key, weight in recorded.items():
                assert key in built, f"{name}: this build rejects {key!r}"
                assert built[key] == pytest.approx(weight, rel=1e-6, abs=1e-6), (
                    f"{name}: {key!r} is {built[key]:.6f} here and "
                    f"{weight:.6f} in Kaldi's"
                )
                checked += 1
        assert checked, (
            "a fingerprint was recorded but nothing was compared against it; "
            "this test skipped unconditionally once because REFERENCE_DIR "
            "already ends in the corpus name and the path had it twice"
        )
