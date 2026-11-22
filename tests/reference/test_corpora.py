"""
Checks that run over **every** committed reference, not just yesno's.

`test_against_reference.py` is written around yesno, whose manifest path is
a module constant. That was fine while yesno was the only capture; with
mini-librispeech committed it would mean 9.9MB of reference data that
nothing reads. These are the corpus-agnostic checks, parametrised over
whatever is in `tests/golden/reference/`.

What each corpus gates, and where, is not uniform and is worth stating:

* **Features and CMVN** are gated here, in tier 1, for every corpus --
  the arrays and statistics are committed, so the comparison needs nothing
  installed. mini-librispeech is what makes this worth having, because its
  filterbank is well conditioned where yesno's is degenerate.
* **Exact artefacts** are gated here by digest.
* **`G.fst`'s fingerprint** is *not* gated here for mini-librispeech:
  checking it means rebuilding `G` from the ARPA model, and that model is a
  13MB download rather than committed data. It is gated in tier 3, where
  the recipe has fetched the LM. yesno's model is in the repository, so
  yesno's fingerprint is gated here once its reference is re-captured.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from pathlib import Path

import pytest

from BenNevis.utils.reference import MANIFEST_NAME, Manifest, sha256_file

TESTS_DIR = Path(__file__).resolve().parent.parent
REFERENCE_ROOT = TESTS_DIR / "golden" / "reference"


def discover():
    """
    Every corpus with a manifest, by name.

    Discovered rather than listed, so committing a capture is enough to put
    it under test -- the failure this guards is reference data that nothing
    reads.
    """
    if not REFERENCE_ROOT.is_dir():
        return []
    return sorted(
        path.name
        for path in REFERENCE_ROOT.iterdir()
        if path.is_dir() and (path / MANIFEST_NAME).is_file()
    )


CORPORA = discover()


@pytest.fixture(params=CORPORA or ["<none>"])
def corpus(request):
    if not CORPORA:
        pytest.skip("no reference capture committed yet")
    name = request.param
    directory = REFERENCE_ROOT / name
    return name, directory, Manifest.read(directory / MANIFEST_NAME)


class TestEveryCorpusIsUnderTest:
    def test_at_least_one_is_committed(self):
        assert CORPORA, "no reference capture in tests/golden/reference"

    def test_yesno_is_among_them(self):
        """
        The corpus tier 3 runs end to end; losing its reference would take
        the WER ledger's backstop with it.
        """
        assert "yesno" in CORPORA

    def test_each_is_discovered_by_its_manifest(self):
        for name in CORPORA:
            assert (REFERENCE_ROOT / name / MANIFEST_NAME).is_file(), name


class TestProvenance:
    def test_it_names_the_image_by_digest(self, corpus):
        """
        A reference whose provenance is unknown is not a reference, and a
        tag that moved underneath the WER ledger would turn it into noise.
        """
        name, _directory, manifest = corpus
        assert manifest.image and manifest.image != "unknown", name
        assert "@sha256:" in manifest.image, name

    def test_it_is_dated(self, corpus):
        name, _directory, manifest = corpus
        assert manifest.captured, name

    def test_it_records_something(self, corpus):
        name, _directory, manifest = corpus
        assert not manifest.is_empty(), name


class TestIntegrity:
    def test_every_stored_artefact_still_matches_its_digest(self, corpus):
        name, directory, manifest = corpus
        changed = [
            relative
            for relative, digest in manifest.exact.items()
            if (directory / relative).is_file()
            and sha256_file(directory / relative) != digest
        ]
        assert not changed, f"{name}: altered since capture: {changed}"

    def test_an_absent_artefact_is_one_recorded_as_large(self, corpus):
        name, directory, manifest = corpus
        missing = [
            relative
            for relative in manifest.exact
            if not (directory / relative).is_file() and relative not in manifest.large
        ]
        assert not missing, f"{name}: recorded but absent: {missing}"

    def test_a_large_artefact_has_a_digest(self, corpus):
        name, _directory, manifest = corpus
        orphans = sorted(set(manifest.large) - set(manifest.exact))
        assert not orphans, f"{name}: {orphans}"

    def test_a_large_artefact_is_not_also_stored(self, corpus):
        name, directory, manifest = corpus
        stored = [
            relative for relative in manifest.large if (directory / relative).is_file()
        ]
        assert not stored, f"{name}: {stored}"


class TestWhatTheCaptureHasToContain:
    def test_the_symbol_table_is_there_whole(self, corpus):
        """
        An input rather than an output: every FST in the directory is
        written against these integer labels, so a replacement cannot be
        driven without them. #38 is where this was got wrong.
        """
        name, directory, manifest = corpus
        if not any(k.endswith("words.txt") for k in manifest.exact):
            pytest.skip(f"{name} has no words.txt in this capture")
        for relative in manifest.exact:
            if relative.endswith("words.txt"):
                assert (directory / relative).is_file(), f"{name}: {relative}"
                assert relative not in manifest.large, f"{name}: {relative}"

    def test_the_feature_configuration_is_recorded(self, corpus):
        """
        A reference taken under different `--dither` is a record of
        something the pipeline no longer does, and nothing else would
        notice.
        """
        name, _directory, manifest = corpus
        assert any(
            k.startswith("recipe/conf/") for k in manifest.exact
        ), f"{name} records no feature configuration"

    def test_both_sides_of_cmvn_are_recorded(self, corpus):
        """
        `raw/` is the input, `features/` the output. The first captures held
        only the output, which was no use for verifying either stage.
        """
        name, _directory, manifest = corpus
        arrays = set(manifest.arrays)
        assert any(k.startswith("raw/") for k in arrays), f"{name}: no raw/"
        assert any(k.startswith("features/") for k in arrays), f"{name}: no features/"


class TestFingerprints:
    def test_a_fingerprint_names_an_artefact_that_was_captured(self, corpus):
        name, _directory, manifest = corpus
        if not manifest.functional:
            pytest.skip(f"{name} was captured before fingerprints")
        for relative in manifest.functional:
            assert relative in manifest.exact, f"{name}: {relative}"

    def test_a_fingerprint_is_not_empty(self, corpus):
        """
        An empty one would pass every comparison while meaning nothing,
        which is the same failure mode as fabricated reference data.
        """
        name, _directory, manifest = corpus
        if not manifest.functional:
            pytest.skip(f"{name} was captured before fingerprints")
        for relative, weights in manifest.functional.items():
            assert weights, f"{name}: {relative} fingerprint is empty"

    def test_the_weights_are_finite(self, corpus):
        import math

        name, _directory, manifest = corpus
        if not manifest.functional:
            pytest.skip(f"{name} was captured before fingerprints")
        for relative, weights in manifest.functional.items():
            for key, weight in weights.items():
                assert math.isfinite(weight), f"{name}: {relative}: {key}"


class TestLexiconFstReproducesItsDigest:
    """
    A Python `L.fst` against the digest Kaldi's recorded.

    This is the digest-only design paying for itself. mini-librispeech's
    `L.fst` is 33MB and its bytes are not committed, but `L` -- unlike `G`
    -- *is* byte-exact, so a 64-character digest and the inputs beside it
    gate the whole thing in tier 1, with no Kaldi and no OpenFST anywhere.

    It needs the dictionary that #40 added and the symbol tables that #38
    kept, which is the "capture the input too" rule earning its keep three
    times over.
    """

    #: How each recipe calls `prepare_lang.sh`. Not derivable from the
    #: capture: the silence probability and whether phones are
    #: position-dependent are arguments, and the reference records what
    #: came out rather than what was asked for. Kept here so adding a
    #: recipe is an explicit edit.
    SETTINGS = {
        "yesno": {"sil_prob": 0.0, "position_dependent_phones": False},
        "mini_librispeech": {
            "sil_prob": 0.5,
            "position_dependent_phones": True,
        },
    }

    def test_every_corpus_has_settings(self, corpus):
        """
        Otherwise a new capture would silently not be checked.
        """
        name, _directory, _manifest = corpus
        assert name in self.SETTINGS, (
            f"{name} has no prepare_lang.sh settings recorded, so its "
            f"L.fst cannot be rebuilt and is not being checked"
        )

    def test_it_matches(self, corpus):
        import hashlib

        from BenNevis.lm.lexicon import build_lexicon_fst_text, read_lexicon

        name, directory, manifest = corpus
        recorded = manifest.exact.get("lang/L.fst.txt")
        if recorded is None:
            pytest.skip(f"{name} has no L.fst in this capture")

        lexicon_path = directory / "dict" / "lexicon.txt"
        words_path = directory / "lang" / "words.txt"
        phones_path = directory / "lang" / "phones.txt"
        silence_path = directory / "dict" / "optional_silence.txt"
        for path in (words_path, phones_path):
            if not path.is_file():
                pytest.skip(f"{name} has no {path.name}")
        if not lexicon_path.is_file():
            pytest.skip(
                f"{name} was captured before the dictionary was, so there "
                f"is no input to rebuild L.fst from"
            )

        def table(path):
            result = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) == 2:
                    result[fields[0]] = int(fields[1])
            return result

        settings = dict(self.SETTINGS[name])
        if silence_path.is_file():
            content = silence_path.read_text(encoding="utf-8").split()
            if content:
                settings["sil_phone"] = content[0]

        built = build_lexicon_fst_text(
            read_lexicon(lexicon_path),
            table(words_path),
            table(phones_path),
            **settings,
        )
        digest = hashlib.sha256(built.encode("utf-8")).hexdigest()
        assert digest == recorded, (
            f"{name}: the Python L.fst hashes to {digest} and Kaldi's to "
            f"{recorded}; {manifest.large.get('lang/L.fst.txt', 0) / 1e6:.1f}"
            f"MB of bytes are not committed, so compare against the "
            f"workflow artefact to see how they differ"
        )
