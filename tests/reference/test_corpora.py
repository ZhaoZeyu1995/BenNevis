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


class TestDisambiguationSymbolCount:
    """
    How many disambiguation symbols the lexicon needed, against what
    `prepare_lang.sh` recorded.

    This is a check available from committed data alone and it is a strong
    one: getting the highest symbol number right across 206,510 entries
    needs prefix detection, homophone counting and the empty-pronunciation
    reservation rule all correct at once. One rule wrong and the number
    moves.

    `disambig.txt` is written as `#0 .. #ndisambig`, and `ndisambig` is
    `add_lex_disambig.pl`'s answer plus the extra symbol reserved for
    silence -- so the count of symbols pins the algorithm's return value.
    """

    #: Symbols `prepare_lang.sh` adds beyond what the lexicon needed: one,
    #: for optional silence in the lexicon FST.
    EXTRA = 1

    def test_it_matches(self, corpus):
        from BenNevis.lm.lexicon import (
            add_disambiguation,
            position_dependent,
            read_lexicon,
        )

        name, directory, _manifest = corpus
        lexicon_path = directory / "dict" / "lexicon.txt"
        disambig_path = directory / "lang" / "phones" / "disambig.int"
        if not lexicon_path.is_file():
            pytest.skip(f"{name} was captured before the dictionary was")
        if not disambig_path.is_file():
            pytest.skip(f"{name} has no disambig.int")

        settings = TestLexiconFstReproducesItsDigest.SETTINGS[name]
        lexicon = read_lexicon(lexicon_path)
        if settings["position_dependent_phones"]:
            lexicon = [(word, position_dependent(pron)) for word, pron in lexicon]

        _out, highest = add_disambiguation(lexicon)
        symbols = len(disambig_path.read_text(encoding="utf-8").split())
        # symbols == ndisambig + 1, and ndisambig == highest + EXTRA.
        expected = highest + self.EXTRA + 1
        assert symbols == expected, (
            f"{name}: disambig.int holds {symbols} symbols, so "
            f"add_lex_disambig returned {symbols - self.EXTRA - 1}; this "
            f"implementation returned {highest}"
        )

    def test_the_silence_symbol_number_follows_from_it(self, corpus):
        """
        `prepare_lang.sh` passes `--sil-disambig='#'$ndisambig`, so being
        one out here puts the wrong symbol on the silence arc of
        L_disambig.fst -- and that is four stages away from where it would
        be noticed.
        """
        name, directory, _manifest = corpus
        disambig_path = directory / "lang" / "phones" / "disambig.int"
        if not disambig_path.is_file():
            pytest.skip(f"{name} has no disambig.int")
        symbols = len(disambig_path.read_text(encoding="utf-8").split())
        assert symbols >= 2, f"{name}: too few symbols to carry #0 and silence"


class TestDisambigLexiconFstReproducesItsDigest:
    """
    A Python `L_disambig.fst` against the digest Kaldi's recorded.

    Four stages -- position suffixes, `add_lex_disambig.pl`,
    `make_lexicon_fst.py` with the silence disambiguation symbol, and
    `fstaddselfloops` -- and the last of those is Kaldi C++ rather than a
    script. mini-librispeech's is 34.5MB and its bytes are not committed,
    so this is a 64-character digest standing in for all of it.

    A cross-check that made the result believable before the digest was
    computed: `L_disambig` has 1,589,729 lines against `L`'s 1,539,598, and
    the difference of 50,131 is the 50,129 entries this implementation
    gives a disambiguation symbol, plus one self-loop, plus the one bridge
    arc the silence symbol adds.
    """

    def test_it_matches(self, corpus):
        import hashlib

        from BenNevis.lm.lexicon import build_disambig_fst_text, read_lexicon

        name, directory, manifest = corpus
        recorded = manifest.exact.get("lang/L_disambig.fst.txt")
        if recorded is None:
            pytest.skip(f"{name} has no L_disambig.fst in this capture")

        lexicon_path = directory / "dict" / "lexicon.txt"
        words_path = directory / "lang" / "words.txt"
        phones_path = directory / "lang" / "phones.txt"
        if not lexicon_path.is_file():
            pytest.skip(f"{name} was captured before the dictionary was")
        for path in (words_path, phones_path):
            if not path.is_file():
                pytest.skip(f"{name} has no {path.name}")

        def table(path):
            result = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if len(fields) == 2:
                    result[fields[0]] = int(fields[1])
            return result

        settings = dict(TestLexiconFstReproducesItsDigest.SETTINGS[name])
        silence_path = directory / "dict" / "optional_silence.txt"
        if silence_path.is_file():
            content = silence_path.read_text(encoding="utf-8").split()
            if content:
                settings["sil_phone"] = content[0]

        built = build_disambig_fst_text(
            read_lexicon(lexicon_path),
            table(words_path),
            table(phones_path),
            **settings,
        )
        digest = hashlib.sha256(built.encode("utf-8")).hexdigest()
        assert digest == recorded, (
            f"{name}: the Python L_disambig.fst hashes to {digest} and "
            f"Kaldi's to {recorded}; "
            f"{manifest.large.get('lang/L_disambig.fst.txt', 0) / 1e6:.1f}MB "
            f"of bytes are not committed, so compare against the workflow "
            f"artefact to see how they differ"
        )


class TestTheLexiconStagesSeparately:
    """
    Each stage of `prepare_lang.sh`'s lexicon pipeline, against the
    intermediate Kaldi wrote for it.

    `L_disambig.fst`'s digest covers four stages at once, so a mismatch
    there says something is wrong and nothing about which. These check the
    two stages that have recorded inputs, each driven by Kaldi's own output
    from the stage before rather than by this implementation's -- so an
    error in stage one cannot make stage two look wrong, or hide a second
    error that cancels it.

    #44 captured these on purpose, before the replacement was written. It
    is the only time today that happened in the right order.
    """

    @staticmethod
    def read_lexiconp(path):
        """A `lexiconp`: word, pronunciation probability, then phones."""
        entries = []
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields:
                entries.append((fields[0], tuple(fields[2:])))
        return entries

    def test_position_dependent_suffixing_matches(self, corpus):
        from BenNevis.lm.lexicon import position_dependent, read_lexicon

        name, directory, _manifest = corpus
        source = directory / "dict" / "lexicon.txt"
        target = directory / "tmp" / "lexiconp.txt"
        if not source.is_file() or not target.is_file():
            pytest.skip(f"{name} has no captured lexicon intermediate")

        settings = TestLexiconFstReproducesItsDigest.SETTINGS[name]
        if not settings["position_dependent_phones"]:
            pytest.skip(f"{name} does not use position-dependent phones")

        mine = [
            (word, position_dependent(pronunciation))
            for word, pronunciation in read_lexicon(source)
        ]
        theirs = self.read_lexiconp(target)
        assert len(mine) == len(theirs), f"{name}: entry count"
        differing = [
            (index, a, b) for index, (a, b) in enumerate(zip(mine, theirs)) if a != b
        ]
        assert not differing, (
            f"{name}: {len(differing)} entries differ, the first at line "
            f"{differing[0][0] + 1}: {differing[0][1]} against "
            f"{differing[0][2]}"
        )

    def test_disambiguation_matches_entry_for_entry(self, corpus):
        """
        Driven by Kaldi's suffixed lexicon, not by this implementation's,
        so the two stages are genuinely independent.
        """
        from BenNevis.lm.lexicon import add_disambiguation

        name, directory, _manifest = corpus
        source = directory / "tmp" / "lexiconp.txt"
        target = directory / "tmp" / "lexiconp_disambig.txt"
        if not source.is_file() or not target.is_file():
            pytest.skip(f"{name} has no captured lexicon intermediates")

        mine, _highest = add_disambiguation(self.read_lexiconp(source))
        theirs = self.read_lexiconp(target)
        assert len(mine) == len(theirs), f"{name}: entry count"
        differing = [
            (index, a, b) for index, (a, b) in enumerate(zip(mine, theirs)) if a != b
        ]
        assert not differing, (
            f"{name}: {len(differing)} entries differ, the first at line "
            f"{differing[0][0] + 1}: {differing[0][1]} against "
            f"{differing[0][2]}"
        )

    def test_the_symbol_count_matches_what_kaldi_recorded(self, corpus):
        """
        `lex_ndisambig` is `add_lex_disambig.pl`'s answer plus the one
        symbol reserved for silence, and that number decides which symbol
        goes on `L_disambig.fst`'s silence arc.
        """
        from BenNevis.lm.lexicon import add_disambiguation

        name, directory, _manifest = corpus
        source = directory / "tmp" / "lexiconp.txt"
        recorded = directory / "tmp" / "lex_ndisambig"
        if not source.is_file() or not recorded.is_file():
            pytest.skip(f"{name} has no captured lexicon intermediates")

        _mine, highest = add_disambiguation(self.read_lexiconp(source))
        expected = int(recorded.read_text(encoding="utf-8").strip())
        assert highest + 1 == expected, (
            f"{name}: lex_ndisambig is {expected}, so add_lex_disambig "
            f"returned {expected - 1}; this returned {highest}"
        )

    def test_the_pass_through_symbols_are_the_disambiguation_ones(self, corpus):
        """
        `fstaddselfloops` gets `#0`'s id on each side. If these were
        anything else the self-loop would carry a label the grammar never
        emits, and the composed graph would silently reject every
        back-off.
        """
        name, directory, _manifest = corpus
        phones_side = directory / "lang" / "phones" / "wdisambig_phones.int"
        words_side = directory / "lang" / "phones" / "wdisambig_words.int"
        if not phones_side.is_file():
            pytest.skip(f"{name} has no captured wdisambig files")

        phones = read_symbol_table_from(directory / "lang" / "phones.txt")
        words = read_symbol_table_from(directory / "lang" / "words.txt")
        assert phones_side.read_text().split() == [str(phones["#0"])], name
        assert words_side.read_text().split() == [str(words["#0"])], name


def read_symbol_table_from(path):
    """An OpenFST symbol table as a mapping."""
    table = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2:
            table[fields[0]] = int(fields[1])
    return table
