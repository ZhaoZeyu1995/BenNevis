"""
`prepare_lang.sh` in Python, gated against Kaldi's captured output.

Tier 1: `BenNevis.lm.prepare` needs nothing but the standard library.

The gate is byte-exactness against `tests/golden/reference/`, discovered
rather than listed, so a corpus whose capture gains a dictionary is picked
up without editing this file -- the same arrangement
`test_corpora.py` has, and for the same reason: a committed reference that
nothing reads is not a test.

**yesno cannot be gated and mini-librispeech's `align_lexicon.int`
cannot.** yesno's capture predates the dictionary being recorded, so there
is no input to rebuild its tables from; mini-librispeech's has the
dictionary but not `phones/align_lexicon.int`. That leaves the one file
the *training* path actually reads ungated on the one corpus that could
gate it, which is a gap in the capture rather than in the code -- it
belongs on the list with `utt2spk`. What is checked instead is its
structure and that its integers resolve through the two tables, both of
which are byte-exact.

The unit tests below cover the decisions that are not guessable from the
shell and that decide the integer ids. Each has a specific failure in
mind: get the suffix lists wrong and every phone id shifts; get the group
order wrong and they shift differently; miss that `words.txt` reserves
three symbols at the end and the grammar back-off label collides with a
word.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import json
import tempfile
from pathlib import Path

import pytest

from BenNevis.lm.prepare import (
    NONSILENCE_SUFFIXES,
    SILENCE_SUFFIXES,
    build_align_lexicon,
    build_phone_map,
    build_phone_symbols,
    build_word_symbols,
    prepare_lang,
    read_phone_groups,
)

REFERENCE = Path(__file__).resolve().parent.parent / "golden" / "reference"

#: What `prepare_lang.sh` was given per corpus, which is also what
#: `tests/recipes/test_lexicon_settings.py` pins against each recipe's own
#: call -- so these cannot drift from the recipes silently.
SETTINGS = {
    "yesno": {"position_dependent_phones": False, "sil_prob": 0.0},
    "mini_librispeech": {"position_dependent_phones": True, "sil_prob": 0.5},
}

#: The out-of-vocabulary word each recipe passes.
OOV = {"yesno": "<UNK>", "mini_librispeech": "<UNK>"}

#: Everything the capture may hold that this can produce.
GATED = (
    "phones.txt",
    "words.txt",
    "oov.txt",
    "oov.int",
    "phones/silence.int",
    "phones/disambig.int",
    "phones/optional_silence.int",
    "phones/wdisambig_phones.int",
    "phones/wdisambig_words.int",
    "phones/align_lexicon.int",
)


def corpora():
    """Captures that have a dictionary to rebuild from."""
    if not REFERENCE.is_dir():
        return []
    return sorted(
        path.name
        for path in REFERENCE.iterdir()
        if (path / "dict" / "lexicon.txt").is_file() and path.name in SETTINGS
    )


#: `prepare_lang` is run once per corpus and the result reused, because
#: this class asks four questions of it and mini-librispeech's 206,510
#: entries are not free. Memoised rather than made a class-scoped fixture,
#: which cannot depend on a function-scoped parameter like `corpus`.
_PREPARED = {}


def _prepare_once(corpus):
    if corpus not in _PREPARED:
        directory = REFERENCE / corpus
        out = Path(tempfile.mkdtemp(prefix=f"bennevis-lang-{corpus}-"))
        prepare_lang(
            directory / "dict",
            OOV[corpus],
            out,
            # The lexicon FSTs are the only slow part and
            # tests/reference/test_corpora.py already gates both by
            # digest, so tier 1 does not pay for them twice.
            write_fsts=False,
            **SETTINGS[corpus],
        )
        _PREPARED[corpus] = (directory, out)
    return _PREPARED[corpus]


@pytest.mark.parametrize(
    "corpus",
    corpora()
    or [
        pytest.param(
            "none", marks=pytest.mark.skip(reason="no capture with a dictionary")
        )
    ],
)
class TestAgainstTheCapture:
    """
    This implementation against Kaldi's, on the real reference.

    **Conditional on the capture's `lang/` being Kaldi's.** The recipes'
    stage 1 calls `python3 -m BenNevis.lm prepare-lang` from the change
    that added this note, so a fresh capture records *these* tables and
    every assertion below becomes the code against itself -- the failure
    the decoding graph's token table already had. The manifest records
    `produced_by`, read off the recipe's own `run.sh`, and this skips with
    that reason rather than passing emptily.

    The committed captures were taken while Kaldi owned the stage, so
    they still compare. A re-capture retires this gate, which is the
    per-stage retirement CLAUDE.md prescribes and not a loss: nine files
    byte-exact is a result that has already been obtained, and
    `tests/reference/compare_lang.py` is where the live comparison goes.
    """

    @pytest.fixture
    def prepared(self, corpus):
        produced_by = (
            json.loads((REFERENCE / corpus / "manifest.json").read_text())
            .get("produced_by", {})
            .get("lang", "kaldi")
        )
        if produced_by != "kaldi":
            pytest.skip(
                f"{corpus}'s lang/ was produced by {produced_by!r}, so "
                f"comparing it with prepare_lang would compare the code "
                f"with itself; the gate retires with the stage"
            )
        return _prepare_once(corpus)

    def test_every_recorded_table_is_byte_exact(self, prepared):
        # The lexicon FSTs are deliberately not checked here:
        # tests/reference/test_corpora.py already gates both by digest,
        # and building them is the only slow part of this.
        directory, out = prepared
        compared = 0
        for name in GATED:
            recorded = directory / "lang" / name
            if not recorded.is_file():
                continue
            compared += 1
            assert (out / name).read_text(encoding="utf-8") == recorded.read_text(
                encoding="utf-8"
            ), f"{name} differs from Kaldi's"
        # A test that compared nothing would pass, which is how the G.fst
        # gate in #41 managed to be a test that could not fail.
        assert compared >= 4, f"only {compared} files were comparable"

    def test_align_lexicon_resolves_through_both_tables(self, prepared):
        """
        Ungated by bytes -- see the module docstring -- so checked for the
        two things that could be wrong about it: that the word is repeated
        and that every integer resolves.
        """
        _directory, out = prepared
        words = dict(
            line.split()
            for line in (out / "words.txt").read_text(encoding="utf-8").splitlines()
        )
        phones = dict(
            line.split()
            for line in (out / "phones.txt").read_text(encoding="utf-8").splitlines()
        )
        text = (out / "phones" / "align_lexicon.txt").read_text().splitlines()
        integers = (out / "phones" / "align_lexicon.int").read_text().splitlines()
        assert len(text) == len(integers) and text
        for line, numbers in zip(text, integers):
            fields = line.split()
            assert fields[0] == fields[1], line
            expected = [words[fields[0]], words[fields[1]]] + [
                phones[p] for p in fields[2:]
            ]
            assert numbers.split() == expected, line

    def test_the_eps_entry_is_the_optional_silence(self, prepared):
        directory, out = prepared
        silence = (directory / "dict" / "optional_silence.txt").read_text().split()[0]
        lines = (out / "phones" / "align_lexicon.txt").read_text().splitlines()
        assert f"<eps> <eps> {silence}" in lines


class TestThePhoneOrder:
    SILENCE = [["SIL"], ["SPN"]]
    NONSILENCE = [["AA", "AA0"], ["B"]]

    def test_silence_keeps_its_bare_form_and_nonsilence_does_not(self):
        """
        The suffix lists are `"" "" "_B" "_E" "_I" "_S"` for silence and
        `"" "_B" "_E" "_I" "_S"` for non-silence, and the first field is
        the map's key -- so silence has five versions including the
        unsuffixed one and non-silence four. Getting this wrong shifts
        every phone id, and with it every integer in the directory.
        """
        assert SILENCE_SUFFIXES == ("", "_B", "_E", "_I", "_S")
        assert NONSILENCE_SUFFIXES == ("_B", "_E", "_I", "_S")
        silence = build_phone_map(self.SILENCE, SILENCE_SUFFIXES, True)
        nonsilence = build_phone_map(self.NONSILENCE, NONSILENCE_SUFFIXES, True)
        assert silence["SIL"] == ["SIL", "SIL_B", "SIL_E", "SIL_I", "SIL_S"]
        assert nonsilence["B"] == ["B_B", "B_E", "B_I", "B_S"]

    def test_the_order_is_silence_then_nonsilence_then_disambig(self):
        symbols, silence, nonsilence = build_phone_symbols(
            self.SILENCE, self.NONSILENCE, ["#0", "#1"], True
        )
        assert symbols[0] == "<eps>"
        assert symbols[1:6] == ["SIL", "SIL_B", "SIL_E", "SIL_I", "SIL_S"]
        assert symbols[-2:] == ["#0", "#1"]
        assert silence[0] == "SIL" and nonsilence[0] == "AA_B"

    def test_the_grouping_decides_the_ids(self):
        """
        `nonsilence_phones.txt` puts a phone and its stress variants on one
        line, and `phones.txt` walks lines then fields -- so `AA0`'s
        versions follow `AA`'s rather than being sorted among them.
        """
        symbols, _s, _n = build_phone_symbols(self.SILENCE, self.NONSILENCE, [], True)
        assert symbols.index("AA_S") < symbols.index("AA0_B")
        assert symbols.index("AA0_S") < symbols.index("B_B")

    def test_without_position_dependence_every_phone_maps_to_itself(self):
        symbols, _s, _n = build_phone_symbols(
            self.SILENCE, self.NONSILENCE, ["#0"], False
        )
        assert symbols == ["<eps>", "SIL", "SPN", "AA", "AA0", "B", "#0"]

    def test_groups_keep_their_shape_when_read(self, tmp_path):
        path = tmp_path / "nonsilence_phones.txt"
        path.write_text("AA AA0 AA1\n\nB\n", encoding="utf-8")
        assert read_phone_groups(path) == [["AA", "AA0", "AA1"], ["B"]]


class TestTheWordOrder:
    def test_it_is_eps_the_sorted_words_then_three_reserved_symbols(self):
        assert build_word_symbols(["ZOO", "APPLE", "APPLE"]) == [
            "<eps>",
            "APPLE",
            "ZOO",
            "#0",
            "<s>",
            "</s>",
        ]

    @pytest.mark.parametrize("reserved", ["<s>", "</s>"])
    def test_a_lexicon_entry_for_a_sentence_boundary_is_refused(self, reserved):
        """
        `prepare_lang.sh` exits rather than renumber, because those two are
        written after the words and a lexicon entry would collide with the
        boundary `arpa2fst` uses.
        """
        with pytest.raises(ValueError, match="collide"):
            build_word_symbols(["A", reserved])


class TestAlignLexicon:
    def test_the_word_is_repeated_and_silence_is_appended(self):
        lines = build_align_lexicon([("YES", ("Y_B", "S_E"))], "SIL")
        assert lines == ["<eps> <eps> SIL", "YES YES Y_B S_E"]

    def test_no_silence_entry_without_an_optional_silence(self):
        assert build_align_lexicon([("A", ("A_S",))], None) == ["A A A_S"]

    def test_duplicates_collapse_and_the_order_is_a_byte_sort(self):
        lines = build_align_lexicon(
            [("B", ("B_S",)), ("A", ("A_S",)), ("B", ("B_S",))], None
        )
        assert lines == ["A A A_S", "B B B_S"]


class TestItRefusesBadInput:
    @pytest.fixture
    def dictionary(self, tmp_path):
        d = tmp_path / "dict"
        d.mkdir()
        (d / "lexicon.txt").write_text("YES Y\nNO N\n<UNK> SPN\n", encoding="utf-8")
        (d / "silence_phones.txt").write_text("SIL\nSPN\n", encoding="utf-8")
        (d / "nonsilence_phones.txt").write_text("Y\nN\n", encoding="utf-8")
        (d / "optional_silence.txt").write_text("SIL\n", encoding="utf-8")
        return d

    def test_it_works_on_a_small_dictionary(self, dictionary, tmp_path):
        prepare_lang(dictionary, "<UNK>", tmp_path / "lang")
        assert (tmp_path / "lang" / "phones.txt").is_file()
        assert (tmp_path / "lang" / "k2" / "L.fst.txt").read_text().strip()

    def test_a_missing_input(self, dictionary, tmp_path):
        (dictionary / "optional_silence.txt").unlink()
        with pytest.raises(ValueError, match="optional_silence.txt is missing"):
            prepare_lang(dictionary, "<UNK>", tmp_path / "lang")

    def test_an_empty_optional_silence(self, dictionary, tmp_path):
        (dictionary / "optional_silence.txt").write_text("\n", encoding="utf-8")
        with pytest.raises(ValueError, match="is empty"):
            prepare_lang(dictionary, "<UNK>", tmp_path / "lang")

    def test_an_oov_word_outside_the_lexicon(self, dictionary, tmp_path):
        with pytest.raises(ValueError, match="not in"):
            prepare_lang(dictionary, "<NOPE>", tmp_path / "lang")

    def test_a_phone_the_lists_do_not_declare(self, dictionary, tmp_path):
        """
        Reported here rather than several steps later, where it surfaces as
        a KeyError from inside the lexicon FST builder.
        """
        (dictionary / "lexicon.txt").write_text(
            "YES Y\nMAYBE M\n<UNK> SPN\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="phone lists do"):
            prepare_lang(dictionary, "<UNK>", tmp_path / "lang")


class TestCompilingTheLexiconFsts:
    """
    `--compile-fsts`, which decides whether the binary `L.fst` and
    `L_disambig.fst` are written beside the text.

    **Nothing in BenNevis's lane reads a binary FST.** Training reads
    `k2/L.fst.txt`; `run/decode_k2.py` builds its graph in k2. The binary
    form exists for Kaldi's `prepare_graph.sh`, which composes
    `L_disambig.fst` against `G.fst`, and for `decode-faster` -- the
    stages goal 2 deletes last and together. So a tree with no OpenFST
    must still get a language directory it can train from, which is why
    `auto` skips rather than fails, and `prepare_graph.sh` is what says so
    loudly when the decoding lane then has nothing to compose.

    The compiling itself is `BenNevis.utils.openfst`, whose argument list
    is pinned in `tests/utils/test_openfst.py`. What these pin is the
    choice of whether to call it.
    """

    @staticmethod
    def dictionary(tmp_path):
        directory = tmp_path / "dict"
        directory.mkdir()
        (directory / "lexicon.txt").write_text(
            "<UNK> SIL\nYES Y\nNO N\n", encoding="utf-8"
        )
        (directory / "silence_phones.txt").write_text("SIL\n", encoding="utf-8")
        (directory / "nonsilence_phones.txt").write_text("Y\nN\n", encoding="utf-8")
        (directory / "optional_silence.txt").write_text("SIL\n", encoding="utf-8")
        return directory

    def test_false_writes_the_text_and_no_binary(self, tmp_path):
        out = tmp_path / "lang"
        prepare_lang(
            self.dictionary(tmp_path),
            "<UNK>",
            out,
            position_dependent_phones=False,
            sil_prob=0.0,
            compile_fsts="false",
        )
        assert (out / "k2" / "L.fst.txt").is_file()
        assert (out / "k2" / "L_disambig.fst.txt").is_file()
        assert not (out / "L.fst").exists()
        assert not (out / "L_disambig.fst").exists()

    def test_auto_without_openfst_skips_and_says_so(
        self, tmp_path, monkeypatch, caplog
    ):
        """
        The case every developer machine is in. It must leave a usable
        language directory and explain what is missing, because the
        failure it defers to is four stages away.
        """
        import BenNevis.utils.openfst as openfst

        monkeypatch.setattr(openfst, "available", lambda: False)
        out = tmp_path / "lang"
        with caplog.at_level("INFO"):
            prepare_lang(
                self.dictionary(tmp_path),
                "<UNK>",
                out,
                position_dependent_phones=False,
                sil_prob=0.0,
                compile_fsts="auto",
            )
        assert not (out / "L.fst").exists()
        assert "OpenFST is not on PATH" in caplog.text
        assert "prepare_graph.sh" in caplog.text

    def test_true_without_openfst_is_an_error(self, tmp_path, monkeypatch):
        """
        A caller that asked for the binaries and silently got none would
        fail inside `fsttablecompose` with something unrelated.
        """
        import BenNevis.utils.openfst as openfst

        monkeypatch.setattr(openfst, "available", lambda: False)
        with pytest.raises(ValueError, match="OpenFST is not on PATH"):
            prepare_lang(
                self.dictionary(tmp_path),
                "<UNK>",
                tmp_path / "lang",
                position_dependent_phones=False,
                sil_prob=0.0,
                compile_fsts="true",
            )

    def test_it_compiles_both_from_the_text_it_wrote(self, tmp_path, monkeypatch):
        """
        From the text rather than by rebuilding, so the binary cannot be
        a different FST from the committed-and-gated text. The text is
        numeric and byte-identical to `fstprint`'s output -- gated against
        the capture for both corpora -- so it needs no symbol tables.
        """
        import BenNevis.utils.openfst as openfst

        compiled = []

        def record(text, output, **kwargs):
            compiled.append((text, output, kwargs))
            output.write_bytes(b"binary")

        monkeypatch.setattr(openfst, "available", lambda: True)
        monkeypatch.setattr(openfst, "compile_fst", record)

        out = tmp_path / "lang"
        prepare_lang(
            self.dictionary(tmp_path),
            "<UNK>",
            out,
            position_dependent_phones=False,
            sil_prob=0.0,
            compile_fsts="true",
        )

        assert [path.name for _text, path, _kw in compiled] == [
            "L.fst",
            "L_disambig.fst",
        ]
        for text, path, kwargs in compiled:
            source = out / "k2" / f"{path.stem}.fst.txt"
            assert text == source.read_text(encoding="utf-8")
            assert not kwargs, f"{path.name} was given {kwargs}"

    def test_an_unknown_choice_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="auto, true or false"):
            prepare_lang(
                self.dictionary(tmp_path),
                "<UNK>",
                tmp_path / "lang",
                position_dependent_phones=False,
                sil_prob=0.0,
                compile_fsts="yes",
            )

    def test_nothing_is_compiled_when_no_text_was_written(self, tmp_path, monkeypatch):
        """
        `write_fsts=False` is what the tier 1 table gate uses, and there
        is nothing to compile from. Reaching for the text anyway would
        raise `FileNotFoundError` in the one path that exists to be cheap.
        """
        import BenNevis.utils.openfst as openfst

        monkeypatch.setattr(openfst, "available", lambda: True)
        monkeypatch.setattr(
            openfst,
            "compile_fst",
            lambda *a, **k: pytest.fail("compiled with no text to compile from"),
        )
        prepare_lang(
            self.dictionary(tmp_path),
            "<UNK>",
            tmp_path / "lang",
            position_dependent_phones=False,
            sil_prob=0.0,
            write_fsts=False,
            compile_fsts="true",
        )
