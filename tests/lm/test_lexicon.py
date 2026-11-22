"""
Tests for the lexicon FST.

Unlike `G.fst`, `L.fst` is compared **byte for byte** with Kaldi's, and
passes. The standard in CLAUDE.md asks for functional equivalence because an
FST's encoding is usually arbitrary -- not because bytes are unwelcome where
they are determined. Here both of the choices `G` leaves free turn out to be
fixed: states are allocated in the order the lexicon lists its words, and
`prepare_lang.sh` leaves the arcs sorted by output label.

That makes the headline test a digest comparison against
`tests/golden/reference/yesno/lang/L.fst.txt`, with the structural tests
below there to say *which* part of the construction broke when it does.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from pathlib import Path

import pytest

from BenNevis.lm.lexicon import build_lexicon_arcs, build_lexicon_fst_text, read_lexicon

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REFERENCE = REPO_ROOT / "tests" / "golden" / "reference" / "yesno" / "lang"
YESNO_LEXICON = REPO_ROOT / "egs" / "yesno" / "asr" / "input" / "lexicon.txt"


def symbol_table(path):
    table = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2:
            table[fields[0]] = int(fields[1])
    return table


WORDS = {"<eps>": 0, "A": 1, "BB": 2, "CCC": 3}
PHONES = {"<eps>": 0, "x": 1, "y": 2, "z": 3}


class TestReadLexicon:
    def test_the_yesno_lexicon_reads(self):
        entries = read_lexicon(YESNO_LEXICON)
        assert entries == [
            ("<SIL>", ("<SIL>",)),
            ("<UNK>", ("<UNK>",)),
            ("YES", ("Y", "E", "S")),
            ("NO", ("N", "O")),
        ]

    def test_file_order_is_kept(self, tmp_path):
        """
        Not sorted. The order decides the state numbering and so the bytes:
        yesno lists YES before NO, so YES takes states 1 and 2 and NO takes
        3. Sorting the entries would swap them and the FST would no longer
        match Kaldi's, while still being a correct lexicon.
        """
        path = tmp_path / "l.txt"
        path.write_text("CCC z\nA x\nBB y\n")
        assert [word for word, _ in read_lexicon(path)] == ["CCC", "A", "BB"]

    def test_an_alternative_pronunciation_is_a_second_entry(self, tmp_path):
        path = tmp_path / "l.txt"
        path.write_text("A x\nA y\n")
        assert read_lexicon(path) == [("A", ("x",)), ("A", ("y",))]

    def test_a_blank_line_is_skipped(self, tmp_path):
        path = tmp_path / "l.txt"
        path.write_text("A x\n\n\nBB y\n")
        assert len(read_lexicon(path)) == 2

    def test_a_word_without_a_pronunciation_raises(self, tmp_path):
        """
        Rather than being dropped. A word that cannot be pronounced can
        never be decoded, and the symptom would be that one word is never
        recognised -- invisible against anything but a targeted test.
        """
        path = tmp_path / "l.txt"
        path.write_text("A x\nBB\n")
        with pytest.raises(ValueError, match="has no pronunciation"):
            read_lexicon(path)


class TestStructure:
    def test_a_single_phone_word_is_a_loop(self):
        arcs, loop = build_lexicon_arcs([("A", ("x",))], WORDS, PHONES)
        assert arcs == [(loop, loop, PHONES["x"], WORDS["A"])]

    def test_a_multi_phone_word_is_a_chain_back_to_the_loop(self):
        arcs, loop = build_lexicon_arcs([("CCC", ("x", "y", "z"))], WORDS, PHONES)
        assert arcs == [
            (loop, 1, PHONES["x"], WORDS["CCC"]),
            (1, 2, PHONES["y"], 0),
            (2, loop, PHONES["z"], 0),
        ]

    def test_the_word_rides_only_the_first_arc(self):
        """
        Emitting it again would decode one utterance as several copies of
        the same word.
        """
        arcs, _loop = build_lexicon_arcs([("CCC", ("x", "y", "z"))], WORDS, PHONES)
        assert [arc[3] for arc in arcs] == [WORDS["CCC"], 0, 0]

    def test_states_are_allocated_in_lexicon_order(self):
        """
        The property that makes the bytes match: the first word listed takes
        the lowest state numbers.
        """
        arcs, _loop = build_lexicon_arcs(
            [("BB", ("x", "y")), ("CCC", ("x", "y", "z"))], WORDS, PHONES
        )
        assert max(arc[1] for arc in arcs[:2]) == 1
        assert {arc[0] for arc in arcs[2:]} == {0, 2, 3}

    def test_the_loop_state_is_the_start_and_the_only_final(self):
        _arcs, loop = build_lexicon_arcs([("A", ("x",))], WORDS, PHONES)
        assert loop == 0

    def test_every_word_is_reachable_from_the_loop(self):
        lexicon = [("A", ("x",)), ("BB", ("x", "y")), ("CCC", ("x", "y", "z"))]
        arcs, loop = build_lexicon_arcs(lexicon, WORDS, PHONES)
        from_loop = {arc[3] for arc in arcs if arc[0] == loop}
        assert from_loop == {WORDS["A"], WORDS["BB"], WORDS["CCC"]}

    def test_a_missing_word_raises(self):
        with pytest.raises(KeyError, match="not in words.txt"):
            build_lexicon_arcs([("ZZZ", ("x",))], WORDS, PHONES)

    def test_a_missing_phone_raises(self):
        with pytest.raises(KeyError, match="not in phones.txt"):
            build_lexicon_arcs([("A", ("q",))], WORDS, PHONES)

    def test_a_non_zero_silence_probability_raises(self):
        """
        Rather than producing a plausible lexicon that decodes differently.
        The shape with optional silence is make_lexicon_fst.pl's, and
        guessing it would be a lexicon that looks right and is not.
        """
        with pytest.raises(NotImplementedError, match="sil_prob"):
            build_lexicon_arcs([("A", ("x",))], WORDS, PHONES, sil_prob=0.5)


class TestByteExactnessWithKaldi:
    @pytest.mark.skipif(
        not (REFERENCE / "L.fst.txt").is_file(),
        reason="no committed L.fst to compare against",
    )
    def test_it_reproduces_kaldis_bytes(self):
        """
        The headline. `egs/yesno/asr` prepares its language directory with
        `--sil_prob 0.0`, which is the construction this builds, so there is
        nothing between the two but the arc order and the state numbering --
        and both of those turn out to be determined rather than arbitrary.
        """
        lexicon = read_lexicon(YESNO_LEXICON)
        built = build_lexicon_fst_text(
            lexicon,
            symbol_table(REFERENCE / "words.txt"),
            symbol_table(REFERENCE / "phones.txt"),
        )
        assert built == (REFERENCE / "L.fst.txt").read_text(encoding="utf-8")

    def test_the_text_is_tab_separated(self):
        lexicon = read_lexicon(YESNO_LEXICON)
        built = build_lexicon_fst_text(
            lexicon,
            symbol_table(REFERENCE / "words.txt"),
            symbol_table(REFERENCE / "phones.txt"),
        )
        for line in built.splitlines():
            assert "\t" in line or line.isdigit(), line

    def test_it_ends_with_a_newline(self):
        built = build_lexicon_fst_text(
            read_lexicon(YESNO_LEXICON),
            symbol_table(REFERENCE / "words.txt"),
            symbol_table(REFERENCE / "phones.txt"),
        )
        assert built.endswith("\n")
