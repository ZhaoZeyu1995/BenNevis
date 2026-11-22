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

import math
from pathlib import Path

import pytest

from BenNevis.lm.lexicon import (
    build_lexicon_arcs,
    build_lexicon_fst_text,
    format_weight,
    position_dependent,
    read_lexicon,
)

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
        assert arcs == [(loop, loop, PHONES["x"], WORDS["A"], 0.0)]

    def test_a_multi_phone_word_is_a_chain_back_to_the_loop(self):
        arcs, loop = build_lexicon_arcs([("CCC", ("x", "y", "z"))], WORDS, PHONES)
        assert arcs == [
            (loop, 1, PHONES["x"], WORDS["CCC"], 0.0),
            (1, 2, PHONES["y"], 0, 0.0),
            (2, loop, PHONES["z"], 0, 0.0),
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

    def test_a_silence_probability_of_one_raises(self):
        """
        Its no-silence arc would cost infinity.
        """
        with pytest.raises(ValueError, match=r"not in \[0, 1\)"):
            build_lexicon_arcs([("A", ("x",))], WORDS, PHONES, sil_prob=1.0)

    def test_a_negative_silence_probability_raises(self):
        with pytest.raises(ValueError, match=r"not in \[0, 1\)"):
            build_lexicon_arcs([("A", ("x",))], WORDS, PHONES, sil_prob=-0.1)


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


class TestPositionDependentPhones:
    """
    `prepare_lang.sh` suffixes a lexicon's phones by word position unless
    told not to, and not being told is the default.

    This is why a builder that ignored it reproduced yesno's bytes exactly
    -- yesno passes `--position-dependent-phones false` -- and could not
    build mini-librispeech's at all: its lexicon says `AH0` while its
    `phones.txt` holds only `AH0_B`, `AH0_E`, `AH0_I` and `AH0_S`.
    """

    def test_a_single_phone_word_takes_s(self):
        assert position_dependent(("x",)) == ("x_S",)

    def test_a_two_phone_word_takes_b_and_e(self):
        assert position_dependent(("x", "y")) == ("x_B", "y_E")

    def test_the_middle_takes_i(self):
        assert position_dependent(("x", "y", "z", "w")) == (
            "x_B",
            "y_I",
            "z_I",
            "w_E",
        )

    def test_an_empty_pronunciation_is_unchanged(self):
        assert position_dependent(()) == ()

    def test_it_is_off_unless_asked_for(self):
        """
        So a caller who has not thought about it gets a KeyError naming the
        phone, rather than silently building against phones that are not in
        the table.
        """
        arcs, _loop = build_lexicon_arcs([("A", ("x",))], WORDS, PHONES)
        assert arcs[0][2] == PHONES["x"]

    def test_the_optional_silence_phone_is_not_suffixed(self):
        """
        Visible in Kaldi's own output: the silence *word* `!SIL` is
        pronounced `SIL_S` while the optional-silence arc carries plain
        `SIL`. Suffixing both would put a phone on the silence arc that
        Kaldi does not.
        """
        phones = dict(PHONES, SIL=9, SIL_S=10, x_S=11)
        arcs, loop = build_lexicon_arcs(
            [("A", ("x",))],
            WORDS,
            phones,
            sil_prob=0.5,
            sil_phone="SIL",
            position_dependent_phones=True,
        )
        silence_arcs = [a for a in arcs if a[0] == 2]
        assert silence_arcs == [(2, loop, phones["SIL"], 0, 0.0)]


class TestOptionalSilence:
    @pytest.fixture
    def phones(self):
        return dict(PHONES, SIL=9)

    def test_the_states_are_kaldis(self, phones):
        """
        start 0, loop 1, silence 2. The numbering matters because it is what
        Kaldi's bytes have, and because the final state is then **not** 0.
        """
        arcs, loop = build_lexicon_arcs([("A", ("x",))], WORDS, phones, sil_prob=0.5)
        assert loop == 1
        assert {a[0] for a in arcs} >= {0, 1, 2}

    def test_the_start_arcs_are_epsilon_on_both_sides(self, phones):
        """
        Not the silence phone. The Perl script of the same name puts
        `silphone:<eps>` here and the Python one puts `<eps>:<eps>`; Kaldi's
        own output settles it, and the capture shows `0 2 0 0`.
        """
        arcs, _loop = build_lexicon_arcs([("A", ("x",))], WORDS, phones, sil_prob=0.5)
        start = sorted(a for a in arcs if a[0] == 0)
        assert [(a[2], a[3]) for a in start] == [(0, 0), (0, 0)]

    def test_the_start_arcs_pay_for_the_choice(self, phones):
        arcs, _loop = build_lexicon_arcs([("A", ("x",))], WORDS, phones, sil_prob=0.25)
        weights = sorted(a[4] for a in arcs if a[0] == 0)
        assert weights == [
            pytest.approx(-math.log(0.75)),
            pytest.approx(-math.log(0.25)),
        ]

    def test_the_last_arc_of_a_word_is_split(self, phones):
        """
        One copy to the loop state paying for no silence, one to the
        silence state paying for it. Non-deterministic on purpose: it is
        how the script avoids epsilon arcs.
        """
        arcs, loop = build_lexicon_arcs(
            [("CCC", ("x", "y", "z"))], WORDS, phones, sil_prob=0.5
        )
        last = [a for a in arcs if a[2] == phones["z"]]
        assert len(last) == 2
        assert {a[1] for a in last} == {loop, 2}

    def test_the_silence_word_is_split_too(self, phones):
        """
        The Perl script skips the split when the last phone is the silence
        phone; the Python one does not. Kaldi's output follows the Python,
        so this pins the one that matches.
        """
        words = dict(WORDS, SIL_WORD=7)
        arcs, _loop = build_lexicon_arcs(
            [("SIL_WORD", ("SIL",))], words, phones, sil_prob=0.5
        )
        word_arcs = [a for a in arcs if a[3] == words["SIL_WORD"]]
        assert len(word_arcs) == 2

    def test_the_silence_state_returns_on_the_silence_phone(self, phones):
        arcs, loop = build_lexicon_arcs([("A", ("x",))], WORDS, phones, sil_prob=0.5)
        assert [a for a in arcs if a[0] == 2] == [(2, loop, phones["SIL"], 0, 0.0)]

    def test_a_disambiguation_symbol_adds_a_state(self, phones):
        """
        `prepare_lang.sh` supplies one for `L_disambig.fst` and not for
        `L.fst`, so both paths have to exist.
        """
        phones = dict(phones, **{"#1": 11})
        arcs, loop = build_lexicon_arcs(
            [("A", ("x",))],
            WORDS,
            phones,
            sil_prob=0.5,
            sil_disambig="#1",
        )
        from_silence = [a for a in arcs if a[0] == 2]
        assert from_silence[0][1] != loop
        bridge = [a for a in arcs if a[2] == phones["#1"]]
        assert bridge and bridge[0][1] == loop

    def test_a_missing_silence_phone_raises(self):
        with pytest.raises(KeyError, match="optional-silence phone"):
            build_lexicon_arcs([("A", ("x",))], WORDS, PHONES, sil_prob=0.5)


class TestWeightFormatting:
    """
    Kaldi's `BaseFloat` is single precision and `fstprint` writes about nine
    significant figures. Writing the double instead gives a correct lexicon
    that is not byte-identical, and this did not arise while every weight
    was zero.
    """

    @pytest.mark.parametrize(
        "value,expected",
        [
            (math.log(2), "0.693147182"),
            (math.log(10), "2.30258512"),
            (0.0, "0"),
        ],
    )
    def test_it_matches_what_kaldi_writes(self, value, expected):
        assert format_weight(value) == expected

    def test_a_zero_weight_is_omitted_from_the_text(self):
        """
        `fstprint` leaves out the semiring One, which is why yesno's L.fst
        has four fields per arc and its G.fst has five.
        """
        text = build_lexicon_fst_text(
            [("A", ("x",))],
            WORDS,
            PHONES,
            sil_prob=0.0,
        )
        for line in text.splitlines():
            assert len(line.split("\t")) in (1, 4), line
