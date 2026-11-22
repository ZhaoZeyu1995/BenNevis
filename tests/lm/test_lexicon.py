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
    add_disambiguation,
    add_self_loops,
    build_disambig_fst_text,
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


class TestDisambiguation:
    """
    `add_lex_disambig.pl`'s algorithm.

    `L` composed with a grammar has to be determinisable, and two things
    stop it being so: a pronunciation that is a proper prefix of another,
    and a pronunciation shared by several words. Both are fixed by
    appending `#1`, `#2`, ... until every ambiguous pronunciation is
    distinct on the input side.

    The headline check is in `tests/reference/test_corpora.py`, which is
    where the committed data is. These pin the rules individually, so a
    failure there says which rule broke.
    """

    def test_an_unambiguous_pronunciation_gets_nothing(self):
        """
        Which is why the symbols are few: mini-librispeech needs fifteen
        for 206,510 entries.
        """
        out, highest = add_disambiguation([("A", ("x",)), ("B", ("y",))])
        assert out == [("A", ("x",)), ("B", ("y",))]
        assert highest == 0

    def test_a_proper_prefix_is_disambiguated(self):
        """
        `x` cannot be told from the start of `x y` without one.
        """
        out, _highest = add_disambiguation([("A", ("x",)), ("AB", ("x", "y"))])
        assert out[0] == ("A", ("x", "#1"))

    def test_the_longer_pronunciation_is_not(self):
        """
        Only the prefix is ambiguous; the word it is a prefix *of* is not.
        """
        out, _highest = add_disambiguation([("A", ("x",)), ("AB", ("x", "y"))])
        assert out[1] == ("AB", ("x", "y"))

    def test_a_homophone_gets_a_distinct_symbol_per_word(self):
        out, highest = add_disambiguation([("B", ("z",)), ("BEE", ("z",))])
        assert out == [("B", ("z", "#1")), ("BEE", ("z", "#2"))]
        assert highest == 2

    def test_three_homophones_count_up(self):
        lexicon = [("A", ("z",)), ("B", ("z",)), ("C", ("z",))]
        out, highest = add_disambiguation(lexicon)
        assert [p[-1] for _w, p in out] == ["#1", "#2", "#3"]
        assert highest == 3

    def test_symbols_restart_for_an_unrelated_pronunciation(self):
        """
        The counter is per pronunciation, not global -- otherwise a big
        lexicon would need thousands of symbols rather than fifteen.
        """
        lexicon = [
            ("A", ("z",)),
            ("B", ("z",)),
            ("C", ("w",)),
            ("D", ("w",)),
        ]
        out, highest = add_disambiguation(lexicon)
        assert [p[-1] for _w, p in out] == ["#1", "#2", "#1", "#2"]
        assert highest == 2

    def test_an_empty_pronunciation_gets_a_reserved_symbol(self):
        """
        It maps to nothing on the input side, so nothing else can
        distinguish it -- a symbol another entry also used would not
        separate them.
        """
        out, highest = add_disambiguation([("A", ()), ("B", ("x",))])
        assert out[0] == ("A", ("#1",))
        assert highest == 1

    def test_a_reserved_symbol_is_skipped_by_everything_else(self):
        """
        The subtlest rule in the script, and the one a reimplementation
        would most likely drop: once a number is reserved for the empty
        pronunciation, the per-pronunciation counters step over it.
        """
        lexicon = [("E", ()), ("A", ("z",)), ("B", ("z",))]
        out, _highest = add_disambiguation(lexicon)
        assert out[0] == ("E", ("#1",))
        # #1 is reserved, so the homophones take #2 and #3.
        assert [p[-1] for _w, p in out[1:]] == ["#2", "#3"]

    def test_two_empty_pronunciations_reserve_two_symbols(self):
        out, highest = add_disambiguation([("A", ()), ("B", ())])
        assert [p for _w, p in out] == [("#1",), ("#2",)]
        assert highest == 2

    def test_the_first_allowed_symbol_is_configurable(self):
        out, highest = add_disambiguation(
            [("A", ("z",)), ("B", ("z",))], first_allowed=5
        )
        assert [p[-1] for _w, p in out] == ["#5", "#6"]
        assert highest == 6

    def test_order_is_preserved(self):
        """
        The state numbering of the FST built from this depends on it.
        """
        lexicon = [("C", ("w",)), ("A", ("x",)), ("B", ("y",))]
        out, _highest = add_disambiguation(lexicon)
        assert [w for w, _p in out] == ["C", "A", "B"]

    def test_the_highest_decides_the_silence_symbol(self):
        """
        `prepare_lang.sh` takes the next number after this for the
        optional-silence disambiguation symbol, so being one out puts the
        wrong symbol on the silence arc. mini-librispeech's `disambig.int`
        holds `#0`..`#16`, which says this returned 15.
        """
        _out, highest = add_disambiguation([("A", ("z",)), ("B", ("z",))])
        assert highest == 2  # so silence would be #3

    def test_an_empty_lexicon_needs_nothing(self):
        assert add_disambiguation([]) == ([], 0)

    def test_it_runs_after_position_dependent_suffixing(self):
        """
        The order `prepare_lang.sh` uses, and it changes the answer: `x`
        and `x y` are prefix-related, but `x_S` and `x_B y_E` are not.
        """
        lexicon = [("A", ("x",)), ("AB", ("x", "y"))]
        suffixed = [(w, position_dependent(p)) for w, p in lexicon]
        out, highest = add_disambiguation(suffixed)
        assert highest == 0
        assert out == suffixed


class TestSelfLoops:
    """
    `fstaddselfloops`, which is Kaldi C++ rather than a script.

    Its own usage text states the rule -- self-loops go "on each final
    state and each state with non-epsilon output symbols on at least one
    arc out of the state" -- and `AddSelfLoops` in
    `fstext/pre-determinize-inl.h` confirms it: weight One, one loop per
    label pair, the destination being the state itself.
    """

    def test_a_final_state_gets_a_loop(self):
        arcs = add_self_loops([], [3], [(7, 8)])
        assert arcs == [(3, 3, 7, 8, 0.0)]

    def test_a_state_with_a_word_on_an_arc_gets_one(self):
        arcs = add_self_loops([(5, 6, 1, 2, 0.0)], [], [(7, 8)])
        assert (5, 5, 7, 8, 0.0) in arcs

    def test_a_state_whose_arcs_are_all_epsilon_does_not(self):
        """
        The output side is what counts, not the input side: an arc emitting
        a phone but no word cannot end a word, so nothing has to pass
        through there.
        """
        arcs = add_self_loops([(5, 6, 1, 0, 0.0)], [], [(7, 8)])
        assert not [a for a in arcs if a[0] == a[1]]

    def test_the_loop_carries_weight_zero(self):
        """
        Weight::One in the tropical semiring, which `fstprint` then omits.
        """
        arcs = add_self_loops([], [3], [(7, 8)])
        assert arcs[0][4] == 0.0

    def test_several_pairs_give_several_loops(self):
        arcs = add_self_loops([], [3], [(7, 8), (9, 10)])
        assert len(arcs) == 2
        assert {(a[2], a[3]) for a in arcs} == {(7, 8), (9, 10)}

    def test_no_pairs_means_no_change(self):
        original = [(5, 6, 1, 2, 0.0)]
        assert add_self_loops(original, [5], []) == original

    def test_the_original_list_is_not_mutated(self):
        original = [(5, 6, 1, 2, 0.0)]
        add_self_loops(original, [5], [(7, 8)])
        assert original == [(5, 6, 1, 2, 0.0)]

    def test_a_lexicon_fst_gets_exactly_one(self):
        """
        Worth pinning as a count rather than a structure. Every arc
        carrying a word leaves the loop state, and the loop state is the
        only final one, so a lexicon FST has exactly one qualifying state
        however big the lexicon is -- mini-librispeech's 206,510 entries
        gain a single arc, `1 1 347 200004`.
        """
        phones = dict(PHONES, SIL=9)
        arcs, loop = build_lexicon_arcs(
            [("A", ("x",)), ("CCC", ("x", "y", "z"))],
            WORDS,
            phones,
            sil_prob=0.5,
        )
        with_loops = add_self_loops(arcs, [loop], [(4, 5)])
        added = [a for a in with_loops if a[0] == a[1] and a[2] == 4]
        assert len(added) == 1
        assert added[0] == (loop, loop, 4, 5, 0.0)


class TestTheWholeDisambigChain:
    """
    `build_disambig_fst_text`, which is all four of `prepare_lang.sh`'s
    steps in one call.

    They are not separable: step three needs the silence disambiguation
    symbol's *number*, and that comes out of step two. Calling the pieces
    by hand with the wrong number puts the wrong symbol on the silence arc,
    four stages from where it shows.
    """

    @pytest.fixture
    def phones(self):
        return dict(PHONES, SIL=9, **{"#0": 20, "#1": 21, "#2": 22, "#3": 23})

    @pytest.fixture
    def words(self):
        return dict(WORDS, **{"#0": 30})

    def test_it_adds_the_pass_through_loop(self, words, phones):
        text = build_disambig_fst_text([("A", ("x",))], words, phones, sil_prob=0.5)
        assert f"1\t1\t{phones['#0']}\t{words['#0']}" in text

    def test_the_silence_symbol_follows_the_lexicon_ones(self, words, phones):
        """
        Two homophones need `#1` and `#2`, so silence takes `#3`.
        """
        text = build_disambig_fst_text(
            [("A", ("z",)), ("BB", ("z",))], words, phones, sil_prob=0.5
        )
        assert f"\t{phones['#3']}\t0" in text

    def test_without_silence_there_is_no_silence_symbol(self, words, phones):
        text = build_disambig_fst_text([("A", ("x",))], words, phones, sil_prob=0.0)
        assert f"\t{phones['#3']}\t" not in text

    def test_it_still_adds_the_pass_through_without_silence(self, words, phones):
        text = build_disambig_fst_text([("A", ("x",))], words, phones, sil_prob=0.0)
        assert f"{phones['#0']}\t{words['#0']}" in text

    def test_the_lexicon_passed_in_is_unsuffixed(self, words, phones):
        """
        The caller gives what `read_lexicon` returns; this applies the
        suffixes and the symbols. Doing it the other way round would
        double-apply them.
        """
        phones = dict(phones, x_S=40)
        text = build_disambig_fst_text(
            [("A", ("x",))],
            words,
            phones,
            position_dependent_phones=True,
        )
        assert f"\t{phones['x_S']}\t" in text
