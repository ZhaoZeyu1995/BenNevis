"""
Tests for the functional fingerprint of an FST.

The fingerprint exists because two decisions that are each right turn out to
conflict. FSTs are compared as functions rather than as bytes, and a large
artefact is recorded by digest without its content -- but a digest answers
only the byte question. `G.fst` is already known not to be byte-equal to a
Python build, so its digest can never match and says nothing at all. On
mini-librispeech, the only corpus where a graph replacement matters, that
left no usable check.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import math

import pytest

from BenNevis.utils.fingerprint import (
    NON_WORDS,
    fingerprint,
    fingerprint_phone_sequences,
    position_dependent_table,
    sample_lexicon_sequences,
    sample_word_sequences,
    shortest_distance,
)

SYMBOLS = {"<eps>": 0, "A": 1, "B": 2, "C": 3, "#0": 4, "<s>": 5, "</s>": 6}


class TestSampling:
    def test_it_is_reproducible(self):
        """
        The whole point: two implementations must be asked the same
        questions, and the questions come from the symbol table rather than
        from either FST.
        """
        first = sample_word_sequences(SYMBOLS.keys())
        second = sample_word_sequences(SYMBOLS.keys())
        assert first == second

    def test_it_does_not_depend_on_iteration_order(self):
        shuffled = {k: SYMBOLS[k] for k in reversed(list(SYMBOLS))}
        assert sample_word_sequences(SYMBOLS.keys()) == sample_word_sequences(
            shuffled.keys()
        )

    def test_non_words_never_appear(self):
        """
        `<eps>` labels nothing and `#0` is the back-off symbol, so a
        sequence containing either says nothing about the language.
        """
        for sequence in sample_word_sequences(SYMBOLS.keys()):
            for word in sequence:
                assert word not in NON_WORDS
                assert not word.startswith("#")

    def test_lengths_are_within_the_bounds(self):
        sequences = sample_word_sequences(SYMBOLS.keys(), min_length=2, max_length=4)
        assert sequences
        for sequence in sequences:
            assert 2 <= len(sequence) <= 4

    def test_a_different_seed_gives_different_questions(self):
        assert sample_word_sequences(SYMBOLS.keys(), seed=1) != (
            sample_word_sequences(SYMBOLS.keys(), seed=2)
        )

    def test_an_empty_vocabulary_gives_nothing(self):
        assert sample_word_sequences(["<eps>", "#0"]) == []

    def test_the_result_is_sorted(self):
        sequences = sample_word_sequences(SYMBOLS.keys())
        assert sequences == sorted(sequences)


class TestShortestDistance:
    def test_a_single_arc(self):
        arcs = {0: [(1, 1, 2.5)]}
        assert shortest_distance(arcs, {1: 0.0}, [1], 0) == pytest.approx(2.5)

    def test_the_final_weight_is_added(self):
        arcs = {0: [(1, 1, 2.5)]}
        assert shortest_distance(arcs, {1: 1.5}, [1], 0) == pytest.approx(4.0)

    def test_the_cheaper_of_two_paths_wins(self):
        """
        Not any path. A back-off FST has several for one sequence, and an
        arc weighted wrong makes some *other* path the cheapest, which
        changes the total silently.
        """
        arcs = {0: [(1, 1, 5.0), (2, 1, 3.0)]}
        assert shortest_distance(arcs, {1: 0.0, 2: 0.0}, [1], 0) == (pytest.approx(3.0))

    def test_an_epsilon_like_arc_consumes_nothing(self):
        arcs = {0: [(1, 9, 1.0)], 1: [(2, 1, 2.0)]}
        assert shortest_distance(
            arcs, {2: 0.0}, [1], 0, epsilon_like=[9]
        ) == pytest.approx(3.0)

    def test_without_the_epsilon_label_there_is_no_path(self):
        arcs = {0: [(1, 9, 1.0)], 1: [(2, 1, 2.0)]}
        assert shortest_distance(arcs, {2: 0.0}, [1], 0) is None

    def test_a_non_final_end_state_is_rejected(self):
        arcs = {0: [(1, 1, 2.5)]}
        assert shortest_distance(arcs, {}, [1], 0) is None

    def test_an_unaccepted_label_is_rejected(self):
        arcs = {0: [(1, 1, 2.5)]}
        assert shortest_distance(arcs, {1: 0.0}, [2], 0) is None

    def test_the_empty_sequence_takes_the_start_state_final_weight(self):
        assert shortest_distance({}, {0: 1.25}, [], 0) == pytest.approx(1.25)

    def test_the_start_state_is_not_assumed_to_be_zero(self):
        """
        Kaldi's G.fst begins at state 2, and reading it as though it began
        at 0 adds that state's back-off weight to every sentence.
        """
        arcs = {2: [(3, 1, 1.0)], 0: [(3, 1, 99.0)]}
        assert shortest_distance(arcs, {3: 0.0}, [1], 2) == pytest.approx(1.0)


class TestFingerprint:
    @pytest.fixture
    def unigram(self):
        """
        A one-state unigram loop: A costs 1, B costs 2, final costs 0.5.
        """
        arcs = {0: [(0, 1, 1.0), (0, 2, 2.0)]}
        return arcs, {0: 0.5}

    def test_it_scores_the_sequences_it_is_given(self, unigram):
        arcs, finals = unigram
        result = fingerprint(arcs, finals, SYMBOLS, 0, [("A",), ("B",), ("A", "B")])
        assert result["A"] == pytest.approx(1.5)
        assert result["B"] == pytest.approx(2.5)
        assert result["A B"] == pytest.approx(3.5)

    def test_a_rejected_sequence_is_left_out(self, unigram):
        """
        Rather than recorded as a sentinel, which a comparison could mistake
        for a number.
        """
        arcs, finals = unigram
        result = fingerprint(arcs, finals, SYMBOLS, 0, [("C",)])
        assert result == {}

    def test_an_out_of_table_word_is_skipped(self, unigram):
        arcs, finals = unigram
        assert fingerprint(arcs, finals, SYMBOLS, 0, [("ZZZ",)]) == {}

    def test_the_keys_are_space_joined(self, unigram):
        arcs, finals = unigram
        result = fingerprint(arcs, finals, SYMBOLS, 0, [("A", "B")])
        assert list(result) == ["A B"]

    def test_it_samples_when_given_no_sequences(self, unigram):
        arcs, finals = unigram
        result = fingerprint(arcs, finals, SYMBOLS, 0)
        assert result

    def test_two_encodings_of_one_function_fingerprint_alike(self):
        """
        The property the whole thing rests on. The same language, written
        with different state numbers, must give the same fingerprint --
        otherwise it would be testing the encoding, which is what the byte
        comparison already does and what the standard rejects.
        """
        one = ({0: [(0, 1, 1.0)]}, {0: 0.5})
        # The same loop, reached through a renumbered state.
        two = ({7: [(7, 1, 1.0)]}, {7: 0.5})
        sequences = [("A",), ("A", "A")]
        assert fingerprint(*one, SYMBOLS, 0, sequences) == fingerprint(
            *two, SYMBOLS, 7, sequences
        )

    def test_a_wrong_arc_weight_is_caught(self):
        """
        And the converse: an FST that is not the same function must
        fingerprint differently, or the check would pass on anything.
        """
        right = ({0: [(0, 1, 1.0)]}, {0: 0.5})
        wrong = ({0: [(0, 1, 1.0 + math.log(2))]}, {0: 0.5})
        sequences = [("A",)]
        assert fingerprint(*right, SYMBOLS, 0, sequences) != fingerprint(
            *wrong, SYMBOLS, 0, sequences
        )

    def test_the_disambiguation_symbol_is_free(self):
        """
        A back-off arc consumes no word, so it must not be matched against
        one -- otherwise every sequence through a back-off would be
        rejected.
        """
        arcs = {0: [(1, SYMBOLS["#0"], 0.25)], 1: [(1, 1, 1.0)]}
        result = fingerprint(arcs, {1: 0.0}, SYMBOLS, 0, [("A",)])
        assert result["A"] == pytest.approx(1.25)


class TestPositionDependentTable:
    """
    Read off the data, because the two recipes differ and a capture
    should not have to be told which one it is of: mini-librispeech takes
    `prepare_lang.sh`'s default, which suffixes, and yesno passes
    `--position-dependent-phones false`.
    """

    def test_a_suffixed_table_is_detected(self):
        assert position_dependent_table(["<eps>", "AH0_B", "AH0_E", "SIL"])

    def test_a_plain_table_is_not(self):
        assert not position_dependent_table(["<eps>", "Y", "N", "SIL"])

    def test_a_disambiguation_symbol_does_not_decide_it(self):
        """`#0` is not a phone, and must not vote either way."""
        assert not position_dependent_table(["<eps>", "Y", "#0", "#1"])

    def test_an_empty_table_is_not(self):
        assert not position_dependent_table([])


class TestSampleLexiconSequences:
    LEXICON = [
        ("YES", ("Y", "E", "S")),
        ("NO", ("N", "O")),
        ("ONE", ("W",)),
    ]

    def test_it_draws_only_words_the_lexicon_pronounces(self):
        drawn = sample_lexicon_sequences(self.LEXICON, count=20)
        assert drawn
        known = {"YES", "NO", "ONE"}
        for words, _phones in drawn:
            assert set(words) <= known

    def test_the_phones_are_the_concatenated_pronunciations(self):
        drawn = sample_lexicon_sequences(self.LEXICON, count=20)
        expansion = {"YES": ("Y", "E", "S"), "NO": ("N", "O"), "ONE": ("W",)}
        for words, phones in drawn:
            expected = tuple(p for word in words for p in expansion[word])
            assert phones == expected

    def test_position_dependence_suffixes_the_phones(self):
        """
        `_S` alone for a one-phone word, otherwise `_B`, `_I`, `_E` -- the
        project's own `position_dependent`, so the expansion cannot drift
        from the lexicon FST it is meant to probe.
        """
        drawn = sample_lexicon_sequences(
            [("YES", ("Y", "E", "S"))],
            count=1,
            min_length=1,
            max_length=1,
            position_dependent_phones=True,
        )
        assert drawn == [(("YES",), ("Y_B", "E_I", "S_E"))]

    def test_a_single_phone_word_takes_the_s_suffix(self):
        drawn = sample_lexicon_sequences(
            [("ONE", ("W",))],
            count=1,
            min_length=1,
            max_length=1,
            position_dependent_phones=True,
        )
        assert drawn == [(("ONE",), ("W_S",))]

    def test_the_first_pronunciation_wins(self):
        """
        A word with alternatives would otherwise make the expected phone
        sequence depend on which was picked, and the whole point is that
        two implementations are asked the same question.
        """
        lexicon = [("READ", ("R", "IY", "D")), ("READ", ("R", "EH", "D"))]
        drawn = sample_lexicon_sequences(lexicon, count=1, min_length=1, max_length=1)
        assert drawn == [(("READ",), ("R", "IY", "D"))]

    def test_the_draw_is_reproducible(self):
        first = sample_lexicon_sequences(self.LEXICON, count=12)
        again = sample_lexicon_sequences(self.LEXICON, count=12)
        assert first == again

    def test_a_different_seed_draws_differently(self):
        first = sample_lexicon_sequences(self.LEXICON, count=12, seed=1)
        other = sample_lexicon_sequences(self.LEXICON, count=12, seed=2)
        assert first != other

    def test_non_words_are_left_out(self):
        lexicon = [("<eps>", ("X",)), ("#0", ("X",)), ("YES", ("Y",))]
        drawn = sample_lexicon_sequences(lexicon, count=10)
        for words, _phones in drawn:
            assert set(words) == {"YES"}

    def test_an_empty_lexicon_draws_nothing(self):
        assert sample_lexicon_sequences([], count=10) == []


class TestFingerprintPhoneSequences:
    #: `A B` costs 1.0, and the FST also accepts `A` alone for 0.5.
    ARCS = {0: [(1, 1, 0.5)], 1: [(2, 2, 0.5)]}
    FINALS = {1: 0.0, 2: 0.0}
    PHONES = {"<eps>": 0, "a": 1, "b": 2, "z": 3}

    def test_it_keys_on_the_word_sequence(self):
        """Readable, and the same question both implementations are asked."""
        drawn = [(("WORD",), ("a", "b"))]
        got = fingerprint_phone_sequences(self.ARCS, self.FINALS, self.PHONES, 0, drawn)
        assert set(got) == {"WORD"}
        assert got["WORD"] == pytest.approx(1.0)

    def test_a_multi_word_key_is_space_joined(self):
        drawn = [(("ONE", "TWO"), ("a", "b"))]
        got = fingerprint_phone_sequences(self.ARCS, self.FINALS, self.PHONES, 0, drawn)
        assert set(got) == {"ONE TWO"}

    def test_a_rejected_sequence_is_left_out(self):
        """
        Not recorded as infinity: a rejection is a fact about the lexicon
        rather than the weights, and an entry both sides agree is
        unreachable would pass the comparison while testing nothing.
        """
        drawn = [(("BAD",), ("b", "a")), (("GOOD",), ("a", "b"))]
        got = fingerprint_phone_sequences(self.ARCS, self.FINALS, self.PHONES, 0, drawn)
        assert set(got) == {"GOOD"}

    def test_a_phone_outside_the_table_is_skipped(self):
        drawn = [(("MISSING",), ("a", "q"))]
        got = fingerprint_phone_sequences(self.ARCS, self.FINALS, self.PHONES, 0, drawn)
        assert got == {}

    def test_nothing_drawn_gives_nothing(self):
        assert (
            fingerprint_phone_sequences(self.ARCS, self.FINALS, self.PHONES, 0, [])
            == {}
        )
