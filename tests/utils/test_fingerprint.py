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
