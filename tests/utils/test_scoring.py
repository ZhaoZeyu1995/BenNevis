"""
Tests for the Python word error rate.

Scoring is the gate the recipe tier exists to produce, so the arithmetic has
to be right before any number it reports means anything. Two independent
checks are used rather than a table of expected values alone:

1. Worked cases whose answer is obvious by hand, including the awkward ones
   (an empty reference, a hypothesis longer than the reference, a repeated
   word).
2. The total edit distance against a separate, deliberately naive
   implementation written here, over randomly generated pairs. The counts
   split three ways and only their sum is uniquely determined, so the sum is
   what is cross-checked.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import random

import pytest

from BenNevis.utils.scoring import (
    Counts,
    levenshtein,
    parse_sclite_wer,
    read_trn,
    score,
    word_error_rate,
)


def naive_distance(reference, hypothesis):
    """
    A plain full-matrix edit distance, as the oracle for the total.

    Written out in full rather than reusing the implementation under test:
    the point is that two different pieces of code agree.
    """
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    table = [[0] * columns for _ in range(rows)]
    for i in range(rows):
        table[i][0] = i
    for j in range(columns):
        table[0][j] = j
    for i in range(1, rows):
        for j in range(1, columns):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            table[i][j] = min(
                table[i - 1][j - 1] + cost,
                table[i - 1][j] + 1,
                table[i][j - 1] + 1,
            )
    return table[-1][-1]


class TestLevenshtein:
    @pytest.mark.parametrize(
        "reference,hypothesis,expected",
        [
            ([], [], (0, 0, 0)),
            (["A"], ["A"], (0, 0, 0)),
            (["A", "B"], ["A", "B"], (0, 0, 0)),
            # one word replaced
            (["A", "B", "C"], ["A", "X", "C"], (1, 0, 0)),
            # one word dropped
            (["A", "B", "C"], ["A", "C"], (0, 1, 0)),
            # one word invented
            (["A", "C"], ["A", "B", "C"], (0, 0, 1)),
            # nothing in common
            (["A", "B"], ["X", "Y"], (2, 0, 0)),
            # everything dropped
            (["A", "B"], [], (0, 2, 0)),
            # everything invented
            ([], ["A", "B"], (0, 0, 2)),
            # a repeated word, which is where a careless implementation
            # collapses two tokens into one
            (["A", "A"], ["A"], (0, 1, 0)),
            (["A"], ["A", "A"], (0, 0, 1)),
        ],
    )
    def test_worked_cases(self, reference, hypothesis, expected):
        assert levenshtein(reference, hypothesis) == expected

    def test_counts_sum_to_the_distance(self):
        """
        Over random pairs, against an independent implementation. Only the
        total is well defined -- the split between substitution, deletion and
        insertion depends on how ties are broken -- so the total is what is
        compared.
        """
        rng = random.Random(1368)
        vocabulary = ["A", "B", "C", "D"]
        for _ in range(400):
            reference = [rng.choice(vocabulary) for _ in range(rng.randint(0, 8))]
            hypothesis = [rng.choice(vocabulary) for _ in range(rng.randint(0, 8))]
            assert sum(levenshtein(reference, hypothesis)) == naive_distance(
                reference, hypothesis
            )

    def test_is_not_symmetric_in_its_counts(self):
        """A deletion one way round is an insertion the other."""
        assert levenshtein(["A", "B"], ["A"]) == (0, 1, 0)
        assert levenshtein(["A"], ["A", "B"]) == (0, 0, 1)


class TestCounts:
    def test_wer_is_errors_over_reference_words(self):
        counts = Counts(substitutions=1, deletions=1, insertions=1, reference_words=10)
        assert counts.errors == 3
        assert counts.wer == pytest.approx(30.0)

    def test_wer_can_exceed_one_hundred(self):
        """
        It genuinely can: insertions are not bounded by the reference length,
        and a gate that clamped would hide a runaway decoder.
        """
        counts = Counts(insertions=20, reference_words=10)
        assert counts.wer == pytest.approx(200.0)

    def test_an_empty_reference_does_not_divide_by_zero(self):
        assert Counts().wer == 0.0
        assert Counts(insertions=2).wer == 200.0

    def test_correct_is_the_matched_reference_words(self):
        counts = Counts(substitutions=2, deletions=1, insertions=5, reference_words=10)
        assert counts.correct == 7

    def test_counts_add(self):
        total = Counts(
            substitutions=1, reference_words=3, sentences=1, sentence_errors=1
        ) + Counts(deletions=2, reference_words=7, sentences=1, sentence_errors=1)
        assert total == Counts(
            substitutions=1,
            deletions=2,
            insertions=0,
            reference_words=10,
            sentences=2,
            sentence_errors=2,
        )
        assert total.wer == pytest.approx(30.0)

    def test_ser_counts_sentences_not_words(self):
        total = Counts(
            substitutions=5, reference_words=50, sentences=10, sentence_errors=1
        )
        assert total.ser == pytest.approx(10.0)

    def test_ser_of_nothing_is_zero(self):
        assert Counts().ser == 0.0


class TestWordErrorRate:
    def test_a_perfect_utterance_is_not_a_sentence_error(self):
        counts = word_error_rate(["A", "B"], ["A", "B"])
        assert counts.sentences == 1
        assert counts.sentence_errors == 0
        assert counts.wer == 0.0

    def test_one_error_makes_one_sentence_error(self):
        counts = word_error_rate(["A", "B"], ["A", "X"])
        assert counts.sentence_errors == 1

    def test_the_denominator_is_the_reference_length(self):
        counts = word_error_rate(["A", "B", "C", "D"], [])
        assert counts.reference_words == 4
        assert counts.wer == pytest.approx(100.0)


class TestReadTrn:
    def test_reads_words_and_ids(self, tmp_path):
        path = tmp_path / "ref.wrd.trn"
        path.write_text("YES NO NO YES (global-1_0_0_1)\nNO NO (global-0_0)\n")
        assert read_trn(path) == {
            "global-1_0_0_1": ["YES", "NO", "NO", "YES"],
            "global-0_0": ["NO", "NO"],
        }

    def test_an_empty_transcription_is_an_empty_list(self, tmp_path):
        path = tmp_path / "hyp.wrd.trn"
        path.write_text(" (global-1)\n")
        assert read_trn(path) == {"global-1": []}

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "hyp.wrd.trn"
        path.write_text("YES (a)\n\n\nNO (b)\n")
        assert read_trn(path) == {"a": ["YES"], "b": ["NO"]}

    def test_whitespace_is_collapsed(self, tmp_path):
        path = tmp_path / "hyp.wrd.trn"
        path.write_text("YES   NO\tYES   (a)\n")
        assert read_trn(path) == {"a": ["YES", "NO", "YES"]}

    def test_a_line_without_an_id_is_refused(self, tmp_path):
        path = tmp_path / "hyp.wrd.trn"
        path.write_text("YES NO\n")
        with pytest.raises(ValueError, match="no parenthesised utterance id"):
            read_trn(path)

    def test_a_repeated_id_is_refused(self, tmp_path):
        """
        Two transcriptions for one utterance means something upstream merged
        the per-job files wrongly, and silently keeping the last would score
        a corrupt file without complaint.
        """
        path = tmp_path / "hyp.wrd.trn"
        path.write_text("YES (a)\nNO (a)\n")
        with pytest.raises(ValueError, match="duplicate id 'a'"):
            read_trn(path)


class TestScore:
    def test_scores_a_whole_set(self):
        reference = {"a": ["YES", "NO"], "b": ["NO", "NO"]}
        hypothesis = {"a": ["YES", "NO"], "b": ["NO", "YES"]}
        counts = score(reference, hypothesis)
        assert counts.sentences == 2
        assert counts.reference_words == 4
        assert counts.substitutions == 1
        assert counts.wer == pytest.approx(25.0)

    def test_a_perfect_set_is_zero(self):
        reference = {"a": ["YES"], "b": ["NO"]}
        assert score(reference, dict(reference)).wer == 0.0

    def test_a_missing_utterance_is_refused(self):
        """
        A decoder that dropped utterances would otherwise be scored only on
        the ones it kept, which is the most flattering possible way to fail.
        """
        with pytest.raises(KeyError, match="missing from"):
            score({"a": ["YES"], "b": ["NO"]}, {"a": ["YES"]})

    def test_a_missing_utterance_can_be_allowed_explicitly(self):
        counts = score({"a": ["YES"], "b": ["NO"]}, {"a": ["YES"]}, require_all=False)
        assert counts.deletions == 1
        assert counts.wer == pytest.approx(50.0)

    def test_an_unexpected_utterance_is_always_refused(self):
        """
        An id in the hypothesis that is not in the reference means the two
        files describe different sets, which no tolerance should accept.
        """
        with pytest.raises(KeyError, match="not in the reference"):
            score({"a": ["YES"]}, {"a": ["YES"], "b": ["NO"]})

    def test_scoring_nothing_is_zero(self):
        assert score({}, {}).wer == 0.0


class TestParseScliteWer:
    # The layout of `sclite -o all`. Treated as a cross-check against the
    # computation above on real output, never as the source of a number, so
    # that a change in sctk's formatting cannot quietly become a WER change.
    SAMPLE = """
    ,-------------------------------------------------------------.
    |                        hyp.wrd.trn                          |
    |-------------------------------------------------------------|
    | SPKR   | # Snt # Wrd | Corr    Sub    Del    Ins    Err  S.Err |
    |--------+-------------+-----------------------------------------|
    | global |   29    232 |  99.6    0.4    0.0    0.0    0.4    3.4 |
    |=============================================================|
    | Sum/Avg|   29    232 |  99.6    0.4    0.0    0.0    0.4    3.4 |
    `-------------------------------------------------------------'
    """

    def test_reads_the_summary_row(self):
        assert parse_sclite_wer(self.SAMPLE) == pytest.approx(0.4)

    def test_a_zero_rate_is_not_confused_with_absence(self):
        assert parse_sclite_wer(self.SAMPLE.replace("0.4    3.4", "0.0    0.0")) == 0.0

    def test_missing_summary_returns_none(self):
        assert parse_sclite_wer("nothing resembling sclite output") is None

    def test_an_empty_string_returns_none(self):
        assert parse_sclite_wer("") is None
