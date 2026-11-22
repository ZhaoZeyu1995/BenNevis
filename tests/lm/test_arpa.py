"""
Tests for the ARPA reader and the back-off scorer.

This is the oracle the FST construction will be checked against, so it has
to be right first. Three kinds of test:

1. Parsing, including the files that should be rejected. A model quietly
   missing entries is a WER mystery rather than a crash, so a header that
   disagrees with its body is an error.
2. The back-off rule on hand-built models whose answers are arithmetic.
3. **Functional equivalence with Kaldi**, by walking the `G.fst` Kaldi
   actually produced -- the one committed in `tests/golden/reference/` -- and
   comparing the weight it assigns to a sentence with the weight this scorer
   assigns. That is the standard set for FSTs in CLAUDE.md: the same
   function, whatever the encoding.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import math
import textwrap
from pathlib import Path

import pytest

from BenNevis.lm.arpa import LOG10_TO_NAT, ArpaModel, NGram, read_arpa

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REFERENCE = REPO_ROOT / "tests" / "golden" / "reference" / "yesno"
YESNO_ARPA = REPO_ROOT / "egs" / "yesno" / "asr" / "input" / "task.arpabo"


def write(tmp_path, text, name="lm.arpa"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text).lstrip("\n"))
    return path


BIGRAM = """
\\data\\
ngram 1=4
ngram 2=2

\\1-grams:
-99\t<s>\t-0.5
-1.0\t</s>
-0.7\tA\t-0.3
-0.8\tB\t-0.2

\\2-grams:
-0.2\t<s>\tA
-0.4\tA\tB

\\end\\
"""


class TestParsing:
    def test_the_yesno_model_reads(self):
        model = read_arpa(YESNO_ARPA)
        assert model.order == 1
        assert model.counts == {1: 4}
        assert model.vocabulary() == ["</s>", "<s>", "NO", "YES"]

    def test_entries_carry_their_probability_and_backoff(self, tmp_path):
        model = read_arpa(write(tmp_path, BIGRAM))
        assert model.entries[("A",)].logprob == pytest.approx(-0.7)
        assert model.entries[("A",)].backoff == pytest.approx(-0.3)
        assert model.entries[("A", "B")].logprob == pytest.approx(-0.4)
        # The highest order carries no back-off.
        assert model.entries[("A", "B")].backoff is None

    def test_the_order_is_the_highest_present(self, tmp_path):
        assert read_arpa(write(tmp_path, BIGRAM)).order == 2

    def test_a_gzipped_model_reads(self, tmp_path):
        import gzip

        plain = write(tmp_path, BIGRAM)
        packed = tmp_path / "lm.arpa.gz"
        with gzip.open(packed, "wt", encoding="utf-8") as handle:
            handle.write(plain.read_text())
        assert read_arpa(packed).entries == read_arpa(plain).entries

    def test_space_separated_fields_are_accepted(self, tmp_path):
        """
        yesno's own file mixes tabs and spaces, so both have to work.
        """
        model = read_arpa(
            write(
                tmp_path,
                """
            \\data\\
            ngram 1=1

            \\1-grams:
            -1.0 A -0.3

            \\end\\
            """,
            )
        )
        assert model.entries[("A",)].backoff == pytest.approx(-0.3)

    def test_a_header_disagreeing_with_the_body_is_refused(self, tmp_path):
        """
        The failure this catches is a truncated or filtered model, which
        otherwise shows up as an unexplainably bad WER.
        """
        with pytest.raises(ValueError, match="declares 5 1-gram"):
            read_arpa(
                write(
                    tmp_path,
                    """
                \\data\\
                ngram 1=5

                \\1-grams:
                -1.0\tA
                -1.0\tB

                \\end\\
                """,
                )
            )

    def test_a_wrong_field_count_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="needs 3 or 4 fields"):
            read_arpa(
                write(
                    tmp_path,
                    """
                \\data\\
                ngram 2=1

                \\2-grams:
                -1.0\tA

                \\end\\
                """,
                )
            )

    def test_an_unknown_section_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="unknown section"):
            read_arpa(
                write(
                    tmp_path,
                    """
                \\data\\
                ngram 1=1

                \\1-grams:
                -1.0\tA

                \\nonsense\\
                """,
                )
            )


class TestBackoffRule:
    @pytest.fixture
    def model(self, tmp_path):
        return read_arpa(write(tmp_path, BIGRAM))

    def test_a_present_bigram_is_used_directly(self, model):
        assert model.logprob(("A",), "B") == pytest.approx(-0.4)

    def test_an_absent_bigram_backs_off(self, model):
        """
        B has no bigram from it, so P(A | B) is backoff(B) + P(A).
        """
        assert model.logprob(("B",), "A") == pytest.approx(-0.2 + -0.7)

    def test_a_missing_backoff_counts_as_zero(self, tmp_path):
        """
        Not an error: an entry that is never a history has no weight to
        give, and ARPA writes nothing rather than 0.
        """
        model = read_arpa(
            write(
                tmp_path,
                """
            \\data\\
            ngram 1=2

            \\1-grams:
            -1.0\tA
            -2.0\tB

            \\end\\
            """,
            )
        )
        assert model.logprob(("A",), "B") == pytest.approx(-2.0)

    def test_history_longer_than_the_order_is_truncated(self, model):
        assert model.logprob(("X", "Y", "A"), "B") == pytest.approx(-0.4)

    def test_an_out_of_vocabulary_word_raises(self, model):
        """
        Rather than a silent zero. A lexicon and a model that disagree about
        the vocabulary is a preparation mistake.
        """
        with pytest.raises(KeyError, match="not in the language model"):
            model.logprob(("A",), "ZZZ")

    def test_a_sentence_sums_its_words_and_the_boundary(self, model):
        # <s> A -> -0.2, A B -> -0.4, B </s> -> backoff(B) + P(</s>)
        expected = -0.2 + -0.4 + (-0.2 + -1.0)
        assert model.score(["A", "B"]) == pytest.approx(expected)

    def test_the_weight_is_the_negative_natural_log(self, model):
        assert model.weight(["A", "B"]) == pytest.approx(
            -model.score(["A", "B"]) * math.log(10.0)
        )

    def test_the_conversion_constant_is_ln_ten(self):
        assert LOG10_TO_NAT == pytest.approx(2.302585092994046)


class TestEmptyAndEdges:
    def test_an_empty_model_has_order_zero(self):
        assert ArpaModel().order == 0
        assert ArpaModel().vocabulary() == []

    def test_an_ngram_knows_its_order(self):
        assert NGram(words=("a", "b"), logprob=-1.0).order == 2

    def test_a_single_word_sentence_scores(self, tmp_path):
        model = read_arpa(write(tmp_path, BIGRAM))
        assert model.score(["A"]) == pytest.approx(-0.2 + (-0.3 + -1.0))


@pytest.mark.skipif(
    not (REFERENCE / "graph" / "ctc" / "G.fst.txt").is_file(),
    reason="no committed G.fst to compare against",
)
class TestFunctionalEquivalenceWithKaldi:
    """
    The scorer against the `G.fst` Kaldi actually built.

    FSTs are compared as functions rather than as bytes -- see CLAUDE.md --
    so the test walks Kaldi's own FST and checks it assigns a sentence the
    same weight this scorer does. yesno's model is a unigram, which makes
    the walk exhaustive rather than sampled: every path through three arcs
    can be checked.
    """

    @pytest.fixture
    def kaldi_fst(self):
        """Kaldi's G.fst as (arcs by state, final weights)."""
        arcs: dict = {}
        finals: dict = {}
        text = (REFERENCE / "graph" / "ctc" / "G.fst.txt").read_text()
        for line in text.splitlines():
            fields = line.split()
            if not fields:
                continue
            if len(fields) >= 4:
                src, dst, ilabel, olabel = (int(f) for f in fields[:4])
                weight = float(fields[4]) if len(fields) > 4 else 0.0
                arcs.setdefault(src, []).append((dst, ilabel, olabel, weight))
            else:
                finals[int(fields[0])] = float(fields[1]) if len(fields) > 1 else 0.0
        return arcs, finals

    @pytest.fixture
    def words(self):
        """The symbol table the FST's labels index."""
        from BenNevis.topos.build import read_symbol_table

        return read_symbol_table(REFERENCE / "lang" / "words.txt")

    @pytest.fixture
    def model(self):
        return read_arpa(YESNO_ARPA)

    def walk(self, kaldi_fst, words, sentence):
        """
        The total weight Kaldi's FST gives a sentence.

        Only well defined because yesno's G is a unigram loop on one state,
        so there is exactly one path per sentence.
        """
        arcs, finals = kaldi_fst
        state = 0
        total = 0.0
        for word in sentence:
            label = words[word]
            matching = [a for a in arcs.get(state, []) if a[1] == label]
            assert (
                len(matching) == 1
            ), f"{word!r} has {len(matching)} arcs from state {state}"
            dst, _, _, weight = matching[0]
            total += weight
            state = dst
        assert state in finals, f"state {state} is not final"
        return total + finals[state]

    @pytest.mark.parametrize(
        "sentence",
        [
            ["YES"],
            ["NO"],
            ["YES", "NO"],
            ["NO", "YES"],
            ["YES"] * 8,
            ["NO"] * 8,
            ["YES", "NO", "NO", "YES", "NO", "YES", "YES", "NO"],
        ],
    )
    def test_the_weights_agree(self, kaldi_fst, words, model, sentence):
        theirs = self.walk(kaldi_fst, words, sentence)
        mine = model.weight(sentence)
        assert mine == pytest.approx(theirs, rel=1e-6, abs=1e-6), (
            f"{' '.join(sentence)}: Kaldi's FST gives {theirs:.6f}, this "
            f"scorer gives {mine:.6f}"
        )

    def test_a_longer_sentence_scales_linearly(self, kaldi_fst, words, model):
        """
        A unigram model has no history, so eight words must cost exactly
        twice four plus the shared boundary. Stated because it is the
        property that would break first if the boundary were handled twice.
        """
        four = model.weight(["YES"] * 4)
        eight = model.weight(["YES"] * 8)
        per_word = (eight - four) / 4
        assert per_word == pytest.approx(LOG10_TO_NAT, rel=1e-6)
