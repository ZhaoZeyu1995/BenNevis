"""
Tests for the ARPA-to-FST construction.

The standard here is functional equivalence rather than bytes -- see
CLAUDE.md -- so these tests are about what the FST *computes*, with one
exception: the structural tests below pin every arc the construction emits,
because that is a statement about the construction that needs no assumption
about normalisation and localises a fault to a single n-gram.

The back-off rule in `BenNevis.lm.arpa` is the oracle for **arc weights**,
and deliberately not for **path totals**. A back-off FST's shortest path can
be cheaper than the rule's score, because backing off early can cost less
than the correct higher-order arc plus its exit path. That is not a defect
here: Kaldi's own `G.fst` does it too, measured on mini-librispeech's 3-gram,
and by the same amount. So the FST is the definition of the language model as
used, and the rule is the definition of its arcs.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import math
import textwrap
from pathlib import Path

import pytest

from BenNevis.lm.arpa import BOS, EOS, LOG10_TO_NAT, read_arpa
from BenNevis.lm.fst import build_arcs, build_fst_text, start_state, total_weight

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REFERENCE = REPO_ROOT / "tests" / "golden" / "reference" / "yesno"
YESNO_ARPA = REPO_ROOT / "egs" / "yesno" / "asr" / "input" / "task.arpabo"

TRIGRAM = """
\\data\\
ngram 1=5
ngram 2=5
ngram 3=3

\\1-grams:
-99\t<s>\t-0.4
-1.0\t</s>
-0.6\tA\t-0.3
-0.7\tB\t-0.25
-0.9\tC\t-0.5

\\2-grams:
-0.3\t<s>\tA\t-0.2
-0.5\tA\tB\t-0.15
-0.6\tB\tC\t-0.35
-0.8\tB\tA\t-0.1
-0.4\tC\t</s>

\\3-grams:
-0.2\t<s>\tA\tB
-0.35\tA\tB\tC
-0.45\tB\tA\tB

\\end\\
"""

SYMBOLS = {"<eps>": 0, "A": 1, "B": 2, "C": 3, "#0": 4}


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "lm.arpa"
    path.write_text(textwrap.dedent(TRIGRAM).lstrip("\n"))
    return read_arpa(path)


@pytest.fixture
def built(model):
    arcs, finals, state_of = build_arcs(model, SYMBOLS)
    return arcs, finals, state_of


class TestStructure:
    """
    Every n-gram becomes one arc, with the weight the rule gives it.

    This is the test that localises a fault: a wrong arc names the n-gram
    that produced it, where a wrong path total names only the sentence.
    """

    def test_every_scoring_ngram_has_an_arc(self, model, built):
        arcs, _finals, state_of = built
        emitted = {(a[0], a[2]) for a in arcs}
        for words in model.entries:
            history, word = words[:-1], words[-1]
            if word in (EOS, BOS) or history not in state_of:
                continue
            assert (state_of[history], SYMBOLS[word]) in emitted, words

    def test_an_arc_carries_the_rules_weight(self, model, built):
        arcs, _finals, state_of = built
        by_key = {(a[0], a[2]): a for a in arcs if a[2] != SYMBOLS["#0"]}
        for words, entry in model.entries.items():
            history, word = words[:-1], words[-1]
            if word in (EOS, BOS) or history not in state_of:
                continue
            arc = by_key[(state_of[history], SYMBOLS[word])]
            assert arc[4] == pytest.approx(-entry.logprob * LOG10_TO_NAT)

    def test_an_arc_is_labelled_the_same_in_and_out(self, built):
        """
        G is an acceptor over words, written as a transducer. A differing
        output label would silently relabel the decoded words.
        """
        arcs, _finals, _state_of = built
        for arc in arcs:
            if arc[2] == SYMBOLS["#0"]:
                continue
            assert arc[2] == arc[3]

    def test_every_state_but_the_unigram_backs_off(self, built):
        arcs, _finals, state_of = built
        backoff = {a[0] for a in arcs if a[2] == SYMBOLS["#0"]}
        assert backoff == set(state_of.values()) - {state_of[()]}

    def test_a_backoff_arc_is_disambig_to_epsilon(self, built):
        """
        `#0:<eps>` rather than `<eps>:<eps>`. The disambiguation symbol is
        what keeps the composed graph determinisable -- two paths differing
        only in how far they backed off would otherwise be identical.
        """
        arcs, _finals, _state_of = built
        backoff = [a for a in arcs if a[2] == SYMBOLS["#0"]]
        assert backoff
        for arc in backoff:
            assert arc[3] == 0

    def test_a_backoff_arc_carries_the_models_weight(self, model, built):
        arcs, _finals, state_of = built
        state_to_history = {v: k for k, v in state_of.items()}
        for arc in arcs:
            if arc[2] != SYMBOLS["#0"]:
                continue
            history = state_to_history[arc[0]]
            expected = -model.backoff_of(history) * LOG10_TO_NAT
            assert arc[4] == pytest.approx(expected), history

    def test_a_backoff_arc_goes_to_the_suffix(self, built):
        arcs, _finals, state_of = built
        state_to_history = {v: k for k, v in state_of.items()}
        for arc in arcs:
            if arc[2] != SYMBOLS["#0"]:
                continue
            source = state_to_history[arc[0]]
            assert state_to_history[arc[1]] == source[1:]

    def test_end_of_sentence_is_a_final_weight_not_an_arc(self, model, built):
        """
        What makes the FST finite: with `</s>` as an arc, every path could
        be extended for ever.
        """
        arcs, finals, state_of = built
        assert SYMBOLS.get(EOS) is None  # not even in the table
        for words, entry in model.entries.items():
            if words[-1] != EOS or words[:-1] not in state_of:
                continue
            state = state_of[words[:-1]]
            assert finals[state] == pytest.approx(-entry.logprob * LOG10_TO_NAT)

    def test_the_unigram_state_is_final(self, built):
        """
        Or a sentence whose last history has no `</s>` entry has nowhere to
        end, and the FST would reject it outright.
        """
        _arcs, finals, state_of = built
        assert state_of[()] in finals

    def test_a_state_only_exists_if_something_extends_it(self, model, built):
        """
        Giving every entry below the highest order a state instead produces
        1,162,307 states for mini-librispeech's 3-gram against arpa2fst's
        249,532 -- functionally the same FST, four and a half times the
        size, composed into a TLG that is already 334MB.
        """
        _arcs, _finals, state_of = built
        extended = {words[:-1] for words in model.entries if len(words) > 1}
        for history in state_of:
            if not history:
                continue
            suffixes = {history[cut:] for cut in range(len(history))}
            assert (
                history in extended or suffixes & extended or history == (BOS,)
            ), history


class TestStartState:
    def test_the_start_is_the_sentence_boundary(self, built):
        _arcs, _finals, state_of = built
        assert start_state(state_of) == state_of[(BOS,)]

    def test_it_is_not_the_unigram_state(self, built):
        """
        The bug this guards. The unigram state is the lowest numbered, so
        inferring the start from the arcs picks it -- and a *unigram* model
        cannot tell the difference, because its history is truncated away.
        yesno agreed by accident; every higher order would be wrong.
        """
        _arcs, _finals, state_of = built
        assert start_state(state_of) != state_of[()]

    def test_a_model_without_a_boundary_falls_back(self, tmp_path):
        path = tmp_path / "x.arpa"
        path.write_text("\\data\\\nngram 1=1\n\n\\1-grams:\n-1.0\tA\n\n\\end\\\n")
        model = read_arpa(path)
        _arcs, _finals, state_of = build_arcs(model, SYMBOLS)
        assert start_state(state_of) == state_of[()]


class TestTextForm:
    def test_the_first_line_is_the_start_state(self, model):
        """
        OpenFST's text format takes the start state from the first line's
        source. Kaldi's own G.fst begins at state 2 for this reason, and
        reading it as though it began at 0 adds that state's back-off
        weight to every sentence.
        """
        text = build_fst_text(model, SYMBOLS)
        _arcs, _finals, state_of = build_arcs(model, SYMBOLS)
        first = text.splitlines()[0].split("\t")
        assert int(first[0]) == start_state(state_of)

    def test_every_arc_line_has_five_fields(self, model):
        text = build_fst_text(model, SYMBOLS)
        for line in text.splitlines():
            assert len(line.split("\t")) in (2, 5), line

    def test_final_states_come_last(self, model):
        """
        Not required by OpenFST, but it keeps the text readable and makes a
        diff between two captures line up.
        """
        lines = [ln.split("\t") for ln in build_fst_text(model, SYMBOLS).splitlines()]
        kinds = [len(f) for f in lines]
        assert kinds == sorted(kinds, reverse=True)

    def test_it_ends_with_a_newline(self, model):
        assert build_fst_text(model, SYMBOLS).endswith("\n")


class TestShortestPathVersusTheRule:
    """
    The property that makes the rule an oracle for arcs and not for totals.

    Backing off early can be cheaper than the correct higher-order arc plus
    the cost of leaving the state it leads to. On this model, every sentence
    containing `B A` is 0.05 log10 cheaper by the FST than by the rule --
    and **Kaldi's own G.fst behaves the same way** on mini-librispeech's
    3-gram, so this is a property of the encoding rather than of this
    construction.
    """

    def test_the_shortest_path_never_exceeds_the_rule(self, model, built):
        """
        One direction does hold unconditionally: the rule describes a real
        path, so the cheapest path cannot cost more than it.
        """
        arcs, finals, state_of = built
        start = start_state(state_of)
        for sentence in (["A"], ["B"], ["A", "B"], ["B", "A"], ["A", "B", "C"]):
            fst = total_weight(arcs, finals, SYMBOLS, sentence, start=start)
            assert fst is not None
            assert fst <= model.weight(sentence) + 1e-9, sentence

    def test_it_is_strictly_cheaper_where_backing_off_pays(self, model, built):
        """
        Pinned rather than merely noted, because the day this stops being
        true the construction has changed in a way worth noticing.
        """
        arcs, finals, state_of = built
        start = start_state(state_of)
        fst = total_weight(arcs, finals, SYMBOLS, ["B", "A"], start=start)
        rule = model.weight(["B", "A"])
        assert rule - fst == pytest.approx(0.05 * math.log(10.0), abs=1e-9)

    def test_when_they_agree_is_not_a_property_worth_claiming(self, model, built):
        """
        It would be convenient if the two agreed except on some
        characterisable set of sentences, and they do not. On this model
        `A B` agrees while `A B C` is cheaper by the FST, and the
        difference is not about which n-grams are present -- it is about
        whether the whole path's arithmetic happens to favour an early
        back-off. So this records that the gap is real and one-sided, and
        makes no claim about where it falls.
        """
        arcs, finals, state_of = built
        start = start_state(state_of)
        gaps = {}
        for sentence in (["A"], ["A", "B"], ["A", "B", "C"], ["B", "A"]):
            fst = total_weight(arcs, finals, SYMBOLS, sentence, start=start)
            gaps[" ".join(sentence)] = model.weight(sentence) - fst
        # One-sided, always.
        assert all(gap >= -1e-9 for gap in gaps.values()), gaps
        # And not uniform, which is the point.
        assert any(gap > 1e-6 for gap in gaps.values()), gaps

    def test_an_unknown_word_has_no_path(self, built):
        arcs, finals, state_of = built
        syms = dict(SYMBOLS, ZZZ=99)
        assert (
            total_weight(arcs, finals, syms, ["ZZZ"], start=start_state(state_of))
            is None
        )


@pytest.mark.skipif(
    not (REFERENCE / "graph" / "ctc" / "G.fst.txt").is_file(),
    reason="no committed G.fst to compare against",
)
class TestAgainstKaldisOwnFst:
    """
    Functional equivalence with the `G.fst` Kaldi produced for yesno.

    A unigram, so this is exhaustive rather than sampled. The same
    comparison on mini-librispeech's 3-gram agrees on 16 of 16 sampled
    sentences, including one where both FSTs deviate from the back-off rule
    by the same 2.05 nats -- but that model is 13MB and not committed, so
    the repeatable test is this one.
    """

    @pytest.fixture
    def words(self):
        table = {}
        for line in (REFERENCE / "lang" / "words.txt").read_text().splitlines():
            fields = line.split()
            if len(fields) == 2:
                table[fields[0]] = int(fields[1])
        return table

    @pytest.fixture
    def kaldi(self):
        arcs, finals, start = {}, {}, None
        for line in (
            (REFERENCE / "graph" / "ctc" / "G.fst.txt").read_text().splitlines()
        ):
            fields = line.split()
            if len(fields) >= 4:
                source = int(fields[0])
                if start is None:
                    start = source
                arcs.setdefault(source, []).append(
                    (
                        int(fields[1]),
                        int(fields[2]),
                        float(fields[4]) if len(fields) > 4 else 0.0,
                    )
                )
            elif fields:
                finals[int(fields[0])] = float(fields[1]) if len(fields) > 1 else 0.0
        return arcs, finals, start

    @staticmethod
    def best(by_source, finals, words, sentence, start, disambig):
        import heapq

        ids = [words[w] for w in sentence]
        seen, queue, answer = {}, [(0.0, start, 0)], None
        while queue:
            cost, state, position = heapq.heappop(queue)
            key = (state, position)
            if key in seen and seen[key] <= cost:
                continue
            seen[key] = cost
            if position == len(ids) and state in finals:
                total = cost + finals[state]
                answer = total if answer is None else min(answer, total)
            for destination, ilabel, weight in by_source.get(state, []):
                if ilabel == disambig:
                    heapq.heappush(queue, (cost + weight, destination, position))
                elif position < len(ids) and ilabel == ids[position]:
                    heapq.heappush(queue, (cost + weight, destination, position + 1))
        return answer

    @pytest.mark.parametrize(
        "sentence",
        [
            ["YES"],
            ["NO"],
            ["YES", "NO"],
            ["NO", "YES"],
            ["YES", "YES", "NO"],
            ["NO"] * 6,
            ["YES", "NO", "NO", "YES", "NO", "YES", "YES", "NO"],
        ],
    )
    def test_the_two_fsts_agree(self, words, kaldi, sentence):
        model = read_arpa(YESNO_ARPA)
        arcs, finals, state_of = build_arcs(model, words)
        mine = {}
        for arc in arcs:
            mine.setdefault(arc[0], []).append((arc[1], arc[2], arc[4]))

        kaldi_arcs, kaldi_finals, kaldi_start = kaldi
        theirs = self.best(
            kaldi_arcs, kaldi_finals, words, sentence, kaldi_start, words["#0"]
        )
        ours = self.best(
            mine, finals, words, sentence, start_state(state_of), words["#0"]
        )
        assert theirs is not None and ours is not None
        assert ours == pytest.approx(
            theirs, rel=1e-6, abs=1e-6
        ), f"{' '.join(sentence)}: Kaldi {theirs:.6f}, this {ours:.6f}"

    def test_the_final_state_count_matches(self, words, kaldi):
        _kaldi_arcs, kaldi_finals, _start = kaldi
        model = read_arpa(YESNO_ARPA)
        _arcs, finals, _state_of = build_arcs(model, words)
        assert len(finals) == len(kaldi_finals)
