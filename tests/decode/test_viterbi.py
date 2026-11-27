"""
Tests for token-passing Viterbi decoding.

Tier 2. Three kinds of test, and the second is the one that matters.

**Hand-written graphs**, as in `test_decoder.py`: the right answer is known
by construction, so a graph with one path tests that the search follows it
and a graph with two tests that the acoustics choose.

**An oracle.** `k2.intersect_dense` is an exhaustive intersection and
`k2.shortest_path` an exact Viterbi over the result, so on a graph small
enough to intersect exhaustively the best path is *known* rather than
agreed. With pruning switched off this search must reproduce it exactly --
both the cost and the word sequence -- and it is checked on hundreds of
random graphs at three factoring thresholds. This is a stronger standard
than `test_decoder.py` could hold `intersect_dense_pruned` to, which is
only fair: that module is being compared with itself.

**The factoring is exactly weight-preserving**, so the threshold must not
change any answer. That is asserted directly, because the whole argument
for factoring rests on it: a token at the group's state costs the group's
minimum, so it survives exactly when a member would have, and the
non-emitting pass then gives each member its true cost under the same
cutoff.

What is *not* tested here is agreement with `decode-faster` on real audio.
Two pruned searches can disagree on a hard utterance while both being
correct, and the recipe's two stages plus the WER ledger are what judge
that -- the same division as for the k2 decoder.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import pytest

k2 = pytest.importorskip("k2")
torch = pytest.importorskip("torch")

from BenNevis.decode.decoder import words_from_best_path  # noqa: E402
from BenNevis.decode.viterbi import (  # noqa: E402
    SearchGraph,
    ViterbiDecoder,
    decode_archive,
    decode_shards,
)

pytestmark = pytest.mark.k2

#: Wide enough that nothing is pruned, so a test about the search is not
#: secretly a test about the beam.
UNPRUNED = dict(beam=1e9, max_active=10**9, min_active=0)

#: One path: tokens A(1) then B(2), emitting word 1.
ONE_PATH = "0 1 1 1 0.0\n1 2 2 0 0.0\n2\n"

#: Two one-token paths: A(1) -> word 1, B(2) -> word 2.
TWO_WORDS = "0 1 1 1 0.0\n0 2 2 2 0.0\n1\n2\n"

#: Four arcs out of state 0 sharing the input label 1, to four different
#: states with four different words and four different weights. This is
#: the shape factoring exists for, in miniature.
PARALLEL = (
    "0 1 1 1 0.5\n" "0 2 1 2 0.25\n" "0 3 1 3 1.5\n" "0 4 1 4 0.75\n" "1\n2\n3\n4\n"
)


def graph_from(text):
    return k2.arc_sort(k2.Fsa.from_openfst(text, acceptor=False))


def peaked(tokens, num_tokens=5):
    """Log probabilities that make `tokens` overwhelmingly likely."""
    logits = torch.full((len(tokens), num_tokens), -20.0)
    for frame, token in enumerate(tokens):
        logits[frame, token] = 0.0
    return logits.log_softmax(-1)


def decode_once(text, tokens, threshold=8, num_tokens=5, **search):
    graph = SearchGraph.from_fsa(graph_from(text), factor_threshold=threshold)
    options = dict(UNPRUNED)
    options.update(search)
    return ViterbiDecoder(graph, **options).decode(peaked(tokens, num_tokens))


class TestItFindsThePath:
    def test_it_follows_the_only_path(self):
        assert decode_once(ONE_PATH, [1, 2]) == [1]

    def test_the_acoustics_choose_between_two(self):
        assert decode_once(TWO_WORDS, [1]) == [1]
        assert decode_once(TWO_WORDS, [2]) == [2]

    def test_a_graph_it_cannot_finish_still_gives_its_best(self):
        """
        Kaldi's `GetBestPath` falls back to the best token anywhere when no
        token reached a final state, and warns. An undertrained model makes
        that common rather than exotic, which is why the fallback is here
        and not an error -- and it is the one place this search deliberately
        differs from `intersect_dense_pruned`, which returns nothing.
        """
        # One frame against a graph that needs two to reach the final state.
        assert decode_once(ONE_PATH, [1]) == [1]

    def test_no_frames_decodes_to_nothing(self):
        graph = SearchGraph.from_fsa(graph_from(ONE_PATH))
        out = ViterbiDecoder(graph, **UNPRUNED).decode(torch.zeros(0, 5))
        assert out == []

    def test_it_rejects_a_batch(self):
        graph = SearchGraph.from_fsa(graph_from(ONE_PATH))
        with pytest.raises(ValueError, match="frames, tokens"):
            ViterbiDecoder(graph, **UNPRUNED).decode(torch.zeros(1, 3, 5))

    def test_it_names_a_model_too_small_for_the_graph(self):
        """
        The graph-and-model mismatch that cost the first real k2 decoding
        run, reported here rather than as an index error from inside the
        search.
        """
        graph = SearchGraph.from_fsa(graph_from(TWO_WORDS))
        with pytest.raises(ValueError, match="different symbol tables"):
            ViterbiDecoder(graph, **UNPRUNED).decode(torch.zeros(4, 2))


class TestFactoringChangesNoAnswer:
    """
    The threshold is a memory and speed knob and must not be anything else.

    On the real 43M-arc graph, thresholds 2, 4, 8 and 16 gave the same WER
    and the same substitution, deletion and insertion counts to the error.
    Here the same claim is made where it can be checked exactly.
    """

    @pytest.mark.parametrize("threshold", [1, 2, 3, 4, 5, 10**9])
    def test_the_cheapest_parallel_arc_wins_whatever_the_threshold(self, threshold):
        # Arc to state 2 is the cheapest at 0.25, so word 2.
        assert decode_once(PARALLEL, [1], threshold=threshold) == [2]

    @pytest.mark.parametrize("threshold", [2, 4, 10**9])
    def test_a_factored_group_keeps_every_member_reachable(self, threshold):
        """
        Not just the cheapest: the whole point is that the others are still
        there, so a later frame can prefer one of them. Here state 3's
        continuation is the only one that accepts token 4.
        """
        text = PARALLEL + "3 5 4 7 0.0\n5\n"
        assert decode_once(text, [1, 4], threshold=threshold) == [3, 7]

    def test_factoring_is_what_the_threshold_says(self):
        """
        The counts, so the knob is known to be doing something -- a test
        that passed whether or not anything was factored would license the
        threshold being ignored.
        """
        fsa = graph_from(PARALLEL)
        off = SearchGraph.from_fsa(fsa, factor_threshold=10**9)
        on = SearchGraph.from_fsa(fsa, factor_threshold=4)
        assert off.num_non_emitting_arcs == 0
        assert off.num_states == fsa.shape[0]
        assert on.num_non_emitting_arcs == 4
        assert on.num_states == fsa.shape[0] + 1
        # And the arc count does not grow: four arcs became one plus four,
        # which is the trade the threshold exists to bound.
        assert on.num_emitting_arcs == off.num_emitting_arcs - 3

    def test_a_group_below_the_threshold_is_left_alone(self):
        on = SearchGraph.from_fsa(graph_from(PARALLEL), factor_threshold=5)
        assert on.num_non_emitting_arcs == 0


class TestAgainstAnExhaustiveIntersection:
    """
    Unpruned, this search must agree with k2 exactly.

    A random graph each time, with self-loops, several arcs per state,
    weighted final states and parallel arcs -- all of which the real graph
    has and a hand-written one tends not to. Where k2 finds no path to a
    final state at the last frame the two deliberately differ, so those
    draws are skipped and the fallback is tested on its own above.
    """

    @staticmethod
    def random_graph(generator):
        states = int(torch.randint(3, 12, (), generator=generator))
        tokens = int(torch.randint(2, 6, (), generator=generator))
        lines = []
        for state in range(states - 1):
            for _ in range(int(torch.randint(1, 5, (), generator=generator))):
                destination = int(torch.randint(0, states - 1, (), generator=generator))
                label = int(torch.randint(0, tokens, (), generator=generator))
                weight = round(float(torch.rand((), generator=generator)) * 3, 3)
                lines.append(f"{state} {destination} {label} {label + 1} {weight}")
            if int(torch.randint(0, 3, (), generator=generator)) == 0:
                weight = round(float(torch.rand((), generator=generator)) * 2, 3)
                lines.append(f"{state} {weight}")
        return "\n".join(lines) + "\n", tokens

    @pytest.mark.parametrize("threshold", [2, 4, 10**9])
    def test_it_reproduces_the_exact_best_path(self, threshold):
        generator = torch.Generator().manual_seed(20261127)
        compared = 0
        for _trial in range(120):
            text, tokens = self.random_graph(generator)
            frames = int(torch.randint(1, 9, (), generator=generator))
            try:
                fsa = graph_from(text)
            except (RuntimeError, ValueError):
                continue
            if fsa.num_arcs == 0:
                continue
            log_probs = torch.randn(1, frames, tokens, generator=generator).log_softmax(
                -1
            )
            segments = torch.tensor([[0, 0, frames]], dtype=torch.int32)
            try:
                lattice = k2.intersect_dense(
                    fsa, k2.DenseFsaVec(log_probs, segments), output_beam=1e9
                )
                best = k2.shortest_path(lattice, use_double_scores=True)
            except (RuntimeError, ValueError):
                continue
            if best.num_arcs == 0:
                continue
            cost = -float(best.get_tot_scores(True, False)[0])
            if cost != cost or abs(cost) > 1e8:
                continue
            compared += 1
            graph = SearchGraph.from_fsa(fsa, factor_threshold=threshold)
            got = ViterbiDecoder(graph, **UNPRUNED).decode(log_probs[0])
            assert got == words_from_best_path(best)[0], text
        # A test that compared nothing would pass, which is how the `G.fst`
        # gate in #41 managed to be a test that could not fail.
        assert compared > 40, f"only {compared} graphs were comparable"


class TestPruning:
    def test_the_beam_can_lose_the_best_path(self):
        """
        Pruning has to be able to bite, or a test of the beam proves
        nothing. Token 2's path costs 1.5 more than token 1's, so a beam
        below that drops it.
        """
        text = "0 1 1 1 0.0\n0 2 1 2 1.5\n1 3 2 0 0.0\n2 3 3 0 0.0\n3\n"
        # Frame 2 wants token 3, which only state 2 accepts -- but with a
        # beam of 1.0 the token at state 2 is already gone.
        assert decode_once(text, [1, 3], beam=1.0) != [2]
        assert decode_once(text, [1, 3], beam=4.0) == [2]

    def test_max_active_keeps_at_most_that_many(self):
        graph = SearchGraph.from_fsa(graph_from(PARALLEL), factor_threshold=10**9)
        decoder = ViterbiDecoder(graph, beam=1e9, max_active=1, min_active=0)
        # Only the cheapest survives the cap, which is the cheapest arc.
        assert decoder.decode(peaked([1])) == [2]

    def test_min_active_widens_the_beam(self):
        """
        Kaldi's `--min-active` loosens a cutoff rather than leave too few
        tokens.

        Three candidates at 0.0, 9.0 and 10.0 above the best. A beam of 1.0
        keeps only the first, so the path through state 2 is gone and token
        3 cannot be taken in the next frame; `min_active` 2 raises the
        cutoff to the second cheapest and brings it back.
        """
        text = (
            "0 1 1 1 0.0\n0 2 1 2 9.0\n0 3 1 3 10.0\n" "1 4 2 0 0.0\n2 4 3 0 0.0\n4\n"
        )
        assert decode_once(text, [1, 3], beam=1.0, min_active=2) == [2]
        assert decode_once(text, [1, 3], beam=1.0, min_active=0) != [2]

    @pytest.mark.parametrize(
        "options, message",
        [
            (dict(beam=0.0), "beam must be positive"),
            (dict(max_active=0), "max_active must be at least 1"),
            (dict(min_active=-1), "min_active must not be negative"),
            (dict(min_active=10, max_active=5), "above max_active"),
        ],
    )
    def test_it_refuses_settings_that_cannot_be_met(self, options, message):
        graph = SearchGraph.from_fsa(graph_from(ONE_PATH))
        settings = dict(beam=16.0, max_active=100, min_active=0)
        settings.update(options)
        with pytest.raises(ValueError, match=message):
            ViterbiDecoder(graph, **settings)


class TestSearchGraph:
    def test_it_refuses_a_graph_that_is_not_arc_sorted(self):
        """
        The grouping is read off the arc order rather than recomputed, so
        an unsorted graph would group silently wrongly -- a decoder that is
        merely a bit worse, which is the hardest kind of defect to notice.
        """
        # Written with label 2 before label 1 and never arc-sorted, which
        # is what a graph straight out of `from_openfst` can look like.
        unsorted = k2.Fsa.from_openfst(
            "0 1 2 2 0.0\n0 2 1 1 0.0\n1\n2\n", acceptor=False
        )
        assert bool(
            (unsorted.arcs.values()[0, 2] > unsorted.arcs.values()[1, 2]).item()
        ), "this graph is already sorted, so the test proves nothing"
        with pytest.raises(ValueError, match="not sorted by"):
            SearchGraph.from_fsa(unsorted)

    def test_final_weights_leave_the_arc_list(self):
        """
        k2 writes a final state as an arc labelled -1; left there, every
        frame would expand -1 as if it were a token.
        """
        graph = SearchGraph.from_fsa(graph_from(TWO_WORDS))
        assert int((graph.emit_label < 0).sum()) == 0
        assert int(torch.isfinite(graph.final_cost).sum()) == 2

    def test_a_final_weight_is_added_to_the_path(self):
        """
        Two equal paths distinguished only by the weight on their final
        state, so the final weight cannot be ignored.
        """
        text = "0 1 1 1 0.0\n0 2 1 2 0.0\n1 5.0\n2 0.0\n"
        assert decode_once(text, [1]) == [2]

    def test_the_token_count_comes_from_the_labels(self):
        graph = SearchGraph.from_fsa(graph_from(PARALLEL))
        assert graph.num_tokens == 2  # labels 0 and 1 are both possible


class TestDecodeArchive:
    def test_it_keeps_archive_order(self):
        graph = SearchGraph.from_fsa(graph_from(TWO_WORDS))
        out = decode_archive(
            graph,
            [("b", peaked([2])), ("a", peaked([1]))],
            {1: "ONE", 2: "TWO"},
            **UNPRUNED,
        )
        assert out == [("b", ["TWO"]), ("a", ["ONE"])]

    def test_it_names_a_misshapen_utterance(self):
        graph = SearchGraph.from_fsa(graph_from(TWO_WORDS))
        with pytest.raises(ValueError, match="utt-7"):
            decode_archive(graph, [("utt-7", torch.zeros(1, 2, 5))], {}, **UNPRUNED)

    def test_a_word_outside_the_table_is_dropped_not_crashed_on(self):
        graph = SearchGraph.from_fsa(graph_from(TWO_WORDS))
        out = decode_archive(graph, [("a", peaked([1]))], {2: "TWO"}, **UNPRUNED)
        assert out == [("a", [])]


class TestShards:
    """
    Several processes must give exactly what one process gives.

    The search is deterministic and a shard is a contiguous block, so this
    is an equality rather than a tolerance -- and it is the only thing that
    makes `--nj` safe to turn on, since a sharding bug would show up as a
    few utterances decoded against the wrong features, which scores as a
    slightly worse WER and nothing else.
    """

    @staticmethod
    def archive(count):
        return [(f"utt-{i}", peaked([1 if i % 2 else 2])) for i in range(count)]

    def test_shards_agree_with_one_process(self):
        graph = SearchGraph.from_fsa(graph_from(TWO_WORDS))
        words = {1: "ONE", 2: "TWO"}
        items = self.archive(9)
        one = decode_archive(graph, iter(items), words, **UNPRUNED)
        for jobs in (1, 2, 3, 4):
            size = -(-len(items) // jobs)
            blocks = [items[i : i + size] for i in range(0, len(items), size)]
            shards = [(lambda b: (lambda: iter(b)))(b) for b in blocks]
            assert (
                decode_shards(graph, shards, words, **UNPRUNED) == one
            ), f"{jobs} shards disagreed with one process"

    def test_a_failing_shard_is_reported_rather_than_dropped(self):
        """
        A short archive scores as a run of deletions and looks like a bad
        model, so a shard that dies has to fail the call.
        """
        graph = SearchGraph.from_fsa(graph_from(TWO_WORDS))

        def broken():
            raise RuntimeError("no features here")
            yield  # pragma: no cover

        shards = [lambda: iter(self.archive(2)), broken]
        with pytest.raises(RuntimeError, match="shard 1"):
            decode_shards(graph, shards, {1: "ONE", 2: "TWO"}, **UNPRUNED)

    def test_a_failing_shard_is_named_the_same_way_without_fork(self, monkeypatch):
        """
        Windows has no `fork`, so `decode_shards` decodes the shards in
        this process -- and used to let the shard's own exception through,
        so the message named no shard there and did on Linux. Pinned
        because the platform that takes this branch is the one nobody runs
        locally.
        """
        import BenNevis.decode.viterbi as module

        monkeypatch.setattr(
            module.multiprocessing, "get_all_start_methods", lambda: ["spawn"]
        )
        graph = SearchGraph.from_fsa(graph_from(TWO_WORDS))

        def broken():
            raise RuntimeError("no features here")
            yield  # pragma: no cover

        shards = [lambda: iter(self.archive(2)), broken]
        with pytest.raises(RuntimeError, match="shard 1"):
            decode_shards(graph, shards, {1: "ONE", 2: "TWO"}, **UNPRUNED)
