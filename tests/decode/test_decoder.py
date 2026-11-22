"""
Tests for one-best decoding with k2.

Tier 2. These are the tests `decode-faster` never had: the entry-point
scripts are reachable from no tier but 3, which is how the regression in #5
got in, and a Python decoder can be tested where a Kaldi binary could not.

The graphs here are tiny and hand-written so the right answer is known by
construction rather than by comparison -- a graph with one path tests that
the decoder follows it, and a graph with two tests that the acoustics
choose between them.

**Deliberately not compared against `decode-faster`'s output.** The two
searches prune differently: `decode-faster` uses a beam on the arc-cost
difference and a cap on active tokens, `intersect_dense_pruned` uses two
beams plus a floor and a cap on active states. A pruned search returns the
best path *it found*, so the two can disagree on a hard utterance while
both being correct. The WER ledger is what gates this.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import pytest

k2 = pytest.importorskip("k2")
torch = pytest.importorskip("torch")

from BenNevis.decode.decoder import decode, decode_archive  # noqa: E402

pytestmark = pytest.mark.k2


def graph_from(text):
    return k2.arc_sort(k2.Fsa.from_openfst(text, acceptor=False))


#: One path: tokens A(1) then B(2), emitting word 1.
ONE_PATH = "0 1 1 1 0.0\n1 2 2 0 0.0\n2\n"

#: Two one-token paths: A(1) -> word 1, B(2) -> word 2.
TWO_WORDS = "0 1 1 1 0.0\n0 2 2 2 0.0\n1\n2\n"


def peaked(tokens, num_tokens=3):
    """Log probabilities that make `tokens` overwhelmingly likely."""
    logits = torch.full((1, len(tokens), num_tokens), -20.0)
    for frame, token in enumerate(tokens):
        logits[0, frame, token] = 0.0
    return logits.log_softmax(-1)


class TestDecode:
    def test_it_follows_the_only_path(self):
        out = decode(
            graph_from(ONE_PATH),
            peaked([1, 2]),
            torch.tensor([2], dtype=torch.int32),
        )
        assert out == [[1]]

    def test_the_acoustics_choose_between_paths(self):
        graph = graph_from(TWO_WORDS)
        for token, word in ((1, 1), (2, 2)):
            out = decode(graph, peaked([token]), torch.tensor([1], dtype=torch.int32))
            assert out == [[word]], f"token {token}"

    def test_epsilon_outputs_are_dropped(self):
        """
        Only the first arc of a word carries it; the rest emit `<eps>`,
        which is 0 and not a word. k2's end-of-path marker is -1 and is
        not either.
        """
        out = decode(
            graph_from(ONE_PATH),
            peaked([1, 2]),
            torch.tensor([2], dtype=torch.int32),
        )
        assert 0 not in out[0]
        assert -1 not in out[0]

    def test_a_frame_count_the_graph_cannot_match_decodes_to_nothing(self):
        """
        `intersect_dense_pruned` needs a path of exactly the frame count,
        so a graph accepting two tokens has no answer for three frames.
        Worth pinning because the symptom is an empty transcription, which
        is easy to mistake for a bad model -- and an empty hypothesis used
        to be scored as its own utterance id.
        """
        out = decode(
            graph_from(ONE_PATH),
            peaked([1, 2, 1]),
            torch.tensor([3], dtype=torch.int32),
        )
        assert out == [[]]

    def test_a_batch_decodes_independently(self):
        graph = graph_from(TWO_WORDS)
        logits = torch.full((2, 1, 3), -20.0)
        logits[0, 0, 1] = 0.0
        logits[1, 0, 2] = 0.0
        out = decode(
            graph,
            logits.log_softmax(-1),
            torch.tensor([1, 1], dtype=torch.int32),
        )
        assert out == [[1], [2]]

    def test_the_acoustic_scale_is_applied(self):
        """
        A graph weight can outvote the acoustics when the scale is small
        enough, which is the whole point of the parameter -- and `aw_1.0`
        in the recipe's output directories has to keep meaning what it did.
        """
        # B is cheaper acoustically, but its arc costs 5 in the graph.
        text = "0 1 1 1 0.0\n0 2 2 2 5.0\n1\n2\n"
        graph = graph_from(text)
        logits = torch.full((1, 1, 3), -20.0)
        logits[0, 0, 1] = -2.0
        logits[0, 0, 2] = 0.0
        scores = logits.log_softmax(-1)
        lengths = torch.tensor([1], dtype=torch.int32)

        loud = decode(graph, scores, lengths, acoustic_scale=10.0)
        quiet = decode(graph, scores, lengths, acoustic_scale=0.01)
        assert loud == [[2]], "a large scale should let the acoustics win"
        assert quiet == [[1]], "a small scale should let the graph win"

    def test_wrong_dimensions_raise(self):
        graph = graph_from(ONE_PATH)
        with pytest.raises(ValueError, match="batch, frames, tokens"):
            decode(
                graph,
                torch.zeros(2, 3),
                torch.tensor([3], dtype=torch.int32),
            )

    def test_a_length_mismatch_raises(self):
        graph = graph_from(ONE_PATH)
        with pytest.raises(ValueError, match="entries for"):
            decode(graph, peaked([1, 2]), torch.tensor([2, 2]))


class TestWordsFromBestPath:
    def test_a_dense_aux_label_graph_works(self):
        graph = graph_from(ONE_PATH)
        assert not isinstance(graph.aux_labels, k2.RaggedTensor)
        out = decode(graph, peaked([1, 2]), torch.tensor([2], dtype=torch.int32))
        assert out == [[1]]

    def test_a_composed_graph_has_ragged_aux_labels(self):
        """
        Composition can put several output labels on one arc, and then
        `aux_labels` is ragged rather than a tensor. Which one it is
        depends on how the graph was built, not on anything the caller
        chose, so both paths have to work.
        """
        first = graph_from("0 1 1 1 0.0\n1\n")
        second = graph_from("0 1 1 1 0.0\n1\n")
        composed = k2.arc_sort(k2.connect(k2.compose(first, second)))
        out = decode(composed, peaked([1]), torch.tensor([1], dtype=torch.int32))
        assert out == [[1]]


class TestDecodeArchive:
    def test_it_decodes_each_utterance(self):
        graph = graph_from(TWO_WORDS)
        idx2word = {1: "YES", 2: "NO"}
        archive = [
            ("utt1", peaked([1])[0]),
            ("utt2", peaked([2])[0]),
        ]
        out = decode_archive(graph, archive, idx2word)
        assert out == [("utt1", ["YES"]), ("utt2", ["NO"])]

    def test_it_keeps_the_archive_order(self):
        """
        The output is merged against `ref.wrd.trn` by position in the
        recipe, so order is not cosmetic.
        """
        graph = graph_from(TWO_WORDS)
        archive = [(f"utt{i}", peaked([1 + i % 2])[0]) for i in range(6)]
        out = decode_archive(graph, archive, {1: "YES", 2: "NO"})
        assert [uttid for uttid, _ in out] == [f"utt{i}" for i in range(6)]

    def test_an_unknown_word_id_is_skipped_not_crashed(self):
        graph = graph_from(TWO_WORDS)
        out = decode_archive(graph, [("u", peaked([1])[0])], {2: "NO"})
        assert out == [("u", [])]

    def test_wrong_dimensions_name_the_utterance(self):
        """
        Rather than failing somewhere inside k2 with no idea which of
        several thousand utterances it was.
        """
        graph = graph_from(TWO_WORDS)
        with pytest.raises(ValueError, match="bad_utt"):
            decode_archive(graph, [("bad_utt", torch.zeros(3))], {1: "YES"})

    def test_an_empty_archive_gives_nothing(self):
        graph = graph_from(TWO_WORDS)
        assert decode_archive(graph, [], {1: "YES"}) == []
