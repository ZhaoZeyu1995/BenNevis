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

from BenNevis.decode.decoder import (  # noqa: E402
    aux_labels_per_arc,
    decode,
    decode_archive,
)

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

    def test_a_composed_graph_decodes(self):
        """
        A graph the caller built by composition, rather than by hand.

        This test used to be called `test_a_composed_graph_has_ragged_aux_labels`
        and asserted no such thing. Measured since: `k2.compose` of two
        transducers returns a **plain** `aux_labels` tensor, so the name
        described a mechanism this construction does not exercise --
        `k2.remove_epsilon` is what makes it ragged, and
        `TestAuxLabelsPerArc` covers that. What is left here is still
        worth keeping: composition is how the real graph is built, and
        the decoder has to follow one.
        """
        first = graph_from("0 1 1 1 0.0\n1\n")
        second = graph_from("0 1 1 1 0.0\n1\n")
        composed = k2.arc_sort(k2.connect(k2.compose(first, second)))
        assert not isinstance(composed.aux_labels, k2.RaggedTensor)
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


class TestDecodeArchiveBatches:
    """
    The batch dimension that was always there and always 1.

    `decode` takes `(batch, frames, tokens)` and
    `intersect_dense_pruned` is built for a batch, but `decode_archive`
    called it with `scores.unsqueeze(0)` per utterance -- so run
    45511013199 spent 4h16m48s on 900 of mini-librispeech's 1,089
    utterances, 17.1 seconds each, against Kaldi's `decode-faster`
    doing all of them at `nj=8` in a fraction of that.

    What these pin is the part a batch can get wrong rather than the
    speed, which no tier-2 graph can show: the answers must not depend
    on how the utterances were grouped, and a short utterance's padding
    must not be decoded as though it were audio.
    """

    def test_a_batch_gives_the_same_answers_as_one_at_a_time(self):
        """
        The property that matters. Lengths deliberately differ within
        the batch, since equal lengths are the case padding cannot get
        wrong.
        """
        graph = graph_from(ONE_PATH)
        archive = [
            ("utt1", peaked([1, 2])[0]),
            ("utt2", peaked([1, 2])[0]),
            ("utt3", peaked([1, 1, 2], num_tokens=3)[0]),
            ("utt4", peaked([1, 2])[0]),
            ("utt5", peaked([1, 2])[0]),
        ]
        idx2word = {1: "YES", 2: "NO"}
        sequential = decode_archive(graph, archive, idx2word, batch_size=1)
        for size in (2, 3, 5, 100):
            assert (
                decode_archive(graph, archive, idx2word, batch_size=size) == sequential
            ), f"batch_size={size} disagrees with one at a time"

    def test_the_padding_is_not_decoded(self):
        """
        The failure a batch invites, and it would look like a plausible
        transcription rather than a crash. `ONE_PATH` accepts exactly
        two frames, so a two-frame utterance batched with a five-frame
        one decodes to its word only if its own frame count reached k2's
        segments; were the padded length used instead, the graph could
        not match five frames and the answer would be empty.
        """
        graph = graph_from(ONE_PATH)
        archive = [
            ("short", peaked([1, 2])[0]),
            ("long", peaked([1, 2, 1, 2, 1])[0]),
        ]
        out = decode_archive(graph, archive, {1: "YES"}, batch_size=2)
        assert out[0] == ("short", ["YES"]), "the short utterance lost its word"
        assert out[1] == ("long", []), "the long one should match no path"

    def test_it_calls_the_search_once_per_batch(self):
        """
        Pinned by counting, because the saving is invisible on a graph
        small enough to test with: the point of the change is one call
        for eight utterances rather than eight.
        """
        from BenNevis.decode import decoder as decoder_module

        graph = graph_from(TWO_WORDS)
        archive = [(f"utt{i}", peaked([1])[0]) for i in range(7)]
        calls = []
        original = decoder_module.decode

        def counted(graph_arg, log_probs, lengths, **kwargs):
            calls.append(int(log_probs.shape[0]))
            return original(graph_arg, log_probs, lengths, **kwargs)

        decoder_module.decode = counted
        try:
            out = decoder_module.decode_archive(
                graph, archive, {1: "YES"}, batch_size=3
            )
        finally:
            decoder_module.decode = original
        assert calls == [3, 3, 1], f"batched as {calls}"
        assert len(out) == 7

    def test_a_batch_size_below_one_is_refused(self):
        """
        Rather than silently decoding nothing, which an archive of
        several thousand utterances would report as every utterance
        empty -- the same shape as a model that learned nothing.
        """
        graph = graph_from(TWO_WORDS)
        for size in (0, -1):
            with pytest.raises(ValueError, match="batch_size"):
                decode_archive(
                    graph, [("u", peaked([1])[0])], {1: "YES"}, batch_size=size
                )

    def test_a_token_count_that_differs_names_the_utterance(self):
        """
        A graph-and-model mismatch, and without this it surfaces as a
        shape error inside torch's padding with no utterance named.
        """
        graph = graph_from(TWO_WORDS)
        archive = [
            ("first", peaked([1], num_tokens=3)[0]),
            ("odd_one", peaked([1], num_tokens=4)[0]),
        ]
        with pytest.raises(ValueError, match="odd_one"):
            decode_archive(graph, archive, {1: "YES"}, batch_size=2)

    def test_wrong_dimensions_still_name_the_utterance_when_batching(self):
        """
        The isolation a batch gives up is memory, not diagnostics: the
        shape check runs before anything is batched, so the utterance is
        still named.
        """
        graph = graph_from(TWO_WORDS)
        archive = [("fine", peaked([1])[0]), ("bad_utt", torch.zeros(3))]
        with pytest.raises(ValueError, match="bad_utt"):
            decode_archive(graph, archive, {1: "YES"}, batch_size=4)


class TestAuxLabelsPerArc:
    """
    The other reduction of `aux_labels`, and the one `bin/align.py` needs.

    `words_from_best_path` flattens the ragged case, which is right for
    *which words were said*. An alignment cannot: it reads `labels[i]`
    and `aux_labels[i]` together to decide which frames a word covers, so
    it needs one entry per arc and a length matching `labels`.

    Getting this wrong broke stages 8 and 9 of the first real
    mini-librispeech run -- `RaggedTensor` has no `.numpy()` -- and yesno
    could not show it, because a four-word lexicon leaves `aux_labels` a
    plain tensor where 206,510 words do not.

    **`k2.remove_epsilon` is what makes it ragged**, measured rather than
    assumed: `k2.compose` of two transducers here returns a plain tensor,
    so the composition in the test above is not the mechanism despite its
    name. Removing an epsilon-input arc that carries a word has to put
    that word somewhere, and the surviving arc is where it goes.
    """

    @staticmethod
    def ragged(text):
        """
        A transducer whose epsilons have been removed, so ragged.

        Arguments
        ---------
        text: str
            OpenFST text for a transducer with at least one
            epsilon-input arc, or `aux_labels` comes back dense.
        """
        fsa = k2.remove_epsilon(k2.Fsa.from_openfst(text, acceptor=False))
        assert isinstance(fsa.aux_labels, k2.RaggedTensor), text
        return fsa

    def test_a_plain_tensor_is_passed_through(self):
        """The dense path must keep writing exactly what it wrote before."""
        best = k2.shortest_path(
            k2.create_fsa_vec([graph_from(ONE_PATH)]), use_double_scores=True
        )
        assert not isinstance(best.aux_labels, k2.RaggedTensor)
        assert aux_labels_per_arc(best) == best.aux_labels.cpu().tolist()

    def test_one_label_per_arc_whatever_the_shape(self):
        """
        The invariant `align.py` asserts, on a genuinely ragged graph.

        This is the test that would have caught the crash: it fails with
        an AttributeError against the old `aux_labels.numpy()`.
        """
        fsa = self.ragged("0 1 1 5 0.0\n1 2 0 7 0.0\n2 3 2 0 0.0\n3\n")
        labels = fsa.labels.cpu().tolist()
        assert len(aux_labels_per_arc(fsa)) == len(labels)

    def test_the_words_survive_epsilon_removal(self):
        """
        Both words are kept, one per arc, in order.

        The input arc carrying `7` was an epsilon and is gone; its word
        has to land on a real arc or the alignment loses a word.
        """
        fsa = self.ragged("0 1 1 5 0.0\n1 2 0 7 0.0\n2 3 2 0 0.0\n3\n")
        assert [x for x in aux_labels_per_arc(fsa) if x > 0] == [5, 7]

    def test_two_words_on_an_arc_is_reported(self):
        """
        Not resolved by dropping one: that would shift every word after
        it and the alignment would look plausible and be wrong.

        Built by hand, because a graph that does this is exactly what has
        not been observed yet -- the point is the behaviour if it occurs.
        """
        fsa = k2.Fsa.from_str("0 1 1 0.0\n1 2 -1 0.0\n2\n", acceptor=True)
        fsa.aux_labels = k2.RaggedTensor([[1], [2, 3]])
        with pytest.raises(ValueError, match="arc 1 carries 2 output labels"):
            aux_labels_per_arc(fsa)

    def test_an_empty_row_is_epsilon_and_a_lone_marker_survives(self):
        """
        The two shapes a wordless arc comes in.

        `remove_epsilon` empties the final arc's row rather than leaving
        `-1` there, so the ragged path reports 0 where the dense path
        reports -1. That asymmetry is k2's, not ours, and is kept rather
        than normalised: changing the dense path would change the ark
        yesno already writes.
        """
        fsa = k2.Fsa.from_str("0 1 1 0.0\n1 2 -1 0.0\n2\n", acceptor=True)
        fsa.aux_labels = k2.RaggedTensor([[], [-1]])
        assert aux_labels_per_arc(fsa) == [0, -1]
