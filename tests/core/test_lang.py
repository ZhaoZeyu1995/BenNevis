"""
Tests for Lang, which loads a lang directory and compiles training graphs.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("k2")

from BenNevis.core.lang import Lang  # noqa: E402
from BenNevis.core.losses import GraphLoss  # noqa: E402

pytestmark = pytest.mark.k2


class TestSymbolTables:
    """The symbol tables drive the network's output size and the graphs."""

    def test_tokens_determine_the_network_output_size(self, lang):
        """
        num_nn_output comes from k2/tokens.txt, which excludes <eps> so that it
        matches the network's output layer exactly.
        """
        assert lang.num_nn_output == 4
        assert lang.tokens == ["<blk>", "a", "b", "c"]

    def test_phones_include_epsilon_but_tokens_do_not(self, lang):
        assert "<eps>" in lang.phones
        assert "<eps>" not in lang.tokens

    @pytest.mark.parametrize("table", ["word", "phone", "token"])
    def test_index_maps_round_trip(self, lang, table):
        to_idx = getattr(lang, f"{table}2idx")
        to_sym = getattr(lang, f"idx2{table}")
        for sym, idx in to_idx.items():
            assert to_sym[idx] == sym

    def test_blank_is_token_zero(self, lang):
        """CTC and every other topology here assume blank at index 0."""
        assert lang.token2idx["<blk>"] == 0


class TestLexicon:
    """The lexicon maps words to phone sequences."""

    def test_align_lexicon_is_loaded_keyed_by_word_id(self, lang):
        assert lang.lexicon[1] == [1]
        assert lang.lexicon[2] == [2]

    def test_wids2pids_expands_a_word_sequence(self, lang):
        assert lang.wids2pids([[1, 2, 3]]) == [[1, 2, 3]]

    def test_wids2pids_handles_a_batch(self, lang):
        assert lang.wids2pids([[1, 2], [3], [2, 2]]) == [[1, 2], [3], [2, 2]]

    def test_wids2pids_preserves_repetition(self, lang):
        """A repeated word must not be collapsed; CTC needs the repeat."""
        assert lang.wids2pids([[2, 2, 2]]) == [[2, 2, 2]]

    def test_l_inv_is_loaded_without_openfst(self, lang):
        """
        k2/L_inv.pt is read with torch.load, so a prebuilt one means no
        fstprint is needed. Verified loadable under weights_only=True, since
        as_dict contains only tensors.
        """
        assert lang.L_inv is not None
        assert lang.L is not None

    def test_l_is_the_inverse_of_l_inv(self, lang):
        assert lang.L.num_arcs == lang.L_inv.num_arcs


class TestCompileTrainingGraph:
    """compile_training_graph composes the topology with the transcript."""

    def test_one_graph_per_utterance(self, lang):
        graph = lang.compile_training_graph([[1, 2, 3], [2, 2]], torch.device("cpu"))
        assert graph.shape[0] == 2

    def test_graph_carries_no_gradient(self, lang):
        graph = lang.compile_training_graph([[1, 2]], torch.device("cpu"))
        assert graph.requires_grad is False

    def test_graph_accepts_its_own_transcript(self, lang):
        """
        Behavioural check: log probabilities that spell the target almost
        exactly should give a loss near zero through the compiled graph.
        """
        target = [1, 2, 3]
        log_probs = torch.full((1, 3, 4), -20.0)
        for t, token in enumerate(target):
            log_probs[0, t, token] = 0.0
        log_probs = log_probs.log_softmax(-1)

        loss = GraphLoss(lang, use_den=False, reduction="sum")(
            log_probs,
            torch.tensor([3], dtype=torch.int32),
            [[1, 2, 3]],
            torch.tensor([3], dtype=torch.int32),
        )

        assert loss.item() == pytest.approx(0.0, abs=1e-3)

    def test_graph_rejects_a_different_transcript(self, lang):
        """The same input scored against the wrong words must cost far more."""
        log_probs = torch.full((1, 3, 4), -20.0)
        for t, token in enumerate([1, 2, 3]):
            log_probs[0, t, token] = 0.0
        log_probs = log_probs.log_softmax(-1)
        args = (torch.tensor([3], dtype=torch.int32),)

        right = GraphLoss(lang, use_den=False, reduction="sum")(
            log_probs, *args, [[1, 2, 3]], torch.tensor([3], dtype=torch.int32)
        )
        wrong = GraphLoss(lang, use_den=False, reduction="sum")(
            log_probs, *args, [[3, 2, 1]], torch.tensor([3], dtype=torch.int32)
        )

        assert wrong.item() > right.item() + 10.0


class TestConstruction:
    """Loading is optional, because each part has its own dependencies."""

    def test_lexicon_and_topo_are_not_loaded_by_default(self, lang_dir):
        """
        Lang(dir) reads only the symbol tables and align_lexicon. The topology
        in particular needs OpenFST's fstprint, so loading it must stay opt-in.
        """
        bare = Lang(str(lang_dir))
        assert not hasattr(bare, "topo")
        assert not hasattr(bare, "L_inv")

    def test_missing_directory_fails(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            Lang(str(tmp_path / "nope"))
