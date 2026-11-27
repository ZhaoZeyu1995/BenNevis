"""
Tests for Lang, which loads a lang directory and compiles training graphs.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import io

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


class TestTheLexiconLoadsWithoutOpenFst:
    """
    The last thing on the training path that needed OpenFST.

    `load_lexicon` read the lexicon FST by shelling out to
    `fstprint {lang_dir}/L.fst`, and `compile_training_graph` genuinely
    needs the result -- so a first training run required OpenFST and
    Kaldi's compiled `L.fst` however much else had been ported. The
    language directory now carries `k2/L.fst.txt`, written in Python by
    `prepare_topo.sh --lexicon`, the same way it already carries
    `k2/topo.yaml`.
    """

    #: Kaldi's `L.fst`: phones in, words out. The fixture's `L_inv.pt` is
    #: this inverted, which is what `Lang` holds.
    LEXICON_FST = "0 0 1 1 0.0\n0 0 2 2 0.0\n0 0 3 3 0.0\n0 0.0\n"

    @staticmethod
    def without_cache(lang_dir, text=None):
        """The fixture's directory with `L_inv.pt` removed."""
        (lang_dir / "k2" / "L_inv.pt").unlink()
        if text is not None:
            (lang_dir / "k2" / "L.fst.txt").write_text(text, encoding="utf-8")
        return lang_dir

    def test_it_reads_the_text_the_preparation_wrote(self, lang_dir):
        directory = self.without_cache(lang_dir, self.LEXICON_FST)
        obj = Lang(str(directory), load_topo=True, load_lexicon=True)
        assert obj.L_inv.num_arcs > 0
        # Phones on the output side once inverted, which is the orientation
        # `compile_training_graph` composes against.
        assert sorted(set(obj.L_inv.labels.tolist())) == [-1, 1, 2, 3]

    def test_nothing_on_the_training_path_shells_out(self, lang_dir, monkeypatch):
        """
        The claim, asserted rather than described: with the text present,
        `Lang` must not run a shell at all. `os.popen` is made to raise, so
        a fallback would fail loudly instead of quietly working on a
        machine that happens to have OpenFST -- which is the only kind of
        machine this has ever been run on.
        """
        import os as os_module
        import subprocess

        directory = self.without_cache(lang_dir, self.LEXICON_FST)

        def forbidden(*args, **kwargs):
            raise AssertionError("the training path shelled out")

        monkeypatch.setattr(os_module, "popen", forbidden)
        monkeypatch.setattr(subprocess, "run", forbidden)
        monkeypatch.setattr(subprocess, "Popen", forbidden)
        monkeypatch.setattr(subprocess, "check_output", forbidden)

        obj = Lang(str(directory), load_topo=True, load_lexicon=True)
        # And the thing training actually asks for, since loading alone
        # would not prove the graph can be compiled.
        graph = obj.compile_training_graph([[1, 2]], torch.device("cpu"))
        assert graph.shape[0] > 0

    def test_the_cache_still_wins(self, lang_dir):
        """
        `k2/L_inv.pt` is checked first, so a directory that has been
        trained from once does not re-read the text.
        """
        (lang_dir / "k2" / "L.fst.txt").write_text("", encoding="utf-8")
        obj = Lang(str(lang_dir), load_topo=True, load_lexicon=True)
        assert obj.L_inv.num_arcs > 0

    def test_it_writes_the_cache_from_the_text(self, lang_dir):
        directory = self.without_cache(lang_dir, self.LEXICON_FST)
        Lang(str(directory), load_topo=True, load_lexicon=True)
        assert (directory / "k2" / "L_inv.pt").is_file()

    def test_the_fallback_warns_and_is_reachable(self, lang_dir, monkeypatch, caplog):
        """
        Five of seven recipes still take this route, so it has to work and
        has to say that it did.
        """
        import os as os_module

        directory = self.without_cache(lang_dir)
        monkeypatch.setattr(
            os_module, "popen", lambda cmd: io.StringIO(self.LEXICON_FST)
        )
        with caplog.at_level("WARNING"):
            obj = Lang(str(directory), load_topo=True, load_lexicon=True)
        assert obj.L_inv.num_arcs > 0
        assert "fstprint" in caplog.text

    def test_an_empty_fstprint_is_an_error_not_an_empty_lexicon(
        self, lang_dir, monkeypatch
    ):
        """
        OpenFST missing from PATH makes `os.popen` yield nothing, and an
        empty lexicon would train against an empty graph -- a run that
        burns hours and converges to nonsense rather than one that fails.
        """
        import os as os_module

        directory = self.without_cache(lang_dir)
        monkeypatch.setattr(os_module, "popen", lambda cmd: io.StringIO(""))
        with pytest.raises(RuntimeError, match="produced nothing"):
            Lang(str(directory), load_topo=True, load_lexicon=True)
