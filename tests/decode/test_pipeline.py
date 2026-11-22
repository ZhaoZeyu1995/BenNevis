"""
Tests for the wiring between the decoder and a recipe.

The `.trn` tests are tier 1 -- they need nothing but the standard library,
and they cover the format that `decode_faster.sh`'s `awk` got wrong on an
empty hypothesis. The end-to-end test is tier 2 and is the strongest check
available without a trained model: a dictionary, an ARPA model and a
topology spec in, the right words out, with no Kaldi and no OpenFST
anywhere in the path.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from pathlib import Path

import pytest

from BenNevis.decode.pipeline import (
    format_trn,
    read_symbol_table,
    read_utt2spk,
    write_trn,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LANG = REPO_ROOT / "tests" / "golden" / "reference" / "yesno" / "lang"
YESNO = REPO_ROOT / "egs" / "yesno" / "asr" / "input"


class TestReaders:
    def test_a_symbol_table_reads(self, tmp_path):
        path = tmp_path / "words.txt"
        path.write_text("<eps> 0\nYES 1\nNO 2\n")
        assert read_symbol_table(path) == {"<eps>": 0, "YES": 1, "NO": 2}

    def test_a_malformed_line_is_skipped(self, tmp_path):
        path = tmp_path / "words.txt"
        path.write_text("<eps> 0\nrubbish\nYES 1\n")
        assert read_symbol_table(path) == {"<eps>": 0, "YES": 1}

    def test_utt2spk_reads(self, tmp_path):
        path = tmp_path / "utt2spk"
        path.write_text("utt1 spkA\nutt2 spkB\n")
        assert read_utt2spk(path) == {"utt1": "spkA", "utt2": "spkB"}


class TestTrnFormat:
    def test_the_form_is_words_then_bracketed_id(self):
        text = format_trn([("u1", ["YES", "NO"])], {"u1": "spk"})
        assert text == "YES NO (spk-u1)\n"

    def test_an_empty_hypothesis_keeps_its_line(self):
        """
        The case `decode_faster.sh` got wrong. Its `awk` stripped the id by
        searching for the first field's text, and with no fields that
        returned the whole line -- so a file name was scored as a
        transcription. The error *count* was the same either way, which is
        why the WER looked plausible while the output was nonsense.
        """
        assert format_trn([("u1", [])], {"u1": "spk"}) == " (spk-u1)\n"

    def test_a_missing_speaker_uses_the_utterance_id(self):
        """
        Rather than dropping the line. A shorter hypothesis than reference
        makes `score` refuse the whole comparison, which hides one odd
        utterance behind a total failure.
        """
        assert format_trn([("u1", ["YES"])], {}) == "YES (u1-u1)\n"

    def test_order_is_preserved(self):
        text = format_trn([("b", ["ONE"]), ("a", ["TWO"])], {"a": "s", "b": "s"})
        assert text.splitlines() == ["ONE (s-b)", "TWO (s-a)"]

    def test_no_hypotheses_gives_empty_text(self):
        assert format_trn([], {}) == ""

    def test_it_round_trips_through_read_trn(self, tmp_path):
        """
        `read_trn` is what scoring uses, so the two have to agree on the
        format rather than each being plausible on its own.
        """
        from BenNevis.utils.scoring import read_trn

        path = tmp_path / "hyp.wrd.trn"
        write_trn(
            [("u1", ["YES", "NO"]), ("u2", ["NO"])],
            {"u1": "s1", "u2": "s2"},
            path,
        )
        # `read_trn` keys on the whole bracketed id, not the utterance
        # alone -- and so does the reference `predict.py` writes, which is
        # what matters: the two have to agree, not match some other idea.
        assert read_trn(path) == {
            "s1-u1": ["YES", "NO"],
            "s2-u2": ["NO"],
        }

    def test_an_empty_hypothesis_round_trips_as_no_words(self, tmp_path):
        from BenNevis.utils.scoring import read_trn

        path = tmp_path / "hyp.wrd.trn"
        write_trn([("u1", [])], {"u1": "s1"}, path)
        assert read_trn(path) == {"s1-u1": []}

    def test_it_creates_the_directory(self, tmp_path):
        path = tmp_path / "deep" / "er" / "hyp.wrd.trn"
        write_trn([("u1", ["YES"])], {"u1": "s"}, path)
        assert path.is_file()


class TestEndToEnd:
    """
    A dictionary, an ARPA model and a topology spec in; the right words out.

    The strongest check available without a trained model, and the one that
    says the parts fit together: `BenNevis.lm` builds `L` and `G`,
    `BenNevis.topos` expands `T`, k2 composes and searches, and nothing in
    the path is Kaldi's or OpenFST's.
    """

    @pytest.fixture(scope="class")
    def graph(self):
        pytest.importorskip("k2")
        from BenNevis.decode.pipeline import build_graph

        return build_graph(
            LANG,
            YESNO / "lexicon.txt",
            YESNO / "task.arpabo",
            "ctc",
            sil_prob=0.0,
            position_dependent_phones=False,
        )

    @pytest.fixture(scope="class")
    def tokens(self):
        from BenNevis.topos import build as topo_build
        from BenNevis.topos import spec as topo_spec

        loaded = topo_spec.load("ctc")
        phones = topo_build.read_phone_table(LANG / "phones.txt")
        return {
            token: index
            for index, token in enumerate(topo_build.build_tokens(loaded, phones))
        }

    @pytest.mark.k2
    def test_the_graph_is_not_trivial(self, graph):
        assert graph.num_arcs > 10
        assert graph.shape[0] > 2

    @pytest.mark.k2
    @pytest.mark.parametrize(
        "phones,expected",
        [
            (["Y", "E", "S"], ["YES"]),
            (["N", "O"], ["NO"]),
            (["Y", "E", "S", "N", "O"], ["YES", "NO"]),
            (["N", "O", "Y", "E", "S"], ["NO", "YES"]),
            (["N", "O", "N", "O"], ["NO", "NO"]),
        ],
    )
    def test_peaked_acoustics_decode_to_the_right_words(
        self, graph, tokens, phones, expected
    ):
        import torch

        from BenNevis.decode.decoder import decode

        ids = [tokens[phone] for phone in phones]
        logits = torch.full((1, len(ids), len(tokens)), -20.0)
        for frame, token in enumerate(ids):
            logits[0, frame, token] = 0.0

        out = decode(
            graph,
            logits.log_softmax(-1),
            torch.tensor([len(ids)], dtype=torch.int32),
        )
        idx2word = {
            index: word for word, index in read_symbol_table(LANG / "words.txt").items()
        }
        words = [idx2word[i] for i in out[0] if i in idx2word]
        assert words == expected

    @pytest.mark.k2
    def test_the_graph_input_alphabet_is_the_output_layer(self, graph, tokens):
        """
        Not phones. This is what the topology on the left buys, and getting
        it wrong would mean the decoder searching for labels the network
        never emits.
        """
        assert int(graph.labels.max()) < len(tokens)

    @pytest.mark.k2
    def test_a_lang_dir_without_disambiguation_symbols_is_refused(self, tmp_path):
        """
        Without them the composition cannot be made determinisable, and the
        failure would otherwise appear much later as a graph that decodes
        badly.
        """
        from BenNevis.decode.pipeline import build_graph

        (tmp_path / "words.txt").write_text("<eps> 0\nYES 1\n")
        (tmp_path / "phones.txt").write_text("<eps> 0\nY 1\n")
        with pytest.raises(ValueError, match="no disambiguation symbols"):
            build_graph(
                tmp_path,
                YESNO / "lexicon.txt",
                YESNO / "task.arpabo",
                "ctc",
            )


class TestTierOneStaysTierOne:
    """
    `pipeline`'s `.trn` handling must import without k2.

    This file's tests are tier 1, and they failed on every platform
    because `BenNevis/decode/__init__.py` imported the k2-dependent
    submodules eagerly -- so importing `BenNevis.decode.pipeline` to format
    a transcription pulled in k2. A machine with k2 installed cannot see
    that, which is exactly why it needs a test rather than care.
    """

    def test_importing_pipeline_does_not_import_k2(self):
        """
        Run in a subprocess with k2 made unimportable, which is the only
        way to check this where k2 is installed.
        """
        import subprocess
        import sys

        program = (
            "import sys\n"
            "class Block:\n"
            "    def find_module(self, name, path=None):\n"
            "        if name == 'k2' or name.startswith('k2.'):\n"
            "            return self\n"
            "    def load_module(self, name):\n"
            "        raise ImportError('k2 is blocked for this test')\n"
            "sys.meta_path.insert(0, Block())\n"
            "from BenNevis.decode.pipeline import format_trn\n"
            "assert format_trn([('u', ['A'])], {'u': 's'}) == 'A (s-u)\\n'\n"
            "assert 'k2' not in sys.modules\n"
            "print('ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, (
            f"importing BenNevis.decode.pipeline without k2 failed:\n"
            f"{result.stdout}\n{result.stderr}"
        )
        assert "ok" in result.stdout

    def test_the_package_exports_without_importing(self):
        """
        `__getattr__` has to cover everything `__all__` promises, or a name
        resolves on one import path and not another.
        """
        import BenNevis.decode as package

        for name in package.__all__:
            assert name in package._SOURCES, name
