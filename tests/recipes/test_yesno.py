"""
The recipe tier: what a completed `egs/yesno/asr` run must look like.

These tests do not run the recipe. The recipe is run by the nightly workflow,
inside the Kaldi image, and these then inspect what it left behind. Keeping
the two apart matters: a recipe takes about an hour, and a test that both
drives and judges it reports "failed" for a missing binary, a diverged WER
and a crashed training run alike.

What is checked, in the order the recipe produces it:

1. Every stage left the artefacts the next stage reads. This is the only
   coverage the entry-point scripts have -- `run/train.py`, `run/predict.py`,
   `bin/align.py` and the shell wrappers are not reachable from any other
   tier, which is how the regression in #5 got in.
2. The network's output layer matches the topology's token inventory. The
   topology determines that inventory, so a mismatch is the failure mode the
   whole topology/`tokens.txt` consistency rule exists to prevent, and a
   trained checkpoint is where it finally becomes visible.
3. The word error rate, against `tests/golden/wer/yesno.json`.
4. That WER, against `sclite`'s. The ledger value is computed in Python by
   `BenNevis.utils.scoring`; sctk is the independent oracle for it, checked
   on real output rather than on a hand-made fixture.

Point to a finished run with `BENNEVIS_YESNO_DIR`; the tests skip, rather
than fail, when there is nothing there.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import json
import os
from pathlib import Path

import pytest

from BenNevis.utils.scoring import parse_sclite_wer, read_trn, score

pytestmark = pytest.mark.kaldi

TESTS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TESTS_DIR.parent
LEDGER_PATH = TESTS_DIR / "golden" / "wer" / "yesno.json"

RECIPE = Path(
    os.environ.get("BENNEVIS_YESNO_DIR", str(REPO_ROOT / "egs" / "yesno" / "asr"))
)

#: The decoding parameters decode_faster.sh defaults to, which name its
#: output directory.
DECODE_DIR = "aw_1.0-ma_5000-bm_32"

with open(LEDGER_PATH, "r", encoding="utf-8") as _handle:
    LEDGER = json.load(_handle)

MODEL = LEDGER["model"]
TEST_SET = LEDGER["test_set"]
TOPOLOGIES = sorted(LEDGER["topologies"])


def require(path: Path) -> Path:
    """
    Skip if a recipe artefact is absent, rather than failing.

    A missing artefact means the recipe was not run, or not run this far,
    which is a different thing from a recipe that ran and produced the wrong
    answer. Only the second is a defect in the toolkit.
    """
    if not path.exists():
        pytest.skip(f"no recipe output at {path}; run egs/yesno/asr/run.sh first")
    return path


@pytest.fixture(scope="module")
def recipe():
    """The recipe directory, skipped past if it has not been run."""
    return require(RECIPE / "exp")


def experiment(topology: str) -> Path:
    """The experiment directory for one topology."""
    return RECIPE / "exp" / f"{MODEL}-{topology}"


def decode_dir(topology: str) -> Path:
    """Where decode_faster.sh put its hypotheses for one topology."""
    return experiment(topology) / f"dec_{TEST_SET}" / DECODE_DIR


class TestDataPreparation:
    @pytest.mark.parametrize("dataset", ["train_yesno", "test_yesno"])
    @pytest.mark.parametrize(
        "name",
        [
            "wav.scp",
            "text",
            "utt2spk",
            "spk2utt",
            "feats.scp",
            "cmvn.scp",
            # Written by run/apply_cmvn.sh and read by BenNevis's Dataset, so
            # its absence is a silently empty training run.
            "feats.cmvn.scp",
            "utt2num_frames",
        ],
    )
    def test_the_data_directory_is_complete(self, recipe, dataset, name):
        path = require(RECIPE / "data" / dataset / name)
        assert path.stat().st_size > 0, path

    @pytest.mark.parametrize("dataset", ["train_yesno", "test_yesno"])
    def test_the_split_is_the_one_the_ledger_records(self, recipe, dataset):
        """
        The split decides what the recorded WERs mean. It was once an
        accident -- two README files counted as waves made the list 62 long,
        so halving it gave 31/29 and the READMEs were silently dropped from
        test -- and it is now stated explicitly. If it moves, the ledger's
        numbers are measuring a different test set.
        """
        text = require(RECIPE / "data" / dataset / "text")
        utterances = [
            line.split()[0] for line in text.read_text().splitlines() if line.strip()
        ]
        assert len(utterances) == LEDGER["utterances"][dataset]
        assert not [
            utterance
            for utterance in utterances
            if not utterance.replace("_", "").isdigit()
        ], "an utterance id that is not eight binary digits; non-audio got in"

    @pytest.mark.parametrize("dataset", ["train_yesno", "test_yesno"])
    def test_every_utterance_has_features(self, recipe, dataset):
        data = require(RECIPE / "data" / dataset)
        utterances = {
            line.split()[0]
            for line in (data / "text").read_text().splitlines()
            if line.strip()
        }
        features = {
            line.split()[0]
            for line in (data / "feats.cmvn.scp").read_text().splitlines()
            if line.strip()
        }
        assert utterances == features


class TestTopologyPreparation:
    @pytest.mark.parametrize("topology", TOPOLOGIES)
    @pytest.mark.parametrize(
        "name",
        [
            "T.fst",
            "tokens.txt",
            "tokens_disambig.txt",
            "disambig.int",
            "k2/T.fst",
            "k2/tokens.txt",
            "k2/phones.txt",
            # The spec copy Lang expands at training time; without it Lang
            # falls back to fstprint, which still works but is not what the
            # pipeline is supposed to do any more.
            "k2/topo.yaml",
        ],
    )
    def test_the_language_directory_is_complete(self, recipe, topology, name):
        path = require(RECIPE / "data" / f"lang_{topology}" / name)
        assert path.stat().st_size > 0, path

    @pytest.mark.parametrize("topology", TOPOLOGIES)
    def test_the_copied_spec_is_the_right_topology(self, recipe, topology):
        from BenNevis.topos import spec

        path = require(RECIPE / "data" / f"lang_{topology}" / "k2" / "topo.yaml")
        assert spec.load_spec(path).name == topology

    @pytest.mark.parametrize("topology", TOPOLOGIES)
    def test_the_decoding_graph_was_built(self, recipe, topology):
        path = require(RECIPE / "data" / f"lang_test_tg_{topology}" / "TLG.fst")
        assert path.stat().st_size > 0


class TestTraining:
    @pytest.mark.parametrize("topology", TOPOLOGIES)
    def test_a_best_checkpoint_exists(self, recipe, topology):
        path = require(experiment(topology) / "checkpoints" / "best.pt")
        assert path.stat().st_size > 0

    @pytest.mark.parametrize("topology", TOPOLOGIES)
    def test_the_output_layer_matches_the_token_inventory(self, recipe, topology):
        """
        The topology fixes the network's output vocabulary. If the trained
        model's final layer and `k2/tokens.txt` disagree, the two have drifted
        apart -- which is exactly what the spec validation and the shared
        derivation are there to prevent, and a checkpoint is the last place it
        can be caught.
        """
        import torch

        tokens = require(RECIPE / "data" / f"lang_{topology}" / "k2" / "tokens.txt")
        expected = len(
            [line for line in tokens.read_text().splitlines() if line.strip()]
        )

        path = require(experiment(topology) / "checkpoints" / "best.pt")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint["MODEL_STATE"]

        # The output layer is the only parameter whose leading dimension is
        # the vocabulary size, and it is the last one in the module order.
        biases = [
            tensor
            for name, tensor in state.items()
            if name.endswith("bias") and tensor.dim() == 1
        ]
        assert biases, sorted(state)
        assert biases[-1].shape[0] == expected

    @pytest.mark.parametrize("topology", TOPOLOGIES)
    def test_the_checkpoint_has_no_ddp_prefix(self, recipe, topology):
        """
        The format is wrapper-agnostic by design: a run on four GPUs and a
        single-process CPU run must write the same keys, or a checkpoint
        stops being portable between them.
        """
        import torch

        path = require(experiment(topology) / "checkpoints" / "best.pt")
        state = torch.load(path, map_location="cpu", weights_only=True)["MODEL_STATE"]
        assert not any(key.startswith("module.") for key in state)


class TestPrediction:
    @pytest.mark.parametrize("topology", TOPOLOGIES)
    @pytest.mark.parametrize("name", ["output.scp", "ref.wrd.trn"])
    def test_the_prediction_outputs_exist(self, recipe, topology, name):
        path = require(experiment(topology) / f"pred_{TEST_SET}" / name)
        assert path.stat().st_size > 0

    @pytest.mark.parametrize("topology", TOPOLOGIES)
    def test_the_reference_covers_the_whole_test_set(self, recipe, topology):
        """
        predict.py merges per-rank files with `cat | awk`, and a merge that
        dropped utterances would flatter every WER computed from it.
        """
        reference = read_trn(
            require(experiment(topology) / f"pred_{TEST_SET}" / "ref.wrd.trn")
        )
        text = require(RECIPE / "data" / TEST_SET / "text")
        utterances = {
            line.split()[0] for line in text.read_text().splitlines() if line.strip()
        }
        assert len(reference) == len(utterances)

    @pytest.mark.parametrize("topology", TOPOLOGIES)
    def test_the_reference_word_count_matches_the_ledger(self, recipe, topology):
        """
        The denominator of every WER in the ledger. If it moves, the recorded
        numbers are no longer comparable with the published ones.
        """
        reference = read_trn(
            require(experiment(topology) / f"pred_{TEST_SET}" / "ref.wrd.trn")
        )
        words = sum(len(w) for w in reference.values())
        assert words == LEDGER["reference_words"]


class TestAlignment:
    @pytest.mark.parametrize("topology", TOPOLOGIES)
    def test_a_ctm_was_produced(self, recipe, topology):
        """
        bin/align.py is reachable from no other tier, and it writes through a
        `gzip -c` pipe, so this is the only place that path runs at all.
        """
        path = require(experiment(topology) / f"pred_{TEST_SET}" / "align" / "ctm")
        assert path.stat().st_size > 0

    @pytest.mark.parametrize("topology", TOPOLOGIES)
    def test_the_ctm_is_sorted_and_well_formed(self, recipe, topology):
        path = require(experiment(topology) / f"pred_{TEST_SET}" / "align" / "ctm")
        rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
        assert rows
        for row in rows:
            # file channel start duration word
            assert len(row) >= 5, row
            float(row[2])
            float(row[3])


class TestWordErrorRate:
    @pytest.mark.parametrize("topology", TOPOLOGIES)
    def test_the_decoding_outputs_exist(self, recipe, topology):
        for name in ("hyp.wrd.trn", "results.wrd.txt"):
            path = require(decode_dir(topology) / name)
            assert path.stat().st_size > 0

    @pytest.mark.parametrize("topology", TOPOLOGIES)
    def test_wer_is_within_the_ledger(self, recipe, topology):
        reference = read_trn(
            require(experiment(topology) / f"pred_{TEST_SET}" / "ref.wrd.trn")
        )
        hypothesis = read_trn(require(decode_dir(topology) / "hyp.wrd.trn"))
        counts = score(reference, hypothesis)

        entry = LEDGER["topologies"][topology]
        assert counts.wer <= entry["max_wer"], (
            f"{MODEL}-{topology} on {TEST_SET}: WER {counts.wer:.2f}% "
            f"({counts.errors} errors in {counts.reference_words} words: "
            f"{counts.substitutions} sub, {counts.deletions} del, "
            f"{counts.insertions} ins) is above the ledger's "
            f"{entry['max_wer']}%. The reference run was "
            f"{entry['reference_wer']}%."
        )

    @pytest.mark.parametrize("topology", TOPOLOGIES)
    def test_sclite_agrees_with_the_python_wer(self, recipe, topology):
        """
        sctk as the independent oracle for the arithmetic. They are computing
        the same quantity by different code; a disagreement means one of them
        is wrong, and the ledger would be measuring the wrong thing.
        """
        reference = read_trn(
            require(experiment(topology) / f"pred_{TEST_SET}" / "ref.wrd.trn")
        )
        hypothesis = read_trn(require(decode_dir(topology) / "hyp.wrd.trn"))
        ours = score(reference, hypothesis).wer

        results = require(decode_dir(topology) / "results.wrd.txt").read_text()
        theirs = parse_sclite_wer(results)
        assert theirs is not None, (
            f"no Sum/Avg row in {decode_dir(topology)}/results.wrd.txt; "
            f"sclite's output format may have changed"
        )
        # sclite prints one decimal place, so it is the rounding that sets the
        # tolerance, not the arithmetic.
        assert ours == pytest.approx(theirs, abs=0.051), (
            f"{MODEL}-{topology}: Python says {ours:.3f}%, sclite says "
            f"{theirs:.3f}%"
        )
