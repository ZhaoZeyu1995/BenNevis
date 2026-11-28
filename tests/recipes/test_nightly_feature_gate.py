"""
Tests for the nightly's feature equivalence gate.

**This gate exists because the committed reference stopped being able to
answer the question.** `tests/golden/reference/*/` holds `raw/`,
`features/` and `cmvn/` -- stage 2's input, output and statistics -- and
the recipes' stage 2 is Python from the change that added this file. So a
fresh capture would make `tests/utils/test_cmvn.py`'s equivalence class
compare the code with itself, which is this repository's most-repeated
defect; the manifest records `features_from` to make that loud, and this
step is where the comparison moved to.

It runs Kaldi's own `compute-fbank-feats`, `compute-cmvn-stats` and
`apply-cmvn` into a scratch directory, on the same `wav.scp` and the same
`conf/fbank.conf`, and compares four things. What these tests pin is the
part that can drift silently: that the step is blocking, that the pipe to
`tee` cannot swallow its status -- the exact defect the graph gate shipped
with -- and that it runs even when the recipe died later on, because
features are stage 2 and a failed run still has them.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nightly.yml"
SCRIPT = REPO_ROOT / "tests" / "reference" / "compare_features.py"
STEP = "Compare the Python features against Kaldi's"

# Verified to import with torch, numpy, k2 and kaldiio all blocked, so
# this stays a tier 1 module: the script reaches for them lazily.
sys.path.insert(0, str(REPO_ROOT / "tests" / "reference"))
import compare_features  # noqa: E402


def steps():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    (job,) = workflow["jobs"].values()
    return job["steps"]


def step():
    for candidate in steps():
        if candidate.get("name") == STEP:
            return candidate
    raise AssertionError(f"no step named {STEP!r}")


class TestTheGateIsWired:
    def test_the_step_exists(self):
        assert step()

    def test_it_runs_on_both_recipes(self):
        """
        Unlike the graph gate, which is mini-librispeech only because
        yesno's graphs are committed. Both recipes compute features, and
        yesno is the one that runs on a schedule -- so this is the gate
        that actually fires without a dispatch.
        """
        condition = str(step().get("if", ""))
        assert "RECIPE" not in condition, condition

    def test_it_runs_even_when_the_recipe_failed(self):
        """
        Features are stage 2. A run that died in training still has them,
        and this is the cheapest way to get the reading.
        """
        assert "always()" in str(step().get("if", ""))

    def test_it_is_blocking(self):
        """
        A gate that cannot fail is the same as no gate. The graph gate
        spent three runs reporting success through real defects because of
        `continue-on-error`.
        """
        assert "continue-on-error" not in step()

    def test_the_pipe_to_tee_cannot_swallow_a_failure(self):
        """
        The command ends in `| tee`, so without `pipefail` the pipeline's
        status is tee's and the gate could not fail however loudly the
        script complained. Removing `continue-on-error` alone would not
        have been enough -- which is exactly what happened to the graph
        gate.
        """
        run = step()["run"]
        assert "| tee" in run
        assert "set -eo pipefail" in run
        assert run.index("set -eo pipefail") < run.index("| tee")

    def test_it_runs_the_comparison_script(self):
        assert "tests/reference/compare_features.py" in step()["run"]
        assert SCRIPT.is_file()

    def test_it_does_not_name_the_data_sets(self):
        """
        yesno's sets are `train_yesno` and `test_yesno` where
        mini-librispeech's are `train_clean_5` and `dev_clean_2`. A
        workflow holding both lists is a list that drifts from the
        recipes, so the script discovers them.
        """
        run = step()["run"]
        for name in ("train_yesno", "test_yesno", "train_clean_5", "dev_clean_2"):
            assert name not in run, f"{name} is named in the workflow"

    def test_it_passes_the_recipe(self):
        assert '--recipe "${RECIPE}"' in step()["run"]


class TestItKeepsWhatItMeasured:
    def test_the_log_lands_under_artefacts(self):
        assert "artefacts/feature-comparison.log" in step()["run"]

    def test_the_upload_collects_it(self):
        """
        The log is the measurement. A gate whose numbers are only in a
        job log loses them to log retention, and the comparison is worth
        keeping for ninety days.
        """
        uploads = [
            candidate
            for candidate in steps()
            if "upload-artifact" in str(candidate.get("uses", ""))
            and "artefacts" in str(candidate.get("with", {}).get("path", ""))
        ]
        assert uploads, "no upload collects artefacts/"


class TestTheScriptRefusesToMeasureNothing:
    def test_discovery_finds_a_prepared_data_directory(self, tmp_path):

        recipe = tmp_path / "recipe"
        prepared = recipe / "data" / "train_x"
        prepared.mkdir(parents=True)
        for name in (
            "wav.scp",
            "spk2utt",
            "utt2spk",
            "feats.scp",
            "cmvn.scp",
            "feats.cmvn.scp",
        ):
            (prepared / name).write_text("x y\n", "utf-8")
        # The kinds of directory that must not be mistaken for a data set.
        for other in ("lang_nosp", "local"):
            (recipe / "data" / other).mkdir(parents=True)

        assert compare_features.discover(recipe) == [Path("data/train_x")]

    def test_it_fails_rather_than_passing_over_no_data(self, tmp_path, monkeypatch):
        """
        A comparison over nothing passes, which is how the `G.fst` gate
        in #41 came to be a test that could not fail.
        """

        recipe = tmp_path / "recipe"
        (recipe / "data").mkdir(parents=True)
        # Pretend Kaldi is present, so the failure under test is the
        # empty discovery rather than the missing binaries.
        monkeypatch.setattr(compare_features, "find_tool", lambda name: "/bin/true")

        assert compare_features.main(["--recipe", str(recipe)]) == 1

    def test_a_missing_kaldi_is_refused_rather_than_skipped(
        self, tmp_path, monkeypatch
    ):
        """
        The comparison needs Kaldi by definition. Carrying on without it
        would report success having compared one side with itself -- the
        same failure as the first two captures, which contained no FSTs
        at all and said nothing about it.
        """

        monkeypatch.setattr(compare_features, "find_tool", lambda name: None)
        assert compare_features.main(["--recipe", str(tmp_path)]) == 1


class TestItDrivesKaldiTheWayMakeFbankDoes:
    """
    The argument lists, because getting one wrong cost a whole nightly.

    `compute-fbank-feats` has **no `--compress` option** -- compression is
    `copy-feats`' job, which is why `make_fbank.sh` pipes one into the
    other:

        compute-fbank-feats --config=... scp:... ark:- \\
          | copy-feats --compress=true ark:- ark,scp:...

    The first version of this harness put `--compress=true` on the
    extractor, from assumption rather than from the script. It exits 255
    with "Invalid option --compress=true", and because this gate is
    `always()` and blocking it failed run 46155782177 -- a nightly whose
    recipe had succeeded and whose language-directory gate had passed.
    **A new gate's first few runs are testing the gate**, for the third
    time in this repository.

    Checked by recording the argument lists rather than by running
    Kaldi, so it runs in tier 1 where the mistake was made rather than
    only in the nightly that pays for it.
    """

    @staticmethod
    def record(monkeypatch, tmp_path):
        """
        Run `kaldi_stage2` with every subprocess replaced, returning the
        argument lists it would have run.
        """
        import subprocess as real_subprocess

        calls = []

        class Pipe:
            """Stands in for a `Popen`, recording its argv."""

            def __init__(self, argv, **kwargs):
                calls.append(list(argv))
                self.returncode = 0
                self.stdout = kwargs.get("stdout") and _Closeable()

            def communicate(self):
                return b"", b""

        class _Closeable:
            def close(self):
                return None

        monkeypatch.setattr(compare_features.subprocess, "Popen", Pipe)
        monkeypatch.setattr(
            compare_features, "run", lambda argv, cwd: calls.append(list(argv))
        )
        monkeypatch.setattr(
            compare_features, "find_tool", lambda name: f"/kaldi/{name}"
        )
        monkeypatch.setattr(
            compare_features.subprocess,
            "CompletedProcess",
            real_subprocess.CompletedProcess,
        )

        compare_features.kaldi_stage2(
            tmp_path / "recipe",
            Path("data/train"),
            tmp_path / "scratch",
            tmp_path / "conf" / "fbank.conf",
        )
        return calls

    @staticmethod
    def find(calls, tool):
        for argv in calls:
            if argv and argv[0].endswith(tool):
                return argv
        raise AssertionError(f"{tool} was never run; got {[a[0] for a in calls]}")

    def test_the_extractor_is_not_given_compress(self, monkeypatch, tmp_path):
        """
        The exact defect. `compute-fbank-feats` rejects the option and
        exits 255.
        """
        calls = self.record(monkeypatch, tmp_path)
        argv = self.find(calls, "compute-fbank-feats")
        assert not [a for a in argv if "compress" in a], argv

    def test_the_compression_is_copy_feats_job(self, monkeypatch, tmp_path):
        calls = self.record(monkeypatch, tmp_path)
        argv = self.find(calls, "copy-feats")
        assert "--compress=true" in argv, argv

    def test_the_two_are_piped(self, monkeypatch, tmp_path):
        """
        `ark:-` out of the extractor and into the compressor, which is
        what `make_fbank.sh` does. Writing an archive to disc in between
        would work and is not what is being reproduced.
        """
        calls = self.record(monkeypatch, tmp_path)
        assert "ark:-" in self.find(calls, "compute-fbank-feats")
        assert "ark:-" in self.find(calls, "copy-feats")

    def test_the_extractor_reads_the_recipes_configuration(self, monkeypatch, tmp_path):
        """
        The same `conf/fbank.conf` our own extractor reads, so the two
        cannot be given different settings -- which is the whole premise
        of the comparison.
        """
        calls = self.record(monkeypatch, tmp_path)
        argv = self.find(calls, "compute-fbank-feats")
        assert any(a.startswith("--config=") and "fbank.conf" in a for a in argv), argv

    def test_the_statistics_are_per_speaker(self, monkeypatch, tmp_path):
        """
        `--spk2utt`, as `steps/compute_cmvn_stats.sh` passes. Per
        utterance would be a different normaliser and the comparison
        would fail for the harness's reason rather than the code's.
        """
        calls = self.record(monkeypatch, tmp_path)
        argv = self.find(calls, "compute-cmvn-stats")
        assert any(a.startswith("--spk2utt=") for a in argv), argv

    def test_the_normalisation_matches_the_recipes(self, monkeypatch, tmp_path):
        """
        `--norm-vars=true`, which is what `run/apply_cmvn.sh` has always
        passed and so what every WER in the ledger was measured under.
        Kaldi's own default is false.
        """
        calls = self.record(monkeypatch, tmp_path)
        argv = self.find(calls, "apply-cmvn")
        assert "--norm-vars=true" in argv, argv
        assert any(a.startswith("--utt2spk=") for a in argv), argv
