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

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nightly.yml"
SCRIPT = REPO_ROOT / "tests" / "reference" / "compare_features.py"
STEP = "Compare the Python features against Kaldi's"


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
        import sys

        sys.path.insert(0, str(REPO_ROOT / "tests" / "reference"))
        import compare_features

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
        import sys

        sys.path.insert(0, str(REPO_ROOT / "tests" / "reference"))
        import compare_features

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
        import sys

        sys.path.insert(0, str(REPO_ROOT / "tests" / "reference"))
        import compare_features

        monkeypatch.setattr(compare_features, "find_tool", lambda name: None)
        assert compare_features.main(["--recipe", str(tmp_path)]) == 1
