"""
Tests for the nightly's graph equivalence gate.

The gate compares the Python `LG` and `TLG` against Kaldi's inside a
recipe run, which is the only place both sides exist: mini-librispeech's
graphs are 138MB and 340MB, so the capture keeps their digests and not
their bytes, and the Python side needs a 3-gram model that is downloaded
rather than committed.

What these tests pin is the part that can drift silently. **The gate
rebuilds the lexicon, so its settings must match the recipe's
`prepare_lang.sh` call** -- exactly the trap already recorded for stage
10, where `--sil_prob` and `--position_dependent_phones` getting out of
step builds a lexicon that is *wrong rather than broken*: the graph
composes, the comparison runs, and it compares the Python graph against a
Kaldi graph built from a different lexicon. Both numbers would look
plausible. So the flags are read out of the workflow and out of the
recipe and checked against each other, rather than against a copy of
either.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nightly.yml"
RECIPE = REPO_ROOT / "egs" / "mini_librispeech" / "asr" / "run.sh"
STEP = "Compare the Python graphs against Kaldi's"


def _steps():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    (job,) = workflow["jobs"].values()
    return job["steps"]


def _step(name):
    for step in _steps():
        if step.get("name") == name:
            return step
    raise AssertionError(f"no step named {name!r}")


class TestTheGateIsWired:
    def test_the_step_exists(self):
        assert _step(STEP)["run"]

    def test_it_runs_only_for_mini_librispeech(self):
        """
        yesno cannot answer the question: there our TLG is 11 states and
        33 arcs against Kaldi's 10 and 29 and the weights agree, because a
        four-word lexicon leaves the second pipeline nothing to compact.
        Running it on yesno would report success that means nothing.
        """
        condition = _step(STEP)["if"]
        assert "egs/mini_librispeech/asr" in condition

    def test_it_runs_even_when_the_recipe_failed(self):
        """
        The graphs are built in stage 4, long before training -- so a run
        that died in stage 5 still has graphs worth comparing, and that is
        the case this gate is cheapest to get from.
        """
        assert _step(STEP)["if"].startswith("always()")

    def test_it_runs_after_the_capture(self):
        names = [step.get("name") for step in _steps()]
        assert names.index("Capture the Kaldi reference") < names.index(STEP)

    def test_it_is_not_blocking_yet_and_says_so(self):
        """
        Deliberately `continue-on-error` until the cost of
        `remove_epsilon` on a 43M-arc TLG is measured. **A check that
        cannot fail the build is a report, not a gate** -- so when it
        becomes blocking, this test changes with it, which is the
        reminder.
        """
        step = _step(STEP)
        assert step.get("continue-on-error") is True, (
            "if the gate is now blocking, good -- update this test and the "
            "comment above the step"
        )


@pytest.fixture(scope="module")
def run_sh():
    """The recipe, so the gate's flags are checked against the source."""
    return RECIPE.read_text()


class TestTheSettingsMatchTheRecipe:
    """
    The gate rebuilds the lexicon, so these must agree with the recipe or
    it compares against a lexicon Kaldi never built.
    """

    def test_the_silence_probability_matches(self, run_sh):
        gate = _step(STEP)["run"]
        match = re.search(r"--sil-prob\s+([0-9.]+)", gate)
        assert match, "the gate passes no --sil-prob"
        recipe = re.search(r"--sil_prob\s+([0-9.]+)", run_sh)
        assert recipe, "stage 10 passes no --sil_prob"
        assert match.group(1) == recipe.group(1)

    def test_position_dependence_matches(self, run_sh):
        gate = _step(STEP)["run"]
        assert ("--position-dependent-phones" in gate) == (
            "--position_dependent_phones" in run_sh
        )

    def test_the_graph_directory_matches(self, run_sh):
        """
        Stage 10 passes the *graph* directory where a language directory
        is expected, because `prepare_topo.sh` writes `tokens.txt` and
        `k2/` there. The gate has to do the same or it reads the wrong
        token table.
        """
        gate = _step(STEP)["run"]
        match = re.search(r"--graph-dir\s+(\S+)", gate)
        assert match, "the gate passes no --graph-dir"
        topo = re.search(r"--topo\s+(\S+)", gate)
        assert topo, "the gate passes no --topo"
        # The recipe writes the directory with the topology interpolated,
        # so the comparison substitutes rather than matching a literal --
        # which is what a first version of this test got wrong.
        template = match.group(1).replace(f"_{topo.group(1)}", "_${topo}")
        assert template in run_sh, (
            f"{match.group(1)} does not correspond to any graph directory "
            f"the recipe builds (looked for {template})"
        )

    def test_the_lexicon_and_model_paths_are_the_recipe_s(self, run_sh):
        gate = _step(STEP)["run"]
        for flag in ("--lexicon", "--arpa"):
            match = re.search(rf"{flag}\s+(\S+)", gate)
            assert match, f"the gate passes no {flag}"
            assert (
                match.group(1) in run_sh
            ), f"{flag} {match.group(1)} is not what the recipe decodes with"


class TestItKeepsWhatItMeasured:
    def test_the_log_lands_under_artefacts(self):
        assert "artefacts/graph-comparison.log" in _step(STEP)["run"]

    def test_the_upload_collects_it(self):
        step = _step("Upload the scores and logs")
        assert "artefacts/graph-comparison.log" in step["with"]["path"]
        assert step["if"] == "always()"
