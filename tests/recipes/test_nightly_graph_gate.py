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

    def test_it_is_blocking(self):
        """
        It was not, for its first four runs, and the condition for
        changing that was one clean verdict on mini-librispeech. Run
        45567166522 gave it: our LG agreed with Kaldi's on all 24
        lexicon-drawn sequences within 0.0009765625 per arc. **A check
        that cannot fail the build is a report, not a gate**, so if this
        ever goes back to `continue-on-error` it should be a decision
        with a reason beside it.
        """
        assert "continue-on-error" not in _step(STEP), (
            "the gate has stopped being able to fail the build; say why "
            "in the step's comment and here"
        )

    def test_the_pipe_to_tee_cannot_swallow_a_failure(self):
        """
        The half of blocking that is easy to miss.

        The command ends in `| tee artefacts/graph-comparison.log`, so
        without `pipefail` the pipeline's status is tee's and a non-zero
        exit from the gate is discarded -- removing `continue-on-error`
        alone would have left a gate that still could not fail. This is
        the same shape as the `G.fst` gate that skipped unconditionally
        and the nightly summary whose `grep` failed a passing run: a
        shell detail deciding whether a check means anything.
        """
        run = _step(STEP)["run"]
        assert "| tee" in run, "if the tee has gone, this test can go too"
        first = run.strip().splitlines()[0]
        assert (
            "pipefail" in first
        ), f"pipefail must be set before the pipeline, not {first!r}"

    def test_both_halves_run(self):
        """
        `--skip-tlg` was taken for as long as the TLG half went through
        `prepare_output_probe`: run 45554380456 showed
        `k2.remove_epsilon` on the 43,263,880-arc TLG stopping the step
        inside a minute with no traceback, which is what a SIGKILL looks
        like. The half is a Dijkstra over (state, position) pairs now,
        which needs no epsilon removal at all, so the flag goes back to
        being an escape.

        The flag still exists, deliberately -- the TLG half is the
        expensive one and may yet have to come off. What this pins is
        that taking it is a decision someone writes a reason for, rather
        than the state the gate quietly sits in.
        """
        assert "--skip-tlg" not in _step(STEP)["run"], (
            "the TLG half is off again; say what it cost here and in the "
            "step's comment"
        )

    def test_the_comments_do_not_contradict_the_step(self):
        """
        They did. #86 made the gate blocking and added a paragraph saying
        so, but left the paragraph above it that said
        "`continue-on-error` deliberately ... **Turn this into a blocking
        gate once the cost is known**" -- in fact the whole comment block
        was there twice, the older copy first. Two comments disagreeing
        about whether a check can fail the build is worse than neither,
        because the stale one is the one a reader meets first and has no
        reason to doubt.

        Only the claims that have a machine-checkable counterpart are
        pinned: a comment promising `continue-on-error` while the step
        has none, and one asking for a change that has been made.
        """
        raw = WORKFLOW.read_text()
        before = raw[: raw.index(f"- name: {STEP}")]
        comments = before[before.rindex("\n\n") :]
        step = _step(STEP)
        if "continue-on-error" not in step:
            assert "continue-on-error` deliberately" not in comments, (
                "the comment says the step cannot fail the build and the "
                "step says otherwise"
            )
        assert "Turn this into a blocking gate" not in comments, (
            "that was done; a comment asking for a change already made "
            "reads as a change still outstanding"
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


class TestTheStepKeepsItsEnvironment:
    """
    The bug that cost the gate its first run.

    The first version wrapped the command in `bash -c '...'` so that
    `path.sh` could be sourced to put `fstprint` on the PATH. Single
    quotes mean the *container's* shell expands `${RECIPE}`, and `RECIPE`
    is a workflow-level `env:` that nothing passed in with `-e`. It became
    `--recipe ""`, every path resolved against `/workspace`, and the
    script died on `data/lang_test_tgsmall_ctc/words.txt` -- after the
    step had already reported success, because `continue-on-error` hides a
    non-zero exit.

    Two assertions, because either alone can be satisfied the wrong way:
    the command must not be wrapped in a single-quoted shell, and any
    workflow variable it uses must either be expanded by the runner or
    passed in explicitly.
    """

    def test_the_command_is_not_wrapped_in_a_quoted_shell(self):
        run = _step(STEP)["run"]
        assert "bash -c '" not in run, (
            "a single-quoted bash -c defers expansion to the container, "
            "where the workflow's env: variables do not exist"
        )

    def test_every_variable_it_uses_reaches_it(self):
        """
        `${RECIPE}` is expanded by the runner here. If a future edit moves
        the command inside a container shell, the variable has to be
        passed with `-e` instead -- this checks one or the other holds.
        """
        run = _step(STEP)["run"]
        import re

        used = set(re.findall(r"\$\{(\w+)\}", run))
        # IMAGE is referenced as "$IMAGE" and is a runner-side variable
        # like the rest; what matters is that nothing is left to a shell
        # inside the container.
        for name in used:
            passed = f"-e {name}" in run
            runner_side = "bash -c" not in run
            assert passed or runner_side, (
                f"${{{name}}} is expanded inside the container but never "
                f"passed in with -e"
            )

    def test_it_does_not_rely_on_path_sh_for_fstprint(self):
        """
        Sourcing `path.sh` is what motivated the shell wrapper. The script
        finds `fstprint` itself, the way `capture.py` does, so the step
        needs no shell at all.
        """
        run = _step(STEP)["run"]
        assert "path.sh" not in run
