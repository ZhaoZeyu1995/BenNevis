"""
Tests for the recipe job's timeout against the platform's own limit.

A hosted GitHub runner kills a job at **360 minutes** whatever the
workflow asks for, so a `timeout-minutes` above that is not a longer
budget -- it is a workflow that no longer states when it expects to be
finished, and the job dies at six hours with GitHub's message rather than
the repository's. Below the measured run time it is the opposite problem:
a slow day reports as a failure.

The ceiling was 300 while mini-librispeech's epochs were believed to take
ten minutes. They take twenty, measured over the twelve epochs of run
45202289714, which was cancelled with only stage 10 outstanding. 350
leaves margin inside the platform limit.

What is pinned is the bracket, not the number: raising 350 to 355 needs no
edit here, and raising it to 400 does.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nightly.yml"

#: What a hosted runner enforces regardless of the workflow, in minutes.
RUNNER_CEILING = 360

#: Four hours of training for twelve epochs, plus graphs and decoding, so
#: a ceiling below this would fail a healthy run.
MEASURED_RUN_MINUTES = 300


def recipe_job() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]["recipe"]


def test_the_job_states_a_timeout():
    """
    Without one the job runs to the platform's six hours.

    A recipe that hangs would then hold a runner for six hours before
    anyone heard about it, and the workflow would say nothing about how
    long it expects to take.
    """
    assert "timeout-minutes" in recipe_job(), "the recipe job has no timeout"


def test_the_timeout_fits_inside_the_runner_ceiling():
    """
    Above 360 the number is a fiction: the runner kills the job anyway.

    Worth asserting rather than trusting, because nothing fails when it
    is wrong -- the job simply dies at six hours with a message that does
    not mention this file.
    """
    assert recipe_job()["timeout-minutes"] <= RUNNER_CEILING


def test_the_timeout_allows_a_measured_run_to_finish():
    """A ceiling under the real run time turns a slow day into a failure."""
    assert recipe_job()["timeout-minutes"] >= MEASURED_RUN_MINUTES
