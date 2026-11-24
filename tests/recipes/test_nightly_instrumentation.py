"""
Tests for the host facts and memory samples the nightly records.

Two dispatches of the same commit trained mini-librispeech at 20m09s and
34m30s an epoch, and the slower pair was then SIGKILLed in epoch 4 --
exit `-9` out of `torchrun`, 213 and 130 minutes in, both well inside the
350-minute ceiling. The second of those reproduced the first exactly:
the same 894 steps an epoch and the same losses to five decimal places,
so the work was identical and the kill was not a one-off.

Everything visible from outside the runner was ruled out. The image
differed only in `charset_normalizer` 3.5.2 to 3.5.3 and `packaging` 26.3
to 26.4, with torch, k2 and numpy the same build and every apt version
identical; `OMP_NUM_THREADS=1` is set for all three runs; the corpus came
from the same cache. That leaves the host, and `ubuntu-latest` promises
nothing about how much memory or how many cores it allocates -- the three
runs landed on three different runners.

So this is instrumentation rather than a fix, and the thing it is for is
a question that cannot be answered after the fact: was the runner short
of memory from the start, or did `python3` grow into it? `MemAvailable`
is sampled on the **host**, because the OOM killer decides on the host's
free memory rather than on anything the container reports, and the three
largest resident processes go beside it so that a growing trainer can be
told from a small runner.

What these tests pin is that the sampler **runs** and that what it writes
is **kept**. Both fail silently otherwise: it is backgrounded with its
output redirected to a file, so a syntax error in it would leave an empty
log and a green step, and a log nothing uploads is discarded with the
runner. The sampler is extracted from the workflow and executed, the way
`tests/recipes/test_decode_faster.py` runs `decode_faster.sh`'s awk --
a reimplementation in the test would pass while the shipped one failed.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nightly.yml"


def _steps():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    (job,) = workflow["jobs"].values()
    return job["steps"]


def _step(name):
    for step in _steps():
        if step.get("name") == name:
            return step
    raise AssertionError(f"no step named {name!r}")


def _sampler_source():
    """The `sample_memory` definition exactly as the workflow ships it."""
    run = _step("Run the recipe")["run"]
    match = re.search(r"^ *sample_memory\(\) \{.*?^ *\}$", run, re.M | re.S)
    assert match, "the recipe step defines no sample_memory function"
    return match.group(0)


class TestItRecordsTheHost:
    """The facts `ubuntu-latest` does not promise."""

    def test_the_host_is_recorded_before_the_recipe_runs(self):
        names = [step.get("name") for step in _steps()]
        assert "Record what the runner is" in names
        assert names.index("Record what the runner is") < names.index(
            "Run the recipe"
        ), "the host must be recorded before the recipe can exhaust it"

    @pytest.mark.parametrize("fact", ["nproc", "MemTotal", "MemAvailable"])
    def test_it_records_the_facts_that_explain_a_kill(self, fact):
        assert fact in _step("Record what the runner is")["run"]

    def test_it_lands_under_artefacts(self):
        assert "artefacts/host.log" in _step("Record what the runner is")["run"]


class TestItSamplesMemory:
    """A SIGKILL is only readable with free memory sampled beside it."""

    def test_the_sampler_writes_under_artefacts(self):
        assert "artefacts/memory.log" in _step("Run the recipe")["run"]

    def test_the_sampler_is_stopped_when_the_step_ends(self):
        run = _step("Run the recipe")["run"]
        assert "trap " in run and "kill" in run, (
            "an unkilled background loop keeps the step alive after the "
            "recipe has finished"
        )

    def test_it_samples_the_host_rather_than_the_container(self):
        sampler = _sampler_source()
        assert "/proc/meminfo" in sampler
        docker = _step("Run the recipe")["run"]
        assert "sample_memory" not in docker.split("docker run")[-1], (
            "the sampler must not be inside the container, whose view of "
            "free memory is not what the OOM killer acts on"
        )

    @pytest.mark.skipif(
        sys.platform != "linux",
        reason="needs /proc/meminfo and GNU ps, so it runs on the Linux tiers",
    )
    def test_the_shipped_sampler_really_emits_a_sample(self):
        """Run the workflow's own function; an empty log is the failure."""
        # The same flags the step sets, because they are what makes a
        # `head -3` in here fatal: SIGPIPE on `ps`, 141 out of the
        # pipeline, and `errexit` ending the loop after one sample.
        script = "set -euo pipefail\n" f"{_sampler_source()}\n" "sample_memory\n"
        try:
            subprocess.run(
                ["bash", "-c", script],
                capture_output=True,
                text=True,
                timeout=6,
                env={"PATH": os.environ["PATH"], "SAMPLE_INTERVAL": "1"},
            )
        except subprocess.TimeoutExpired as expired:
            proc = expired
        else:
            raise AssertionError("the sampler returned; it should loop")
        out = (proc.stdout or b"").decode()
        lines = out.splitlines()
        assert len(lines) >= 2, (
            f"sampler stopped after {len(lines)} sample(s); a loop that "
            "dies silently is the failure mode, stderr "
            f"{(proc.stderr or b'').decode()!r}"
        )
        first = lines[0]
        assert re.fullmatch(
            r"\d{2}:\d{2}:\d{2} avail=\d+MiB( \S+=\d+MiB)+ ?",
            first,
        ), (
            f"sampler wrote {first!r}, stderr " f"{(proc.stderr or b'').decode()!r}"
        )


class TestItKeepsWhatItMeasured:
    """A log nothing uploads is discarded with the runner."""

    @pytest.mark.parametrize("log", ["host.log", "memory.log"])
    def test_the_upload_collects_it(self, log):
        step = _step("Upload the scores and logs")
        assert f"artefacts/{log}" in step["with"]["path"]
        assert step["if"] == "always()", (
            "a run that was killed is exactly the run whose numbers are "
            "wanted, so the upload cannot be conditional on success"
        )
