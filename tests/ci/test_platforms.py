"""
Tests that CI still covers every platform the library claims to support.

Tier 1 stopped running on macOS on 23 November, because the macOS pool
failed to assign a runner four times in one day -- `cancelled`, with
`runner_name: ""`, zero steps and exactly 15m02s of queueing each time --
and every one of those cost a re-run before a green PR could merge. That
was a judgement about allocation, not about coverage: it is only safe
because tiers 1b and 2 still run there, carrying torch and k2.

So what is pinned here is the thing that judgement depends on. Dropping
one tier's macOS job is a trade; dropping the last one would end macOS
coverage altogether, and nothing would say so -- CI would simply go green
on two platforms while CLAUDE.md claimed three. The same holds for
Windows, where `as_posix()` and the `bash` probe are three real tier 1
breakages that no other platform showed.

The matrix shape is read rather than asserted, so putting macOS back into
tier 1 needs no change here.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from pathlib import Path
from typing import Dict, Set

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"

#: The platforms the library supports, and so the ones CI must exercise.
#: Not the recipes, which are Linux and macOS only -- see CLAUDE.md.
SUPPORTED = {"ubuntu-latest", "macos-latest", "windows-latest"}

#: The jobs that run tests, by the id they carry in the workflow.
TEST_JOBS = ("test-tier1", "test-torch", "test-k2")


def jobs() -> Dict[str, dict]:
    """
    The workflow's jobs.

    Returns
    -------
    dict
        Job id to job body.
    """
    return yaml.safe_load(CI.read_text())["jobs"]


def platforms_of(job_id: str) -> Set[str]:
    """
    Every operating system a job runs on.

    Arguments
    ---------
    job_id: str
        The job's id in the workflow.

    Returns
    -------
    set of str
        The runner labels, gathered from both the matrix's `os` axis and
        any `include` entries, since the two are combined at run time and
        reading only one would miss half the matrix.
    """
    matrix = jobs()[job_id]["strategy"]["matrix"]
    # `or []` and not a default: an `include:` key left behind with its
    # last entry removed parses as None, and iterating that raised a
    # TypeError where the point is to report the missing platform.
    found = set(matrix.get("os") or [])
    for entry in matrix.get("include") or []:
        if "os" in entry:
            found.add(entry["os"])
    return found


def test_the_test_jobs_are_all_present():
    """Guards the ids the rest of the file rests on."""
    missing = [j for j in TEST_JOBS if j not in jobs()]
    assert not missing, f"{missing} no longer exist in ci.yml"


@pytest.mark.parametrize("platform", sorted(SUPPORTED))
def test_every_supported_platform_is_tested_somewhere(platform):
    """
    A platform CI stops running is a platform the library stops supporting.

    Tier 1 may drop one, as it did with macOS over runner allocation. The
    last tier covering a platform may not.
    """
    covering = [j for j in TEST_JOBS if platform in platforms_of(j)]
    assert covering, f"no CI job runs on {platform}"


@pytest.mark.parametrize("job_id", TEST_JOBS)
def test_no_job_runs_on_an_unsupported_platform(job_id):
    """A runner label nobody meant to add is a cost with no coverage."""
    extra = platforms_of(job_id) - SUPPORTED
    assert not extra, f"{job_id} runs on {extra}"


def test_linux_runs_every_tier():
    """
    The cheapest platform is the one that must never be narrowed.

    Linux is where the Python sweep lives and where tier 3 runs, so a tier
    that stopped running there would be a tier that effectively stopped
    running.
    """
    for job_id in TEST_JOBS:
        assert "ubuntu-latest" in platforms_of(job_id), job_id
