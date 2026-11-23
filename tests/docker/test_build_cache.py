"""
Tests that the image build writes no cache into the Actions budget.

`cache-to: type=gha,mode=max` was removed on measurement. From the
21 November build, the layers a cache could save -- apt, the venv, and
the two pip installs -- came to 58 seconds between them, while exporting
the cache took 209.9. The Actions cache is also scoped per branch, so a
pull request touching `docker/` exported its own full copy that master
could never read: the two together held 5.22GiB, 88% of everything
cached for this repository, against the 10GB budget the nightly's corpus
caches compete for.

That budget is the reason this is pinned rather than left to judgement.
An eviction does not fail anything -- a corpus cache simply stops
hitting, and the only symptom is a nightly that downloads 1.2GiB again,
which looks like a slow day. So the test is about protecting something
whose loss is silent.

The Dockerfile is single stage, which is what makes the trade so
one-sided: `mode=max` differs from `mode=min` only in intermediate
stages, and there are none, so the cache was exporting the whole image
including the 2.1GiB Kaldi base -- a base that is pinned by digest and
comes from Docker Hub in 67.6 seconds either way.

If this is reinstated, the numbers above are what to beat, and
`tests/recipes/test_nightly_cache.py` has the other side of the budget.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker.yml"


def build_step() -> dict:
    """
    The step that builds the image.

    Returns
    -------
    dict
        The step, as the workflow declares it. Looked up by its action
        rather than by name, so renaming the step cannot make the tests
        below pass by finding nothing.
    """
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["build"]["steps"]
    # Split on the version, not a prefix match: `startswith` was the first
    # try and `docker/build-push-action-renamed@v6` satisfied it, so a
    # mutation that renamed the action away passed all three tests.
    matches = [
        s
        for s in steps
        if str(s.get("uses", "")).split("@", 1)[0] == "docker/build-push-action"
    ]
    assert len(matches) == 1, f"{len(matches)} steps build the image"
    return matches[0]


def test_the_image_is_still_built():
    """Guards the lookup the other tests rest on."""
    assert build_step()["with"]["file"] == "docker/Dockerfile"


def test_the_build_exports_no_cache():
    """
    `cache-to` is what spends the budget, so there is none.

    58 seconds of cached layers is not worth 5.22GiB of a 10GB budget
    shared with the corpus caches, nor 209.9 seconds of export.
    """
    assert "cache-to" not in build_step()["with"], build_step()["with"]


def test_the_build_names_no_gha_cache():
    """
    Neither direction, and nowhere in the file.

    `cache-from: type=gha` alone would store nothing, but it would also
    read a cache that nothing writes -- which is not a bound, just a
    slower way to miss.
    """
    assert "cache-from" not in build_step()["with"], build_step()["with"]
    # Comments stripped, because the comment that records why the cache
    # went says `type=gha,mode=max` and is the most useful line in the
    # file. What is forbidden is configuring it, not naming it.
    active = [
        line
        for line in WORKFLOW.read_text().splitlines()
        if not line.lstrip().startswith("#")
    ]
    assert "type=gha" not in "\n".join(active)
