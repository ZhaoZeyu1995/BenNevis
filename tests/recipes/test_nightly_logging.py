"""
Tests for the nightly job's log volume.

A job log is capped at about 2MB. `wget`'s default dot progress writes one
line per 50KB, so mini-librispeech's 1.2GiB of corpus and language models
produces some 25,000 lines of dots and spends the whole budget before the
recipe has begun -- twice, in run 45202289714, whose two attempts were cut
off 59 bytes apart at 2,117,576 and 2,117,517 bytes, both mid-download.

That is worse than noise. A log that stops mid-step looks exactly like a
job that stopped mid-step, and the silence was read here as a symptom of
the lost runner it had nothing to do with. So the volume is a correctness
concern about the project's own diagnosis, not housekeeping.

`.github/wgetrc` is the fix, passed to the container as `WGETRC` so that
Kaldi's download scripts -- which are reached through the `local/` symlink
and are not ours to edit -- need no change. What these tests pin is the
part that is easy to get wrong in the quiet direction: `quiet = on` would
suppress wget's error messages too, which trades one unreadable log for
another.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nightly.yml"
WGETRC = REPO_ROOT / ".github" / "wgetrc"

#: Where the workflow mounts the working tree inside the container.
MOUNT = "/workspace"


def recipe_step() -> dict:
    """
    The step that runs the recipe, and so the one that downloads.

    Returns
    -------
    dict
        The step, as the workflow declares it.
    """
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["recipe"]["steps"]
    matches = [s for s in steps if s.get("name") == "Run the recipe"]
    assert len(matches) == 1, f"{len(matches)} steps named 'Run the recipe'"
    return matches[0]


def settings() -> dict:
    """
    The wgetrc, parsed.

    Returns
    -------
    dict
        One entry per `key = value` line, keys lowercased and stripped,
        comments and blank lines dropped.
    """
    parsed = {}
    for line in WGETRC.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        key, _, value = line.partition("=")
        parsed[key.strip().lower()] = value.strip().lower()
    return parsed


def test_the_recipe_step_passes_wgetrc():
    """
    Without this the dots come back, and nothing fails to say so.

    The path is checked against the mount rather than against itself: the
    workflow names a path inside the container, so a file moved in the
    repository leaves `wget` reading a startup file that is not there --
    which it ignores silently.
    """
    run = recipe_step()["run"]
    # The mount, not assumed: wget errors out when WGETRC names a file
    # that is not there, so a path that is right in the repository and
    # wrong in the container would break every download in stage 0.
    assert f'-v "$PWD:{MOUNT}"' in run, f"the tree is not mounted at {MOUNT}"
    passed = [line.strip() for line in run.splitlines() if "WGETRC=" in line]
    assert len(passed) == 1, f"WGETRC appears {len(passed)} times in the step"
    container_path = passed[0].split("WGETRC=", 1)[1].split()[0].rstrip("\\").strip()
    assert container_path.startswith(f"{MOUNT}/"), container_path
    in_repo = REPO_ROOT / Path(container_path).relative_to(MOUNT)
    assert in_repo.is_file(), f"{container_path} is not a file in the tree"
    assert in_repo == WGETRC, f"{in_repo} is not {WGETRC}"


def test_the_wgetrc_silences_the_progress_output():
    """`verbose = off` is wget's -nv: no dots, errors kept."""
    assert settings().get("verbose") == "off", settings()


def test_the_wgetrc_does_not_silence_the_errors():
    """
    `quiet` would hide what the log is kept for.

    The point of trimming the log is to be able to read a failure in it, so
    suppressing wget's error messages would defeat the change that
    introduced this file.
    """
    assert settings().get("quiet") in (None, "off"), settings()


def test_the_wgetrc_sets_nothing_unexplained():
    """
    A startup file applies to every download the recipe makes.

    Anything beyond the two settings above changes how the corpus is
    fetched -- timeouts, retries, proxies -- which is a decision about the
    recipe rather than about its log, and belongs in a PR that says so.
    """
    assert set(settings()) <= {"verbose", "quiet"}, settings()
