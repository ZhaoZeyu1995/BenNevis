"""
Tests for the nightly job's log volume.

The log a job serves *while it runs* holds about 2MB, and `wget`'s default
dot progress writes one line per 50KiB -- so mini-librispeech's 1.2GiB of
corpus and language models produces some 25,000 lines of dots and spends
that before the recipe has begun. Run 45202289714 showed it twice, its two
attempts cut off 59 bytes apart at 2,117,576 and 2,117,517 bytes, both
mid-download. Its finished log was 3,009,128 bytes and complete, so the
limit is on the live view and on a job whose runner dies before uploading.

That is worse than noise. A log that stops mid-step looks exactly like a
job that stopped mid-step, and the silence was read here as a symptom of
the lost runner it had nothing to do with -- and for three hours after
that, "still training" could not be told from "hung". So the volume is a
correctness concern about the project's own diagnosis, not housekeeping.

Two things keep it readable. `.github/wgetrc`, passed to the container as
`WGETRC` so that Kaldi's download scripts -- reached through the `local/`
symlink and not ours to edit -- need no change, coarsens the progress
meter to `dot:giga`: a line per 50MiB instead of per 50KiB, so the same
1.2GiB costs about 25 lines and the transfer rate survives. And the whole
recipe output is tee'd to an artefact, so the cap cannot hide what the
recipe did even when something else floods the log.

What these tests pin is the part that is easy to get wrong in the quiet
direction. `quiet = on` would suppress wget's error messages, and
`verbose = off` -- which was the first fix here -- suppresses the rate
along with the dots; either trades one unreadable log for another. And a
tee'd log that nothing uploads is worse than no tee at all, because it
looks like a record and is discarded with the runner.

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


def test_the_wgetrc_coarsens_the_progress_meter():
    """
    One line per 50MiB rather than per 50KiB.

    `dot:giga` is one dot per MiB and fifty to a line, which turns
    mini-librispeech's 1.2GiB from some 25,000 lines into about 25.
    """
    assert settings().get("progress") == "dot:giga", settings()


def test_the_wgetrc_silences_nothing():
    """
    Both of wget's quiet knobs are the wrong fix, for different reasons.

    `quiet = on` drops the error messages, which is what the log is kept
    for. `verbose = off` keeps them but drops the transfer rate, which is
    worth having when openslr is slow enough to cost half an hour -- it
    was the first fix here and the meter is strictly better.
    """
    assert settings().get("quiet") in (None, "off"), settings()
    assert settings().get("verbose") in (None, "on"), settings()


def test_the_wgetrc_sets_nothing_unexplained():
    """
    A startup file applies to every download the recipe makes.

    Anything beyond the settings above changes how the corpus is fetched
    -- timeouts, retries, proxies -- which is a decision about the recipe
    rather than about its log, and belongs in a PR that says so.
    """
    assert set(settings()) <= {"progress", "verbose", "quiet"}, settings()


def tee_target() -> str:
    """
    The file the recipe step tees its output into.

    Returns
    -------
    str
        The path as the step writes it, relative to the working directory.
    """
    lines = [line for line in recipe_step()["run"].splitlines() if "| tee " in line]
    assert len(lines) == 1, f"the recipe step tees {len(lines)} times"
    return lines[0].split("| tee ", 1)[1].split()[0].strip('"')


def test_the_recipe_output_is_kept_whole():
    """
    The job log is capped; the tee'd copy is not.

    Without this, three hours of recipe output was unreadable on run
    45202289714 and "still training" could not be told from "hung".
    """
    target = tee_target()
    assert target.startswith("artefacts/"), target
    # The directory has to exist before the tee opens the file, and the
    # runner creates it in an earlier step -- deliberately, since a
    # container creating it first would own it as root.
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["recipe"]["steps"]
    names = [s.get("name") for s in steps]
    made = [
        n for n, s in zip(names, steps) if "mkdir -p artefacts" in str(s.get("run"))
    ]
    assert made, "nothing creates the artefacts directory"
    assert names.index(made[0]) < names.index("Run the recipe")


def test_the_kept_output_is_uploaded():
    """
    A tee'd log nothing uploads is discarded with the runner.

    It also has to upload on failure, since a run that broke is the one
    whose log is worth reading.
    """
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["recipe"]["steps"]
    uploads = [
        s
        for s in steps
        if str(s.get("uses", "")).startswith("actions/upload-artifact")
        and tee_target() in str(s.get("with", {}).get("path", ""))
    ]
    assert len(uploads) == 1, f"{tee_target()} is uploaded {len(uploads)} times"
    assert str(uploads[0].get("if")) == "always()", uploads[0].get("if")
