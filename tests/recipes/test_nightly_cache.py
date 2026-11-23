"""
Tests for the corpus caches in `.github/workflows/nightly.yml`.

A cache is silent when it is wrong. One keyed to the wrong recipe, or
pointed at a directory the recipe does not write, simply never hits: every
run downloads the corpus exactly as it did before and nothing reports a
thing. The only symptom is a bill in minutes, which is why these are
checked here rather than observed.

What is pinned is the pairing rather than the contents. `actions/cache` is
`post-if: success()`, so mini-librispeech saves through the split
`restore`/`save` pair to survive a failed recipe -- and a split pair is two
copies of a path list and a key that must agree, or the job restores one
set of files and stores another. That is the drift this file exists to
catch.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from pathlib import Path
from typing import Dict, List

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nightly.yml"


def load_workflow() -> Dict:
    """
    Read the nightly workflow.

    Returns
    -------
    dict
        The parsed workflow. Note YAML 1.1 reads a bare `on` as the boolean
        True, so the trigger block is not under the string key.
    """
    return yaml.safe_load(WORKFLOW.read_text())


def recipe_options() -> List[str]:
    """
    The recipes the workflow offers to a dispatch.

    Returns
    -------
    list of str
        Paths under `egs/`, as the `recipe` input lists them.
    """
    workflow = load_workflow()
    triggers = workflow.get("on", workflow.get(True))
    return list(triggers["workflow_dispatch"]["inputs"]["recipe"]["options"])


def cache_steps() -> List[Dict]:
    """
    Every step of the recipe job that uses the cache action.

    Returns
    -------
    list of dict
        The steps, in the order the workflow runs them.
    """
    steps = load_workflow()["jobs"]["recipe"]["steps"]
    return [s for s in steps if str(s.get("uses", "")).startswith("actions/cache")]


def paths_of(step: Dict) -> List[str]:
    """
    The included paths of a cache step, with exclusions dropped.

    Arguments
    ---------
    step: dict
        A step using one of the cache actions.

    Returns
    -------
    list of str
        One path per line of the step's `path`, without the `!` entries.
    """
    lines = [line.strip() for line in str(step["with"]["path"]).splitlines()]
    return [line for line in lines if line and not line.startswith("!")]


def recipe_of(step: Dict) -> str:
    """
    The recipe a cache step is conditioned on.

    Arguments
    ---------
    step: dict
        A step using one of the cache actions.

    Returns
    -------
    str
        The recipe path named in the step's `if`.
    """
    condition = str(step.get("if", ""))
    matches = [r for r in recipe_options() if f"'{r}'" in condition]
    assert len(matches) == 1, f"{step.get('name')!r} names {matches} recipes"
    return matches[0]


def test_the_workflow_parses():
    assert WORKFLOW.is_file()
    assert cache_steps(), "no cache steps found, so nothing below is checked"


@pytest.mark.parametrize("recipe", recipe_options())
def test_every_recipe_has_its_corpus_cached(recipe):
    """A recipe added without a cache downloads its corpus every run."""
    restores = [
        s
        for s in cache_steps()
        if not s["uses"].endswith("/save@v4") and recipe_of(s) == recipe
    ]
    assert len(restores) == 1, f"{recipe} has {len(restores)} restore steps"


@pytest.mark.parametrize("step", cache_steps(), ids=lambda s: s["name"])
def test_a_cached_path_belongs_to_its_recipe(step):
    """
    The guard and the path have to name the same recipe.

    A cache restored for yesno into mini-librispeech's tree, or the other
    way about, never hits and never complains.
    """
    recipe = recipe_of(step)
    for path in paths_of(step):
        assert path.startswith(f"{recipe}/"), f"{path} is not under {recipe}"
        assert (REPO_ROOT / recipe).is_dir(), f"{recipe} is not in the tree"


@pytest.mark.parametrize("step", cache_steps(), ids=lambda s: s["name"])
def test_a_cached_directory_is_one_the_recipe_writes(step):
    """
    The directory's name must appear in the recipe that fills it.

    yesno untars into `waves_yesno` and mini-librispeech into `corpus`,
    both written literally in `run.sh`, so a renamed or mistyped directory
    is caught here instead of showing up as a cache that never hits.
    """
    recipe = recipe_of(step)
    run_sh = (REPO_ROOT / recipe / "run.sh").read_text()
    for path in paths_of(step):
        directory = Path(path).relative_to(recipe).parts[0]
        assert directory in run_sh, f"{recipe}/run.sh never mentions {directory}"


def test_restore_and_save_agree():
    """
    A split pair must restore and store the same thing.

    Two copies of a path list and a key is the cost of saving on failure,
    and the two drifting apart would leave the job restoring one set of
    files and caching another -- which looks like a cache that keeps
    missing.
    """
    saves = [s for s in cache_steps() if s["uses"].endswith("/save@v4")]
    assert saves, "no save steps, so no pair to check"
    for save in saves:
        restores = [
            s
            for s in cache_steps()
            if s["uses"].endswith("/restore@v4") and recipe_of(s) == recipe_of(save)
        ]
        assert len(restores) == 1, f"{recipe_of(save)} has {len(restores)} restores"
        restore = restores[0]
        assert save["with"]["key"] == restore["with"]["key"]
        assert save["with"]["path"] == restore["with"]["path"]
        assert restore.get("id"), "a restore in a pair needs an id for cache-hit"
        assert restore["id"] in str(save["if"]), "the save ignores the restore's hit"


def test_a_split_pair_saves_on_failure():
    """
    The whole point of the split is `always()`.

    `actions/cache` runs its save with `post-if: success()`, so a recipe
    that fails after the download saves nothing and the retry pays for the
    corpus again. Without `always()` here the pair is more code than the
    combined action for no gain at all.
    """
    saves = [s for s in cache_steps() if s["uses"].endswith("/save@v4")]
    assert saves
    for save in saves:
        assert "always()" in str(save.get("if", "")), f"{save['name']} skips on failure"
        assert save.get("continue-on-error") is True, (
            f"{save['name']} can fail the job: with always() it runs after an "
            "earlier failure, and cache/save rejects a path that is not there"
        )
