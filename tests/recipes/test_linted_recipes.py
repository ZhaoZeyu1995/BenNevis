"""
Every Python file in a recipe is linted.

Tier 1: it reads the CI workflow and the pre-commit config as YAML.

**A recipe can hold Python now**, since `local/prepare_data.py` replaced
yesno's shell driver and three Perl scripts, and recipe code has no claim
to a lower standard than library code. But `egs/` cannot simply be handed
to the linters, and the reason is symlinks: a recipe directory holds
`run -> BenNevis/run`, which **isort is deliberately not run over**, and
`utils`, `steps` and `local -> tools/kaldi/...`, whose own Python would be
linted on any machine that has a Kaldi checkout. Measured: `isort egs/`
reports three BenNevis files, and `black egs/` walks sixty-four.

So the linters name each recipe file. That is a list, and this is what
stops it falling behind -- the same shape as `test_run_stages.py`
asserting its own membership and `test_decode_k2_settings.py` asserting
its own.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PRE_COMMIT = REPO_ROOT / ".pre-commit-config.yaml"

#: The linters that must cover a recipe's Python, by step name.
LINTERS = ("black", "isort", "flake8")


def recipe_python():
    """
    Every real Python file under `egs/`, as a POSIX path.

    Symlinked directories are not followed -- that is the whole point --
    so this walks only what the repository itself holds.
    """
    found = []
    for recipe in sorted(REPO_ROOT.glob("egs/*/*")):
        if not recipe.is_dir():
            continue
        for child in sorted(recipe.iterdir()):
            if child.is_symlink():
                continue
            if child.is_file() and child.suffix == ".py":
                found.append(child.relative_to(REPO_ROOT).as_posix())
            elif child.is_dir():
                for path in sorted(child.rglob("*.py")):
                    if not any(part.is_symlink() for part in [path, *path.parents]):
                        found.append(path.relative_to(REPO_ROOT).as_posix())
    return sorted(set(found))


def linter_command(name):
    workflow = yaml.safe_load(WORKFLOW.read_text())
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if step.get("name") == name:
                return step["run"]
    raise AssertionError(f"no step named {name!r}")


def test_a_recipe_holds_python_at_all():
    """
    A parametrised suite over an empty list passes, which is how the
    `G.fst` gate in #41 came to be a test that could not fail.
    """
    assert recipe_python(), "no recipe holds a Python file"


@pytest.mark.parametrize("linter", LINTERS)
def test_every_recipe_python_file_is_named(linter):
    command = linter_command(linter)
    for path in recipe_python():
        assert path in command, (
            f"{path} is not linted by {linter}. Add it to the {linter} step "
            f"in .github/workflows/ci.yml -- by path, not by passing `egs/`, "
            f"which follows the recipe symlinks into BenNevis/ and into any "
            f"Kaldi checkout."
        )


@pytest.mark.parametrize("linter", LINTERS)
def test_the_linters_are_not_given_the_whole_directory(linter):
    """
    The hazard this file exists for, pinned so it cannot come back as a
    convenience. `egs/` would be shorter and would lint a Kaldi checkout.
    """
    command = linter_command(linter)
    assert " egs/\n" not in command and not command.rstrip().endswith(
        " egs/"
    ), f"{linter} is given egs/, which follows the recipe symlinks"


def test_pre_commit_covers_recipe_python():
    """
    pre-commit passes the staged paths rather than a directory, so it
    cannot traverse a symlink -- but its `files` pattern still has to
    match, or a recipe's Python is formatted in CI and not on the
    developer's machine, which is the worst way round.
    """
    config = yaml.safe_load(PRE_COMMIT.read_text())
    patterns = {}
    for repo in config["repos"]:
        for hook in repo["hooks"]:
            if hook["id"] in LINTERS:
                patterns[hook["id"]] = hook.get("files", "")

    import re

    for name, pattern in patterns.items():
        if not pattern:
            continue
        for path in recipe_python():
            assert re.match(pattern, path), (
                f"pre-commit's {name} pattern {pattern!r} does not match " f"{path}"
            )
