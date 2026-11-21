"""
Tests for the dependency list the Docker image installs.

The image does not install BenNevis, only its dependencies, because it runs
against a mounted working tree. That makes it easy for the image's dependency
list to drift from `pyproject.toml` without anyone noticing until a recipe
fails on an import, so the list is derived rather than written down, and these
tests pin the derivation.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "docker"))

import deps  # noqa: E402

PYPROJECT = REPO_ROOT / "pyproject.toml"

# `tomllib` is standard from 3.11. deps.py runs only inside the image, whose
# Python is 3.12, so needing it there is not a constraint -- but tier 1 sweeps
# back to 3.10, where reading pyproject.toml is impossible without a
# third-party parser that this tier is not going to install for one file.
# The tests that do not read it still run everywhere.
needs_tomllib = pytest.mark.skipif(
    importlib.util.find_spec("tomllib") is None,
    reason="tomllib is standard from 3.11; the image's Python is 3.12",
)


class TestRequirementName:
    @pytest.mark.parametrize(
        "requirement,expected",
        [
            ("torch", "torch"),
            ("pytest>=7.0", "pytest"),
            ("hydra-core", "hydra-core"),
            ("numpy==1.26.4", "numpy"),
            ("torch<3", "torch"),
            ("torch!=2.0.0", "torch"),
            ("torch~=2.15", "torch"),
            ("uvicorn[standard]", "uvicorn"),
            ("tomli; python_version < '3.11'", "tomli"),
            ("  Torch  ", "torch"),
        ],
    )
    def test_strips_everything_but_the_name(self, requirement, expected):
        assert deps.requirement_name(requirement) == expected


@needs_tomllib
class TestDependencies:
    def test_torch_is_excluded(self):
        """
        torch is pinned in the Dockerfile to the version the k2 wheel was
        built against. Installing it again from this list, unpinned, would
        silently replace that build and break k2.
        """
        names = [deps.requirement_name(r) for r in deps.dependencies(PYPROJECT)]
        assert "torch" not in names
        assert "torchaudio" not in names

    def test_everything_else_is_kept(self):
        """
        Derived from pyproject rather than listed, so a dependency added there
        reaches the image without anyone remembering to update the Dockerfile.
        """
        import tomllib

        with open(PYPROJECT, "rb") as handle:
            declared = tomllib.load(handle)["project"]["dependencies"]

        expected = {
            deps.requirement_name(r)
            for r in declared
            if deps.requirement_name(r) not in deps.PINNED_ELSEWHERE
        }
        got = {deps.requirement_name(r) for r in deps.dependencies(PYPROJECT)}
        assert expected <= got

    def test_pytest_is_added(self):
        """The container runs the tests, and pytest is not a runtime dep."""
        names = [deps.requirement_name(r) for r in deps.dependencies(PYPROJECT)]
        assert "pytest" in names

    def test_k2_is_not_in_the_list(self):
        """
        k2 comes from its own index with a local version suffix, so it cannot
        be a pyproject dependency and must not appear here either.
        """
        names = [deps.requirement_name(r) for r in deps.dependencies(PYPROJECT)]
        assert "k2" not in names

    def test_the_declared_dependencies_are_not_empty(self):
        assert len(deps.dependencies(PYPROJECT)) > 1

    def test_version_specifiers_survive(self):
        """
        The requirement strings are passed through, not just the names: a pin
        in pyproject must reach pip.
        """
        for requirement in deps.dependencies(PYPROJECT):
            assert requirement == requirement.strip()
            assert requirement


class TestCli:
    @needs_tomllib
    def test_prints_one_per_line(self, capsys):
        assert deps.main([str(PYPROJECT)]) == 0
        out = capsys.readouterr().out.splitlines()
        assert out == deps.dependencies(PYPROJECT)

    def test_wrong_arity_is_refused(self, capsys):
        assert deps.main([]) == 2
        assert "usage" in capsys.readouterr().err
