"""
Tests that the pinned versions agree across the places they appear.

CLAUDE.md already warns that the k2 wheel is built against an exact torch
version and must be bumped with it. Now that there are two places saying so
-- the tier 2 job in .github/workflows/ci.yml and docker/Dockerfile -- the
warning is not enough on its own: a mismatch means the container tests a
different torch from the one CI tests, and the symptom would be a k2 import
error inside a nightly job rather than anything pointing at the cause.

These are plain text checks on purpose. Parsing the workflow with a YAML
library would need pyyaml to read a `run:` block and then a regex over it
anyway, so the regex is applied directly.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"

#: `k2==1.24.4.dev20261112+cpu.torch2.15.0`, wherever it is written.
K2_RE = re.compile(r"k2==([0-9][^\s\"'\\]*)")
#: `torch==2.15.0`, including the Dockerfile's `"torch==${TORCH_VERSION}"`.
TORCH_RE = re.compile(r"torch==([0-9][^\s\"'\\]*)")
#: `ARG TORCH_VERSION=2.15.0`
ARG_RE = re.compile(r"^ARG\s+(\w+)=(\S+)\s*$", re.MULTILINE)


def docker_args():
    """The Dockerfile's build arguments and their defaults."""
    return dict(ARG_RE.findall(DOCKERFILE.read_text()))


class TestK2Pin:
    def test_ci_pins_k2_exactly(self):
        assert K2_RE.findall(CI.read_text())

    def test_the_dockerfile_and_ci_pin_the_same_k2(self):
        from_ci = set(K2_RE.findall(CI.read_text()))
        from_docker = {docker_args()["K2_VERSION"]}
        assert from_ci == from_docker, (
            f"ci.yml pins k2 {sorted(from_ci)} but docker/Dockerfile pins "
            f"{sorted(from_docker)}; the container would test a different k2"
        )

    def test_the_k2_pin_names_its_torch(self):
        """
        The local version suffix records which torch the wheel was built
        against, and it must be the torch that is installed beside it.
        """
        k2_version = docker_args()["K2_VERSION"]
        torch_version = docker_args()["TORCH_VERSION"]
        assert f"torch{torch_version}" in k2_version, (
            f"k2 {k2_version} was built against a different torch from the "
            f"pinned {torch_version}"
        )


class TestTorchPin:
    def test_the_dockerfile_and_ci_pin_the_same_torch(self):
        from_ci = set(TORCH_RE.findall(CI.read_text()))
        assert from_ci, "ci.yml no longer pins torch exactly"
        assert from_ci == {docker_args()["TORCH_VERSION"]}


class TestKaldiPin:
    def test_kaldi_is_pinned_to_a_full_commit(self):
        """
        Not a branch and not an abbreviated hash. A floating Kaldi would make
        an old result impossible to reproduce, which is the whole reason the
        pre-refactor history is kept intact.
        """
        commit = docker_args()["KALDI_COMMIT"]
        assert re.fullmatch(r"[0-9a-f]{40}", commit), commit


class TestWorkflowWiring:
    def test_the_docker_workflow_builds_only_when_it_changes(self):
        """
        A Kaldi build is most of an hour. It must not be on the path of an
        ordinary push.
        """
        workflow = (REPO_ROOT / ".github" / "workflows" / "docker.yml").read_text()
        assert "paths:" in workflow
        assert "docker/**" in workflow
        assert "pyproject.toml" in workflow

    def test_the_dockerfile_installs_the_derived_requirements(self):
        """
        The image must install the list `deps.py` produces, not a list copied
        into the Dockerfile, which is what the tests in test_deps.py assume.
        """
        text = DOCKERFILE.read_text()
        assert "deps.py" in text
        assert "-r /tmp/build/requirements.txt" in text

    def test_the_dockerfile_does_not_install_bennevis(self):
        """
        The container runs against a mounted tree, reached through PYTHONPATH.
        An installed copy would shadow it and the nightly job would test the
        image's BenNevis rather than the commit under test.
        """
        text = DOCKERFILE.read_text()
        assert "pip install --no-cache-dir -e ." not in text
        assert "pip install -e ." not in text
