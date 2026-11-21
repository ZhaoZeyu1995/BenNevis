"""
Shared test fixtures.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import subprocess
import sys
import textwrap

import pytest

# Importing a module in a subprocess with one package forced to fail. Run out
# of process so the import block cannot leak into the rest of the session, and
# so the result is unaffected by what previous tests already imported.
_SCRIPT = """
import sys

class _Blocker:
    def find_module(self, name, path=None):
        return self if name == {blocked!r} else None

    def find_spec(self, name, path=None, target=None):
        if name == {blocked!r}:
            raise ImportError("blocked by test: {blocked} must not be required")
        return None

sys.meta_path.insert(0, _Blocker())
import {module}  # noqa: F401
assert {blocked!r} not in sys.modules, "{blocked} was imported after all"
print("OK")
"""


@pytest.fixture
def import_without():
    """
    Assert that a module imports while a given package is unavailable.

    Used to pin decouplings that would otherwise rot: a dependency reappearing
    in an import chain is easy to do by accident and invisible locally, where
    everything happens to be installed.

    Returns
    -------
    check: Callable[[str, str], None]
        Takes the module to import and the package that must not be needed,
        and fails the test if the import does not succeed without it.
    """

    def check(module: str, blocked: str) -> None:
        script = textwrap.dedent(_SCRIPT).format(module=module, blocked=blocked)
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert result.returncode == 0, (
            f"importing {module} without the {blocked} package failed:\n"
            f"{result.stderr}"
        )
        assert "OK" in result.stdout

    return check


@pytest.fixture(scope="session")
def process_group(tmp_path_factory):
    """
    A real single-process gloo process group, for the duration of the session.

    `Trainer` requires an initialised process group: it reads the rank and
    world size in `__init__` and all-reduces losses. Rather than stub those
    out, the tests run against a genuine one-rank group, so the distributed
    code paths are the ones actually exercised.

    A FileStore is used rather than TCP so that nothing depends on a free
    port, which is a common source of flakiness in CI.
    """
    import torch.distributed as dist

    if dist.is_initialized():  # pragma: no cover - defensive
        yield
        return

    store_path = tmp_path_factory.mktemp("pg") / "store"
    store = dist.FileStore(str(store_path), 1)
    dist.init_process_group(backend="gloo", store=store, rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()
