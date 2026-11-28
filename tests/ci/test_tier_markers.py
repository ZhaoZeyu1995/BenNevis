"""
A test that needs a heavy dependency carries the marker that selects it.

Tier 1: it reads the test files as text.

**This exists because one file of eighteen did not, and the tests in it
never ran anywhere.** `tests/utils/test_features.py` called
`pytest.importorskip("torch")` and carried no `torch` marker, so:

* in **tier 1** -- which installs pytest and pyyaml and nothing else --
  everything below that line skipped;
* in **tier 1b**, which selects `-m torch`, the whole file was
  *deselected* for want of the marker.

So `TestFbank` and the filterbank gate ran only on a machine that
happened to have torch **and** used the tier-1 selection -- a developer's
laptop. The gate had been written, waited several captures for its input,
and would have gone on not running after it arrived.

**A test that never runs is harder to notice than one that cannot fail**,
because the suite does not even report a skip: `-m torch` deselects it
silently, and the deselected count is a number nobody reads. That is the
failure this file is here to make loud.

The rule it pins: if a module reaches for torch, k2, numpy, kaldiio or
torchaudio, something in it must carry `@pytest.mark.torch` or
`@pytest.mark.k2`, or the tier that installs that dependency will not
select it.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import re
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent.parent

#: Dependencies that tier 1 does not install, so a module reaching for
#: one is not a tier 1 module however it is written.
HEAVY = ("torch", "torchaudio", "k2", "numpy", "kaldiio", "numba")

#: What selects a module into a tier that has those installed. `kaldi`
#: and `gpu` count too: tier 3 and tier 4 install everything.
MARKERS = ("torch", "k2", "kaldi", "gpu")

_IMPORTORSKIP = re.compile(r"importorskip\(\s*[\"']([a-z_0-9.]+)[\"']")
_MARKER = re.compile(r"pytest\.mark\.(\w+)")


def modules():
    return sorted(
        path for path in TESTS_DIR.rglob("test_*.py") if "__pycache__" not in str(path)
    )


def heavy_imports(text):
    """Which heavy dependencies a module importorskips."""
    found = {
        name.split(".")[0]
        for name in _IMPORTORSKIP.findall(text)
        if name.split(".")[0] in HEAVY
    }
    return sorted(found)


def markers(text):
    """Which tier markers appear anywhere in the module."""
    return sorted({name for name in _MARKER.findall(text) if name in MARKERS})


def needing_a_marker():
    out = []
    for path in modules():
        text = path.read_text(encoding="utf-8")
        if heavy_imports(text):
            out.append(path.relative_to(TESTS_DIR).as_posix())
    return out


def test_some_module_needs_a_marker():
    """
    A parametrised suite over an empty list passes, which is how the
    `G.fst` gate in #41 came to be a test that could not fail.
    """
    assert needing_a_marker(), "no module importorskips a heavy dependency"


@pytest.mark.parametrize("name", needing_a_marker())
def test_it_carries_a_tier_marker(name):
    """
    Otherwise the tier that installs the dependency deselects it and the
    tier that selects it has not installed it -- so it runs nowhere.
    """
    path = TESTS_DIR / name
    text = path.read_text(encoding="utf-8")
    assert markers(text), (
        f"{name} importorskips {', '.join(heavy_imports(text))} but carries "
        f"no {' or '.join('@pytest.mark.' + m for m in MARKERS)}. Tier 1 does "
        f"not install those, and the tier that does selects by marker -- so "
        f"every test in it would run nowhere."
    )


def test_the_detector_finds_the_known_modules():
    """
    The other half: a regex that matched nothing would make the test
    above pass for every module, which is exactly the shape of defect it
    is meant to catch.
    """
    found = needing_a_marker()
    assert len(found) >= 15, f"only {len(found)} modules detected: {found}"
    assert "utils/test_features.py" in found
    assert "core/test_trainer.py" in found


def test_a_module_with_no_heavy_import_is_not_required_to_be_marked():
    """
    Tier 1 is most of the suite and must stay unmarked, so the rule has
    to be one-directional. `tests/topos/` is the clearest case: the specs
    are YAML and the expander is pure Python.
    """
    unmarked = [
        path.relative_to(TESTS_DIR).as_posix()
        for path in modules()
        if not heavy_imports(path.read_text(encoding="utf-8"))
    ]
    assert len(unmarked) > 20, f"only {len(unmarked)} tier 1 modules?"
