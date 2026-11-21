"""
Tests for the WER ledger itself.

The ledger is the gate the recipe tier measures against, so its shape is
worth pinning even though nothing in it is computed: a typo in a topology
name would silently exclude that topology from the gate, which is the kind
of failure that is noticed a year later.

Tier 1, because reading a JSON file needs nothing.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import json
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TESTS_DIR.parent
LEDGER = TESTS_DIR / "golden" / "wer" / "yesno.json"
RECIPE = REPO_ROOT / "egs" / "yesno" / "asr"


@pytest.fixture(scope="module")
def ledger():
    with open(LEDGER, "r", encoding="utf-8") as handle:
        return json.load(handle)


class TestShape:
    def test_has_the_required_keys(self, ledger):
        for key in ("model", "test_set", "reference_words", "topologies"):
            assert key in ledger

    def test_every_entry_has_both_numbers(self, ledger):
        for name, entry in ledger["topologies"].items():
            assert set(entry) == {"reference_wer", "max_wer"}, name
            assert isinstance(entry["reference_wer"], (int, float)), name
            assert isinstance(entry["max_wer"], (int, float)), name

    def test_the_gate_is_not_below_the_reference(self, ledger):
        """
        A gate under the published number would fail on a correct run, which
        trains everyone to raise it -- the one thing that must not happen to
        a ledger.
        """
        for name, entry in ledger["topologies"].items():
            assert entry["max_wer"] >= entry["reference_wer"], name

    def test_the_gate_is_not_absurdly_loose(self, ledger):
        """
        Above a few percent on a 232-word set the gate stops distinguishing a
        working topology from a broken one.
        """
        for name, entry in ledger["topologies"].items():
            assert entry["max_wer"] <= 5.0, name


class TestAgreesWithTheRecipe:
    def test_it_covers_every_topology_the_recipe_trains(self, ledger):
        """
        The gate must cover what the recipe actually runs. A topology in
        run.sh but not here would be trained nightly and never checked.
        """
        text = (RECIPE / "run.sh").read_text()
        line = next(line for line in text.splitlines() if line.startswith("topos="))
        topologies = line.split("=", 1)[1].strip().strip('"').split()
        assert sorted(topologies) == sorted(ledger["topologies"])

    def test_every_topology_has_a_spec(self, ledger):
        """
        And each one must still exist as a topology, so that a renamed spec
        cannot leave a stale ledger entry behind.
        """
        from BenNevis.topos import spec

        available = set(spec.available())
        assert set(ledger["topologies"]) <= available

    def test_the_model_matches_the_recipe_default(self, ledger):
        text = (RECIPE / "run.sh").read_text()
        line = next(line for line in text.splitlines() if line.startswith("model="))
        assert line.split("=", 1)[1].strip() == ledger["model"]
