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
        for key in (
            "model",
            "test_set",
            "reference_words",
            "utterances",
            "topologies",
        ):
            assert key in ledger

    def test_the_word_count_follows_from_the_split(self, ledger):
        """
        Every yesno utterance is eight words -- the file name is eight binary
        digits, one per spoken word -- so the two numbers cannot move
        independently.
        """
        test_utterances = ledger["utterances"][ledger["test_set"]]
        assert ledger["reference_words"] == 8 * test_utterances

    def test_the_split_covers_the_whole_corpus(self, ledger):
        """
        Sixty waves, all used, none twice. A split that lost some would make
        the baseline quietly easier.
        """
        assert sum(ledger["utterances"].values()) == 60

    def test_every_entry_has_its_numbers(self, ledger):
        expected = {
            "reference_wer",
            "measured_wer",
            "measured_errors",
            "max_wer",
        }
        for name, entry in ledger["topologies"].items():
            assert set(entry) == expected, name
            for key in expected:
                assert isinstance(entry[key], (int, float)), f"{name}.{key}"

    def test_the_gate_is_above_what_was_measured(self, ledger):
        """
        A gate below the recorded run would fail on a correct one, which
        trains everyone to raise it -- the one thing that must not happen to
        a ledger.
        """
        for name, entry in ledger["topologies"].items():
            assert entry["max_wer"] > entry["measured_wer"], name

    def test_the_measured_value_agrees_with_its_error_count(self, ledger):
        """
        The two are the same fact twice. If they disagree, one was edited by
        hand.
        """
        words = ledger["reference_words"]
        for name, entry in ledger["topologies"].items():
            implied = 100.0 * entry["measured_errors"] / words
            assert abs(entry["measured_wer"] - implied) < 0.01, name

    def test_the_measurement_records_where_it_came_from(self, ledger):
        """
        A number with no provenance cannot be reproduced or doubted.
        """
        measured = ledger["measured"]
        for key in ("run", "date", "image"):
            assert measured.get(key), key
        assert "@sha256:" in measured["image"]

    def test_the_measurement_is_close_to_the_published_result(self, ledger):
        """
        The point of the exercise: this repository still reproduces the
        author's pre-refactor numbers. Three word errors of latitude, across
        a change of device, torch version and most of the toolkit.
        """
        allowance = 3 * 100.0 / ledger["reference_words"]
        for name, entry in ledger["topologies"].items():
            drift = abs(entry["measured_wer"] - entry["reference_wer"])
            assert drift <= allowance, (
                f"{name}: measured {entry['measured_wer']}% against the "
                f"published {entry['reference_wer']}%"
            )

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
