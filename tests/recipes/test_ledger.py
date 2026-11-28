"""
Tests for the WER ledgers themselves.

A ledger is the gate the recipe tier measures against, so its shape is
worth pinning even though nothing in it is computed: a typo in a topology
name would silently exclude that topology from the gate, which is the kind
of failure that is noticed a year later.

**The ledgers are discovered rather than listed**, which is the lesson
`tests/reference/test_corpora.py` already carries: this module used to
have yesno's path as a constant, so committing a second ledger would have
added a file no test looked at. A ledger's name gives its recipe, and the
corpus-specific facts live in their own classes below -- yesno's word
count follows from its file names, and mini-librispeech's does not.

Tier 1, because reading a JSON file needs nothing.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import json
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TESTS_DIR.parent
LEDGER_DIR = TESTS_DIR / "golden" / "wer"

#: Which recipe each ledger gates. A ledger whose corpus has several
#: recipes would need this to say which; none does yet.
RECIPES = {
    "yesno": "egs/yesno/asr",
    "mini_librispeech": "egs/mini_librispeech/asr",
}

#: The largest allowance a gate may carry, in absolute percentage points.
#: Three points is a few hundred word errors on the largest set here and
#: is never a plausible "noise" band -- the failures these corpora exist
#: to catch are a broken feature pipeline or a decoder that deletes
#: everything, which cost tens of points.
MAX_ALLOWANCE = 3.0


def ledgers():
    if not LEDGER_DIR.is_dir():
        return []
    return sorted(path.stem for path in LEDGER_DIR.glob("*.json"))


def load(name):
    with open(LEDGER_DIR / f"{name}.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def test_there_are_ledgers_to_check():
    """
    A parametrised suite over an empty list passes. That is how the
    `G.fst` gate in #41 managed to be a test which could not fail.
    """
    assert ledgers(), f"no ledger found in {LEDGER_DIR}"


@pytest.mark.parametrize("name", ledgers())
class TestEveryLedger:
    def test_it_names_a_recipe_that_exists(self, name):
        assert name in RECIPES, (
            f"{name}.json gates no known recipe; add it to RECIPES so the "
            f"checks below can find its run.sh"
        )
        assert (REPO_ROOT / RECIPES[name]).is_dir()

    def test_it_has_the_required_keys(self, name):
        ledger = load(name)
        for key in (
            "model",
            "test_set",
            "reference_words",
            "utterances",
            "topologies",
            "measured",
        ):
            assert key in ledger, key

    def test_the_test_set_is_one_of_the_splits(self, name):
        ledger = load(name)
        assert ledger["test_set"] in ledger["utterances"]

    def test_every_entry_has_its_numbers(self, name):
        ledger = load(name)
        for topology, entry in ledger["topologies"].items():
            for key in ("measured_wer", "measured_errors", "max_wer"):
                assert isinstance(entry.get(key), (int, float)), f"{topology}.{key}"
            # `reference_wer` must be *present* and may be null: a corpus
            # with no published pre-refactor number has nothing to put
            # there, and leaving the key out would make that
            # indistinguishable from forgetting it.
            assert "reference_wer" in entry, topology

    def test_the_gate_is_above_what_was_measured(self, name):
        """
        A gate below the recorded run would fail on a correct one, which
        trains everyone to raise it -- the one thing that must not happen
        to a ledger.
        """
        for topology, entry in load(name)["topologies"].items():
            assert entry["max_wer"] > entry["measured_wer"], topology

    def test_the_gate_is_not_absurdly_loose(self, name):
        """
        Past a few points the gate stops distinguishing a working
        topology from a broken one, and a ledger that cannot fail is the
        same as no ledger.
        """
        for topology, entry in load(name)["topologies"].items():
            allowance = entry["max_wer"] - entry["measured_wer"]
            assert allowance <= MAX_ALLOWANCE, f"{topology}: {allowance:.2f} points"

    def test_the_measured_value_agrees_with_its_error_count(self, name):
        """
        The two are the same fact twice. If they disagree, one was edited
        by hand.
        """
        ledger = load(name)
        words = ledger["reference_words"]
        for topology, entry in ledger["topologies"].items():
            implied = 100.0 * entry["measured_errors"] / words
            assert abs(entry["measured_wer"] - implied) < 0.01, topology

    def test_a_breakdown_sums_to_the_error_count(self, name):
        """
        Where the substitutions, deletions and insertions are recorded,
        they are the error count split three ways and must add up.
        Optional because yesno's entry predates recording them.
        """
        checked = 0
        for topology, entry in load(name)["topologies"].items():
            breakdown = entry.get("measured_breakdown")
            if breakdown is None:
                continue
            assert set(breakdown) == {"substitutions", "deletions", "insertions"}
            assert sum(breakdown.values()) == entry["measured_errors"], topology
            checked += 1
        if checked == 0:
            pytest.skip(f"{name} records no breakdown")

    def test_the_measurement_records_where_it_came_from(self, name):
        """
        A number with no provenance cannot be reproduced or doubted.
        """
        measured = load(name)["measured"]
        for key in ("run", "date", "image"):
            assert measured.get(key), key
        assert "@sha256:" in measured["image"]

    def test_it_says_which_features_it_was_measured_on(self, name):
        """
        Dropping pitch moved every recipe's input from 83 dimensions to
        80, so a WER with no feature era attached cannot be compared with
        one taken after. Both eras are legitimate; silence is not.
        """
        features = load(name).get("features")
        assert features, "no `features` block"
        for key in ("measured_at", "current"):
            assert "dimension" in features.get(key, ""), key

    def test_it_covers_every_topology_the_recipe_trains(self, name):
        """
        The gate must cover what the recipe actually runs. A topology in
        run.sh but not here would be trained and never checked -- and one
        here that the recipe does not train would be a gate on nothing.
        """
        run = REPO_ROOT / RECIPES[name] / "run.sh"
        line = next(
            line
            for line in run.read_text(encoding="utf-8").splitlines()
            if line.startswith("topos=")
        )
        topologies = line.split("=", 1)[1].strip().strip('"').split()
        assert sorted(topologies) == sorted(load(name)["topologies"])

    def test_every_topology_has_a_spec(self, name):
        """
        Each entry must still exist as a topology, so a renamed spec
        cannot leave a stale ledger entry behind. Read off the spec
        directory rather than through `BenNevis.topos`, which would make
        this tier 1b.
        """
        available = {
            path.stem
            for path in (REPO_ROOT / "BenNevis" / "topos" / "specs").glob("*.yaml")
        }
        assert available, "no topology specs found"
        assert set(load(name)["topologies"]) <= available

    def test_the_model_matches_the_recipe_default(self, name):
        run = REPO_ROOT / RECIPES[name] / "run.sh"
        line = next(
            line
            for line in run.read_text(encoding="utf-8").splitlines()
            if line.startswith("model=")
        )
        assert line.split("=", 1)[1].strip() == load(name)["model"]


class TestYesno:
    """
    What is true of yesno and of no other corpus.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def ledger():
        return load("yesno")

    def test_the_word_count_follows_from_the_split(self, ledger):
        """
        Every yesno utterance is eight words -- the file name is eight
        binary digits, one per spoken word -- so the two numbers cannot
        move independently.
        """
        assert ledger["reference_words"] == 8 * ledger["utterances"]["test_yesno"]

    def test_the_split_covers_the_whole_corpus(self, ledger):
        """
        Sixty waves, all used, none twice. A split that lost some would
        make the baseline quietly easier.
        """
        assert sum(ledger["utterances"].values()) == 60

    def test_the_measurement_is_close_to_the_published_result(self, ledger):
        """
        The point of the exercise: this repository still reproduces the
        author's pre-refactor numbers. Three word errors of latitude,
        across a change of device, torch version and most of the toolkit.
        """
        allowance = 3 * 100.0 / ledger["reference_words"]
        for name, entry in ledger["topologies"].items():
            drift = abs(entry["measured_wer"] - entry["reference_wer"])
            assert drift <= allowance, (
                f"{name}: measured {entry['measured_wer']}% against the "
                f"published {entry['reference_wer']}%"
            )


class TestMiniLibrispeech:
    """
    What is true of mini-librispeech and of no other corpus.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def ledger():
        return load("mini_librispeech")

    def test_it_has_no_published_reference(self, ledger):
        """
        yesno's entry is judged against the author's own CUDA runs. This
        recipe has never published a WER, so `reference_wer` is null and
        `measured_wer` is the only anchor -- which is why the gate is
        wider and why the file says so.
        """
        assert "none" in ledger["reference_source"]
        for name, entry in ledger["topologies"].items():
            assert entry["reference_wer"] is None, name

    def test_the_python_decoder_is_what_is_gated(self, ledger):
        """
        `measured_wer` must be BenNevis's own search, not Kaldi's. Both
        scored the same checkpoint and the Kaldi number is a cross-check
        that will leave with Kaldi; gating on it would make the ledger
        depend on a dependency being deleted.
        """
        for name, entry in ledger["topologies"].items():
            assert "kaldi_wer" in entry, name
            assert entry["measured_wer"] != entry["kaldi_wer"], name
            # The run this records: the Python search deletes less than
            # decode-faster and scores better overall.
            assert entry["measured_wer"] < entry["kaldi_wer"], name
            assert (
                entry["measured_breakdown"]["deletions"] / ledger["reference_words"]
                < entry["kaldi_breakdown"]["deletions"] / 100.0
            ), name

    def test_the_validation_loss_is_recorded_per_epoch(self, ledger):
        """
        The loss curve is what says whether more epochs would help, and
        it is the first thing wanted when a later run scores differently.
        """
        losses = ledger["training"]["valid_loss"]
        assert len(losses) == ledger["training"]["epochs"]
        assert losses == sorted(losses, reverse=True), "the loss must fall"
