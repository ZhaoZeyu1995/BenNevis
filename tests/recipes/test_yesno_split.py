"""
Tests for the yesno data split.

The split decides which utterances the published WERs in
`egs/yesno/asr/README.md` were measured on, so it is not a detail. It was
also, until recently, an accident: the corpus tarball contains `README` and
`README~` beside the sixty waves, `prepare_data.sh` listed the directory
without filtering, and so the list was 62 long. Half of 62 is 31, which put
the first 31 waves in training and the remaining 29 plus both READMEs in
test. Feature extraction then failed on the READMEs, Kaldi warned "Probably
a serious error", `fix_data_dir.sh` silently dropped them, and the test set
was 29 while claiming 31.

Filtering the list fixes that, but naively it also moves one utterance from
training to test, because half of 60 is 30. The split count is therefore
stated explicitly as 31 to keep the membership identical to the runs the
published numbers came from.

These tests drive the real Perl script over a synthetic listing with the two
READMEs in it, so they pin that membership rather than the explanation of it.

Tier 1: the script is Perl, which every runner has, and the test skips if it
does not.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RECIPE = REPO_ROOT / "egs" / "yesno" / "asr"
SPLIT = RECIPE / "local" / "create_yesno_waves_test_train.pl"

pytestmark = pytest.mark.skipif(
    shutil.which("perl") is None, reason="the split script is Perl"
)

#: The corpus's file names are eight binary digits, one per spoken word.
WAVES = sorted(
    f"{a}_{b}_{c}_{d}_{e}_{f}_{g}_{h}.wav"
    for a in (0, 1)
    for b in (0, 1)
    for c in (0, 1)
    for d in (0, 1)
    for e in (0, 1)
    for f in (0, 1)
    for g in (0, 1)
    for h in (0, 1)
)[:60]

#: What `ls -1` produced before the filter, under LC_ALL=C: digits sort
#: before `R`, so both READMEs land at the end and therefore in the test half.
UNFILTERED = WAVES + ["README", "README~"]


def run_split(tmp_path, entries, train_count=None):
    """Run the real script and return its (train, test) lists."""
    listing = tmp_path / "waves_all.list"
    listing.write_text("".join(f"{entry}\n" for entry in entries))
    train = tmp_path / "waves.train"
    test = tmp_path / "waves.test"

    command = ["perl", str(SPLIT), str(listing), str(test), str(train)]
    if train_count is not None:
        command.append(str(train_count))
    subprocess.run(command, check=True, cwd=tmp_path)

    return train.read_text().split(), test.read_text().split()


class TestTheSplitIsPreserved:
    def test_filtering_with_an_explicit_count_reproduces_the_old_membership(
        self, tmp_path
    ):
        """
        The point of the whole exercise: the waves in each half must be the
        same ones as before the READMEs stopped being counted, or the
        published numbers stop being comparable.
        """
        before_train, before_test = run_split(tmp_path, UNFILTERED)
        after_train, after_test = run_split(tmp_path, WAVES, train_count=31)

        assert after_train == before_train
        assert after_test == [entry for entry in before_test if entry.endswith(".wav")]

    def test_the_counts_are_thirty_one_and_twenty_nine(self, tmp_path):
        train, test = run_split(tmp_path, WAVES, train_count=31)
        assert len(train) == 31
        assert len(test) == 29
        assert len(train) + len(test) == 60

    def test_no_wave_is_in_both_halves(self, tmp_path):
        train, test = run_split(tmp_path, WAVES, train_count=31)
        assert not set(train) & set(test)

    def test_every_wave_is_used(self, tmp_path):
        train, test = run_split(tmp_path, WAVES, train_count=31)
        assert sorted(train + test) == sorted(WAVES)

    def test_an_even_split_would_have_moved_one_across(self, tmp_path):
        """
        Why the count is stated rather than left to halving. This is the
        change that was not made, and the test records why.
        """
        explicit_train, _ = run_split(tmp_path, WAVES, train_count=31)
        halved_train, halved_test = run_split(tmp_path, WAVES)

        assert len(halved_train) == 30
        assert len(halved_test) == 30
        moved = set(explicit_train) - set(halved_train)
        assert len(moved) == 1
        assert moved.pop() in halved_test


class TestTheScriptItself:
    def test_the_count_defaults_to_half(self, tmp_path):
        """
        Unchanged behaviour when no count is given, so nothing else that
        uses this script is affected.
        """
        train, test = run_split(tmp_path, [f"{i}.wav" for i in range(10)])
        assert len(train) == 5
        assert len(test) == 5

    def test_an_odd_list_rounds_the_training_half_down(self, tmp_path):
        train, test = run_split(tmp_path, [f"{i}.wav" for i in range(9)])
        assert len(train) == 4
        assert len(test) == 5

    def test_the_order_within_each_half_is_the_list_order(self, tmp_path):
        entries = [f"{i:02d}.wav" for i in range(10)]
        train, test = run_split(tmp_path, entries, train_count=3)
        assert train == entries[:3]
        assert test == entries[3:]

    def test_asking_for_more_than_the_list_holds_is_refused(self, tmp_path):
        """
        Rather than silently producing an empty test set, which would score
        perfectly.
        """
        with pytest.raises(subprocess.CalledProcessError):
            run_split(tmp_path, [f"{i}.wav" for i in range(5)], train_count=99)

    def test_all_of_the_list_can_go_to_training(self, tmp_path):
        train, test = run_split(tmp_path, [f"{i}.wav" for i in range(5)], train_count=5)
        assert len(train) == 5
        assert test == []


class TestDataPreparationFilters:
    def test_prepare_data_lists_only_waves(self):
        """
        The filter is in shell, so this reads it. Weak on its own, which is
        why the membership tests above exist; its job is to catch the filter
        being dropped.
        """
        text = (RECIPE / "local" / "prepare_data.sh").read_text()
        listing = [
            line
            for line in text.splitlines()
            if "waves_all.list" in line and not line.strip().startswith("#")
        ]
        assert listing, "nothing writes waves_all.list any more"
        assert any(".wav" in line for line in listing), listing

    def test_prepare_data_states_the_split_count(self):
        text = (RECIPE / "local" / "prepare_data.sh").read_text()
        called = [
            line
            for line in text.splitlines()
            if "create_yesno_waves_test_train.pl" in line
            and not line.strip().startswith("#")
        ]
        assert len(called) == 1, called
        assert called[0].rstrip().endswith(" 31"), called[0]
