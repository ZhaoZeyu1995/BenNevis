"""
yesno's data preparation in Python, against the Perl it replaces.

Tier 1: the Python needs only the standard library, and the Perl half
skips where `perl` is absent or the scripts have been removed.

**The Perl is the oracle, not the explanation.** `prepare_data.sh` and
its three scripts are what every WER published in
`egs/yesno/asr/README.md` was measured through, and the one thing a
reimplementation must not do is change which utterance is in which half
-- halving sixty gives thirty and moves one across, which would make
those numbers no longer comparable. So the scripts are kept in the tree
and driven here, and the Python is required to agree with them **byte for
byte** on `wav.scp` and `text`. That is a stronger standard than
restating the 31/29 split in a constant and asserting the constant, which
is what `tests/recipes/test_yesno_split.py` does for the split script
alone.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RECIPE = REPO_ROOT / "egs" / "yesno" / "asr"
LOCAL = RECIPE / "local"
PREPARE = LOCAL / "prepare_data.py"

sys.path.insert(0, str(LOCAL))
import prepare_data  # noqa: E402

#: The corpus's names: eight binary digits, one per spoken word. The real
#: tarball holds sixty of the 256 possibilities.
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

perl_only = pytest.mark.skipif(
    shutil.which("perl") is None or os.name == "nt",
    reason="the scripts being compared against are Perl, and `perl` on a "
    "Windows runner is not a given",
)


@pytest.fixture
def corpus(tmp_path):
    """
    An unpacked corpus, with the two READMEs the real tarball holds.

    They are the whole reason the split is what it is, so a fixture
    without them could not show that the Python filters as the fixed
    shell does.
    """
    waves = tmp_path / "waves_yesno"
    waves.mkdir()
    for name in WAVES:
        (waves / name).write_bytes(b"")
    (waves / "README").write_text("not audio\n", encoding="utf-8")
    (waves / "README~").write_text("not audio either\n", encoding="utf-8")
    return waves


def run_python(corpus, tmp_path, *extra):
    """Run the entry point as the recipe does, from the recipe directory."""
    done = subprocess.run(
        [
            sys.executable,
            str(PREPARE),
            str(corpus),
            "--data",
            str(tmp_path / "data"),
            *extra,
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    return tmp_path / "data"


class TestItAgreesWithThePerl:
    @perl_only
    def test_the_split_is_the_perls(self, corpus, tmp_path):
        """
        Driven through `create_yesno_waves_test_train.pl` itself, with the
        same explicit count the shell passes.
        """
        listing = tmp_path / "waves_all.list"
        listing.write_text(
            "".join(f"{name}\n" for name in sorted(WAVES)), encoding="utf-8"
        )
        subprocess.run(
            [
                "perl",
                str(LOCAL / "create_yesno_waves_test_train.pl"),
                str(listing),
                str(tmp_path / "waves.test"),
                str(tmp_path / "waves.train"),
                "31",
            ],
            check=True,
            capture_output=True,
        )
        theirs_train = (tmp_path / "waves.train").read_text().split()
        theirs_test = (tmp_path / "waves.test").read_text().split()

        mine_train, mine_test = prepare_data.split(prepare_data.list_waves(corpus))

        assert mine_train == theirs_train
        assert mine_test == theirs_test

    @perl_only
    @pytest.mark.parametrize("half", ["train", "test"])
    def test_wav_scp_is_byte_identical(self, corpus, tmp_path, half):
        names = prepare_data.list_waves(corpus)
        training, testing = prepare_data.split(names)
        chosen = training if half == "train" else testing

        listing = tmp_path / f"waves.{half}"
        listing.write_text("".join(f"{n}\n" for n in chosen), encoding="utf-8")
        theirs = subprocess.run(
            [
                "perl",
                str(LOCAL / "create_yesno_wav_scp.pl"),
                str(corpus),
                str(listing),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

        rows = prepare_data.tables(corpus, chosen)["wav.scp"]
        mine = "".join(f"{key} {rest}\n" for key, rest in rows)
        assert mine == theirs

    @perl_only
    @pytest.mark.parametrize("half", ["train", "test"])
    def test_the_transcriptions_are_byte_identical(self, corpus, tmp_path, half):
        names = prepare_data.list_waves(corpus)
        training, testing = prepare_data.split(names)
        chosen = training if half == "train" else testing

        listing = tmp_path / f"waves.{half}"
        listing.write_text("".join(f"{n}\n" for n in chosen), encoding="utf-8")
        theirs = subprocess.run(
            ["perl", str(LOCAL / "create_yesno_txt.pl"), str(listing)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

        rows = prepare_data.tables(corpus, chosen)["text"]
        mine = "".join(f"{key} {rest}\n" for key, rest in rows)
        assert mine == theirs


class TestTheSplitItself:
    def test_it_is_thirty_one_and_twenty_nine(self, corpus):
        """
        Not half of sixty. Halving would move one utterance across and
        make the WERs published in the recipe's README no longer
        comparable with a new run.
        """
        training, testing = prepare_data.split(prepare_data.list_waves(corpus))
        assert (len(training), len(testing)) == (31, 29)

    def test_the_readmes_are_not_utterances(self, corpus):
        """
        The defect this filter exists for: under `LC_ALL=C` digits sort
        before `R`, so both READMEs land at the end and therefore in the
        test half, where feature extraction fails on them and
        `fix_data_dir` drops them -- leaving a test set of 29 that claimed
        31. A real audio failure looks identical.
        """
        names = prepare_data.list_waves(corpus)
        assert len(names) == 60
        assert not [name for name in names if "README" in name]

    def test_every_wave_is_used_once(self, corpus):
        training, testing = prepare_data.split(prepare_data.list_waves(corpus))
        assert sorted(training + testing) == sorted(WAVES)
        assert not set(training) & set(testing)

    def test_the_order_is_bytes_not_locale(self, corpus):
        """
        `ls -1` under the `LC_ALL=C` the recipe's `path.sh` exports. A
        locale that folded the underscore would reorder the list and move
        the split.
        """
        names = prepare_data.list_waves(corpus)
        assert names == sorted(names, key=lambda name: name.encode("utf-8"))


class TestTheTranscriptions:
    @pytest.mark.parametrize(
        "name, expected",
        [
            ("0_0_0_0_0_0_0_0", "NO NO NO NO NO NO NO NO"),
            ("1_1_1_1_1_1_1_1", "YES YES YES YES YES YES YES YES"),
            ("1_0_1_0_1_0_1_0", "YES NO YES NO YES NO YES NO"),
            ("0_0_0_1_1_1_0_0.wav", "NO NO NO YES YES YES NO NO"),
        ],
    )
    def test_the_name_is_the_transcription(self, name, expected):
        assert prepare_data.transcribe(name) == expected

    def test_every_utterance_is_eight_words(self, corpus):
        """
        The arithmetic `tests/golden/wer/yesno.json` rests on: 232
        reference words is 29 utterances times eight.
        """
        for name in prepare_data.list_waves(corpus):
            assert len(prepare_data.transcribe(name).split()) == 8


class TestTheDataDirectories:
    def test_it_writes_the_four_files_for_both_halves(self, corpus, tmp_path):
        data = run_python(corpus, tmp_path)
        for dataset, count in (("train_yesno", 31), ("test_yesno", 29)):
            for name in ("wav.scp", "text", "utt2spk", "spk2utt"):
                path = data / dataset / name
                assert path.is_file(), f"{dataset}/{name}"
            assert len((data / dataset / "text").read_text().splitlines()) == count

    def test_every_utterance_is_the_global_speaker(self, corpus, tmp_path):
        """
        Kaldi's own choice for this corpus, and the reason
        `run/apply_cmvn.sh` clamps its job count to the speaker count --
        `split_scp.pl` refuses to split into more parts than there are
        speakers, which is why the recipe runs at `nj=1`.
        """
        data = run_python(corpus, tmp_path)
        for dataset in ("train_yesno", "test_yesno"):
            speakers = {
                line.split()[1]
                for line in (data / dataset / "utt2spk").read_text().splitlines()
            }
            assert speakers == {"global"}
            spk2utt = (data / dataset / "spk2utt").read_text().splitlines()
            assert len(spk2utt) == 1
            assert spk2utt[0].split()[0] == "global"

    def test_spk2utt_lists_exactly_utt2spk_s_utterances(self, corpus, tmp_path):
        data = run_python(corpus, tmp_path)
        for dataset in ("train_yesno", "test_yesno"):
            utterances = [
                line.split()[0]
                for line in (data / dataset / "utt2spk").read_text().splitlines()
            ]
            listed = (data / dataset / "spk2utt").read_text().split()[1:]
            assert listed == utterances

    def test_the_wave_paths_stay_relative_when_given_relative(self, tmp_path):
        """
        `create_yesno_wav_scp.pl` wrote `$waves_dir/$name` with the
        directory as it was given -- `waves_yesno` -- and every stage of
        the recipe runs from the recipe directory. Making them absolute
        would be a gratuitous difference from every recorded run.
        """
        waves = tmp_path / "waves_yesno"
        waves.mkdir()
        for name in WAVES[:2]:
            (waves / name).write_bytes(b"")
        rows = prepare_data.tables(Path("waves_yesno"), WAVES[:2])["wav.scp"]
        assert all(rest.startswith("waves_yesno/") for _key, rest in rows)

    def test_the_tables_are_sorted(self, corpus, tmp_path):
        """
        `fix_data_dir` would sort them later, but it also reports what it
        changed -- and a data directory that always needs sorting makes
        that report meaningless.
        """
        data = run_python(corpus, tmp_path)
        for dataset in ("train_yesno", "test_yesno"):
            for name in ("wav.scp", "text", "utt2spk"):
                lines = (data / dataset / name).read_text().splitlines()
                keys = [line.split()[0] for line in lines]
                assert keys == sorted(keys, key=lambda k: k.encode("utf-8"))


class TestItRefusesNonsense:
    def test_a_missing_corpus_is_an_error(self, tmp_path):
        done = subprocess.run(
            [sys.executable, str(PREPARE), str(tmp_path / "absent")],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 1
        assert "not a directory" in done.stderr

    def test_a_corpus_with_no_waves_is_an_error(self, tmp_path):
        """
        Two empty data directories read downstream as a model that
        recognises nothing rather than as a missing corpus.
        """
        empty = tmp_path / "waves_yesno"
        empty.mkdir()
        (empty / "README").write_text("only this\n", encoding="utf-8")
        done = subprocess.run(
            [sys.executable, str(PREPARE), str(empty)],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 1
        assert "no .wav files" in done.stderr

    def test_a_train_count_past_the_corpus_is_an_error(self, corpus, tmp_path):
        done = subprocess.run(
            [
                sys.executable,
                str(PREPARE),
                str(corpus),
                "--data",
                str(tmp_path / "data"),
                "--train-count",
                "100",
            ],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 1
        assert "but the corpus has" in done.stderr

    def test_a_corpus_of_the_wrong_size_warns(self, tmp_path):
        """
        Not fatal -- a changed corpus is the author's business -- but it
        moves the split and so every published number, which is worth
        saying rather than discovering from a WER.
        """
        waves = tmp_path / "waves_yesno"
        waves.mkdir()
        for name in WAVES[:40]:
            (waves / name).write_bytes(b"")
        done = subprocess.run(
            [
                sys.executable,
                str(PREPARE),
                str(waves),
                "--data",
                str(tmp_path / "data"),
            ],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0
        assert "no longer what it was" in done.stderr


class TestTheOraclesAreKept:
    """
    The three Perl scripts have no caller, and that is deliberate.

    They are what every WER published in `egs/yesno/asr/README.md` was
    measured through, and the tests above require this implementation to
    agree with them byte for byte. Deleting them would leave the split
    and the transcription mapping pinned only by constants restated in a
    test -- which is the weaker standard, and the one that cannot catch a
    constant and an implementation changed together.

    So removing them should be a decision with a replacement, not a
    tidy-up of files that look dead.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "create_yesno_waves_test_train.pl",
            "create_yesno_wav_scp.pl",
            "create_yesno_txt.pl",
        ],
    )
    def test_the_perl_is_still_there(self, name):
        assert (LOCAL / name).is_file(), (
            f"{name} is the oracle this implementation is compared against; "
            f"see this class's docstring before removing it"
        )

    def test_the_shell_driver_is_gone(self):
        """
        The other half: `prepare_data.sh` *is* replaced, so keeping it
        would leave two implementations and no statement of which runs.
        Its other two outputs -- `data/local/waves_all.list` and a copy of
        `task.arpabo` -- were read by nothing.
        """
        assert not (LOCAL / "prepare_data.sh").exists()


class TestTheRecipeCallsIt:
    @staticmethod
    def stage_one():
        text = (RECIPE / "run.sh").read_text(encoding="utf-8").replace("\\\n", " ")
        import re

        match = re.search(
            r"if \[ \$\{stage\} -le 1 \].*?\n(.*?)\nfi\n", text, re.DOTALL
        )
        assert match, "yesno has no stage 1"
        return match.group(1)

    def test_stage_one_runs_the_python(self):
        assert "local/prepare_data.py" in self.stage_one()
        assert "local/prepare_data.sh" not in self.stage_one()

    def test_stage_one_needs_nothing_from_kaldi(self):
        """
        What this change is for. yesno's stage 1 was the last part of the
        training path that reached into the Kaldi checkout -- through
        `utils/prepare_lang.sh` until #106, and through
        `utils/utt2spk_to_spk2utt.pl` inside `prepare_data.sh` until now.
        `local/prepare_dict.sh` stays and is pure `cp` and `grep`.
        """
        body = self.stage_one()
        for marker in ("utils/", "steps/", "run.pl", "utt2spk_to_spk2utt"):
            assert marker not in body, f"stage 1 still uses {marker}"
