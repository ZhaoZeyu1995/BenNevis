"""
Every recipe's stage 2 calls the Python extractor, not Kaldi's.

Tier 1: this reads `run.sh` as text.

The point is not that the Python exists -- `run/make_fbank.py` has
existed since #102 while every recipe went on calling
`steps/make_fbank.sh`, which is exactly the gap this file closes. A
replacement nothing calls is not a replacement, and the way that state
persists is by nobody asserting the call site.

**Recipes are discovered rather than listed**, the lesson
`tests/reference/test_corpora.py` and `tests/recipes/test_ledger.py`
both carry: a new recipe, or a converted one left off a list, would
otherwise be unchecked. `egs/librispeech` and `egs/timit` have never run
in CI, which is precisely why their call sites want a cheap test -- the
`idim: 83` mismatch in those three recipes survived for the same reason.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: What stage 2 must no longer invoke. `make_fbank_pitch.sh` is here as
#: well as `make_fbank.sh` because pitch was dropped: a recipe reaching
#: for it would silently be 83 dimensions again.
KALDI_STAGE2 = (
    "steps/make_fbank.sh",
    "steps/make_fbank_pitch.sh",
    "steps/compute_cmvn_stats.sh",
    "utils/fix_data_dir.sh",
    "apply-cmvn",
    "compute-cmvn-stats",
)

#: What it must invoke instead. `apply_cmvn` is matched without its
#: extension because the shim and the script are both acceptable.
PYTHON_STAGE2 = (
    "run/make_fbank.py",
    "run/compute_cmvn_stats.py",
    "run/fix_data_dir.py",
    "run/apply_cmvn",
)


def recipes():
    return sorted(
        path.parent.relative_to(REPO_ROOT).as_posix()
        for path in REPO_ROOT.glob("egs/*/*/run.sh")
    )


def extracts_features(recipe):
    text = (REPO_ROOT / recipe / "run.sh").read_text(encoding="utf-8")
    return "make_fbank" in text


def feature_recipes():
    return [recipe for recipe in recipes() if extracts_features(recipe)]


def script(recipe):
    return (REPO_ROOT / recipe / "run.sh").read_text(encoding="utf-8")


def test_there_are_recipes_to_check():
    """
    A parametrised suite over an empty list passes, which is how the
    `G.fst` gate in #41 came to be a test that could not fail.
    """
    assert feature_recipes(), "no recipe extracts features"


@pytest.mark.parametrize("recipe", feature_recipes())
class TestStageTwoIsPython:
    def test_it_calls_no_kaldi_feature_script(self, recipe):
        text = script(recipe)
        for name in KALDI_STAGE2:
            assert name not in text, f"{recipe} still calls {name}"

    def test_it_calls_every_python_entry_point(self, recipe):
        text = script(recipe)
        for name in PYTHON_STAGE2:
            assert name in text, f"{recipe} does not call {name}"

    def test_the_entry_points_exist_and_are_executable(self, recipe):
        """
        A recipe invokes these by path, so a missing execute bit fails
        the stage after the corpus has been downloaded -- which is what
        `run/make_fbank.py` was missing when this test was written.

        The bit itself is only checked off Windows, where there is none
        and `st_mode` says nothing; the recipes are POSIX-only anyway.
        """
        for name in PYTHON_STAGE2:
            candidates = [
                REPO_ROOT / "BenNevis" / (name + suffix)
                for suffix in ("", ".py", ".sh")
            ]
            found = [path for path in candidates if path.is_file()]
            assert found, f"{name} does not exist"
            if os.name != "nt":
                assert found[0].stat().st_mode & 0o111, f"{found[0]} is not executable"

    def test_it_reads_the_recipes_own_fbank_configuration(self, recipe):
        """
        `--conf conf/fbank.conf` is what stops the two extractors being
        given different settings -- it is the same file Kaldi's reads.
        Without it the Python falls back to Kaldi's defaults, which are
        not this recipe's: `num_mel_bins` would be 23 rather than 80.
        """
        assert "--conf conf/fbank.conf" in script(recipe), recipe
        assert (REPO_ROOT / recipe / "conf" / "fbank.conf").is_file()

    def test_it_compresses_the_archive(self, recipe):
        """
        `make_fbank.sh` wrote compressed archives, so `--compress` is
        what keeps the data the trainer reads like for like across the
        switch. Writing uncompressed would change CMVN's inputs and hence
        every WER in the ledger -- a measured decision of the same kind
        as dropping pitch, and not one to take by deleting a flag.
        """
        assert "--compress" in script(recipe), recipe

    def test_it_writes_the_frame_counts(self, recipe):
        """
        The duration-based samplers batch on `utt2num_frames`; without it
        they fall back to reading every matrix.
        """
        assert "--write-utt2num-frames" in script(recipe), recipe

    def test_it_does_not_pass_a_job_count_to_apply_cmvn(self, recipe):
        """
        `--nj` is gone rather than ignored, so a recipe still passing it
        would fail with an argument error. Measured: the whole of
        mini-librispeech's train_clean_5 normalises in under a second.
        """
        for line in script(recipe).splitlines():
            if "apply_cmvn" in line:
                assert "--nj" not in line, f"{recipe}: {line.strip()}"
