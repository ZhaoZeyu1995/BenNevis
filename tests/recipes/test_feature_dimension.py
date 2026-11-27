"""
A recipe's configured `idim` must match the features its own stage makes.

Tier 1: it reads the recipe's shell and YAML as text.

**This was already wrong in three recipes and nothing noticed.**
`egs/librispeech/asr{1,2,3}` call `steps/make_fbank.sh`, which writes 80
filterbank coefficients and no pitch, while their model configuration said
`idim: 83`. A run would have died at the first forward pass on a shape
mismatch -- 80 features into an 83-wide input layer -- and the only reason
nobody met it is that those recipes have never been run in CI. Dropping
pitch fixed them as a side effect, which is exactly why the fix is worth
less than the test.

The asymmetry that allows it is worth naming: `odim` is `null` in every
model configuration and derived from the language directory at training
time, so a topology change cannot desynchronise it. `idim` is written out
by hand. Deriving it from the features would make this class of mistake
impossible, and that is a change to the configuration contract rather
than a test -- so it is recorded here and in CLAUDE.md instead of done
quietly.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: How many dimensions each feature extractor adds beyond the filterbank.
#: `make_fbank_pitch.sh` appends the three that
#: `process-kaldi-pitch-feats` produces -- the POV feature, the normalised
#: log pitch and the delta pitch. No recipe uses it any more, and it is
#: listed so that a recipe going back to it stays consistent rather than
#: silently wrong.
EXTRA_DIMENSIONS = {"make_fbank.sh": 0, "make_fbank_pitch.sh": 3}

#: Kaldi's own default, for a recipe whose `fbank.conf` does not say.
DEFAULT_MEL_BINS = 23


def recipes():
    return sorted(
        path.parent.relative_to(REPO_ROOT).as_posix()
        for path in REPO_ROOT.glob("egs/*/*/run.sh")
    )


def feature_script(recipe: str) -> str:
    """Which of Kaldi's feature scripts the recipe's stage 2 runs."""
    text = (REPO_ROOT / recipe / "run.sh").read_text(encoding="utf-8")
    found = {
        name
        for name in EXTRA_DIMENSIONS
        if re.search(rf"steps/{re.escape(name)}\b", text)
    }
    assert len(found) <= 1, f"{recipe} runs more than one of {found}"
    return found.pop() if found else ""


def mel_bins(recipe: str) -> int:
    path = REPO_ROOT / recipe / "conf" / "fbank.conf"
    if not path.is_file():
        return DEFAULT_MEL_BINS
    match = re.search(r"^--num-mel-bins=(\d+)", path.read_text(encoding="utf-8"), re.M)
    return int(match.group(1)) if match else DEFAULT_MEL_BINS


def configured_idim(recipe: str):
    """
    Every `idim` in the recipe's model configurations, by file.

    `None` for `idim: null`, which is the value that means "derive it", so
    a config writing a number out is distinguishable from one that does
    not mention `idim` at all. Matching `(\\d+)` here instead -- which is
    what this did first -- makes every case *skip* once the configs say
    `null`, and a test that skips everywhere passes while checking
    nothing.
    """
    out = {}
    for path in sorted((REPO_ROOT / recipe / "conf" / "model").glob("*.yaml")):
        match = re.search(r"^\s+idim:\s*(\S+)", path.read_text(encoding="utf-8"), re.M)
        if match:
            raw = match.group(1)
            out[path.relative_to(REPO_ROOT).as_posix()] = (
                None if raw == "null" else int(raw)
            )
    return out


@pytest.mark.parametrize("recipe", recipes())
def test_idim_is_derived_rather_than_written_out(recipe):
    """
    `idim: null`, the same contract `odim` has.

    It used to be a number, and three recipes had it wrong --
    `egs/librispeech/asr{1,2,3}` said 83 while running
    `steps/make_fbank.sh`, which writes 80. Deriving it from the features
    makes that impossible rather than merely detectable, which is why this
    test replaced one that compared the number against
    `--num-mel-bins`: there is no number left to compare.

    `run/train.py` still honours an explicit value and checks it against
    the data, so pinning a dimension stays possible -- it just cannot be
    silently wrong. That check is tested in `tests/utils/test_feature_dim.py`.
    """
    configured = configured_idim(recipe)
    if not configured:
        pytest.skip(f"{recipe} configures no idim")
    for path, value in configured.items():
        assert value is None, (
            f"{path} writes idim out as {value}. It is derived from the "
            f"features at training time; a number here can disagree with "
            f"them, which is what it did in three recipes."
        )


def test_the_idim_check_actually_reads_something():
    """
    Because it just did not. The reader matched `(\\d+)`, which stopped
    matching when the configs became `idim: null`, so all seven cases
    skipped and the suite passed green while checking nothing -- the same
    shape as the `G.fst` gate in #41.
    """
    found = {r: configured_idim(r) for r in recipes()}
    with_idim = {r: v for r, v in found.items() if v}
    assert len(with_idim) >= 6, f"only {sorted(with_idim)} were read"


def test_every_recipe_was_checked_or_skipped_for_a_stated_reason():
    """
    So the parametrisation cannot quietly cover nothing -- the failure mode
    the `G.fst` gate in #41 had, where a path was built wrongly and the
    test skipped unconditionally.
    """
    with_features = [r for r in recipes() if feature_script(r)]
    assert len(with_features) >= 6, with_features


def test_no_recipe_extracts_pitch_any_more():
    """
    Pitch was a GMM-era addition for tonal languages; these corpora are
    English and Hebrew, and every modern end-to-end recipe uses plain
    filterbank or the waveform. Pinned because going back to it silently
    would change every model's input width and re-anchor the WER ledger
    without saying so.
    """
    using = [r for r in recipes() if feature_script(r) == "make_fbank_pitch.sh"]
    assert not using, f"{using} extract pitch; see CLAUDE.md on why it went"


def test_no_recipe_still_carries_a_pitch_configuration():
    """
    `conf/pitch.conf` outlived its use in `egs/librispeech/asr*`, which
    had already moved to `make_fbank.sh` -- a dead configuration file that
    made the `idim: 83` above look deliberate.
    """
    stale = sorted(
        p.relative_to(REPO_ROOT).as_posix()
        for p in REPO_ROOT.glob("egs/*/*/conf/pitch.conf")
    )
    assert not stale, stale


def test_odim_is_derived_rather_than_written_out():
    """
    The counterpart to `idim`, and the reason this test exists: `odim` is
    `null` and filled in from the language directory, so a topology change
    cannot desynchronise it. Asserted so that nobody "fixes" it by writing
    a number in.
    """
    checked = 0
    for recipe in recipes():
        for path in sorted((REPO_ROOT / recipe / "conf" / "model").glob("*.yaml")):
            text = path.read_text(encoding="utf-8")
            if "odim" not in text:
                continue
            checked += 1
            assert re.search(r"^\s+odim:\s*null", text, re.M), (
                f"{path.relative_to(REPO_ROOT)} writes odim out; it is "
                f"determined by the language directory at training time"
            )
    assert checked >= 6, checked
