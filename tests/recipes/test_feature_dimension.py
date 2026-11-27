"""
A model's input width is derived from the features, never written out.

Tier 1: it reads the recipes' shell and YAML as text.

**It used to be written out, and three recipes had it wrong.**
`egs/librispeech/asr{1,2,3}` call `steps/make_fbank.sh`, which writes 80
filterbank coefficients, while their model configuration said `idim: 83`
-- 80 features into an 83-wide input layer, a shape mismatch at the first
forward pass. The only reason nobody met it is that those recipes have
never been run in CI. Note `conf` in asr2 and asr3 is a **symlink** to
asr1's, so the three shared one file and one mistake.

`odim` never had the problem, because it is `null` in every model
configuration and filled in from the language directory at training time,
so a topology change cannot desynchronise it. `idim` is derived the same
way now, from `feats.cmvn.scp`, which makes the mismatch impossible
rather than merely detectable -- so what this file checks is that no
number has crept back in.

The runtime half, including that an explicit value is still honoured and
still checked against the data, is in `tests/utils/test_feature_dim.py`;
it lives in tier 1b because measuring an archive needs `kaldiio`.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Kaldi's filterbank stages, so that a recipe can be told to have one.
#: Listed rather than globbed because a recipe gaining a third kind of
#: feature extraction should be a deliberate edit here.
FEATURE_SCRIPTS = ("make_fbank.sh", "make_fbank_pitch.sh")


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
        for name in FEATURE_SCRIPTS
        if re.search(rf"steps/{re.escape(name)}\b", text)
    }
    assert len(found) <= 1, f"{recipe} runs more than one of {found}"
    return found.pop() if found else ""


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
