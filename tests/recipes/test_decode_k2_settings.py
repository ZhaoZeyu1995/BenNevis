"""
Stage 10's lexicon settings must match how its recipe prepared the lexicon.

`run/decode_k2.py` rebuilds `L_disambig` from the dictionary, so it has to
be told the same two things `prepare_lang.sh` was: the optional-silence
probability and whether phones are position-dependent. Both differ between
the two recipes:

| recipe           | --sil_prob | position-dependent |
| ---------------- | ---------- | ------------------ |
| yesno            | 0.0        | no                 |
| mini_librispeech | 0.5 (default) | yes (default)   |

Getting either wrong builds a lexicon that is **wrong rather than
broken** -- the graph composes, the decoder runs, and the transcriptions
are quietly worse. That is the failure this pins, and it is the same shape
as the token table: a difference in an argument nobody reads, surfacing as
a WER nobody can explain.

Tier 1: it reads the shell scripts as text.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Recipes with a k2 decoding stage, and what their `prepare_lang.sh` call
#: implies stage 10 must pass. Listed rather than derived, so adding a
#: recipe is a deliberate edit -- and the first test below checks the list
#: against what is actually on disc, so it cannot go stale silently.
EXPECTED = {
    "egs/yesno/asr": {"sil_prob": "0.0", "position_dependent": False},
    "egs/mini_librispeech/asr": {
        "sil_prob": "0.5",
        "position_dependent": True,
    },
}


def run_sh(recipe: str) -> str:
    return (REPO_ROOT / recipe / "run.sh").read_text(encoding="utf-8")


def stage_ten(text: str) -> str:
    """The body of stage 10, or an empty string if the recipe has none."""
    match = re.search(r"if \[ \$\{stage\} -le 10 \].*?\n(.*?)\nfi\n", text, re.DOTALL)
    return match.group(1) if match else ""


def prepare_lang_call(text: str) -> str:
    """
    The `prepare_lang.sh` invocation, joined across its line continuations.
    """
    joined = text.replace("\\\n", " ")
    for line in joined.splitlines():
        if "prepare_lang.sh" in line and not line.strip().startswith("#"):
            return line
    return ""


class TestEveryRecipeWithStageTenIsListed:
    def test_the_list_matches_what_is_on_disc(self):
        """
        A recipe that gains stage 10 without an entry here would go
        unchecked, which is the only way this test could quietly stop
        mattering.
        """
        found = {
            str(path.parent.relative_to(REPO_ROOT).as_posix())
            for path in REPO_ROOT.glob("egs/*/*/run.sh")
            if stage_ten(path.read_text(encoding="utf-8"))
        }
        assert found == set(EXPECTED), (
            f"recipes with a stage 10: {sorted(found)}; listed here: "
            f"{sorted(EXPECTED)}"
        )


@pytest.mark.parametrize("recipe", sorted(EXPECTED))
class TestTheSettingsAgree:
    def test_it_has_a_stage_ten(self, recipe):
        assert stage_ten(run_sh(recipe)), recipe

    def test_the_silence_probability_matches_prepare_lang(self, recipe):
        """
        yesno passes `--sil_prob 0.0` to `prepare_lang.sh`, so stage 10
        must too -- which it does by taking the default. mini_librispeech
        passes nothing, so `prepare_lang.sh`'s own default of 0.5 applies
        and stage 10 has to say so explicitly.
        """
        text = run_sh(recipe)
        body = stage_ten(text)
        expected = EXPECTED[recipe]["sil_prob"]
        match = re.search(r"--sil_prob\s+([0-9.]+)", body)
        given = match.group(1) if match else "0.0"
        assert given == expected, (
            f"{recipe}: stage 10 uses sil_prob {given}, but its "
            f"prepare_lang.sh call implies {expected}"
        )

    def test_position_dependence_matches_prepare_lang(self, recipe):
        body = stage_ten(run_sh(recipe))
        given = "--position_dependent_phones" in body
        assert given == EXPECTED[recipe]["position_dependent"], (
            f"{recipe}: stage 10 "
            f"{'passes' if given else 'omits'} --position_dependent_phones, "
            f"which disagrees with its prepare_lang.sh call"
        )

    def test_prepare_lang_still_says_what_we_think(self, recipe):
        """
        The other half: this reads the recipe's actual `prepare_lang.sh`
        line, so the table above is checked against the source of truth
        rather than against itself. If someone changes the preparation,
        this fails and the stage-10 flags have to follow.
        """
        call = prepare_lang_call(run_sh(recipe))
        assert call, f"{recipe} has no prepare_lang.sh call"

        explicit_zero = "--sil_prob 0.0" in call
        expected_zero = EXPECTED[recipe]["sil_prob"] == "0.0"
        assert explicit_zero == expected_zero, (
            f"{recipe}: prepare_lang.sh "
            f"{'sets' if explicit_zero else 'does not set'} --sil_prob 0.0"
        )

        turned_off = "--position-dependent-phones false" in call
        assert turned_off == (not EXPECTED[recipe]["position_dependent"]), (
            f"{recipe}: prepare_lang.sh "
            f"{'disables' if turned_off else 'leaves on'} "
            f"position-dependent phones"
        )

    def test_the_lexicon_it_rebuilds_from_is_the_one_prepared(self, recipe):
        """
        `prepare_lang.sh`'s first positional argument is the dictionary,
        and stage 10 must rebuild from the same one -- yesno's is
        `data/local/dict` and mini_librispeech's is `data/local/dict_nosp`.
        """
        text = run_sh(recipe)
        call = prepare_lang_call(text)
        dicts = re.findall(r"(data/local/dict\w*)", call)
        assert dicts, f"{recipe}: no dictionary in the prepare_lang.sh call"
        prepared = dicts[0]

        body = stage_ten(text)
        match = re.search(r"--lexicon\s+(\S+)", body)
        assert match, f"{recipe}: stage 10 passes no --lexicon"
        assert match.group(1).startswith(prepared), (
            f"{recipe}: stage 10 rebuilds from {match.group(1)} but the "
            f"lexicon was prepared from {prepared}"
        )

    def test_it_decodes_into_its_own_directory(self, recipe):
        """
        `dec_k2_` rather than `dec_`, so the Kaldi decode is not
        overwritten -- the comparison needs both.
        """
        body = stage_ten(run_sh(recipe))
        assert "dec_k2_" in body, recipe

    def test_it_stops_the_recipe_on_failure(self, recipe):
        """
        Every other stage does. A k2 decode that failed silently would
        leave the summary table missing half its rows and nothing saying
        why.
        """
        body = stage_ten(run_sh(recipe))
        assert "exit 1" in body, recipe
