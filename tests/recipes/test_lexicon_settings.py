"""
Stage 3's lexicon settings must match how its recipe prepared the lexicon.

`prepare_topo.sh --lexicon` rebuilds `L.fst` from the dictionary in Python
and writes it as `k2/L.fst.txt`, which is what takes OpenFST off the
training path. It therefore has to be told the same two things
`prepare_lang.sh` was -- the optional-silence probability and whether
phones are position-dependent -- and both differ between the two recipes:

| recipe           | --sil-prob    | position-dependent |
| ---------------- | ------------- | ------------------ |
| yesno            | 0.0           | no                 |
| mini_librispeech | 0.5 (default) | yes (default)      |

This is the same trap as stage 10's, one stage earlier and with a worse
blast radius: stage 10 getting it wrong makes one decode quietly worse,
while stage 3 getting it wrong makes every **training** run compose
against a lexicon that is wrong rather than broken. The graph composes, the
loss falls, and the model is trained against pronunciations the dictionary
does not have.

Tier 1: it reads the shell scripts as text.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Recipes that pass `--lexicon`, and what their `prepare_lang.sh` call
#: implies stage 3 must pass with it. `prepare_topo.sh` is called by seven
#: recipes and the flag is optional, because the other five have not had
#: their settings checked -- they keep the `fstprint` fallback, which
#: warns. The first test below pins which recipes are in which group, so
#: converting one is a deliberate edit and forgetting to is visible.
EXPECTED = {
    "egs/yesno/asr": {
        "lexicon": "data/local/dict/lexicon.txt",
        "sil_prob": "0.0",
        "position_dependent": False,
    },
    "egs/mini_librispeech/asr": {
        "lexicon": "data/local/dict_nosp/lexicon.txt",
        "sil_prob": "0.5",
        "position_dependent": True,
    },
}


def run_sh(recipe: str) -> str:
    return (REPO_ROOT / recipe / "run.sh").read_text(encoding="utf-8")


def joined(text: str) -> str:
    """The script with its line continuations folded away."""
    return text.replace("\\\n", " ")


def call(text: str, script: str) -> str:
    """The invocation of `script`, or an empty string."""
    for line in joined(text).splitlines():
        if script in line and not line.strip().startswith("#"):
            return line
    return ""


def recipes_calling_prepare_topo():
    found = []
    for path in sorted((REPO_ROOT / "egs").glob("*/*/run.sh")):
        if "prepare_topo.sh" in path.read_text(encoding="utf-8"):
            found.append(path.parent.relative_to(REPO_ROOT).as_posix())
    return found


class TestTheListMatchesWhatIsOnDisc:
    def test_every_listed_recipe_exists_and_passes_the_lexicon(self):
        for recipe in EXPECTED:
            assert (REPO_ROOT / recipe / "run.sh").is_file(), recipe
            assert "--lexicon" in call(run_sh(recipe), "prepare_topo.sh"), recipe

    def test_the_recipes_left_out_are_named_rather_than_forgotten(self):
        """
        A recipe that calls `prepare_topo.sh` without `--lexicon` still
        works and still needs OpenFST once. That is a deliberate gap, so
        it is asserted rather than left to be discovered -- and this test
        fails when one is converted without being listed, which is the
        direction that matters.
        """
        on_disc = set(recipes_calling_prepare_topo())
        assert set(EXPECTED) <= on_disc, set(EXPECTED) - on_disc
        for recipe in sorted(on_disc - set(EXPECTED)):
            assert "--lexicon" not in call(run_sh(recipe), "prepare_topo.sh"), (
                f"{recipe} passes --lexicon but is not in EXPECTED, so its "
                f"settings are not checked against its prepare_lang.sh call"
            )


@pytest.mark.parametrize("recipe", sorted(EXPECTED))
class TestTheSettingsAgree:
    def test_the_silence_probability_matches_prepare_lang(self, recipe):
        topo = call(run_sh(recipe), "prepare_topo.sh")
        match = re.search(r"--sil-prob\s+(\S+)", topo)
        assert match, f"{recipe}: stage 3 passes no --sil-prob"
        assert match.group(1) == EXPECTED[recipe]["sil_prob"], recipe

        lang = call(run_sh(recipe), "prepare_lang.sh")
        given = re.search(r"--sil[-_]prob\s+(\S+)", lang)
        if given:
            assert given.group(1) == match.group(1), (
                f"{recipe}: prepare_lang.sh was given --sil_prob "
                f"{given.group(1)} but prepare_topo.sh is given "
                f"{match.group(1)}"
            )
        else:
            # prepare_lang.sh's own default is 0.5.
            assert match.group(1) == "0.5", (
                f"{recipe}: prepare_lang.sh takes its default --sil_prob of "
                f"0.5 but prepare_topo.sh is given {match.group(1)}"
            )

    def test_position_dependence_matches_prepare_lang(self, recipe):
        topo = call(run_sh(recipe), "prepare_topo.sh")
        match = re.search(r"--position-dependent-phones\s+(\S+)", topo)
        passed = bool(match) and match.group(1) == "true"
        assert passed == EXPECTED[recipe]["position_dependent"], recipe

        lang = call(run_sh(recipe), "prepare_lang.sh")
        given = re.search(r"--position-dependent-phones\s+(\S+)", lang)
        # Not passing it to prepare_lang.sh means position-dependent, which
        # is why mini-librispeech's lexicon says AH0 and its phones.txt
        # holds only AH0_B, AH0_E, AH0_I and AH0_S.
        expected = True if not given else given.group(1) != "false"
        assert passed == expected, (
            f"{recipe}: prepare_lang.sh implies position-dependent="
            f"{expected} but prepare_topo.sh is given {passed}"
        )

    def test_it_rebuilds_from_the_lexicon_prepare_lang_read(self, recipe):
        """
        Not from the recipe's input copy. yesno's `prepare_dict.sh` copies
        `input/lexicon.txt` into `data/local/dict/`, so the two agree today
        and would diverge the moment a preparation step transformed the
        lexicon.
        """
        topo = call(run_sh(recipe), "prepare_topo.sh")
        match = re.search(r"--lexicon\s+(\S+)", topo)
        assert match, recipe
        assert match.group(1) == EXPECTED[recipe]["lexicon"], recipe

        lang = call(run_sh(recipe), "prepare_lang.sh")
        fields = [f for f in lang.split() if not f.startswith("-")]
        prepared = next((f for f in fields if "dict" in f), None)
        assert prepared, f"{recipe}: cannot find prepare_lang.sh's dictionary"
        assert match.group(1).startswith(prepared.rstrip("/")), (
            f"{recipe}: stage 3 rebuilds from {match.group(1)} but the "
            f"lexicon was prepared from {prepared}"
        )

    def test_stage_three_and_stage_ten_agree_with_each_other(self, recipe):
        """
        Both rebuild a lexicon FST from the dictionary, so they must be
        given the same settings -- and they are written out in two places,
        four lines apart in one file and two hundred in the other.
        """
        text = run_sh(recipe)
        topo = call(text, "prepare_topo.sh")
        decode = call(text, "decode_k2.py")
        if not decode:
            pytest.skip(f"{recipe} has no stage 10")
        topo_sil = re.search(r"--sil-prob\s+(\S+)", topo)
        decode_sil = re.search(r"--sil_prob\s+(\S+)", decode)
        topo_value = topo_sil.group(1) if topo_sil else "0.0"
        decode_value = decode_sil.group(1) if decode_sil else "0.0"
        assert float(topo_value) == float(decode_value), recipe

        topo_pdp = re.search(r"--position-dependent-phones\s+(\S+)", topo)
        assert (bool(topo_pdp) and topo_pdp.group(1) == "true") == (
            "--position_dependent_phones" in decode
        ), recipe


class TestPrepareTopoItself:
    @staticmethod
    def script() -> str:
        return (REPO_ROOT / "BenNevis" / "run" / "prepare_topo.sh").read_text(
            encoding="utf-8"
        )

    def test_the_flag_defaults_match_the_python_defaults(self):
        """
        `prepare_topo.sh` defaults `--sil-prob` to 0.0 and
        position-dependence to false, which is **not**
        `prepare_lang.sh`'s default -- it is the conservative one, so a
        recipe that forgets to pass them gets a loud mismatch against its
        `phones.txt` rather than a quietly different lexicon.
        """
        text = self.script()
        assert re.search(r"^sil_prob=0\.0\s*$", text, re.M)
        assert re.search(r"^position_dependent_phones=false\s*$", text, re.M)
        assert re.search(r"^lexicon=\s*$", text, re.M)

    def test_it_writes_the_text_only_when_told_to(self):
        text = self.script()
        assert "k2/L.fst.txt" in text
        assert 'if [ -n "$lexicon" ]' in text

    def test_it_says_so_when_it_does_not(self):
        """
        Silence here would mean a recipe quietly kept needing OpenFST.
        """
        text = self.script()
        tail = text[text.index('if [ -n "$lexicon" ]') :]
        assert "else" in tail and "fstprint" in tail

    def test_it_fails_rather_than_writing_an_empty_lexicon(self):
        text = self.script()
        assert "does not exist" in text
        assert "failed to build" in text

    def test_it_forwards_both_settings_to_the_builder(self):
        """
        A recipe passing `--position-dependent-phones true` is worth
        nothing if the script does not hand it on, and dropping it is
        invisible from the recipe's side -- a mutation that removed `$pdp`
        from the command line passed every other test here.

        Checked on the invocation itself rather than on the whole file, so
        the flag has to reach `BenNevis.lm lexicon-fst` and not merely
        appear somewhere in a comment.
        """
        text = self.script().replace("\\\n", " ")
        call = next(
            line
            for line in text.splitlines()
            if "BenNevis.lm lexicon-fst" in line and not line.strip().startswith("#")
        )
        assert "--sil-prob $sil_prob" in call, call
        assert "$pdp" in call, call
        # And `$pdp` has to be set from the flag, not left empty.
        assert '--position-dependent-phones"' in text
        assert 'if [ "$position_dependent_phones" == "true" ]' in text
