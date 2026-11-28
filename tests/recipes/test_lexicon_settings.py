"""
Stage 3's lexicon settings must match how its recipe prepared the lexicon.

`prepare_topo.sh --lexicon` rebuilds `L.fst` from the dictionary in Python
and writes it as `k2/L.fst.txt`, which is what takes OpenFST off the
training path. It therefore has to be told the same two things
stage 1's `prepare-lang` is -- the optional-silence probability and whether
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

#: Recipes that pass `--lexicon`, and what their `prepare-lang` call
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


#: How stage 1 now spells the call `prepare_lang.sh` used to be. Named
#: once, because three tests across two files look for it.
PREPARE_LANG = "BenNevis.lm prepare-lang"


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
                f"settings are not checked against its prepare-lang call"
            )


@pytest.mark.parametrize("recipe", sorted(EXPECTED))
class TestTheSettingsAgree:
    def test_the_silence_probability_matches_prepare_lang(self, recipe):
        topo = call(run_sh(recipe), "prepare_topo.sh")
        match = re.search(r"--sil-prob\s+(\S+)", topo)
        assert match, f"{recipe}: stage 3 passes no --sil-prob"
        assert match.group(1) == EXPECTED[recipe]["sil_prob"], recipe

        lang = call(run_sh(recipe), PREPARE_LANG)
        given = re.search(r"--sil-prob\s+(\S+)", lang)
        # **Required rather than optional now.** Stage 1 used to be
        # `prepare_lang.sh`, whose own default of 0.5 applied when the
        # recipe passed nothing -- so this test had to encode that
        # default and mini-librispeech's setting lived in Kaldi rather
        # than in the recipe. The Python entry point is given both
        # settings explicitly, which makes the agreement readable from
        # the recipe alone and removes the implicit default that is the
        # whole shape of this trap.
        assert given, f"{recipe}: stage 1 passes no --sil-prob to prepare-lang"
        assert given.group(1) == match.group(1), (
            f"{recipe}: prepare-lang is given --sil-prob {given.group(1)} "
            f"but prepare_topo.sh is given {match.group(1)}"
        )

    def test_position_dependence_matches_prepare_lang(self, recipe):
        topo = call(run_sh(recipe), "prepare_topo.sh")
        match = re.search(r"--position-dependent-phones\s+(\S+)", topo)
        passed = bool(match) and match.group(1) == "true"
        assert passed == EXPECTED[recipe]["position_dependent"], recipe

        lang = call(run_sh(recipe), PREPARE_LANG)
        given = re.search(r"--position-dependent-phones\s+(\S+)", lang)
        # Explicit in stage 1, for the reason above. Position dependence
        # is why mini-librispeech's lexicon says AH0 and its phones.txt
        # holds only AH0_B, AH0_E, AH0_I and AH0_S.
        assert given, (
            f"{recipe}: stage 1 passes no --position-dependent-phones to "
            f"prepare-lang"
        )
        expected = given.group(1) != "false"
        assert passed == expected, (
            f"{recipe}: prepare-lang implies position-dependent="
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

        lang = call(run_sh(recipe), PREPARE_LANG)
        fields = [f for f in lang.split() if not f.startswith("-")]
        prepared = next((f for f in fields if "dict" in f), None)
        assert prepared, f"{recipe}: cannot find prepare-lang's dictionary"
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
    """
    The defaults and the plumbing, now in `prepare_topo.py`.

    These used to read `prepare_topo.sh` as shell text. The logic moved to
    Python when stage 3 stopped needing Kaldi's `utils/parse_options.sh`
    to read its own flags, so they read the Python -- and by calling it
    rather than by matching its source, which is the stronger check and
    was not available while it was a shell script.
    """

    @staticmethod
    def defaults():
        from BenNevis.run.prepare_topo import parse

        return parse(["--topo", "ctc", "src", "des"])

    def test_the_defaults_are_the_conservative_ones(self):
        """
        Not `prepare-lang`'s, which are `prepare_lang.sh`'s own: 0.5 and
        position-dependent; these are 0.0 and not, so a recipe that
        forgets to pass them gets a loud mismatch against its
        `phones.txt` rather than a quietly different lexicon.
        """
        args = self.defaults()
        assert args.sil_prob == 0.0
        assert args.position_dependent_phones == "false"
        assert args.lexicon is None

    def test_the_fsts_are_compiled_only_when_available(self):
        assert self.defaults().compile_fsts == "auto"

    def test_the_booleans_take_kaldis_spellings(self):
        """
        `--position-dependent-phones true`, not a bare flag: the shim
        passes a recipe's existing command line through untouched and that
        is what the recipes write.
        """
        from BenNevis.run.prepare_topo import parse

        args = parse(
            [
                "--topo",
                "ctc",
                "--position-dependent-phones",
                "true",
                "src",
                "des",
            ]
        )
        assert args.position_dependent_phones == "true"
        with pytest.raises(SystemExit):
            parse(["--topo", "ctc", "--position-dependent-phones", "yes", "s", "d"])

    def test_the_lexicon_text_is_written_only_when_asked_for(self):
        source = (REPO_ROOT / "BenNevis" / "run" / "prepare_topo.py").read_text(
            encoding="utf-8"
        )
        assert "k2/L.fst.txt" in source or "L.fst.txt" in source
        assert "if args.lexicon is not None:" in source

    def test_it_says_so_when_it_does_not(self):
        """
        Silence here would mean a recipe quietly kept needing OpenFST.
        """
        source = (REPO_ROOT / "BenNevis" / "run" / "prepare_topo.py").read_text(
            encoding="utf-8"
        )
        tail = source[source.index("if args.lexicon is not None:") :]
        assert "else:" in tail and "fstprint" in tail

    def test_both_settings_reach_the_builder(self):
        """
        A recipe passing `--position-dependent-phones true` is worth
        nothing if the script does not hand it on, and in the shell version
        a mutation removing `$pdp` from the command line passed every other
        test here. In Python it is an argument rather than a string, so it
        is checked by calling the function.
        """
        from BenNevis.run.prepare_topo import write_lexicon_text

        source = (REPO_ROOT / "BenNevis" / "run" / "prepare_topo.py").read_text(
            encoding="utf-8"
        )
        body = source[source.index("def write_lexicon_text") :]
        body = body[: body.index("\ndef ")]
        assert "sil_prob=args.sil_prob" in body
        assert 'args.position_dependent_phones == "true"' in body
        assert write_lexicon_text is not None
