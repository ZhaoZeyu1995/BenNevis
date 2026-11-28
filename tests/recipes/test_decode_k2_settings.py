"""
Stage 10's lexicon settings must match how its recipe prepared the lexicon.

`run/decode_k2.py` rebuilds `L_disambig` from the dictionary, so it has to
be told the same two things stage 1's `prepare-lang` is: the optional-silence
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

#: Recipes with a k2 decoding stage, and what their `prepare-lang` call
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
    Stage 1's `prepare-lang` invocation, joined across its continuations.

    `utils/prepare_lang.sh` until stage 1 became Python; the settings are
    the same two and are now stated explicitly rather than falling back
    to a Kaldi default.
    """
    joined = text.replace("\\\n", " ")
    for line in joined.splitlines():
        if "BenNevis.lm prepare-lang" in line and not line.strip().startswith("#"):
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
        yesno passes `--sil-prob 0.0` to `prepare-lang`, so stage 10
        must too -- which it does by taking the default.
        mini_librispeech passes 0.5, so stage 10 has to say so
        explicitly. Both are explicit in stage 1 now; while stage 1 was
        `prepare_lang.sh`, mini-librispeech's 0.5 came from that
        script's own default and so lived in Kaldi rather than in the
        recipe.
        """
        text = run_sh(recipe)
        body = stage_ten(text)
        expected = EXPECTED[recipe]["sil_prob"]
        match = re.search(r"--sil_prob\s+([0-9.]+)", body)
        given = match.group(1) if match else "0.0"
        assert given == expected, (
            f"{recipe}: stage 10 uses sil_prob {given}, but its "
            f"prepare-lang call implies {expected}"
        )

    def test_position_dependence_matches_prepare_lang(self, recipe):
        body = stage_ten(run_sh(recipe))
        given = "--position_dependent_phones" in body
        assert given == EXPECTED[recipe]["position_dependent"], (
            f"{recipe}: stage 10 "
            f"{'passes' if given else 'omits'} --position_dependent_phones, "
            f"which disagrees with its prepare-lang call"
        )

    def test_prepare_lang_still_says_what_we_think(self, recipe):
        """
        The other half: this reads the recipe's actual `prepare-lang`
        line, so the table above is checked against the source of truth
        rather than against itself. If someone changes the preparation,
        this fails and the stage-10 flags have to follow.
        """
        call = prepare_lang_call(run_sh(recipe))
        assert call, f"{recipe} has no prepare-lang call"

        sil = re.search(r"--sil-prob\s+(\S+)", call)
        assert sil, f"{recipe}: stage 1 passes no --sil-prob"
        assert sil.group(1) == EXPECTED[recipe]["sil_prob"], (
            f"{recipe}: prepare-lang is given --sil-prob {sil.group(1)} "
            f"where this table says {EXPECTED[recipe]['sil_prob']}"
        )

        position = re.search(r"--position-dependent-phones\s+(\S+)", call)
        assert position, f"{recipe}: stage 1 passes no --position-dependent-phones"
        assert (position.group(1) == "true") == EXPECTED[recipe][
            "position_dependent"
        ], (
            f"{recipe}: prepare-lang is given position-dependent-phones "
            f"{position.group(1)} where this table says "
            f"{EXPECTED[recipe]['position_dependent']}"
        )

    def test_the_lexicon_it_rebuilds_from_is_the_one_prepared(self, recipe):
        """
        `prepare-lang`'s first positional argument is the dictionary,
        and stage 10 must rebuild from the same one -- yesno's is
        `data/local/dict` and mini_librispeech's is `data/local/dict_nosp`.
        """
        text = run_sh(recipe)
        call = prepare_lang_call(text)
        dicts = re.findall(r"(data/local/dict\w*)", call)
        assert dicts, f"{recipe}: no dictionary in the prepare-lang call"
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


class TestTheTwoSearchesArePrunedTheSame:
    """
    Stage 10's token-passing defaults must be `decode_faster.sh`'s.

    The two stages exist to compare *searches* on one checkpoint, so every
    other knob has to agree or the comparison measures the settings
    instead. `decode_faster.sh` sets the beam to 32 and `max_active` to
    5000, and `BenNevis.decode.viterbi` takes the same two parameters with
    the same meaning -- so the defaults in `run/decode_k2.py` are read out
    of the shell script here rather than restated.

    This does not apply to `--search k2`: `intersect_dense_pruned` prunes
    with two beams and a floor and a cap on active *states*, which do not
    correspond to Kaldi's one beam and one cap, and pretending they did is
    what the module docstring warns against.
    """

    @staticmethod
    def shell_default(name: str) -> str:
        text = (REPO_ROOT / "BenNevis" / "run" / "decode_faster.sh").read_text(
            encoding="utf-8"
        )
        match = re.search(rf"^{name}=['\"]?([0-9.]+)['\"]?\s*$", text, re.M)
        assert match, f"decode_faster.sh no longer sets {name}"
        return match.group(1)

    @staticmethod
    def python_default(name: str) -> str:
        text = (REPO_ROOT / "BenNevis" / "run" / "decode_k2.py").read_text(
            encoding="utf-8"
        )
        match = re.search(
            rf'add_argument\(\s*"--{name}",\s*type=\w+,\s*default=([0-9.]+)',
            text,
        )
        assert match, f"decode_k2.py no longer has a --{name} default"
        return match.group(1)

    def test_the_beam_matches(self):
        assert float(self.python_default("beam")) == float(self.shell_default("beam"))

    def test_max_active_matches(self):
        assert int(float(self.python_default("max_active"))) == int(
            float(self.shell_default("max_active"))
        )

    def test_the_acoustic_scale_matches(self):
        assert float(self.python_default("acoustic_scale")) == float(
            self.shell_default("acoustic_scale")
        )

    def test_token_passing_is_the_default_search(self):
        """
        Pinned because the reason is a measurement, not a preference:
        `intersect_dense_pruned` lost the mini-librispeech comparison three
        runs in a row, the last by SIGKILL at 14,988 MiB with 1,000 of
        1,089 utterances decoded.
        """
        text = (REPO_ROOT / "BenNevis" / "run" / "decode_k2.py").read_text(
            encoding="utf-8"
        )
        match = re.search(
            r'add_argument\(\s*"--search",.*?default="([a-z0-9]+)"', text, re.S
        )
        assert match and match.group(1) == "viterbi"
