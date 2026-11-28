"""
Stage 1 builds its language directory in Python, where it is converted.

Tier 1: it reads `run.sh` as text.

**Only the recipes whose settings have been checked are converted**, and
that is the point of this file rather than an omission. `prepare-lang`
has to be given the same `--sil-prob` and `--position-dependent-phones`
that `prepare_lang.sh` was, and getting them wrong builds a lexicon that
is *wrong rather than broken*: the tables come out, the FSTs compose, and
every transcription is quietly worse. yesno and mini-librispeech have
had theirs checked against their own calls; `egs/librispeech/asr{1,2,3}`
and `egs/timit/asr_{char,phn}` have not, and keep Kaldi's script.

So the invariant here is a **coupling** rather than a list: a recipe
converted at stage 1 is exactly a recipe converted at stage 3, because
both rebuild a lexicon from the dictionary and both need the same two
settings. `tests/recipes/test_lexicon_settings.py` is what checks the
values agree; this checks the two sets of recipes are the same set, so
converting one stage without the other fails here.

The other thing pinned is that **the language model left stage 1**.
`local/prepare_lm.sh` and `local/format_lms.sh` ran there and need
`arpa2fst`, `fstcompile`, `fstcompose` and `fstinfo` for an artefact only
decoding reads -- so stage 1 needed OpenFST for something training never
touches. They run in stage 4 now, beside the graphs they feed.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PREPARE_LANG = "BenNevis.lm prepare-lang"

#: The LM formatting scripts, which must not be in stage 1.
#: `run/format_lm.sh` is ours, written because Kaldi's `format_lms.sh`
#: ends in `validate_lang.pl` and that validates a GMM language
#: directory; see that script's header.
LM_SCRIPTS = (
    "local/prepare_lm.sh",
    "local/format_lms.sh",
    "run/format_lm.sh",
)


def recipes():
    return sorted(
        path.parent.relative_to(REPO_ROOT).as_posix()
        for path in REPO_ROOT.glob("egs/*/*/run.sh")
    )


def script(recipe):
    return (REPO_ROOT / recipe / "run.sh").read_text(encoding="utf-8")


def joined(recipe):
    """The script with its line continuations folded away."""
    return script(recipe).replace("\\\n", " ")


def stage(recipe, number):
    """
    The commands of one stage, with comments stripped.

    Read from the guard expression rather than from the `echo`, because
    the guard is what decides what runs.

    **Comments are stripped, and that is not tidiness.** The first
    version kept them, and a comment reading "run/format_lm.sh, not
    local/format_lms.sh" made `test_it_moved_rather_than_vanished` pass
    for a stage that no longer ran `local/format_lms.sh` at all -- the
    test matched the explanation instead of the command. Same family as
    `"G.fst" in name` matching `LG.fst.txt` and `--compile-fsts` reading
    as a binary called `fsts`: a scan for a command name must look at
    the commands.
    """
    match = re.search(
        rf"if \[ \$\{{stage\}} -le {number} \].*?\n(.*?)\nfi\n",
        joined(recipe),
        re.DOTALL,
    )
    if not match:
        return ""
    return "\n".join(
        line for line in match.group(1).splitlines() if not line.strip().startswith("#")
    )


#: `prepare-lang`'s flags that take a value, so a positional scan can
#: step over them. Listed rather than inferred: a flag added here without
#: being listed would be counted as a positional and read as a path.
VALUED_FLAGS = ("--sil-prob", "--position-dependent-phones", "--compile-fsts")


def call_arguments(recipe):
    """
    The tokens `prepare-lang` is given, up to the shell's `||` or `;`.

    The joined line carries the recipe's error handling -- `|| { echo ...
    exit 1; }` -- and scanning that as arguments is how the first version
    of this test counted eleven positionals.
    """
    line = next(line for line in joined(recipe).splitlines() if PREPARE_LANG in line)
    tail = line.split(PREPARE_LANG, 1)[1]
    for terminator in ("||", ";", "&&"):
        tail = tail.split(terminator, 1)[0]
    return tail.split()


def strip_flags(arguments):
    """The positional arguments, stepping over each flag and its value."""
    out = []
    skip = False
    for token in arguments:
        if skip:
            skip = False
            continue
        if token in VALUED_FLAGS:
            skip = True
            continue
        if token.startswith("-"):
            continue
        out.append(token)
    return out


def positional_arguments(recipe):
    return len(strip_flags(call_arguments(recipe)))


def converted_at_stage_one():
    return [recipe for recipe in recipes() if PREPARE_LANG in script(recipe)]


def converted_at_stage_three():
    """
    Recipes whose `prepare_topo.sh` call rebuilds the lexicon in Python,
    which is what `--lexicon` means.
    """
    return [recipe for recipe in recipes() if "--lexicon" in joined(recipe)]


def test_some_recipe_is_converted():
    """
    A parametrised suite over an empty list passes, which is how the
    `G.fst` gate in #41 came to be a test that could not fail.
    """
    assert converted_at_stage_one(), "no recipe prepares its lang in Python"


def test_the_two_conversions_cover_the_same_recipes():
    """
    The coupling, and the reason this file is not a list. Both stages
    rebuild a lexicon from the dictionary and both need the same two
    settings, so a recipe converted at one and not the other has its
    settings checked at one end only -- which is how a *wrong rather
    than broken* lexicon gets in.
    """
    assert converted_at_stage_one() == converted_at_stage_three(), (
        f"stage 1 converted: {converted_at_stage_one()}; stage 3 "
        f"converted: {converted_at_stage_three()}"
    )


@pytest.mark.parametrize("recipe", converted_at_stage_one())
class TestAConvertedRecipe:
    def test_it_does_not_call_kaldis_prepare_lang(self, recipe):
        assert "utils/prepare_lang.sh" not in script(recipe), recipe

    def test_the_call_is_in_stage_one(self, recipe):
        """
        Not merely somewhere in the file. A language directory built in a
        later stage would be built after the features that depend on
        nothing and before nothing that needs it.
        """
        assert PREPARE_LANG in stage(recipe, 1), recipe

    def test_it_states_both_settings_explicitly(self, recipe):
        """
        `prepare-lang` defaults to `prepare_lang.sh`'s own 0.5 and
        position-dependent, so a recipe could leave them out and be
        correct today. Stating them keeps the agreement readable from the
        recipe alone, which is what
        `tests/recipes/test_lexicon_settings.py` checks -- and it is the
        implicit default that made mini-librispeech's silence probability
        live in Kaldi rather than in its own recipe.
        """
        line = next(
            line for line in joined(recipe).splitlines() if PREPARE_LANG in line
        )
        assert re.search(r"--sil-prob\s+\S+", line), recipe
        assert re.search(r"--position-dependent-phones\s+(true|false)", line), recipe

    def test_it_passes_no_temporary_directory(self, recipe):
        """
        `prepare_lang.sh` took one -- `data/local/lang` -- and left
        `lexiconp.txt` and `lexiconp_disambig.txt` in it, which the
        capture records as `tmp/`. The Python needs none, so a fourth
        positional argument is a call written against the shell script
        and would be read as the output directory.
        """
        assert positional_arguments(recipe) == 3, recipe

    def test_the_three_positionals_are_dictionary_oov_and_output(self, recipe):
        """
        The other half of the count: three arguments in the wrong order
        would also be three.
        """
        arguments = call_arguments(recipe)
        positional = [a for a in strip_flags(arguments)]
        assert "dict" in positional[0], f"{recipe}: {positional}"
        assert positional[1].strip('"') == "<UNK>", f"{recipe}: {positional}"
        assert "lang" in positional[2], f"{recipe}: {positional}"


class TestTheLanguageModelLeftStageOne:
    @pytest.mark.parametrize("recipe", converted_at_stage_one())
    def test_stage_one_formats_no_language_model(self, recipe):
        """
        The LM scripts need `arpa2fst`, `fstcompile`, `fstcompose` and
        `fstinfo` for `G.fst`, which only decoding reads. Leaving them in
        stage 1 would mean the stages training needs still wanted OpenFST.
        """
        body = stage(recipe, 1)
        for name in LM_SCRIPTS:
            assert name not in body, f"{recipe}: stage 1 still runs {name}"

    @pytest.mark.parametrize("recipe", converted_at_stage_one())
    def test_it_moved_rather_than_vanished(self, recipe):
        """
        The other half, and the one that matters: `prepare_graph.sh`
        composes against `G.fst`, so a recipe that stopped building it
        would fail in stage 4 -- and this test is the difference between
        moving the step and deleting it.
        """
        body = stage(recipe, 4)
        assert any(name in body for name in LM_SCRIPTS), (
            f"{recipe}: stage 4 formats no language model, so G.fst is "
            f"never built and prepare_graph.sh has nothing to compose"
        )

    @pytest.mark.parametrize("recipe", converted_at_stage_one())
    def test_the_model_is_formatted_before_the_graphs(self, recipe):
        body = stage(recipe, 4)
        lm = min(body.index(name) for name in LM_SCRIPTS if name in body)
        assert "prepare_graph.sh" in body, recipe
        assert lm < body.index("prepare_graph.sh"), (
            f"{recipe}: stage 4 builds the graphs before formatting the "
            f"language model they compose against"
        )
