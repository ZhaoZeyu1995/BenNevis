"""
Tests for the nightly's language-directory gate, and its settings parser.

Tier 1. The parser half needs nothing but the repository's own `run.sh`
files; the workflow half reads the YAML.

**The gate exists because the committed reference stopped being able to
answer the question.** Stage 1 builds its language directory in Python,
so a fresh capture's `lang/` and `dict/` are our own output and
`tests/lm/test_prepare_lang.py` would compare the code with itself.

What is worth testing here is **the settings parser**, because it is the
part that can be wrong while looking right. `prepare-lang` must be
compared against a Kaldi directory built with the *same* `--sil-prob` and
`--position-dependent-phones`; a mismatch there builds a lexicon that is
wrong rather than broken, and the comparison would then report a
difference that is the harness's fault, or -- worse, on the day the two
settings happen to agree -- hide one that is not.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nightly.yml"
SCRIPT = REPO_ROOT / "tests" / "reference" / "compare_lang.py"
STEP = "Compare the Python language directory against Kaldi's"

sys.path.insert(0, str(REPO_ROOT / "tests" / "reference"))
import compare_lang  # noqa: E402


def steps():
    workflow = yaml.safe_load(WORKFLOW.read_text())
    (job,) = workflow["jobs"].values()
    return job["steps"]


def step():
    for candidate in steps():
        if candidate.get("name") == STEP:
            return candidate
    raise AssertionError(f"no step named {STEP!r}")


def converted():
    """Recipes whose stage 1 is the Python one."""
    return sorted(
        path.parent.relative_to(REPO_ROOT).as_posix()
        for path in REPO_ROOT.glob("egs/*/*/run.sh")
        if "BenNevis.lm prepare-lang" in path.read_text(encoding="utf-8")
    )


class TestTheGateIsWired:
    def test_the_step_exists_and_runs_the_script(self):
        assert "tests/reference/compare_lang.py" in step()["run"]
        assert SCRIPT.is_file()

    def test_it_runs_even_when_the_recipe_failed(self):
        assert "always()" in str(step().get("if", ""))

    def test_it_is_blocking(self):
        assert "continue-on-error" not in step()

    def test_the_pipe_to_tee_cannot_swallow_a_failure(self):
        run = step()["run"]
        assert "| tee" in run
        assert "set -eo pipefail" in run
        assert run.index("set -eo pipefail") < run.index("| tee")

    def test_it_does_not_restate_the_settings(self):
        """
        The settings come out of the recipe. A workflow carrying its own
        copy of `--sil-prob` is a copy that drifts from the recipe, which
        is the trap this repository has recorded three times -- stage 3,
        stage 10 and the graph gate.
        """
        run = step()["run"]
        for flag in ("--sil-prob", "--position-dependent-phones", "--dict"):
            assert flag not in run, f"{flag} is restated in the workflow"

    def test_the_log_lands_under_artefacts(self):
        assert "artefacts/lang-comparison.log" in step()["run"]


@pytest.mark.parametrize("recipe", converted())
class TestTheSettingsParser:
    def test_it_finds_the_call(self, recipe):
        assert compare_lang.stage_one_call(REPO_ROOT / recipe)

    def test_it_stops_at_the_shells_error_handling(self, recipe):
        """
        The joined line carries `|| { echo ...; exit 1; }`, and reading
        that as arguments is how a positional scan comes up with eleven.
        """
        call = compare_lang.stage_one_call(REPO_ROOT / recipe)
        assert "||" not in call
        assert "echo" not in call

    def test_the_settings_match_the_recipes_own_flags(self, recipe):
        """
        Against the recipe's text rather than against a table here, so
        this cannot drift from what the recipe says.
        """
        text = (REPO_ROOT / recipe / "run.sh").read_text(encoding="utf-8")
        settings = compare_lang.settings_from(
            compare_lang.stage_one_call(REPO_ROOT / recipe)
        )
        assert f"--sil-prob {settings['sil_prob']:g}" in text.replace("\\\n", " ")
        spelt = "true" if settings["position_dependent_phones"] else "false"
        assert f"--position-dependent-phones {spelt}" in text.replace("\\\n", " ")

    def test_the_three_positionals_are_read_in_order(self, recipe):
        settings = compare_lang.settings_from(
            compare_lang.stage_one_call(REPO_ROOT / recipe)
        )
        assert "dict" in str(settings["dict_dir"])
        assert settings["oov_word"] == "<UNK>"
        assert "lang" in str(settings["out_dir"])

    def test_the_oov_word_loses_its_quotes(self, recipe):
        """
        The recipe writes `"<UNK>"` because the shell would otherwise
        read the angle brackets as a redirection. Passing the quotes
        through to `prepare_lang` would look for a word that is not in
        the lexicon and raise.
        """
        settings = compare_lang.settings_from(
            compare_lang.stage_one_call(REPO_ROOT / recipe)
        )
        assert '"' not in str(settings["oov_word"])


class TestTheParserRefusesWhatItCannotRead:
    def test_a_call_without_settings_is_refused(self):
        """
        Rather than filling in `prepare-lang`'s defaults, which would
        compare our 0.5 against a Kaldi directory built with whatever the
        recipe meant.
        """
        with pytest.raises(ValueError, match="no settings"):
            compare_lang.settings_from(" data/local/dict '<UNK>' data/lang")

    def test_a_fourth_positional_is_refused(self):
        """
        A call written against the shell script, which took a temporary
        directory. Treating the fourth as the output would compare the
        wrong directory and report agreement on nothing.
        """
        call = (
            " --sil-prob 0.0 --position-dependent-phones false "
            "data/local/dict '<UNK>' data/local/lang data/lang"
        )
        with pytest.raises(ValueError, match="three positional"):
            compare_lang.settings_from(call)

    def test_an_unconverted_recipe_is_skipped_rather_than_failed(self, tmp_path):
        """
        Five recipes still use Kaldi's script, deliberately, because
        their settings have never been checked. The gate must not report
        them as failures.
        """
        recipe = tmp_path / "recipe"
        recipe.mkdir()
        (recipe / "run.sh").write_text("utils/prepare_lang.sh a b c d\n", "utf-8")
        assert compare_lang.stage_one_call(recipe) == ""
