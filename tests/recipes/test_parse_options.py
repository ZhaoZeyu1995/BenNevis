"""
A recipe reads its own options without Kaldi.

Tier 1, and it sources the shell that ships rather than a copy -- the
same arrangement as `tests/recipes/test_decode_faster.py` with its awk.

**`run.sh` could not parse `--stage` without a Kaldi checkout.** It
sourced `utils/parse_options.sh`, and `utils/` is a symlink into
`tools/kaldi`, so every recipe needed Kaldi for an argument parser long
after every stage `--stage` selects had become Python. That is the same
defect `prepare_topo.sh` had, found the same way: by trying to run the
thing with no Kaldi there.

What these pin is the semantics, because seven recipes and four
`run/*.sh` are written against Kaldi's and a near-miss would be worse
than an error. The one that earns its place is **an unknown option being
fatal**: `--stop_stge 4` must not quietly run every stage, and a parser
that ignored it would look fine on every correct command line.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PARSER = REPO_ROOT / "BenNevis" / "run" / "parse_options.sh"

pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="bash exists on a Windows runner and does not work; the recipes "
    "are POSIX-only anyway",
)

#: Driving a whole `run.sh` needs the venv, because its first line is
#: `. ./env.sh`, which activates `tools/venv` and fails without it. Tier 1
#: installs pytest and pyyaml and no venv, so those tests skip there --
#: the parser's own semantics are covered by the direct tests above, and
#: what the whole-recipe tests add is that it is wired in, which a
#: developer's machine and the nightly both check.
needs_venv = pytest.mark.skipif(
    not (REPO_ROOT / "tools" / "venv" / "bin" / "activate").is_file(),
    reason="a recipe's env.sh activates tools/venv, which tier 1 has not",
)


def parse(declarations, arguments, report='echo "stage=$stage"'):
    """
    Source the parser with some variables declared and some arguments.

    Arguments
    ---------
    declarations: str
        Shell setting the caller's defaults, as a recipe's top does.
    arguments: list
        The command line.
    report: str
        What to print afterwards.

    Returns
    -------
    subprocess.CompletedProcess
    """
    program = f"set -u\n{declarations}\n" f'. "{PARSER}" || exit 1\n' f"{report}\n"
    return subprocess.run(
        ["bash", "-c", program, "run.sh", *arguments],
        capture_output=True,
        text=True,
    )


class TestItSetsWhatItIsGiven:
    def test_an_option_sets_its_variable(self):
        done = parse("stage=0", ["--stage", "4"])
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "stage=4"

    def test_a_dash_becomes_an_underscore(self):
        """
        `--stop-stage` and `--stop_stage` both set `stop_stage`, which is
        Kaldi's rule and what `run/apply_cmvn.sh`'s old `--norm-vars`
        relied on.
        """
        done = parse(
            "stop_stage=100",
            ["--stop-stage", "7"],
            report='echo "stop_stage=$stop_stage"',
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "stop_stage=7"

    def test_several_options_are_read(self):
        done = parse(
            "stage=0\nstop_stage=100\ntopos=ctc",
            ["--stage", "2", "--stop_stage", "3", "--topos", "ctc mmictc"],
            report='echo "$stage $stop_stage $topos"',
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "2 3 ctc mmictc"

    def test_a_value_may_be_empty(self):
        done = parse(
            'hydra_opts=""',
            ["--hydra_opts", ""],
            report='echo "[$hydra_opts]"',
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "[]"

    def test_a_value_with_spaces_survives(self):
        """
        `--hydra_opts "trainer.max_epochs=2 model=rnnp"` is a single
        argument, and splitting it would pass a stray token to the
        trainer.
        """
        done = parse(
            'hydra_opts=""',
            ["--hydra_opts", "a=1 b=2"],
            report='echo "[$hydra_opts]"',
        )
        assert done.stdout.strip() == "[a=1 b=2]"


class TestItRefusesWhatItCannotRead:
    def test_an_unknown_option_is_fatal(self):
        """
        The test that earns this file. `--stop_stge 4` must not quietly
        run every stage, and a parser that ignored unknown options would
        pass every other test here.
        """
        done = parse("stage=0\nstop_stage=100", ["--stop_stge", "4"])
        assert done.returncode != 0
        assert "unknown option" in done.stderr
        assert "--stop_stge" in done.stderr

    def test_the_equals_form_is_refused(self):
        """
        Kaldi refuses it, so a recipe written for one parser and run
        under the other would silently take a default.
        """
        done = parse("stage=0", ["--stage=4"])
        assert done.returncode != 0
        assert "--name=value is not supported" in done.stderr

    def test_an_option_with_no_value_is_refused(self):
        done = parse("stage=0", ["--stage"])
        assert done.returncode != 0
        assert "needs a value" in done.stderr

    def test_a_boolean_must_be_true_or_false(self):
        """
        Catches `--write-utt2num-frames yes`, which shell would treat as
        a true-ish string and Python as a true value whatever it said.
        """
        done = parse(
            "norm_vars=true",
            ["--norm-vars", "yes"],
            report='echo "$norm_vars"',
        )
        assert done.returncode != 0
        assert "is a boolean" in done.stderr

    def test_a_boolean_accepts_false(self):
        done = parse(
            "norm_vars=true",
            ["--norm-vars", "false"],
            report='echo "$norm_vars"',
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "false"

    def test_a_non_boolean_may_take_any_value(self):
        """
        The other half: a parser that applied the boolean rule to every
        variable would reject `--topos ctc`.
        """
        done = parse("topos=ctc", ["--topos", "mmictc"], report='echo "$topos"')
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "mmictc"


class TestItStopsAtThePositionals:
    def test_the_positional_arguments_survive(self):
        done = parse(
            "nj=4",
            ["--nj", "8", "data/train", "data/lang"],
            report='echo "$# $1 $2"',
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "2 data/train data/lang"

    def test_it_stops_at_the_first_non_option(self):
        """
        So that a positional argument which happens to start with a
        value, or an option meant for something downstream, is left
        alone.
        """
        done = parse(
            "nj=4",
            ["data/train", "--nj", "8"],
            report='echo "$# $nj"',
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "3 4"

    def test_no_arguments_at_all_is_fine(self):
        done = parse("stage=0", [], report='echo "$# $stage"')
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "0 0"


class TestHelp:
    def test_it_prints_the_message_the_caller_set(self):
        done = parse(
            'stage=0\nhelp_message="Usage: run.sh [--stage N]"',
            ["--help"],
        )
        assert done.returncode == 0
        assert "Usage: run.sh" in done.stderr

    def test_it_says_so_when_the_caller_set_none(self):
        done = parse("stage=0", ["-h"])
        assert done.returncode == 0
        assert "no help message" in done.stderr


def shell_scripts():
    """
    Every recipe's `run.sh` and every `BenNevis/run/*.sh`.

    Discovered rather than listed: the point of this parser is that
    nothing needs Kaldi's, and a script left off a list would be the one
    that still did.
    """
    return sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in [
            *REPO_ROOT.glob("egs/*/*/run.sh"),
            *REPO_ROOT.glob("BenNevis/run/*.sh"),
        ]
    )


def sources_a_parser(path):
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    return any(
        line.strip().startswith(". ./") and "parse_options.sh" in line
        for line in text.splitlines()
    )


def test_some_script_parses_options():
    assert [p for p in shell_scripts() if sources_a_parser(p)]


@pytest.mark.parametrize("path", [p for p in shell_scripts() if sources_a_parser(p)])
class TestEveryScriptUsesThisParser:
    def test_it_does_not_source_kaldis(self, path):
        """
        `utils/` is a symlink into the Kaldi checkout, so sourcing Kaldi's
        parser means the script cannot read its own flags without Kaldi --
        whatever else it has been ported to.
        """
        for line in (REPO_ROOT / path).read_text(encoding="utf-8").splitlines():
            if line.strip().startswith(". ./"):
                assert "utils/parse_options.sh" not in line, f"{path}: {line}"

    def test_it_sources_this_one(self, path):
        assert "./run/parse_options.sh" in (REPO_ROOT / path).read_text(
            encoding="utf-8"
        ), path


@needs_venv
@pytest.mark.parametrize("recipe", ["egs/yesno/asr", "egs/mini_librispeech/asr"])
class TestTheRecipesStillParseTheirFlags:
    def test_the_recipe_still_refuses_an_inverted_range(self, recipe):
        """
        End to end over the real `run.sh`: driven with `--stop_stage`
        below `--stage`, so the recipe refuses the range and exits before
        running anything. If the parser did not set the two variables,
        this would run stage 0 instead.
        """
        done = subprocess.run(
            ["bash", "./run.sh", "--stage", "9", "--stop_stage", "1"],
            cwd=REPO_ROOT / recipe,
            capture_output=True,
            text=True,
        )
        assert done.returncode != 0
        assert "is past --stop_stage" in done.stdout + done.stderr

    def test_an_unknown_flag_stops_the_recipe(self, recipe):
        """
        The whole value of the parser, over the real recipe: a typo must
        not run every stage with the defaults.
        """
        done = subprocess.run(
            ["bash", "./run.sh", "--stop_stge", "1"],
            cwd=REPO_ROOT / recipe,
            capture_output=True,
            text=True,
        )
        assert done.returncode != 0
        assert "unknown option" in done.stdout + done.stderr
