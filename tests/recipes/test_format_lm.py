"""
Building the test-time grammar, and why Kaldi's own script is not used.

Tier 1: it reads the shell as text and, where it can, runs it.

`local/format_lms.sh` ends in `utils/validate_lang.pl`, and that
validates a **GMM** language directory. It requires nine files BenNevis
does not write -- `phones/context_indep.txt`, four `.csl` lists,
`phones/roots.txt`, `phones/sets.txt`, `phones/extra_questions.{txt,int}`
and `topo` -- every one of which only a decision-tree build or an HMM
topology reads. BenNevis trains no GMM and has no HMM topology, so
writing them would be nine more things to keep correct in order to
satisfy a validator whose subject does not exist here.

Stage 4 failed on exactly that, in run 46165159785, with nine
`fail to open` lines and then `ERROR (see error messages above)`.

**yesno could not have shown it.** Its `local/prepare_lm.sh` runs
`arpa2fst` and some `fstinfo` diagnostics and never validates, and yesno
is the recipe that runs nightly. "Nothing in BenNevis reads it" is not
"nothing in the recipe reads it", and the recipe that runs most often is
the one least able to say -- the second time that has cost a dispatch,
after `spk2gender`.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "BenNevis" / "run" / "format_lm.sh"

posix_only = pytest.mark.skipif(
    os.name == "nt",
    reason="bash exists on a Windows runner and does not work; the recipes "
    "are POSIX-only anyway",
)


def body():
    return SCRIPT.read_text(encoding="utf-8")


def commands():
    """The script's lines with comments stripped."""
    return "\n".join(
        line for line in body().splitlines() if not line.strip().startswith("#")
    )


class TestItBuildsTheGrammar:
    def test_it_exists_and_is_executable(self):
        assert SCRIPT.is_file()
        if os.name != "nt":
            assert SCRIPT.stat().st_mode & 0o111

    def test_it_runs_arpa2fst_with_kaldis_flags(self):
        """
        `--disambig-symbol=#0` and `--read-symbol-table`, as Kaldi's
        script passes. Without the symbol table the integers would be
        arpa2fst's own and would match nothing else in the directory.
        """
        text = commands()
        assert "arpa2fst" in text
        assert "--disambig-symbol='#0'" in text or '--disambig-symbol="#0"' in text
        assert "--read-symbol-table=" in text

    def test_it_arcsorts_on_the_input_side(self):
        """
        `fsttablecompose` in `prepare_graph.sh` takes `L_disambig.fst`
        and `G.fst`; the grammar is the second argument, which wants
        ilabel sorting. Kaldi's `prepare_nolm.sh` does the same.
        """
        assert "--sort_type=ilabel" in commands()

    def test_it_does_not_validate_the_lang_directory(self):
        """
        The whole reason this script exists. Running it would demand nine
        GMM files that are deliberately absent.
        """
        assert "validate_lang" not in commands()

    def test_the_header_says_why_not(self):
        """
        A reader meeting a missing validation step needs the reason, not
        its absence -- and the reason is not obvious, since validating
        sounds strictly good.
        """
        text = body()
        assert "validate_lang.pl" in text
        assert "GMM" in text
        for name in ("topo", "roots.txt", "sets.txt"):
            assert name in text, name

    def test_it_names_what_covers_the_checks_it_skips(self):
        """
        Dropping a validator is only safe if its useful checks live
        somewhere. The header has to say where, or the next reader
        cannot tell a considered omission from an oversight.
        """
        text = body()
        assert "compare_graphs.py" in text
        assert "prepare_graph.sh" in text

    def test_it_builds_only_what_the_recipe_decodes_against(self):
        """
        Kaldi's script built `tgsmall` and `tgmed`; the recipe decodes
        against `tgsmall` alone, so the second was a graph nothing read.
        """
        assert "tgmed" not in commands()
        assert "tgsmall" in body()


class TestItFailsLoudly:
    def test_it_sets_pipefail(self):
        """
        The command is `gunzip | arpa2fst | fstarcsort`, so without
        `pipefail` a failure in arpa2fst would be reported as
        fstarcsort's success and leave a truncated G.fst. The same trap
        as the nightly's `| tee`.
        """
        text = commands()
        assert "set -eo pipefail" in text
        assert text.index("set -eo pipefail") < text.index("arpa2fst")

    def test_it_checks_its_inputs(self):
        text = commands()
        assert "words.txt" in text
        assert "does not exist" in text

    def test_it_checks_the_grammar_is_not_empty(self):
        """
        `arpa2fst` writing nothing would leave a file that composes to an
        empty graph and decodes every utterance as silence -- wrong
        rather than broken.
        """
        assert "-s " in commands()
        assert "is empty" in commands()

    @posix_only
    def test_a_missing_model_is_refused(self, tmp_path):
        """
        Run for real, with no Kaldi: it must fail on its own input check
        rather than on `arpa2fst` being absent, so the message a reader
        gets names the missing file.
        """
        recipe = tmp_path / "recipe"
        (recipe / "run").mkdir(parents=True)
        (recipe / "data" / "lang").mkdir(parents=True)
        (recipe / "data" / "lang" / "words.txt").write_text("<eps> 0\n", "utf-8")
        (recipe / "data" / "local" / "lm").mkdir(parents=True)
        # A path.sh that sets nothing, so the script can be sourced
        # without a Kaldi tree.
        (recipe / "path.sh").write_text("true\n", "utf-8")
        for name in ("format_lm.sh", "parse_options.sh"):
            (recipe / "run" / name).write_bytes(
                (REPO_ROOT / "BenNevis" / "run" / name).read_bytes()
            )
        done = subprocess.run(
            ["bash", "./run/format_lm.sh", "--src-dir", "data/lang", "data/local/lm"],
            cwd=recipe,
            capture_output=True,
            text=True,
        )
        assert done.returncode != 0
        assert "lm_tgsmall.arpa.gz does not exist" in done.stderr


class TestTheRecipeUsesIt:
    def test_mini_librispeech_calls_it_and_not_kaldis(self):
        text = (REPO_ROOT / "egs" / "mini_librispeech" / "asr" / "run.sh").read_text()
        commands_only = "\n".join(
            line for line in text.splitlines() if not line.strip().startswith("#")
        )
        assert "run/format_lm.sh" in commands_only
        assert "local/format_lms.sh" not in commands_only

    def test_it_is_given_the_language_directory_stage_one_wrote(self):
        """
        `--src-dir` must be the directory `prepare-lang` produced, or the
        grammar is built against a different `words.txt` and every
        integer in it means something else.

        Comments are filtered before the scan, because the comment above
        the call names `local/format_lms.sh` and the first version of
        this test matched it -- the third time today that a scan for a
        command found an explanation instead.
        """
        text = (REPO_ROOT / "egs" / "mini_librispeech" / "asr" / "run.sh").read_text()
        joined = text.replace("\\\n", " ")
        lines = [
            line for line in joined.splitlines() if not line.strip().startswith("#")
        ]
        line = next(line for line in lines if "run/format_lm.sh" in line)
        assert "--src-dir ${lang}" in line
        # and `${lang}` is what stage 1 passes to prepare-lang, so the two
        # cannot be given different directories
        assert "${lang}" in next(
            candidate for candidate in lines if "BenNevis.lm prepare-lang" in candidate
        )
