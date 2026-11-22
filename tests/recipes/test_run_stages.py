"""
Tests for the recipe's stage selection.

`run.sh` is ten stages, each reading what the ones before it wrote, and
`--stage`/`--stop_stage` choose which to run. Getting that wrong is quiet
and expensive: an off-by-one skips a stage whose output the next one needs,
and the failure appears somewhere else entirely.

The guard expressions are extracted from the script and executed, rather
than reimplemented here, so these tests are about the conditions that ship.
The stage bodies are replaced with an echo, which is what makes it possible
to test the selection without Kaldi.

Tier 1: bash and awk only, and it skips where there is no bash.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EGS = REPO_ROOT / "egs"

#: Every recipe that has been given stage guards. A recipe without them is
#: not a failure here -- the librispeech and timit recipes still lack them --
#: but one that has them must have them consistently.
STAGED = [
    path for path in sorted(EGS.glob("*/*/run.sh")) if "${stage}" in path.read_text()
]
assert STAGED, "no recipe has stage guards; has run.sh been restructured?"

RECIPE = EGS / "yesno" / "asr"
RUN = RECIPE / "run.sh"

# Executing the guards needs a working bash, which Windows does not have for
# this purpose: `shutil.which("bash")` finds C:\Windows\System32\bash.exe,
# the WSL launcher, which exists whether or not a distribution is installed
# and exits 1 when one is not. So the presence of bash is not the question on
# Windows -- and the recipes are Linux and macOS only anyway (see CLAUDE.md).
#
# The structural tests below are pure text and run everywhere; only the ones
# that run the guards are skipped.
needs_bash = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="running the stage guards needs a working bash; on Windows the "
    "bash on PATH is the WSL launcher, and the recipes are not Windows",
)

# Negative stage numbers are part of the convention, not a mistake:
# egs/librispeech/asr3 uses stage -1 for an optional data download.
GUARD_RE = re.compile(
    r"^if \[ \$\{stage\} -le (?P<number>-?\d+) \] "
    r"&& \[ \$\{stop_stage\} -ge (?P<same>-?\d+) \]; then$",
    re.MULTILINE,
)
LABEL_RE = re.compile(
    r'^\s*echo "\$0: stage (?P<number>-?\d+): (?P<label>[^"]+)"',
    re.MULTILINE | re.IGNORECASE,
)

#: Recipes brought up to the fuller convention: they announce each stage,
#: refuse an inverted range, and stop on a failing stage. yesno gained all
#: three after a stage failed silently and Kaldi's warning about two
#: unreadable files went unnoticed; mini_librispeech was written with them.
#:
#: The stage guards themselves predate this work -- egs/timit and
#: egs/librispeech/asr3 already used them, and asr3 uses stage -1 for an
#: optional download. What those recipes lack is the announcing and the
#: stopping: timit has nine guards and no announcements at all. They are
#: listed here rather than exempted silently, and the membership is itself
#: asserted, so a recipe cannot drift in or out unnoticed. Converting them is
#: its own piece of work on recipes that cannot currently be run to check.
CONVERTED = {"yesno/asr", "mini_librispeech/asr"}


def guards(run=None):
    """The stage guards in the script, in order, as (number, condition)."""
    text = (run or RUN).read_text()
    found = []
    for match in GUARD_RE.finditer(text):
        assert match.group("number") == match.group("same"), (
            f"stage guard compares {match.group('number')} with "
            f"{match.group('same')}; one of them is a typo"
        )
        found.append((int(match.group("number")), match.group(0)))
    assert found, "no stage guards found; has run.sh been restructured?"
    return found


def select(stage, stop_stage):
    """
    Which stages run for a given pair, using the script's own conditions.

    Each guard is executed with its body replaced by an echo, so the answer
    comes from the real expressions rather than from a copy of them.
    """
    lines = [f"stage={stage}", f"stop_stage={stop_stage}"]
    for number, condition in guards():
        lines.append(condition)
        lines.append(f'    echo "{number}"')
        lines.append("fi")
    completed = subprocess.run(
        ["bash", "-c", "\n".join(lines)],
        check=True,
        capture_output=True,
        text=True,
    )
    return [int(line) for line in completed.stdout.split()]


@pytest.mark.parametrize(
    "run", STAGED, ids=[p.relative_to(EGS).parent.as_posix() for p in STAGED]
)
class TestEveryStagedRecipe:
    """
    The structural rules, applied to every recipe that has stage guards.

    Written as a parametrised class so a new recipe is covered by adding the
    guards rather than by remembering to add tests.
    """

    def test_the_stages_are_contiguous(self, run):
        """
        A gap is a stage nobody can select by number and a repeat is two
        stages that cannot be run apart. The first number is not required to
        be zero: asr3 starts at -1 for an optional download.
        """
        numbers = [number for number, _ in guards(run)]
        assert numbers == list(range(numbers[0], numbers[0] + len(numbers))), numbers

    def test_every_stage_announces_itself_where_converted(self, run):
        name = run.relative_to(EGS).parent.as_posix()
        text = run.read_text()
        labelled = {int(m.group("number")) for m in LABEL_RE.finditer(text)}
        numbers = {number for number, _ in guards(run)}
        if name in CONVERTED:
            assert labelled == numbers, (
                f"{name} is listed as converted but labels {sorted(labelled)} "
                f"for stages {sorted(numbers)}"
            )
        else:
            assert (
                labelled != numbers
            ), f"{name} now announces every stage; add it to CONVERTED"

    def test_the_options_are_declared_before_parse_options(self, run):
        text = run.read_text()
        assert text.index("\nstage=") < text.index("parse_options.sh")
        assert text.index("\nstop_stage=") < text.index("parse_options.sh")

    def test_an_inverted_range_is_refused_where_the_recipe_does_so(self, run):
        """
        Only the converted recipes do this; see CONVERTED.
        """
        name = run.relative_to(EGS).parent.as_posix()
        if name not in CONVERTED:
            pytest.skip(f"{name} has not been converted")
        text = run.read_text()
        assert "-gt ${stop_stage}" in text

    def test_it_stops_on_a_failing_stage_where_converted(self, run):
        """
        `set -e` plus `pipefail`. Without them a stage can fail and the recipe
        carries on with partial inputs, which is how Kaldi warning about two
        unreadable files went unnoticed on yesno.

        Asserted only for the recipes converted so far, and the membership of
        that set is itself checked below -- so a recipe cannot be quietly
        added to the exemption.
        """
        name = run.relative_to(EGS).parent.as_posix()
        text = run.read_text()
        stops = "set -eo pipefail" in text or "set -euo pipefail" in text
        if name in CONVERTED:
            assert stops, f"{name} is listed as converted but does not stop"
        else:
            assert not stops, f"{name} now stops on failure; add it to CONVERTED"


class TestStageNumbering:
    def test_the_stages_are_contiguous_from_zero(self):
        """
        A gap would mean a stage nobody can select by number, and a repeat
        would mean two stages that cannot be run separately.
        """
        numbers = [number for number, _ in guards()]
        assert numbers == list(range(len(numbers))), numbers

    def test_every_stage_announces_itself(self):
        """
        So a log says which stage is running. Finding out from the first
        error message instead is how a re-run from the wrong stage goes
        unnoticed.
        """
        text = RUN.read_text()
        labelled = {int(m.group("number")) for m in LABEL_RE.finditer(text)}
        assert labelled == {number for number, _ in guards()}

    def test_the_options_are_declared_before_parse_options(self):
        """
        Kaldi's parse_options.sh only accepts a flag whose variable already
        exists, so --stage would be rejected outright otherwise.
        """
        text = RUN.read_text()
        assert text.index("\nstage=") < text.index("parse_options.sh")
        assert text.index("\nstop_stage=") < text.index("parse_options.sh")

    def test_the_default_runs_everything(self):
        text = RUN.read_text()
        stage = re.search(r"^stage=(\d+)$", text, re.MULTILINE)
        stop = re.search(r"^stop_stage=(\d+)$", text, re.MULTILINE)
        assert stage and stop
        assert int(stage.group(1)) == 0
        assert int(stop.group(1)) >= len(guards()) - 1


@needs_bash
class TestSelection:
    def test_the_default_selects_every_stage(self):
        numbers = [number for number, _ in guards()]
        assert select(0, 100) == numbers

    def test_a_single_stage_selects_only_itself(self):
        for number, _ in guards():
            assert select(number, number) == [number]

    def test_a_range_is_inclusive_at_both_ends(self):
        assert select(2, 4) == [2, 3, 4]

    def test_starting_late_skips_the_earlier_stages(self):
        numbers = [number for number, _ in guards()]
        assert select(5, 100) == [n for n in numbers if n >= 5]

    def test_stopping_early_skips_the_later_stages(self):
        assert select(0, 1) == [0, 1]

    def test_an_inverted_range_selects_nothing(self):
        """
        The script refuses this before reaching any stage, but the guards
        must not quietly run something if it ever got through.
        """
        assert select(5, 2) == []


class TestInvertedRangeIsRefused:
    def test_the_script_rejects_a_stage_past_stop_stage(self):
        """
        Rather than exiting 0 having done nothing, which in CI is
        indistinguishable from success.
        """
        text = RUN.read_text()
        assert "-gt ${stop_stage}" in text
        guard = text[text.index("-gt ${stop_stage}") :]
        assert "exit 1" in guard[:400]
