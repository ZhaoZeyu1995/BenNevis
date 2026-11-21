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

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RECIPE = REPO_ROOT / "egs" / "yesno" / "asr"
RUN = RECIPE / "run.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="the recipe is bash"
)

GUARD_RE = re.compile(
    r"^if \[ \$\{stage\} -le (?P<number>\d+) \] "
    r"&& \[ \$\{stop_stage\} -ge (?P<same>\d+) \]; then$",
    re.MULTILINE,
)
LABEL_RE = re.compile(
    r'^\s*echo "\$0: stage (?P<number>\d+): (?P<label>[^"]+)"$', re.MULTILINE
)


def guards():
    """The stage guards in the script, in order, as (number, condition)."""
    text = RUN.read_text()
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
