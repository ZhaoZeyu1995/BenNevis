"""
Tests for the WER extraction in `.github/workflows/nightly.yml`.

The parsers are pulled out of the workflow and run, so the test is about
the code that ships -- the same approach `test_decode_faster.py` takes to
the awk in `run/decode_faster.sh`, and for the same reason: a parser
reimplemented in the test would pass while the shipped one failed.

This exists because the shipped one did fail. The step sets `pipefail` and
its shell is `bash -e`, so `grep -E '...' "$f" | awk '...'` aborted the
whole step as soon as a file it could not match appeared -- which is what
the k2 decoder's results are, since they come from
`BenNevis.utils.scoring` rather than sclite. A successful recipe reported
failure, twice, including one scheduled run.

The sample rows are taken verbatim from a real run rather than invented.
CLAUDE.md warns against pinning a parser against a hand-made sclite
fixture, and the warning stands: what is pinned here is the *field
arithmetic*, which is the part that was wrong in a different place -- the
last three fields are Err, S.Err and a bar, so `NF-1` silently gives the
sentence error rate, a bigger number that looks plausible.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "nightly.yml"

#: A real sclite summary, copied from run 45072040967. Note `| Sum/Avg|`
#: with no space before the second bar, so awk sees `Sum/Avg|` as one
#: field -- which is why the row has twelve fields and not thirteen.
SCLITE = (
    "      | SPKR    | # Snt # Wrd  | Corr     Sub    Del     Ins     Err"
    "  S.Err  |\n"
    "      | Sum/Avg|   29    232 | 99.6    0.0     0.4    0.0    0.4"
    "    3.4 |\n"
)

#: What `BenNevis.utils.scoring` writes, from the same run.
K2 = "WER 0.43%  [1/232 words, 0 sub, 1 del, 0 ins]\n"

posix_only = pytest.mark.skipif(
    os.name == "nt",
    reason="bash on a Windows runner is the WSL launcher and does not work",
)


def extract(tool: str) -> str:
    """
    One parser, lifted out of the workflow.

    Taken a line at a time rather than with a regex over the whole file:
    the awk fragment contains `$(NF-2)`, so anything that stops at the
    first `)` truncates it in the middle of a quoted string, and the
    result fails to parse for a reason that has nothing to do with the
    workflow.

    Arguments
    ---------
    tool: str
        The command the assignment begins with, `awk` or `sed`.

    Returns
    -------
    str
        The shell fragment between `wer=$(` and the closing bracket.
    """
    prefix = f"wer=$({tool} "
    for line in WORKFLOW.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix) and stripped.endswith(")"):
            return stripped[len("wer=$(") : -1]
    raise AssertionError(
        f"no `{prefix}...)` line in {WORKFLOW.name}; the summary step's "
        f"WER extraction has moved or changed shape"
    )


def run(fragment: str, contents: str, tmp_path: Path) -> str:
    """Run a `wer=$(...)` fragment against a file and return what it set."""
    target = tmp_path / "results.wrd.txt"
    target.write_text(contents, encoding="utf-8")
    script = (
        "set -eo pipefail\n"
        f'd="{target}"\n'
        f"wer=$({fragment})\n"
        'printf "%s" "$wer"\n'
    )
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, (
        f"the fragment failed under `set -eo pipefail`, which is what the "
        f"workflow uses:\n{done.stderr}"
    )
    return done.stdout


@pytest.fixture
def sclite_parser():
    return extract("awk")


@pytest.fixture
def scoring_parser():
    return extract("sed")


class TestTheWorkflowStillHasThem:
    def test_the_sclite_parser_is_awk_not_grep(self):
        """
        The fix. `grep` exits non-zero when it matches nothing, and under
        `pipefail` that fails the step -- so a file in the other format
        turned a successful recipe into a red run.
        """
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "grep -E '\\| *Sum/Avg'" not in text
        assert "awk '/\\| *Sum\\/Avg/" in text

    def test_both_parsers_are_present(self, sclite_parser, scoring_parser):
        assert "Sum" in sclite_parser
        assert "WER" in scoring_parser


@posix_only
class TestSclite:
    def test_it_reads_the_word_error_rate(self, sclite_parser, tmp_path):
        """
        `NF-2`, not `NF-1`. The last three fields are Err, S.Err and a
        bar, so the sentence error rate sits where a careless index lands
        -- 3.4 against 0.4 here, which is the kind of wrong number that
        reads as a real regression.
        """
        assert run(sclite_parser, SCLITE, tmp_path) == "0.4"

    def test_a_file_in_the_other_format_gives_nothing(self, sclite_parser, tmp_path):
        """
        Empty rather than failing, which is the whole point: this parser
        is tried first on every file, including the k2 ones.
        """
        assert run(sclite_parser, K2, tmp_path) == ""

    def test_an_empty_file_gives_nothing(self, sclite_parser, tmp_path):
        assert run(sclite_parser, "", tmp_path) == ""

    def test_a_file_with_no_summary_row_gives_nothing(self, sclite_parser, tmp_path):
        assert run(sclite_parser, "nothing to see\n", tmp_path) == ""


@posix_only
class TestScoring:
    def test_it_reads_the_word_error_rate(self, scoring_parser, tmp_path):
        assert run(scoring_parser, K2, tmp_path) == "0.43"

    def test_a_zero_rate_reads_as_zero_not_empty(self, scoring_parser, tmp_path):
        """
        Four of yesno's eight topologies score 0.00, and an empty result
        would show them as "unknown" -- which looks like a parse failure
        rather than a perfect run.
        """
        line = "WER 0.00%  [0/232 words, 0 sub, 0 del, 0 ins]\n"
        assert run(scoring_parser, line, tmp_path) == "0.00"

    def test_a_file_in_the_other_format_gives_nothing(self, scoring_parser, tmp_path):
        assert run(scoring_parser, SCLITE, tmp_path) == ""

    def test_an_empty_file_gives_nothing(self, scoring_parser, tmp_path):
        assert run(scoring_parser, "", tmp_path) == ""


@posix_only
class TestTogether:
    """
    The two in sequence, as the workflow runs them: sclite first, then the
    scoring fallback. Every file one of the two decoders writes must come
    out with a number.
    """

    @pytest.mark.parametrize(
        "contents,expected",
        [
            (SCLITE, "0.4"),
            (K2, "0.43"),
            ("WER 0.00%  [0/232 words, 0 sub, 0 del, 0 ins]\n", "0.00"),
        ],
    )
    def test_one_of_them_parses_it(
        self, sclite_parser, scoring_parser, contents, expected, tmp_path
    ):
        first = run(sclite_parser, contents, tmp_path)
        wer = first or run(scoring_parser, contents, tmp_path)
        assert wer == expected

    def test_an_unparseable_file_yields_neither_a_number_nor_a_failure(
        self, sclite_parser, scoring_parser, tmp_path
    ):
        """
        It becomes "unknown" in the table and increments the step's
        `unparsed` count. The step must not abort: the capture is already
        uploaded by then, and losing the summary over one odd file would
        hide seven good rows.
        """
        junk = "this is not a results file\n"
        assert run(sclite_parser, junk, tmp_path) == ""
        assert run(scoring_parser, junk, tmp_path) == ""
