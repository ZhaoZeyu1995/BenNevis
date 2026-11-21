"""
Tests for the transcription formatting in `run/decode_faster.sh`.

The script's last awk turns Kaldi's `uttid word word ...` into NIST's
`word word ... (speaker-uttid)`. It did that with
`substr($0, index($0, $2))`, which is correct until the hypothesis is empty:
`$2` is then unset, `index($0, "")` is 0, and `substr($0, 0)` returns the
whole line -- so a decoder that emitted nothing had its **utterance id**
recorded as the hypothesis.

That is how the first complete recipe run scored: an undertrained model
decoded nothing, and every hypothesis came out as its own file name. The
error count happens to be the same either way -- one substitution plus N-1
deletions against N deletions -- so the WER looked plausible while the
transcription was nonsense, and anything reading `hyp.wrd.txt` downstream
saw a word that was never spoken.

The awk program is extracted from the script and run, rather than copied
here, so these tests are about the code that ships.

Tier 1: awk is on every runner, and the tests skip if it is not.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "BenNevis" / "run" / "decode_faster.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("awk") is None, reason="the formatting step is awk"
)


def awk_program() -> str:
    """
    The awk program the script uses to build hyp.wrd.trn.

    Pulled out of the shell rather than duplicated, so that a change to the
    script is a change to what these tests exercise.
    """
    text = SCRIPT.read_text()
    match = re.search(
        r"^awk '(?P<program>.*?)' \$data_dir/utt2spk", text, re.MULTILINE | re.DOTALL
    )
    assert match, "the awk that writes hyp.wrd.trn is no longer recognisable"
    return match.group("program")


def format_hypotheses(tmp_path, utt2spk, hypotheses):
    """Run the real awk over synthetic inputs and return its lines."""
    utt2spk_path = tmp_path / "utt2spk"
    utt2spk_path.write_text("".join(f"{u} {s}\n" for u, s in utt2spk))
    hyp_path = tmp_path / "hyp.wrd.txt"
    hyp_path.write_text("".join(f"{line}\n" for line in hypotheses))

    completed = subprocess.run(
        ["awk", awk_program(), str(utt2spk_path), str(hyp_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.splitlines()


class TestFormatting:
    def test_a_normal_hypothesis(self, tmp_path):
        lines = format_hypotheses(tmp_path, [("u1", "spk")], ["u1 YES NO NO YES"])
        assert lines == ["YES NO NO YES (spk-u1)"]

    def test_an_empty_hypothesis_is_empty(self, tmp_path):
        """
        The bug. The utterance id must not become the transcription.
        """
        lines = format_hypotheses(tmp_path, [("u1", "spk")], ["u1"])
        assert len(lines) == 1
        words, _, identifier = lines[0].rpartition(" (")
        assert words.strip() == "", f"hypothesis is {words!r}, not empty"
        assert identifier == "spk-u1)"

    def test_an_empty_hypothesis_does_not_contain_its_own_id(self, tmp_path):
        """
        Stated separately because this is the specific corruption: scoring a
        file name as a spoken word. If an id ever coincided with a real word
        it would even be counted correct.
        """
        lines = format_hypotheses(tmp_path, [("1_0_0_1", "global")], ["1_0_0_1"])
        words = lines[0].rpartition(" (")[0]
        assert "1_0_0_1" not in words

    def test_a_single_word_hypothesis(self, tmp_path):
        lines = format_hypotheses(tmp_path, [("u1", "spk")], ["u1 YES"])
        assert lines == ["YES (spk-u1)"]

    def test_several_utterances_keep_their_order(self, tmp_path):
        lines = format_hypotheses(
            tmp_path,
            [("u1", "a"), ("u2", "b"), ("u3", "a")],
            ["u1 YES", "u2", "u3 NO NO"],
        )
        assert lines == ["YES (a-u1)", " (b-u2)", "NO NO (a-u3)"]

    def test_the_speaker_comes_from_utt2spk(self, tmp_path):
        lines = format_hypotheses(tmp_path, [("u1", "speaker42")], ["u1 YES"])
        assert lines == ["YES (speaker42-u1)"]

    def test_extra_whitespace_is_not_carried_into_the_transcription(self, tmp_path):
        lines = format_hypotheses(tmp_path, [("u1", "spk")], ["u1   YES   NO"])
        words = lines[0].rpartition(" (")[0]
        assert not words.startswith(" ")
        assert words.split() == ["YES", "NO"]


class TestTheOutputIsScoreable:
    """
    The result has to be readable by the scorer, including the empty case.
    """

    def test_the_scorer_reads_an_empty_hypothesis_as_no_words(self, tmp_path):
        from BenNevis.utils.scoring import read_trn, score

        lines = format_hypotheses(
            tmp_path, [("u1", "spk"), ("u2", "spk")], ["u1 YES NO", "u2"]
        )
        hyp_path = tmp_path / "hyp.wrd.trn"
        hyp_path.write_text("".join(f"{line}\n" for line in lines))

        ref_path = tmp_path / "ref.wrd.trn"
        ref_path.write_text("YES NO (spk-u1)\nYES NO (spk-u2)\n")

        counts = score(read_trn(ref_path), read_trn(hyp_path))
        # Two reference words missing entirely: two deletions, no
        # substitution from a spurious utterance id.
        assert counts.deletions == 2
        assert counts.substitutions == 0
        assert counts.wer == pytest.approx(50.0)
