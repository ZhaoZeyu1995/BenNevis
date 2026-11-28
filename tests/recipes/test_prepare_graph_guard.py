"""
`prepare_graph.sh`'s input check, which could not fail.

Tier 1, and it runs the shell it ships rather than a copy -- the same
arrangement as `tests/recipes/test_decode_faster.py`, where a
reimplementation of the awk in the test would have passed while the
shipped parser failed.

The guard read:

    if [ ! -f $src/$file ]; then
        echo "$0: $src/$file is missing!" || exit 1;
    fi

`||` binds to `echo`, `echo` always succeeds, so the `exit` was
unreachable: the script announced the missing input and carried on to
fail inside `fsttablecompose` with something unrelated. Same family as
the nightly's `grep | awk` under `pipefail`, the `| tee` that made a gate
unable to fail, and the `G.fst` gate that skipped unconditionally -- a
shell detail deciding whether a check means anything.

It matters more now than it did. `L_disambig.fst` is written by
`python3 -m BenNevis.lm prepare-lang`, which **skips it when OpenFST is
absent**, because nothing in BenNevis's lane reads a binary FST. So a
missing `L_disambig.fst` is a reachable state on a machine with no Kaldi,
and this is where it has to be said out loud.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "BenNevis" / "run" / "prepare_graph.sh"

posix_only = pytest.mark.skipif(
    os.name == "nt",
    reason="bash exists on a Windows runner and does not work; the recipes "
    "are POSIX-only anyway",
)


def guard() -> str:
    """
    The `for file in L_disambig.fst G.fst` loop, lifted from the script.

    Lifted rather than copied, so the test is about the code that ships.
    """
    text = SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        r"^(for file in L_disambig\.fst G\.fst; do\n.*?^done)$",
        text,
        re.DOTALL | re.MULTILINE,
    )
    assert match, "the input guard is no longer recognisable in the script"
    return match.group(1)


def run_guard(src: Path):
    """Run the guard with `$src` set, returning the completed process."""
    program = (
        f'set -u\nsrc="{src}"\n0="prepare_graph.sh"\n{guard()}\necho REACHED-THE-END\n'
    )
    return subprocess.run(["bash", "-c", program], capture_output=True, text=True)


class TestTheGuardIsRecognisable:
    def test_it_is_still_in_the_script(self):
        assert "L_disambig.fst" in guard()
        assert "G.fst" in guard()

    def test_it_does_not_use_the_unreachable_echo_form(self):
        """
        The specific defect, pinned by shape as well as by behaviour: the
        behavioural test below would also pass for a guard rewritten some
        other broken way, but this names the thing that was wrong.
        """
        assert 'is missing!" || exit 1' not in SCRIPT.read_text(encoding="utf-8")


@posix_only
class TestItExits:
    def test_a_missing_lexicon_fst_stops_the_script(self, tmp_path):
        """
        The case `prepare-lang` makes reachable: no OpenFST, so no binary
        `L_disambig.fst`, so nothing for `fsttablecompose` to read.
        """
        src = tmp_path / "lang"
        src.mkdir()
        (src / "G.fst").write_bytes(b"")
        done = run_guard(src)
        assert done.returncode != 0, done.stdout
        assert "REACHED-THE-END" not in done.stdout

    def test_a_missing_grammar_stops_the_script(self, tmp_path):
        src = tmp_path / "lang"
        src.mkdir()
        (src / "L_disambig.fst").write_bytes(b"")
        done = run_guard(src)
        assert done.returncode != 0, done.stdout
        assert "REACHED-THE-END" not in done.stdout

    def test_both_present_carries_on(self, tmp_path):
        """
        The other half. A guard that rejected every directory would also
        pass the two tests above, which is the one-sided test this
        repository has written before -- see the `build_graph` phone
        threshold.
        """
        src = tmp_path / "lang"
        src.mkdir()
        (src / "L_disambig.fst").write_bytes(b"")
        (src / "G.fst").write_bytes(b"")
        done = run_guard(src)
        assert done.returncode == 0, done.stderr
        assert "REACHED-THE-END" in done.stdout

    def test_it_says_where_the_lexicon_fst_comes_from(self, tmp_path):
        """
        The message is the whole value of the guard: a reader on a
        machine with no OpenFST needs to know that stage 1 skipped the
        file deliberately and which flag makes that loud.
        """
        src = tmp_path / "lang"
        src.mkdir()
        (src / "G.fst").write_bytes(b"")
        done = run_guard(src)
        combined = done.stdout + done.stderr
        assert "prepare-lang" in combined
        assert "--compile-fsts true" in combined
        assert "OpenFST" in combined

    def test_the_message_goes_to_stderr(self, tmp_path):
        """
        The recipe tees its log; a diagnostic on stdout would be
        interleaved with Kaldi's progress and read as progress.
        """
        src = tmp_path / "lang"
        src.mkdir()
        (src / "G.fst").write_bytes(b"")
        done = run_guard(src)
        assert "is missing" in done.stderr
        assert "is missing" not in done.stdout
