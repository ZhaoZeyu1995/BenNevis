"""
Compiling OpenFST text, and the flags that decide what the binary is.

Tier 1: the invocation is built in Python and `subprocess.run` is
replaced, so this needs no OpenFST -- which is the point. OpenFST is on
PATH in the Kaldi image and nowhere else this runs, so a test that needed
it would only ever run in tier 3, a long way from the code.

What is pinned is the **argument list**, because every one of its parts
changes the file that comes out and none of them fails loudly:

* `--keep_isymbols=false` -- with the tables attached the binary carries
  them and is a different file, which Kaldi's own stages neither write
  nor expect;
* `--sort_type=olabel` -- `fsttablecompose` requires its first argument
  sorted on the output side, and an unsorted graph composes to the wrong
  thing rather than to an error;
* two `subprocess` calls rather than a shell pipeline, so a failure in
  `fstcompile` cannot be hidden by `fstarcsort` succeeding -- the same
  trap as the nightly's `| tee` and `prepare_graph.sh`'s `echo || exit`.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import subprocess

import pytest

from BenNevis.utils import openfst


class Recorder:
    """Stands in for `subprocess.run`, recording each call."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.fail_on and argv[0] == self.fail_on:
            raise subprocess.CalledProcessError(1, argv)
        return subprocess.CompletedProcess(argv, 0, stdout=b"binary", stderr=b"")


@pytest.fixture
def recorder(monkeypatch):
    recorder = Recorder()
    monkeypatch.setattr(openfst.subprocess, "run", recorder)
    return recorder


class TestAvailability:
    def test_both_binaries_are_required(self, monkeypatch):
        """
        A tree with one and not the other would fail halfway and leave a
        truncated FST, so `available()` is an and rather than an or.
        """
        monkeypatch.setattr(
            openfst.shutil,
            "which",
            lambda name: "/bin/fstcompile" if name == "fstcompile" else None,
        )
        assert not openfst.available()

    def test_it_is_true_when_both_are_there(self, monkeypatch):
        monkeypatch.setattr(openfst.shutil, "which", lambda name: f"/bin/{name}")
        assert openfst.available()

    def test_the_names_are_the_two_it_runs(self):
        assert openfst.BINARIES == ("fstcompile", "fstarcsort")


class TestTheInvocation:
    def test_numeric_text_needs_no_symbol_tables(self, recorder, tmp_path):
        """
        `BenNevis.lm.lexicon` writes numeric text, byte-identical to
        `fstprint`'s output, so compiling it without tables gives the
        same binary as Kaldi's symbolic text compiled with them. Passing
        tables it does not need would be harmless and passing
        `--keep_isymbols` without tables would not be.
        """
        openfst.compile_fst("0 1 2 3\n1\n", tmp_path / "out.fst")
        (compile_argv, _), (sort_argv, _) = recorder.calls
        assert compile_argv == ["fstcompile"]
        assert sort_argv == ["fstarcsort", "--sort_type=olabel"]

    def test_symbolic_text_drops_the_tables_from_the_binary(self, recorder, tmp_path):
        openfst.compile_fst(
            "0 1 A x\n1\n",
            tmp_path / "out.fst",
            isymbols=tmp_path / "i.txt",
            osymbols=tmp_path / "o.txt",
        )
        compile_argv = recorder.calls[0][0]
        assert f"--isymbols={tmp_path / 'i.txt'}" in compile_argv
        assert f"--osymbols={tmp_path / 'o.txt'}" in compile_argv
        assert "--keep_isymbols=false" in compile_argv
        assert "--keep_osymbols=false" in compile_argv

    def test_keeping_the_symbols_is_possible_and_not_the_default(
        self, recorder, tmp_path
    ):
        """
        `run/prepare_topo.py` keeps them for `k2/T.fst` and drops them
        for `T.fst`, so both have to work.
        """
        openfst.compile_fst(
            "0 1 A x\n1\n",
            tmp_path / "out.fst",
            isymbols=tmp_path / "i.txt",
            osymbols=tmp_path / "o.txt",
            keep_symbols=True,
        )
        compile_argv = recorder.calls[0][0]
        assert "--keep_isymbols=false" not in compile_argv

    def test_the_text_reaches_fstcompile_and_the_output_is_arcsorted(
        self, recorder, tmp_path
    ):
        """
        The order matters: compile, then sort. Sorting text would do
        nothing and compiling a sorted binary again is not possible.
        """
        openfst.compile_fst("0 1 2 3\n1\n", tmp_path / "out.fst")
        compile_kwargs = recorder.calls[0][1]
        sort_kwargs = recorder.calls[1][1]
        assert compile_kwargs["input"] == b"0 1 2 3\n1\n"
        assert sort_kwargs["input"] == b"binary"
        assert (tmp_path / "out.fst").read_bytes() == b"binary"

    def test_both_calls_check_their_status(self, recorder, tmp_path):
        openfst.compile_fst("0 1 2 3\n1\n", tmp_path / "out.fst")
        for _argv, kwargs in recorder.calls:
            assert kwargs["check"] is True

    def test_a_different_sort_type_is_possible(self, recorder, tmp_path):
        openfst.compile_fst("0\n", tmp_path / "out.fst", sort_type="ilabel")
        assert recorder.calls[1][0] == ["fstarcsort", "--sort_type=ilabel"]


class TestFailuresAreNotSwallowed:
    def test_a_failing_compile_does_not_reach_the_sort(self, monkeypatch, tmp_path):
        """
        A shell pipeline would report the sort's status, which is the
        defect `set -o pipefail` exists for and the one the nightly's
        `| tee` had. Two calls with `check=True` cannot do that.
        """
        recorder = Recorder(fail_on="fstcompile")
        monkeypatch.setattr(openfst.subprocess, "run", recorder)
        with pytest.raises(subprocess.CalledProcessError):
            openfst.compile_fst("nonsense", tmp_path / "out.fst")
        assert len(recorder.calls) == 1
        assert not (tmp_path / "out.fst").exists()

    def test_a_failing_sort_leaves_no_output(self, monkeypatch, tmp_path):
        recorder = Recorder(fail_on="fstarcsort")
        monkeypatch.setattr(openfst.subprocess, "run", recorder)
        with pytest.raises(subprocess.CalledProcessError):
            openfst.compile_fst("0\n", tmp_path / "out.fst")
        assert not (tmp_path / "out.fst").exists()

    @pytest.mark.skipif(
        openfst.available(),
        reason="this is the no-OpenFST case; inside the Kaldi image "
        "fstcompile exists and fails differently",
    )
    def test_a_missing_binary_raises_rather_than_returning(self, tmp_path):
        """
        Callers test `available()` first and decide whether that is
        fatal. Reaching here with no OpenFST is a caller's mistake, and a
        silent no-op would leave the stage that needs the binary to fail
        with something unrelated instead.
        """
        with pytest.raises(FileNotFoundError):
            openfst.compile_fst("0\n", tmp_path / "out.fst")
