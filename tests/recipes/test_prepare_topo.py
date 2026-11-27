"""
`run/prepare_topo.py`: stage 3 without Kaldi.

Tier 1. The shell version needed Kaldi for two things that had nothing to
do with what it produced: it sourced **Kaldi's `utils/parse_options.sh`**
to read its own flags, and `utils/` is a symlink into `tools/kaldi`, so it
could not parse `--topo` without a Kaldi checkout; and it ran `fstcompile`
unconditionally for `T.fst` and `k2/T.fst`, which **nothing in the Python
lane reads**. So stage 3 failed without Kaldi over an argument parser and
an artefact, long after everything it produces had become Python.

The most important test here is not that it runs. It is that its symbol
tables are **byte-identical to `python3 -m BenNevis.topos`**, because
moving the logic into a second entry point is exactly how two
implementations of one thing come to disagree -- and the thing they would
disagree about is the network's output indices, which a checkpoint is only
valid while they stay put.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "BenNevis" / "run" / "prepare_topo.py"

sys.path.insert(0, str(REPO_ROOT))

from BenNevis.run.prepare_topo import main  # noqa: E402

PHONES = "<eps> 0\nSIL 1\nY 2\nN 3\n#0 4\n#1 5\n"
WORDS = "<eps> 0\n<SIL> 1\nYES 2\nNO 3\n#0 4\n"
LEXICON = "<SIL> SIL\nYES Y\nNO N\n"

#: Every topology, since the tables differ per topology and a drift
#: between the two entry points could easily affect only one of them.
TOPOLOGIES = ["ctc", "2state", "2state-1", "3state-skip", "mmictc"]


@pytest.fixture
def source(tmp_path):
    src = tmp_path / "lang"
    (src / "phones").mkdir(parents=True)
    (src / "phones.txt").write_text(PHONES, encoding="utf-8")
    (src / "words.txt").write_text(WORDS, encoding="utf-8")
    (src / "phones" / "align_lexicon.int").write_text("2 2 2\n3 3 3\n")
    dictionary = tmp_path / "dict"
    dictionary.mkdir()
    (dictionary / "lexicon.txt").write_text(LEXICON, encoding="utf-8")
    (dictionary / "optional_silence.txt").write_text("SIL\n", encoding="utf-8")
    return src


@pytest.fixture
def no_openfst(monkeypatch, tmp_path):
    """A PATH with no OpenFST on it, whatever the developer has installed."""
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    return empty


class TestItRunsWithoutOpenFst:
    def test_it_writes_everything_the_python_lane_reads(
        self, source, tmp_path, no_openfst
    ):
        des = tmp_path / "out"
        assert (
            main(
                [
                    "--topo",
                    "ctc",
                    "--lexicon",
                    str(tmp_path / "dict" / "lexicon.txt"),
                    "--sil-prob",
                    "0.0",
                    str(source),
                    str(des),
                ]
            )
            == 0
        )
        for name in (
            "phones.txt",
            "words.txt",
            "tokens.txt",
            "tokens_disambig.txt",
            "disambig.int",
            "k2/tokens.txt",
            "k2/phones.txt",
            "k2/topo.yaml",
            "k2/L.fst.txt",
        ):
            assert (des / name).is_file(), name
            assert (des / name).read_text(encoding="utf-8").strip(), f"{name} is empty"
        # Everything Lang reads is there, and the two FSTs are not.
        assert not (des / "T.fst").exists()
        assert not (des / "k2" / "T.fst").exists()

    def test_the_source_directory_is_copied_not_moved(
        self, source, tmp_path, no_openfst
    ):
        des = tmp_path / "out"
        main(["--topo", "ctc", str(source), str(des)])
        assert (source / "phones.txt").is_file()
        # Including the parts of it nothing here writes, which is how
        # `phones/align_lexicon.int` reaches the training path.
        assert (des / "phones" / "align_lexicon.int").is_file()

    def test_it_says_what_it_skipped(self, source, tmp_path, no_openfst, caplog):
        with caplog.at_level("INFO"):
            main(["--topo", "ctc", str(source), str(tmp_path / "out")])
        assert "not on PATH" in caplog.text
        assert "prepare_graph.sh" in caplog.text

    def test_an_existing_destination_is_replaced(self, source, tmp_path, no_openfst):
        des = tmp_path / "out"
        des.mkdir()
        (des / "stale.txt").write_text("from a previous topology", encoding="utf-8")
        main(["--topo", "ctc", str(source), str(des)])
        assert not (des / "stale.txt").exists()


class TestTheTablesCannotDriftFromTheOtherEntryPoint:
    """
    Byte-identical to `python3 -m BenNevis.topos`, for every topology.

    Two entry points computing the network's output indices is the setup
    for a silent disagreement, and a checkpoint is only valid while those
    indices stay put -- deriving them a second way is what made the first
    real k2 decoding run read `<blk>` as `<eps>`.
    """

    @staticmethod
    def topos_cli(*argv):
        result = subprocess.run(
            [sys.executable, "-m", "BenNevis.topos", *argv],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    @pytest.mark.parametrize("topo", TOPOLOGIES)
    def test_every_table_matches(self, topo, source, tmp_path, no_openfst):
        des = tmp_path / f"out-{topo}"
        assert main(["--topo", topo, str(source), str(des)]) == 0
        phones = str(source / "phones.txt")
        k2_phones = str(des / "k2" / "phones.txt")
        for name, argv in [
            ("tokens_disambig.txt", ("tokens", topo, phones)),
            ("tokens.txt", ("tokens", topo, phones, "--no-disambig")),
            (
                "k2/tokens.txt",
                ("tokens", topo, phones, "--no-eps", "--no-disambig"),
            ),
            ("disambig.int", ("disambig", topo, phones)),
            ("k2/phones.txt", ("phones", phones, "--no-disambig")),
            ("k2/topo.yaml", ("spec", topo)),
        ]:
            assert (des / name).read_text(encoding="utf-8") == self.topos_cli(
                *argv
            ), f"{topo}: {name} differs from `python3 -m BenNevis.topos`"
        # And the one it does not take from that CLI, for completeness.
        assert (des / "k2" / "phones.txt").read_text(encoding="utf-8") == (
            self.topos_cli("phones", k2_phones, "--no-disambig")
        )


class TestCompileFsts:
    def test_true_without_openfst_fails_rather_than_skipping(
        self, source, tmp_path, no_openfst
    ):
        assert (
            main(
                [
                    "--topo",
                    "ctc",
                    "--compile-fsts",
                    "true",
                    str(source),
                    str(tmp_path / "o"),
                ]
            )
            == 1
        )

    def test_false_skips_even_when_openfst_is_there(
        self, source, tmp_path, monkeypatch
    ):
        """
        Checked with stubs on PATH, so the flag is known to be what decides
        rather than the environment the test happens to run in.
        """
        stubs = tmp_path / "bin"
        stubs.mkdir()
        for name in ("fstcompile", "fstarcsort"):
            tool = stubs / name
            tool.write_text("#!/bin/sh\ncat\n", encoding="utf-8")
            tool.chmod(0o755)
        monkeypatch.setenv("PATH", f"{stubs}:{os.environ['PATH']}")
        des = tmp_path / "out"
        assert (
            main(["--topo", "ctc", "--compile-fsts", "false", str(source), str(des)])
            == 0
        )
        assert not (des / "T.fst").exists()

    @pytest.mark.skipif(os.name == "nt", reason="the recipes are POSIX only")
    def test_auto_compiles_when_openfst_is_there(self, source, tmp_path, monkeypatch):
        stubs = tmp_path / "bin"
        stubs.mkdir()
        for name in ("fstcompile", "fstarcsort"):
            tool = stubs / name
            tool.write_text("#!/bin/sh\ncat\n", encoding="utf-8")
            tool.chmod(0o755)
        monkeypatch.setenv("PATH", f"{stubs}:{os.environ['PATH']}")
        des = tmp_path / "out"
        assert main(["--topo", "ctc", str(source), str(des)]) == 0
        # The stubs pass text through, so the files exist and hold the
        # topology text rather than a compiled FST -- which is enough to
        # show the tools were invoked at all.
        assert (des / "T.fst").is_file()
        assert (des / "k2" / "T.fst").is_file()
        assert (des / "T.fst").stat().st_size > 0


class TestItRefusesBadInput:
    def test_an_unknown_topology(self, source, tmp_path, no_openfst):
        assert main(["--topo", "nonsense", str(source), str(tmp_path / "o")]) == 1

    def test_a_missing_source(self, tmp_path, no_openfst):
        assert main(["--topo", "ctc", str(tmp_path / "nope"), str(tmp_path / "o")]) == 1

    def test_a_missing_lexicon(self, source, tmp_path, no_openfst):
        assert (
            main(
                [
                    "--topo",
                    "ctc",
                    "--lexicon",
                    str(tmp_path / "nope.txt"),
                    str(source),
                    str(tmp_path / "o"),
                ]
            )
            == 1
        )


class TestTheShimAndItsCallers:
    def test_the_shim_execs_the_python_and_sources_no_kaldi(self):
        """
        It must not source `path.sh` or Kaldi's `parse_options.sh`: the
        whole point is that this stage needs no Kaldi on PATH.
        """
        text = (REPO_ROOT / "BenNevis" / "run" / "prepare_topo.sh").read_text(
            encoding="utf-8"
        )
        assert "prepare_topo.py" in text
        assert "parse_options.sh" not in re.sub(r"^#.*$", "", text, flags=re.M)
        assert "path.sh" not in re.sub(r"^#.*$", "", text, flags=re.M)

    @pytest.mark.skipif(os.name == "nt", reason="the recipes are POSIX only")
    def test_the_shim_passes_its_arguments_through(self, source, tmp_path):
        """
        The shim's own PATH is left alone: whether OpenFST is absent is
        tested above, and stripping PATH down here only took `bash` and
        `dirname` away with it.
        """
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("no bash")
        des = tmp_path / "out"
        result = subprocess.run(
            [
                bash,
                str(REPO_ROOT / "BenNevis" / "run" / "prepare_topo.sh"),
                "--topo",
                "ctc",
                str(source),
                str(des),
            ],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHONPATH": str(REPO_ROOT),
                "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
            },
        )
        assert result.returncode == 0, result.stderr
        assert (des / "k2" / "topo.yaml").is_file()

    def test_prepare_graph_requires_the_compiled_fst(self):
        """
        Kaldi's lane still composes against `T.fst`, so it must ask for it
        explicitly and say so clearly when stage 3 skipped it -- otherwise
        the failure is `fsttablecompose` complaining about a missing file
        four stages from the cause.
        """
        text = (REPO_ROOT / "BenNevis" / "run" / "prepare_graph.sh").read_text(
            encoding="utf-8"
        )
        assert "--compile-fsts true" in text
        assert "T.fst was not written" in text
