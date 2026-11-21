"""
Tests for the topology command line interface.

`prepare_topo.sh` calls this instead of the fifteen scripts it used to, so its
output format is a contract: `fst` must print exactly what the old scripts
printed, since it feeds the same `fstcompile` pipeline.

Invoked in-process rather than through a subprocess, so the tests stay in
tier 1 and run everywhere, and a traceback points at the real frame.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from pathlib import Path

import pytest

from BenNevis.topos import __main__ as cli
from BenNevis.topos import spec

TESTS_DIR = Path(__file__).resolve().parent.parent
PHONES = str(TESTS_DIR / "fixtures" / "phones.txt")
GOLDEN_DIR = TESTS_DIR / "golden" / "topos"


def run(capsys, *argv):
    """Run the CLI and return its stdout."""
    status = cli.main(list(argv))
    assert status == 0
    return capsys.readouterr().out


class TestFst:
    @pytest.mark.parametrize("name", spec.available())
    def test_the_lang_variant_matches_its_golden(self, capsys, name):
        """
        The output is byte-compared against what the scripts produced, because
        it is piped straight into fstcompile.
        """
        out = run(capsys, "fst", name, PHONES)
        assert out == (GOLDEN_DIR / f"{name}.fst.txt").read_text()

    @pytest.mark.parametrize("name", spec.available())
    def test_the_k2_variant_matches_its_golden(self, capsys, name):
        out = run(capsys, "fst", name, PHONES, "--variant", "k2")
        assert out == (GOLDEN_DIR / f"{name}_k2.fst.txt").read_text()

    def test_lang_is_the_default_variant(self, capsys):
        explicit = run(capsys, "fst", "2state", PHONES, "--variant", "lang")
        default = run(capsys, "fst", "2state", PHONES)
        assert default == explicit

    def test_an_unknown_variant_is_refused(self, capsys):
        with pytest.raises(SystemExit):
            cli.main(["fst", "2state", PHONES, "--variant", "nope"])

    def test_an_unknown_topology_is_refused(self):
        with pytest.raises(FileNotFoundError, match="available:"):
            cli.main(["fst", "no-such-topo", PHONES])


class TestTokens:
    def test_the_default_table_is_tokens_disambig(self, capsys):
        out = run(capsys, "tokens", "2state", PHONES)
        assert out.splitlines()[0] == "<eps> 0"
        assert out.splitlines()[-1] == "#1 9"

    def test_no_disambig_gives_tokens_txt(self, capsys):
        out = run(capsys, "tokens", "2state", PHONES, "--no-disambig")
        assert "#0" not in out
        assert out.splitlines()[0] == "<eps> 0"

    def test_no_eps_and_no_disambig_gives_the_k2_table(self, capsys):
        out = run(capsys, "tokens", "2state", PHONES, "--no-eps", "--no-disambig")
        assert out.splitlines()[0] == "<blk> 0"
        assert "<eps>" not in out
        assert "#0" not in out

    def test_the_ids_are_contiguous_from_zero(self, capsys):
        out = run(capsys, "tokens", "3state-skip-1", PHONES)
        ids = [int(line.split()[1]) for line in out.splitlines()]
        assert ids == list(range(len(ids)))


class TestPhones:
    def test_the_table_is_renumbered_from_zero(self, capsys):
        out = run(capsys, "phones", PHONES)
        assert out.splitlines()[0] == "<eps> 0"
        assert out.splitlines()[-1] == "#1 6"

    def test_no_disambig_gives_the_k2_table(self, capsys):
        out = run(capsys, "phones", PHONES, "--no-disambig")
        assert "#" not in out
        assert out.splitlines()[-1] == "B 4"


class TestListAndDescribe:
    def test_list_prints_every_topology(self, capsys):
        out = run(capsys, "list")
        assert set(out.split()) == set(spec.available())

    def test_describe_prints_the_summary_and_the_tokens(self, capsys):
        out = run(capsys, "describe", "3state-skip")
        assert "3state-skip" in out
        assert "{phone}_2" in out
        assert "lang" in out and "k2" in out

    def test_no_command_is_refused(self):
        with pytest.raises(SystemExit):
            cli.main([])


class TestRoundTrip:
    def test_the_fst_and_the_token_table_are_consistent(self, capsys):
        """
        The two commands prepare_topo.sh calls must agree: every input label
        in the FST has to exist in the token table it is compiled against.
        """
        for name in spec.available():
            fst = run(capsys, "fst", name, PHONES)
            table = run(capsys, "tokens", name, PHONES)
            known = {line.split()[0] for line in table.splitlines()}
            for line in fst.splitlines():
                fields = line.split()
                if len(fields) >= 4:
                    assert fields[2] in known, f"{name}: {fields[2]} not in tokens"

    def test_the_k2_fst_matches_the_k2_token_table(self, capsys):
        for name in spec.available():
            fst = run(capsys, "fst", name, PHONES, "--variant", "k2")
            table = run(capsys, "tokens", name, PHONES, "--no-eps", "--no-disambig")
            known = {line.split()[0] for line in table.splitlines()}
            for line in fst.splitlines():
                fields = line.split()
                if len(fields) >= 4:
                    assert fields[2] in known, f"{name}: {fields[2]} not in k2 tokens"
