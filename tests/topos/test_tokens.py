"""
Tests for the token and phone symbol tables derived from a topology.

These tables used to be built by `awk` in `prepare_topo.sh`, separately for
each of the eight topologies, which meant the token inventory and the FST were
defined in two different places in two different languages. They are derived
from the spec now.

The expected values below were taken from the shell pipeline they replace,
run against the same fixture phone table, so this is a port rather than a
reinterpretation.

One thing the shell got away with by luck: it selected disambiguation symbols
with `grep "#[^\\d]"`, which works only because POSIX grep treats `\\d` inside
a bracket expression as the literal characters `\\` and `d`. Under a grep where
`\\d` is a digit class -- ugrep, or `grep -P` -- the pattern stops matching
`#0` and the tokens come out as `#0_0`, `#0_1`, which would then fail to
compile against the symbol table. Deriving the tables in Python removes that
dependence on the grep dialect.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from pathlib import Path

import pytest

from BenNevis.topos import build, spec

TESTS_DIR = Path(__file__).resolve().parent.parent
PHONES = TESTS_DIR / "fixtures" / "phones.txt"

# <eps> 0 / <SIL> 1 / AA 2 / AE 3 / B 4 / #0 5 / #1 6
PHONE_SYMBOLS = ["<eps>", "<SIL>", "AA", "AE", "B", "#0", "#1"]


@pytest.fixture
def phones():
    return build.read_phone_table(PHONES)


class TestReadPhoneTable:
    def test_only_the_symbol_column_is_read(self, phones):
        assert phones == PHONE_SYMBOLS

    def test_blank_lines_are_ignored(self, tmp_path):
        """
        A trailing newline would otherwise become a phone named "", and the
        scripts this replaces would have emitted an arc labelled with it.
        """
        path = tmp_path / "phones.txt"
        path.write_text("<eps> 0\n\nAA 1\n\n\n")
        assert build.read_phone_table(path) == ["<eps>", "AA"]


class TestTokenInventories:
    """The three tables the pipeline needs, for each topology."""

    def test_ctc_tokens_are_the_phones_themselves(self, phones):
        got = build.build_tokens(spec.load("ctc"), phones)
        assert got == ["<eps>", "<blk>", "AA", "AE", "B", "#0", "#1"]

    def test_a_two_state_topology_splits_each_phone(self, phones):
        got = build.build_tokens(spec.load("2state"), phones)
        assert got == [
            "<eps>",
            "<blk>",
            "AA_0",
            "AA_1",
            "AE_0",
            "AE_1",
            "B_0",
            "B_1",
            "#0",
            "#1",
        ]

    def test_a_three_state_topology_splits_each_phone_three_ways(self, phones):
        got = build.build_tokens(spec.load("3state-skip"), phones)
        assert got == [
            "<eps>",
            "<blk>",
            "AA_0",
            "AA_1",
            "AA_2",
            "AE_0",
            "AE_1",
            "AE_2",
            "B_0",
            "B_1",
            "B_2",
            "#0",
            "#1",
        ]

    def test_tokens_txt_drops_the_disambiguation_symbols(self, phones):
        got = build.build_tokens(spec.load("2state"), phones, with_disambig=False)
        assert got == ["<eps>", "<blk>", "AA_0", "AA_1", "AE_0", "AE_1", "B_0", "B_1"]

    def test_k2_tokens_drop_eps_and_disambiguation(self, phones):
        """
        This table *is* the network's output inventory, so its length is the
        number of output units. <eps> is not something a network can emit and
        the disambiguation symbols are not acoustic.
        """
        got = build.build_tokens(
            spec.load("2state"), phones, with_eps=False, with_disambig=False
        )
        assert got == ["<blk>", "AA_0", "AA_1", "AE_0", "AE_1", "B_0", "B_1"]

    def test_silence_is_not_a_token(self, phones):
        for name in spec.available():
            tokens = build.build_tokens(spec.load(name), phones)
            assert not any("<SIL>" in token for token in tokens), name

    @pytest.mark.parametrize("name", spec.available())
    def test_the_token_count_follows_from_the_phone_count(self, phones, name):
        """
        Three emittable phones, so the inventory is blank + 3 x the per-phone
        tokens. Pins the arithmetic the output layer size depends on.
        """
        topology = spec.load(name)
        got = build.build_tokens(topology, phones, with_eps=False, with_disambig=False)
        assert len(got) == 1 + 3 * len(topology.tokens_per_phone)

    @pytest.mark.parametrize("name", spec.available())
    def test_every_arc_input_label_appears_in_the_table(self, phones, name):
        """
        The FST and the token table must agree, or graph composition fails
        with nothing pointing back at the topology.
        """
        topology = spec.load(name)
        tokens = set(build.build_tokens(topology, phones))
        text = build.build_openfst_text(topology, "lang", phones)
        for line in text.splitlines():
            fields = line.split()
            if len(fields) >= 4:
                assert fields[2] in tokens, f"{name}: {fields[2]} missing"


class TestPhoneSymbols:
    def test_the_full_table_is_passed_through(self, phones):
        assert build.build_phone_symbols(phones) == PHONE_SYMBOLS

    def test_k2_phones_drop_the_disambiguation_symbols(self, phones):
        got = build.build_phone_symbols(phones, with_disambig=False)
        assert got == ["<eps>", "<SIL>", "AA", "AE", "B"]


class TestSymbolTableRendering:
    def test_ids_are_the_positions(self):
        assert build.write_symbol_table(["<eps>", "<blk>", "AA"]) == (
            "<eps> 0\n<blk> 1\nAA 2\n"
        )

    def test_an_empty_table_renders_as_nothing(self):
        assert build.write_symbol_table([]) == ""

    def test_symbol_ids_round_trips(self):
        symbols = ["<eps>", "<blk>", "AA"]
        assert build.symbol_ids(symbols) == {"<eps>": 0, "<blk>": 1, "AA": 2}
