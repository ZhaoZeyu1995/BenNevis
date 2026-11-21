"""
Tests for the integer-labelled OpenFST text that k2 consumes.

`Lang` used to obtain this text by compiling the symbolic text with
`fstcompile`, sorting it with `fstarcsort` and printing it back with
`fstprint | awk`. `build.build_openfst_int_text` produces it directly, which
is what takes OpenFST out of the training path.

The oracle is the golden files: they were produced by the fifteen
`get_token_fst_*.py` scripts, and the symbol substitution is reimplemented
independently here rather than being borrowed from the code under test, so a
mistake in either mapping shows up as a mismatch.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from pathlib import Path

import pytest

from BenNevis.topos import build, spec

TESTS_DIR = Path(__file__).resolve().parent.parent
GOLDEN_DIR = TESTS_DIR / "golden" / "topos"
PHONES = TESTS_DIR / "fixtures" / "phones.txt"

EPS = "<eps>"


def _outputs():
    """Golden name -> (topology name, variant name), as test_build does."""
    outputs = {}
    for name in spec.available():
        topology = spec.load(name)
        for variant in sorted(topology.variants):
            label = name if variant == "lang" else f"{name}_{variant}"
            outputs[label] = (name, variant)
    return outputs


OUTPUTS = _outputs()
TOPO_NAMES = sorted(OUTPUTS)


def tables(topology: spec.TopologySpec, variant: str, phones):
    """
    The symbol tables `prepare_topo.sh` compiles each variant against.

    The `lang` variant is compiled with `--isymbols=tokens_disambig.txt
    --osymbols=phones.txt`, and the `k2` variant with
    `--isymbols=k2/tokens.txt --osymbols=k2/phones.txt`.
    """
    if variant == "k2":
        tokens = build.build_tokens(
            topology, phones, with_eps=False, with_disambig=False
        )
        phone_symbols = build.build_phone_symbols(phones, with_disambig=False)
    else:
        tokens = build.build_tokens(topology, phones)
        phone_symbols = build.build_phone_symbols(phones)
    return build.symbol_ids(tokens), build.symbol_ids(phone_symbols)


def substitute(text: str, isymbols: dict, osymbols: dict) -> str:
    """
    Independently rewrite symbolic OpenFST text with integer labels.

    Deliberately a reimplementation rather than a call into `build`: it is the
    oracle, and `fstprint`'s conventions are what it reproduces -- a zero
    weight appended to every arc, and a final state with weight one printed as
    the state id alone.
    """
    lines = []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) == 1:
            lines.append(fields[0])
            continue
        assert len(fields) == 4, f"unexpected FST line: {line!r}"
        src, dst, ilabel, olabel = fields
        lines.append(f"{src} {dst} {isymbols[ilabel]} {osymbols[olabel]} 0.0")
    return "\n".join(lines) + "\n"


class TestAgainstGolden:
    """The integer text must match the old scripts' output, substituted."""

    @pytest.mark.parametrize("name", TOPO_NAMES)
    def test_matches_substituted_golden(self, name):
        topology_name, variant = OUTPUTS[name]
        topology = spec.load(topology_name)
        phones = build.read_phone_table(PHONES)
        isymbols, osymbols = tables(topology, variant, phones)

        golden = (GOLDEN_DIR / f"{name}.fst.txt").read_text()
        expected = substitute(golden, isymbols, osymbols)

        actual = build.build_openfst_int_text(
            topology, variant, phones, isymbols=isymbols, osymbols=osymbols
        )
        assert actual == expected

    @pytest.mark.parametrize("name", TOPO_NAMES)
    def test_same_shape_as_symbolic(self, name):
        """One line per line, and the state columns are untouched."""
        topology_name, variant = OUTPUTS[name]
        topology = spec.load(topology_name)
        phones = build.read_phone_table(PHONES)
        isymbols, osymbols = tables(topology, variant, phones)

        symbolic = build.build_openfst_text(topology, variant, phones).splitlines()
        integer = build.build_openfst_int_text(
            topology, variant, phones, isymbols=isymbols, osymbols=osymbols
        ).splitlines()

        assert len(symbolic) == len(integer)
        for before, after in zip(symbolic, integer):
            assert before.split()[:2] == after.split()[:2]


class TestWeights:
    """Every arc carries an explicit zero weight; finals carry none."""

    @pytest.mark.parametrize("name", TOPO_NAMES)
    def test_arc_and_final_field_counts(self, name):
        topology_name, variant = OUTPUTS[name]
        topology = spec.load(topology_name)
        phones = build.read_phone_table(PHONES)
        isymbols, osymbols = tables(topology, variant, phones)
        text = build.build_openfst_int_text(
            topology, variant, phones, isymbols=isymbols, osymbols=osymbols
        )

        saw_arc = False
        saw_final = False
        for line in text.splitlines():
            fields = line.split()
            if len(fields) == 1:
                saw_final = True
                int(fields[0])
            else:
                saw_arc = True
                assert len(fields) == 5, line
                assert fields[4] == "0.0", line
                for field in fields[:4]:
                    assert int(field) >= 0, line
        assert saw_arc and saw_final

    def test_text_is_newline_terminated(self):
        topology = spec.load("ctc")
        phones = build.read_phone_table(PHONES)
        isymbols, osymbols = tables(topology, "k2", phones)
        text = build.build_openfst_int_text(
            topology, "k2", phones, isymbols=isymbols, osymbols=osymbols
        )
        assert text.endswith("\n")
        assert "\n\n" not in text


class TestLabelsComeFromTheTable:
    """The ids are the table's, not the position in some recomputed list."""

    def test_shifted_table_shifts_the_labels(self):
        topology = spec.load("ctc")
        phones = build.read_phone_table(PHONES)
        isymbols, osymbols = tables(topology, "k2", phones)
        shifted = {name: index + 100 for name, index in isymbols.items()}

        plain = build.build_openfst_int_text(
            topology, "k2", phones, isymbols=isymbols, osymbols=osymbols
        )
        moved = build.build_openfst_int_text(
            topology, "k2", phones, isymbols=shifted, osymbols=osymbols
        )
        assert plain != moved
        for before, after in zip(plain.splitlines(), moved.splitlines()):
            fields_before = before.split()
            fields_after = after.split()
            if len(fields_before) == 1:
                assert fields_before == fields_after
                continue
            assert int(fields_after[2]) == int(fields_before[2]) + 100
            assert fields_after[3] == fields_before[3]

    def test_missing_input_symbol_is_named(self):
        topology = spec.load("ctc")
        phones = build.read_phone_table(PHONES)
        isymbols, osymbols = tables(topology, "k2", phones)
        del isymbols["AA"]
        with pytest.raises(KeyError, match="input symbol 'AA'"):
            build.build_openfst_int_text(
                topology, "k2", phones, isymbols=isymbols, osymbols=osymbols
            )

    def test_missing_output_symbol_is_named(self):
        topology = spec.load("ctc")
        phones = build.read_phone_table(PHONES)
        isymbols, osymbols = tables(topology, "k2", phones)
        del osymbols["AE"]
        with pytest.raises(KeyError, match="output symbol 'AE'"):
            build.build_openfst_int_text(
                topology, "k2", phones, isymbols=isymbols, osymbols=osymbols
            )

    def test_k2_variant_needs_no_eps_input(self):
        """
        The k2 input table has no `<eps>`, which is only usable because the
        k2 variant is epsilon-free on the input side. Rendering it with that
        table is therefore the real check of that invariant.
        """
        for name in spec.available():
            topology = spec.load(name)
            if "k2" not in topology.variants:
                continue
            phones = build.read_phone_table(PHONES)
            isymbols, osymbols = tables(topology, "k2", phones)
            assert EPS not in isymbols
            build.build_openfst_int_text(
                topology, "k2", phones, isymbols=isymbols, osymbols=osymbols
            )


class TestExpand:
    """`expand` is the single definition both renderers consume."""

    @pytest.mark.parametrize("name", TOPO_NAMES)
    def test_renderers_agree_with_expand(self, name):
        topology_name, variant = OUTPUTS[name]
        topology = spec.load(topology_name)
        phones = build.read_phone_table(PHONES)
        arcs, finals = build.expand(topology, variant, phones)

        lines = build.build_openfst_text(topology, variant, phones).splitlines()
        assert len(lines) == len(arcs) + len(finals)
        for arc, line in zip(arcs, lines):
            src, dst, ilabel, olabel = arc
            assert line.split() == [str(src), str(dst), ilabel, olabel]
        for state, line in zip(finals, lines[len(arcs) :]):
            assert line == str(state)

    def test_hub_is_always_final(self):
        for name in spec.available():
            topology = spec.load(name)
            phones = build.read_phone_table(PHONES)
            for variant in topology.variants:
                _, finals = build.expand(topology, variant, phones)
                assert finals[0] == 0


class TestReadSymbolTable:
    """Reading a symbol table takes the ids from the file."""

    def test_reads_names_and_ids(self, tmp_path):
        path = tmp_path / "tokens.txt"
        path.write_text("<blk> 0\nAA 1\nAE 2\n")
        assert build.read_symbol_table(path) == {"<blk>": 0, "AA": 1, "AE": 2}

    def test_ids_are_not_the_line_order(self, tmp_path):
        """A table whose ids are not 0..N-1 must not be renumbered."""
        path = tmp_path / "tokens.txt"
        path.write_text("<blk> 0\nAA 7\nAE 3\n")
        assert build.read_symbol_table(path) == {"<blk>": 0, "AA": 7, "AE": 3}

    def test_blank_lines_are_ignored(self, tmp_path):
        path = tmp_path / "tokens.txt"
        path.write_text("<blk> 0\n\nAA 1\n\n")
        assert build.read_symbol_table(path) == {"<blk>": 0, "AA": 1}

    def test_malformed_line_names_the_file_and_line(self, tmp_path):
        path = tmp_path / "tokens.txt"
        path.write_text("<blk> 0\nAA\n")
        with pytest.raises(ValueError, match=r"tokens\.txt:2"):
            build.read_symbol_table(path)

    def test_round_trips_write_symbol_table(self, tmp_path):
        topology = spec.load("3state-skip")
        phones = build.read_phone_table(PHONES)
        tokens = build.build_tokens(topology, phones)
        path = tmp_path / "tokens_disambig.txt"
        path.write_text(build.write_symbol_table(tokens))
        assert build.read_symbol_table(path) == build.symbol_ids(tokens)
