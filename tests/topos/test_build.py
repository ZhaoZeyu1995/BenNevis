"""
Tests for the declarative topology specifications in BenNevis/topos/.

These lock down the semantics of every topology in two ways:

1. Golden-file comparison. The fifteen `get_token_fst_*.py` scripts that used
   to define these topologies are gone, but their output is preserved in
   tests/golden/topos/ and compared byte for byte, so the rewrite is pinned to
   what the scripts actually produced rather than to what the specs claim.
   Regenerate deliberately with tests/regenerate_topo_golden.py and review the
   diff; a change there is a change to a topology, not a formatting detail.
2. Structural invariants. These hold for any phone table and would catch a
   semantic break even if the golden files were regenerated carelessly.

The golden files keep the names the scripts had: `<topo>` for the variant that
builds `T.fst` and `<topo>_k2` for the one that builds `k2/T.fst`.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from pathlib import Path

import pytest

from BenNevis.topos import build, spec

TESTS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TESTS_DIR.parent
GOLDEN_DIR = TESTS_DIR / "golden" / "topos"
PHONES = TESTS_DIR / "fixtures" / "phones.txt"

# Symbols that are structural rather than emittable units.
EPS = "<eps>"
BLANK = "<blk>"
SILENCE = "<SIL>"


def _outputs():
    """
    Every FST the specs can produce, keyed by the name of its golden file.

    Returns
    -------
    Dict[str, Tuple[str, str]]
        Golden name -> (topology name, variant name).
    """
    outputs = {}
    for name in spec.available():
        topology = spec.load(name)
        for variant in sorted(topology.variants):
            label = name if variant == "lang" else f"{name}_{variant}"
            outputs[label] = (name, variant)
    return outputs


OUTPUTS = _outputs()
TOPO_NAMES = sorted(OUTPUTS)


def generate(name: str, phones: Path = PHONES) -> str:
    """Expand one topology output and return its OpenFST text."""
    topology_name, variant = OUTPUTS[name]
    return build.build_openfst_text(
        spec.load(topology_name), variant, build.read_phone_table(phones)
    )


def parse_arcs(text: str):
    """
    Split OpenFST text output into arcs and final states.

    Returns
    -------
    arcs: list of (src, dst, ilabel, olabel)
    finals: list of int
    """
    arcs = []
    finals = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) == 1:
            finals.append(int(fields[0]))
        elif len(fields) == 2:
            # final state with a weight
            finals.append(int(fields[0]))
        elif len(fields) in (4, 5):
            src, dst, ilabel, olabel = fields[0], fields[1], fields[2], fields[3]
            arcs.append((int(src), int(dst), ilabel, olabel))
        else:
            raise AssertionError(f"Unparseable FST line: {line!r}")
    return arcs, finals


def phone_table(path: Path = PHONES):
    """Return the phone symbols from a Kaldi-style symbol table."""
    return [line.split()[0] for line in path.read_text().splitlines() if line.strip()]


def emittable_phones(path: Path = PHONES):
    """Return the phones a topology is expected to model, excluding structure."""
    return [
        p
        for p in phone_table(path)
        if p not in (EPS, SILENCE) and not p.startswith("#")
    ]


def test_every_topology_output_has_a_golden_file():
    """A new topology must arrive with its golden file, or it is untested."""
    have = {p.name[: -len(".fst.txt")] for p in GOLDEN_DIR.glob("*.fst.txt")}
    assert set(TOPO_NAMES) == have, (
        f"Topology outputs without golden files: {sorted(set(TOPO_NAMES) - have)}; "
        f"golden files without topology outputs: {sorted(have - set(TOPO_NAMES))}. "
        "Run tests/regenerate_topo_golden.py."
    )


@pytest.mark.parametrize("name", TOPO_NAMES)
def test_matches_golden(name):
    """The generator reproduces its recorded output exactly."""
    golden = (GOLDEN_DIR / f"{name}.fst.txt").read_text()
    assert generate(name) == golden, (
        f"Topology {name} no longer matches its golden file. If this change is "
        "intended, rerun tests/regenerate_topo_golden.py and review the diff."
    )


@pytest.mark.parametrize("name", TOPO_NAMES)
def test_output_is_parseable_openfst_text(name):
    """Every line is either an arc or a final state."""
    arcs, finals = parse_arcs(generate(name))
    assert arcs, f"{name} produced no arcs"
    assert finals, f"{name} declared no final state"


@pytest.mark.parametrize("name", TOPO_NAMES)
def test_start_state_is_the_hub(name):
    """
    State 0 is the shared hub: every topology both leaves it and returns to it,
    and it is final so that an empty token sequence is accepted.
    """
    arcs, finals = parse_arcs(generate(name))
    assert any(src == 0 for src, _, _, _ in arcs), f"{name} never leaves state 0"
    assert any(dst == 0 for _, dst, _, _ in arcs), f"{name} never returns to state 0"
    assert 0 in finals, f"{name} does not make state 0 final"


@pytest.mark.parametrize("name", TOPO_NAMES)
def test_all_states_reachable_from_the_hub(name):
    """An unreachable state is dead weight and usually a numbering bug."""
    arcs, _ = parse_arcs(generate(name))
    outgoing = {}
    for src, dst, _, _ in arcs:
        outgoing.setdefault(src, set()).add(dst)
    seen = {0}
    stack = [0]
    while stack:
        for nxt in outgoing.get(stack.pop(), ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    states = {s for s, _, _, _ in arcs} | {d for _, d, _, _ in arcs}
    assert (
        states == seen
    ), f"{name} has states unreachable from 0: {sorted(states - seen)}"


@pytest.mark.parametrize("name", TOPO_NAMES)
def test_every_phone_is_emittable(name):
    """
    Each modelled phone appears as an output label, so no phone is silently
    droppable from the lexicon.
    """
    arcs, _ = parse_arcs(generate(name))
    produced = {olabel for _, _, _, olabel in arcs}
    for phone in emittable_phones():
        assert phone in produced, f"{name} cannot output phone {phone}"


@pytest.mark.parametrize("name", [n for n in TOPO_NAMES if n.endswith("_k2")])
def test_k2_variants_have_no_epsilon_input_labels(name):
    """
    The defining property of the _k2 variants, per BenNevis/bin/topos/README.md:
    their input side is epsilon-free so that it matches the neural network's
    output vocabulary directly.
    """
    arcs, _ = parse_arcs(generate(name))
    offenders = [a for a in arcs if a[2] == EPS]
    assert not offenders, f"{name} has <eps> input labels: {offenders[:5]}"


@pytest.mark.parametrize("name", [n for n in TOPO_NAMES if not n.endswith("_k2")])
def test_non_k2_variants_pass_disambiguation_symbols_through(name):
    """
    The decoding-graph variants must keep #N symbols as hub self-loops so that
    T.fst can be composed with L_disambig.fst.
    """
    arcs, _ = parse_arcs(generate(name))
    disambig = [p for p in phone_table() if p.startswith("#")]
    for sym in disambig:
        assert (0, 0, sym, sym) in arcs, f"{name} drops disambiguation symbol {sym}"


def test_blank_self_loop_on_hub():
    """
    Every topology uses a blank self-loop on the hub to absorb frames between
    tokens. CTC is included: that loop is what makes it CTC.
    """
    for name in TOPO_NAMES:
        arcs, _ = parse_arcs(generate(name))
        assert (0, 0, BLANK, EPS) in arcs, f"{name} has no <blk> self-loop on state 0"


@pytest.mark.parametrize("name", TOPO_NAMES)
def test_output_is_deterministic(name):
    """Two runs agree; the generators must not depend on dict or set ordering."""
    assert generate(name) == generate(name)
