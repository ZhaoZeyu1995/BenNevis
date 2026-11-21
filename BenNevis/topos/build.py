"""
Expanding a topology specification into an FST and the symbol tables.

The expansion walks the phone table once, assigning each phone a contiguous
block of state ids and emitting that phone's arcs with the block's base
substituted in. The arc order is part of the output, not an accident of it:
the text produced here is compared byte for byte against the output of the
scripts this replaces, so the per-phone arcs come out in spec order, the
disambiguation self-loops at the point in the phone table where they occur,
and the inter-phone arcs after all blocks, destination-major.

The symbol tables were previously built by `awk` in `prepare_topo.sh`, once
per topology, which meant the token inventory and the FST were defined in two
different places in two different languages. They are derived from the same
spec here.

Functions
---------
read_phone_table
    Read a Kaldi-style `phones.txt`.
read_symbol_table
    Read a symbol table as a name-to-id mapping.
expand
    Expand one variant into arcs and final states.
build_openfst_text
    Render one variant as OpenFST text with symbol names.
build_openfst_int_text
    Render one variant as OpenFST text with integer labels, for k2.
build_tokens
    Derive the token inventory.
write_symbol_table
    Render a symbol table as `symbol id` lines.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from BenNevis.topos.spec import (
    BLANK,
    DISAMBIG_SELF_LOOP,
    EPS,
    FINALS_HUB_AND_BASES,
    TopologySpec,
)

HUB_STATE = 0


def is_disambig(symbol: str) -> bool:
    """
    Whether a phone-table entry is a disambiguation symbol.

    Arguments
    ---------
    symbol: str
        The entry.

    Returns
    -------
    bool
        True for `#0`, `#1`, and so on.
    """
    return symbol.startswith("#")


def read_phone_table(path: Path) -> List[str]:
    """
    Read the symbols from a Kaldi-style `phones.txt`.

    Only the symbol column is used; the ids are reassigned by whatever
    consumes the table. Blank lines are ignored, so a trailing newline does
    not become a phone -- the scripts this replaces would have produced an
    arc labelled with the empty string.

    Arguments
    ---------
    path: Path
        The phone table.

    Returns
    -------
    List[str]
        The symbols, in file order.
    """
    symbols = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            symbols.append(line.split()[0])
    return symbols


def read_symbol_table(path: Path) -> Dict[str, int]:
    """
    Read an OpenFST symbol table as a mapping from name to id.

    The ids are taken from the file rather than from the line order. They
    agree for every table BenNevis writes, but `k2/tokens.txt` fixes the
    neural network's output indices, so a table that had been edited or
    written by something else must not be silently renumbered.

    Arguments
    ---------
    path: Path
        The symbol table, in OpenFST's `symbol id` text format.

    Returns
    -------
    Dict[str, int]
        The mapping.
    """
    table: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"{path}:{number}: expected 'symbol id', got {line!r}")
            table[fields[0]] = int(fields[1])
    return table


def _plan(spec: TopologySpec, variant_name: str, phones: Sequence[str]):
    """
    Walk the phone table and decide what each entry contributes.

    Separated from the emission so that the state-id arithmetic can be
    inspected and tested on its own; it is the part that is easy to get
    subtly wrong.

    Arguments
    ---------
    spec: TopologySpec
        The topology.
    variant_name: str
        Which variant to plan for.
    phones: Sequence[str]
        The phone-table symbols.

    Returns
    -------
    Tuple[List[Tuple[str, int]], List[str]]
        The expandable phones with their block bases, and the disambiguation
        symbols that get a hub self-loop.
    """
    variant = spec.variant(variant_name)
    blocks: List[Tuple[str, int]] = []
    disambigs: List[str] = []
    next_state = 1

    for phone in phones:
        if phone in spec.skip_phones:
            continue
        if is_disambig(phone):
            if variant.disambig == DISAMBIG_SELF_LOOP:
                disambigs.append(phone)
                if variant.disambig_advances:
                    next_state += variant.stride
            continue
        blocks.append((phone, next_state))
        next_state += variant.stride

    return blocks, disambigs


def _resolve(state: Optional[int], base: int) -> int:
    """Turn an arc's state reference into a concrete state id."""
    return HUB_STATE if state is None else base + state


def _format_arc(src: int, dst: int, ilabel: str, olabel: str) -> str:
    """Render one arc as an OpenFST text line."""
    return f"{src} {dst} {ilabel} {olabel}"


def expand(
    spec: TopologySpec,
    variant_name: str,
    phones: Sequence[str],
) -> Tuple[List[Tuple[int, int, str, str]], List[int]]:
    """
    Expand one variant of a topology into arcs and final states.

    This is the single definition of a topology's shape. Both renderers below
    consume it, so the text handed to `fstcompile` and the integer text handed
    to `k2.Fsa.from_openfst` cannot describe different machines.

    Arguments
    ---------
    spec: TopologySpec
        The topology.
    variant_name: str
        "lang" or "k2".
    phones: Sequence[str]
        The phone-table symbols, in order.

    Returns
    -------
    Tuple[List[Tuple[int, int, str, str]], List[int]]
        The arcs as `(src, dst, ilabel, olabel)` with labels as symbol names,
        and the final states. The order is part of the contract: it is
        compared byte for byte against the scripts this replaces.
    """
    variant = spec.variant(variant_name)
    blocks, disambigs = _plan(spec, variant_name, phones)
    arcs: List[Tuple[int, int, str, str]] = []

    if spec.blank:
        arcs.append((HUB_STATE, HUB_STATE, BLANK, EPS))

    # Per-phone blocks, interleaved with the disambiguation self-loops at the
    # position where they appear in the phone table. Kaldi writes them last,
    # so in practice they come after every block, but the order is taken from
    # the table rather than assumed.
    emitted_disambigs = set()
    for phone in phones:
        if phone in spec.skip_phones:
            continue
        if is_disambig(phone):
            if phone in disambigs and phone not in emitted_disambigs:
                emitted_disambigs.add(phone)
                arcs.append((HUB_STATE, HUB_STATE, phone, phone))
            continue
        base = dict(blocks)[phone]
        for arc in variant.arcs:
            arcs.append(
                (
                    _resolve(arc.src, base),
                    _resolve(arc.dst, base),
                    arc.ilabel.format(phone=phone),
                    arc.olabel.format(phone=phone),
                )
            )

    # Arcs between distinct phone blocks, destination-major: for each
    # destination block, every other block in turn. CTC's cross-token
    # transitions.
    for arc in variant.inter_phone_arcs:
        for dst_phone, dst_base in blocks:
            for src_phone, src_base in blocks:
                if src_base == dst_base:
                    continue
                arcs.append(
                    (
                        _resolve(arc.src, src_base),
                        _resolve(arc.dst, dst_base),
                        arc.ilabel.format(phone=dst_phone, dst_phone=dst_phone),
                        arc.olabel.format(phone=dst_phone, dst_phone=dst_phone),
                    )
                )

    finals = [HUB_STATE]
    if variant.finals == FINALS_HUB_AND_BASES:
        finals.extend(base for _, base in blocks)

    return arcs, finals


def build_openfst_text(
    spec: TopologySpec,
    variant_name: str,
    phones: Sequence[str],
) -> str:
    """
    Expand one variant of a topology into OpenFST text, with symbol names.

    Arguments
    ---------
    spec: TopologySpec
        The topology.
    variant_name: str
        "lang" or "k2".
    phones: Sequence[str]
        The phone-table symbols, in order.

    Returns
    -------
    str
        The FST in OpenFST's text format, newline-terminated, ready for
        `fstcompile` with the matching `--isymbols` and `--osymbols`.
    """
    arcs, finals = expand(spec, variant_name, phones)
    lines = [_format_arc(*arc) for arc in arcs]
    lines.extend(str(state) for state in finals)
    return "\n".join(lines) + "\n"


def _lookup(symbol: str, table: Dict[str, int], which: str, topology: str) -> int:
    """
    Resolve one symbol to its id, naming the table in the error.

    Arguments
    ---------
    symbol: str
        The symbol to resolve.
    table: Dict[str, int]
        The symbol table.
    which: str
        Which table it is, for the error message.
    topology: str
        The topology name, for the error message.

    Returns
    -------
    int
        The symbol's id.
    """
    if symbol not in table:
        raise KeyError(
            f"topology {topology!r}: {which} symbol {symbol!r} is not in the "
            f"symbol table, so the FST and the table disagree"
        )
    return table[symbol]


def build_openfst_int_text(
    spec: TopologySpec,
    variant_name: str,
    phones: Sequence[str],
    *,
    isymbols: Dict[str, int],
    osymbols: Dict[str, int],
) -> str:
    """
    Expand one variant into OpenFST text with integer labels and weights.

    This is the form `k2.Fsa.from_openfst` wants, and it is what the pipeline
    used to obtain by compiling the symbolic text with `fstcompile` and
    printing it back with `fstprint`, piped through an `awk` that appended a
    zero weight to every arc. Producing it directly removes OpenFST from the
    training path: nothing about the round trip was adding information.

    Arc order is preserved rather than sorted. The old path ran
    `fstarcsort --sort_type=olabel`, which `k2.compose` discards anyway --
    it arc-sorts its first argument itself.

    Arguments
    ---------
    spec: TopologySpec
        The topology.
    variant_name: str
        "lang" or "k2".
    phones: Sequence[str]
        The phone-table symbols, in order.
    isymbols: Dict[str, int]
        The input symbol table, mapping token name to id. For the `k2`
        variant this is `k2/tokens.txt`, whose ids are the network's output
        indices, so it must be the table on disc rather than one recomputed
        here.
    osymbols: Dict[str, int]
        The output symbol table, mapping phone name to id.

    Returns
    -------
    str
        The FST in OpenFST's text format with integer labels, newline
        terminated.

    Raises
    ------
    KeyError
        If an arc uses a symbol that is not in the relevant table.
    """
    arcs, finals = expand(spec, variant_name, phones)
    lines = [
        f"{src} {dst} "
        f"{_lookup(ilabel, isymbols, 'input', spec.name)} "
        f"{_lookup(olabel, osymbols, 'output', spec.name)} 0.0"
        for src, dst, ilabel, olabel in arcs
    ]
    lines.extend(str(state) for state in finals)
    return "\n".join(lines) + "\n"


def build_tokens(
    spec: TopologySpec,
    phones: Sequence[str],
    *,
    with_eps: bool = True,
    with_disambig: bool = True,
) -> List[str]:
    """
    Derive a token inventory from a phone table.

    The three tables the pipeline needs are the three combinations of the two
    flags that are actually used:

    - `tokens_disambig.txt`: `with_eps=True, with_disambig=True`, the input
      symbols of `T.fst`.
    - `tokens.txt`: `with_eps=True, with_disambig=False`.
    - `k2/tokens.txt`: `with_eps=False, with_disambig=False`, which is exactly
      the neural network's output inventory, so its length is the number of
      output units.

    Arguments
    ---------
    spec: TopologySpec
        The topology.
    phones: Sequence[str]
        The phone-table symbols.
    with_eps: bool
        Whether to include `<eps>` at position 0.
    with_disambig: bool
        Whether to append the disambiguation symbols.

    Returns
    -------
    List[str]
        The token symbols, in order; the index of each is its id.
    """
    tokens: List[str] = []
    if with_eps:
        tokens.append(EPS)
    if spec.blank:
        tokens.append(BLANK)

    for phone in phones:
        if phone in spec.skip_phones or is_disambig(phone):
            continue
        tokens.extend(spec.tokens_for_phone(phone))

    if with_disambig:
        tokens.extend(phone for phone in phones if is_disambig(phone))

    return tokens


def build_phone_symbols(
    phones: Sequence[str], *, with_disambig: bool = True
) -> List[str]:
    """
    The output symbol table for a topology FST.

    Arguments
    ---------
    phones: Sequence[str]
        The phone-table symbols.
    with_disambig: bool
        Whether to keep the disambiguation symbols. `k2/phones.txt` drops
        them; `phones.txt` keeps them.

    Returns
    -------
    List[str]
        The symbols, in order.
    """
    if with_disambig:
        return list(phones)
    return [phone for phone in phones if not is_disambig(phone)]


def write_symbol_table(symbols: Sequence[str]) -> str:
    """
    Render a symbol table as OpenFST's `symbol id` text.

    Arguments
    ---------
    symbols: Sequence[str]
        The symbols, in id order.

    Returns
    -------
    str
        The table, newline-terminated.
    """
    return "".join(f"{symbol} {index}\n" for index, symbol in enumerate(symbols))


def symbol_ids(symbols: Sequence[str]) -> Dict[str, int]:
    """
    Map each symbol to its id.

    Arguments
    ---------
    symbols: Sequence[str]
        The symbols, in id order.

    Returns
    -------
    Dict[str, int]
        The mapping.
    """
    return {symbol: index for index, symbol in enumerate(symbols)}
