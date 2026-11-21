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
build_openfst_text
    Expand one variant into OpenFST text.
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


def build_openfst_text(
    spec: TopologySpec,
    variant_name: str,
    phones: Sequence[str],
) -> str:
    """
    Expand one variant of a topology into OpenFST text.

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
        `fstcompile` or for `k2.Fsa.from_openfst`.
    """
    variant = spec.variant(variant_name)
    blocks, disambigs = _plan(spec, variant_name, phones)
    lines: List[str] = []

    if spec.blank:
        lines.append(_format_arc(HUB_STATE, HUB_STATE, BLANK, EPS))

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
                lines.append(_format_arc(HUB_STATE, HUB_STATE, phone, phone))
            continue
        base = dict(blocks)[phone]
        for arc in variant.arcs:
            lines.append(
                _format_arc(
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
                lines.append(
                    _format_arc(
                        _resolve(arc.src, src_base),
                        _resolve(arc.dst, dst_base),
                        arc.ilabel.format(phone=dst_phone, dst_phone=dst_phone),
                        arc.olabel.format(phone=dst_phone, dst_phone=dst_phone),
                    )
                )

    finals = [HUB_STATE]
    if variant.finals == FINALS_HUB_AND_BASES:
        finals.extend(base for _, base in blocks)
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
