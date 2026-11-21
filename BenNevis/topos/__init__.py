"""
Declarative topology definitions.

A topology is described by a YAML file in `specs/` and expanded into an FST
and the matching symbol tables by `build`. This replaces the fifteen
`get_token_fst_*.py` scripts, which each printed OpenFST text directly and
shared their logic by copy and paste.

Each spec covers one topology family and both of the FSTs it produces: the
`lang` variant for `T.fst`, whose input symbols include `<eps>` and the
disambiguation symbols, and the `k2` variant for `k2/T.fst`, whose input
symbols are exactly the neural network's outputs.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from BenNevis.topos.build import (
    build_openfst_int_text,
    build_openfst_text,
    build_phone_symbols,
    build_tokens,
    expand,
    read_phone_table,
    read_symbol_table,
    symbol_ids,
    write_symbol_table,
)
from BenNevis.topos.spec import TopologySpec, available, load, load_spec

__all__ = [
    "TopologySpec",
    "available",
    "load",
    "load_spec",
    "build_openfst_int_text",
    "build_openfst_text",
    "build_phone_symbols",
    "build_tokens",
    "expand",
    "read_phone_table",
    "read_symbol_table",
    "symbol_ids",
    "write_symbol_table",
]
