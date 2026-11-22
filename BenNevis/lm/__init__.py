"""
Language model handling, in Python.

Currently the ARPA side of what Kaldi's `arpa2fst` does: reading a
back-off n-gram model and turning it into the weighted FST the decoding
graph composes against.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from BenNevis.lm.arpa import LOG10_TO_NAT, ArpaModel, NGram, read_arpa
from BenNevis.lm.fst import build_arcs, build_fst_text, start_state, total_weight
from BenNevis.lm.lexicon import build_lexicon_arcs, build_lexicon_fst_text, read_lexicon

__all__ = [
    "LOG10_TO_NAT",
    "ArpaModel",
    "NGram",
    "read_arpa",
    "build_arcs",
    "build_fst_text",
    "start_state",
    "total_weight",
    "read_lexicon",
    "build_lexicon_arcs",
    "build_lexicon_fst_text",
]
