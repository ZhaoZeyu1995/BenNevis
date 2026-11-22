"""
Decoding, in Python, with k2 rather than Kaldi's WFST decoder.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from BenNevis.decode.graph import build_decoding_graph, compose_lg, load_openfst_text

__all__ = ["build_decoding_graph", "compose_lg", "load_openfst_text"]
