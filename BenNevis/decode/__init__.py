"""
Decoding, in Python, with k2 rather than Kaldi's WFST decoder.

**Nothing is imported eagerly here, and that is the point.** `graph` and
`decoder` need k2; `pipeline`'s `.trn` handling needs nothing but the
standard library. A package `__init__` that imports the heavy submodules
makes *every* submodule as expensive as the heaviest one -- importing
`BenNevis.decode.pipeline` to format a transcription pulled in k2 and broke
tier 1 and tier 1b on every platform, which a machine with k2 installed
cannot see.

So the convenience names resolve on first use through `__getattr__`. A
caller who wants `format_trn` gets it without k2; a caller who wants
`decode` gets the import error they would have got anyway.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from typing import Any

__all__ = [
    "build_decoding_graph",
    "compose_lg",
    "load_openfst_text",
    "decode",
    "decode_archive",
    "build_graph",
    "format_trn",
    "write_trn",
]

#: Which submodule each exported name lives in, so `__getattr__` can find
#: it without importing the others.
_SOURCES = {
    "build_decoding_graph": "graph",
    "compose_lg": "graph",
    "load_openfst_text": "graph",
    "decode": "decoder",
    "decode_archive": "decoder",
    "build_graph": "pipeline",
    "format_trn": "pipeline",
    "write_trn": "pipeline",
}


def __getattr__(name: str) -> Any:
    """
    Resolve an exported name by importing only the submodule holding it.

    Arguments
    ---------
    name: str
        The attribute being looked up.

    Returns
    -------
    Any
        The attribute.

    Raises
    ------
    AttributeError
        For a name this package does not export.
    """
    module = _SOURCES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f"{__name__}.{module}"), name)


def __dir__():
    return sorted(__all__)
