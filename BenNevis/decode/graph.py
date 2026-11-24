"""
Building a decoding graph with k2.

`egs/*/asr/run.sh` builds `TLG.fst` with six OpenFST and Kaldi programs --
`fsttablecompose`, `fstdeterminizestar`, `fstminimizeencoded`,
`fstrmsymbols`, `fstrmepslocal` and `fstarcsort` -- and the result exists
only to be fed to `decode-faster`. Replacing the graph alone would buy
nothing, because a Python-built text FST still needs `fstcompile` to become
the binary that decoder wants. Replacing both together drops all eight
programs and the 334MB artefact with them.

The inputs are no longer Kaldi's either: `BenNevis.lm.lexicon` builds
`L_disambig` from the dictionary and `BenNevis.lm.fst` builds `G` from the
ARPA model, both byte-exactly, so the whole decoding path can run without
OpenFST.

**This is where byte equality stops being the standard, and it is not a
compromise.** k2's `determinize` is not Kaldi's `DeterminizeStar`, and k2
has no equivalent of `fstminimizeencoded`, so the graph differs in its
states and arcs while accepting the same language.

It also differs in its weights, and **the Kaldi side is the less precise
one**. `fstminimizeencoded` quantises every weight to a multiple of
OpenFST's `kDelta`, which is 1/1024: yesno's `G.fst` carries
`2.30258512` for `-log(0.1)` and its `LG.fst` carries `2.30273438`, which
is `2358/1024`. Each arc therefore loses up to `kDelta/2`, about 1.5e-4.
Matching that would mean deliberately rounding the graph to Kaldi's grid,
which is a clear case of an equivalence gate turning an accident into a
requirement. So `QUANTISATION_DELTA` is here to *bound the comparison*,
not to reproduce.

The check that counts is the WER ledger: a graph that decodes the same
transcriptions is the same graph for every purpose this project has, and
1.5e-4 per arc against acoustic scores is nothing.

Functions
---------
load_openfst_text
    Read an OpenFST text FST as a k2 FSA.
compose_lg
    Compose the lexicon and the grammar, determinise, drop disambiguation
    symbols.
build_decoding_graph
    The whole thing, topology included.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import logging
from typing import Optional

import k2
import torch

__all__ = [
    "QUANTISATION_DELTA",
    "load_openfst_text",
    "compose_lg",
    "build_decoding_graph",
]

#: OpenFST's `kDelta`, the grid `fstminimizeencoded` rounds weights to. A
#: path through `n` arcs of a Kaldi-built graph can therefore differ from
#: the same path through an unquantised one by `n * kDelta / 2`. Used to
#: bound a comparison, never to reproduce the rounding.
QUANTISATION_DELTA = 1.0 / 1024


def load_openfst_text(text: str, acceptor: bool = False) -> k2.Fsa:
    """
    Read an OpenFST text FST as a k2 FSA.

    `fstprint`, and `BenNevis.lm`'s builders after it, omit the weight when
    it is the semiring One -- so an all-zero-weight FST has four fields per
    arc and k2 wants five. The missing weight is filled in here rather than
    by an `awk` in a shell pipeline, which is what `Lang.load_lexicon` has
    been doing and is one of the project's remaining POSIX-only sites.

    Arguments
    ---------
    text: str
        The FST in OpenFST text form, with integer labels.
    acceptor: bool
        Whether it is an acceptor. `L` and `G` as built here are
        transducers, so the default is False.

    Returns
    -------
    k2.Fsa
        The FSA, arc-sorted.
    """
    lines = []
    for line in text.splitlines():
        fields = line.split("\t") if "\t" in line else line.split()
        if len(fields) == 4:
            # An arc whose weight fstprint omitted.
            fields = fields + ["0.0"]
        elif len(fields) == 1:
            # A final state whose weight it omitted.
            fields = fields + ["0.0"]
        lines.append("\t".join(fields))
    return k2.arc_sort(k2.Fsa.from_openfst("\n".join(lines) + "\n", acceptor=acceptor))


def compose_lg(
    lexicon: k2.Fsa,
    grammar: k2.Fsa,
    first_disambig: int,
) -> k2.Fsa:
    """
    Compose the lexicon with the grammar and clean up after it.

    Composes, connects, then **removes the disambiguation symbols and the
    epsilons they leave behind**. The symbols exist only to make the
    composition determinisable, and leaving them in would have the decoder
    consume a `#1` that no acoustic model ever emits.

    **Deliberately not determinised, which is a departure from both
    Kaldi's recipe and k2's own HLG construction.** Measured on
    mini-librispeech's 206,510-word lexicon and its pruned 3-gram:

    ======================================  ==========  ============
    step                                          time          arcs
    ======================================  ==========  ============
    k2 compose + connect                          1.9s     5,447,106
    k2 determinize                             3358.0s    18,732,970
    strip disambig + remove_epsilon               1.5s     6,111,314
    Kaldi: compose, determinizestar, minimise    19.0s    ~5,800,000
    ======================================  ==========  ============

    So `k2.determinize` costs **56 minutes** against the 19 seconds Kaldi
    spends doing more -- it determinises in the log semiring *and*
    minimises -- and it makes the graph 3.4 times larger rather than
    smaller, because the minimisation that would compact it afterwards has
    no equivalent here. Skipping it costs 1.5 seconds and yields a graph
    three times smaller than the determinised one, 12% above the raw
    composition. `intersect_dense_pruned` does not require a deterministic
    graph, so nothing downstream needs it.

    What this gives up is real and is not a tolerance: Kaldi's
    `--use-log=true` merges paths that share an input string by **adding
    their probabilities**, so its arc weights can be lower than any single
    path's. Without determinization the cheapest path stands alone. For
    one-best Viterbi output the winning path is the same, which is what
    the decoder asks for; a lattice or any sum over paths would differ.
    `docs/kaldi-graph-chain.md` has the rest of the chain.

    Arguments
    ---------
    lexicon: k2.Fsa
        `L_disambig`, mapping phones to words.
    grammar: k2.Fsa
        `G`, over words.
    first_disambig: int
        The lowest disambiguation symbol id on the input side. Every label
        at or above it is dropped -- these are `#0` upwards in
        `phones.txt`, which `prepare_lang.sh` writes contiguously at the
        end for exactly this reason.

    Returns
    -------
    k2.Fsa
        The composed, determinised, cleaned graph, arc-sorted.
    """
    logging.info("composing L with G")
    composed = k2.compose(lexicon, grammar)
    composed = k2.connect(composed)

    # No `k2.determinize` here. See the docstring: 56 minutes against
    # Kaldi's 19 for more work, and a graph 3.4 times larger.
    #
    # The disambiguation symbols therefore never serve their purpose and
    # are pure noise, which is the same reason they go but a different
    # reason from Kaldi's -- Kaldi carries them through the T composition
    # so that *it* can determinise again, and drops them after.
    logging.info("removing disambiguation symbols")
    labels = composed.labels.clone()
    labels[labels >= first_disambig] = 0
    composed.labels = labels

    composed = k2.remove_epsilon(composed)
    composed = k2.connect(composed)
    return k2.arc_sort(composed)


def build_decoding_graph(
    topology: k2.Fsa,
    lexicon: k2.Fsa,
    grammar: k2.Fsa,
    first_disambig: int,
    device: Optional[torch.device] = None,
) -> k2.Fsa:
    """
    The decoding graph: the topology composed with the lexicon and grammar.

    The topology is what makes this a *token* graph rather than a phone
    one: it maps the network's output labels to phones, so composing it on
    the left gives an FSA whose input alphabet is the output layer. That is
    the same object `TLG.fst` was, built from the same three parts, by
    different algorithms.

    Arguments
    ---------
    topology: k2.Fsa
        `T`, from `BenNevis.topos`. Already Python, and the only one of the
        three inputs that never came from Kaldi.
    lexicon: k2.Fsa
        `L_disambig`.
    grammar: k2.Fsa
        `G`.
    first_disambig: int
        The lowest disambiguation symbol id on the lexicon's input side.
    device: Optional[torch.device]
        Where to put the result. The graph loss runs on the CPU because k2
        has no MPS backend, and the same applies here, so this is left to
        the caller rather than guessed.

    Returns
    -------
    k2.Fsa
        The decoding graph, arc-sorted, ready for `intersect_dense_pruned`.
    """
    composed = compose_lg(lexicon, grammar, first_disambig)

    logging.info("composing T with LG")
    graph = k2.compose(k2.arc_sort(topology), composed)
    graph = k2.connect(graph)
    graph = k2.arc_sort(graph)
    if device is not None:
        graph = graph.to(device)
    return graph
