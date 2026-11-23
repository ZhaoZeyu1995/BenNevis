"""
One-best decoding with k2, in place of Kaldi's `decode-faster`.

`decode-faster` takes a compiled `TLG.fst`, an archive of log probabilities
and three search parameters, and writes word ids. This does the same with
`k2.intersect_dense_pruned` and `k2.shortest_path`, which removes the last
Kaldi binary from the decoding path -- the graph it searches is built by
`BenNevis.decode.graph` from a dictionary and an ARPA model, and the error
rate is computed by `BenNevis.utils.scoring`, so nothing between the
acoustic model and the WER is Kaldi's any more.

**The search is not Kaldi's and the result is not expected to be
identical.** `decode-faster` prunes by a beam on the arc-cost difference
and a cap on active tokens; `intersect_dense_pruned` prunes by two beams
and both a floor and a cap on active states. The parameters do not
correspond one to one, and a pruned search returns the best path *it
found*, so two implementations can disagree on a hard utterance while both
being correct. The check is the WER ledger, which is what gates this.

Why the acoustic scale multiplies rather than divides: Kaldi's
`--acoustic-scale` weights the acoustic score against the graph, and the
convention here follows it, so a value of 1.0 means equal weight and the
recipe's `aw_1.0` directories mean the same thing as before.

Functions
---------
decode
    One batch of log probabilities against a graph.
decode_archive
    An archive of them, utterance by utterance.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import k2
import torch

__all__ = [
    "DEFAULT_SEARCH_BEAM",
    "DEFAULT_OUTPUT_BEAM",
    "DEFAULT_MIN_ACTIVE",
    "DEFAULT_MAX_ACTIVE",
    "decode",
    "decode_archive",
    "words_from_best_path",
]

#: `decode-faster`'s defaults were beam 16, max-active 7000. These are k2's
#: own conventions for the two beams it prunes with, which do not
#: correspond one to one -- see the module docstring.
DEFAULT_SEARCH_BEAM = 20.0
DEFAULT_OUTPUT_BEAM = 8.0
DEFAULT_MIN_ACTIVE = 30
DEFAULT_MAX_ACTIVE = 10000


def words_from_best_path(best: k2.Fsa) -> List[List[int]]:
    """
    The word ids along each best path, epsilons dropped.

    `aux_labels` is a plain tensor for a graph whose arcs carry one output
    label each and a ragged tensor once composition has put several on an
    arc, and which one it is depends on how the graph was built rather than
    on anything the caller chose -- so both are handled here instead of at
    every call site.

    Arguments
    ---------
    best: k2.Fsa
        The result of `k2.shortest_path`, an FsaVec of one path each.

    Returns
    -------
    List[List[int]]
        Word ids per utterance, in order.
    """
    if isinstance(best.aux_labels, k2.RaggedTensor):
        paths = _ragged_paths(best, best.aux_labels)
    else:
        paths = _dense_paths(best)
    # 0 is <eps> and -1 is k2's end-of-path marker; neither is a word.
    return [[label for label in path if label > 0] for path in paths]


def aux_labels_per_arc(best: k2.Fsa) -> List[int]:
    """
    One output label per arc of a single best path.

    This is the other reduction of `aux_labels`, and the distinction
    matters. `words_from_best_path` flattens the ragged case, which is
    right for *which words were said* and wrong for an alignment:
    `bin/align.py` reads `labels[i]` and `aux_labels[i]` together to work
    out which frames a word covers, so it needs exactly one entry per arc
    and a length that matches `labels`.

    `-1` is k2's end-of-path marker and is preserved, because the dense
    path has always carried it through and the caller's walk is written
    against that. `0` is `<eps>`.

    Arguments
    ---------
    best: k2.Fsa
        One path, as `k2.shortest_path` returns it. Not an FsaVec of
        several: the caller aligns one utterance at a time, and
        `_arc_ranges` is what splits a batch.

    Returns
    -------
    List[int]
        One label per arc, in arc order.

    Raises
    ------
    ValueError
        If an arc carries more than one word. An alignment places a word
        across a span of frames, and two words on one arc cannot be laid
        out in time -- so this is reported rather than resolved by
        dropping one, which would silently shift every word after it.
    """
    aux = best.aux_labels
    if not isinstance(aux, k2.RaggedTensor):
        return aux.cpu().tolist()
    labels = []
    for arc, row in enumerate(aux.tolist()):
        words = [label for label in row if label > 0]
        if len(words) > 1:
            raise ValueError(
                f"arc {arc} carries {len(words)} output labels, {words}; "
                "an alignment cannot place more than one word on an arc"
            )
        if words:
            labels.append(words[0])
        elif -1 in row:
            labels.append(-1)
        else:
            labels.append(0)
    return labels


def _arc_ranges(best: k2.Fsa) -> Iterable[Tuple[int, int]]:
    """
    The half-open arc range of each FSA in an FsaVec.

    An FsaVec's arcs are a three-level ragged shape -- FSA, then state,
    then arc -- so `row_splits(1)` gives *state* boundaries per FSA and not
    arc boundaries. Using it directly ran the first utterance's slice into
    the second's: a batch of two decoded as `[[1, 2], [2]]` where the
    answer was `[[1], [2]]`. A batch of one would never have shown it.
    """
    shape = best.arcs.shape()
    states = shape.row_splits(1).cpu().tolist()
    arcs = shape.row_splits(2).cpu().tolist()
    for first_state, last_state in zip(states[:-1], states[1:]):
        yield arcs[first_state], arcs[last_state]


def _dense_paths(best: k2.Fsa) -> Iterable[Sequence[int]]:
    """Per-path aux labels where `aux_labels` is a plain tensor."""
    labels = best.aux_labels.cpu().tolist()
    for start, end in _arc_ranges(best):
        yield labels[start:end]


def _ragged_paths(best: k2.Fsa, aux: k2.RaggedTensor) -> Iterable[Sequence[int]]:
    """Per-path aux labels where `aux_labels` is ragged."""
    per_arc = aux.tolist()
    for start, end in _arc_ranges(best):
        flat: List[int] = []
        for arc in range(start, end):
            flat.extend(per_arc[arc])
        yield flat


def decode(
    graph: k2.Fsa,
    log_probs: torch.Tensor,
    lengths: torch.Tensor,
    acoustic_scale: float = 1.0,
    search_beam: float = DEFAULT_SEARCH_BEAM,
    output_beam: float = DEFAULT_OUTPUT_BEAM,
    min_active_states: int = DEFAULT_MIN_ACTIVE,
    max_active_states: int = DEFAULT_MAX_ACTIVE,
) -> List[List[int]]:
    """
    Decode one batch of log probabilities against a graph.

    Arguments
    ---------
    graph: k2.Fsa
        The decoding graph, arc-sorted, on the same device as `log_probs`.
    log_probs: torch.Tensor
        `(batch, frames, tokens)`, log-softmaxed.
    lengths: torch.Tensor
        `(batch,)` valid frame counts.
    acoustic_scale: float
        How much to weight the acoustic score against the graph. Follows
        Kaldi's `--acoustic-scale`, so the recipe's `aw_1.0` keeps meaning
        what it did.
    search_beam: float
        k2's search beam.
    output_beam: float
        k2's output beam.
    min_active_states: int
        The floor on active states, which Kaldi has no equivalent of.
    max_active_states: int
        The cap, which corresponds loosely to `--max-active`.

    Returns
    -------
    List[List[int]]
        Word ids per utterance.
    """
    if log_probs.dim() != 3:
        raise ValueError(
            f"log_probs must be (batch, frames, tokens), got "
            f"{tuple(log_probs.shape)}"
        )
    if lengths.numel() != log_probs.shape[0]:
        raise ValueError(
            f"lengths has {lengths.numel()} entries for "
            f"{log_probs.shape[0]} utterances"
        )

    scaled = log_probs * acoustic_scale
    # k2 wants (sequence, start frame, frame count) on the CPU as int32,
    # whatever device the scores live on.
    segments = torch.stack(
        [
            torch.arange(scaled.shape[0], dtype=torch.int32),
            torch.zeros(scaled.shape[0], dtype=torch.int32),
            lengths.detach().cpu().to(torch.int32),
        ],
        dim=1,
    )
    dense = k2.DenseFsaVec(scaled, segments)
    lattice = k2.intersect_dense_pruned(
        graph,
        dense,
        search_beam=search_beam,
        output_beam=output_beam,
        min_active_states=min_active_states,
        max_active_states=max_active_states,
    )
    # Double scores, for the reason the graph loss keeps them: with float
    # the accumulation has produced NaNs.
    best = k2.shortest_path(lattice, use_double_scores=True)
    return words_from_best_path(best)


def decode_archive(
    graph: k2.Fsa,
    utterances: Iterable[Tuple[str, "torch.Tensor"]],
    idx2word: Dict[int, str],
    acoustic_scale: float = 1.0,
    device: Optional[torch.device] = None,
    **search,
) -> List[Tuple[str, List[str]]]:
    """
    Decode an archive of log probabilities, one utterance at a time.

    One at a time rather than batched, deliberately: `decode-faster` is
    sequential, so this keeps the comparison honest, and an utterance that
    exhausts memory then names itself instead of taking a batch down with
    it. Batching is a performance question for later, and a measured one.

    Arguments
    ---------
    graph: k2.Fsa
        The decoding graph.
    utterances: Iterable[Tuple[str, torch.Tensor]]
        Utterance id and its `(frames, tokens)` log probabilities.
    idx2word: Dict[int, str]
        The word symbol table, inverted -- `Lang` already has one.
    acoustic_scale: float
        See `decode`.
    device: Optional[torch.device]
        Where to run. k2 has no MPS backend, so a caller on an Apple
        machine wants the CPU here; left explicit rather than guessed.
    **search
        Passed to `decode`.

    Returns
    -------
    List[Tuple[str, List[str]]]
        Utterance id and the words decoded for it.
    """
    results: List[Tuple[str, List[str]]] = []
    for index, (uttid, scores) in enumerate(utterances):
        if scores.dim() != 2:
            raise ValueError(
                f"{uttid}: expected (frames, tokens), got " f"{tuple(scores.shape)}"
            )
        batch = scores.unsqueeze(0)
        if device is not None:
            batch = batch.to(device)
        lengths = torch.tensor([scores.shape[0]], dtype=torch.int32)
        ids = decode(
            graph,
            batch,
            lengths,
            acoustic_scale=acoustic_scale,
            **search,
        )[0]
        results.append((uttid, [idx2word[i] for i in ids if i in idx2word]))
        if index and index % 100 == 0:
            logging.info("decoded %d utterances", index)
    return results
