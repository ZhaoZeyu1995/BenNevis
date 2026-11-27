"""
Frame-synchronous Viterbi decoding, in place of `k2.intersect_dense_pruned`.

This is Kaldi's `FasterDecoder` written in torch tensor operations: one
token per graph state, a beam and a cap on active states, a back-pointer
per token and no lattice. It exists because the k2 route could not be made
to decode mini-librispeech inside a CI run, and because tuning it was
measured and found not to help.

Why, in numbers, on the real 43,263,880-arc `TLG` and 1,100-frame
utterances:

======================================================  ===========  =====
search                                                   ms a frame    WER
======================================================  ===========  =====
`intersect_dense_pruned`, search 20 / output 8             8.60       41.0%
the same, output beam 2 instead of 8                       8.66       41.0%
the same, max_active 5000 instead of 10000                 7.86       39.5%
this, beam 20 / max_active 7000                            **0.98**   41.0%
this, beam 32 / max_active 5000, which is the recipe's      3.57       38.0%
======================================================  ===========  =====

So the two pruning knobs k2 offers are worth -0.7% and 9%, and there is no
setting of them that closes a 19x gap. **The gap was never k2's CPU
implementation; it was the graph, and what k2 cannot do with it.** The
profile says it plainly: the states the search actually visits have a mean
out-degree of **168**, against the graph's own average of 9.31 and Kaldi's
3.66. One state carries 208,941 arcs over 106 distinct input labels, and
366 such states hold 52.5% of the whole graph. They are the word-boundary
states, and the cause is that `L_disambig` is a flower -- Kaldi's
`make_lexicon_fst.py` gives every pronunciation its own chain, so the loop
state has one arc per entry -- which Kaldi's `fstdeterminizestar` turns
into a trie. We do not determinise, because `k2.determinize` costs 56
minutes on this graph and returns 3.4 times the arcs.

**Label factoring** recovers most of what determinisation would, locally,
in one vectorised pass, and exactly. Where a state has *k* > 1 arcs sharing
an input label, they are replaced by one emitting arc into a new state
carrying the group's minimum weight, and *k* non-emitting arcs out of it
carrying the residuals. Every path keeps its exact weight, since
`min + (w - min) == w`. What it buys is that those *k* arcs are reached
only once the label has survived the beam **with its acoustic score
applied**: at a word boundary a handful of the 106 possible first phones
are acoustically plausible in a given frame, so the search expands a few
hundred arcs where it used to expand 206,721. Measured: 12,735 emitting
plus 10,727 non-emitting candidate arcs a frame, against 338,909 before,
and Kaldi's own ~18,300.

**This is not an approximation and there is nothing to trade.** A token at
the new state costs the group's minimum, so it survives exactly when some
member of the group would have, and the non-emitting pass then gives each
member its true cost and prunes it against the same cutoff. The surviving
token set is identical to the unfactored search's. Four factoring
thresholds on the real graph give the same WER and the same error counts
to the error, and 293 random graphs agree with `k2.intersect_dense`
followed by `k2.shortest_path` -- an exhaustive intersection and an exact
Viterbi -- on both the best path's cost and its word sequence, at three
thresholds each.

The pruning parameters are Kaldi's and mean what they mean there, which is
what makes the two searches comparable: `beam` on the cost difference from
the best token, `max_active` and `min_active` on the number of them. One
difference is worth knowing: `FasterDecoder` applies its cutoff to the
tokens it is about to expand, while this applies it to the candidates as
they are created. Those are the same set one frame apart -- a candidate
kept here is a token expanded there -- so the surviving sets agree; doing
it at creation simply means never building a token that will not be kept,
which is what `next_weight_cutoff` is for in the original.

Memory is the other half of why this exists. `intersect_dense_pruned`
returns a lattice, and on a 1,100-frame utterance with `max_active` 10,000
that lattice is gigabytes: run 45918000128 was SIGKILLed with 1,000 of
1,089 utterances decoded, having touched 14,988 MiB. Token passing keeps
one token per state and a back-pointer, so what it holds is the graph plus
some tens of megabytes.

Functions and classes
---------------------
SearchGraph
    A decoding graph prepared for this search: two CSRs, emitting and not.
ViterbiDecoder
    The search itself.
decode_archive
    An archive of log probabilities, one utterance at a time.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import logging
import multiprocessing
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "DEFAULT_BEAM",
    "DEFAULT_MAX_ACTIVE",
    "DEFAULT_MIN_ACTIVE",
    "DEFAULT_FACTOR_THRESHOLD",
    "SearchGraph",
    "ViterbiDecoder",
    "decode_archive",
    "decode_shards",
]

#: `decode-faster`'s own defaults, so a run of each is comparable without
#: translating anything. `run/decode_faster.sh` overrides the beam to 32
#: and `max_active` to 5000, and `run/decode_k2.py` follows it.
DEFAULT_BEAM = 16.0
DEFAULT_MAX_ACTIVE = 7000
DEFAULT_MIN_ACTIVE = 20

#: Factor a group of arcs sharing an input label only once there are this
#: many of them. Measured on mini-librispeech's `TLG`: at 8, 100,141 groups
#: cover 55.2% of all 43M arcs and the arc count grows by 0.2%, where at 2
#: it takes 2,848,972 new states to cover 74.9% and costs 6.6%. All four
#: thresholds tried gave identical WER and within 20% of the same speed,
#: so this is chosen for its memory.
DEFAULT_FACTOR_THRESHOLD = 8

#: `float("inf")` as a tensor fill, kept as a name because it means "no
#: token here" in two different arrays.
_INF = float("inf")

#: Sentinel for "no arc chosen yet" in the tie-breaking scatter. Larger
#: than any arc index a graph could have and still fit in int64.
_NO_ARC = 2**62


@dataclass
class SearchGraph:
    """
    A decoding graph in the form this search wants.

    Two compressed-sparse-row arc lists over one state numbering: the
    emitting arcs, which consume a frame, and the non-emitting ones, which
    do not. A graph with no parallel arcs has an empty second list, and a
    factored one has the groups' members there.

    `arc` maps each arc back to the source FSA's arc index, so the word
    labels stay where k2 put them and are read only for the few arcs on a
    best path. `-1` marks an arc this preparation invented, which carries
    no word.

    Attributes
    ----------
    num_states: int
        Including the states factoring introduced, which come last.
    num_tokens: int
        One more than the largest input label, i.e. what the acoustic
        model must emit.
    """

    num_states: int
    num_tokens: int
    emit_row_splits: torch.Tensor
    emit_dst: torch.Tensor
    emit_label: torch.Tensor
    emit_cost: torch.Tensor
    emit_arc: torch.Tensor
    eps_row_splits: torch.Tensor
    eps_dst: torch.Tensor
    eps_cost: torch.Tensor
    eps_arc: torch.Tensor
    eps_key: torch.Tensor
    final_cost: torch.Tensor
    aux_row_splits: Optional[torch.Tensor]
    aux_values: torch.Tensor

    @property
    def num_emitting_arcs(self) -> int:
        return int(self.emit_dst.numel())

    @property
    def num_non_emitting_arcs(self) -> int:
        return int(self.eps_dst.numel())

    def words_of(self, arc: int, non_emitting: bool) -> List[int]:
        """
        The word ids an arc of this graph emits.

        Arguments
        ---------
        arc: int
            An index into `eps_*` if `non_emitting`, else into `emit_*`.
        non_emitting: bool
            Which arc list the index is for.

        Returns
        -------
        List[int]
            Word ids, with `<eps>` and k2's `-1` marker dropped.
        """
        source = int(self.eps_arc[arc] if non_emitting else self.emit_arc[arc])
        if source < 0:
            return []
        if self.aux_row_splits is None:
            label = int(self.aux_values[source])
            return [label] if label > 0 else []
        start = int(self.aux_row_splits[source])
        end = int(self.aux_row_splits[source + 1])
        return [int(x) for x in self.aux_values[start:end] if x > 0]

    @classmethod
    def from_fsa(
        cls,
        fsa,
        factor_threshold: int = DEFAULT_FACTOR_THRESHOLD,
    ) -> "SearchGraph":
        """
        Prepare a k2 FSA for this search, factoring its parallel arcs.

        Arguments
        ---------
        fsa: k2.Fsa
            The decoding graph, arc-sorted. `aux_labels` carries the words
            and may be a plain tensor or a `k2.RaggedTensor`.
        factor_threshold: int
            The smallest group of arcs sharing an input label that is
            worth factoring. 1 or below factors every group, which costs a
            state per distinct `(state, label)` pair for no measured gain;
            a number above the largest group factors nothing, which is the
            unfactored search and is how the equivalence is tested.

        Returns
        -------
        SearchGraph
            Ready for `ViterbiDecoder`.

        Raises
        ------
        ValueError
            If the arcs are not sorted by `(state, label)`, since the
            grouping is read off their order rather than recomputed; or if
            an input label other than k2's `-1` final marker is negative.
        """
        import k2  # local, so importing this module needs no k2

        arcs = fsa.arcs.values()
        src = arcs[:, 0].long()
        dst = arcs[:, 1].long()
        label = arcs[:, 2].long()
        cost = -fsa.scores.detach().double()
        source_arc = torch.arange(src.numel(), dtype=torch.int64)
        num_states = int(fsa.shape[0])

        # k2 writes a final state as an arc labelled -1 into a single
        # super-final state, where OpenFST writes a final weight. Token
        # passing wants the second form: left in the arc list, every frame
        # would expand -1 as though it were a token a network could emit.
        is_final = label < 0
        final_cost = torch.full((num_states,), _INF, dtype=torch.float64)
        final_cost[src[is_final]] = cost[is_final]
        keep = ~is_final
        src, dst, label, cost, source_arc = (
            src[keep],
            dst[keep],
            label[keep],
            cost[keep],
            source_arc[keep],
        )
        if label.numel() and int(label.min()) < 0:
            raise ValueError(
                "an input label below zero survived dropping the final arcs; "
                "this search treats -1 as finality and every other label as "
                "a token the acoustic model emits"
            )
        num_tokens = int(label.max()) + 1 if label.numel() else 0

        groups, group_min = _group_by_state_and_label(src, label, cost)
        graph = _factor(
            src,
            dst,
            label,
            cost,
            source_arc,
            groups,
            group_min,
            num_states,
            factor_threshold,
        )

        aux = fsa.aux_labels
        if isinstance(aux, k2.RaggedTensor):
            aux_row_splits = aux.shape.row_splits(1).clone()
            aux_values = aux.values.clone()
        else:
            aux_row_splits = None
            aux_values = aux.detach().clone()

        total_states = graph["num_states"]
        final_cost = torch.cat(
            [
                final_cost,
                torch.full((total_states - num_states,), _INF, dtype=torch.float64),
            ]
        )
        return cls(
            num_states=total_states,
            num_tokens=num_tokens,
            final_cost=final_cost.float(),
            aux_row_splits=aux_row_splits,
            aux_values=aux_values,
            **{k: v for k, v in graph.items() if k != "num_states"},
        )


def _group_by_state_and_label(
    src: torch.Tensor, label: torch.Tensor, cost: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Which group of arcs sharing `(state, label)` each arc belongs to.

    The groups are read off the arc order rather than sorted for, because
    `k2.arc_sort` already leaves them contiguous -- and that is asserted,
    since a graph that was not arc-sorted would group silently wrongly and
    give a decoder that is merely a bit off.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The group index of every arc, and each group's minimum cost.
    """
    if src.numel() == 0:
        return (
            torch.zeros(0, dtype=torch.int64),
            torch.zeros(0, dtype=torch.float64),
        )
    width = int(label.max()) + 2
    key = src * width + label
    if bool((key[1:] < key[:-1]).any()):
        raise ValueError(
            "the arcs are not sorted by (state, label); call k2.arc_sort on "
            "the graph before preparing it for this search"
        )
    boundary = torch.ones(key.numel(), dtype=torch.bool)
    boundary[1:] = key[1:] != key[:-1]
    groups = torch.cumsum(boundary.long(), 0) - 1
    count = int(groups[-1]) + 1
    group_min = torch.full((count,), _INF, dtype=torch.float64)
    group_min.scatter_reduce_(0, groups, cost, reduce="amin")
    return groups, group_min


def _factor(
    src: torch.Tensor,
    dst: torch.Tensor,
    label: torch.Tensor,
    cost: torch.Tensor,
    source_arc: torch.Tensor,
    groups: torch.Tensor,
    group_min: torch.Tensor,
    num_states: int,
    threshold: int,
) -> Dict[str, object]:
    """
    Split the arcs into an emitting list and a non-emitting one.

    Groups smaller than `threshold` stay as ordinary emitting arcs. Each
    larger one becomes a single emitting arc into a state of its own,
    carrying the group's minimum cost, plus one non-emitting arc per
    member carrying what that member costs above the minimum.
    """
    sizes = torch.bincount(groups, minlength=group_min.numel())
    factored = sizes >= max(threshold, 2)
    rank = torch.cumsum(factored.long(), 0) - 1
    num_new = int(factored.sum())
    total_states = num_states + num_new

    member_factored = factored[groups]
    plain = ~member_factored
    first = torch.zeros(groups.numel(), dtype=torch.bool)
    first[0] = True
    first[1:] = groups[1:] != groups[:-1]
    heads = first & member_factored

    emit_src = torch.cat([src[plain], src[heads]])
    emit_label = torch.cat([label[plain], label[heads]])
    emit_dst = torch.cat([dst[plain], num_states + rank[groups[heads]]])
    emit_cost = torch.cat([cost[plain], group_min[groups[heads]]])
    emit_arc = torch.cat(
        [source_arc[plain], torch.full((num_new,), -1, dtype=torch.int64)]
    )
    # Stable, so arcs of one state keep the label order they came in and
    # the grouping of the *next* preparation of this graph is the same.
    order = torch.argsort(emit_src, stable=True)

    # The members of a group, **sorted by their residual**, which is what
    # lets the search expand only a prefix of them -- see `_prefix_limits`.
    group_of_member = groups[member_factored]
    eps_src = num_states + rank[group_of_member]
    eps_cost = cost[member_factored] - group_min[group_of_member]
    key = _residual_key(eps_src, eps_cost)
    inside = torch.argsort(key)

    return dict(
        num_states=total_states,
        emit_row_splits=_row_splits(emit_src[order], total_states),
        emit_dst=emit_dst[order].int(),
        emit_label=emit_label[order].int(),
        emit_cost=emit_cost[order].float(),
        emit_arc=emit_arc[order].int(),
        eps_row_splits=_row_splits(eps_src[inside], total_states),
        eps_dst=dst[member_factored][inside].int(),
        eps_cost=eps_cost[inside].float(),
        eps_arc=source_arc[member_factored][inside].int(),
        eps_key=key[inside],
    )


#: Steps per nat in the key the non-emitting prefix is found by. Coarse on
#: purpose: a coarser key only ever selects a *superset* of the arcs within
#: the beam, and the exact float comparison follows, so this trades nothing
#: but a few wasted comparisons.
_KEY_SCALE = 64

#: Residuals above this many nats are clamped together. A residual is how
#: much dearer a member of a group is than its cheapest, so one this large
#: is outside any beam anyone decodes with; clamping keeps the key small
#: and, being monotone, keeps it sorted.
_KEY_CEILING = 1024


def _residual_key(src: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """
    A single sorted key over all non-emitting arcs, state major.

    `torch.searchsorted` wants one sorted sequence, and the residuals are
    sorted only *within* a state -- so the state index is folded into the
    key above the residual. That makes the whole array non-decreasing and
    one `searchsorted` can answer every active token's question at once.
    """
    quantised = torch.clamp(
        (residual * _KEY_SCALE).floor().long(), 0, _KEY_CEILING * _KEY_SCALE
    )
    return src * (_KEY_CEILING * _KEY_SCALE + 1) + quantised


def _prefix_limits(
    graph: "SearchGraph", active: torch.Tensor, budget: torch.Tensor
) -> torch.Tensor:
    """
    How many of each state's non-emitting arcs can be within the cutoff.

    A token at cost *c* under a cutoff *k* can only use an arc whose
    residual is at most *k - c*, and the residuals are sorted -- so the
    answer is a prefix, and the prefix length is one `searchsorted`
    against the key. The quantised budget is rounded **up** so the prefix
    is a superset of the arcs that will survive the exact comparison; it
    is never short, which would silently lose paths.

    This is the whole of the second optimisation, and the measurement that
    asked for it: at beam 32 the closure was evaluating 162,533 arcs a
    frame and keeping 24% of them.
    """
    quantised = torch.clamp(
        (budget * _KEY_SCALE).ceil().long(), 0, _KEY_CEILING * _KEY_SCALE
    )
    width = _KEY_CEILING * _KEY_SCALE + 1
    bound = torch.searchsorted(graph.eps_key, active * width + quantised, right=True)
    starts = graph.eps_row_splits[active]
    return torch.clamp(bound - starts, min=0)


def _arcs_within(
    row_splits: torch.Tensor, active: torch.Tensor, counts: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    `_arcs_of`, but taking only the first `counts[i]` arcs of each state.

    Used for the non-emitting pass, where the arcs of a state are sorted
    by residual so a prefix is all that can survive the cutoff.
    """
    total = int(counts.sum())
    if total == 0:
        empty = torch.zeros(0, dtype=torch.int64)
        return empty, empty
    token = torch.repeat_interleave(
        torch.arange(active.numel(), dtype=torch.int64), counts
    )
    shift = row_splits[active] - (torch.cumsum(counts, 0) - counts)
    arc = shift[token] + torch.arange(total, dtype=torch.int64)
    return arc, token


def _row_splits(src: torch.Tensor, num_states: int) -> torch.Tensor:
    """CSR boundaries for arcs already ordered by source state."""
    out = torch.zeros(num_states + 1, dtype=torch.int64)
    torch.cumsum(torch.bincount(src, minlength=num_states), 0, out=out[1:])
    return out


def _arcs_of(
    row_splits: torch.Tensor, active: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Every arc out of `active`, and which of them each arc belongs to.

    The ragged expansion is three `repeat_interleave`s and an `arange`,
    which is the whole reason this search is affordable in Python: one
    frame is a handful of tensor operations over some ten thousand
    elements rather than a loop over them.
    """
    starts = row_splits[active]
    counts = row_splits[active + 1] - starts
    total = int(counts.sum())
    if total == 0:
        empty = torch.zeros(0, dtype=torch.int64)
        return empty, empty
    token = torch.repeat_interleave(
        torch.arange(active.numel(), dtype=torch.int64), counts
    )
    # `base - offset` is one value per token, so it is a gather rather than
    # two more `repeat_interleave`s -- which were 10% of the whole search.
    shift = starts - (torch.cumsum(counts, 0) - counts)
    arc = shift[token] + torch.arange(total, dtype=torch.int64)
    return arc, token


class ViterbiDecoder:
    """
    `FasterDecoder`'s search over a `SearchGraph`.

    One instance may decode any number of utterances; the scratch arrays it
    keeps are the size of the graph's state count and are reset only where
    they were touched, so a frame costs nothing in proportion to the graph.

    Arguments
    ---------
    graph: SearchGraph
        The prepared graph.
    beam: float
        Kaldi's `--beam`: how far above the best token's cost a token may
        be and still be kept.
    max_active: int
        Kaldi's `--max-active`, a cap on the number of tokens.
    min_active: int
        Kaldi's `--min-active`, a floor: the beam is widened rather than
        leave fewer than this active.
    acoustic_scale: float
        What the log probabilities are multiplied by, as Kaldi's
        `--acoustic-scale`.
    beam_delta: float
        Kaldi's `--beam-delta`, which it calls an obscure setting: how much
        slack the adaptive beam gets over whichever cutoff `max_active` or
        `min_active` forced.
    max_epsilon_rounds: int
        How many times the non-emitting closure may iterate in one frame.
        Factoring introduces exactly one level, so one round settles it;
        the loop is general because a graph from elsewhere need not be.
    """

    def __init__(
        self,
        graph: SearchGraph,
        beam: float = DEFAULT_BEAM,
        max_active: int = DEFAULT_MAX_ACTIVE,
        min_active: int = DEFAULT_MIN_ACTIVE,
        acoustic_scale: float = 1.0,
        beam_delta: float = 0.5,
        max_epsilon_rounds: int = 16,
    ) -> None:
        if beam <= 0:
            raise ValueError(f"beam must be positive, got {beam}")
        if max_active < 1:
            raise ValueError(f"max_active must be at least 1, got {max_active}")
        if min_active < 0:
            raise ValueError(f"min_active must not be negative, got {min_active}")
        if min_active > max_active:
            raise ValueError(
                f"min_active {min_active} is above max_active {max_active}, so "
                f"no token count can satisfy both"
            )
        self.graph = graph
        self.beam = beam
        self.max_active = max_active
        self.min_active = min_active
        self.acoustic_scale = acoustic_scale
        self.beam_delta = beam_delta
        self.max_epsilon_rounds = max_epsilon_rounds
        self._best = torch.full((graph.num_states,), _INF, dtype=torch.float64)
        self._arc = torch.full((graph.num_states,), _NO_ARC, dtype=torch.int64)
        #: Counted rather than timed, because a timing cannot be asserted:
        #: candidate arcs a frame is the quantity the factoring exists to
        #: reduce, and the one to watch if a graph changes shape.
        self.counters = dict(frames=0, emitting=0, non_emitting=0, rounds=0)

    def _cutoff(self, cost: torch.Tensor) -> Tuple[float, float]:
        """
        Kaldi's `GetCutoff`, in its own order.

        `max_active` first, and if it binds tighter than the beam it wins;
        then `min_active`, and if it is looser than the beam it wins. The
        order matters when both would apply.

        **Two values, because Kaldi's `GetCutoff` returns two.** One is the
        cutoff for the tokens in hand; the other is the `adaptive_beam`
        that `ProcessEmitting` then prunes the *next* frame's candidates
        with, at `best_new + adaptive_beam`. They are not the same number
        and the coupling is the point: when `max_active` tightens the
        cutoff the candidate beam tightens with it, and when `min_active`
        loosens it the candidate beam loosens too, both by `beam_delta`.
        Splitting them -- applying only the plain beam to candidates --
        was tried here and makes `min_active` inert.

        Returns
        -------
        Tuple[float, float]
            The cutoff on the costs given, and the beam to prune the next
            frame's candidates with.

        **One token apart from Kaldi, deliberately.** `GetCutoff` takes
        `tmp_array_[max_active]` after an `nth_element`, which is the
        *(max_active + 1)*-th smallest cost, so `--max-active 5000` leaves
        5,001 tokens; `kthvalue(cost, max_active)` here is the
        5,000th. The same shift applies to `min_active`. The names are
        taken to mean what they say, and a one-token difference in a cap of
        five thousand is not worth reproducing an off-by-one for -- but it
        does mean the two searches are not expected to agree token for
        token at the cap.
        """
        count = cost.numel()
        best = float(cost.min())
        cutoff = best + self.beam
        if count > self.max_active:
            kth = float(torch.kthvalue(cost, self.max_active).values)
            if kth < cutoff:
                return kth, kth - best + self.beam_delta
        if count > self.min_active > 0:
            kth = float(torch.kthvalue(cost, self.min_active).values)
            if kth > cutoff:
                return kth, kth - best + self.beam_delta
        return cutoff, self.beam

    def _cap(
        self,
        active: torch.Tensor,
        cost: torch.Tensor,
        history: List[Tuple[torch.Tensor, torch.Tensor, bool]],
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply the cutoff to the token list, which is where Kaldi applies it.

        **This was wrong and the profile is what showed it.**
        `FasterDecoder` calls `GetCutoff` on `toks_` at the *start* of a
        frame, so `max_active` bounds the token list *after* the
        non-emitting closure has added to it. This applied it to the
        emitting candidates instead, and the consequence was large: at beam
        32 the emitting pass left 3,837 tokens, the closure took that to
        **18,038**, and nothing capped them -- so `--max-active 5000` was
        letting 3.6 times its own limit into the next frame. With factoring
        that is not a detail, because the closure's job is to add a token
        per member of every group it reaches.

        The last generation in `history` is aligned with `active` by
        construction, so dropping tokens means indexing that generation by
        the same mask -- otherwise the next generation's parent indices
        would point into a list that no longer exists, which is a
        back-pointer bug and would read as a plausible-but-wrong
        transcription.
        """
        if active.numel() <= 1:
            return active, cost, self.beam
        cutoff, adaptive = self._cutoff(cost)
        keep = cost <= cutoff
        if bool(keep.all()):
            return active, cost, adaptive
        parent, arc, non_emitting = history[-1]
        history[-1] = (parent[keep], arc[keep], non_emitting)
        return active[keep], cost[keep], adaptive

    def _recombine(
        self, dst: torch.Tensor, cost: torch.Tensor, arc: torch.Tensor
    ) -> torch.Tensor:
        """
        One token per state, the cheapest.

        Two scatters rather than one: the first finds each state's best
        cost, the second picks the smallest arc index among whatever tied,
        so which token survives does not depend on the order a reduction
        happened to visit them in. The scratch arrays are reset only at
        the states this frame touched.
        """
        self._best.scatter_reduce_(0, dst, cost, reduce="amin")
        is_best = cost == self._best[dst]
        self._arc.scatter_reduce_(0, dst[is_best], arc[is_best], reduce="amin")
        chosen = is_best & (self._arc[dst] == arc)
        self._best[dst] = _INF
        self._arc[dst] = _NO_ARC
        return chosen

    def decode(self, log_probs: torch.Tensor) -> List[int]:
        """
        The word ids of the best path through one utterance.

        Arguments
        ---------
        log_probs: torch.Tensor
            `(frames, tokens)`, log-softmaxed.

        Returns
        -------
        List[int]
            Word ids in order, `<eps>` dropped.

        Raises
        ------
        ValueError
            If `log_probs` is not two dimensional, or has fewer tokens than
            the graph's input labels need -- which is the graph-and-model
            mismatch that cost the first real k2 decoding run, and is
            cheaper to name here than to diagnose from the search.
        """
        if log_probs.dim() != 2:
            raise ValueError(
                f"log_probs must be (frames, tokens), got " f"{tuple(log_probs.shape)}"
            )
        if log_probs.shape[1] < self.graph.num_tokens:
            raise ValueError(
                f"the graph has input labels up to "
                f"{self.graph.num_tokens - 1} but the acoustic model emits "
                f"{log_probs.shape[1]} tokens; the two were built from "
                f"different symbol tables"
            )
        g = self.graph
        active = torch.zeros(1, dtype=torch.int64)
        cost = torch.zeros(1, dtype=torch.float64)
        # One entry per generation: where each of its tokens came from in
        # the generation before, and on which arc. A frame is one emitting
        # generation and however many non-emitting ones it took to settle.
        history: List[Tuple[torch.Tensor, torch.Tensor, bool]] = []
        # The start state alone needs no cutoff, so the first frame's
        # candidates are pruned with the plain beam.
        adaptive = self.beam

        for frame in range(log_probs.shape[0]):
            acoustic = -(log_probs[frame].double() * self.acoustic_scale)
            arc, token = _arcs_of(g.emit_row_splits, active)
            if arc.numel() == 0:
                break
            self.counters["emitting"] += int(arc.numel())
            candidate = (
                cost[token]
                + g.emit_cost[arc].double()
                + acoustic[g.emit_label[arc].long()]
            )
            # Only the beam here. `max_active` belongs to the token list and
            # is applied below, which is where Kaldi applies it -- see
            # `_cap`.
            cutoff = float(candidate.min()) + adaptive
            alive = candidate <= cutoff
            arc, token, candidate = arc[alive], token[alive], candidate[alive]
            if arc.numel() == 0:
                break
            dst = g.emit_dst[arc].long()
            chosen = self._recombine(dst, candidate, arc)
            active, cost = dst[chosen], candidate[chosen]
            # int32, because this is the only thing the search allocates in
            # proportion to the utterance: 1,500 frames by 7,000 tokens is
            # 91MiB at four bytes a field and twice that at eight.
            history.append((token[chosen].int(), arc[chosen].int(), False))
            self.counters["frames"] += 1
            active, cost = self._close(active, cost, cutoff, history)
            active, cost, adaptive = self._cap(active, cost, history)

        if not history:
            return []
        # Kaldi prefers a token in a final state and falls back to the best
        # token anywhere when the search reached none, which an undertrained
        # model makes common rather than exotic.
        total = cost + g.final_cost[active].double()
        if not bool(torch.isfinite(total).any()):
            total = cost
        index = int(torch.argmin(total))

        words: List[int] = []
        for parent, arc, non_emitting in reversed(history):
            which = int(arc[index])
            if which >= 0:
                words.extend(reversed(g.words_of(which, non_emitting)))
            index = int(parent[index])
        words.reverse()
        return words

    def _close(
        self,
        active: torch.Tensor,
        cost: torch.Tensor,
        cutoff: float,
        history: List[Tuple[torch.Tensor, torch.Tensor, bool]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Kaldi's `ProcessNonemitting`, under the frame's own cutoff.

        The tokens already present compete with the new ones, because a
        non-emitting arc may reach a state the emitting pass reached more
        cheaply -- so each round recombines the union rather than appending.
        """
        g = self.graph
        if g.num_non_emitting_arcs == 0:
            return active, cost
        # A frontier, as `ProcessNonemitting` keeps a queue: after the first
        # round only a state whose cost *improved* can give anything new, so
        # re-expanding the whole token list repeats work that is guaranteed
        # to lose the recombination. It was doing exactly that -- 1,670
        # expansions for 1,092 frames, and the extra 578 found nothing,
        # because the states factoring introduces stay active and get
        # expanded again every frame.
        frontier, frontier_cost = active, cost
        # Where each frontier token sits in the token list, because the
        # history records parents as indices into that list. Carried rather
        # than looked up: `active` is `dst[chosen]` in arc order and so is
        # **not** sorted, and a `searchsorted` here gave wrong parents --
        # which the oracle caught and no hand-written graph did.
        frontier_index = torch.arange(active.numel(), dtype=torch.int64)
        for _round in range(self.max_epsilon_rounds):
            # The out-degrees first, since they are two gathers: most
            # second rounds end here, without paying for a `searchsorted`
            # over the whole non-emitting arc list.
            starts = g.eps_row_splits[frontier]
            counts = g.eps_row_splits[frontier + 1] - starts
            if int(counts.sum()) == 0:
                return active, cost
            # Only the prefix of each state's arcs that can be within the
            # cutoff, found by one `searchsorted` against the sorted
            # residuals. Expanding all of them and pruning afterwards was
            # what this did first, and at beam 32 that evaluated 162,533
            # arcs a frame to keep 24% of them.
            limits = torch.minimum(
                counts, _prefix_limits(g, frontier, cutoff - frontier_cost)
            )
            arc, token = _arcs_within(g.eps_row_splits, frontier, limits)
            if arc.numel() == 0:
                return active, cost
            self.counters["non_emitting"] += int(arc.numel())
            candidate = frontier_cost[token] + g.eps_cost[arc].double()
            # The prefix is a superset, because the key is quantised and
            # rounded outwards, so the exact comparison still decides.
            alive = candidate <= cutoff
            if not bool(alive.any()):
                return active, cost
            arc, token, candidate = arc[alive], token[alive], candidate[alive]
            self.counters["rounds"] += 1
            # The frontier's tokens are a subset of `active`, so a candidate
            # has to compete with the whole token list and not just with the
            # frontier -- a non-emitting arc may reach a state the emitting
            # pass already reached more cheaply.
            held = active.numel()
            dst = torch.cat([active, g.eps_dst[arc].long()])
            merged = torch.cat([cost, candidate])
            from_arc = torch.cat([torch.full((held,), -1, dtype=torch.int64), arc])
            parent = torch.cat(
                [torch.arange(held, dtype=torch.int64), frontier_index[token]]
            )
            chosen = self._recombine(dst, merged, from_arc)
            won = chosen.clone()
            won[:held] = False
            if int(won.sum()) == 0:
                return active, cost
            # Positions in the token list this round is about to become,
            # for the next round's parents.
            rank = torch.cumsum(chosen.long(), 0) - 1
            active, cost = dst[chosen], merged[chosen]
            history.append((parent[chosen].int(), from_arc[chosen].int(), True))
            # Only what improved can give anything in the round after.
            frontier, frontier_cost = dst[won], merged[won]
            frontier_index = rank[won]
        logging.warning(
            "the non-emitting closure did not settle in %d rounds; the graph "
            "has a chain of non-emitting arcs longer than this search expects",
            self.max_epsilon_rounds,
        )
        return active, cost


def decode_archive(
    graph: SearchGraph,
    utterances: Iterable[Tuple[str, torch.Tensor]],
    idx2word: Dict[int, str],
    acoustic_scale: float = 1.0,
    beam: float = DEFAULT_BEAM,
    max_active: int = DEFAULT_MAX_ACTIVE,
    min_active: int = DEFAULT_MIN_ACTIVE,
    report_every: int = 100,
) -> List[Tuple[str, List[str]]]:
    """
    Decode an archive of log probabilities, one utterance at a time.

    One at a time with no apology this time: the search is frame
    synchronous over one utterance's own active set, so batching buys
    nothing here -- which is the opposite of `intersect_dense_pruned`,
    where a batch was worth 2.65x because the work is tensor-shaped. What
    that gets back is the isolation batching cost: an utterance that fails
    names itself instead of taking seven others with it.

    Arguments
    ---------
    graph: SearchGraph
        The prepared decoding graph.
    utterances: Iterable[Tuple[str, torch.Tensor]]
        Utterance id and its `(frames, tokens)` log probabilities.
    idx2word: Dict[int, str]
        The word symbol table, inverted.
    acoustic_scale: float
        See `ViterbiDecoder`.
    beam: float
        See `ViterbiDecoder`.
    max_active: int
        See `ViterbiDecoder`.
    min_active: int
        See `ViterbiDecoder`.
    report_every: int
        Log a line after this many utterances, so a long run can be told
        from a hung one. Zero turns it off.

    Returns
    -------
    List[Tuple[str, List[str]]]
        Utterance id and its words, in archive order -- which the recipe
        relies on, since it merges this against `ref.wrd.trn` by position.
    """
    decoder = ViterbiDecoder(
        graph,
        beam=beam,
        max_active=max_active,
        min_active=min_active,
        acoustic_scale=acoustic_scale,
    )
    results: List[Tuple[str, List[str]]] = []
    for uttid, scores in utterances:
        if scores.dim() != 2:
            raise ValueError(
                f"{uttid}: expected (frames, tokens), got {tuple(scores.shape)}"
            )
        ids = decoder.decode(scores)
        results.append((uttid, [idx2word[i] for i in ids if i in idx2word]))
        if report_every and len(results) % report_every == 0:
            counters = decoder.counters
            logging.info(
                "decoded %d utterances (%.0f emitting and %.0f non-emitting "
                "candidate arcs a frame)",
                len(results),
                counters["emitting"] / max(counters["frames"], 1),
                counters["non_emitting"] / max(counters["frames"], 1),
            )
    return results


def decode_shards(
    graph: SearchGraph,
    shards: Sequence[Callable[[], Iterable[Tuple[str, torch.Tensor]]]],
    idx2word: Dict[int, str],
    acoustic_scale: float = 1.0,
    beam: float = DEFAULT_BEAM,
    max_active: int = DEFAULT_MAX_ACTIVE,
    min_active: int = DEFAULT_MIN_ACTIVE,
) -> List[Tuple[str, List[str]]]:
    """
    Decode several shards of an archive at once, one process each.

    This is what `decode_faster.sh` gets from `run.pl JOB=1:8`, and without
    it a comparison of the two stages is a comparison of one core against
    four. Threads do not substitute: the tensors a frame touches are tens
    of thousands of elements, so four torch threads were measured at 1.44x
    against one, where four processes are 3.3x.

    **Processes by `fork`, and the reason is memory.** The search only ever
    reads the graph, so a forked child shares its 934MiB by
    copy-on-write and never copies it; the alternative -- each process
    building or loading its own -- is 934MiB each and does not fit a
    15.6GiB runner at four. `share_memory_()` would also work and is not
    used: it needs `/dev/shm`, which is 64MB inside Docker by default, and
    the nightly runs the recipe inside Docker. This is the same pattern
    torch's own `DataLoader` uses for its workers on POSIX.

    Each shard's results are written to a file of its own and concatenated
    in shard order, so a block-wise split restores the archive's order
    exactly -- which the recipe depends on. A shard that fails takes the
    whole call down with its exit status named, rather than leaving a
    short archive that would score as a run of deletions.

    Arguments
    ---------
    graph: SearchGraph
        The prepared decoding graph. Already built when this is called, so
        that the children inherit it rather than build it.
    shards: Sequence[Callable[[], Iterable[Tuple[str, torch.Tensor]]]]
        One callable per shard, each yielding that shard's utterances when
        called. Callables rather than iterables because the point is for
        the *child* to read the features: materialising 1,089 utterances of
        mini-librispeech in the parent is 1.6GB before any of it is
        decoded.
    idx2word: Dict[int, str]
        The word symbol table, inverted.
    acoustic_scale: float
        See `ViterbiDecoder`.
    beam: float
        See `ViterbiDecoder`.
    max_active: int
        See `ViterbiDecoder`.
    min_active: int
        See `ViterbiDecoder`.

    Returns
    -------
    List[Tuple[str, List[str]]]
        Every shard's results, concatenated in shard order.

    Raises
    ------
    RuntimeError
        If a shard fails, naming which -- on every platform, including the
        ones that cannot fork and decode the shards in this process. A
        short archive scores as a run of deletions and reads as a bad
        model, so this must never be allowed to pass for success.
    """
    search = dict(
        acoustic_scale=acoustic_scale,
        beam=beam,
        max_active=max_active,
        min_active=min_active,
    )
    if len(shards) == 1:
        return decode_archive(graph, shards[0](), idx2word, **search)
    if "fork" not in multiprocessing.get_all_start_methods():
        logging.warning(
            "this platform cannot fork, so %d shards will be decoded one "
            "after another in this process",
            len(shards),
        )
        results: List[Tuple[str, List[str]]] = []
        for index, shard in enumerate(shards):
            try:
                results.extend(
                    decode_archive(graph, shard(), idx2word, report_every=0, **search)
                )
            except Exception as failure:
                # Named the same way the forked path names it, so a failing
                # shard reads the same on every platform. Windows took this
                # branch and reported the shard's own exception instead,
                # which is a difference nobody would notice until they were
                # reading a Windows log to find out which shard broke.
                raise RuntimeError(
                    f"decoding failed in 1 of {len(shards)} shards: "
                    f"shard {index} raised {failure!r}"
                ) from failure
        return results

    context = multiprocessing.get_context("fork")
    directory = Path(tempfile.mkdtemp(prefix="bennevis-decode-"))
    # One torch thread in the parent *before* forking, for two reasons.
    # Oversubscription is the obvious one: four shards of four threads on
    # four cores spend the machine on contention, which is why `run.pl`
    # gives each Kaldi job one core. The other is that forking a process
    # with a live thread pool is what Python 3.12 warns about and what can
    # deadlock a child inside OpenMP -- with one thread there is no pool to
    # inherit. Restored afterwards, since the caller did not ask for this.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        workers = []
        for index, shard in enumerate(shards):
            output = directory / f"shard.{index}.pickle"
            process = context.Process(
                target=_decode_one_shard,
                args=(graph, shard, idx2word, search, output, index),
                daemon=False,
            )
            process.start()
            workers.append((process, output, index))
        results = []
        failures = []
        for process, output, index in workers:
            process.join()
            if process.exitcode != 0:
                failures.append(f"shard {index} exited with {process.exitcode}")
                continue
            if not output.is_file():
                failures.append(f"shard {index} wrote nothing")
                continue
            with output.open("rb") as handle:
                results.extend(pickle.load(handle))
        if failures:
            raise RuntimeError(
                "decoding failed in "
                + f"{len(failures)} of {len(shards)} shards: "
                + "; ".join(failures)
            )
        return results
    finally:
        torch.set_num_threads(threads)
        shutil.rmtree(directory, ignore_errors=True)


def _decode_one_shard(
    graph: SearchGraph,
    shard: Callable[[], Iterable[Tuple[str, torch.Tensor]]],
    idx2word: Dict[int, str],
    search: Dict[str, object],
    output: Path,
    index: int,
) -> None:
    """
    One shard, in a forked child, writing its results where told.

    One torch thread, because the parent's four would otherwise be
    multiplied by the number of shards and spend the machine on contention
    -- the same reason `run.pl` gives each Kaldi job a single core.
    """
    torch.set_num_threads(1)
    results = decode_archive(
        graph,
        shard(),
        idx2word,
        report_every=0,
        **search,
    )
    with output.open("wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("shard %d decoded %d utterances", index, len(results))
