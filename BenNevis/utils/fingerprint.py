"""
Functional fingerprints for a weighted FST.

An FST is compared as a function rather than as bytes -- see CLAUDE.md --
but a large one is recorded in the capture by digest alone, and a digest
answers only the byte question. `G.fst` is already known not to be
byte-equal to a Python build, because `arpa2fst` numbers its states in
whatever order it meets histories, so its digest can never match and would
say nothing at all.

A fingerprint closes that: a fixed set of word sequences, and the weight the
FST assigns each one. A few kilobytes, encoding-independent, and exactly the
comparison the standard asks for.

The sequences are drawn with a fixed seed from the symbol table, which makes
them reproducible from committed data rather than from the FST -- so two
implementations are asked the same questions. Any sequence of in-vocabulary
words works, because a back-off language model FST accepts all of them: the
back-off arcs reach the unigram state, which has an arc for every word.

That last property is what makes this straightforward for `G`. It is **not**
true of `L` or `TLG`, whose input alphabets are phones and tokens and which
accept only sequences their lexicon permits, so those need sequences drawn
from the lexicon rather than at random. Not done here.

Functions
---------
sample_word_sequences
    Draw reproducible word sequences from a symbol table.
shortest_distance
    The least weight an FST gives a label sequence.
fingerprint
    Both together: a map from sequence to weight.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import heapq
import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "NON_WORDS",
    "sample_word_sequences",
    "shortest_distance",
    "fingerprint",
]

#: Symbols that are not words and must never appear in a sampled sequence.
#: `<eps>` labels nothing, `#0` is the back-off disambiguation symbol, and
#: the sentence boundaries are a start history and a final weight rather
#: than arcs -- so each would make a sequence that says nothing about the
#: language the FST accepts.
NON_WORDS = ("<eps>", "#0", "<s>", "</s>", "<UNK>")

#: Enough sequences that a wrong arc weight is very unlikely to be missed,
#: few enough that the fingerprint stays a few kilobytes.
DEFAULT_COUNT = 64

#: Short sequences localise a fault; long ones exercise back-off depth.
DEFAULT_MIN_LENGTH = 1
DEFAULT_MAX_LENGTH = 6

#: Fixed, so the same questions are asked of every implementation.
DEFAULT_SEED = 20261122


def sample_word_sequences(
    words: Iterable[str],
    count: int = DEFAULT_COUNT,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: int = DEFAULT_SEED,
) -> List[Tuple[str, ...]]:
    """
    Draw word sequences reproducibly from a vocabulary.

    Arguments
    ---------
    words: Iterable[str]
        The symbol table's keys. Non-words are filtered out.
    count: int
        How many sequences.
    min_length: int
        The shortest sequence, in words.
    max_length: int
        The longest.
    seed: int
        The random seed, fixed so the sequences are reproducible.

    Returns
    -------
    List[Tuple[str, ...]]
        The sequences, sorted so the fingerprint's order does not depend on
        the vocabulary's iteration order.
    """
    vocabulary = sorted(
        word for word in words if word not in NON_WORDS and not word.startswith("#")
    )
    if not vocabulary:
        return []
    rng = random.Random(seed)
    sequences = {
        tuple(
            rng.choice(vocabulary) for _ in range(rng.randint(min_length, max_length))
        )
        for _ in range(count)
    }
    return sorted(sequences)


def shortest_distance(
    arcs: Mapping[int, Sequence[Tuple[int, int, float]]],
    finals: Mapping[int, float],
    labels: Sequence[int],
    start: int,
    epsilon_like: Sequence[int] = (),
) -> Optional[float]:
    """
    The least weight the FST gives a label sequence.

    Dijkstra over (state, labels consumed), because an FST with back-off
    arcs has several paths for one sequence and the tropical semiring's
    answer is the cheapest.

    Arguments
    ---------
    arcs: Mapping[int, Sequence[Tuple[int, int, float]]]
        Arcs by source state, each (destination, input label, weight).
    finals: Mapping[int, float]
        Final weights by state.
    labels: Sequence[int]
        The input labels to consume, in order.
    start: int
        The start state.
    epsilon_like: Sequence[int]
        Labels that consume nothing -- the back-off disambiguation symbol,
        and `<eps>` itself.

    Returns
    -------
    Optional[float]
        The least total, or None if the sequence is not accepted.
    """
    free = set(epsilon_like)
    best: Dict[Tuple[int, int], float] = {}
    queue: List[Tuple[float, int, int]] = [(0.0, start, 0)]
    answer: Optional[float] = None

    while queue:
        cost, state, position = heapq.heappop(queue)
        key = (state, position)
        if key in best and best[key] <= cost:
            continue
        best[key] = cost

        if position == len(labels) and state in finals:
            total = cost + finals[state]
            answer = total if answer is None else min(answer, total)

        for destination, label, weight in arcs.get(state, ()):
            if label in free:
                heapq.heappush(queue, (cost + weight, destination, position))
            elif position < len(labels) and label == labels[position]:
                heapq.heappush(queue, (cost + weight, destination, position + 1))
    return answer


def fingerprint(
    arcs: Mapping[int, Sequence[Tuple[int, int, float]]],
    finals: Mapping[int, float],
    symbols: Mapping[str, int],
    start: int,
    sequences: Optional[Sequence[Tuple[str, ...]]] = None,
    disambig: str = "#0",
) -> Dict[str, float]:
    """
    A map from word sequence to the weight the FST gives it.

    Arguments
    ---------
    arcs: Mapping[int, Sequence[Tuple[int, int, float]]]
        Arcs by source state, each (destination, input label, weight).
    finals: Mapping[int, float]
        Final weights by state.
    symbols: Mapping[str, int]
        The word symbol table.
    start: int
        The start state.
    sequences: Optional[Sequence[Tuple[str, ...]]]
        The sequences to score, or None to sample them from `symbols`.
    disambig: str
        The back-off disambiguation symbol, which consumes no word.

    Returns
    -------
    Dict[str, float]
        Space-joined sequence to weight. A sequence the FST does not accept
        is left out rather than recorded as a sentinel, so a comparison
        cannot mistake "rejected" for a number.
    """
    if sequences is None:
        sequences = sample_word_sequences(symbols.keys())
    free = [0]
    if disambig in symbols:
        free.append(symbols[disambig])

    result: Dict[str, float] = {}
    for sequence in sequences:
        if any(word not in symbols for word in sequence):
            continue
        weight = shortest_distance(
            arcs,
            finals,
            [symbols[word] for word in sequence],
            start,
            epsilon_like=free,
        )
        if weight is not None:
            result[" ".join(sequence)] = weight
    return result
