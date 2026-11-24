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
accept only sequences their lexicon permits -- a sequence sampled from
`words.txt` is rejected by them almost surely, and a fingerprint of
rejections compares nothing.

`sample_lexicon_sequences` closes that for the phone-input graphs. It draws
word sequences from the words the *lexicon* pronounces and returns each one
with the phone sequence `L` would accept for it, built with the project's
own `position_dependent` so the suffixing cannot drift from the lexicon FST
it is meant to probe.

Why this matters more than it sounds: mini-librispeech's `LG.fst.txt` is
138,155,979 bytes, so the capture keeps its digest and not its bytes, and
`G.fst` aside there was **no way to gate the graph on the one corpus where a
graph replacement matters**. The yesno equivalence test reads Kaldi's
committed `LG.fst.txt` directly, which works only because yesno's is 20KB.

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
    "fingerprint_phone_sequences",
    "position_dependent_table",
    "sample_lexicon_sequences",
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


def sample_lexicon_sequences(
    lexicon: Sequence[Tuple[str, Sequence[str]]],
    count: int = DEFAULT_COUNT,
    min_length: int = DEFAULT_MIN_LENGTH,
    max_length: int = DEFAULT_MAX_LENGTH,
    seed: int = DEFAULT_SEED,
    position_dependent_phones: bool = False,
) -> List[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """
    Draw word sequences the lexicon can pronounce, with their phones.

    Sampling from `words.txt` is no use for a phone-input FST: almost every
    sequence it draws is one the lexicon cannot pronounce, and an FST that
    rejects every question is indistinguishable from one that disagrees
    about every answer. These sequences are drawn from the words the
    lexicon actually has entries for, so the graph accepts them.

    The **first** pronunciation of each word is used, in file order. A word
    with alternatives would otherwise make the expected phone sequence
    depend on which one was picked, and the point is that two
    implementations are asked exactly the same question.

    Arguments
    ---------
    lexicon: Sequence[Tuple[str, Sequence[str]]]
        Entries as `read_lexicon` returns them, in file order.
    count: int
        How many sequences to draw.
    min_length: int
        The shortest sequence, in words.
    max_length: int
        The longest.
    seed: int
        The random seed, fixed so the sequences are reproducible.
    position_dependent_phones: bool
        Whether to suffix the phones `_B`, `_I`, `_E`, `_S`. Must match
        the setting the lexicon FST was built with: mini-librispeech takes
        the default of `prepare_lang.sh`, which is on, while yesno passes
        `--position-dependent-phones false`.

    Returns
    -------
    List[Tuple[Tuple[str, ...], Tuple[str, ...]]]
        Each sequence as its words and the phones `L` accepts for it,
        sorted so the order does not depend on a dictionary's iteration
        order.
    """
    from BenNevis.lm.lexicon import position_dependent

    first: Dict[str, Tuple[str, ...]] = {}
    for word, pronunciation in lexicon:
        if word in NON_WORDS or word.startswith("#"):
            continue
        if word not in first:
            first[word] = tuple(pronunciation)
    vocabulary = sorted(first)
    if not vocabulary:
        return []

    rng = random.Random(seed)
    sequences = {
        tuple(
            rng.choice(vocabulary) for _ in range(rng.randint(min_length, max_length))
        )
        for _ in range(count)
    }

    drawn = []
    for words in sorted(sequences):
        phones: List[str] = []
        for word in words:
            pronunciation = first[word]
            if position_dependent_phones:
                phones.extend(position_dependent(pronunciation))
            else:
                phones.extend(pronunciation)
        drawn.append((words, tuple(phones)))
    return drawn


def position_dependent_table(phones: Iterable[str]) -> bool:
    """
    Whether a phone table carries position-dependent suffixes.

    Read off the data rather than configured per corpus, because the two
    recipes differ -- mini-librispeech takes `prepare_lang.sh`'s default,
    which is on, and yesno passes `--position-dependent-phones false` --
    and a capture should not need to be told which recipe it is of.

    Arguments
    ---------
    phones: Iterable[str]
        The phone symbol table's keys.

    Returns
    -------
    bool
        True if any phone ends in `_B`, `_I`, `_E` or `_S`.
    """
    return any(
        phone.endswith(("_B", "_I", "_E", "_S"))
        for phone in phones
        if not phone.startswith("#") and phone not in NON_WORDS
    )


def fingerprint_phone_sequences(
    arcs: Mapping[int, Sequence[Tuple[int, int, float]]],
    finals: Mapping[int, float],
    phones: Mapping[str, int],
    start: int,
    drawn: Sequence[Tuple[Tuple[str, ...], Tuple[str, ...]]],
) -> Dict[str, float]:
    """
    What weight a phone-input FST gives each lexicon-drawn sequence.

    Keyed by the **word** sequence rather than the phone sequence, for two
    reasons: it is what a reader can interpret, and it is the question
    both implementations are asked, leaving the phone expansion to
    `sample_lexicon_sequences` so it cannot differ between them.

    A sequence the FST rejects is **left out** rather than recorded as
    infinity. A rejection is a fact about the lexicon rather than about the
    weights, and an entry that both implementations agree is unreachable
    would pass the comparison while testing nothing.

    Arguments
    ---------
    arcs: Mapping[int, Sequence[Tuple[int, int, float]]]
        Arcs by source, as `destination`, `input label`, `weight`.
    finals: Mapping[int, float]
        Final weights by state.
    phones: Mapping[str, int]
        The phone symbol table, mapping name to the FST's input label.
    start: int
        The start state.
    drawn: Sequence[Tuple[Tuple[str, ...], Tuple[str, ...]]]
        Word and phone sequences, as `sample_lexicon_sequences` returns.

    Returns
    -------
    Dict[str, float]
        Space-joined word sequence to weight, for the accepted ones.
    """
    recorded: Dict[str, float] = {}
    for words, phone_sequence in drawn:
        missing = [phone for phone in phone_sequence if phone not in phones]
        if missing:
            continue
        sequence = [phones[phone] for phone in phone_sequence]
        weight = shortest_distance(arcs, finals, sequence, start)
        if weight is not None:
            recorded[" ".join(words)] = weight
    return recorded


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
