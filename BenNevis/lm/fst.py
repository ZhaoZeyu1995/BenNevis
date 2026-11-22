"""
Turning an ARPA back-off model into a weighted FST.

This is the other half of replacing Kaldi's `arpa2fst`. The construction is
the standard one, and worth stating because the code is short enough to look
like it is doing less than it is:

* **A state per history.** The empty history is the unigram state; `<s>` is
  the start. A state exists for every n-gram that can *be* a history, which
  is every entry below the highest order.
* **An arc per n-gram.** The entry `h w` becomes an arc from `state(h)` to
  `state((h + w) truncated to order - 1)`, labelled `w:w`, weighted
  `-logprob * ln(10)`.
* **A back-off arc per history**, from `state(h)` to `state(h[1:])`,
  labelled `disambig:<eps>` and weighted by the back-off. The disambiguation
  symbol rather than `<eps>` on the input side is what keeps the composed
  graph determinisable -- two paths differing only in how far they backed
  off would otherwise be indistinguishable.
* **`</s>` is a final weight, not an arc.** This is what makes the FST
  finite: without it every path could be extended for ever.

The result is not byte-identical to `arpa2fst`'s and is not meant to be --
see the equivalence standard in CLAUDE.md. It is checked by assigning the
same weight to the same word sequences, against the back-off rule in
`BenNevis.lm.arpa` and against the `G.fst` Kaldi actually produced.

Functions
---------
build_fst_text
    The OpenFST text form, with integer labels.
total_weight
    Walk a built FST and total a word sequence's weight; the checking tool.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from BenNevis.lm.arpa import BOS, EOS, LOG10_TO_NAT, ArpaModel

__all__ = ["build_fst_text", "build_arcs", "start_state", "total_weight", "Arc"]

#: One arc: source, destination, input label, output label, weight.
Arc = Tuple[int, int, int, int, float]

#: Kaldi's default, and what `prepare_lang.sh` puts in `words.txt`.
DISAMBIG = "#0"


def _weight(logprob: float) -> float:
    """An ARPA base-ten log probability as an FST weight."""
    return -logprob * LOG10_TO_NAT


def build_arcs(
    model: ArpaModel,
    symbols: Dict[str, int],
    disambig: str = DISAMBIG,
) -> Tuple[List[Arc], Dict[int, float], Dict[Tuple[str, ...], int]]:
    """
    The arcs, final weights and state numbering of the model's FST.

    Arguments
    ---------
    model: ArpaModel
        The parsed model.
    symbols: Dict[str, int]
        The word symbol table, mapping a word to its integer id. Every word
        the model scores must be in it.
    disambig: str
        The symbol labelling back-off arcs on the input side.

    Returns
    -------
    Tuple[List[Arc], Dict[int, float], Dict[Tuple[str, ...], int]]
        The arcs, the final weights by state, and the history-to-state map.
        The last is returned for the tests rather than used here: it is the
        only way to say anything about the *structure* of a construction
        whose state numbering is deliberately not Kaldi's.

    Raises
    ------
    KeyError
        If a word the model scores is absent from the symbol table, or the
        disambiguation symbol is. Both mean the lexicon and the model were
        prepared from different vocabularies, which is a mistake that would
        otherwise show up as a bad WER.
    """
    if disambig not in symbols:
        raise KeyError(
            f"the symbol table has no {disambig!r}; prepare_lang.sh adds it, "
            f"and the back-off arcs cannot be labelled without it"
        )
    disambig_id = symbols[disambig]

    # A history needs a state only if some n-gram actually extends it.
    # Giving one to every entry below the highest order instead produces
    # 1,162,307 states for mini-librispeech's 3-gram against arpa2fst's
    # 249,532 -- functionally the same FST, since the extra states carry
    # nothing but a back-off arc, but 4.7 times the states, and G is
    # composed and determinised into a TLG that is already 334MB.
    histories = {()}
    if (BOS,) in model.entries:
        # Only where the model has one. Adding it unconditionally gives a
        # model with no sentence boundary a start state reachable by
        # nothing, whose only arc is a free back-off to the unigram state --
        # harmless but a lie about the model's shape.
        histories.add((BOS,))
    for words in model.entries:
        if len(words) > 1:
            histories.add(words[:-1])
    # A history must be a state before an arc can point at it, and the
    # longest-suffix search below relies on every suffix of a state also
    # being one.
    for history in list(histories):
        for cut in range(1, len(history)):
            histories.add(history[cut:])

    # The numbering is this construction's own. Sorted, so it is at least
    # deterministic across runs and platforms -- which byte-comparing two
    # *Python* captures still needs, even though Kaldi's numbering is not
    # being reproduced.
    ordered = sorted(histories, key=lambda h: (len(h), h))
    state_of = {history: index for index, history in enumerate(ordered)}

    def truncate(history: Tuple[str, ...]) -> Tuple[int, Tuple[str, ...]]:
        """The longest suffix of `history` that is a state."""
        candidate = history
        while candidate not in state_of:
            candidate = candidate[1:]
        return state_of[candidate], candidate

    arcs: List[Arc] = []
    finals: Dict[int, float] = {}

    for words, entry in sorted(
        model.entries.items(), key=lambda kv: (len(kv[0]), kv[0])
    ):
        history, word = words[:-1], words[-1]
        if history not in state_of:
            # A history the model gives probabilities for but that cannot be
            # reached -- it exists only at the highest order. Nothing to
            # hang the arc on, and backing off reaches the same weight.
            continue
        source = state_of[history]

        if word == EOS:
            # Not an arc: the sentence ends here.
            finals[source] = _weight(entry.logprob)
            continue
        if word == BOS:
            # <s> is a history, never a prediction. arpa2fst drops it too.
            continue

        if word not in symbols:
            raise KeyError(
                f"{word!r} is in the language model but not in the symbol "
                f"table; the lexicon and the model disagree"
            )
        destination, _ = truncate(words)
        arcs.append(
            (source, destination, symbols[word], symbols[word], _weight(entry.logprob))
        )

    # Back-off arcs. Every state but the unigram one backs off to its
    # suffix; the unigram state has nothing shorter to fall back to.
    for history in ordered:
        if not history:
            continue
        source = state_of[history]
        destination, _ = truncate(history[1:])
        arcs.append(
            (source, destination, disambig_id, 0, _weight(model.backoff_of(history)))
        )

    # The unigram state must be final, or a sentence whose last history has
    # no </s> entry has nowhere to end. Its weight is </s>'s unigram.
    unigram = state_of[()]
    if unigram not in finals and (EOS,) in model.entries:
        finals[unigram] = _weight(model.entries[(EOS,)].logprob)

    return arcs, finals, state_of


def start_state(state_of: Dict[Tuple[str, ...], int]) -> int:
    """
    The state a sentence starts from.

    `<s>`'s history, falling back to the unigram state for a model that has
    no `<s>` at all. Named rather than inlined because getting it wrong is
    invisible on a unigram model and wrong on every other -- the lowest
    numbered state is the unigram one, and starting there scores every
    sentence as though it had no beginning.

    Arguments
    ---------
    state_of: Dict[Tuple[str, ...], int]
        The history-to-state map from `build_arcs`.

    Returns
    -------
    int
        The start state.
    """
    return state_of.get((BOS,), state_of[()])


def build_fst_text(
    model: ArpaModel,
    symbols: Dict[str, int],
    disambig: str = DISAMBIG,
) -> str:
    """
    The model as OpenFST text, with integer labels.

    The start state is whichever state `<s>` is, because OpenFST's text
    format takes the source of the first line as the start -- so the arcs
    out of `<s>` are written first.

    Arguments
    ---------
    model: ArpaModel
        The parsed model.
    symbols: Dict[str, int]
        The word symbol table.
    disambig: str
        The symbol labelling back-off arcs.

    Returns
    -------
    str
        Text suitable for `fstcompile`, or for `k2.Fsa.from_openfst`.
    """
    arcs, finals, state_of = build_arcs(model, symbols, disambig)
    start = start_state(state_of)

    lines: List[str] = []
    # The start state's arcs first, so OpenFST reads the right start state.
    for arc in sorted(arcs, key=lambda a: (a[0] != start, a[0], a[2], a[1])):
        lines.append(f"{arc[0]}\t{arc[1]}\t{arc[2]}\t{arc[3]}\t{arc[4]:.8g}")
    for state in sorted(finals):
        lines.append(f"{state}\t{finals[state]:.8g}")
    return "\n".join(lines) + "\n"


def total_weight(
    arcs: Sequence[Arc],
    finals: Dict[int, float],
    symbols: Dict[str, int],
    sentence: Iterable[str],
    start: int = 0,
    disambig: str = DISAMBIG,
) -> Optional[float]:
    """
    The least weight the FST gives a word sequence.

    A shortest-path search rather than a single walk, because a back-off FST
    has several paths for one sentence -- taking a back-off arc or not -- and
    the semiring's answer is the cheapest. That is the whole reason the
    construction needs checking: an arc weighted wrong makes some *other*
    path the cheapest, and the total silently changes.

    Arguments
    ---------
    arcs: Sequence[Arc]
        The arcs, as `build_arcs` returns them.
    finals: Dict[int, float]
        The final weights.
    symbols: Dict[str, int]
        The word symbol table.
    sentence: Iterable[str]
        The words, without boundaries.
    start: int
        The state to start from, which must be `<s>`'s state for the total
        to mean the same as the back-off rule's. Not inferred: the lowest
        numbered state is the *unigram* state, and starting there silently
        scores every sentence as though it had no beginning -- which a
        unigram model cannot tell apart, so it would agree on yesno and be
        wrong everywhere else.
    disambig: str
        The symbol labelling back-off arcs, which consume no word.

    Returns
    -------
    Optional[float]
        The cheapest total, or None if the sentence has no accepting path.
    """
    disambig_id = symbols[disambig]
    by_source: Dict[int, List[Arc]] = {}
    for arc in arcs:
        by_source.setdefault(arc[0], []).append(arc)

    words = [symbols[word] for word in sentence]
    # Dijkstra over (state, words consumed). Back-off arcs consume nothing,
    # so the search has to allow them without advancing the position.
    import heapq

    best: Dict[Tuple[int, int], float] = {}
    queue: List[Tuple[float, int, int]] = [(0.0, start, 0)]
    answer: Optional[float] = None

    while queue:
        cost, state, position = heapq.heappop(queue)
        key = (state, position)
        if key in best and best[key] <= cost:
            continue
        best[key] = cost

        if position == len(words) and state in finals:
            total = cost + finals[state]
            answer = total if answer is None else min(answer, total)

        for _, destination, ilabel, _olabel, weight in by_source.get(state, []):
            if ilabel == disambig_id:
                heapq.heappush(queue, (cost + weight, destination, position))
            elif position < len(words) and ilabel == words[position]:
                heapq.heappush(queue, (cost + weight, destination, position + 1))

    return answer
