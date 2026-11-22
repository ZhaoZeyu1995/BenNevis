"""
Reading an ARPA back-off n-gram model, and scoring with it.

This is the first half of replacing Kaldi's `arpa2fst`. The FST construction
is built on top of it, and this module exists separately because it is the
**oracle** for that construction: the back-off rule applied directly to the
model is an independent way of computing what the FST must compute, so the
two can be compared without reference to Kaldi at all.

ARPA holds base-ten logarithms; FST weights are negative natural logarithms,
which is the convention OpenFST's tropical semiring uses and what Kaldi
writes. The conversion is the only arithmetic here that is easy to get
silently wrong, so `LOG10_TO_NAT` is named rather than inlined.

The back-off rule, for a sequence `w` after history `h`:

    if (h, w) is in the model:          its log probability
    else:                               backoff(h) + score(h[1:], w)

with `score` on the empty history falling back to the unigram, and a word
absent even as a unigram being an error rather than a silent zero -- a
lexicon and a language model that disagree about the vocabulary is a
preparation mistake, and one that would otherwise show up as an
unexplainably bad WER.

Classes
-------
NGram
    One entry: its words, its log probability and its back-off weight.
ArpaModel
    The whole model, with the orders it holds and a scorer.

Functions
---------
read_arpa
    Parse an ARPA file.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import gzip
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = ["LOG10_TO_NAT", "NGram", "ArpaModel", "read_arpa"]

#: ARPA stores base-ten logarithms; FST weights are negative natural logs.
#: A weight is therefore `-logprob * LOG10_TO_NAT`.
LOG10_TO_NAT = math.log(10.0)

#: The sentence boundary symbols ARPA uses. `<s>` never labels an arc -- it
#: is the start history -- and `</s>` becomes a final weight rather than an
#: arc, which is what makes an FST of a language model finite.
BOS = "<s>"
EOS = "</s>"

#: `\data\`, `\2-grams:`, `\end\`
_SECTION_RE = re.compile(r"^\\(?:(?P<order>\d+)-grams:|(?P<other>data\\|end\\))\s*$")
_COUNT_RE = re.compile(r"^ngram\s+(?P<order>\d+)\s*=\s*(?P<count>\d+)\s*$")


@dataclass(frozen=True)
class NGram:
    """
    One n-gram entry.

    Attributes
    ----------
    words: Tuple[str, ...]
        The sequence, in order, so `("the", "cat")` is a bigram.
    logprob: float
        Its base-ten log probability, as the file gives it.
    backoff: Optional[float]
        The base-ten log back-off weight for using this sequence as a
        history, or None where the file gives none -- which is the case for
        the highest order, and for any entry that is never a history.
    """

    words: Tuple[str, ...]
    logprob: float
    backoff: Optional[float] = None

    @property
    def order(self) -> int:
        """How many words the entry covers."""
        return len(self.words)


@dataclass
class ArpaModel:
    """
    A back-off n-gram model.

    Attributes
    ----------
    entries: Dict[Tuple[str, ...], NGram]
        Every entry, keyed by its word sequence.
    counts: Dict[int, int]
        The counts the `\\data\\` section declared, by order. Kept so a file
        whose header disagrees with its body can be rejected.
    """

    entries: Dict[Tuple[str, ...], NGram] = field(default_factory=dict)
    counts: Dict[int, int] = field(default_factory=dict)

    @property
    def order(self) -> int:
        """The highest order present."""
        return max((len(key) for key in self.entries), default=0)

    def vocabulary(self) -> List[str]:
        """
        Every word that appears as a unigram, sorted.

        Returns
        -------
        List[str]
            The unigram vocabulary, which is what the model can score.
        """
        return sorted(key[0] for key in self.entries if len(key) == 1)

    def backoff_of(self, history: Tuple[str, ...]) -> float:
        """
        The back-off weight of a history, in base ten logs.

        Arguments
        ---------
        history: Tuple[str, ...]
            The history.

        Returns
        -------
        float
            Its back-off weight, or 0.0 where the model gives none -- which
            is what a missing back-off means, not an error.
        """
        entry = self.entries.get(history)
        if entry is None or entry.backoff is None:
            return 0.0
        return entry.backoff

    def logprob(self, history: Sequence[str], word: str) -> float:
        """
        The base-ten log probability of a word after a history.

        The back-off rule, applied directly. This is the oracle the FST
        construction is checked against, so it is written to be obviously
        the rule rather than efficiently.

        Arguments
        ---------
        history: Sequence[str]
            The preceding words, longest first in the usual sense -- that
            is, `("the", "black")` for predicting after "the black".
        word: str
            The word to score.

        Returns
        -------
        float
            Its base-ten log probability.

        Raises
        ------
        KeyError
            If the word is not even a unigram. A model and a lexicon that
            disagree about the vocabulary is a preparation mistake, and
            returning a silent zero would surface as an unexplainably bad
            WER instead.
        """
        history = tuple(history)
        # Only as much history as the model's order allows.
        if len(history) >= self.order:
            history = history[len(history) - self.order + 1 :]

        while True:
            entry = self.entries.get(history + (word,))
            if entry is not None:
                return entry.logprob
            if not history:
                raise KeyError(
                    f"{word!r} is not in the language model, not even as a "
                    f"unigram; the lexicon and the model disagree about the "
                    f"vocabulary"
                )
            # Back off: pay the history's weight and shorten it.
            penalty = self.backoff_of(history)
            history = history[1:]
            shorter = self.logprob(history, word)
            return penalty + shorter

    def score(self, sentence: Iterable[str]) -> float:
        """
        The base-ten log probability of a whole sentence, with boundaries.

        Arguments
        ---------
        sentence: Iterable[str]
            The words, without `<s>` or `</s>`.

        Returns
        -------
        float
            The summed log probability, including the final `</s>`.
        """
        words = list(sentence)
        total = 0.0
        history: Tuple[str, ...] = (BOS,)
        for word in words:
            total += self.logprob(history, word)
            history = history + (word,)
        total += self.logprob(history, EOS)
        return total

    def weight(self, sentence: Iterable[str]) -> float:
        """
        The same score as an FST weight: a negative natural logarithm.

        Arguments
        ---------
        sentence: Iterable[str]
            The words, without boundaries.

        Returns
        -------
        float
            `-score * ln(10)`, which is what an FST path's weight sums to.
        """
        return -self.score(sentence) * LOG10_TO_NAT


def _open(path: Path):
    """Open an ARPA file, transparently gunzipping a `.gz`."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def read_arpa(path: Path) -> ArpaModel:
    """
    Parse an ARPA back-off model.

    Arguments
    ---------
    path: Path
        The file, optionally gzipped.

    Returns
    -------
    ArpaModel
        The parsed model.

    Raises
    ------
    ValueError
        If a section header is unrecognised, an entry has the wrong number
        of fields for its order, or the body disagrees with the counts the
        header declared. All three mean the file is not what it claims, and
        a model quietly missing entries is a WER mystery rather than a
        crash.
    """
    model = ArpaModel()
    order: Optional[int] = None
    seen: Dict[int, int] = {}

    with _open(Path(path)) as handle:
        for number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue

            count = _COUNT_RE.match(line)
            if count is not None:
                model.counts[int(count.group("order"))] = int(count.group("count"))
                continue

            section = _SECTION_RE.match(line)
            if section is not None:
                if section.group("order") is not None:
                    order = int(section.group("order"))
                else:
                    order = None
                continue

            if line.startswith("\\"):
                raise ValueError(f"{path}:{number}: unknown section {line!r}")

            if order is None:
                # Outside any n-gram section: the \data\ preamble's other
                # lines, which carry nothing this needs.
                continue

            # Any whitespace, not tabs-or-spaces-but-not-both. The real
            # LibriSpeech 3-gram writes a tab after the probability and
            # *spaces* between the words, so splitting on tab alone sees
            # `-1.486134` and `<UNK> </s>` and calls it a malformed bigram.
            # ARPA words cannot contain whitespace, so this is unambiguous.
            fields = line.split()
            if len(fields) not in (order + 1, order + 2):
                raise ValueError(
                    f"{path}:{number}: a {order}-gram needs {order + 1} or "
                    f"{order + 2} fields, got {len(fields)}: {line!r}"
                )

            logprob = float(fields[0])
            words = tuple(fields[1 : 1 + order])
            backoff = float(fields[order + 1]) if len(fields) == order + 2 else None
            model.entries[words] = NGram(words=words, logprob=logprob, backoff=backoff)
            seen[order] = seen.get(order, 0) + 1

    for declared_order, declared in model.counts.items():
        if seen.get(declared_order, 0) != declared:
            raise ValueError(
                f"{path}: the header declares {declared} {declared_order}-gram(s) "
                f"but the body holds {seen.get(declared_order, 0)}"
            )
    return model
