"""
Word error rate, computed in Python from NIST transcription files.

Scoring has been Kaldi's job: `run/decode_faster.sh` ends by invoking
`sclite` from sctk, which prints a table for a human to read. That is fine
for reading and poor for a regression gate, which needs a number a test can
compare. Parsing `sclite`'s table back is possible but fragile, and it keeps
sctk on the measurement path for a calculation that is thirty lines of
Python.

So the error rate is computed here, and `sclite` becomes the independent
oracle rather than the source: the recipe tier asserts that the two agree on
real output. That is a stronger check than either alone, and it is what makes
a WER ledger possible without Kaldi.

The `.trn` format is NIST's, and is what `decode_faster.sh` and
`run/predict.py` already write:

    YES NO NO YES (global-1_0_1_1)

The words come first and the final parenthesised field is the utterance id,
conventionally `speaker-utterance`.

Functions
---------
read_trn
    Read a `.trn` file into an id-to-words mapping.
levenshtein
    The edit distance between two sequences, with its edit counts.
word_error_rate
    Score one pair of word sequences.
score
    Score a whole reference against a whole hypothesis.
parse_sclite_wer
    Pull the overall error rate out of `sclite`'s table, for cross-checking.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Counts",
    "read_trn",
    "levenshtein",
    "word_error_rate",
    "score",
    "parse_sclite_wer",
]

#: `YES NO (global-1_0_1_1)` -- words, then a parenthesised id.
_TRN_RE = re.compile(r"^(?P<words>.*?)\s*\((?P<utterance>[^()]*)\)\s*$")

#: The `Sum/Avg` row of `sclite -o all`: speaker, sentences, words, then
#: correct, substitutions, deletions, insertions, error rate, sentence error.
_SCLITE_SUM_RE = re.compile(
    r"\|\s*Sum/Avg\s*\|\s*(\d+)\s+(\d+)\s*\|"
    r"\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
)


@dataclass(frozen=True)
class Counts:
    """
    The result of scoring, as counts rather than a single rate.

    Counts rather than a percentage because they add up across utterances,
    which a percentage does not: the corpus error rate is the total errors
    over the total reference words, never the mean of per-utterance rates.

    Attributes
    ----------
    substitutions: int
        Reference words replaced by a different word.
    deletions: int
        Reference words missing from the hypothesis.
    insertions: int
        Hypothesis words with no reference word.
    reference_words: int
        The length of the reference, which is the denominator.
    sentences: int
        How many utterances were scored.
    sentence_errors: int
        How many of them had at least one error.
    """

    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_words: int = 0
    sentences: int = 0
    sentence_errors: int = 0

    @property
    def errors(self) -> int:
        """The total edit distance."""
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        """
        The word error rate as a percentage.

        An empty reference scores 0.0 when the hypothesis is also empty and
        100.0 times the insertions otherwise, rather than dividing by zero.
        """
        if self.reference_words == 0:
            return 0.0 if self.errors == 0 else 100.0 * self.errors
        return 100.0 * self.errors / self.reference_words

    @property
    def ser(self) -> float:
        """The sentence error rate as a percentage."""
        if self.sentences == 0:
            return 0.0
        return 100.0 * self.sentence_errors / self.sentences

    @property
    def correct(self) -> int:
        """Reference words matched exactly."""
        return self.reference_words - self.substitutions - self.deletions

    def __add__(self, other: "Counts") -> "Counts":
        """Accumulate two results."""
        if not isinstance(other, Counts):  # pragma: no cover - defensive
            return NotImplemented
        return Counts(
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
            reference_words=self.reference_words + other.reference_words,
            sentences=self.sentences + other.sentences,
            sentence_errors=self.sentence_errors + other.sentence_errors,
        )


def read_trn(path: Path) -> Dict[str, List[str]]:
    """
    Read a NIST `.trn` file.

    Arguments
    ---------
    path: Path
        The file, whose lines are `word word ... (utterance-id)`.

    Returns
    -------
    Dict[str, List[str]]
        Utterance id to its words. Blank lines are skipped; an utterance with
        no words maps to an empty list.

    Raises
    ------
    ValueError
        If a line has no parenthesised utterance id, or if an id repeats.
    """
    transcriptions: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            match = _TRN_RE.match(line.rstrip("\n"))
            if match is None:
                raise ValueError(
                    f"{path}:{number}: no parenthesised utterance id in "
                    f"{line.strip()!r}"
                )
            utterance = match.group("utterance").strip()
            if utterance in transcriptions:
                raise ValueError(f"{path}:{number}: duplicate id {utterance!r}")
            transcriptions[utterance] = match.group("words").split()
    return transcriptions


def levenshtein(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> Tuple[int, int, int]:
    """
    The edit distance between two word sequences, as its three edit counts.

    A standard two-row dynamic program with unit costs, which is what NIST
    scoring uses. Ties are broken towards substitution, then deletion, then
    insertion; the total distance is the same whichever way a tie goes, and
    only the split between the three counts can differ.

    Arguments
    ---------
    reference: Sequence[str]
        The reference words.
    hypothesis: Sequence[str]
        The hypothesis words.

    Returns
    -------
    Tuple[int, int, int]
        Substitutions, deletions and insertions.
    """
    # Each cell holds (substitutions, deletions, insertions); the distance is
    # their sum. Carrying the breakdown costs nothing here and is what makes
    # the output comparable with sclite's table.
    previous: List[Tuple[int, int, int]] = [
        (0, 0, index) for index in range(len(hypothesis) + 1)
    ]

    for i, reference_word in enumerate(reference, start=1):
        current: List[Tuple[int, int, int]] = [(0, i, 0)]
        for j, hypothesis_word in enumerate(hypothesis, start=1):
            if reference_word == hypothesis_word:
                current.append(previous[j - 1])
                continue
            substitution = previous[j - 1]
            deletion = previous[j]
            insertion = current[j - 1]
            best = min(
                (sum(substitution) + 1, 0, substitution),
                (sum(deletion) + 1, 1, deletion),
                (sum(insertion) + 1, 2, insertion),
            )
            _, which, (subs, dels, inss) = best
            if which == 0:
                current.append((subs + 1, dels, inss))
            elif which == 1:
                current.append((subs, dels + 1, inss))
            else:
                current.append((subs, dels, inss + 1))
        previous = current

    return previous[-1]


def word_error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> Counts:
    """
    Score one utterance.

    Arguments
    ---------
    reference: Sequence[str]
        The reference words.
    hypothesis: Sequence[str]
        The hypothesis words.

    Returns
    -------
    Counts
        The counts for this utterance alone.
    """
    substitutions, deletions, insertions = levenshtein(reference, hypothesis)
    errors = substitutions + deletions + insertions
    return Counts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=len(reference),
        sentences=1,
        sentence_errors=1 if errors else 0,
    )


def score(
    reference: Dict[str, List[str]],
    hypothesis: Dict[str, List[str]],
    *,
    require_all: bool = True,
) -> Counts:
    """
    Score a whole reference against a whole hypothesis.

    Arguments
    ---------
    reference: Dict[str, List[str]]
        Utterance id to reference words, as `read_trn` returns.
    hypothesis: Dict[str, List[str]]
        Utterance id to hypothesis words.
    require_all: bool
        Whether every reference utterance must appear in the hypothesis. True
        by default: a decoder that silently dropped utterances would
        otherwise score well on the ones it kept, which is the most
        flattering possible way to fail.

    Returns
    -------
    Counts
        The accumulated counts, from which `wer` follows.

    Raises
    ------
    KeyError
        If `require_all` and an utterance is missing, or if the hypothesis
        has an utterance the reference does not.
    """
    missing = sorted(set(reference) - set(hypothesis))
    if missing and require_all:
        raise KeyError(
            f"{len(missing)} utterance(s) in the reference are missing from "
            f"the hypothesis, the first being {missing[0]!r}"
        )
    unexpected = sorted(set(hypothesis) - set(reference))
    if unexpected:
        raise KeyError(
            f"{len(unexpected)} utterance(s) in the hypothesis are not in "
            f"the reference, the first being {unexpected[0]!r}"
        )

    total = Counts()
    for utterance, words in reference.items():
        total = total + word_error_rate(words, hypothesis.get(utterance, []))
    return total


def parse_sclite_wer(text: str) -> Optional[float]:
    """
    The overall word error rate from `sclite -o all` output.

    Used to cross-check the computation above against sctk on real output,
    rather than as the source of the number. `sclite`'s table is formatted
    for a human, so this is deliberately narrow: it reads the `Err` column of
    the `Sum/Avg` row and returns None if that row is not there, leaving the
    caller to decide whether a missing number is a failure.

    Arguments
    ---------
    text: str
        The contents of a `results.wrd.txt`.

    Returns
    -------
    Optional[float]
        The error rate as a percentage, or None if no summary row was found.
    """
    match = _SCLITE_SUM_RE.search(text)
    if match is None:
        return None
    return float(match.group(7))
