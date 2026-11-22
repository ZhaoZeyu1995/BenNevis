"""
Building the lexicon FST, `L.fst`, from a pronunciation dictionary.

`L` maps phone sequences to words: each word becomes a chain of arcs
emitting its phones, carrying the word on the first arc's output and
`<eps>` on the rest, returning to the loop state so any word can follow
any other.

Unlike `G.fst`, this one *can* be byte-exact, and is. Both of the choices
that `G` leaves arbitrary are pinned here: states are allocated in the
order the lexicon lists its words, and the arcs come out sorted by output
label, which is what `prepare_lang.sh`'s `fstarcsort --sort_type=olabel`
leaves behind. So this is checked against Kaldi's bytes rather than only
against its function -- the standard in CLAUDE.md asks for functional
equivalence because an FST's encoding is usually arbitrary, not because
bytes are unwelcome where they are determined.

**Optional silence is not handled yet.** `prepare_lang.sh --sil_prob 0.0`,
which `egs/yesno/asr` uses, gives the plain construction above.
A non-zero silence probability adds a start state, a silence state and
weighted entry arcs, and the exact shape is `make_lexicon_fst.pl`'s rather
than something to infer -- so a non-zero probability raises instead of
producing a plausible lexicon that decodes differently.

Functions
---------
read_lexicon
    Parse a Kaldi `lexicon.txt`.
build_lexicon_fst_text
    The OpenFST text form, with integer labels.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

__all__ = [
    "Lexicon",
    "read_lexicon",
    "build_lexicon_arcs",
    "build_lexicon_fst_text",
]

#: A word and the phones it is pronounced as.
Lexicon = List[Tuple[str, Tuple[str, ...]]]


def read_lexicon(path: Path) -> Lexicon:
    """
    Parse a Kaldi `lexicon.txt`.

    One entry per line: the word, then its phones, whitespace separated. The
    order is kept, because it decides the state numbering and so the bytes.

    Arguments
    ---------
    path: Path
        The lexicon.

    Returns
    -------
    Lexicon
        The entries, in file order. A word may appear more than once, which
        is how alternative pronunciations are written.

    Raises
    ------
    ValueError
        If a line has a word and no phones. A word that cannot be
        pronounced can never be decoded, and silently dropping it would
        show up only as that word never being recognised.
    """
    entries: Lexicon = []
    for number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        fields = raw.split()
        if not fields:
            continue
        if len(fields) < 2:
            raise ValueError(f"{path}:{number}: {fields[0]!r} has no pronunciation")
        entries.append((fields[0], tuple(fields[1:])))
    return entries


def build_lexicon_arcs(
    lexicon: Lexicon,
    words: Dict[str, int],
    phones: Dict[str, int],
    sil_prob: float = 0.0,
) -> Tuple[List[Tuple[int, int, int, int]], int]:
    """
    The arcs of `L.fst`, and its single final state.

    Arguments
    ---------
    lexicon: Lexicon
        The entries, in the order that fixes the state numbering.
    words: Dict[str, int]
        The word symbol table.
    phones: Dict[str, int]
        The phone symbol table.
    sil_prob: float
        The optional-silence probability. Only 0.0 is supported.

    Returns
    -------
    Tuple[List[Tuple[int, int, int, int]], int]
        The arcs as (source, destination, phone, word), and the loop state,
        which is both the start and the only final state.

    Raises
    ------
    NotImplementedError
        For a non-zero silence probability.
    KeyError
        If a word or phone is missing from its symbol table.
    """
    if sil_prob != 0.0:
        raise NotImplementedError(
            f"sil_prob={sil_prob} is not supported yet; only 0.0, which is "
            f"what prepare_lang.sh --sil_prob 0.0 produces. A non-zero "
            f"probability adds a start state, a silence state and weighted "
            f"entry arcs, and that shape is make_lexicon_fst.pl's rather "
            f"than something to infer"
        )

    loop = 0
    next_state = 1
    arcs: List[Tuple[int, int, int, int]] = []

    for word, pronunciation in lexicon:
        if word not in words:
            raise KeyError(f"{word!r} is in the lexicon but not in words.txt")
        for phone in pronunciation:
            if phone not in phones:
                raise KeyError(f"{phone!r} is in the lexicon but not in phones.txt")

        # The word rides the first arc's output; the rest emit <eps>.
        source = loop
        for index, phone in enumerate(pronunciation):
            last = index == len(pronunciation) - 1
            destination = loop if last else next_state
            if not last:
                next_state += 1
            arcs.append(
                (
                    source,
                    destination,
                    phones[phone],
                    words[word] if index == 0 else 0,
                )
            )
            source = destination

    return arcs, loop


def build_lexicon_fst_text(
    lexicon: Lexicon,
    words: Dict[str, int],
    phones: Dict[str, int],
    sil_prob: float = 0.0,
) -> str:
    """
    `L.fst` as OpenFST text, with integer labels.

    Sorted by output label, which is what `prepare_lang.sh`'s
    `fstarcsort --sort_type=olabel` leaves -- so the bytes match Kaldi's.

    Arguments
    ---------
    lexicon: Lexicon
        The entries.
    words: Dict[str, int]
        The word symbol table.
    phones: Dict[str, int]
        The phone symbol table.
    sil_prob: float
        The optional-silence probability. Only 0.0 is supported.

    Returns
    -------
    str
        Text suitable for `fstcompile`, or for `k2.Fsa.from_openfst`.
    """
    arcs, final = build_lexicon_arcs(lexicon, words, phones, sil_prob)
    # By source, then output label, then input label: an arc-sorted FST as
    # OpenFST writes one. The final state's line goes where Kaldi puts it,
    # which is after the start state's arcs rather than at the end.
    ordered = sorted(arcs, key=lambda a: (a[0], a[3], a[2]))
    lines: List[str] = []
    emitted_final = False
    for arc in ordered:
        if arc[0] > final and not emitted_final:
            lines.append(f"{final}")
            emitted_final = True
        lines.append(f"{arc[0]}\t{arc[1]}\t{arc[2]}\t{arc[3]}")
    if not emitted_final:
        lines.append(f"{final}")
    return "\n".join(lines) + "\n"
