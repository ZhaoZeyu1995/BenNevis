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

**Optional silence** adds a start state, a silence state, and a split on
each word's *last* arc: one copy to the loop state paying for no silence,
one to the silence state paying for it. Deliberately non-deterministic,
which is how `make_lexicon_fst.py` avoids epsilon arcs. The shape here is
that script's and was read from it rather than inferred -- and then checked,
because the Perl script of the same name has a *different* shape (its start
arcs carry the silence phone rather than `<eps>`, and it skips the split
after a silence word). Kaldi's own output settles which: `<eps>:<eps>` to
both states, and no special case.

**Weights are float32.** Kaldi's `BaseFloat` is single precision and
`fstprint` writes about nine significant figures, so `-log(0.5)` comes out
as `0.693147182` rather than `0.6931471805599453`. Computing in double and
writing the double would be a correct lexicon that is not byte-identical,
so the weight is narrowed before it is formatted. This did not arise for
`--sil_prob 0.0`, where every weight is zero and `fstprint` omits it
entirely.

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

import math
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

__all__ = [
    "Arc",
    "Lexicon",
    "read_lexicon",
    "build_lexicon_arcs",
    "build_lexicon_fst_text",
    "format_weight",
    "position_dependent",
    "add_disambiguation",
]

#: A word and the phones it is pronounced as.
Lexicon = List[Tuple[str, Tuple[str, ...]]]

#: One arc: source, destination, input (phone), output (word), weight.
Arc = Tuple[int, int, int, int, float]


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


def position_dependent(
    pronunciation: Tuple[str, ...],
) -> Tuple[str, ...]:
    """
    A pronunciation with Kaldi's word-position suffixes attached.

    `prepare_lang.sh` does this unless `--position-dependent-phones false`,
    and it is the default -- so mini-librispeech's lexicon is written in
    `AH0` while its `phones.txt` only has `AH0_B`, `AH0_E`, `AH0_I` and
    `AH0_S`. yesno turns it off, which is why a lexicon FST that ignored
    this reproduced yesno's bytes exactly and could not build
    mini-librispeech's at all.

    The suffixes are `_S` for a word of one phone, otherwise `_B` for the
    first, `_E` for the last and `_I` for the rest.

    Note this applies to the **lexicon's** phones. The optional-silence
    phone is used unsuffixed, which is visible in Kaldi's own output: the
    silence word `!SIL` is pronounced `SIL_S` while the optional-silence
    arc carries plain `SIL`.

    Arguments
    ---------
    pronunciation: Tuple[str, ...]
        The phones, as the lexicon writes them.

    Returns
    -------
    Tuple[str, ...]
        The phones with suffixes.
    """
    if not pronunciation:
        return pronunciation
    if len(pronunciation) == 1:
        return (f"{pronunciation[0]}_S",)
    return tuple(
        (
            f"{phone}_B"
            if index == 0
            else f"{phone}_E" if index == len(pronunciation) - 1 else f"{phone}_I"
        )
        for index, phone in enumerate(pronunciation)
    )


def build_lexicon_arcs(
    lexicon: Lexicon,
    words: Dict[str, int],
    phones: Dict[str, int],
    sil_prob: float = 0.0,
    sil_phone: str = "SIL",
    sil_disambig: Optional[str] = None,
    position_dependent_phones: bool = False,
) -> Tuple[List[Arc], int]:
    """
    The arcs of `L.fst`, and its final state.

    Arguments
    ---------
    lexicon: Lexicon
        The entries, in the order that fixes the state numbering.
    words: Dict[str, int]
        The word symbol table.
    phones: Dict[str, int]
        The phone symbol table.
    sil_prob: float
        The optional-silence probability. Zero gives the plain loop; a value
        strictly between zero and one adds the silence structure.
    sil_phone: str
        The optional-silence phone, `optional_silence.txt`'s content.
    sil_disambig: Optional[str]
        A disambiguation symbol for the silence arc, which
        `prepare_lang.sh` supplies for `L_disambig.fst` and not for `L.fst`.
    position_dependent_phones: bool
        Whether to suffix the lexicon's phones by word position, which
        `prepare_lang.sh` does unless told not to. Defaults to False here
        because that is what a caller who has not thought about it wants to
        be told about by a KeyError, rather than silently getting phones
        that are not in the table.

    Returns
    -------
    Tuple[List[Arc], int]
        The arcs as (source, destination, phone, word, weight), and the
        final state -- which is the loop state, and is **not** state 0 once
        silence is in play.

    Raises
    ------
    ValueError
        If the silence probability is not in [0, 1). One is rejected
        because its no-silence cost is infinite.
    KeyError
        If a word or phone is missing from its symbol table.
    """
    if not 0.0 <= sil_prob < 1.0:
        raise ValueError(
            f"sil_prob={sil_prob} is not in [0, 1); a probability of one "
            f"would make the no-silence arc cost infinity"
        )

    def check(word: str, pronunciation: Tuple[str, ...]) -> None:
        if word not in words:
            raise KeyError(f"{word!r} is in the lexicon but not in words.txt")
        for phone in pronunciation:
            if phone not in phones:
                raise KeyError(f"{phone!r} is in the lexicon but not in phones.txt")

    arcs: List[Arc] = []

    if sil_prob == 0.0:
        loop = 0
        next_state = 1
    else:
        # start 0, loop 1, silence 2 -- the numbering make_lexicon_fst.py
        # uses, and so the numbering Kaldi's bytes have.
        loop = 1
        silence = 2
        next_state = 3
        sil_cost = -math.log(sil_prob)
        no_sil_cost = -math.log(1.0 - sil_prob)
        if sil_phone not in phones:
            raise KeyError(
                f"the optional-silence phone {sil_phone!r} is not in " f"phones.txt"
            )
        arcs.append((0, loop, 0, 0, no_sil_cost))
        arcs.append((0, silence, 0, 0, sil_cost))
        if sil_disambig is None:
            arcs.append((silence, loop, phones[sil_phone], 0, 0.0))
        else:
            if sil_disambig not in phones:
                raise KeyError(
                    f"the silence disambiguation symbol {sil_disambig!r} is "
                    f"not in phones.txt"
                )
            disambig_state = next_state
            next_state += 1
            arcs.append((silence, disambig_state, phones[sil_phone], 0, 0.0))
            arcs.append((disambig_state, loop, phones[sil_disambig], 0, 0.0))

    for word, raw_pronunciation in lexicon:
        pronunciation = (
            position_dependent(raw_pronunciation)
            if position_dependent_phones
            else raw_pronunciation
        )
        check(word, pronunciation)
        source = loop
        # Every phone but the last walks forward through fresh states.
        for index in range(len(pronunciation) - 1):
            arcs.append(
                (
                    source,
                    next_state,
                    phones[pronunciation[index]],
                    words[word] if index == 0 else 0,
                    0.0,
                )
            )
            source = next_state
            next_state += 1

        last = len(pronunciation) - 1
        phone = phones[pronunciation[last]] if last >= 0 else 0
        # The word rides the final arc too when the pronunciation is one
        # phone long, which is the `i <= 0` in Kaldi's script.
        label = words[word] if last <= 0 else 0
        if sil_prob == 0.0:
            arcs.append((source, loop, phone, label, 0.0))
        else:
            # Split: one way out paying for no silence, one paying for it.
            # Non-deterministic on purpose -- it is how the script avoids
            # epsilon arcs, and it is why a fingerprint of this FST has to
            # take a shortest path rather than walk it.
            arcs.append((source, loop, phone, label, no_sil_cost))
            arcs.append((source, silence, phone, label, sil_cost))

    return arcs, loop


def format_weight(weight: float) -> str:
    """
    A weight as `fstprint` writes it.

    Kaldi's `BaseFloat` is single precision and `fstprint` writes about nine
    significant figures, so `-log(0.5)` is `0.693147182` and not
    `0.6931471805599453`. Narrowing before formatting is the difference
    between a correct lexicon and a byte-identical one.

    Arguments
    ---------
    weight: float
        The weight, computed in double.

    Returns
    -------
    str
        Its text form.
    """
    return f"{struct.unpack('f', struct.pack('f', weight))[0]:.9g}"


def build_lexicon_fst_text(
    lexicon: Lexicon,
    words: Dict[str, int],
    phones: Dict[str, int],
    sil_prob: float = 0.0,
    sil_phone: str = "SIL",
    sil_disambig: Optional[str] = None,
    position_dependent_phones: bool = False,
) -> str:
    """
    `L.fst` as OpenFST text, with integer labels.

    Written as `fstprint` writes it: arcs grouped by source state in state
    order, a final state's line after its own arcs rather than at the end,
    arcs sorted by output label -- which is what `prepare_lang.sh`'s
    `fstarcsort --sort_type=olabel` leaves -- and a zero weight omitted
    rather than printed.

    Arguments
    ---------
    lexicon: Lexicon
        The entries.
    words: Dict[str, int]
        The word symbol table.
    phones: Dict[str, int]
        The phone symbol table.
    sil_prob: float
        The optional-silence probability.
    sil_phone: str
        The optional-silence phone.
    sil_disambig: Optional[str]
        A disambiguation symbol for the silence arc.
    position_dependent_phones: bool
        Whether to suffix the lexicon's phones by word position.

    Returns
    -------
    str
        Text suitable for `fstcompile`, or for `k2.Fsa.from_openfst`.
    """
    arcs, final = build_lexicon_arcs(
        lexicon,
        words,
        phones,
        sil_prob,
        sil_phone,
        sil_disambig,
        position_dependent_phones,
    )
    ordered = sorted(arcs, key=lambda a: (a[0], a[3], a[2], a[1]))

    lines: List[str] = []
    emitted_final = False
    for arc in ordered:
        if arc[0] > final and not emitted_final:
            lines.append(f"{final}")
            emitted_final = True
        fields = [str(arc[0]), str(arc[1]), str(arc[2]), str(arc[3])]
        if arc[4] != 0.0:
            fields.append(format_weight(arc[4]))
        lines.append("\t".join(fields))
    if not emitted_final:
        lines.append(f"{final}")
    return "\n".join(lines) + "\n"


def add_disambiguation(
    lexicon: Lexicon,
    first_allowed: int = 1,
) -> Tuple[Lexicon, int]:
    """
    Append disambiguation symbols where a lexicon needs them, as
    `add_lex_disambig.pl` does.

    `L.fst` composed with a grammar has to be determinisable, and two
    things stop it being so. A pronunciation that is a **proper prefix** of
    another cannot be told from the start of that other one. And a
    pronunciation shared by **several words** -- a homophone -- cannot say
    which word it was. Both are fixed by appending `#1`, `#2`, ... so that
    every ambiguous pronunciation becomes distinct on the input side.

    A pronunciation that is neither a prefix nor shared gets nothing, which
    is why the symbols are few: mini-librispeech needs fifteen for 206,510
    entries.

    This runs on the phones **as the FST will see them**, so after any
    position-dependent suffixing -- `prepare_lang.sh` applies the suffixes
    first and disambiguates the result, which its own documented example of
    `lexiconp_disambig.txt` shows.

    The empty pronunciation is a special case with a symbol **reserved** to
    it alone: it maps to nothing on the input side, so it cannot be
    distinguished by anything else, and a symbol another entry also used
    would not separate them.

    Arguments
    ---------
    lexicon: Lexicon
        The entries, in order.
    first_allowed: int
        The lowest symbol number this may use. `prepare_lang.sh` leaves it
        at one.

    Returns
    -------
    Tuple[Lexicon, int]
        The entries with symbols appended, and the highest symbol number
        used -- which decides the *silence* disambiguation symbol's number,
        since `prepare_lang.sh` takes the next one after this.
    """
    counts: Dict[Tuple[str, ...], int] = {}
    for _word, pronunciation in lexicon:
        counts[pronunciation] = counts.get(pronunciation, 0) + 1

    # Every proper prefix of every pronunciation, including the empty one.
    prefixes: set = set()
    for _word, pronunciation in lexicon:
        for cut in range(len(pronunciation)):
            prefixes.add(pronunciation[:cut])

    highest = first_allowed - 1
    reserved: set = set()
    last_used: Dict[Tuple[str, ...], int] = {}
    result: Lexicon = []

    for word, pronunciation in lexicon:
        if pronunciation not in prefixes and counts[pronunciation] == 1:
            result.append((word, pronunciation))
            continue

        if not pronunciation:
            # Reserved to the empty pronunciation alone; see above.
            highest += 1
            reserved.add(highest)
            result.append((word, (f"#{highest}",)))
            continue

        previous = last_used.get(pronunciation)
        current = first_allowed if previous is None else previous + 1
        while current in reserved:
            current += 1
        highest = max(highest, current)
        last_used[pronunciation] = current
        result.append((word, pronunciation + (f"#{current}",)))

    return result, highest
