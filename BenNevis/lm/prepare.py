"""
`prepare_lang.sh` in Python: the symbol tables and index files.

Stage 1 is the last of the pipeline the Python lane reads from and cannot
produce. It needs three things out of Kaldi's `utils/prepare_lang.sh`:
`phones.txt` and `words.txt`, which fix every integer label in the
directory, and `phones/align_lexicon.int`, which `Dataset` reads through
`Lang.wids2pids`. The FSTs it also produces are already Python and
byte-exact -- see `BenNevis.lm.lexicon` -- so what is left is table
construction, read off `prepare_lang.sh` rather than guessed.

Four of its decisions are not guessable from the shell, and three of them
decide the integers:

**The phone order is silence, then non-silence, then disambiguation**, each
flattened in input order, and `<eps>` first. So a phone's id depends on
which line of `silence_phones.txt` or `nonsilence_phones.txt` it appears
on and where on that line -- `nonsilence_phones.txt` groups phones per
line, `AA AA0 AA1 AA2`, and the group order is part of the answer.

**A silence phone keeps its bare form and a non-silence phone does not.**
The phone map is built with the suffix list `"" "" "_B" "_E" "_I" "_S"`
for silence and `"" "_B" "_E" "_I" "_S"` for non-silence; the first field
is the key, so silence maps to five versions *including* the unsuffixed
one and non-silence to four. That is why mini-librispeech's `phones.txt`
reads `SIL SIL_B SIL_E SIL_I SIL_S` but has no bare `AA`, and why the
optional-silence phone is unsuffixed while the silence *word* takes
`SIL_S`.

**`words.txt` ends with three symbols that are not words**: `#0`, `<s>`
and `</s>`, numbered after the last word, and `prepare_lang.sh` *fails* if
`<s>` or `</s>` is in the lexicon rather than silently renumbering.

**`align_lexicon.txt` is built from the position-suffixed lexicon**, each
line being `word word phones...` with the word repeated, plus a
`<eps> <silphone>` entry, then sorted and uniqued -- a byte sort in the C
locale, which is what Kaldi's `path.sh` exports.

What is *not* produced here is everything only GMM training reads:
`phones/sets.txt`, `roots.txt`, `extra_questions.txt`, `word_boundary.*`
and the `topo` file. BenNevis trains no GMM, and writing files nothing
reads would be four more things to keep correct.

Functions
---------
read_phone_groups
    `silence_phones.txt` or `nonsilence_phones.txt`, groups preserved.
build_phone_map
    A phone to its word-position-dependent versions.
build_phone_symbols
    `phones.txt`'s order.
build_word_symbols
    `words.txt`'s order.
build_align_lexicon
    `phones/align_lexicon.txt`'s lines.
prepare_lang
    All of it, into a directory.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "SILENCE_SUFFIXES",
    "NONSILENCE_SUFFIXES",
    "read_phone_groups",
    "build_phone_map",
    "build_phone_symbols",
    "build_word_symbols",
    "build_align_lexicon",
    "prepare_lang",
]

#: The versions a **silence** phone expands to, in order. The bare form is
#: included, because `prepare_lang.sh` writes the suffix list as
#: `"" "" "_B" "_E" "_I" "_S"` and the first field is the map's key -- so
#: the value keeps one unsuffixed copy.
SILENCE_SUFFIXES = ("", "_B", "_E", "_I", "_S")

#: And a **non-silence** phone's, which has no bare form: its suffix list
#: is `"" "_B" "_E" "_I" "_S"`, the first field being consumed as the key.
NONSILENCE_SUFFIXES = ("_B", "_E", "_I", "_S")

#: Written after the last word of `words.txt`, in this order. `#0` is the
#: grammar back-off symbol the language model FST needs; the other two are
#: sentence boundaries that `arpa2fst` consumes as final weights rather
#: than as arcs.
WORD_SUFFIX_SYMBOLS = ("#0", "<s>", "</s>")


def read_phone_groups(path: Path) -> List[List[str]]:
    """
    A phone list file, with its grouping preserved.

    The grouping is not cosmetic: `nonsilence_phones.txt` puts a phone and
    its stress variants on one line, `AA AA0 AA1 AA2`, and `phones.txt` is
    built by walking the lines in order and the fields within a line in
    order -- so the grouping decides the integer ids.

    Arguments
    ---------
    path: Path
        `silence_phones.txt` or `nonsilence_phones.txt`.

    Returns
    -------
    List[List[str]]
        One list per non-empty line.
    """
    groups = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields:
            groups.append(fields)
    return groups


def build_phone_map(
    groups: Sequence[Sequence[str]],
    suffixes: Sequence[str],
    position_dependent: bool,
) -> Dict[str, List[str]]:
    """
    Each phone to the versions of it that appear in `phones.txt`.

    Arguments
    ---------
    groups: Sequence[Sequence[str]]
        As `read_phone_groups` returns.
    suffixes: Sequence[str]
        `SILENCE_SUFFIXES` or `NONSILENCE_SUFFIXES`.
    position_dependent: bool
        False maps every phone to itself, which is Kaldi's identity
        `phone_map.txt` for a recipe passing
        `--position-dependent-phones false`.

    Returns
    -------
    Dict[str, List[str]]
        Phone to its versions, in suffix order.
    """
    mapping: Dict[str, List[str]] = {}
    for group in groups:
        for phone in group:
            if position_dependent:
                mapping[phone] = [f"{phone}{suffix}" for suffix in suffixes]
            else:
                mapping[phone] = [phone]
    return mapping


def _flatten(
    groups: Sequence[Sequence[str]], mapping: Dict[str, List[str]]
) -> List[str]:
    """Every version of every phone, line by line and field by field."""
    out: List[str] = []
    for group in groups:
        for phone in group:
            out.extend(mapping[phone])
    return out


def build_phone_symbols(
    silence: Sequence[Sequence[str]],
    nonsilence: Sequence[Sequence[str]],
    disambig: Sequence[str],
    position_dependent: bool,
) -> Tuple[List[str], List[str], List[str]]:
    """
    `phones.txt`'s order, and the two lists it is assembled from.

    Arguments
    ---------
    silence: Sequence[Sequence[str]]
        `silence_phones.txt`'s groups.
    nonsilence: Sequence[Sequence[str]]
        `nonsilence_phones.txt`'s groups.
    disambig: Sequence[str]
        `#0` upwards, as `phones/disambig.txt` lists them.
    position_dependent: bool
        Whether to expand the word-position suffixes.

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        The symbols in `phones.txt` order, then the flattened silence and
        non-silence lists -- which are `phones/silence.txt` and
        `phones/nonsilence.txt` and are needed for their `.int` files.
    """
    silence_map = build_phone_map(silence, SILENCE_SUFFIXES, position_dependent)
    nonsilence_map = build_phone_map(
        nonsilence, NONSILENCE_SUFFIXES, position_dependent
    )
    silence_flat = _flatten(silence, silence_map)
    nonsilence_flat = _flatten(nonsilence, nonsilence_map)
    symbols = ["<eps>"] + silence_flat + nonsilence_flat + list(disambig)
    return symbols, silence_flat, nonsilence_flat


def build_word_symbols(words: Iterable[str]) -> List[str]:
    """
    `words.txt`'s order: `<eps>`, the sorted words, then `#0 <s> </s>`.

    Arguments
    ---------
    words: Iterable[str]
        The lexicon's words, in any order and with repeats.

    Returns
    -------
    List[str]
        The symbols, index being the id.

    Raises
    ------
    ValueError
        If `<s>` or `</s>` is in the lexicon. `prepare_lang.sh` exits
        rather than renumber, because those two are written after the
        words and a lexicon entry for either would collide with the
        sentence boundary `arpa2fst` uses.
    """
    unique = sorted(set(words))
    for reserved in ("<s>", "</s>"):
        if reserved in unique:
            raise ValueError(
                f"{reserved} is in the lexicon, and words.txt writes it "
                f"after the words as a sentence boundary; the two would "
                f"collide"
            )
    return ["<eps>"] + unique + list(WORD_SUFFIX_SYMBOLS)


def build_align_lexicon(
    lexicon: Sequence[Tuple[str, Sequence[str]]],
    silence_phone: Optional[str],
) -> List[str]:
    """
    `phones/align_lexicon.txt`'s lines: `word word phones...`.

    The word is repeated because the first field is a key and the second
    is the word the pronunciation spells -- they differ only for the
    `<eps>` entry, which maps to the optional-silence phone so that a
    lattice can align silence that no word asked for.

    Sorted and uniqued as a byte sort, which is what Kaldi's `path.sh`
    exports `LC_ALL=C` for; Python's string order is codepoint order and
    agrees with it for the ASCII these files hold.

    Arguments
    ---------
    lexicon: Sequence[Tuple[str, Sequence[str]]]
        The **position-suffixed** lexicon, as `prepare_lang.sh`'s
        `tmp/lexiconp.txt` holds it -- not the dictionary's own copy.
    silence_phone: Optional[str]
        The optional silence, unsuffixed. None omits the `<eps>` entry,
        which `prepare_lang.sh` does when there is no optional silence.

    Returns
    -------
    List[str]
        The lines, without trailing newlines.
    """
    lines = [f"{word} {word} {' '.join(phones)}" for word, phones in lexicon]
    if silence_phone:
        lines.append(f"<eps> <eps> {silence_phone}")
    return sorted(set(lines))


def _write(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _table(symbols: Sequence[str]) -> List[str]:
    return [f"{symbol} {index}" for index, symbol in enumerate(symbols)]


def _compile_lexicon_fsts(out_dir: Path, compile_fsts: str) -> None:
    """
    Compile `k2/L.fst.txt` and `k2/L_disambig.fst.txt` to binary FSTs.

    Arguments
    ---------
    out_dir: Path
        The language directory, already holding the two text FSTs.
    compile_fsts: str
        `auto`, `true` or `false`; see `prepare_lang`.

    Raises
    ------
    ValueError
        On an unknown choice, or on `true` with no OpenFST -- a caller
        that asked for the binaries and silently got none would fail four
        stages later, inside `fsttablecompose`.

    Notes
    -----
    The text carries **numeric** labels and is byte-identical to
    `fstprint`'s output -- verified against the capture for both corpora
    -- so it compiles without symbol tables and gives the same binary as
    Kaldi's symbolic text compiled with them. `L_disambig.fst.txt`
    already has `fstaddselfloops`' arc folded in, which
    `build_disambig_fst_text` does, so there is no separate step for it.
    """
    if compile_fsts not in ("auto", "true", "false"):
        raise ValueError(
            f"compile_fsts must be auto, true or false, got {compile_fsts!r}"
        )
    if compile_fsts == "false":
        return

    from BenNevis.utils.openfst import available, compile_fst

    if not available():
        if compile_fsts == "true":
            raise ValueError(
                "compile_fsts is true but OpenFST is not on PATH, so L.fst "
                "and L_disambig.fst cannot be written"
            )
        logging.info(
            "OpenFST is not on PATH, so L.fst and L_disambig.fst will not be "
            "written. Nothing in BenNevis's lane reads them; Kaldi's "
            "prepare_graph.sh does, and fails saying so."
        )
        return

    for name in ("L", "L_disambig"):
        text = (out_dir / "k2" / f"{name}.fst.txt").read_text(encoding="utf-8")
        compile_fst(text, out_dir / f"{name}.fst")
    logging.info("compiled %s and %s", out_dir / "L.fst", out_dir / "L_disambig.fst")


def prepare_lang(
    dict_dir: Path,
    oov_word: str,
    out_dir: Path,
    position_dependent_phones: bool = True,
    sil_prob: float = 0.5,
    write_fsts: bool = True,
    compile_fsts: str = "auto",
) -> None:
    """
    Write a language directory from a dictionary, without Kaldi.

    Produces what the Python lane reads -- `phones.txt`, `words.txt`,
    `phones/align_lexicon.{txt,int}` -- plus the index files the capture
    records, plus `k2/L.fst.txt` and `k2/L_disambig.fst.txt` through the
    byte-exact builders in `BenNevis.lm.lexicon`.

    Deliberately **not** produced: `phones/sets.txt`, `roots.txt`,
    `extra_questions.txt`, `word_boundary.*` and `topo`, all of which only
    GMM training reads. BenNevis trains no GMM.

    Arguments
    ---------
    dict_dir: Path
        A Kaldi dictionary directory: `lexicon.txt`,
        `silence_phones.txt`, `nonsilence_phones.txt`,
        `optional_silence.txt`.
    oov_word: str
        The out-of-vocabulary word, `prepare_lang.sh`'s second positional
        argument. It must be in the lexicon.
    out_dir: Path
        Where to write the directory.
    position_dependent_phones: bool
        `prepare_lang.sh`'s default is True, and so is this one -- unlike
        the stage-3 flags, because here the name says what it does and
        there is no shell default to be conservative against.
    sil_prob: float
        The optional-silence probability, for the lexicon FSTs.
    write_fsts: bool
        Whether to write `k2/L.fst.txt` and `k2/L_disambig.fst.txt`. False
        skips the only expensive part -- 1.5 million lines each on
        mini-librispeech -- for a caller that wants the tables, which is
        what the tier 1 gate on the tables wants, since
        `tests/reference/test_corpora.py` already gates both FSTs by
        digest and tier 1 has to stay instant.
    compile_fsts: str
        Whether to also write the **binary** `L.fst` and
        `L_disambig.fst`, which is what `prepare_lang.sh` did and what
        Kaldi's `prepare_graph.sh` composes against. `auto` compiles them
        when OpenFST is on PATH and says what it skipped when it is not,
        `true` requires it, `false` never does -- the same three choices
        as `run/prepare_topo.py --compile-fsts`, and for the same reason:
        **nothing in BenNevis's own lane reads a binary FST.** Training
        reads `k2/L.fst.txt`, so a tree with no OpenFST gets a language
        directory it can train from; the decoding lane is Kaldi's and
        fails pointing here.

    Raises
    ------
    ValueError
        If a required input is missing, if `oov_word` is not in the
        lexicon, or if the lexicon uses a phone the phone lists do not
        declare -- which would otherwise produce a `phones.txt` that the
        lexicon FST cannot be built against, several steps later.
    """
    from BenNevis.lm.lexicon import (
        add_disambiguation,
        build_disambig_fst_text,
        build_lexicon_fst_text,
        position_dependent,
        read_lexicon,
    )

    dict_dir = Path(dict_dir)
    out_dir = Path(out_dir)
    for name in (
        "lexicon.txt",
        "silence_phones.txt",
        "nonsilence_phones.txt",
        "optional_silence.txt",
    ):
        if not (dict_dir / name).is_file():
            raise ValueError(f"{dict_dir / name} is missing")

    lexicon = read_lexicon(dict_dir / "lexicon.txt")
    silence = read_phone_groups(dict_dir / "silence_phones.txt")
    nonsilence = read_phone_groups(dict_dir / "nonsilence_phones.txt")
    silence_phone = (
        (dict_dir / "optional_silence.txt").read_text(encoding="utf-8").split()
    )
    if not silence_phone:
        raise ValueError(f"{dict_dir / 'optional_silence.txt'} is empty")
    silence_phone = silence_phone[0]

    # The number of disambiguation symbols is the lexicon's answer plus one
    # for silence, which is the rule `add_lex_disambig.pl` has and a
    # reimplementation drops: the empty pronunciation gets a symbol
    # reserved to it alone. Counting them here rather than guessing is what
    # fixes `phones.txt`'s tail.
    _entries, ndisambig = add_disambiguation(
        [
            (
                (word, position_dependent(tuple(phones)))
                if position_dependent_phones
                else (word, tuple(phones))
            )
            for word, phones in lexicon
        ]
    )
    disambig = [f"#{n}" for n in range(ndisambig + 2)]

    symbols, silence_flat, nonsilence_flat = build_phone_symbols(
        silence, nonsilence, disambig, position_dependent_phones
    )
    phone_ids = {symbol: index for index, symbol in enumerate(symbols)}
    word_symbols = build_word_symbols(word for word, _ in lexicon)
    word_ids = {symbol: index for index, symbol in enumerate(word_symbols)}

    if oov_word not in word_ids:
        raise ValueError(
            f"the out-of-vocabulary word {oov_word!r} is not in "
            f"{dict_dir / 'lexicon.txt'}, so oov.int would be empty and "
            f"every unknown word would map to nothing"
        )

    suffixed = [
        (
            word,
            (
                position_dependent(tuple(phones))
                if position_dependent_phones
                else tuple(phones)
            ),
        )
        for word, phones in lexicon
    ]
    unknown = sorted(
        {phone for _word, phones in suffixed for phone in phones} - set(phone_ids)
    )
    if unknown:
        raise ValueError(
            f"the lexicon uses {len(unknown)} phone(s) the phone lists do "
            f"not declare: {unknown[:8]}. phones.txt is built from the "
            f"lists, so the lexicon FST could not be built against it"
        )

    _write(out_dir / "phones.txt", _table(symbols))
    _write(out_dir / "words.txt", _table(word_symbols))
    _write(out_dir / "oov.txt", [oov_word])
    _write(out_dir / "oov.int", [str(word_ids[oov_word])])

    phones_dir = out_dir / "phones"
    _write(phones_dir / "silence.txt", silence_flat)
    _write(phones_dir / "nonsilence.txt", nonsilence_flat)
    _write(phones_dir / "disambig.txt", disambig)
    _write(phones_dir / "optional_silence.txt", [silence_phone])
    _write(phones_dir / "wdisambig.txt", ["#0"])
    for name, listing in (
        ("silence", silence_flat),
        ("nonsilence", nonsilence_flat),
        ("disambig", disambig),
        ("optional_silence", [silence_phone]),
    ):
        _write(phones_dir / f"{name}.int", [str(phone_ids[p]) for p in listing])
    _write(phones_dir / "wdisambig_phones.int", [str(phone_ids["#0"])])
    _write(phones_dir / "wdisambig_words.int", [str(word_ids["#0"])])

    align = build_align_lexicon(suffixed, silence_phone)
    _write(phones_dir / "align_lexicon.txt", align)
    _write(
        phones_dir / "align_lexicon.int",
        [
            " ".join(
                [str(word_ids[f[0]]), str(word_ids[f[1]])]
                + [str(phone_ids[p]) for p in f[2:]]
            )
            for f in (line.split() for line in align)
        ],
    )

    if not write_fsts:
        logging.info(
            "wrote %s: %d phones, %d words, %d align_lexicon entries " "(no FSTs)",
            out_dir,
            len(symbols),
            len(word_symbols),
            len(align),
        )
        return

    # The FSTs, which were already Python and are byte-exact against
    # Kaldi's. Written as text under k2/ because that is where `Lang`
    # reads the lexicon from and nothing needs them compiled.
    k2 = out_dir / "k2"
    k2.mkdir(parents=True, exist_ok=True)
    (k2 / "L.fst.txt").write_text(
        build_lexicon_fst_text(
            lexicon,
            word_ids,
            phone_ids,
            sil_prob=sil_prob,
            sil_phone=silence_phone,
            position_dependent_phones=position_dependent_phones,
        ),
        encoding="utf-8",
    )
    (k2 / "L_disambig.fst.txt").write_text(
        build_disambig_fst_text(
            lexicon,
            word_ids,
            phone_ids,
            sil_prob=sil_prob,
            sil_phone=silence_phone,
            position_dependent_phones=position_dependent_phones,
        ),
        encoding="utf-8",
    )
    _compile_lexicon_fsts(out_dir, compile_fsts)
    logging.info(
        "wrote %s: %d phones, %d words, %d align_lexicon entries",
        out_dir,
        len(symbols),
        len(word_symbols),
        len(align),
    )
