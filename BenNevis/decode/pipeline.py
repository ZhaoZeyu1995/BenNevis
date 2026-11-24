"""
Wiring the k2 decoder into a recipe.

The pieces -- `BenNevis.lm` for `L` and `G`, `BenNevis.topos` for `T`,
`BenNevis.decode.graph` and `.decoder` for the search,
`BenNevis.utils.scoring` for the error rate -- are each tested on their own.
This is what joins them, and it lives here rather than in `run/decode.py`
because an entry-point script is reachable from no tier but 3, which is how
the regression in #5 got in. Everything here is tested; the script is
argument parsing.

Functions
---------
build_graph
    A decoding graph from a recipe's dictionary, language model and
    topology.
format_trn
    Hypotheses in the `.trn` form `sclite` and `read_trn` both read.
write_trn
    The same, to a file.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

__all__ = [
    "read_symbol_table",
    "read_utt2spk",
    "format_trn",
    "write_trn",
    "build_graph",
]


def read_symbol_table(path: Path) -> Dict[str, int]:
    """
    An OpenFST symbol table as a mapping.

    Arguments
    ---------
    path: Path
        `words.txt` or `phones.txt`.

    Returns
    -------
    Dict[str, int]
        Symbol to integer id.
    """
    table: Dict[str, int] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2:
            table[fields[0]] = int(fields[1])
    return table


def read_symbol_table_or_list(path: Path) -> Dict[str, int]:
    """
    A symbol table that may be two columns or one.

    `prepare_topo.sh` writes `k2/tokens.txt` and `k2/phones.txt` through
    `BenNevis.topos`, which emits `symbol id` pairs, while `Lang` reads the
    same files with `read_keys` and takes the *line order* as the id. Both
    readings have to give the same answer, so a one-column file is numbered
    by position and a two-column one is trusted as written -- and a
    two-column file whose ids are not 0..n-1 in order would mean the two
    disagree, which is worth an error rather than a silent difference.

    Arguments
    ---------
    path: Path
        The table.

    Returns
    -------
    Dict[str, int]
        Symbol to integer id.

    Raises
    ------
    ValueError
        If the ids are not consecutive from zero in file order, since
        `Lang` assumes they are.
    """
    table: Dict[str, int] = {}
    for position, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines()
    ):
        fields = line.split()
        if not fields:
            continue
        if len(fields) == 1:
            table[fields[0]] = len(table)
            continue
        index = int(fields[1])
        if index != position:
            raise ValueError(
                f"{path}:{position + 1}: {fields[0]!r} has id {index} but is "
                f"on line {position + 1}; Lang reads this file by line order, "
                f"so the two would disagree about the network's outputs"
            )
        table[fields[0]] = index
    return table


def read_utt2spk(path: Path) -> Dict[str, str]:
    """
    Kaldi's `utt2spk`.

    Arguments
    ---------
    path: Path
        The file.

    Returns
    -------
    Dict[str, str]
        Utterance id to speaker id.
    """
    mapping: Dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            mapping[fields[0]] = fields[1]
    return mapping


def format_trn(
    hypotheses: Iterable[Tuple[str, Sequence[str]]],
    utt2spk: Dict[str, str],
) -> str:
    """
    Hypotheses as `.trn`: the words, then the id in brackets.

    The form is `WORDS (speaker-utterance)`, which is what
    `decode_faster.sh`'s `awk` produced and what `read_trn` and `sclite`
    both read. An utterance with no speaker is written with its own id in
    place of one rather than skipped: dropping it would make the hypothesis
    shorter than the reference, and `score` would then refuse the whole
    comparison rather than report one odd utterance.

    An **empty** hypothesis keeps its line, with nothing before the
    bracket. That is the case `decode_faster.sh` got wrong -- its `awk`
    stripped the id by searching for the first field's text, which returned
    the whole line when the decoder emitted nothing, so a file name was
    scored as a transcription.

    Arguments
    ---------
    hypotheses: Iterable[Tuple[str, Sequence[str]]]
        Utterance id and its decoded words.
    utt2spk: Dict[str, str]
        Utterance id to speaker id.

    Returns
    -------
    str
        The `.trn` text, one line per utterance.
    """
    lines: List[str] = []
    for uttid, words in hypotheses:
        speaker = utt2spk.get(uttid, uttid)
        lines.append(f"{' '.join(words)} ({speaker}-{uttid})")
    return "\n".join(lines) + "\n" if lines else ""


def write_trn(
    hypotheses: Iterable[Tuple[str, Sequence[str]]],
    utt2spk: Dict[str, str],
    path: Path,
) -> None:
    """
    Write hypotheses as `.trn`.

    Arguments
    ---------
    hypotheses: Iterable[Tuple[str, Sequence[str]]]
        Utterance id and its decoded words.
    utt2spk: Dict[str, str]
        Utterance id to speaker id.
    path: Path
        Where to write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_trn(hypotheses, utt2spk), encoding="utf-8")


def build_graph(
    lang_dir: Path,
    lexicon: Path,
    arpa: Path,
    topology: str,
    variant: str = "k2",
    sil_prob: float = 0.0,
    position_dependent_phones: bool = False,
    device=None,
):
    """
    A decoding graph from a recipe's own inputs.

    Takes the dictionary and the ARPA model rather than the language
    directory's compiled FSTs, because reading those needs OpenFST and
    building them does not -- `BenNevis.lm` reproduces both byte for byte.
    The symbol tables *are* read from the language directory, for the same
    reason `Lang` reads them rather than recomputing: those integer ids are
    what every other artefact is written against.

    Arguments
    ---------
    lang_dir: Path
        The language directory, for `words.txt` and `phones.txt`.
    lexicon: Path
        `lexicon.txt` from the recipe's dictionary.
    arpa: Path
        The ARPA language model, optionally gzipped.
    topology: str
        Which topology, e.g. `ctc`.
    variant: str
        Which variant of it. `k2` is the epsilon-free one the network's
        output layer requires.
    sil_prob: float
        The optional-silence probability `prepare_lang.sh` was given.
    position_dependent_phones: bool
        Whether it was given position-dependent phones.
    device
        Where to put the graph. k2 has no MPS backend, so this is left to
        the caller.

    Returns
    -------
    k2.Fsa
        The decoding graph, arc-sorted.
    """
    from BenNevis.decode.graph import build_decoding_graph, load_openfst_text
    from BenNevis.lm import build_fst_text, read_arpa
    from BenNevis.lm.lexicon import build_disambig_fst_text, read_lexicon
    from BenNevis.topos import build as topo_build
    from BenNevis.topos import spec as topo_spec

    lang_dir = Path(lang_dir)
    words = read_symbol_table(lang_dir / "words.txt")
    phones = read_symbol_table(lang_dir / "phones.txt")

    # Checked before any work: a language directory with no
    # disambiguation symbols cannot give a determinisable composition, and
    # discovering that after building a 206,510-entry lexicon wastes the
    # time and reports the wrong error.
    disambig = [value for key, value in phones.items() if key.startswith("#")]
    if not disambig:
        raise ValueError(
            f"{lang_dir}/phones.txt has no disambiguation symbols, so the "
            f"composition cannot be made determinisable"
        )
    first_disambig = min(disambig)

    # `compose_lg` strips the disambiguation symbols with a *threshold*,
    # `labels >= first_disambig`, where Kaldi's `fstrmsymbols` takes the
    # explicit list in `disambig.int`. The two agree only while the `#`
    # symbols occupy a contiguous suffix of `phones.txt` -- true for both
    # corpora here, yesno at 8..9 of 10 and mini-librispeech at 347..363
    # of 364 -- and if an ordinary phone ever sorted above them the
    # threshold would map that phone to epsilon. The graph would still
    # compose and the decoder would still run, so this is the "wrong
    # rather than broken" failure the stage-10 settings note warns about,
    # and nothing downstream would report it. Cheaper to assert than to
    # diagnose.
    intruders = sorted(
        key
        for key, value in phones.items()
        if value >= first_disambig and not key.startswith("#")
    )
    if intruders:
        raise ValueError(
            f"{lang_dir}/phones.txt has {len(intruders)} symbol(s) numbered "
            f"at or above the first disambiguation symbol "
            f"#{first_disambig}: {intruders[:8]}. Stripping the "
            f"disambiguation symbols by threshold would map these to "
            f"epsilon and silently corrupt the graph"
        )

    logging.info("building L_disambig from %s", lexicon)
    lexicon_fsa = load_openfst_text(
        build_disambig_fst_text(
            read_lexicon(lexicon),
            words,
            phones,
            sil_prob=sil_prob,
            position_dependent_phones=position_dependent_phones,
        )
    )

    logging.info("building G from %s", arpa)
    grammar = load_openfst_text(build_fst_text(read_arpa(arpa), words))

    logging.info("expanding the %s topology", topology)
    loaded = topo_spec.load(topology)
    phone_list = topo_build.read_phone_table(lang_dir / "phones.txt")

    # **Read, never recompute.** `k2/tokens.txt` fixes the network's output
    # indices and a checkpoint is only valid while they stay put -- the rule
    # is in CLAUDE.md and breaking it is what made the first real decoding
    # run fail. `prepare_topo.sh` writes that table with neither `<eps>` nor
    # the disambiguation symbols, so recomputing it with the defaults gave
    # ten labels where the network emits seven, every one shifted by one:
    # the decoder read `<blk>` as `<eps>` and `Y` as `<UNK>`.
    tokens_path = lang_dir / "k2" / "tokens.txt"
    if tokens_path.is_file():
        tokens = read_symbol_table_or_list(tokens_path)
    else:
        logging.warning(
            "%s has no k2/tokens.txt; deriving the table instead, which is "
            "only safe if the network was built from the same spec",
            lang_dir,
        )
        tokens = {
            token: index
            for index, token in enumerate(
                topo_build.build_tokens(
                    loaded, phone_list, with_eps=False, with_disambig=False
                )
            )
        }

    # Likewise the output side: the k2 variant maps tokens to phones without
    # the disambiguation symbols, because `compose_lg` has already stripped
    # them from LG by this point.
    k2_phones_path = lang_dir / "k2" / "phones.txt"
    if k2_phones_path.is_file():
        osymbols = read_symbol_table_or_list(k2_phones_path)
        out_phone_list = topo_build.read_phone_table(k2_phones_path)
    else:
        osymbols = phones
        out_phone_list = phone_list

    topology_fsa = load_openfst_text(
        topo_build.build_openfst_int_text(
            loaded,
            variant,
            out_phone_list,
            isymbols=tokens,
            osymbols=osymbols,
        )
    )
    logging.info(
        "topology: %d tokens on the input side, %d arcs",
        len(tokens),
        topology_fsa.num_arcs,
    )

    return build_decoding_graph(
        topology_fsa, lexicon_fsa, grammar, first_disambig, device=device
    )
