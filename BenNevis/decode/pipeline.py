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
    tokens = {
        token: index
        for index, token in enumerate(topo_build.build_tokens(loaded, phone_list))
    }
    topology_fsa = load_openfst_text(
        topo_build.build_openfst_int_text(
            loaded,
            variant,
            phone_list,
            isymbols=tokens,
            osymbols=phones,
        )
    )

    return build_decoding_graph(
        topology_fsa, lexicon_fsa, grammar, first_disambig, device=device
    )
