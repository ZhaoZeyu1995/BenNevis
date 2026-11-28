"""
Capture what Kaldi produced from a completed recipe run.

Run inside the Kaldi image, against a recipe that has finished, this writes
`tests/golden/reference/<corpus>/` and the manifest describing it. That
directory is then committed, and the Kaldi-removal work compares against the
committed bytes rather than against a Kaldi it has to keep working.

Why capture rather than compare live: recent Kaldi cannot fetch the OpenFST
version its own source needs, and the OpenBLAS it pins no longer compiles on
current hardware. The image that still works is pinned by digest, and its
output is worth recording while it can still be produced. A capture also
moves the comparison into the cheap test tiers, which is where the migration
work actually happens.

What is captured, and why each one:

- The symbol tables and lexicon FSTs from `prepare_lang.sh`. These are the
  part of the pipeline BenNevis does not yet build, and the part a Python
  lexicon has to reproduce. As text, compared as bytes.
- `G.fst` and `TLG.fst` as text, for the same reason.
- CMVN statistics in full: two rows per speaker, so there is no reason not to.
- Feature matrices from **both sides of CMVN**: `feats.scp` as the feature
  extractor wrote it, and `feats.cmvn.scp` after normalisation. The pair is
  what lets a Python replacement for either stage be driven with the real
  input and compared against the real output.
- A few matrices in full and per-utterance statistics for the rest. Every
  matrix would be 12MB of float32 per side for yesno and buy nothing over a
  handful.
- The decoded hypotheses and the reference transcriptions, per topology.
- The feature configuration, `conf/fbank.conf` and `conf/pitch.conf`. These
  decide what the features are, so a capture that does not record them cannot
  be distinguished from one taken under different settings.

Usage
-----
    python3 tests/reference/capture.py egs/yesno/asr tests/golden/reference

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from BenNevis.utils.fingerprint import (  # noqa: E402
    fingerprint,
    fingerprint_phone_sequences,
    position_dependent_table,
    sample_lexicon_sequences,
    sample_word_sequences,
)
from BenNevis.utils.reference import (  # noqa: E402
    FEATURE_TOLERANCE,
    MANIFEST_NAME,
    Manifest,
    array_stats,
    sha256_bytes,
)

#: How many feature matrices to keep whole. Enough for a genuine elementwise
#: comparison across more than one utterance; the rest are kept as statistics.
FULL_FEATURE_UTTERANCES = 5

#: The feature configuration, copied verbatim. These files decide what the
#: captured features are, so a reference that does not record them cannot be
#: told apart from one taken under different settings -- which matters most
#: for `--dither`, where a change silently makes every captured feature
#: unreproducible.
#: `conf/pitch.conf` is listed although no recipe uses it any more: pitch
#: was dropped, and a capture taken from an older checkout still holds
#: one, so the list has to be able to record it. A missing file is noted
#: on stderr rather than being an error, which is what makes that work.
RECIPE_CONF = ("conf/fbank.conf", "conf/pitch.conf")

#: Text artefacts copied verbatim from the language directory. These are what
#: a Python replacement for `prepare_lang.sh` has to reproduce.
LANG_TEXT = (
    "phones.txt",
    "words.txt",
    "oov.txt",
    "oov.int",
    "phones/align_lexicon.int",
    "phones/disambig.int",
    # The word-disambiguation symbols `fstaddselfloops` is given when
    # L_disambig.fst is built -- the phone side and the word side of the
    # `#0` pass-through that lets G's back-off arcs traverse L. Two lines
    # each, and without them that stage has no recorded inputs at all.
    "phones/wdisambig_phones.int",
    "phones/wdisambig_words.int",
    "phones/silence.int",
    "phones/optional_silence.int",
)

#: The *inputs* `prepare_lang.sh` reads, from `data/local/dict`. The rule
#: the capture already follows for CMVN and for the symbol tables: whatever
#: stage is being replaced needs its input captured as well as its output,
#: because otherwise there is nothing to drive the replacement with.
#:
#: Learned by writing a Python `L.fst` and finding that mini-librispeech's
#: lexicon was not in the reference, so the only thing the build could be
#: checked against was yesno's four-word dictionary.
#: `prepare_lang.sh`'s working directory, which holds the one intermediate
#: a replacement needs: the lexicon after `add_lex_disambig.pl` has added
#: disambiguation symbols. L_disambig.fst is four stages -- disambiguate the
#: lexicon, build the FST, add the pass-through self-loops, arc-sort -- and
#: only its end result has a digest. A mismatch would say that something
#: among four stages is wrong and nothing about which, on a 34MB artefact
#: whose bytes are not committed. This is the intermediate that splits that
#: in two.
#:
#: Captured proactively rather than after being caught out, which the
#: dictionary, the symbol tables, CMVN's inputs and `utt2spk` all were.
TMP_TEXT = (
    "lexiconp_disambig.txt",
    "lexiconp.txt",
    "lex_ndisambig",
)

DICT_TEXT = (
    "lexicon.txt",
    "lexiconp.txt",
    "silence_phones.txt",
    "nonsilence_phones.txt",
    "optional_silence.txt",
    "extra_questions.txt",
)

#: Binary FSTs, converted to text with fstprint so that a diff is readable
#: and a replacement can be compared without OpenFST.
LANG_FSTS = ("L.fst", "L_disambig.fst")

#: How many lexicon-drawn sequences a phone-input graph's fingerprint
#: holds. Fewer than the 64 word sequences `G` gets: the search runs over
#: a graph around three times `G`'s size, and this is a gate rather than a
#: survey of the language.
LEXICON_FINGERPRINT_COUNT = 24
GRAPH_FSTS = ("G.fst", "TLG.fst", "LG.fst")

#: Text files in a per-topology graph directory, captured because they are
#: the symbol tables the graphs' labels are written against -- and because
#: **the two token tables are not interchangeable.** `prepare_topo.sh`
#: writes `tokens.txt` with `<eps>` for Kaldi's `T.fst`, and
#: `k2/tokens.txt` with neither `<eps>` nor the disambiguation symbols for
#: ours, because a network cannot emit `<eps>`. So the same token has a
#: different id on each side -- measured on mini-librispeech, `<blk>` is 1
#: in Kaldi's table and 0 in ours, `SIL` 2 and 1 -- and any comparison of
#: `TLG` has to go through token *names*. Comparing raw ids would line
#: `<blk>` up against `SIL`, which is the shift that broke the first real
#: k2 decoding run.
#:
#: Capturing them also removes a reconstruction: measuring our own TLG
#: meant rebuilding both tables by hand, which is a second implementation
#: of something the recipe already writes.
GRAPH_TEXT = ("tokens.txt", "k2/tokens.txt", "k2/phones.txt")


#: Where each Kaldi tool lives inside the tree, relative to KALDI_ROOT.
TOOL_LOCATIONS = {
    "fstprint": "tools/openfst/bin",
    "copy-feats": "src/featbin",
}


@contextlib.contextmanager
def working_directory(path: Path):
    """
    Run a block with the working directory set to `path`.

    `contextlib.chdir` would do, and is Python 3.11, which the image has
    -- written out because the other tiers are 3.12 and this file is
    imported by their tests. Restored on the way out however the block
    ends, since a capture that left the process somewhere else would
    write the rest of itself into the recipe.

    Arguments
    ---------
    path: Path
        The directory to run in.
    """
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def find_kaldi_tool(name: str) -> Optional[str]:
    """
    Locate a Kaldi tool, on PATH or inside the Kaldi tree.

    The image has them, but only on the PATH that a recipe's `path.sh`
    builds. A capture invoked directly -- as the nightly workflow does --
    has no such PATH, which is how the first captures came out without a
    single FST in them: `shutil.which` found nothing and the code quietly
    carried on. Looking under `KALDI_ROOT` as well makes the capture
    independent of how it was invoked.

    Arguments
    ---------
    name: str
        The tool, which must be one of TOOL_LOCATIONS.

    Returns
    -------
    Optional[str]
        Its path, or None if it is nowhere to be found.
    """
    found = shutil.which(name)
    if found:
        return found

    kaldi_root = os.environ.get("KALDI_ROOT", "/opt/kaldi")
    candidate = Path(kaldi_root) / TOOL_LOCATIONS[name] / name
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def find_fstprint() -> Optional[str]:
    """Locate `fstprint`; see `find_kaldi_tool`."""
    return find_kaldi_tool("fstprint")


def fstprint(binary: str, path: Path) -> Optional[bytes]:
    """
    Print a binary FST as text.

    Arguments
    ---------
    binary: str
        The fstprint executable, as `find_fstprint` located it.
    path: Path
        The compiled FST.

    Returns
    -------
    Optional[bytes]
        Its text form, or None if the call failed. A failure is reported on
        stderr rather than raised, so that one unreadable FST does not lose
        the rest of the capture -- but it is never silent.
    """
    try:
        completed = subprocess.run(
            [binary, str(path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        sys.stderr.write(f"capture: fstprint {path} failed: {error.stderr!r}\n")
        return None
    return completed.stdout


def read_archive(scp: Path, copy_feats: Optional[str] = None) -> Dict[str, "object"]:
    """
    Read a Kaldi feature archive referenced by an scp file.

    With `copy_feats`, the archive is first rewritten uncompressed by Kaldi
    itself and that copy is read instead. This is not a convenience: it is
    what makes a bit-exact comparison possible at all.

    `make_fbank_pitch.sh` writes compressed archives, and Kaldi's
    `CompressedMatrix` quantises each column to 8 bits. Reading such an
    archive with `kaldiio` gives *its* decompression, not Kaldi's, and the
    two differ in the last few bits -- which is why the Python CMVN agreed
    with `apply-cmvn` to 4e-6 rather than exactly, even with Kaldi's
    algorithm reproduced operation for operation. Letting Kaldi decompress,
    and reading the uncompressed result losslessly, removes the round trip
    from the comparison.

    Arguments
    ---------
    scp: Path
        An scp mapping utterance ids to archive offsets.
    copy_feats: Optional[str]
        The `copy-feats` executable. None reads the archive directly, which
        is lossless for an uncompressed one.

    Returns
    -------
    Dict[str, numpy.ndarray]
        Utterance id to feature matrix, in scp order.
    """
    import kaldiio

    if copy_feats is None:
        return {key: value for key, value in kaldiio.load_scp_sequential(str(scp))}

    with tempfile.TemporaryDirectory() as workspace:
        plain = Path(workspace) / "plain.ark"
        subprocess.run(
            [
                copy_feats,
                "--compress=false",
                f"scp:{scp}",
                f"ark:{plain}",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {key: value for key, value in kaldiio.load_ark(str(plain))}


#: Artefacts whose *lines* are reproducible but whose line *order* is not,
#: normalised by sorting before they are stored and digested.
#:
#: `extra_questions.txt` is the one. Two captures from the same commit in
#: the same image produced the same five lines in a different order --
#: Kaldi's `prepare_dict.sh` builds it by iterating a Perl hash, and Perl
#: has randomised hash order per process since 5.18. So its bytes cannot be
#: compared and its digest means nothing unless the order is pinned.
#:
#: Sorting here rather than at comparison time, so that the stored bytes and
#: the recorded digest agree and the integrity check still works. This is a
#: normalisation of how the reference is *read*, the same kind of thing as
#: having Kaldi rewrite the feature archives uncompressed -- not an edit to
#: make a test pass, and the recipe is untouched.
#:
#: Nothing yet depends on this file: it feeds the decision-tree questions in
#: GMM training, not the lexicon FSTs. It is fixed now because the failure
#: it would cause is a test that fails on a *re-capture* rather than on a
#: change, which is the hardest kind to read.
UNORDERED_TEXT = ("extra_questions.txt",)


def normalise_unordered(relative: str, payload: bytes) -> bytes:
    """
    Sort the lines of an artefact whose order is not reproducible.

    Arguments
    ---------
    relative: str
        Its path within the capture directory.
    payload: bytes
        The content as captured.

    Returns
    -------
    bytes
        The content with its lines sorted, for an artefact named in
        UNORDERED_TEXT; unchanged otherwise.
    """
    if PurePosixPath(relative).name not in UNORDERED_TEXT:
        return payload
    text = payload.decode("utf-8", errors="replace")
    lines = sorted(line for line in text.splitlines())
    if not lines:
        return payload
    return ("\n".join(lines) + "\n").encode("utf-8")


#: Above this, an exact artefact's digest is recorded but its bytes are not
#: stored. An exact artefact is compared *by digest*, so the content is not
#: needed to tell whether a replacement reproduces it -- only to see how it
#: differs when it does not. mini-librispeech's TLG.fst is 334MB as text,
#: against yesno's 20KB, because it carries a 200k-word lexicon and a real
#: 3-gram model; storing it would put 600MB of generated data in the
#: repository to answer a question a 64-character digest answers.
#:
#: The cost is that a mismatch cannot be diffed from the committed reference.
#: The uploaded workflow artefact holds the bytes for ninety days.
MAX_EXACT_BYTES = 1 << 20

#: Artefacts stored whole however large they are, because they are *inputs*
#: to a replacement rather than outputs of one. A symbol table fixes the
#: integer labels every FST in the directory is written against, so a
#: Python `arpa2fst` cannot be driven at all without it, let alone compared
#: -- and "any stage being replaced needs its input captured as well as its
#: output" is the rule the capture already follows for CMVN's two sides.
#: mini-librispeech's `words.txt` is 3.0MB, which is above the threshold and
#: nothing like the 334MB the threshold exists for.
#:
#: Matched on the path's suffix, so `lang/words.txt` and
#: `lang_nosp/words.txt` both qualify.
#: `lexicon.txt` is here for the same reason as the symbol tables: it is the
#: input a Python `make_lexicon_fst` is driven by, and mini-librispeech's is
#: several megabytes because the dictionary has 200k words.
ALWAYS_STORED = (
    "words.txt",
    "phones.txt",
    "tokens.txt",
    "lexicon.txt",
    "lexiconp.txt",
    "lexiconp_disambig.txt",
)


def is_input_artefact(relative: str) -> bool:
    """
    Whether an artefact is stored whole regardless of its size.

    Arguments
    ---------
    relative: str
        Its path within the capture directory.

    Returns
    -------
    bool
        True for the symbol tables in `ALWAYS_STORED`.
    """
    return PurePosixPath(relative).name in ALWAYS_STORED


def store(
    manifest: Manifest,
    destination: Path,
    relative: str,
    payload: bytes,
    max_bytes: int = MAX_EXACT_BYTES,
) -> None:
    """
    Record one exact artefact's digest, and its bytes if they are small.

    Arguments
    ---------
    manifest: Manifest
        The manifest to record it in.
    destination: Path
        The capture directory.
    relative: str
        Its path within that directory, which is also its manifest key.
    payload: bytes
        The content. An artefact named in UNORDERED_TEXT has its lines
        sorted first; see `normalise_unordered`.
    max_bytes: int
        Above this the digest is recorded without the bytes, unless the
        artefact is one `is_input_artefact` keeps whole. See
        MAX_EXACT_BYTES and ALWAYS_STORED.
    """
    payload = normalise_unordered(relative, payload)
    manifest.exact[relative] = sha256_bytes(payload)
    if len(payload) > max_bytes and not is_input_artefact(relative):
        manifest.large[relative] = len(payload)
        return
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def parse_fst_text(text: bytes) -> Tuple[dict, dict, Optional[int]]:
    """
    An fstprint dump as arcs by source, final weights, and the start state.

    OpenFST's text format takes the start state from the **first line's
    source**, which is not necessarily 0: Kaldi's `G.fst` for
    mini-librispeech begins at state 2, and reading it as though it began
    at 0 adds that state's back-off weight to every sentence.

    Arguments
    ---------
    text: bytes
        The dump.

    Returns
    -------
    Tuple[dict, dict, Optional[int]]
        Arcs by source as (destination, input label, weight), final weights
        by state, and the start state or None for an empty FST.
    """
    arcs: dict = {}
    finals: dict = {}
    start: Optional[int] = None
    for line in text.decode("utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 4:
            source = int(fields[0])
            if start is None:
                start = source
            arcs.setdefault(source, []).append(
                (
                    int(fields[1]),
                    int(fields[2]),
                    float(fields[4]) if len(fields) > 4 else 0.0,
                )
            )
        elif fields:
            finals[int(fields[0])] = float(fields[1]) if len(fields) > 1 else 0.0
    return arcs, finals, start


def record_shape(manifest: Manifest, relative: str, text: bytes) -> None:
    """
    Record how large an FST is, in states, arcs and final states.

    A `large` artefact keeps its digest and not its bytes, so its *size*
    was knowable only as a byte count and anything about its structure had
    to be inferred from it. That is how `docs/kaldi-graph-chain.md` came
    to say Kaldi's `LG.fst` holds "about 5.8M arcs": 138,155,979 bytes
    divided by a guess at bytes per line. The number matters, because it
    is what decides whether the `fstdeterminizestar`/`fstminimizeencoded`
    pair we skip is compacting the graph or not, and a replacement cannot
    be judged against an estimate.

    Arguments
    ---------
    manifest: Manifest
        The manifest to record it in.
    relative: str
        The FST's path within the capture, which is its manifest key.
    text: bytes
        The fstprint dump.
    """
    arcs, finals, _ = parse_fst_text(text)
    # States are counted as every state that appears as a source or a
    # destination, not as the highest index seen: an FST text need not
    # mention every state as a source, and a final state with no outgoing
    # arcs appears only as a destination.
    states = set(arcs) | set(finals)
    for outgoing in arcs.values():
        states.update(destination for destination, _, _ in outgoing)
    manifest.shapes[relative] = {
        "states": len(states),
        "arcs": sum(len(outgoing) for outgoing in arcs.values()),
        "finals": len(finals),
    }


def record_fingerprint(
    manifest: Manifest,
    relative: str,
    text: bytes,
    words: Dict[str, int],
) -> None:
    """
    Record what weights an FST gives a fixed set of word sequences.

    The standard for an FST is functional equivalence, but a large one is
    kept as a digest alone and a digest answers only the byte question --
    and `G.fst` is already known not to be byte-equal to a Python build, so
    its digest can never match and says nothing. This is what makes the
    right comparison possible without storing 61MB.

    Arguments
    ---------
    manifest: Manifest
        The manifest to record into.
    relative: str
        The FST's path within the capture directory.
    text: bytes
        Its fstprint dump.
    words: Dict[str, int]
        The word symbol table its output labels index.
    """
    arcs, finals, start = parse_fst_text(text)
    if start is None or not words:
        return
    sequences = sample_word_sequences(words.keys())
    recorded = fingerprint(arcs, finals, words, start, sequences)
    if recorded:
        manifest.functional[relative] = recorded


def record_lexicon_fingerprint(
    manifest: Manifest,
    relative: str,
    text: bytes,
    phones: Dict[str, int],
    lexicon: Path,
    count: int = LEXICON_FINGERPRINT_COUNT,
) -> None:
    """
    Record what a phone-input FST gives sequences its lexicon permits.

    `record_fingerprint` samples from `words.txt`, which works for `G`
    because a back-off LM accepts any sequence of in-vocabulary words. A
    phone-input graph does not: it accepts only what the lexicon
    pronounces, so a randomly sampled word sequence is rejected almost
    surely and a fingerprint of rejections compares nothing.

    This is the gate mini-librispeech has been missing. Its `LG.fst.txt`
    is 138,155,979 bytes, so the capture keeps the digest and not the
    bytes -- and the digest can never match a Python build whose state
    numbering differs. The yesno equivalence test reads Kaldi's committed
    `LG.fst.txt` directly, which works only at 20KB.

    Position dependence is read off the phone table rather than
    configured, because the two recipes differ and a capture should not
    have to be told which one it is of.

    Arguments
    ---------
    manifest: Manifest
        The manifest to record into.
    relative: str
        The FST's path within the capture directory.
    text: bytes
        Its fstprint dump.
    phones: Dict[str, int]
        The phone symbol table its input labels index.
    lexicon: Path
        `prepare_lang.sh`'s input dictionary, as captured under `dict/`.
    count: int
        How many sequences to draw. Lower than the word-sequence default,
        because the shortest-distance search runs over a graph some three
        times `G`'s size and the fingerprint is a gate rather than a
        sample of the whole language.
    """
    from BenNevis.lm.lexicon import read_lexicon

    if not phones or not lexicon.is_file():
        return
    arcs, finals, start = parse_fst_text(text)
    if start is None:
        return
    drawn = sample_lexicon_sequences(
        read_lexicon(lexicon),
        count=count,
        position_dependent_phones=position_dependent_table(phones.keys()),
    )
    recorded = fingerprint_phone_sequences(arcs, finals, phones, start, drawn)
    if recorded:
        manifest.functional[relative] = recorded


def capture_language(
    manifest: Manifest,
    recipe: Path,
    destination: Path,
    topologies: List[str],
    printer: Optional[str],
    lang_dir: str = "data/lang",
    graph_prefix: str = "data/lang_test_tg_",
    dict_dir: str = "data/local/dict",
    tmp_dir: str = "data/local/lang",
) -> None:
    """
    Capture the language directories: the dictionary, symbol tables and FSTs.

    Arguments
    ---------
    manifest: Manifest
        The manifest to record into.
    recipe: Path
        The recipe directory.
    destination: Path
        The capture directory.
    topologies: List[str]
        The topologies the recipe built.
    printer: Optional[str]
        The fstprint executable, or None to skip the compiled FSTs.
    lang_dir: str
        The language directory `prepare_lang.sh` produced, relative to the
        recipe. yesno's is `data/lang`; mini_librispeech's is
        `data/lang_nosp`, Kaldi's own name for a lexicon without
        position-dependent silence modelling.
    graph_prefix: str
        What a per-topology graph directory is called, minus the topology.
        yesno's are `data/lang_test_tg_<topo>`, mini_librispeech's
        `data/lang_test_tgsmall_<topo>` after the small 3-gram model Kaldi's
        format_lms.sh builds.
    dict_dir: str
        The dictionary `prepare_lang.sh` reads, relative to the recipe. Its
        contents are the *input* to the stage, so a replacement has
        something to run on.
    tmp_dir: str
        `prepare_lang.sh`'s working directory, which holds the
        disambiguated lexicon. yesno's is `data/local/lang`;
        mini_librispeech's is `data/local/lang_tmp_nosp`.
    """
    tmp_base = recipe / tmp_dir
    for name in TMP_TEXT:
        source = tmp_base / name
        if source.is_file():
            store(manifest, destination, f"tmp/{name}", source.read_bytes())

    dict_base = recipe / dict_dir
    for name in DICT_TEXT:
        source = dict_base / name
        if source.is_file():
            store(manifest, destination, f"dict/{name}", source.read_bytes())

    base = recipe / lang_dir
    for name in LANG_TEXT:
        source = base / name
        if source.is_file():
            store(manifest, destination, f"lang/{name}", source.read_bytes())

    if printer is None:
        return

    for name in LANG_FSTS:
        source = base / name
        if not source.is_file():
            continue
        text = fstprint(printer, source)
        if text is not None:
            relative = f"lang/{name}.txt"
            store(manifest, destination, relative, text)
            record_shape(manifest, relative, text)

    # The word symbol table the graph FSTs' output labels index. Read from
    # the language directory rather than recomputed, for the same reason
    # training reads it: these indices are what everything else is written
    # against.
    words: Dict[str, int] = {}
    words_path = base / "words.txt"
    if words_path.is_file():
        for line in words_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                words[parts[0]] = int(parts[1])

    # The phone table LG's *input* labels index, and the dictionary the
    # lexicon-drawn sequences come from. Both read from the directories
    # Kaldi wrote, for the same reason as `words` above: these indices are
    # what the FSTs were written against.
    phones: Dict[str, int] = {}
    phones_path = base / "phones.txt"
    if phones_path.is_file():
        for line in phones_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                phones[parts[0]] = int(parts[1])
    dictionary = dict_base / "lexicon.txt"

    for topology in topologies:
        graph = recipe / f"{graph_prefix}{topology}"
        for name in GRAPH_TEXT:
            source = graph / name
            if source.is_file():
                store(
                    manifest,
                    destination,
                    f"graph/{topology}/{name}",
                    source.read_bytes(),
                )
        for name in GRAPH_FSTS:
            source = graph / name
            if not source.is_file():
                continue
            text = fstprint(printer, source)
            if text is not None:
                relative = f"graph/{topology}/{name}.txt"
                store(manifest, destination, relative, text)
                record_shape(manifest, relative, text)
                if name == "G.fst":
                    # G accepts any sequence of in-vocabulary words,
                    # because its back-off arcs reach the unigram state.
                    record_fingerprint(manifest, relative, text, words)
                elif name == "LG.fst":
                    # LG's input side is phones, so it accepts only what
                    # the lexicon pronounces -- hence sequences drawn from
                    # the lexicon rather than from words.txt. This is the
                    # only functional gate the graph has on a corpus whose
                    # bytes are too large to commit.
                    record_lexicon_fingerprint(
                        manifest, relative, text, phones, dictionary
                    )


def capture_features(
    manifest: Manifest,
    recipe: Path,
    destination: Path,
    datasets: List[str],
    copy_feats: Optional[str] = None,
) -> None:
    """
    Capture feature matrices and CMVN statistics.

    A few matrices whole, the rest as statistics, and all of the CMVN
    statistics, which are two rows per speaker.

    Arguments
    ---------
    manifest: Manifest
        The manifest to record into.
    recipe: Path
        The recipe directory.
    destination: Path
        The capture directory.
    datasets: List[str]
        The data directories to capture.
    copy_feats: Optional[str]
        The `copy-feats` executable, used to have Kaldi decompress its own
        archives; see `read_archive`. None reads them with kaldiio, which is
        lossless only for an uncompressed archive.
    """
    import numpy as np

    for dataset in datasets:
        data = recipe / "data" / dataset
        # Both sides of CMVN. `feats.scp` is what compute-cmvn-stats reads
        # and apply-cmvn transforms; `feats.cmvn.scp` is what it wrote and
        # what training consumes. The first captures held only the second,
        # which leaves a replacement for either stage with no input to be
        # driven with and nothing to compare against -- so the pair is the
        # point, and the same utterances are kept on both sides.
        # Collected explicitly rather than read off the loop variable
        # below: with neither archive present that loop `continue`s and
        # leaves nothing bound, which is how the first version of the
        # waveform capture crashed on a data directory with no features.
        whole_keys: List[str] = []
        for kind, name in (("features", "feats.cmvn.scp"), ("raw", "feats.scp")):
            scp = data / name
            if not scp.is_file():
                sys.stderr.write(f"capture: no {scp}, skipping {kind}\n")
                continue

            features = read_archive(scp, copy_feats=copy_feats)
            keys = sorted(features)

            whole = keys[:FULL_FEATURE_UTTERANCES]
            if whole and not whole_keys:
                whole_keys = list(whole)
            if whole:
                arrays = {key: np.asarray(features[key]) for key in whole}
                relative = f"{kind}/{dataset}.npz"
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target, **arrays)
                for key in whole:
                    manifest.arrays[f"{kind}/{dataset}/{key}"] = {
                        "path": relative,
                        "key": key,
                        "shape": list(arrays[key].shape),
                        "dtype": str(arrays[key].dtype),
                        "tolerance": {
                            "rtol": FEATURE_TOLERANCE.rtol,
                            "atol": FEATURE_TOLERANCE.atol,
                        },
                    }

            for key in keys:
                manifest.stats[f"{kind}/{dataset}/{key}"] = array_stats(features[key])

        # The **waveforms** for the utterances whose features are kept
        # whole. Feature extraction is the one stage whose input was never
        # captured, and the cost of that was not theoretical: the "1.2%
        # linear-energy residual" between torchaudio and Kaldi was chased
        # through six refuted hypotheses before anyone could compute a
        # filterbank from the audio and find that the *reference* is
        # quantised to 8 bits by `CompressedMatrix`. With the audio here,
        # that comparison is a tier 1b test instead of a research project.
        #
        # Stored as the samples rather than as a file, so the capture does
        # not depend on a corpus layout or on an audio decoder, and so the
        # digest is of the numbers a feature extractor actually receives.
        capture_waveforms(manifest, recipe, data, destination, dataset, whole_keys)

        cmvn = data / "cmvn.scp"
        if cmvn.is_file():
            # Statistics are double and never compressed, so they are read
            # directly: passing them through copy-feats would narrow them to
            # float and lose the precision the whole normaliser rests on.
            stats = read_archive(cmvn)
            arrays = {key: np.asarray(value) for key, value in stats.items()}
            relative = f"cmvn/{dataset}.npz"
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **arrays)
            for key, value in arrays.items():
                manifest.arrays[f"cmvn/{dataset}/{key}"] = {
                    "path": relative,
                    "key": key,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "tolerance": {
                        "rtol": FEATURE_TOLERANCE.rtol,
                        "atol": FEATURE_TOLERANCE.atol,
                    },
                }


def capture_waveforms(
    manifest: Manifest,
    recipe: Path,
    data: Path,
    destination: Path,
    dataset: str,
    keys: List[str],
) -> None:
    """
    The audio a feature extractor is given, for a few utterances.

    `wav.scp` can name a file or a command ending in `|`, and may be
    paired with a `segments` file; `kaldiio.load_scp` handles all three,
    which is why it is used here rather than a wave reader. Without this
    the capture records what came *out* of feature extraction and nothing
    that went in, which is the gap that made the filterbank residual
    unmeasurable -- see the comment at the call site.

    **The paths in `wav.scp` are relative to the recipe directory, and
    this is read from somewhere else.** mini-librispeech's entries are
    `flac -c -d -s ./corpus/LibriSpeech/... |`, which a Kaldi recipe can
    write because every one of its stages runs from its own directory --
    so the first capture that reached this function died on
    `FLAC__STREAM_DECODER_INIT_STATUS_ERROR_OPENING_FILE` with the
    recipe's whole five-hour run already spent. The reads therefore happen
    with the working directory set to `recipe`, which is reproducing the
    recipe's own assumption rather than guessing at it. yesno's paths are
    relative too and only looked fine because its capture predates this
    function.

    Arguments
    ---------
    manifest: Manifest
        The manifest to record into.
    recipe: Path
        The recipe directory, which is what `wav.scp`'s relative paths
        and piped commands are written against.
    data: Path
        The data directory.
    destination: Path
        The capture directory.
    dataset: str
        Its name, for the manifest keys.
    keys: List[str]
        The utterances to keep, which are the ones whose features are kept
        whole -- the pair is the point.
    """
    import kaldiio
    import numpy as np

    # Absolute before the chdir below, because a relative destination
    # would otherwise be written inside the recipe.
    scp = (data / "wav.scp").resolve()
    destination = destination.resolve()
    if not scp.is_file():
        sys.stderr.write(f"capture: no {scp}, skipping waveforms\n")
        return
    segments = (data / "segments").resolve()

    arrays = {}
    rates = {}
    with working_directory(recipe):
        try:
            loader = kaldiio.load_scp(
                str(scp), segments=str(segments) if segments.is_file() else None
            )
        except Exception as failure:  # pragma: no cover - corpus dependent
            sys.stderr.write(f"capture: cannot read {scp}: {failure!r}\n")
            return
        for key in keys:
            if key not in loader:
                sys.stderr.write(f"capture: no audio for {key}\n")
                continue
            try:
                rate, samples = loader[key]
            except Exception as failure:
                # A corpus whose audio cannot be decoded is a gap in the
                # capture, not a reason to lose the rest of a run that
                # has already cost hours. Said loudly, since a silent
                # skip here is what made the waveforms missing and
                # unexplained once before.
                sys.stderr.write(
                    f"capture: cannot decode {key} from {scp}: {failure!r}\n"
                )
                continue
            arrays[key] = np.asarray(samples)
            rates[key] = int(rate)
    if not arrays:
        sys.stderr.write(f"capture: no waveforms captured for {dataset}\n")
        return

    relative = f"audio/{dataset}.npz"
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    for key, value in arrays.items():
        manifest.arrays[f"audio/{dataset}/{key}"] = {
            "path": relative,
            "key": key,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sample_rate": rates[key],
            # Exact: these are the samples themselves, not a computation
            # over them, so any difference is a different recording.
            "tolerance": {"rtol": 0.0, "atol": 0.0},
        }


def capture_transcriptions(
    manifest: Manifest,
    recipe: Path,
    destination: Path,
    topologies: List[str],
    model: str,
    test_set: str,
    decode_dir: str,
) -> None:
    """
    Capture the reference and decoded transcriptions for each topology.

    Arguments
    ---------
    manifest: Manifest
        The manifest to record into.
    recipe: Path
        The recipe directory.
    destination: Path
        The capture directory.
    topologies: List[str]
        The topologies the recipe decoded.
    model: str
        The model name, as the experiment directories are prefixed.
    test_set: str
        The decoded data set.
    decode_dir: str
        The decoding parameter directory name.
    """
    for topology in topologies:
        experiment = recipe / "exp" / f"{model}-{topology}"
        reference = experiment / f"pred_{test_set}" / "ref.wrd.trn"
        if reference.is_file():
            store(
                manifest,
                destination,
                f"transcriptions/{topology}/ref.wrd.trn",
                reference.read_bytes(),
            )
        hypothesis = experiment / f"dec_{test_set}" / decode_dir / "hyp.wrd.trn"
        if hypothesis.is_file():
            store(
                manifest,
                destination,
                f"transcriptions/{topology}/hyp.wrd.trn",
                hypothesis.read_bytes(),
            )


def main(argv=None) -> int:
    """
    Capture a reference from a finished recipe run.

    Arguments
    ---------
    argv: Optional[List[str]]
        The arguments, defaulting to sys.argv[1:].

    Returns
    -------
    int
        0 if anything was captured, 1 if nothing was.
    """
    parser = argparse.ArgumentParser(
        description="Capture Kaldi's output from a finished recipe run.",
    )
    parser.add_argument("recipe", type=Path, help="the recipe directory")
    parser.add_argument(
        "destination",
        type=Path,
        help="where to write <corpus>/; usually tests/golden/reference",
    )
    parser.add_argument("--corpus", default="yesno")
    parser.add_argument("--model", default="rnnp")
    parser.add_argument("--test-set", default="test_yesno")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["train_yesno", "test_yesno"],
    )
    parser.add_argument("--decode-dir", default="aw_1.0-ma_5000-bm_32")
    parser.add_argument(
        "--lang-dir",
        default="data/lang",
        help=(
            "the language directory prepare_lang.sh produced, relative to "
            "the recipe; data/lang_nosp for mini_librispeech"
        ),
    )
    parser.add_argument(
        "--graph-prefix",
        default="data/lang_test_tg_",
        help=(
            "what a per-topology graph directory is called, minus the "
            "topology; data/lang_test_tgsmall_ for mini_librispeech"
        ),
    )
    parser.add_argument(
        "--dict-dir",
        default="data/local/dict",
        help=(
            "the dictionary prepare_lang.sh reads, whose contents are the "
            "input a Python replacement has to be driven by"
        ),
    )
    parser.add_argument(
        "--tmp-dir",
        default="data/local/lang",
        help=(
            "prepare_lang.sh's working directory, which holds the "
            "disambiguated lexicon; data/local/lang_tmp_nosp for "
            "mini_librispeech"
        ),
    )
    parser.add_argument(
        "--topos",
        nargs="+",
        default=None,
        help="topologies to capture; by default whatever the recipe built",
    )
    parser.add_argument(
        "--allow-no-fstprint",
        action="store_true",
        help=(
            "capture everything else if fstprint cannot be found, instead of "
            "failing. The compiled FSTs are the point of the capture for the "
            "lexicon and graph work, so their absence is an error by default."
        ),
    )
    args = parser.parse_args(argv)

    recipe = args.recipe.resolve()
    if not (recipe / "data").is_dir():
        sys.stderr.write(f"capture: no data directory in {recipe}\n")
        return 1

    topologies = args.topos
    if topologies is None:
        prefix = Path(args.graph_prefix)
        parent = recipe / prefix.parent
        stem = prefix.name
        topologies = sorted(
            path.name[len(stem) :] for path in parent.glob(f"{stem}*") if path.is_dir()
        )
    if not topologies:
        sys.stderr.write("capture: no prepared graphs found\n")

    # Resolved before anything is written, so a capture that cannot include
    # the FSTs fails before producing a directory that looks complete.
    printer = find_fstprint()
    if printer is None:
        message = (
            "capture: fstprint was not found on PATH or under "
            f"{os.environ.get('KALDI_ROOT', '/opt/kaldi')}/tools/openfst/bin. "
            "The lexicon and graph FSTs are what a Python replacement for "
            "prepare_lang.sh and the graph compilation will be compared "
            "against, so a capture without them is not much use."
        )
        if not args.allow_no_fstprint:
            sys.stderr.write(message + " Pass --allow-no-fstprint to proceed.\n")
            return 1
        sys.stderr.write(message + " Continuing without them.\n")

    out = (args.destination / args.corpus).resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest = Manifest(
        corpus=args.corpus,
        captured=datetime.date.today().isoformat(),
        # Recorded by the workflow, which knows the digest it pulled. A
        # reference whose provenance is unknown is not a reference.
        image=os.environ.get("BENNEVIS_IMAGE", "unknown"),
        recipe=str(args.recipe),
        notes=[
            "Captured from a completed run of the recipe named above, inside "
            "the image named above.",
            "The `exact` entries are compared as bytes: there is no reason "
            "for a correct reimplementation to differ by a character.",
            "The `arrays` and `stats` entries carry a tolerance, because "
            "bit-exact features are not achievable across implementations "
            "(FFT and accumulation order differ).",
            f"{FULL_FEATURE_UTTERANCES} feature matrices per data set are "
            "stored whole; the rest are statistics, because every matrix "
            "would be megabytes and buy nothing over a handful.",
            "Feature archives are rewritten uncompressed by Kaldi's own "
            "copy-feats before being read, so the values are the ones Kaldi "
            "worked with rather than another decompression of them. Without "
            "that, a comparison cannot be better than about 4e-6.",
        ],
    )

    for name in RECIPE_CONF:
        source = recipe / name
        if source.is_file():
            store(manifest, out, f"recipe/{name}", source.read_bytes())
        else:
            sys.stderr.write(f"capture: no {source}\n")

    capture_language(
        manifest,
        recipe,
        out,
        topologies,
        printer,
        lang_dir=args.lang_dir,
        dict_dir=args.dict_dir,
        tmp_dir=args.tmp_dir,
        graph_prefix=args.graph_prefix,
    )
    capture_features(
        manifest, recipe, out, args.datasets, copy_feats=find_kaldi_tool("copy-feats")
    )
    capture_transcriptions(
        manifest,
        recipe,
        out,
        topologies,
        args.model,
        args.test_set,
        args.decode_dir,
    )

    manifest.write(out / MANIFEST_NAME)

    exact, arrays, stats = manifest.names()
    sys.stdout.write(
        f"capture: wrote {out}\n"
        f"  {len(exact)} exact artefact(s)\n"
        f"  {len(arrays)} array(s) stored whole\n"
        f"  {len(stats)} array(s) as statistics\n"
    )
    if manifest.is_empty():
        sys.stderr.write("capture: nothing was captured\n")
        return 1

    # Silence is not success: say which of the FSTs the capture exists for
    # are actually in it, so an incomplete one is visible in the log rather
    # than discovered when a migration PR has nothing to compare against.
    if printer is not None:
        expected = ["lang/L.fst.txt", "lang/L_disambig.fst.txt"] + [
            f"graph/{topology}/TLG.fst.txt" for topology in topologies
        ]
        absent = [name for name in expected if name not in manifest.exact]
        if absent:
            sys.stderr.write(
                f"capture: {len(absent)} expected FST(s) are missing, the "
                f"first being {absent[0]}; the recipe may not have built "
                f"them\n"
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
