"""
Tests that `docs/kaldi-graph-chain.md` still describes the shipped chain.

The document exists because four things the Kaldi graph pipeline does are
not guessable from the shell -- determinization in the log semiring, the
epsilon removal folded into it, where the 1/1024 quantisation enters, and
why the disambiguation symbols are carried through the T composition. The
Python port differs on all four, and goal 2 cannot be done faithfully
without knowing it.

A document that drifts from the script is worse than none, because it
reads as authoritative. `BenNevis/run/prepare_graph.sh` is the single
source of truth -- each recipe reaches it through a `run` symlink -- so
these compare the two directly, in both directions: a binary dropped from
the script and a binary added to it both fail.

Deliberately not a comparison of the whole pipeline text. The document
explains rather than quotes, and pinning it line for line would make it
unmaintainable prose.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "BenNevis" / "run" / "prepare_graph.sh"
DOC = REPO_ROOT / "docs" / "kaldi-graph-chain.md"

#: Any `fst...` binary, as either script or document writes it.
BINARY = re.compile(r"\b(fst[a-z]+)\b")

#: Flags the document makes a claim about, so they cannot change quietly.
FLAGS = ("--use-log=true", "--sort_type=ilabel")


def binaries_in(path):
    """
    The OpenFST and Kaldi binaries a file mentions.

    Arguments
    ---------
    path: Path
        The script or the document.

    Returns
    -------
    set of str
        One entry per distinct binary name.
    """
    return set(BINARY.findall(path.read_text()))


def binaries_in_the_doc_chain():
    """
    The binaries the document's fenced block claims the chain runs.

    The prose names others legitimately -- the source directories
    `fstbin` and `fstext`, and `fstminimize` inside a quoted usage string
    -- so only the block that presents itself as the shipped pipeline is
    compared against the script.

    Returns
    -------
    set of str
        One entry per distinct binary name in the fenced bash block.
    """
    blocks = re.findall(r"```bash\n(.*?)```", DOC.read_text(), re.DOTALL)
    assert len(blocks) == 1, f"{DOC.name} has {len(blocks)} bash blocks"
    return set(BINARY.findall(blocks[0]))


def test_both_files_are_there():
    assert SCRIPT.is_file(), SCRIPT
    assert DOC.is_file(), DOC


def test_the_document_covers_every_binary_the_script_runs():
    """
    A stage nobody wrote down is a stage that gets ported by guesswork.

    Compared against the fenced block and not the whole document: the
    prose mentioning a binary somewhere is not the same as the chain
    listing it, and reading the whole file let a stage go missing from
    the chain while a mention elsewhere kept this green.
    """
    missing = binaries_in(SCRIPT) - binaries_in_the_doc_chain()
    assert not missing, f"{DOC.name}'s chain omits {sorted(missing)}"


def test_every_stage_is_explained_and_not_only_listed():
    """
    The document's value is the prose, not the transcription.

    A binary added to the chain block and never explained leaves the next
    person with the shell they already had.
    """
    text = DOC.read_text()
    start = text.find("## Stage by stage")
    end = text.find("## Where", start)
    assert start != -1 and end > start, "the explaining section is gone"
    # Bounded at both ends: the divergence table below it names every
    # binary too, so reading to the end of the file would let this pass
    # on a stage whose explanation had been deleted.
    section = text[start:end]
    for binary in sorted(binaries_in(SCRIPT)):
        assert binary in section, f"{binary} is listed but never explained"


def test_the_document_invents_no_stages():
    """
    The other direction, which is the one that rots.

    A binary dropped from the script leaves the document describing a
    pipeline that no longer exists, and nothing else would notice.
    """
    extra = binaries_in_the_doc_chain() - binaries_in(SCRIPT)
    assert not extra, f"{DOC.name} describes {sorted(extra)}, not in the script"


def test_the_flags_the_document_explains_are_still_used():
    """
    `--use-log=true` is the central claim: it determinizes in the log
    semiring, so merged paths are log-added rather than min-ed. If it ever
    goes, the document's explanation of the weights goes with it.
    """
    script = SCRIPT.read_text()
    for flag in FLAGS:
        assert flag in script, f"{SCRIPT.name} no longer passes {flag}"
        assert flag in DOC.read_text(), f"{DOC.name} does not explain {flag}"


def test_the_document_records_the_minimisation_precondition():
    """
    The finding a reader would otherwise assume the other way.

    Textbook minimisation is defined on a DFA, so the obvious reading of
    the shell pipeline is that `fstminimizeencoded` is only available
    after `fstdeterminizestar` -- and therefore that dropping
    determinization dropped minimisation with it. OpenFST's
    `AcceptorMinimize` falls back to Hopcroft for non-deterministic input,
    and `Minimize`'s own comment allows it in an idempotent semiring,
    which the tropical semiring is. The document has to say so, because
    the mistake is the kind that silently narrows what anyone considers
    doing next.
    """
    text = DOC.read_text()
    assert "Hopcroft" in text, "the fallback that makes this work is unnamed"
    assert "idempotent" in text, "the condition on the semiring is unstated"
    assert "separable" in text, "the conclusion for our own chain is missing"


def test_the_quantisation_rule_is_given_rather_than_only_its_effect():
    """
    `floor(w / delta + 0.5) * delta` is what a replacement must copy to
    land on Kaldi's grid; "quantised to 1/1024" is not enough to
    reproduce, and the 2358/1024 example is only evidence that the rule is
    right.
    """
    text = DOC.read_text()
    assert "std::floor(Value() / delta + 0.5F) * delta" in text
    assert "2358/1024" in text


def test_the_document_names_the_variant_the_script_actually_calls():
    """
    `fstrmepslocal` with no flags is `RemoveEpsLocalSpecial`.

    `fstrmepslocal.cc` defaults `use_log = false` and
    `stochastic_in_log = true`, and that combination picks the special
    variant rather than the plain `RemoveEpsLocal` the header documents
    first. A replacement written against the wrong one preserves
    stochasticity in the wrong semiring, which is not something any test
    of ours would notice.
    """
    text = DOC.read_text()
    assert "RemoveEpsLocalSpecial" in text


def test_the_document_records_that_rmepslocal_merges_arc_pairs():
    """
    The operation `k2.remove_epsilon` has no counterpart for.

    `remove-eps-local.h`: it "does not just remove epsilon arcs; it also
    combines pairs of input-epsilon and output-epsilon arcs into one." A
    reader who takes the stage for epsilon removal alone will substitute
    `k2.remove_epsilon` and lose the arc-pair merge, which is part of how
    the stage shrinks a graph.
    """
    text = DOC.read_text()
    assert "(x:eps)(eps:y)" in text, "the merge is not shown"
