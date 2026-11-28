"""
Compare the Python decoding graphs against Kaldi's, inside the recipe.

This is the equivalence gate for mini-librispeech, and it has to run here
rather than in a test tier because of what it needs. `tests/decode/`
compares against yesno's committed `LG.fst.txt` and `TLG.fst.txt`, which
works only because they are 20KB and 313 bytes; mini-librispeech's are
138,155,979 and 340,397,504 bytes, kept in the capture as digests alone.
And our side cannot be rebuilt offline either, because the 3-gram model
is downloaded rather than committed. Both sides exist only inside a
recipe run, so the comparison belongs there.

**yesno cannot answer this question.** Measured: on yesno our `TLG` is 11
states and 33 arcs against Kaldi's 10 and 29, and the weights agree -- a
four-word lexicon leaves Kaldi's second pipeline nothing to compact. On
mini-librispeech ours is 43,263,880 arcs. A replacement validated on
yesno alone would look finished.

What is compared, and why each side is probed differently:

* **LG on its input side.** Its input alphabet is phones, so it is asked
  for the weight of a phone sequence, drawn from the words the lexicon
  pronounces -- sampling from `words.txt` would draw sequences the
  lexicon cannot say, and a comparison of rejections compares nothing.
* **TLG on its output side.** Its input alphabet is tokens, and under a
  CTC topology a word sequence has many token realisations, so the
  question a decoder asks is the minimum over all of them. Probing one
  hand-picked token sequence would compare an alignment no decoder need
  take. It also avoids the token tables, whose ids differ by one between
  Kaldi's `tokens.txt` and our `k2/tokens.txt`. The graph is inverted so
  the words are its input labels and then **searched**, because
  `k2.remove_epsilon` on 43,263,880 arcs does not fit the runner and the
  Dijkstra does not need it.

Weights are compared within `n * kDelta / 2` -- Kaldi's own quantisation,
since `fstminimizeencoded` rounds every arc to a multiple of 1/1024.
Kaldi is the less precise side; this bounds the comparison rather than
reproducing the rounding.

The sizes are reported whatever the weights do, because they are the open
question: being three times Kaldi's arc count is not a correctness
failure but it is a cost, and the number belongs in the job log.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# So that `capture.py` beside this file is importable however this is
# invoked.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))


def fstprint(printer: str, path: Path) -> str:
    """Dump a compiled FST as OpenFST text."""
    result = subprocess.run(
        [printer, str(path)], capture_output=True, text=True, check=True
    )
    return result.stdout


def read_table(path: Path) -> Dict[str, int]:
    """A symbol table as a name-to-id mapping."""
    table: Dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2:
            table[fields[0]] = int(fields[1])
    return table


def say(label: str, started: float) -> None:
    print(f"  {label:44s} {time.time() - started:8.1f}s", flush=True)


def size(label: str, fsa) -> None:
    print(
        f"  {label:44s} {fsa.shape[0]:>12,} states {fsa.num_arcs:>14,} arcs",
        flush=True,
    )


def report(failures: List[str], half: str) -> None:
    """
    Print one half's mismatches as soon as that half is known.

    Arguments
    ---------
    failures: List[str]
        The mismatches found so far in this half.
    half: str
        Which half, for the summary line.

    Notes
    -----
    Waiting until the end was a mistake worth not repeating: the TLG
    half was killed inside a minute on mini-librispeech, taking the LG
    result with it even though the LG half had already finished.
    """
    for failure in failures:
        print(f"MISMATCH: {failure}", flush=True)
    print(
        f"  {half}: {len(failures)} mismatch(es)" if failures else f"  {half}: clean",
        flush=True,
    )


def compare_lg(ours, theirs, phones, drawn, delta) -> Tuple[int, List[str]]:
    """
    Compare LG on phone sequences the lexicon permits.

    Returns the number compared and the failures, so the caller can tell
    "agreed on nothing" from "agreed on everything" -- a gate that
    compared nothing must not pass.
    """
    from BenNevis.decode.graph import LabelProbe

    # One probe per graph, not per question. The pack is per graph, and
    # paying it per sequence is what made this half 125.7s where it is
    # now six seconds.
    their_probe = LabelProbe(theirs)
    our_probe = LabelProbe(ours)

    failures: List[str] = []
    compared = 0
    for words, phone_sequence in drawn:
        if any(phone not in phones for phone in phone_sequence):
            continue
        ids = [phones[phone] for phone in phone_sequence]
        # Searched rather than composed. `k2.intersect` cannot do this on
        # Kaldi's LG either way: with `treat_epsilons_specially=False`
        # label 0 is an ordinary symbol and every path with an input
        # epsilon is rejected, which made the gate report `compared LG on
        # 0 sequences`; with it True the epsilons match, but an n-gram
        # back-off arc is an input epsilon and those form cycles, so
        # `k2.shortest_path` fails with `dest_state > state_idx01`.
        # `LabelProbe` searches with a Dijkstra over
        # (state, position) pairs, which tolerates the cycles because the
        # weights are non-negative.
        kaldi = their_probe.weight(ids)
        mine = our_probe.weight(ids)
        key = " ".join(words)
        if kaldi is None and mine is None:
            # Both reject it: a fact about the lexicon and grammar, not a
            # disagreement, and nothing to compare.
            continue
        if kaldi is None or mine is None:
            failures.append(
                f"LG {key}: Kaldi "
                f"{'rejects' if kaldi is None else f'{kaldi:.6f}'}, ours "
                f"{'rejects' if mine is None else f'{mine:.6f}'}"
            )
            continue
        bound = (len(words) + 1) * delta
        compared += 1
        if abs(kaldi - mine) > bound:
            failures.append(
                f"LG {key}: Kaldi {kaldi:.6f}, ours {mine:.6f}, gap "
                f"{abs(kaldi - mine):.6f} exceeds {bound:.6f}"
            )
    return compared, failures


def compare_tlg(ours, theirs, words_table, drawn, delta) -> Tuple[int, List[str]]:
    """
    Compare TLG on word sequences, searched on the output side.

    **Inverted and then searched, not inverted and epsilon-freed.** The
    first version of this half called `prepare_output_probe`, whose
    `k2.remove_epsilon` is what a 43,263,880-arc graph does not survive:
    run 45554380456 printed `preparing the output probes` and stopped
    inside a minute with no traceback, which is what a SIGKILL looks
    like. `LabelProbe` needs no removal at all, because a
    Dijkstra over (state, position) pairs tolerates the cycles that
    epsilon arcs form -- every arc of a graph built from `-log`
    probabilities costs nothing or more, so going round a cycle can only
    make a path dearer.

    The epsilons it skips are the ones that matter here: most of a token
    graph's arcs emit no word at all, 20 of 29 even in yesno's `TLG`, and
    once the graph is inverted those are input epsilons. The label is
    read out of `words.txt` rather than assumed, because 0 is `<eps>` on
    this side and `<blk>` on the token side -- a probe hardcoding it
    would look identical while skipping the wrong thing.

    Both probes are packed once and reused across the sequences. The
    pack is a property of the graph, so asking 24 questions of a
    per-question probe repacked 43M and 12.9M arcs 24 times each.

    Returns the number compared and the failures, like `compare_lg`, so
    a half that compared nothing cannot pass for a half that agreed.
    """
    import k2

    from BenNevis.decode.graph import LabelProbe

    epsilon = words_table.get("<eps>", 0)
    started = time.time()
    their_inverted = k2.invert(theirs)
    say("Kaldi TLG inverted", started)
    size("Kaldi TLG inverted", their_inverted)
    started = time.time()
    our_inverted = k2.invert(ours)
    say("our TLG inverted", started)
    size("our TLG inverted", our_inverted)
    started = time.time()
    their_probe = LabelProbe(their_inverted, epsilon=epsilon)
    our_probe = LabelProbe(our_inverted, epsilon=epsilon)
    say("packed both TLG probes", started)

    failures: List[str] = []
    compared = 0
    for sentence, _phones in drawn:
        if any(word not in words_table for word in sentence):
            continue
        ids = [words_table[word] for word in sentence]
        kaldi = their_probe.weight(ids)
        mine = our_probe.weight(ids)
        key = " ".join(sentence)
        if kaldi is None and mine is None:
            continue
        if kaldi is None or mine is None:
            failures.append(
                f"TLG {key}: Kaldi "
                f"{'rejects' if kaldi is None else f'{kaldi:.6f}'}, ours "
                f"{'rejects' if mine is None else f'{mine:.6f}'}"
            )
            continue
        bound = (len(sentence) + 1) * delta
        compared += 1
        if abs(kaldi - mine) > bound:
            failures.append(
                f"TLG {key}: Kaldi {kaldi:.6f}, ours {mine:.6f}, gap "
                f"{abs(kaldi - mine):.6f} exceeds {bound:.6f}"
            )
    return compared, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument(
        "--graph-dir",
        required=True,
        help="the per-topology graph directory, relative to the recipe; "
        "this is what stage 10 passes as the language directory, because "
        "prepare_topo.sh writes tokens.txt and k2/ there",
    )
    parser.add_argument("--lexicon", required=True)
    parser.add_argument("--arpa", required=True)
    parser.add_argument("--topo", default="ctc")
    parser.add_argument("--sil-prob", type=float, default=0.0)
    parser.add_argument(
        "--position-dependent-phones",
        action="store_true",
        help="must match the recipe's prepare_lang.sh call; getting it "
        "wrong builds a lexicon that is wrong rather than broken",
    )
    parser.add_argument(
        "--fstprint",
        default=None,
        help="the fstprint executable; found the way capture.py finds it "
        "when omitted, because it is in the image but only on the PATH a "
        "recipe's path.sh builds",
    )
    parser.add_argument("--sequences", type=int, default=24)
    parser.add_argument(
        "--skip-tlg",
        action="store_true",
        help="compare LG only. The escape, kept now that the TLG half no "
        "longer calls remove_epsilon: the Dijkstra is affordable where "
        "that was not, but it is still the expensive half",
    )
    args = parser.parse_args()

    import k2  # noqa: F401  -- imported early so a missing wheel fails fast

    from BenNevis.decode.graph import QUANTISATION_DELTA, compose_lg, load_openfst_text
    from BenNevis.decode.pipeline import build_graph
    from BenNevis.lm import build_fst_text, read_arpa
    from BenNevis.lm.lexicon import build_disambig_fst_text, read_lexicon
    from BenNevis.utils.fingerprint import (
        position_dependent_table,
        sample_lexicon_sequences,
    )

    # Found rather than assumed on the PATH: fstprint lives in the image
    # but only on the PATH a recipe's `path.sh` builds, and sourcing that
    # from the workflow is what cost this step its environment once
    # already. `capture.py` solved the same problem; reuse it rather than
    # keep a second answer.
    printer = args.fstprint
    if printer is None:
        from capture import find_fstprint

        printer = find_fstprint()
        if printer is None:
            print(
                "FAILED: no fstprint, so Kaldi's graphs cannot be read",
                flush=True,
            )
            return 1
    print(f"  fstprint: {printer}", flush=True)

    graph_dir = args.recipe / args.graph_dir
    lexicon_path = args.recipe / args.lexicon
    arpa_path = args.recipe / args.arpa

    # Checked rather than assumed, and the reason is this step's own
    # `always()`. It runs after a failed recipe on purpose -- the graphs
    # are built in stage 4 and a run that died in training still has them
    # -- but a run that died *before* stage 4 has none, and then reading
    # `words.txt` raised a bare FileNotFoundError traceback on top of the
    # real failure. Run 46164030031 is the example: stage 2 exited 1, and
    # this step then added eight lines of pathlib stack to the diagnosis.
    # The status is unchanged -- a gate that cannot compare has not
    # passed -- but what it prints now names the cause.
    missing = [
        path
        for path in (
            graph_dir / "words.txt",
            graph_dir / "phones.txt",
            lexicon_path,
            arpa_path,
        )
        if not path.is_file()
    ]
    if missing:
        print(
            "FAILED: nothing to compare -- these inputs do not exist:",
            flush=True,
        )
        for path in missing:
            print(f"  {path}", flush=True)
        print(
            "FAILED: the graphs are built in stage 4 and the lexicon and "
            "model in stages 0 and 1, so a recipe that stopped before "
            "them leaves this gate nothing to do. Read the recipe step's "
            "own failure first; this one is downstream of it.",
            flush=True,
        )
        return 1

    words = read_table(graph_dir / "words.txt")
    phones = read_table(graph_dir / "phones.txt")
    first_disambig = min(value for key, value in phones.items() if key.startswith("#"))

    print(f"comparing {args.topo} graphs in {graph_dir}", flush=True)
    print(
        f"  position-dependent phones: {args.position_dependent_phones} "
        f"(phone table says {position_dependent_table(phones.keys())})",
        flush=True,
    )

    # Ours, through the shipped code path, so the gate tests what ships.
    started = time.time()
    lexicon_fsa = load_openfst_text(
        build_disambig_fst_text(
            read_lexicon(lexicon_path),
            words,
            phones,
            sil_prob=args.sil_prob,
            position_dependent_phones=args.position_dependent_phones,
        )
    )
    say("built our L_disambig", started)
    started = time.time()
    grammar = load_openfst_text(build_fst_text(read_arpa(arpa_path), words))
    say("built our G", started)
    started = time.time()
    our_lg = compose_lg(lexicon_fsa, grammar, first_disambig)
    say("built our LG", started)
    size("our LG", our_lg)

    started = time.time()
    their_lg = load_openfst_text(
        fstprint(printer, graph_dir / "LG.fst"), acceptor=False
    )
    say("loaded Kaldi's LG", started)
    size("Kaldi LG", their_lg)

    drawn = sample_lexicon_sequences(
        read_lexicon(lexicon_path),
        count=args.sequences,
        position_dependent_phones=args.position_dependent_phones,
    )
    print(f"  drew {len(drawn)} lexicon sequences", flush=True)

    started = time.time()
    compared_lg, failures = compare_lg(
        our_lg, their_lg, phones, drawn, QUANTISATION_DELTA
    )
    say(f"compared LG on {compared_lg} sequences", started)
    # Reported here rather than only at the end, because the TLG half is
    # the expensive one and has already been killed mid-way once: a crash
    # there took the LG verdict with it, and the LG verdict is the half
    # that costs nothing.
    report(failures, "LG")

    compared_tlg = 0
    if not args.skip_tlg:
        started = time.time()
        our_tlg = build_graph(
            graph_dir,
            lexicon_path,
            arpa_path,
            args.topo,
            sil_prob=args.sil_prob,
            position_dependent_phones=args.position_dependent_phones,
        )
        say("built our TLG", started)
        size("our TLG", our_tlg)
        started = time.time()
        their_tlg = load_openfst_text(
            fstprint(printer, graph_dir / "TLG.fst"), acceptor=False
        )
        say("loaded Kaldi's TLG", started)
        size("Kaldi TLG", their_tlg)
        started = time.time()
        compared_tlg, tlg_failures = compare_tlg(
            our_tlg, their_tlg, words, drawn, QUANTISATION_DELTA
        )
        say(f"compared TLG on {compared_tlg} sequences", started)
        report(tlg_failures, "TLG")
        failures.extend(tlg_failures)

    print(flush=True)

    # A gate that compared nothing must not pass. Agreement on an empty
    # set is how the G.fst gate came to skip unconditionally.
    if compared_lg == 0:
        print("FAILED: no LG sequence could be compared", flush=True)
        return 1
    if not args.skip_tlg and compared_tlg == 0:
        print("FAILED: no TLG sequence could be compared", flush=True)
        return 1
    if failures:
        print(f"FAILED: {len(failures)} mismatch(es)", flush=True)
        return 1
    print(
        f"OK: LG agreed on {compared_lg} sequences, TLG on {compared_tlg}, "
        f"within {QUANTISATION_DELTA} per arc",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
