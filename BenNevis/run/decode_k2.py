#!/usr/bin/env python3
"""
Decode with k2 instead of Kaldi's `decode-faster`.

Deliberately thin: everything it does lives in `BenNevis.decode.pipeline`,
`BenNevis.decode.decoder` and `BenNevis.utils.scoring`, all of which are
tested in tiers 1 and 2. An entry-point script is reachable from no tier but
3, which is how the regression in #5 got in, so there is as little here as
possible.

Added *beside* `decode_faster.sh` rather than in place of it. One nightly
then decodes the same models both ways and the WER ledger can compare them
directly, which is the only check that can tell whether two differently
pruned searches agree -- and it is how this gets judged, since nothing in
tier 1 or 2 can.

**Two searches live behind `--search`, and the default is no longer k2's.**
`intersect_dense_pruned` could not decode mini-librispeech inside a CI run:
three attempts in a row lost the measurement, the last SIGKILLed at 14,988
MiB with 1,000 of 1,089 utterances done. `--search viterbi` is Kaldi's own
token passing written in torch -- see `BenNevis.decode.viterbi` for why
that was the fix and what was measured on the way -- and it decodes the
same graph in under a gigabyte at Kaldi's own beam. `--search k2` keeps the
older route, because it is the one the yesno equivalence was established
with and two searches checking each other is the only comparison either of
them has.

Usage
-----
    python3 run/decode_k2.py <data_dir> <lang_dir> <predict_dir> <decode_dir> \
        --lexicon <lexicon.txt> --arpa <lm.arpa> --topo <topo>

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import argparse
import logging
import sys
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Decode with k2 rather than Kaldi's decode-faster."
    )
    parser.add_argument("data_dir", help="the data directory, for utt2spk")
    parser.add_argument("lang_dir", help="for words.txt and phones.txt")
    parser.add_argument("predict_dir", help="holding output.scp and ref.wrd.trn")
    parser.add_argument("decode_dir", help="where to write the hypotheses")
    parser.add_argument("--lexicon", required=True, help="the recipe's lexicon.txt")
    parser.add_argument("--arpa", required=True, help="the ARPA language model")
    parser.add_argument("--topo", required=True, help="which topology")
    parser.add_argument("--variant", default="k2")
    parser.add_argument("--sil_prob", type=float, default=0.0)
    parser.add_argument(
        "--position_dependent_phones", action="store_true", default=False
    )
    parser.add_argument("--acoustic_scale", type=float, default=1.0)
    parser.add_argument(
        "--search",
        choices=("viterbi", "k2"),
        default="viterbi",
        help="which search. `viterbi` is Kaldi's token passing written in "
        "torch, which decodes mini-librispeech's dev_clean_2 in about "
        "0.6 to 2.0 seconds an utterance against "
        "`intersect_dense_pruned`'s 5.2, at the same WER or better, and "
        "in under a gigabyte rather than fifteen. `k2` is the older route, "
        "kept because it is what the yesno equivalence was established "
        "with and because the two searches checking each other is the "
        "only comparison either of them has",
    )
    # Kaldi's token passing. The defaults are `run/decode_faster.sh`'s own,
    # not this file's preference, so stage 10 and stage 7 differ in their
    # *search* and in nothing else -- which is the whole point of having
    # both. `tests/recipes/test_decode_k2_settings.py` reads them out of
    # both files and fails if they drift apart.
    parser.add_argument("--beam", type=float, default=32.0)
    parser.add_argument("--max_active", type=int, default=5000)
    parser.add_argument("--min_active", type=int, default=20)
    parser.add_argument(
        "--nj",
        type=int,
        default=1,
        help="how many processes to decode in, for `--search viterbi`. "
        "The recipe passes its own ${nj}, which is what decode_faster.sh "
        "gets, so the two stages are given the same machine. Forked, so "
        "the children share the prepared graph by copy-on-write rather "
        "than each holding a gigabyte of their own",
    )
    parser.add_argument(
        "--factor_threshold",
        type=int,
        default=8,
        help="the smallest group of arcs sharing an input label that is "
        "worth factoring into a non-emitting group; see "
        "BenNevis.decode.viterbi. A number larger than any group turns "
        "factoring off, which is 9x slower and gives the same answer",
    )
    # `intersect_dense_pruned`. Kept for `--search k2`.
    parser.add_argument("--search_beam", type=float, default=20.0)
    parser.add_argument("--output_beam", type=float, default=8.0)
    parser.add_argument("--min_active_states", type=int, default=30)
    parser.add_argument("--max_active_states", type=int, default=10000)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="how many utterances to decode at once, for `--search k2` "
        "only: the token-passing search is frame synchronous over one "
        "utterance's own active set, so a batch buys it nothing. 8 rather "
        "than 1 because run 45511013199 spent 4h16m48s on 900 of 1,089 "
        "utterances with a batch of one, and run 45715720923 measured the "
        "batch at 2.65x; 8 is also what decode_faster.sh runs at nj=8",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    import kaldiio
    import torch

    from BenNevis.decode.pipeline import (
        build_graph,
        read_symbol_table,
        read_utt2spk,
        write_trn,
    )
    from BenNevis.utils.scoring import read_trn, score

    lang_dir = Path(args.lang_dir)
    predict_dir = Path(args.predict_dir)
    decode_dir = Path(args.decode_dir)

    scp = predict_dir / "output.scp"
    if not scp.is_file():
        logging.error("no %s; run predict first", scp)
        return 1

    graph = build_graph(
        lang_dir,
        Path(args.lexicon),
        Path(args.arpa),
        args.topo,
        variant=args.variant,
        sil_prob=args.sil_prob,
        position_dependent_phones=args.position_dependent_phones,
    )
    logging.info("decoding graph: %d states, %d arcs", graph.shape[0], graph.num_arcs)

    words = read_symbol_table(lang_dir / "words.txt")
    idx2word = {index: word for word, index in words.items()}
    utt2spk = read_utt2spk(Path(args.data_dir) / "utt2spk")

    def utterances():
        for uttid, array in kaldiio.load_scp_sequential(str(scp)):
            # Copied because kaldiio hands back a read-only view and
            # torch warns once per process about non-writable tensors.
            yield uttid, torch.from_numpy(array.copy())

    def shards(count):
        """
        `count` callables, each yielding one contiguous block of the scp.

        Contiguous rather than round-robin, because the blocks are
        concatenated in order afterwards and the recipe merges the result
        against `ref.wrd.trn` by position -- the same reason
        `split_scp.pl` splits the way it does.

        Each callable reads its own features, in its own process. The
        parent only ever holds the scp's lines: mini-librispeech's 1,089
        utterances are 1.6GB of log probabilities, which is not something
        to materialise in order to hand it to a child.
        """
        entries = []
        for line in scp.read_text(encoding="utf-8").splitlines():
            fields = line.split(None, 1)
            if len(fields) == 2:
                entries.append((fields[0], fields[1]))
        count = max(1, min(count, len(entries)))
        size = -(-len(entries) // count)
        blocks = [entries[i : i + size] for i in range(0, len(entries), size)]

        def reader(block):
            def read():
                for uttid, location in block:
                    yield uttid, torch.from_numpy(kaldiio.load_mat(location).copy())

            return read

        return [reader(block) for block in blocks]

    if args.search == "viterbi":
        import gc

        from BenNevis.decode.viterbi import SearchGraph
        from BenNevis.decode.viterbi import decode_shards
        from BenNevis.utils.memory import trim_malloc

        search_graph = SearchGraph.from_fsa(
            graph, factor_threshold=args.factor_threshold
        )
        logging.info(
            "prepared for token passing: %d states, %d emitting and %d "
            "non-emitting arcs",
            search_graph.num_states,
            search_graph.num_emitting_arcs,
            search_graph.num_non_emitting_arcs,
        )
        # The composed FSA is a gigabyte and nothing reads it again; the
        # trim is there because preparing the graph leaves several times
        # that in free heap glibc does not hand back on its own, which is
        # the same lesson the trainer learned in #84.
        del graph
        gc.collect()
        trim_malloc()
        hypotheses = decode_shards(
            search_graph,
            shards(args.nj),
            idx2word,
            acoustic_scale=args.acoustic_scale,
            beam=args.beam,
            max_active=args.max_active,
            min_active=args.min_active,
        )
    else:
        from BenNevis.decode.decoder import decode_archive

        hypotheses = decode_archive(
            graph,
            utterances(),
            idx2word,
            acoustic_scale=args.acoustic_scale,
            batch_size=args.batch_size,
            search_beam=args.search_beam,
            output_beam=args.output_beam,
            min_active_states=args.min_active_states,
            max_active_states=args.max_active_states,
        )
    logging.info("decoded %d utterances", len(hypotheses))

    output = decode_dir / f"aw_{args.acoustic_scale}"
    write_trn(hypotheses, utt2spk, output / "hyp.wrd.trn")

    reference = predict_dir / "ref.wrd.trn"
    if reference.is_file():
        counts = score(read_trn(reference), read_trn(output / "hyp.wrd.trn"))
        text = (
            f"WER {counts.wer:.2f}%  "
            f"[{counts.errors}/{counts.reference_words} words, "
            f"{counts.substitutions} sub, {counts.deletions} del, "
            f"{counts.insertions} ins]\n"
        )
        (output / "results.wrd.txt").write_text(text, encoding="utf-8")
        logging.info(text.strip())
    else:
        logging.warning("no %s, so nothing was scored", reference)

    return 0


if __name__ == "__main__":
    sys.exit(main())
