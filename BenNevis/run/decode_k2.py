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
    parser.add_argument("--search_beam", type=float, default=20.0)
    parser.add_argument("--output_beam", type=float, default=8.0)
    parser.add_argument("--min_active_states", type=int, default=30)
    parser.add_argument("--max_active_states", type=int, default=10000)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    import kaldiio
    import torch

    from BenNevis.decode.decoder import decode_archive
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

    hypotheses = decode_archive(
        graph,
        utterances(),
        idx2word,
        acoustic_scale=args.acoustic_scale,
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
