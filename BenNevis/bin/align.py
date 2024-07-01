#!/usr/bin/env python3
"""
Note that this script is supposed to be run with a higher level script, run/align.sh

This script is used to align the log-probabilities with the text using the K2 library.
The script is divided into several sections, each of which is described below.

1. Importing libraries and setting up the environment
2. Reading the configuration
3. Reading the log-probabilities and the text
4. Reading the segments and the reco2file_and_channel, if provided
5. Aligning the log-probabilities with the text
6. Writing the alignments to disk, including the alignments, word-level alignments, and the ctm file

Usage:
    python align.py [options]

Example:
    align.py \
        --output_ctm_path=exp/model-topo/dec_test/acwt_1.0-maxac_5000-beam_32/align/ctm \
        --output_align_path=exp/model-topo/dec_test/acwt_1.0-maxac_5000-beam_32/align/ali.ark.gz \
        --output_word_align_path=exp/model-topo/dec_test/acwt_1.0-maxac_5000-beam_32/align/word_ali.ark.gz \
        --frame_shift=0.02 \
        --segments=data/test/segments \
        --reco2file_and_channel=data/test/reco2file_and_channel
        data/lang_topo \
        exp/model-topo/pred_test/output.scp \
        exp/model-topo/dec_test/acwt_1.0-maxac_5000-beam_32/hyp.wrd.txt

    Note: The segments and reco2file_and_channel are optional arguments. You only need to provide them if you have them.
        Besides, segments and reco2file_and_channel have to be provided together.

Here is a list of the files you will get:
    `ctm`: The ctm file
    `ali.ark.gz`: The state-level alignments in Kaldi ark format
    `word_ali.ark.gz`: The word-level alignments in Kaldi ark format

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""

import os
import k2
import torch
import numpy as np
import kaldiio
import logging
import argparse
from BenNevis.core.lang import Lang
from BenNevis.utils.data import read_dict


def read_segments(segments: str) -> dict:
    segments = {}
    with open(segments, "r") as f:
        for line in f:
            utt, rec, start, end = line.strip().split()
            segments[utt] = (rec, start, end)
    return segments


def read_reco2file_and_channel(reco2file_and_channel: str) -> dict:
    reco2file_and_channel = {}
    with open(reco2file_and_channel, "r") as f:
        for line in f:
            rec, file, channel = line.strip().split()
            reco2file_and_channel[rec] = (file, channel)
    return reco2file_and_channel


def get_lattice(
    log_prob: torch.Tensor,
    log_prob_len: torch.Tensor,
    decoding_graph: k2.fsa.Fsa,
) -> k2.fsa.Fsa:
    """
    Get the decoding lattice from a decoding graph and  log_softmax output.
    Note that this is modified from k2's `get_lattice` function but we use
    `intersect_dense` instead of `intersect_dense_pruned`.
    Arguments
    ---------
    log_prob : torch.Tensor
        A 3-D tensor of shape (N, T, C), where
        N is the number of utterances in a batch,
        T is the number of frames, and
        C is the number of classes (or tokens).
    log_prob_len : torch.Tensor
        A 1-D tensor of shape (N,) that contains the number of frames
        for each utterance.
    decoding_graph : k2.fsa.Fsa
        The decoding graph. It is usually obtained by calling the `Lang.compile_training_graph`

    Returns
    -------
      An FsaVec containing the decoding result. It has axes [utt][state][arc].
    """
    assert log_prob.ndim == 3, log_prob.shape
    assert log_prob_len.ndim == 1, log_prob_len.shape
    assert log_prob.size(0) == log_prob_len.size(0), (
        log_prob.shape,
        log_prob_len.shape,
    )

    batch_size = log_prob.size(0)

    supervision_segment = (
        torch.stack(
            [
                torch.arange(batch_size),
                torch.zeros(batch_size),
                log_prob_len.cpu(),
            ],
        )
        .t()
        .to(torch.int32)
    )

    dense_fsa_vec = k2.DenseFsaVec(
        log_prob,
        supervision_segment,
    )

    lattice = k2.intersect_dense(
        decoding_graph,
        dense_fsa_vec,
        5.0,
    )

    return lattice


def main(args):
    lang = Lang(args.lang_dir, load_topo=True, load_lexicon=True)
    utt2log_prob = kaldiio.load_scp(args.log_prob_scp)
    text = read_dict(args.text)
    if args.ignore_labels:
        ignore_labels = [int(label) for label in args.ignore_labels.split(",")]
    else:
        ignore_labels = []

    if args.segments is not None:
        segments = read_segments(args.segments)
    if args.reco2file_and_channel is not None:
        reco2file_and_channel = read_reco2file_and_channel(args.reco2file_and_channel)

    os.makedirs(os.path.dirname(args.output_ctm_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_align_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_word_align_path), exist_ok=True)

    count = 0

    with kaldiio.WriteHelper(
        f"ark:| gzip -c > {args.output_align_path}"
    ) as writer, kaldiio.WriteHelper(
        f"ark:| gzip -c > {args.output_word_align_path}"
    ) as word_writer, open(
        args.output_ctm_path, "w"
    ) as ctm_file:
        ctm = ""
        for utt, log_prob in utt2log_prob.items():
            logging.info(f"Processing utterance {utt}")
            log_prob = torch.tensor(np.array([log_prob]))
            log_prob_len = torch.tensor([log_prob.shape[1]])
            targets = [
                [
                    (
                        lang.word2idx[word]
                        if word in lang.word2idx
                        else lang.word2idx["<UNK>"]
                    )
                    for word in text[utt].split()
                ]
            ]
            channel = (
                "1"
                if args.reco2file_and_channel is None or args.segments is None
                else reco2file_and_channel[segments[utt][0]][1]
            )

            graph = lang.compile_training_graph(targets, log_prob.device)

            lattice = get_lattice(
                log_prob=log_prob,
                log_prob_len=log_prob_len,
                decoding_graph=graph,
            )

            best_path = k2.shortest_path(lattice, use_double_scores=True)
            labels = best_path.labels.numpy()
            aux_labels = best_path.aux_labels.numpy()
            logging.info("Best path labels: " + str(labels))
            logging.info("Best path aux labels: " + str(aux_labels))
            writer(utt, labels)
            word_writer(utt, aux_labels)
            labels = labels.tolist()
            aux_labels = aux_labels.tolist()
            assert len(labels) == len(aux_labels)

            p = 0
            aligns = []
            e = []
            while p < len(labels):
                if not aux_labels[p]:
                    if e:
                        if labels[p] and labels[p] not in ignore_labels:
                            e[2] = p + 1
                else:
                    if e:
                        aligns.append(e)
                    e = [aux_labels[p], p, p + 1]
                p += 1
            for e in aligns:
                ctm += f"{utt} {channel} {e[1]*args.frame_shift:.3f} {args.frame_shift*(e[2]-e[1]):.3f} {lang.idx2word[e[0]]}\n"
            count += 1
        ctm_file.write(ctm)
        logging.info(f"Processed {count} utterances")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(
        description="Align the log-probabilities with the text using the K2 library",
    )
    parser.add_argument("lang_dir", type=str, help="The language directory")
    parser.add_argument(
        "log_prob_scp", type=str, help="The log-probabilities in Kaldi scp format"
    )
    parser.add_argument(
        "text",
        type=str,
        help=(
            "The text in Kaldi format with the utterance ID "
            "and the text separated by a space in each line"
        ),
    )
    parser.add_argument(
        "--output_ctm_path",
        type=str,
        required=True,
        help="The path to output the ctm file",
    )
    parser.add_argument(
        "--output_align_path",
        type=str,
        required=True,
        help="The path to output the state-level alignments",
    )
    parser.add_argument(
        "--output_word_align_path",
        type=str,
        required=True,
        help="The path to output the word-level alignments",
    )
    parser.add_argument(
        "--frame_shift",
        type=float,
        required=True,
        help="The frame shift in seconds at the output end of the model.",
    )
    parser.add_argument("--segments", type=str, default=None, help="The segments file")
    parser.add_argument(
        "--reco2file_and_channel",
        type=str,
        default=None,
        help="The reco2file_and_channel file",
    )
    parser.add_argument(
        "--ignore_labels",
        type=str,
        default="",
        help="The input labels to ignore when generating the ctm file. The input should be a sequence of input label ids separated by a comma. For example, '0,1,2' will ignore the labels 0, 1, and 2. Default is an empty string. The ignored labels will be treated as silence.",
    )

    args = parser.parse_args()
    main(args)
