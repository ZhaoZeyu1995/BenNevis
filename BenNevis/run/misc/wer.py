#!/usr/bin/env python3
"""
This is a python implementation of the word error rate (WER) calculation.
The algorithm is based on dynamic programming, and the implementation is
accelerated by Numba.
The main feature of this implementation is that the words with parenthesis
are considered as optional, which means that the words with parenthesis
can be ignored or mistakenly recognised in the calculation of the WER.
That said, for a word with parenthesis, the algorithm will allow a deletion
or a substitution of the word without penalty.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""
import numpy as np
import time
from typing import List
from numba import jit
from numba.typed import List as NumbaList
import argparse


@jit(nopython=True)
def wer(hyp: NumbaList[str], ref: NumbaList[str], res: NumbaList[str]):
    len_hyp = len(hyp)
    len_ref = len(ref)
    record = np.full((len_ref + 1, len_hyp + 1), 1e10, dtype=np.int64)
    hist = np.zeros((len_ref + 1, len_hyp + 1, 3), dtype=np.int64)
    # coordinate plus a flag:
    # 0 for deletion,
    # 1 for insertion,
    # 2 for substitution,
    # 3 for match,
    # 4 for optional substitution
    # 5 for optional deletion

    # initialize
    for i in range(len_hyp + 1):
        record[0, i] = i
        if i > 0:
            hist[0, i] = [0, i - 1, 1]  # insertion

    for i in range(1, len_ref + 1):
        if ref[i - 1].startswith("(") and ref[i - 1].endswith(")"):
            record[i, 0] = record[i - 1, 0]
            hist[i, 0] = [i - 1, 0, 5]  # optional deletion
        else:
            record[i, 0] = record[i - 1, 0] + 1
            hist[i, 0] = [i - 1, 0, 0]  # deletion

    # recursion
    for i in range(1, len_ref + 1):
        for j in range(1, len_hyp + 1):
            if ref[i - 1].startswith("(") and ref[i - 1].endswith(")"):
                if (
                    record[i - 1, j] <= record[i, j - 1] + 1
                    and record[i - 1, j] <= record[i - 1, j - 1]
                ):
                    record[i, j] = record[i - 1, j]
                    hist[i, j] = [i - 1, j, 5]  # optional deletion
                elif record[i - 1, j - 1] <= record[i - 1, j]:
                    record[i, j] = record[i - 1, j - 1]
                    hist[i, j] = [i - 1, j - 1, 4]  # optional substitution
                else:
                    record[i, j] = record[i, j - 1] + 1
                    hist[i, j] = [i, j - 1, 1]  # insertion
            else:
                if ref[i - 1] == hyp[j - 1]:
                    if (
                        record[i - 1, j - 1] <= record[i - 1, j] + 1
                        and record[i - 1, j - 1] <= record[i, j - 1] + 1
                    ):
                        record[i, j] = record[i - 1, j - 1]
                        hist[i, j] = [i - 1, j - 1, 3]  # match
                    elif record[i - 1, j] <= record[i, j - 1]:
                        record[i, j] = record[i - 1, j] + 1
                        hist[i, j] = [i - 1, j, 0]  # deletion
                    else:
                        record[i, j] = record[i, j - 1] + 1
                        hist[i, j] = [i, j - 1, 1]  # insertion
                else:
                    if (
                        record[i - 1, j - 1] <= record[i - 1, j]
                        and record[i - 1, j - 1] <= record[i, j - 1]
                    ):
                        record[i, j] = record[i - 1, j - 1] + 1
                        hist[i, j] = [i - 1, j - 1, 2]  # substitution
                    elif record[i - 1, j] <= record[i, j - 1]:
                        record[i, j] = record[i - 1, j] + 1
                        hist[i, j] = [i - 1, j, 0]  # deletion
                    else:
                        record[i, j] = record[i, j - 1] + 1
                        hist[i, j] = [i, j - 1, 1]  # insertion

    # backtrace
    (i, j) = (len_ref, len_hyp)

    def tostr(flag: int):
        if flag == 0:
            return "D"
        elif flag == 1:
            return "I"
        elif flag == 2:
            return "S"
        elif flag == 3:
            return ""
        elif flag == 4:
            return "SO"
        elif flag == 5:
            return "DO"
        else:
            raise ValueError

    while i > 0 or j > 0:
        res.insert(0, tostr(hist[i, j][2]))
        (i, j) = (hist[i, j][0], hist[i, j][1])

    return record[len_ref, len_hyp]


def run(hyp: List[str], ref: List[str]):
    # if hyp is empty, then all should be deletions
    if len(hyp) == 0:
        error = len(ref) - len(
            [x for x in ref if x.startswith("(") and x.endswith(")")]
        )
        res = ["D" if x.startswith("(") and x.endswith(")") else "DO" for x in ref]
        return error, res
    hyp = NumbaList[str](hyp)
    ref = NumbaList[str](ref)
    res = NumbaList[str](["eos"])
    return wer(hyp, ref, res), list(res)[:-1]


def main(hyp_path: str, ref_path: str, format: str, output_path: str):
    assert format in ["text", "trn"], "format should be either 'text' or 'trn'"
    if format == "text":
        hyp_dict = dict()
        with open(hyp_path, "r") as f:
            for line in f:
                lc = line.strip().split()
                uttid = lc[0]
                hyp = lc[1:]
                hyp_dict[uttid] = hyp
        ref_dict = dict()
        with open(ref_path, "r") as f:
            for line in f:
                lc = line.strip().split()
                uttid = lc[0]
                ref = lc[1:]
                assert len(ref) > 0, "empty reference for uttid: {}".format(uttid)
                ref_dict[uttid] = ref
    elif format == "trn":
        hyp_dict = dict()
        with open(hyp_path, "r") as f:
            for line in f:
                lc = line.strip().split()
                uttid = lc[-1]
                hyp = lc[:-1]
                hyp_dict[uttid] = hyp
        ref_dict = dict()
        with open(ref_path, "r") as f:
            for line in f:
                lc = line.strip().split()
                uttid = lc[-1]
                ref = lc[:-1]
                assert len(ref) > 0, "empty reference for uttid: {}".format(uttid)
                ref_dict[uttid] = ref

    f = open(output_path, "w")
    Nerror = 0
    Ndel = 0
    Nins = 0
    Nsub = 0
    NdelOpt = 0
    NsubOpt = 0
    Ntotal = 0
    Ncor = 0
    NSetenceCor = 0
    NSetence = 0
    fc = ""
    for uttid, hyp in hyp_dict.items():
        ref = ref_dict[uttid]
        Ntotal += len(ref) - len([x for x in ref if x.startswith("(") and x.endswith(")")])
        NSetence += 1
        error, res = run(hyp, ref)
        align(hyp, ref, res)
        if error == 0:
            NSetenceCor += 1

        Nerror += error
        Ndel += res.count("D")
        Nins += res.count("I")
        Nsub += res.count("S")
        NdelOpt += res.count("DO")
        NsubOpt += res.count("SO")
        Ncor += res.count("")

        fc += "id: %s\n" % (uttid)
        fc += "Scores: (#C #S #D #I #DO #SO) %d %d %d %d %d %d\n" % (
            res.count(""),
            res.count("S"),
            res.count("D"),
            res.count("I"),
            res.count("DO"),
            res.count("SO"),
        )

        max_lengths = [
            max(len(word1), len(word2), len(word3))
            for word1, word2, word3 in zip(hyp, ref, res)
        ]

        hyp_txt = " ".join([f"{word:<{max_lengths[i]}}" for i, word in enumerate(hyp)])
        ref_txt = " ".join([f"{word:<{max_lengths[i]}}" for i, word in enumerate(ref)])
        res_txt = " ".join([f"{word:<{max_lengths[i]}}" for i, word in enumerate(res)])

        fc += "Hyp: %s\n" % hyp_txt
        fc += "Ref: %s\n" % ref_txt
        fc += "Eva: %s\n" % res_txt
        fc += "\n"

    fc = "\n" + fc
    fc = "SER: %.2f%%\n" % ((NSetence - NSetenceCor) / NSetence * 100) + fc
    fc = "#Sentence: %d\n" % NSetence + fc
    fc = "WER: %.2f%%\n" % (Nerror / Ntotal * 100) + fc
    fc = (
        "RATE: (#C #S #D #I #DO #SO) %.2f%% %.2f%% %.2f%% %.2f%% %.2f%% %.2f%%\n"
        % (
            Ncor / Ntotal * 100,
            Nsub / Ntotal * 100,
            Ndel / Ntotal * 100,
            Nins / Ntotal * 100,
            NdelOpt / Ntotal * 100,
            NsubOpt / Ntotal * 100,
        )
        + fc
    )
    fc = "Total: (#C #S #D #I #DO #SO) %d %d %d %d %d %d\n" % (Ncor, Nsub, Ndel, Nins, NdelOpt, NsubOpt) + fc
    f.write(fc)
    f.close()


def debug():
    # a simple example
    hyp = ["a", "b", "c", "d"]
    ref = ["a", "b", "c", "(e)"]
    start = time.time()
    error, res = run(hyp, ref)
    align(hyp, ref, res)
    max_lengths = [
        max(len(word1), len(word2), len(word3))
        for word1, word2, word3 in zip(hyp, ref, res)
    ]

    formatted_list1 = " ".join(
        [f"{word:<{max_lengths[i]}}" for i, word in enumerate(hyp)]
    )
    formatted_list2 = " ".join(
        [f"{word:<{max_lengths[i]}}" for i, word in enumerate(ref)]
    )
    formatted_list3 = " ".join(
        [f"{word:<{max_lengths[i]}}" for i, word in enumerate(res)]
    )

    print(formatted_list1)
    print(formatted_list2)
    print(formatted_list3)
    print(time.time() - start)

    # a more complex example
    hyp = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    ref = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "k",
        "l",
        "(m)",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
    ]
    start = time.time()
    error, res = run(hyp, ref)
    print(error)
    align(hyp, ref, res)
    max_lengths = [
        max(len(word1), len(word2), len(word3))
        for word1, word2, word3 in zip(hyp, ref, res)
    ]

    formatted_list1 = " ".join(
        [f"{word:<{max_lengths[i]}}" for i, word in enumerate(hyp)]
    )
    formatted_list2 = " ".join(
        [f"{word:<{max_lengths[i]}}" for i, word in enumerate(ref)]
    )
    formatted_list3 = " ".join(
        [f"{word:<{max_lengths[i]}}" for i, word in enumerate(res)]
    )

    print(formatted_list1)
    print(formatted_list2)
    print(formatted_list3)
    print(time.time() - start)

    # a real sentence example
    hyp = ["the", "cat", "is", "on", "the", "mat"]
    ref = ["the", "cat", "is", "(not)", "on", "the", "mat"]
    start = time.time()
    error, res = run(hyp, ref)
    print(error)
    align(hyp, ref, res)
    max_lengths = [
        max(len(word1), len(word2), len(word3))
        for word1, word2, word3 in zip(hyp, ref, res)
    ]

    formatted_list1 = " ".join(
        [f"{word:<{max_lengths[i]}}" for i, word in enumerate(hyp)]
    )
    formatted_list2 = " ".join(
        [f"{word:<{max_lengths[i]}}" for i, word in enumerate(ref)]
    )
    formatted_list3 = " ".join(
        [f"{word:<{max_lengths[i]}}" for i, word in enumerate(res)]
    )

    print(formatted_list1)
    print(formatted_list2)
    print(formatted_list3)
    print(time.time() - start)

    # a real sentence example from wikipedia
    hyp = ["quick", "brown", "jumped", "over", "the", "lazy", "dog"]
    ref = ["(the)", "fast", "brown", "fox", "jumped", "over", "the", "(sleepy)", "dog"]
    start = time.time()
    error, res = run(hyp, ref)
    print(error)
    align(hyp, ref, res)

    max_lengths = [
        max(len(word1), len(word2), len(word3))
        for word1, word2, word3 in zip(hyp, ref, res)
    ]

    formatted_list1 = " ".join(
        [f"{word:<{max_lengths[i]}}" for i, word in enumerate(hyp)]
    )
    formatted_list2 = " ".join(
        [f"{word:<{max_lengths[i]}}" for i, word in enumerate(ref)]
    )
    formatted_list3 = " ".join(
        [f"{word:<{max_lengths[i]}}" for i, word in enumerate(res)]
    )

    print(formatted_list1)
    print(formatted_list2)
    print(formatted_list3)
    print(time.time() - start)


def align(hyp: List[str], ref: List[str], res: List[str]):
    N = len(res)
    for i in range(N):
        item = res[i]
        if item == "D":
            hyp.insert(i, "*" * len(ref[i]))
        elif item == "I":
            ref.insert(i, "*" * len(hyp[i]))
        elif item == "S":
            pass
        elif item == "":
            pass
        elif item == "SO":
            pass
        elif item == "DO":
            hyp.insert(i, "*" * len(ref[i]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        """
        This is a WER calculator where the words with parenthesis
        in the transcriptions are considered as optional.

        Currently, it supports two different formats of the input files:
            * text format: <uttid> <word1> <word2> ...
            * trn format: <word1> <word2> ... <uttid>
        Please make sure to specify the format of the input files,
        and provide the input files with the correct format.
        """
    )
    parser.add_argument("hyp", type=str, help="The path to the hypothesis file.")
    parser.add_argument("ref", type=str, help="The path to the reference file.")
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="trn",
        help="The format of the input files. It should be either 'text' or 'trn'.",
    )
    parser.add_argument("output", type=str, help="The path to the output file.")
    args = parser.parse_args()

    main(args.hyp, args.ref, args.format, args.output)
