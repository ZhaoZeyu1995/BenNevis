#!/bin/bash
# Build a test-time language directory with a grammar FST in it.
#
# Replaces `local/format_lms.sh`, which is Kaldi's, for one reason: its
# last step is `utils/validate_lang.pl`, and that validates a **GMM**
# language directory. It requires nine files BenNevis deliberately does
# not write --
#
#   phones/context_indep.txt   phones/roots.txt
#   phones/nonsilence.csl      phones/sets.txt
#   phones/silence.csl         phones/extra_questions.{txt,int}
#   phones/optional_silence.csl
#   phones/disambig.csl        topo
#
# -- all of which only a decision-tree build and an HMM topology read.
# BenNevis trains no GMM and has no HMM topology, so writing them would
# be nine more things to keep correct in order to satisfy a validator
# whose subject we do not have. `python3 -m BenNevis.lm prepare-lang`
# writes what is actually read, and stage 4 failed on the difference.
#
# **yesno could not have shown this**: its `local/prepare_lm.sh` runs
# `arpa2fst` and some `fstinfo` diagnostics and never validates, and
# yesno is the recipe that runs nightly. Same shape as `spk2gender` --
# "nothing in BenNevis reads it" is not "nothing in the recipe reads
# it", and the recipe that runs most often is the one least able to say.
#
# What is lost by not validating, and what covers it instead:
#   * L.fst and L_disambig.fst olabel-sorted -- `BenNevis.utils.openfst`
#     arc-sorts on that side, and `tests/utils/test_openfst.py` pins it.
#   * G.fst ilabel-sorted -- `fstarcsort` below, as Kaldi's own
#     `prepare_nolm.sh` does.
#   * the symbol tables and oov files agreeing -- gated byte-exactly
#     against Kaldi's in `tests/lm/test_prepare_lang.py`.
#   * G.fst being the right function -- `tests/reference/compare_graphs.py`
#     compares it against Kaldi's on sampled word sequences, which is a
#     far stronger check than existence.
#   * the graphs' inputs being present -- `run/prepare_graph.sh` checks.
#
# Only `tgsmall` is built, where Kaldi's script also built `tgmed`. The
# recipe decodes against `tgsmall` alone, so the second was a graph
# nothing read.
#
# Usage:
# run/format_lm.sh --src-dir data/lang_nosp --suffix tgsmall <lm-dir>
# Authors:
#  * Zeyu Zhao (The University of Edinburgh) 2026

. ./path.sh || exit 1

echo "$0 $@"  # Print the command line for logging

src_dir=data/lang
suffix=tgsmall

. ./run/parse_options.sh || exit 1

set -eo pipefail

if [ $# != 1 ]; then
    echo "Usage: run/format_lm.sh [options] <lm-dir>" >&2
    echo "  --src-dir <dir>   # the language directory to copy, default data/lang" >&2
    echo "  --suffix <name>   # which lm_<name>.arpa.gz to read, default tgsmall" >&2
    exit 1
fi

lm_dir=$1
arpa=${lm_dir}/lm_${suffix}.arpa.gz
test=${src_dir}_test_${suffix}

for required in "${src_dir}/words.txt" "${arpa}"; do
    if [ ! -f "${required}" ]; then
        echo "$0: ${required} does not exist" >&2
        exit 1
    fi
done

rm -rf "${test}"
mkdir -p "${test}"
cp -r "${src_dir}"/* "${test}"

gunzip -c "${arpa}" | \
    arpa2fst --disambig-symbol='#0' \
             --read-symbol-table="${test}/words.txt" - - | \
    fstarcsort --sort_type=ilabel > "${test}/G.fst"

if [ ! -s "${test}/G.fst" ]; then
    echo "$0: ${test}/G.fst is empty" >&2
    exit 1
fi

echo "$0: wrote ${test}/G.fst from ${arpa}"
