#!/bin/bash
# This script copies a language directory and creates a trivial word-loop G.fst in the new directory.
# This is useful to get decoding results without a language model.
# Usage:
#  run/prepare_nolm.sh <src-dir> <des-dir>
# e.g.:
#  run/prepare_nolm.sh data/lang_test_bg data/lang_test_nolm
# Authors:
#  * Zeyu Zhao (The University of Edinburgh) 2024

. ./path.sh || exit 1

if [ $# != 2 ]; then
  echo "Usage: run/prepare_nolm.sh <src-dir> <des-dir>"
  echo "e.g.:"
  echo " run/prepare_nolm.sh data/lang_test_bg data/lang_test_nolm"
  echo " where <src-dir> is the source language directory, and <des-dir> is the destination language directory."
  echo " Please make sure that <des-dir> does not exist already."
  exit 1
fi

src=$1
des=$2

if [ -d $des ]; then
  echo "$0: $des already exists, please check and remove it manually"
  exit 1
fi

cp -r $src $des

trivial_gfst.py $des/words.txt | fstcompile --isymbols=$des/words.txt --osymbols=$des/words.txt | fstarcsort --sort_type=ilabel > $des/G.fst

