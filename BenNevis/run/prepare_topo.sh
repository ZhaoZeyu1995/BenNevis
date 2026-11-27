#!/bin/bash
# A shim over run/prepare_topo.py, kept so that a recipe's existing command
# line works unchanged.
#
# The shell version of this script is gone. It sourced Kaldi's
# utils/parse_options.sh to read its own flags -- and utils/ is a symlink
# into tools/kaldi, so it could not parse --topo without a Kaldi checkout --
# and it ran fstcompile unconditionally for two FSTs that nothing in the
# Python lane reads. Stage 3 therefore needed Kaldi for an argument parser
# and an artefact, long after everything it produces had become Python.
#
# Deliberately does NOT source path.sh: the point is that this stage needs
# no Kaldi on PATH. run/prepare_topo.py finds OpenFST itself when it is
# there and says so when it is not.
#
# Usage:
# run/prepare_topo.sh --topo [topo] [--lexicon <lexicon.txt>] \
#     [--sil-prob P] [--position-dependent-phones true|false] \
#     [--compile-fsts auto|true|false] <src_lang_dir> <des_lang_dir>
# Authors:
#  * Zeyu Zhao (The University of Edinburgh) 2024

echo "$0 $@"  # Print the command line for logging

exec python3 "$(dirname "$0")/prepare_topo.py" "$@"
