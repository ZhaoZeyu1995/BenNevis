#!/bin/bash
# A shim over run/apply_cmvn.py, kept so that a recipe's existing command
# line works unchanged.
#
# The shell version is gone. It sourced Kaldi's utils/parse_options.sh to
# read its own flags, split the data with utils/split_data.sh, ran
# apply-cmvn under run.pl once per job and cat'ed the per-job scp files
# back together -- four Kaldi dependencies for arithmetic that
# BenNevis.utils.cmvn has been doing bit-exactly since the first stage of
# the migration.
#
# Deliberately does NOT source path.sh: the point is that this stage needs
# no Kaldi on PATH.
#
# --nj is gone rather than ignored, and the measurement is in
# run/apply_cmvn.py's docstring: the whole of mini-librispeech's
# train_clean_5 normalises and writes in under a second, so a job count
# could only ever go wrong. A recipe still passing it gets an argument
# error, which is the loud failure rather than the quiet one.
#
# Usage:
# run/apply_cmvn.sh [--norm-vars true|false] [--norm-means true|false] \
#     <data-dir>
# Authors:
#  * Zeyu Zhao (The University of Edinburgh) 2024

echo "$0 $@"  # Print the command line for logging

exec python3 "$(dirname "$0")/apply_cmvn.py" "$@"
