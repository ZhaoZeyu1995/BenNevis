# The environment every recipe's stages run in. All seven recipes share
# this one file through a symlink, so a change here changes all of them.
#
# **A missing Kaldi is not fatal any more, and that is the point.** This
# used to end with
#
#     [ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo ... && exit 1
#
# and `exit` inside a sourced file exits the caller -- so `. ./path.sh` at
# the top of run.sh made the *whole recipe* unrunnable without a Kaldi
# checkout, including the stages that no longer need one. Stages 0 to 3
# and 5 are Python now: data preparation, the language directory, the
# features, CMVN, the topologies and training. Hard-failing here was the
# last thing standing between that and a recipe that can train with no
# Kaldi at all.
#
# What still needs Kaldi says so itself, where it is needed and with the
# reason: `prepare_graph.sh` checks for `L_disambig.fst` and `T.fst` and
# names the flag that writes them, `run/prepare_topo.py` says what it
# skipped, and `BenNevis.lm prepare-lang` does the same. That is the
# pattern -- the stage that cannot proceed complains, rather than the
# environment refusing to be set up.
#
# The Kaldi-present behaviour is unchanged, PATH order included:
# BenNevis/bin first, then OpenFST, sctk and Kaldi's utils, then
# whatever was there.
#
# Authors:
#  * Zeyu Zhao (The University of Edinburgh) 2024

MAIN_ROOT=$PWD/../../..
KALDI_ROOT=$MAIN_ROOT/tools/kaldi

if [ -f $KALDI_ROOT/tools/config/common_path.sh ]; then
    export PATH=$MAIN_ROOT/BenNevis/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sctk/bin:$KALDI_ROOT/egs/wsj/s5/utils:$PATH
    export LD_LIBRARY_PATH=$KALDI_ROOT/tools/openfst/lib:$KALDI_ROOT/src/lib:$LD_LIBRARY_PATH
    . $KALDI_ROOT/tools/config/common_path.sh
else
    export PATH=$MAIN_ROOT/BenNevis/bin:$PATH
    echo >&2 "path.sh: no Kaldi under $KALDI_ROOT."
    echo >&2 "path.sh: the Python stages do not need one; the stages that do"
    echo >&2 "path.sh: will say so when they are reached."
fi

export LC_ALL=C

export PYTHONPATH=$MAIN_ROOT:$PYTHONPATH

export OMP_NUM_THREADS=1
export PYTHONIOENCODING=UTF-8
