#!/bin/bash 
# This script is used to prepare the decoding graph for decoding.
# It first copies the source language directory to the destination directory.
# Then it generates several files by running run/prepare_topo.sh in the destination directory.
# Finally, it generates the decoding graph, TLG.fst, in the destination directory.
# Usage:
# run/prepare_graph.sh --topo [topo] [options] <srcdir> <tmpdir> <desdir>
# e.g.:
# run/prepare_graph.sh --topo mmictc data/lang_tg data/local/lang_tg_mmictc_tmp data/lang_tg_mmictc
# Authors:
#  * Zeyu Zhao (The University of Edinburgh) 2024

. ./path.sh || exit 1

echo "$0 $@"

topo=
nondeterministic=false

. ./run/parse_options.sh

if [ -z $topo ]; then
    echo "$0 --topo cannot be blank!"
    echo "$0: --topo must be ctc|mmictc|mmictc-1|2state|2state-1|3state-skip|3state-skip-1|3state-skip-2"
    exit 1
fi

if [ $# != 3 ]; then
    echo "Usage: run/prepare_graph.sh --topo [topo] [options] <srcdir> <tmpdir> <desdir>"
    echo "  --topo: ctc|mmictc|mmictc-1|2state|2state-1|3state-skip|3state-skip-1|3state-skip-2 # The topology"
    echo "  --nondeterministic: true|false # Whether the topology is nondeterministic or not. Default: false (deterministic)"
    echo "  Note that currently, there is one topology, mmictc-1, which is nondeterministic. The rest are deterministic."
    echo "  <srcdir>: The source directory containing L_disambig.fst and G.fst"
    echo "  <tmpdir>: The temporary directory to store the intermediate files"
    echo "  <desdir>: The destination directory to store the final graphs"
    echo "e.g. run/prepare_graph.sh --topo mmictc data/lang_tg data/local/lang_tg_mmictc_tmp data/lang_tg_mmictc"
    exit 1
fi

src=$1
tmp=$2
des=$3

# `echo "..." || exit 1` was what this said, which never exits: the `||`
# binds to echo, echo always succeeds, and the script carried on to fail
# inside fsttablecompose with something unrelated. Same family as the
# nightly's `grep | awk` under pipefail and the G.fst gate that skipped
# unconditionally -- a shell detail deciding whether a check means
# anything. It matters more now that L_disambig.fst is written by
# `BenNevis.lm prepare-lang`, which skips it when OpenFST is absent.
for file in L_disambig.fst G.fst; do
    if [ ! -f $src/$file ]; then
        echo "$0: $src/$file is missing." >&2
        if [ "$file" = "L_disambig.fst" ]; then
            echo "$0: stage 1 writes it with 'python3 -m BenNevis.lm prepare-lang'," >&2
            echo "$0: which skips it when OpenFST is not on PATH. This stage is" >&2
            echo "$0: Kaldi's decoding lane and needs OpenFST; pass" >&2
            echo "$0: --compile-fsts true there to make that a loud failure." >&2
        fi
        exit 1
    fi
done

if [ -d $des ]; then
    echo "$0: $des has already existed. Removing it..."
    rm -rf $des
fi

if [ -d $tmp ]; then
    echo "$0: $tmp has already existed. Removing it..."
    rm -rf $tmp
fi
mkdir -p ${tmp}

echo "$0: Making T.fst in ${des}..."
# --compile-fsts true because this script composes against $des/T.fst,
# where the Python lane reads neither T.fst nor k2/T.fst and lets stage 3
# skip them when OpenFST is absent.
run/prepare_topo.sh --compile-fsts true --topo ${topo} $src $des || exit 1;

if [ ! -f $des/T.fst ]; then
    echo "$0: $des/T.fst was not written, so TLG.fst cannot be composed." >&2
    echo "$0: This stage is Kaldi's decoding lane and needs OpenFST." >&2
    exit 1
fi

echo "$0: Preparing decoding graph TLG.fst in ${des}..."
fsttablecompose $des/L_disambig.fst $des/G.fst | fstdeterminizestar --use-log=true | \
  fstminimizeencoded | fstarcsort --sort_type=ilabel > $tmp/LG.fst || exit 1;

if [[ $topo == 'mmictc-1' ]]; then
    nondeterministic=true
else
    nondeterministic=false
fi

if [[ $nondeterministic == 'false' ]]; then 
    fsttablecompose $des/T.fst $tmp/LG.fst | fstdeterminizestar --use-log=true | \
       fstrmsymbols $des/disambig.int | fstrmepslocal | fstminimizeencoded | fstarcsort --sort_type=ilabel > $des/TLG.fst || exit 1;
else
    fsttablecompose $des/T.fst $tmp/LG.fst | \
       fstrmsymbols $des/disambig.int | fstrmepslocal | fstminimizeencoded | fstarcsort --sort_type=ilabel > $des/TLG.fst || exit 1;
fi

fsttablecompose $des/L_disambig.fst $des/G.fst | fstdeterminizestar --use-log=true | \
  fstrmsymbols $des/phones/disambig.int | fstrmepslocal | fstminimizeencoded | fstarcsort --sort_type=ilabel > $des/LG.fst || exit 1;

echo "$0: Removing $tmp..."
rm -rf $tmp
echo "$0: Done!"
