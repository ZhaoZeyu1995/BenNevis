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

. ./utils/parse_options.sh

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

for file in L_disambig.fst G.fst; do
    if [ ! -f $src/$file ]; then
        echo "$0: $src/$file is missing!" || exit 1;
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
run/prepare_topo.sh --topo ${topo} $src $des || exit 1;

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
