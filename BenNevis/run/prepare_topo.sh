#!/bin/bash
# This script copies the source lang directory to the destination lang directory and prepares the following files.
# 1. tokens.txt, the input symbol table for T.fst, which contains <eps>
# 2. disambig.int, the ids of the disambiguation symbols in tokens_disambig.txt, used when composing with L_disambig.fst
# 3. tokens_disambig.txt, the input symbol table for T.fst, which contains <eps> and the disambiguation symbols
# 4. T.fst, the topo FST for decoding; its input symbols are tokens_disambig.txt and its output symbols are phones.txt
# 5. k2/T.fst, the topo FST for k2 training; its input symbols are k2/tokens.txt, which contains neither <eps> nor the
#    disambiguation symbols, so it matches the neural network's outputs directly
# 6. k2/tokens.txt, the input symbol table for k2/T.fst
# 7. k2/phones.txt, the output symbol table for k2/T.fst, without the disambiguation symbols
# 8. k2/topo.yaml, the topology specification, copied in so that `Lang` can build the training FSA in Python
#    rather than reading k2/T.fst back with `fstprint`. A copy rather than a reference, so that a language
#    directory stays reproducible when the bundled specifications change.
#
# The topology itself, and the token inventory derived from it, come from
# BenNevis/topos/specs/<topo>.yaml. This script used to carry one branch per
# topology, each repeating the same awk pipelines, and each calling one of
# fifteen generator scripts; all of that now lives in the spec.
#
# Usage:
# run/prepare_topo.sh --topo [topo] <src_lang_dir> <des_lang_dir>
# Run `python3 -m BenNevis.topos list` for the available topologies.
# e.g.
# run/prepare_topo.sh --topo mmictc data/lang data/lang_mmictc
# Authors:
#  * Zeyu Zhao (The University of Edinburgh) 2024

. ./path.sh || exit 1

echo "$0 $@"  # Print the command line for logging

topo=

. ./utils/parse_options.sh

if [ -z $topo ]; then
    echo "--topo cannot be blank!"
    echo "$0: Supported topologies: $(python3 -m BenNevis.topos list | tr '\n' ' ')"
    exit 1
fi

if [ $# != 2 ]; then
    echo "Usage: run/prepare_topo.sh --topo [topo] <src_lang_dir> <des_lang_dir>"
    echo "  --topo: $(python3 -m BenNevis.topos list | tr '\n' ' ')"
    echo "e.g. run/prepare_topo.sh --topo mmictc data/lang data/lang_mmictc"
    exit 1
fi

src=$1
des=$2

# Fail before touching anything if the topology does not exist, rather than
# part way through writing the destination directory.
if ! python3 -m BenNevis.topos describe $topo > /dev/null; then
    echo "$0: Unknown topology: $topo"
    echo "$0: Supported topologies: $(python3 -m BenNevis.topos list | tr '\n' ' ')"
    exit 1
fi

echo "$0: Using $topo topology"

if [ -d $des ]; then
    echo "$0: $des already exists, removing it..."
    rm -rf $des
fi
cp -r $src $des
mkdir -p $des/k2

# Symbol tables. These were built with grep and awk, once per topology, which
# meant the token inventory and the FST were defined in two different places;
# both come from the spec now. The grep for disambiguation symbols was also
# subtly dialect-dependent: `grep "#[^\d]"` only selects #0 where \d inside a
# bracket expression means the literal characters \ and d, which is POSIX but
# not what ugrep or grep -P do.
python3 -m BenNevis.topos tokens   $topo $des/phones.txt                          > $des/tokens_disambig.txt
python3 -m BenNevis.topos tokens   $topo $des/phones.txt --no-disambig            > $des/tokens.txt
python3 -m BenNevis.topos tokens   $topo $des/phones.txt --no-eps --no-disambig   > $des/k2/tokens.txt
python3 -m BenNevis.topos disambig $topo $des/phones.txt                          > $des/disambig.int
python3 -m BenNevis.topos phones         $des/phones.txt --no-disambig            > $des/k2/phones.txt

# T.fst, for decoding.
python3 -m BenNevis.topos fst $topo $des/phones.txt |\
    fstcompile --isymbols=$des/tokens_disambig.txt --osymbols=$des/phones.txt \
    --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > $des/T.fst

# k2/T.fst, for training. The k2 variant is epsilon-free on the input side by
# construction, so unlike the previous ctc branch there is nothing for
# fstrmepsilon to remove and it can be compiled against k2/tokens.txt directly.
#
# Training no longer reads this file: `Lang` builds the FSA from k2/topo.yaml
# below. It is still written because it is a useful artefact to inspect with
# the OpenFST tools, and because a checkout that has one but no specification
# must stay loadable.
python3 -m BenNevis.topos fst $topo $des/k2/phones.txt --variant k2 |\
    fstcompile --isymbols=$des/k2/tokens.txt --osymbols=$des/k2/phones.txt |\
    fstarcsort --sort_type=olabel > $des/k2/T.fst

# The specification itself, which is what `Lang` expands at training time.
python3 -m BenNevis.topos spec $topo > $des/k2/topo.yaml
