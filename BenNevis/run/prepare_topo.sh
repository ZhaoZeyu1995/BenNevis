#!/bin/bash
# This script copies the source lang directory to the destination lang directory and prepares the following files.
# 1. tokens.txt, the input symbol table for T.fst, which contains <eps> 
# 2. disambig.txt, the disambig symbols in T.fst. The disambig symbols are in the form of #1, #2, ... and are useful when composing T.fst with L_disambig.fst.
# 3. disambig.int, the integer representation of disambig symbols
# 4. tokens_disambig.txt, the input symbol table for T.fst, which contains <eps> and disambig symbols
# 5. T.fst, the topo FST for training decoding, as its input symols are tokens_disambig.txt and output symbols are phones.txt
# 6. k2/T.fst, the topo FST for k2 training, as its input symbols are k2/tokens.txt, which does not contain <eps> or disambig symbols, and output symbols are k2/phones.txt
# 7. k2/tokens.txt, the input symbol table for k2/T.fst, which does not contain <eps> or disambig symbols
# 8. k2/phones.txt, the output symbol table for k2/T.fst, which does not contain disambig symbols
# Usage:
# run/prepare_topo.sh --topo [topo] <src_lang_dir> <des_lang_dir>
# --topo: ctc|mmictc|mmictc-1|2state|2state-1|3state-skip|3state-skip-1|3state-skip-2
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
    exit 1
fi

if [ $# != 2 ]; then
    echo "Usage: run/prepare_topo.sh --topo [topo] <src_lang_dir> <des_lang_dir>"
    echo "  --topo: ctc|mmictc|mmictc-1|2state|2state-1|3state-skip|3state-skip-1|3state-skip-2"
    echo "e.g. run/prepare_topo.sh --topo mmictc data/lang data/lang_mmictc"
    exit 1
fi

src=$1
des=$2

if [ -d $des ]; then
    echo "$0: $des already exists, removing it..."
    rm -rf $des
fi
cp -r $src $des
mkdir -p $des/k2

case $topo in
    ctc)
        echo "$0: Using ctc topology"
        (echo "<eps>"; echo "<blk>"; cat $des/phones.txt | grep -v "<eps>" | cut -f 1 -d" ") | grep -v "<SIL>" | awk '{print $1 " " NR-1}' > $des/tokens_disambig.txt

        cat $des/tokens_disambig.txt | grep -v "#[^\d]" > $des/tokens.txt
        cat $des/tokens_disambig.txt | grep "#[^\d]" | awk '{print $2}' > $des/disambig.int

        get_token_fst_ctc.py $des/phones.txt |\
            fstcompile --isymbols=$des/tokens_disambig.txt --osymbols=$des/phones.txt \
            --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > $des/T.fst

        # For k2

        cat $des/tokens.txt | grep -v "<eps>" | awk '{print $1 " " NR-1}' > $des/k2/tokens.txt # no <eps>
        cat $des/phones.txt | grep -v "#[^\d]" | awk '{print $1 " " NR-1}' > $des/k2/phones.txt # no disambig symbols

        get_token_fst_ctc.py $des/k2/phones.txt |\
            fstcompile --isymbols=$des/tokens.txt --osymbols=$des/k2/phones.txt \
            --keep_isymbols=false --keep_osymbols=false | fstrmepsilon |\
            fstprint --isymbols=$des/tokens.txt --osymbols=$des/k2/phones.txt |\
            fstcompile --isymbols=$des/k2/tokens.txt --osymbols=$des/k2/phones.txt |\
            fstarcsort --sort_type=olabel > $des/k2/T.fst
        ;;
    mmictc)
        echo "$0: Using mmictc topology"
        (echo "<eps>"; echo "<blk>"; \
            cat $des/phones.txt | grep -v "<eps>" | grep -v "<SIL>" | grep -v "#[^\d]" | awk '{print $1"_0"; print $1"_1"}'; \
            cat $des/phones.txt | grep "#[^\d]") | awk '{print $1 " " NR-1}' > $des/tokens_disambig.txt

        cat $des/tokens_disambig.txt | grep -v "#[^\d]" > $des/tokens.txt
        cat $des/tokens_disambig.txt | grep "#[^\d]" | awk '{print $2}' > $des/disambig.int

        get_token_fst_mmictc.py $des/phones.txt |\
            fstcompile --isymbols=$des/tokens_disambig.txt --osymbols=$des/phones.txt \
            --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > $des/T.fst

        # For k2

        cat $des/tokens.txt | grep -v "<eps>" | awk '{print $1 " " NR-1}' > $des/k2/tokens.txt
        cat $des/phones.txt | grep -v "#[^\d]" | awk '{print $1 " " NR-1}' > $des/k2/phones.txt # no disambig symbols

        get_token_fst_mmictc_k2.py $des/k2/phones.txt |\
            fstcompile --isymbols=$des/k2/tokens.txt --osymbols=$des/k2/phones.txt |\
            fstarcsort --sort_type=olabel > $des/k2/T.fst
        ;;
    mmictc-1)
        echo "$0: Using mmictc-1 topology"
        (echo "<eps>"; echo "<blk>"; \
            cat $des/phones.txt | grep -v "<eps>" | grep -v "<SIL>" | grep -v "#[^\d]" | awk '{print $1"_0"; print $1"_1"}'; \
            cat $des/phones.txt | grep "#[^\d]") | awk '{print $1 " " NR-1}' > $des/tokens_disambig.txt

        cat $des/tokens_disambig.txt | grep -v "#[^\d]" > $des/tokens.txt
        cat $des/tokens_disambig.txt | grep "#[^\d]" | awk '{print $2}' > $des/disambig.int

        get_token_fst_mmictc-1.py $des/phones.txt |\
            fstcompile --isymbols=$des/tokens_disambig.txt --osymbols=$des/phones.txt \
            --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > $des/T.fst

        # For k2

        cat $des/tokens.txt | grep -v "<eps>" | awk '{print $1 " " NR-1}' > $des/k2/tokens.txt
        cat $des/phones.txt | grep -v "#[^\d]" | awk '{print $1 " " NR-1}' > $des/k2/phones.txt # no disambig symbols

        get_token_fst_mmictc-1_k2.py $des/k2/phones.txt |\
            fstcompile --isymbols=$des/k2/tokens.txt --osymbols=$des/k2/phones.txt |\
            fstarcsort --sort_type=olabel > $des/k2/T.fst
        ;;
    2state)
        echo "$0: Using 2state topology"
        (echo "<eps>"; echo "<blk>"; \
            cat $des/phones.txt | grep -v "<eps>" | grep -v "<SIL>" | grep -v "#[^\d]" | awk '{print $1"_0"; print $1"_1"}'; \
            cat $des/phones.txt | grep "#[^\d]") | awk '{print $1 " " NR-1}' > $des/tokens_disambig.txt

        cat $des/tokens_disambig.txt | grep -v "#[^\d]" > $des/tokens.txt
        cat $des/tokens_disambig.txt | grep "#[^\d]" | awk '{print $2}' > $des/disambig.int

        get_token_fst_2state.py $des/phones.txt |\
            fstcompile --isymbols=$des/tokens_disambig.txt --osymbols=$des/phones.txt \
            --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > $des/T.fst

        # For k2

        cat $des/tokens.txt | grep -v "<eps>" | awk '{print $1 " " NR-1}' > $des/k2/tokens.txt
        cat $des/phones.txt | grep -v "#[^\d]" | awk '{print $1 " " NR-1}' > $des/k2/phones.txt # no disambig symbols

        get_token_fst_2state_k2.py $des/k2/phones.txt |\
            fstcompile --isymbols=$des/k2/tokens.txt --osymbols=$des/k2/phones.txt |\
            fstarcsort --sort_type=olabel > $des/k2/T.fst
        ;;
    2state-1)
        echo "$0: Using 2state-1 topology"
        (echo "<eps>"; echo "<blk>"; \
            cat $des/phones.txt | grep -v "<eps>" | grep -v "<SIL>" | grep -v "#[^\d]" | awk '{print $1"_0"; print $1"_1"}'; \
            cat $des/phones.txt | grep "#[^\d]") | awk '{print $1 " " NR-1}' > $des/tokens_disambig.txt

        cat $des/tokens_disambig.txt | grep -v "#[^\d]" > $des/tokens.txt
        cat $des/tokens_disambig.txt | grep "#[^\d]" | awk '{print $2}' > $des/disambig.int

        get_token_fst_2state-1.py $des/phones.txt |\
            fstcompile --isymbols=$des/tokens_disambig.txt --osymbols=$des/phones.txt \
            --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > $des/T.fst

        # For k2

        cat $des/tokens.txt | grep -v "<eps>" | awk '{print $1 " " NR-1}' > $des/k2/tokens.txt
        cat $des/phones.txt | grep -v "#[^\d]" | awk '{print $1 " " NR-1}' > $des/k2/phones.txt # no disambig symbols

        get_token_fst_2state-1_k2.py $des/k2/phones.txt |\
            fstcompile --isymbols=$des/k2/tokens.txt --osymbols=$des/k2/phones.txt |\
            fstarcsort --sort_type=olabel > $des/k2/T.fst
        ;;
    3state-skip)
        echo "$0: Using 3state-skip topology"
        (echo "<eps>"; echo "<blk>"; \
            cat $des/phones.txt | grep -v "<eps>" | grep -v "<SIL>" | grep -v "#[^\d]" | awk '{print $1"_0"; print $1"_1"; print $1"_2"}'; \
            cat $des/phones.txt | grep "#[^\d]") | awk '{print $1 " " NR-1}' > $des/tokens_disambig.txt

        cat $des/tokens_disambig.txt | grep -v "#[^\d]" > $des/tokens.txt
        cat $des/tokens_disambig.txt | grep "#[^\d]" | awk '{print $2}' > $des/disambig.int

        get_token_fst_3state-skip.py $des/phones.txt |\
            fstcompile --isymbols=$des/tokens_disambig.txt --osymbols=$des/phones.txt \
            --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > $des/T.fst

        # For k2

        cat $des/tokens.txt | grep -v "<eps>" | awk '{print $1 " " NR-1}' > $des/k2/tokens.txt
        cat $des/phones.txt | grep -v "#[^\d]" | awk '{print $1 " " NR-1}' > $des/k2/phones.txt # no disambig symbols

        get_token_fst_3state-skip_k2.py $des/k2/phones.txt |\
            fstcompile --isymbols=$des/k2/tokens.txt --osymbols=$des/k2/phones.txt |\
            fstarcsort --sort_type=olabel > $des/k2/T.fst
        ;;
    3state-skip-1)
        echo "$0: Using 3state-skip-1 topology"
        (echo "<eps>"; echo "<blk>"; \
            cat $des/phones.txt | grep -v "<eps>" | grep -v "<SIL>" | grep -v "#[^\d]" | awk '{print $1"_0"; print $1"_1"; print $1"_2"}'; \
            cat $des/phones.txt | grep "#[^\d]") | awk '{print $1 " " NR-1}' > $des/tokens_disambig.txt

        cat $des/tokens_disambig.txt | grep -v "#[^\d]" > $des/tokens.txt
        cat $des/tokens_disambig.txt | grep "#[^\d]" | awk '{print $2}' > $des/disambig.int

        get_token_fst_3state-skip-1.py $des/phones.txt |\
            fstcompile --isymbols=$des/tokens_disambig.txt --osymbols=$des/phones.txt \
            --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > $des/T.fst

        # For k2

        cat $des/tokens.txt | grep -v "<eps>" | awk '{print $1 " " NR-1}' > $des/k2/tokens.txt
        cat $des/phones.txt | grep -v "#[^\d]" | awk '{print $1 " " NR-1}' > $des/k2/phones.txt # no disambig symbols

        get_token_fst_3state-skip-1_k2.py $des/k2/phones.txt |\
            fstcompile --isymbols=$des/k2/tokens.txt --osymbols=$des/k2/phones.txt |\
            fstarcsort --sort_type=olabel > $des/k2/T.fst
        ;;
    3state-skip-2)
        echo "$0: Using 3state-skip-2 topology"
        (echo "<eps>"; echo "<blk>"; \
            cat $des/phones.txt | grep -v "<eps>" | grep -v "<SIL>" | grep -v "#[^\d]" | awk '{print $1"_0"; print $1"_1"; print $1"_2"}'; \
            cat $des/phones.txt | grep "#[^\d]") | awk '{print $1 " " NR-1}' > $des/tokens_disambig.txt

        cat $des/tokens_disambig.txt | grep -v "#[^\d]" > $des/tokens.txt
        cat $des/tokens_disambig.txt | grep "#[^\d]" | awk '{print $2}' > $des/disambig.int

        get_token_fst_3state-skip-2.py $des/phones.txt |\
            fstcompile --isymbols=$des/tokens_disambig.txt --osymbols=$des/phones.txt \
            --keep_isymbols=false --keep_osymbols=false | fstarcsort --sort_type=olabel > $des/T.fst

        # For k2

        cat $des/tokens.txt | grep -v "<eps>" | awk '{print $1 " " NR-1}' > $des/k2/tokens.txt
        cat $des/phones.txt | grep -v "#[^\d]" | awk '{print $1 " " NR-1}' > $des/k2/phones.txt # no disambig symbols

        get_token_fst_3state-skip-2_k2.py $des/k2/phones.txt |\
            fstcompile --isymbols=$des/k2/tokens.txt --osymbols=$des/k2/phones.txt |\
            fstarcsort --sort_type=olabel > $des/k2/T.fst
        ;;
    *)
        echo "$0: Unknown topology: $topo"
        echo "$0: Supported topologies: ctc, mmictc, mmictc-1, 2state, 2state-1, 3state-skip, 3state-skip-1, 3state-skip-2"
        exit 1
        ;;
esac
