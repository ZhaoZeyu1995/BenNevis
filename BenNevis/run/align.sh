#!/bin/bash
# This is a wrapper script that conducts aligning with run/align.py.
# Usage:
# run/align.sh [options] <data_dir> <lang_dir> <predict_dir> [<decode_dir>]
# e.g.:
# run/align.sh data/test data/lang_test_tg_topo exp/model-topo/pred_test exp/model-topo/decode_test
# Authors:
# * Zeyu Zhao (The University of Edinburgh) 2024

. ./path.sh || exit 1
. ./env.sh || exit 1

echo "$0 $@"  # Print the command line for logging

nj=4
frame_shift=0.02

. ./utils/parse_options.sh

if [ $# -ne 4 -a $# -ne 3 ]; then
    echo "Usage: run/align.sh [options] <data_dir> <lang_dir> <predict_dir> [<decode_dir>]"
    echo "Options:"
    echo "  --nj <nj>                # default: 4, the number of jobs."
    echo "  --frame_shift <float>    # default: 0.02, the frame shift in seconds at the output of the neural network."
    echo "e.g.:"
    echo " run/align.sh data/test data/lang_test_tg_topo exp/model-topo/pred_test exp/model-topo/decode_test"
    exit 1
fi

data_dir=$1
lang_dir=$2
predict_dir=$3
decode_dir=$4

# check whether align to the ground truth or the decoding results
if [ -z $decode_dir ]; then
    echo "$0: The <decode_dir> is not specified."
    echo "$0: The $data_dir/text will be used as hyp. Please make sure this is what you want."
    text=$data_dir/text
    output_dir=$predict_dir
else
    text=$decode_dir/hyp.wrd.txt
    output_dir=$decode_dir
fi

if [ ! -f $text ]; then
    echo "$0: Cannot find $text!"
    echo "$0: Please make sure the $text exists." && exit 1
fi

# split the $predict_dir/output.scp to $output_dir/split$nj/output.JOB.scp if not exist
if [ -f $predict_dir/output.scp ]; then
    run.pl JOB=1:$nj $output_dir/split$nj/log/split_scp.JOB.log utils/split_scp.pl -j $nj JOB --one-based \
        $predict_dir/output.scp $output_dir/split$nj/output.JOB.scp
else
    echo "$0: Cannot find output.scp in $predict_dir!"
    echo "$0: Please run run/predict.sh first to generate the output.scp." && exit 1
fi

# conduct aligning

if [ ! -d $output_dir/align ]; then
    mkdir -p $output_dir/align
fi

opts="--frame_shift $frame_shift"
if [ -f $data_dir/reco2file_and_channel ]; then
    opts="$opts --reco2file_and_channel $data_dir/reco2file_and_channel"
    if [ -f $data_dir/segments ]; then
        opts="$opts --segments $data_dir/segments"
    else
        echo "$0: Cannot find $data_dir/segments!"
        echo "$0: Please make sure the $data_dir/segments exists when reco2file_and_channel exists." && exit 1
    fi
fi

run.pl JOB=1:$nj $output_dir/align/log/align.JOB.log \
    align.py \
        $opts \
        --output_ctm_path=$output_dir/split$nj/ctm.JOB \
        --output_align_path=$output_dir/align/ali.JOB.ark.gz \
        --output_word_align_path=$output_dir/align/word.ali.JOB.ark.gz \
        $lang_dir \
        $output_dir/split$nj/output.JOB.scp \
        $text

# merge the ctm
for i in $(seq $nj); do
    cat $output_dir/split$nj/ctm.$i
done > $output_dir/align/ctm

# sort the ctm
cat $output_dir/align/ctm | sort > $output_dir/align/tmp && mv $output_dir/align/tmp $output_dir/align/ctm


