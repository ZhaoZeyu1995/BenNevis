#!/bin/bash
# Apply CMVN for a data directory
# This script assumes that there is feats.scp and cmvn.scp in the data directory
# and creates feats.cmvn.scp in the data directory, which will be loaded by BenNevis.
# Usage:
# run/apply_cmvn.sh [options] <data-dir>
# e.g.:
# run/apply_cmvn.sh data/train
# Authors:
#   * Zeyu Zhao (The University of Edinburgh) 2024

. ./path.sh || exit 1;

echo "$0 $@"  # Print the command line for logging

nj=4
norm_vars=true

. ./utils/parse_options.sh

if [ $# != 1 ]; then
    echo "Usage: apply_cmvn.sh [options] <data-dir>"
    echo "Options:"
    echo "  --nj <nj>                 # default: 4, the number of jobs."
    echo "  --norm-vars <true|false>  # default: true, whether to normalize the variance."
    echo "e.g.:"
    echo " apply_cmvn.sh --nj 4 --norm-vars true data/train"
    exit 1
fi

data=$1
num_spk=$(cat "$data/spk2utt" | wc -l)

if [ $num_spk -lt $nj ]; then
    echo "$0: Warning: The number of speakers is small than the number of jobs."
    echo "$0: Changing the number of jobs to the number of speakers automatically."
    nj=$num_spk
fi

./utils/split_data.sh $data $nj || exit 1;

abs_data="$PWD/$data"
logdir=$data/split$nj/log

run.pl JOB=1:$nj $logdir/apply-cmvn.JOB.log \
    apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data/split${nj}/JOB/utt2spk \
    scp:$data/split${nj}/JOB/cmvn.scp \
    scp:$data/split${nj}/JOB/feats.scp \
    ark,scp:$abs_data/split${nj}/JOB/feats.cmvn.ark,$data/split${nj}/JOB/feats.cmvn.scp

for i in $(seq $nj); do
    cat $data/split${nj}/${i}/feats.cmvn.scp
done > $data/feats.cmvn.scp
