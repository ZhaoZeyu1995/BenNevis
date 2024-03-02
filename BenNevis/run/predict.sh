#!/bin/bash
# This is a wrapper script that conducts predicting with run/predict.py.
# Usage:
# run/predict.sh [options] <data_dir> <lang_dir> <ckpt_dir> <predict_dir>
# e.g.:
# run/predict.sh data/test data/lang_test_tg_topo exp/model-topo/checkpoints/best.pt exp/model-topo/pred_test
# Authors:
#  * Zeyu Zhao (The University of Edinburgh) 2024

. ./path.sh || exit 1;
. ./env.sh || exit 1

ngpu=2
pin_memory=true
batch_size=4
num_workers=4

. ./utils/parse_options.sh

if [ $# != 4 ]; then
    echo "Usage: run/predict.sh [options] <data_dir> <lang_dir> <ckpt_dir> <predict_dir>"
    echo "Options:"
    echo "  --ngpu <ngpu>               # default: 2, the number of GPUs."
    echo "  --pin-memory <true|false>   # default: true, whether to use pin_memory in DataLoader."
    echo "  --batch-size <batch_size>   # default: 4, the batch size."
    echo "  --num-workers <num_workers> # default: 4, the number of workers in DataLoader."
    echo "e.g.:"
    echo " run/predict.sh data/test data/lang_test_tg_topo exp/model-topo/checkpoints/best.pt exp/model-topo/pred_test"
    exit 1
fi

data_dir=$1
lang_dir=$2
ckpt_dir=$3
predict_dir=$4

opts="--batch_size ${batch_size} --num_workers ${num_workers}"
if [ $pin_memory == "false" ]; then
    opts="${opts} --not_pin_memory"
fi

torchrun --standalone --nproc_per_node=${ngpu} \
    run/predict.py \
    $opts \
    $data_dir \
    $lang_dir \
    $ckpt_dir \
    $predict_dir
