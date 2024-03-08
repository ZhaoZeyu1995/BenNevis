#!/bin/bash
# Authors:
#  * Zeyu Zhao (The University of Edinburgh) 2024

. ./env.sh || exit 1
. ./path.sh || exit 1

nj=16
ngpu=4
topos="ctc mmictc mmictc-1 2state 2state-1 3state-skip 3state-skip-1 3state-skip-2"
model="wav2vec2"

# data
datadir=$LOCAL_HOME/data/librispeech
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11

stage=0
stop_stage=7

train_set=train_960
dev_set=dev_clean
recog_sets="dev_clean dev_other test_clean test_other"

acwts="0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4"
lms="nolm tgsmall"

. ./utils/parse_options.sh

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Download the data
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "$0: stage -1: Data Download"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        local/download_and_untar.sh ${datadir} ${data_url} ${part}
    done
fi

# Data preparation
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "$0: stage 0: data preparation"
    for part in dev-clean test-clean dev-other test-other train-clean-100 train-clean-360 train-other-500; do
        local/data_prep.sh $datadir/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
    done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "$0: stage 1: dictionary preparation"
    # download the LM resources
    local/download_lm.sh $lm_url data/local/lm

    local/prepare_dict.sh data/local/lm data/local/dict_train data/local/dict

    utils/prepare_lang.sh --position-dependent-phones false --sil_prob 0.0 data/local/dict_train \
        "<UNK>" data/local/lang_tr_tmp data/lang_tr
    utils/prepare_lang.sh --position-dependent-phones false --sil_prob 0.0 data/local/dict \
        "<UNK>" data/local/lang_tmp data/lang

    local/format_lms.sh --src-dir data/lang data/local/lm
    run/prepare_nolm.sh data/lang data/lang_test_nolm
fi


# Feature extraction
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    for x in train_clean_100 train_clean_360 train_other_500 dev_clean dev_other test_clean test_other; do
        steps/make_fbank_pitch.sh --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} fbank
        steps/compute_cmvn_stats.sh data/${x} exp/compute_cmvn_stats/${x} cmvn || exit 1;
        utils/fix_data_dir.sh data/${x}
        run/apply_cmvn.sh --nj ${nj} data/${x}
    done
    utils/combine_data.sh --extra-files "feats.cmvn.scp" data/train_460 data/train_clean_100 data/train_clean_360
    utils/combine_data.sh --extra-files "feats.cmvn.scp" data/train_960 data/train_460 data/train_other_500
fi

# Topology preparation
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    for topo in ${topos}; do
        run/prepare_topo.sh --topo ${topo} data/lang_tr data/lang_${topo} || exit 1;
    done
fi

# Graph preparation
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    for topo in ${topos}; do
        for lm in $lms; do
            ./run/prepare_graph.sh --topo ${topo} data/lang_test_${lm} data/local/tmp data/lang_test_${lm}_${topo} || exit 1;
        done
    done
fi

# Model training
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    for topo in ${topos}; do
        if [ $topo == "ctc" ]; then
            torchrun --standalone --nproc_per_node=${ngpu} \
                run/train.py \
                model=${model} \
                opts=${model} \
                data.lang=data/lang_${topo} \
                data.train_ds=data/$train_set \
                data.valid_ds=data/$dev_set \
                loss.kwargs.use_den=false \
                logger.name=${model}-${topo} \
                hydra.run.dir=exp/${model}-$topo || exit 1;
        else
            torchrun --standalone --nproc_per_node=${ngpu} \
                run/train.py \
                model=${model} \
                opts=${model} \
                data.lang=data/lang_${topo} \
                data.train_ds=data/$train_set \
                data.valid_ds=data/$dev_set \
                logger.name=${model}-${topo} \
                hydra.run.dir=exp/${model}-${topo} || exit 1;
        fi
    done
fi

# Prediction
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    for topo in ${topos}; do
        for x in ${recog_sets}; do
            ./run/predict.sh --ngpu ${ngpu} \
                data/${x} data/lang_test_nolm_${topo} \
                exp/${model}-${topo}/checkpoints/best.pt exp/${model}-${topo}/pred_${x} || exit 1;
        done
    done
fi

# Decoding
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    for topo in ${topos}; do
        for x in ${recog_sets}; do
            for lm in $lms; do
                if [ $lm == "nolm" ]; then
                    ./run/decode_faster.sh --nj ${nj} \
                        data/${x} data/lang_test_nolm_${topo} \
                        exp/${model}-${topo}/pred_${x} exp/${model}-${topo}/dec_nolm_${x} || exit 1;
                else
                    for acwt in ${acwts}; do
                        ./run/decode_faster.sh --nj ${nj} --acoustic_scale $acwt \
                            data/${x} data/lang_test_${lm}_${topo} \
                            exp/${model}-${topo}/pred_${x} exp/${model}-${topo}/dec_${lm}_${x} || exit 1;
                    done
                fi
            done
        done
    done
fi

# Align with the ground truth
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    for topo in ${topos}; do
        for x in ${recog_sets}; do
            run/align.sh data/${x} data/lang_test_nolm_${topo} exp/${model}-${topo}/pred_${x} || exit 1;
        done
    done
fi

# Align with the decoding results
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    for topo in ${topos}; do
        for x in ${recog_sets}; do
            for lm in $lms; do
                if [ $lm == "nolm" ]; then
                    run/align.sh --nj ${nj} data/${x} data/lang_test_nolm_${topo} exp/${model}-${topo}/pred_${x} exp/${model}-${topo}/dec_nolm_${x}/aw_1.0-ma_5000-bm_32 || exit 1;
                else
                    for acwt in ${acwts}; do
                        run/align.sh --nj ${nj} data/${x} data/lang_test_${lm}_${topo} exp/${model}-${topo}/pred_${x} exp/${model}-${topo}/dec_${lm}_${x}/aw_${acwt}-ma_5000-bm_32 || exit 1;
                    done
                fi
            done
        done
    done
fi
