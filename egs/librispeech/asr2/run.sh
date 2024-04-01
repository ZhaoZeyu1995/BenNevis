#!/bin/bash
# Authors:
#  * Zeyu Zhao (The University of Edinburgh) 2024

. ./env.sh || exit 1
. ./path.sh || exit 1

nj=16
ngpu=4
topos="ctc mmictc mmictc-1 2state 2state-1 3state-skip 3state-skip-1 3state-skip-2"

# data
datadir=$LOCAL_HOME/data/librispeech
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11

stage=0
stop_stage=7

# tokens related
mtype="bpe"  # bpe or unigram
mtokens=1000  # number of tokens

train_set=train_960
dev_set=dev_clean
recog_sets="dev_clean dev_other test_clean test_other"
model="wav2vec2.large.lv60k"
opts="wav2vec2"
hydra_opts=""

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

    local/prepare_dict.sh --mtype ${mtype} --mtokens ${mtokens} \
        data/local/lm data/local/dict_train_${mtype}_${mtokens} data/local/dict_${mtype}_${mtokens}

    utils/prepare_lang.sh --position-dependent-phones false --sil_prob 0.0 data/local/dict_train_${mtype}_${mtokens} \
        "<UNK>" data/local/lang_tr_${mtype}_${mtokens}_tmp data/lang_tr_${mtype}_${mtokens}
    utils/prepare_lang.sh --position-dependent-phones false --sil_prob 0.0 data/local/dict_${mtype}_${mtokens} \
        "<UNK>" data/local/lang_${mtype}_${mtokens}_tmp data/lang_${mtype}_${mtokens}

    local/format_lms.sh --src-dir data/lang_${mtype}_${mtokens} data/local/lm
    run/prepare_nolm.sh data/lang_${mtype}_${mtokens} data/lang_${mtype}_${mtokens}_test_nolm
fi


# Feature extraction
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    for x in train_clean_100 train_clean_360 train_other_500 dev_clean dev_other test_clean test_other; do
        steps/make_fbank.sh --nj ${nj} --write_utt2num_frames true \
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
        run/prepare_topo.sh --topo ${topo} data/lang_tr_${mtype}_${mtokens} data/lang_${mtype}_${mtokens}_${topo} || exit 1;
    done
fi

# Graph preparation
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    for topo in ${topos}; do
        for lm in $lms; do
            ./run/prepare_graph.sh --topo ${topo} data/lang_${mtype}_${mtokens}_test_${lm} data/local/tmp data/lang_${mtype}_${mtokens}_test_${lm}_${topo} || exit 1;
        done
    done
fi

# Model training
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    for topo in ${topos}; do
        if [ $topo == "ctc" ]; then
            torchrun --standalone --nproc_per_node=${ngpu} \
                run/train.py \
                data.lang=data/lang_${mtype}_${mtokens}_${topo} \
                data.train_ds=data/${train_set} \
                data.valid_ds=data/${dev_set} \
                model=${model} \
                opts=${opts} \
                loss.kwargs.use_den=false \
                logger.name=${model}-${mtype}-${mtokens}-${topo} \
                hydra.run.dir=exp/${model}-${mtype}-${mtokens}-${topo} \
                ${hydra_opts} || exit 1;
        else
            torchrun --standalone --nproc_per_node=${ngpu} \
                run/train.py \
                data.lang=data/lang_${mtype}_${mtokens}_${topo} \
                data.train_ds=data/${train_set} \
                data.valid_ds=data/${dev_set} \
                model=${model} \
                opts=${opts} \
                logger.name=${model}-${mtype}-${mtokens}-${topo} \
                hydra.run.dir=exp/${model}-${mtype}-${mtokens}-${topo} \
                ${hydra_opts} || exit 1;
        fi
    done
fi

# Prediction
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    for topo in ${topos}; do
        for x in ${recog_sets}; do
            ./run/predict.sh --ngpu ${ngpu} \
                data/${x} data/lang_${mtype}_${mtokens}_test_nolm_${topo} \
                exp/${model}-${mtype}-${mtokens}-${topo}/checkpoints/best.pt exp/${model}-${mtype}-${mtokens}-${topo}/pred_${x} || exit 1;
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
                        data/${x} data/lang_${mtype}_${mtokens}_test_nolm_${topo} \
                        exp/${model}-${mtype}-${mtokens}-${topo}/pred_${x} exp/${model}-${mtype}-${mtokens}-${topo}/dec_nolm_${x} || exit 1;
                else
                    for acwt in ${acwts}; do
                        ./run/decode_faster.sh --nj ${nj} --acoustic_scale $acwt \
                            data/${x} data/lang_${mtype}_${mtokens}_test_${lm}_${topo} \
                            exp/${model}-${mtype}-${mtokens}-${topo}/pred_${x} exp/${model}-${mtype}-${mtokens}-${topo}/dec_${lm}_${x} || exit 1;
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
            run/align.sh --nj ${nj} data/${x} data/lang_${mtype}_${mtokens}_test_nolm_${topo} exp/${model}-${mtype}-${mtokens}-${topo}/pred_${x} || exit 1;
        done
    done
fi

# Align with the decoding results
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    for topo in ${topos}; do
        for x in ${recog_sets}; do
            for lm in $lms; do
                if [ $lm == "nolm" ]; then
                    run/align.sh --nj ${nj} data/${x} data/lang_${mtype}_${mtokens}_test_nolm_${topo} \
                        exp/${model}-${mtype}-${mtokens}-${topo}/pred_${x} \
                        exp/${model}-${mtype}-${mtokens}-${topo}/dec_nolm_${x}/aw_1.0-ma_5000-bm_32 || exit 1;
                else
                    for acwt in ${acwts}; do
                        run/align.sh --nj ${nj} data/${x} data/lang_${mtype}_${mtokens}_test_${lm}_${topo} \
                            exp/${model}-${mtype}-${mtokens}-${topo}/pred_${x} \
                            exp/${model}-${mtype}-${mtokens}-${topo}/dec_${lm}_${x}/aw_${acwt}-ma_5000-bm_32 || exit 1;
                    done
                fi
            done
        done
    done
fi
