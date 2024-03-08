#!/bin/bash

. ./env.sh || exit 1
. ./path.sh || exit 1

nj=4
ngpu=2
topos="ctc mmictc mmictc-1 2state 2state-1 3state-skip 3state-skip-1 3state-skip-2"

timit=/group/corporapublic/timit/original
trans_type=char

stage=0
stop_stage=100

train_set=train
dev_set=dev
recog_sets="dev test"

. ./utils/parse_options.sh

# Data preparation
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    local/timit_data_prep.sh ${timit} ${trans_type} || exit 1
    local/timit_format_data.sh
    local/prepare_dict.sh
    utils/prepare_lang.sh --position-dependent-phones false --sil_prob 0.0 data/local/dict "<UNK>" data/local/lang_tmp data/lang
    run/prepare_nolm.sh data/lang data/lang_test_nolm
fi


# Feature extraction
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    for x in train dev test; do
        steps/make_fbank_pitch.sh --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} data/${x}/fbank
        steps/compute_cmvn_stats.sh data/${x} exp/compute_cmvn_stats/${x} data/${x}/cmvn || exit 1;
        utils/fix_data_dir.sh data/${x}
        run/apply_cmvn.sh data/${x}
    done
fi

# Topology preparation
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    for topo in ${topos}; do
        run/prepare_topo.sh --topo ${topo} data/lang data/lang_${topo} || exit 1;
    done
fi

# Graph preparation
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    for topo in ${topos}; do
        ./run/prepare_graph.sh --topo ${topo} data/lang_test_nolm data/local/tmp data/lang_test_nolm_${topo} || exit 1;
    done
fi

# Model training
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    for topo in ${topos}; do
        if [ $topo == "ctc" ]; then
            torchrun --standalone --nproc_per_node=${ngpu} \
                run/train.py \
                data.lang=data/lang_${topo} \
                data.train_ds=data/$train_set \
                data.valid_ds=data/$dev_set \
                loss.kwargs.use_den=false \
                logger.name=rnnp-${topo} \
                hydra.run.dir=exp/rnnp-$topo || exit 1;
        else
            torchrun --standalone --nproc_per_node=${ngpu} \
                run/train.py \
                data.lang=data/lang_${topo} \
                data.train_ds=data/$train_set \
                data.valid_ds=data/$dev_set \
                logger.name=rnnp-${topo} \
                hydra.run.dir=exp/rnnp-${topo} || exit 1;
        fi
    done
fi

# Prediction
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    for topo in ${topos}; do
        for x in ${recog_sets}; do
            ./run/predict.sh --ngpu ${ngpu} \
                data/${x} data/lang_test_nolm_${topo} \
                exp/rnnp-${topo}/checkpoints/best.pt exp/rnnp-${topo}/pred_${x} || exit 1;
        done
    done
fi

# Decoding
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    for topo in ${topos}; do
        for x in ${recog_sets}; do
            ./run/decode_faster.sh --nj ${nj} \
                data/${x} data/lang_test_nolm_${topo} \
                exp/rnnp-${topo}/pred_${x} exp/rnnp-${topo}/dec_nolm_${x} || exit 1;
        done
    done
fi

# Align with the ground truth
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    for topo in ${topos}; do
        for x in ${recog_sets}; do
            run/align.sh data/${x} data/lang_test_nolm_${topo} exp/rnnp-${topo}/pred_${x} || exit 1;
        done
    done
fi

# Align with the decoding results
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    for topo in ${topos}; do
        for x in ${recog_sets}; do
            run/align.sh data/${x} data/lang_test_nolm_${topo} exp/rnnp-${topo}/pred_${x} exp/rnnp-${topo}/dec_nolm_${x}/aw_1.0-ma_5000-bm_32 || exit 1;
        done
    done
fi
