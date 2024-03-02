#!/bin/bash

. ./env.sh || exit 1
. ./path.sh || exit 1

nj=4
ngpu=2
topos="ctc mmictc mmictc-1 2state 2state-1 3state-skip 3state-skip-1 3state-skip-2"

train_yesno=train_yesno
test_base_name=test_yesno

# Download the dataset
if [ ! -d waves_yesno ]; then
  wget http://www.openslr.org/resources/1/waves_yesno.tar.gz || exit 1;
  # was:
  # wget http://sourceforge.net/projects/kaldi/files/waves_yesno.tar.gz || exit 1;
  tar -xvzf waves_yesno.tar.gz || exit 1;
fi

# Data preparation
local/prepare_data.sh waves_yesno
local/prepare_dict.sh
utils/prepare_lang.sh --position-dependent-phones false --sil_prob 0.0 data/local/dict "<UNK>" data/local/lang data/lang
local/prepare_lm.sh

# Feature extraction
fbankdir=fbank
cmvndir=cmvn
for x in train_yesno test_yesno; do 
    steps/make_fbank_pitch.sh --nj ${nj} --write_utt2num_frames true \
        data/${x} exp/make_fbank/${x} data/${x}/${fbankdir}
    steps/compute_cmvn_stats.sh data/${x} exp/compute_cmvn_stats/${x} data/${x}/$cmvndir || exit 1;
    utils/fix_data_dir.sh data/${x}
    run/apply_cmvn.sh data/${x}
done

# Topology preparation
for topo in ${topos}; do
    run/prepare_topo.sh --topo ${topo} data/lang data/lang_${topo} || exit 1;
done

# Graph preparation
for topo in ${topos}; do
    run/prepare_graph.sh --topo $topo data/lang_test_tg data/local/lang_tg_${topo}_tmp data/lang_test_tg_${topo} || exit 1;
done

 #Model training
for topo in ${topos}; do
    if [ $topo == "ctc" ]; then
        torchrun --standalone --nproc_per_node=${ngpu} \
            run/train.py \
            data.lang=data/lang_${topo} \
            data.train_ds=data/train_yesno \
            data.valid_ds=data/test_yesno \
            loss.kwargs.use_den=false \
            logger.name=rnnp-ctc \
            hydra.run.dir=exp/rnnp-${topo} || exit 1;
    else
        torchrun --standalone --nproc_per_node=${ngpu} \
            run/train.py \
            data.lang=data/lang_${topo} \
            data.train_ds=data/train_yesno \
            data.valid_ds=data/test_yesno \
            logger.name=rnnp-${topo} \
            hydra.run.dir=exp/rnnp-${topo} || exit 1;
    fi
done

# Prediction
for topo in ${topos}; do
    run/predict.sh --ngpu ${ngpu} \
        data/test_yesno data/lang_test_tg_${topo} \
        exp/rnnp-${topo}/checkpoints/best.pt exp/rnnp-${topo}/pred_test_yesno || exit 1;
done

# Decoding
for topo in ${topos}; do
    ./run/decode_faster.sh --nj ${nj} \
        data/test_yesno data/lang_test_tg_${topo} \
        exp/rnnp-${topo}/pred_test_yesno exp/rnnp-${topo}/dec_test_yesno || exit 1;
done

# Align with the ground truth
for topo in ${topos}; do
    run/align.sh data/test_yesno data/lang_test_tg_${topo} exp/rnnp-${topo}/pred_test_yesno || exit 1;
done

# Align with the decoding results
for topo in ${topos}; do
    run/align.sh data/test_yesno data/lang_test_tg_${topo} exp/rnnp-${topo}/pred_test_yesno exp/rnnp-${topo}/dec_test_yesno/aw_1.0-ma_5000-bm_32 || exit 1;
done
