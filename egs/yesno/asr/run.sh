#!/bin/bash

. ./env.sh || exit 1
. ./path.sh || exit 1

nj=4
ngpu=1
topos="ctc mmictc mmictc-1 2state 2state-1 3state-skip 3state-skip-1 3state-skip-2"

train_set=train_yesno
dev_set=test_yesno
recog_sets="test_yesno"
model=rnnp
hydra_opts=""

. ./utils/parse_options.sh || exit 1

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
            data.train_ds=data/${train_set} \
            data.valid_ds=data/${dev_set} \
            model=${model} \
            loss.kwargs.use_den=false \
            logger.name=${model}-${topo} \
            hydra.run.dir=exp/${model}-${topo} \
            ${hydra_opts} || exit 1;
    else
        torchrun --standalone --nproc_per_node=${ngpu} \
            run/train.py \
            data.lang=data/lang_${topo} \
            data.train_ds=data/${train_set} \
            data.valid_ds=data/${dev_set} \
            model=${model} \
            logger.name=${model}-${topo} \
            hydra.run.dir=exp/${model}-${topo} \
            ${hydra_opts} || exit 1;
    fi
done

# Prediction
for topo in ${topos}; do
    for x in ${recog_sets}; do
        ./run/predict.sh --ngpu ${ngpu} \
            data/${x} data/lang_test_tg_${topo} \
            exp/${model}-${topo}/checkpoints/best.pt exp/${model}-${topo}/pred_${x} || exit 1;
    done
done

# Decoding
for topo in ${topos}; do
    for x in ${recog_sets}; do
        ./run/decode_faster.sh --nj ${nj} \
            data/${x} data/lang_test_tg_${topo} \
            exp/${model}-${topo}/pred_${x} exp/${model}-${topo}/dec_${x} || exit 1;
    done
done

# Align with the ground truth
for topo in ${topos}; do
    for x in ${recog_sets}; do
        run/align.sh --nj ${nj} data/${x} data/lang_test_tg_${topo} exp/${model}-${topo}/pred_${x} || exit 1;
    done
done

# Align with the decoding results
for topo in ${topos}; do
    for x in ${recog_sets}; do
        run/align.sh --nj ${nj} data/${x} data/lang_test_tg_${topo} exp/${model}-${topo}/pred_${x} exp/${model}-${topo}/dec_${x}/aw_1.0-ma_5000-bm_32 || exit 1;
    done
done
