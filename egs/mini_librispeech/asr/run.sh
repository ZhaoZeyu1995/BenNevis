#!/bin/bash
# mini-librispeech: five hours of training audio at 16 kHz.
#
# Added for two reasons, both about measurement rather than about the corpus.
#
# yesno sits at nought to two word errors, which makes it a fine regression
# gate and a useless instrument: nothing that improves a model can be shown
# to improve it. And yesno's filterbank is degenerate -- eighty mel bins over
# 20-4000 Hz against a 256-point FFT, so fourteen of them span a single FFT
# bin -- which makes it the worst available corpus on which to judge a
# feature reimplementation. Both problems go away at 16 kHz with real data.
#
# The data preparation is Kaldi's own, through the local symlink to
# egs/mini_librispeech/s5/local, so this recipe does not reimplement what
# Kaldi already does correctly. Everything from the topologies onward is
# BenNevis's, as in egs/yesno/asr.
#
# Stage 0 to 4 need no training and take minutes; that is the path used to
# capture a 16 kHz reference for the Kaldi migration. Stage 5 onward is
# hours, so run it deliberately.
#
# Usage:
#   ./run.sh                              # everything
#   ./run.sh --stage 0 --stop_stage 4     # data, features and graphs only
#   ./run.sh --topos ctc --stage 5        # train one topology and decode
#
# Authors:
#  * Zeyu Zhao (The University of Edinburgh) 2026

set -eo pipefail

. ./env.sh || exit 1
. ./path.sh || exit 1

nj=8
ngpu=1
topos="ctc"

corpus=./corpus
data_url=www.openslr.org/resources/31
lm_url=www.openslr.org/resources/11

train_set=train_clean_5
dev_set=dev_clean_2
recog_sets="dev_clean_2"
model=rnnp
hydra_opts=""

# The language directory Kaldi's preparation produces, and the one carrying
# the small 3-gram G.fst that decoding composes against. "nosp" is Kaldi's
# own name for a lexicon without word-position-dependent silence modelling.
lang=data/lang_nosp
test_lang=data/lang_nosp_test_tgsmall

#   0  download the corpus and the language model
#   1  data, dict, lang and LM          5  train
#   2  features and CMVN                6  predict
#   3  topologies                       7  decode
#   4  decoding graphs                  8  align against the reference
#                                       9  align against the decoding output
stage=0
stop_stage=100

. ./utils/parse_options.sh || exit 1

if [ ${stage} -gt ${stop_stage} ]; then
    echo "$0: --stage ${stage} is past --stop_stage ${stop_stage}; nothing to do" >&2
    exit 1
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "$0: stage 0: download the corpus and the language model"
    mkdir -p ${corpus}
    for part in dev-clean-2 train-clean-5; do
        local/download_and_untar.sh ${corpus} ${data_url} ${part} || exit 1;
    done
    local/download_lm.sh ${lm_url} ${corpus} data/local/lm || exit 1;
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "$0: stage 1: data, dict, lang and LM"
    for part in dev-clean-2 train-clean-5; do
        local/data_prep.sh ${corpus}/LibriSpeech/${part} \
            data/$(echo ${part} | sed s/-/_/g) || exit 1;
    done
    # --stage 3 skips the grapheme-to-phoneme stages, which need Sequitur and
    # only matter for words outside the supplied lexicon.
    local/prepare_dict.sh --stage 3 --nj ${nj} --cmd run.pl \
        data/local/lm data/local/lm data/local/dict_nosp || exit 1;
    utils/prepare_lang.sh data/local/dict_nosp \
        "<UNK>" data/local/lang_tmp_nosp ${lang} || exit 1;
    local/format_lms.sh --src-dir ${lang} data/local/lm || exit 1;
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "$0: stage 2: features and CMVN"
    fbankdir=fbank
    cmvndir=cmvn
    for x in ${train_set} ${dev_set}; do
        steps/make_fbank_pitch.sh --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} data/${x}/${fbankdir} \
            || { echo "$0: make_fbank_pitch.sh failed for ${x}" >&2; exit 1; }
        steps/compute_cmvn_stats.sh data/${x} exp/compute_cmvn_stats/${x} \
            data/${x}/${cmvndir} || exit 1;
        utils/fix_data_dir.sh data/${x} || exit 1;
        run/apply_cmvn.sh --nj ${nj} data/${x} || exit 1;
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "$0: stage 3: topologies"
    for topo in ${topos}; do
        run/prepare_topo.sh --topo ${topo} ${lang} data/lang_${topo} || exit 1;
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "$0: stage 4: decoding graphs"
    for topo in ${topos}; do
        run/prepare_graph.sh --topo ${topo} ${test_lang} \
            data/local/lang_tgsmall_${topo}_tmp \
            data/lang_test_tgsmall_${topo} || exit 1;
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "$0: stage 5: train"
    for topo in ${topos}; do
        if [ ${topo} == "ctc" ]; then
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
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "$0: stage 6: predict"
    for topo in ${topos}; do
        for x in ${recog_sets}; do
            ./run/predict.sh --ngpu ${ngpu} \
                data/${x} data/lang_test_tgsmall_${topo} \
                exp/${model}-${topo}/checkpoints/best.pt \
                exp/${model}-${topo}/pred_${x} || exit 1;
        done
    done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "$0: stage 7: decode"
    for topo in ${topos}; do
        for x in ${recog_sets}; do
            ./run/decode_faster.sh --nj ${nj} \
                data/${x} data/lang_test_tgsmall_${topo} \
                exp/${model}-${topo}/pred_${x} \
                exp/${model}-${topo}/dec_${x} || exit 1;
        done
    done
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "$0: stage 8: align against the reference"
    for topo in ${topos}; do
        for x in ${recog_sets}; do
            run/align.sh --nj ${nj} data/${x} data/lang_test_tgsmall_${topo} \
                exp/${model}-${topo}/pred_${x} || exit 1;
        done
    done
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "$0: stage 9: align against the decoding output"
    for topo in ${topos}; do
        for x in ${recog_sets}; do
            run/align.sh --nj ${nj} data/${x} data/lang_test_tgsmall_${topo} \
                exp/${model}-${topo}/pred_${x} \
                exp/${model}-${topo}/dec_${x}/aw_1.0-ma_5000-bm_32 || exit 1;
        done
    done
fi
