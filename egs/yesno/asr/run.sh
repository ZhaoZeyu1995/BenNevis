#!/bin/bash
# Stop on the first failing stage.
#
# Most stages below had no `|| exit 1` while a few did, which looks accidental
# rather than intended: a stage could fail and the recipe would carry on to
# the next one with missing or partial inputs. That is how Kaldi printing
# "Probably a serious error" about two unreadable files went unnoticed for as
# long as it did -- the recipe just kept going.
#
# pipefail matters as much as errexit here, because almost everything in a
# Kaldi recipe is a pipeline, and without it only the last command's status
# is seen.
set -eo pipefail

. ./env.sh || exit 1
. ./path.sh || exit 1

# One, not four, and not an arbitrary choice: prepare_data.sh assigns every
# utterance the single speaker "global", and Kaldi's split_scp.pl refuses to
# split data into more parts than there are speakers. With nj=4 this recipe
# cannot get past feature extraction. yesno is sixty files, so there is
# nothing to gain from parallelism here anyway.
nj=1
ngpu=1
topos="ctc mmictc mmictc-1 2state 2state-1 3state-skip 3state-skip-1 3state-skip-2"

train_set=train_yesno
dev_set=test_yesno
recog_sets="test_yesno"
model=rnnp
hydra_opts=""

# Which stages to run, inclusive. Every stage reads what the ones before it
# wrote, so re-running from the middle is only safe when the earlier output
# is still there -- which, after a completed run, it is. Without these, an
# iteration on training re-did data preparation, prepare_lang, feature
# extraction and graph compilation to get there: minutes on yesno and hours
# on a real corpus.
#
#   0  download the corpus          5  train
#   1  data, dict, lang and LM      6  predict
#   2  features and CMVN            7  decode
#   3  topologies                   8  align against the reference
#   4  decoding graphs              9  align against the decoding output
#                                  10  decode again with k2, for comparison
stage=0
stop_stage=100

. ./utils/parse_options.sh || exit 1

if [ ${stage} -gt ${stop_stage} ]; then
    echo "$0: --stage ${stage} is past --stop_stage ${stop_stage}; nothing to do" >&2
    exit 1
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "$0: stage 0: download the corpus"
    if [ ! -d waves_yesno ]; then
      wget http://www.openslr.org/resources/1/waves_yesno.tar.gz || exit 1;
      # was:
      # wget http://sourceforge.net/projects/kaldi/files/waves_yesno.tar.gz || exit 1;
      tar -xvzf waves_yesno.tar.gz || exit 1;
    fi
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "$0: stage 1: data, dict, lang and LM"
    local/prepare_data.sh waves_yesno || { echo "$0: prepare_data.sh failed" >&2; exit 1; }
    local/prepare_dict.sh || { echo "$0: prepare_dict.sh failed" >&2; exit 1; }
    utils/prepare_lang.sh --position-dependent-phones false --sil_prob 0.0 \
        data/local/dict "<UNK>" data/local/lang data/lang \
        || { echo "$0: prepare_lang.sh failed" >&2; exit 1; }
    local/prepare_lm.sh || { echo "$0: prepare_lm.sh failed" >&2; exit 1; }
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "$0: stage 2: features and CMVN"
    fbankdir=fbank
    cmvndir=cmvn
    for x in train_yesno test_yesno; do 
        steps/make_fbank.sh --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} data/${x}/${fbankdir} \
            || { echo "$0: make_fbank.sh failed for ${x}" >&2; exit 1; }
        steps/compute_cmvn_stats.sh data/${x} exp/compute_cmvn_stats/${x} data/${x}/$cmvndir || exit 1;
        utils/fix_data_dir.sh data/${x} || exit 1;
        run/apply_cmvn.sh data/${x} || exit 1;
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "$0: stage 3: topologies"
    for topo in ${topos}; do
        run/prepare_topo.sh \
            --lexicon data/local/dict/lexicon.txt --sil-prob 0.0 \
            --topo ${topo} data/lang data/lang_${topo} || exit 1;
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "$0: stage 4: decoding graphs"
    for topo in ${topos}; do
        run/prepare_graph.sh --topo $topo data/lang_test_tg data/local/lang_tg_${topo}_tmp data/lang_test_tg_${topo} || exit 1;
    done
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "$0: stage 5: train"
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
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "$0: stage 6: predict"
    for topo in ${topos}; do
        for x in ${recog_sets}; do
            ./run/predict.sh --ngpu ${ngpu} \
                data/${x} data/lang_test_tg_${topo} \
                exp/${model}-${topo}/checkpoints/best.pt exp/${model}-${topo}/pred_${x} || exit 1;
        done
    done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "$0: stage 7: decode"
    for topo in ${topos}; do
        for x in ${recog_sets}; do
            ./run/decode_faster.sh --nj ${nj} \
                data/${x} data/lang_test_tg_${topo} \
                exp/${model}-${topo}/pred_${x} exp/${model}-${topo}/dec_${x} || exit 1;
        done
    done
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "$0: stage 8: align against the reference"
    for topo in ${topos}; do
        for x in ${recog_sets}; do
            run/align.sh --nj ${nj} data/${x} data/lang_test_tg_${topo} exp/${model}-${topo}/pred_${x} || exit 1;
        done
    done
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
    echo "$0: stage 9: align against the decoding output"
    for topo in ${topos}; do
        for x in ${recog_sets}; do
            run/align.sh --nj ${nj} data/${x} data/lang_test_tg_${topo} exp/${model}-${topo}/pred_${x} exp/${model}-${topo}/dec_${x}/aw_1.0-ma_5000-bm_32 || exit 1;
        done
    done
fi

# Beside stage 7 rather than in place of it, so one run decodes the same
# models both ways and the two WERs can be compared directly. Two pruned
# searches cannot be compared any other way: decode-faster prunes on the
# arc-cost difference and a token cap, k2 on two beams and a state floor and
# cap, and each returns the best path it found.
#
# The graph is built from the dictionary and the ARPA model by BenNevis.lm,
# not from the language directory's compiled FSTs, so this stage uses no
# OpenFST and no Kaldi binary at all.
#
# data/local/dict/lexicon.txt rather than input/lexicon.txt, which
# prepare_dict.sh copies it from: the two are identical here, but the one
# prepare_lang.sh actually read is the one to rebuild from, and a
# preparation step that ever transformed the lexicon would make them
# differ with nothing to say so.
if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
    echo "$0: stage 10: decode again with k2, for comparison"
    for topo in ${topos}; do
        for x in ${recog_sets}; do
            python3 run/decode_k2.py \
                data/${x} data/lang_test_tg_${topo} \
                exp/${model}-${topo}/pred_${x} exp/${model}-${topo}/dec_k2_${x} \
                --lexicon data/local/dict/lexicon.txt \
                --arpa input/task.arpabo \
                --topo ${topo} \
                --nj ${nj} \
                || { echo "$0: decode_k2.py failed for ${topo}/${x}" >&2; exit 1; }
        done
    done
fi
