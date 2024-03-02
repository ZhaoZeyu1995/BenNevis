#!/bin/bash 
# This is a wrapper script that conducts decoding with the Kaldi decode-faster decoder.
# The script is used to decode the output posterior probabilities from the neural network.
# Usage:
# run/decode_faster.sh [options] <data_dir> <lang_dir> <predict_dir> <decode_dir>
# e.g.:
# run/decode_faster.sh data/test_dev93 data/lang_eval_test_tg_ctc exp/wav2vec2-ctc/pred_test_dev93 exp/wav2vec2-ctc/decode_test_dev93_bd_tg
# Authors:
#   * Zeyu Zhao (The University of Edinburgh) 2024

. ./path.sh

echo "$0 $@"  # Print the command line for logging

nj=10
max_active="5000"
acoustic_scale="1.0"
beam='32'

. ./utils/parse_options.sh

if [ $# != 4 ]; then
  echo "Usage: run/decode_faster.sh [options] <data_dir> <lang_dir> <predict_dir> <decode_dir>"
  echo "     --nj                        # default: 10, the number of jobs."
  echo "     --max_active                # default: 5000, the max_active value for decode-faster in Kaldi."
  echo "     --acoustic_scale            # default: 1.0, the acoustic_scale value for decode-faster in Kaldi."
  echo "     --beam                      # default: 32, the decoding beam for decode-faster in Kaldi."
  echo "e.g.:"
  echo " $0 data/test_dev93 data/lang_eval_test_tg_ctc exp/wav2vec2-ctc/pred_test_dev93 exp/wav2vec2-ctc/decode_test_dev93_bd_tg"
  echo " where <data_dir> is the directory containing the test data, <lang_dir> is the directory containing the decoding graph,"
  echo " and <predict_dir> should contain a output.scp file storing all of the output posterior probabilities,"
  echo " and <decode_dir> is the directory to store the decoding results."
  exit 1
fi

data_dir=$1
lang_dir=$2
predict_dir=$3
decode_dir=$4

graph=${lang_dir}/TLG.fst

if [ ! -f $graph ]; then
    echo "$0: Cannot find $graph!"
    echo "$0: Please run run/prepare_graph.sh first to prepare the decoding graph."
    exit 1
fi

if [ ! -d $decode_dir ]; then
    mkdir -p $decode_dir
fi

# split the $predict_dir/output.scp to $decode_dir/split$nj/output.JOB.scp
if [ ! -d $predict_dir/split$nj ]; then
    if [ -f $predict_dir/output.scp ]; then
        run.pl JOB=1:$nj $decode_dir/split$nj/log/split_scp.JOB.log utils/split_scp.pl -j $nj JOB --one-based $predict_dir/output.scp $decode_dir/split$nj/output.JOB.scp
    else
        echo "$0: Cannot find output.scp in $predict_dir!"
    fi
fi

acwt=$acoustic_scale
maxac=$max_active

output_dir=$decode_dir/aw_${acwt}-ma_${maxac}-bm_${beam}
if [ ! -d $output_dir ]; then
    mkdir -p $output_dir
fi
run.pl JOB=1:$nj $output_dir/split${nj}/log/decode-faster.JOB.log decode-faster --max-active=$maxac --acoustic-scale=$acwt --beam=$beam --word-symbol-table=${lang_dir}/words.txt ${graph} "ark:copy-feats scp:$decode_dir/split$nj/output.JOB.scp  ark:-|" "ark,t:$output_dir/split$nj/hyp.JOB.wrd"

for i in $(seq $nj); do
    utils/int2sym.pl -f 2- $lang_dir/words.txt $output_dir/split$nj/hyp.$i.wrd | cat || exit 1
done > $output_dir/hyp.wrd.txt || exit 1

awk 'NR==FNR {spk[$1]=$2; next} {printf "%s (%s-%s)\n", substr($0, index($0,$2)), spk[$1], $1}' $data_dir/utt2spk $output_dir/hyp.wrd.txt > $output_dir/hyp.wrd.trn

sclite -r $predict_dir/ref.wrd.trn trn -h $output_dir/hyp.wrd.trn trn -i rm -o all stdout > $output_dir/results.wrd.txt
