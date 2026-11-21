#!/usr/bin/env bash

mkdir -p data/local
local=`pwd`/local
scripts=`pwd`/scripts

export PATH=$PATH:`pwd`/../../../tools/irstlm/bin

echo "Preparing train and test data"

train_base_name=train_yesno
test_base_name=test_yesno
waves_dir=$1

# Only the audio. The corpus tarball also contains README and README~, and
# listing them here made them utterances: feature extraction then failed on
# them, Kaldi warned "Probably a serious error", fix_data_dir.sh silently
# dropped them, and the test set was quietly 29 rather than the 31 it
# claimed. A real audio failure would have looked exactly the same.
ls -1 $waves_dir | grep '\.wav$' > data/local/waves_all.list

cd data/local

# 31, rather than half of however many lines there are. With the two README
# files counted the list was 62 long and half of it was 31, so train took the
# first 31 waves and test the rest; stating 31 explicitly keeps exactly that
# membership now that the list is 60 long, which is what the published
# results in ../README.md were measured on. An even 30/30 split would move
# one utterance across and make them no longer comparable.
../../local/create_yesno_waves_test_train.pl waves_all.list waves.test waves.train 31

../../local/create_yesno_wav_scp.pl ${waves_dir} waves.test > ${test_base_name}_wav.scp

../../local/create_yesno_wav_scp.pl ${waves_dir} waves.train > ${train_base_name}_wav.scp

../../local/create_yesno_txt.pl waves.test > ${test_base_name}.txt

../../local/create_yesno_txt.pl waves.train > ${train_base_name}.txt

cp ../../input/task.arpabo lm_tg.arpa

cd ../..

# This stage was copied from WSJ example
for x in train_yesno test_yesno; do 
  mkdir -p data/$x
  cp data/local/${x}_wav.scp data/$x/wav.scp
  cp data/local/$x.txt data/$x/text
  cat data/$x/text | awk '{printf("%s global\n", $1);}' > data/$x/utt2spk
  utils/utt2spk_to_spk2utt.pl <data/$x/utt2spk >data/$x/spk2utt
done

