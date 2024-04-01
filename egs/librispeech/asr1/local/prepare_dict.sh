#!/usr/bin/env bash
# Prepares two character-based dictionaries for LibriSpeech,
# one for training and the other for evaluation.
# The training dictionary contains all the words in the training and dev sets,
# while the evaluation dictionary contains all the words in the official vocabulary
# plus the words in the training sets.
# We do so because during training, the training graph is constructed on-the-fly
# and a bigger vocabulary means a bigger lexicon FST, which is not computationally efficient.
# Usually, it will not be an issue for small toy datasets, like timit and yesno.
# However, for large datasets, like LibriSpeech, it is better to use a smaller lexicon FST for training.
# Authors:
#   * Zeyu Zhao (The University of Edinburgh) 2024

. ./path.sh || exit 1


lm_dir=$1
dst_dir=$2
dst_eval_dir=$3
tmpdir=data/local/dict_tmp

[ -d $tmpdir ] && rm -rf $tmpdir
mkdir -p $tmpdir
[ -d $dst_dir ] && rm -rf $dst_dir
mkdir -p $dst_dir
[ -d $dst_eval_dir ] && rm -rf $dst_eval_dir
mkdir -p $dst_eval_dir

vocab=$lm_dir/librispeech-vocab.txt
[ ! -f $vocab ] && echo "$0: vocabulary file not found at $vocab" && exit 1;

for x in train_clean_100 train_clean_360 train_other_500; do
    [ ! -f data/$x/text ] && echo "$0: expected file data/$x/text to exist" && exit 1;
done

for x in train_clean_100 train_clean_360 train_other_500; do
    cat data/$x/text | cut -d' ' -f2- | tr ' ' '\n' 
done | grep -v -w '<UNK>' | grep -v -w '<SIL>' | sort | uniq > $tmpdir/words.train
for x in dev_clean dev_other; do
    [ ! -f data/$x/text ] && echo "$0: expected file data/$x/text to exist" && exit 1;
    cat data/$x/text | cut -d' ' -f2- | tr ' ' '\n' 
done | grep -v -w '<UNK>' | grep -v -w '<SIL>' | sort | uniq > $tmpdir/words.dev
cat $vocab | sort | uniq > $tmpdir/words.vocab

cat $tmpdir/words.train $tmpdir/words.vocab | sort | uniq > $tmpdir/words.eval
cat $tmpdir/words.train $tmpdir/words.dev | sort | uniq > $tmpdir/words.train_dev

# Prepare the dictionary for evaluation
echo "$0: Preparing the dictionary in $dst_eval_dir for evaluation"

echo "<SIL>" > $dst_eval_dir/optional_silence.txt
echo "<SIL>" > $dst_eval_dir/silence_phones.txt
echo "<SIL> <SIL>" > $dst_eval_dir/lexicon.txt
echo "<UNK> <UNK>" >> $dst_eval_dir/lexicon.txt
echo "<UNK>" > $dst_eval_dir/nonsilence_phones.txt
# get character-level dictionary
cat $tmpdir/words.eval | sed 's/./& /g' | awk '{print $0, "<eow>"}' > $tmpdir/raw.lexicon.txt
paste -d ' ' $tmpdir/words.eval $tmpdir/raw.lexicon.txt | sort | uniq >> $dst_eval_dir/lexicon.txt
cat $tmpdir/raw.lexicon.txt | tr ' ' '\n' | sort | uniq | grep -v -w "<SIL>" | awk 'NF>0' >> $dst_eval_dir/nonsilence_phones.txt

echo "$0: Done"

# Prepare the dictionary for training
echo "$0: Preparing the dictionary in $dst_dir for training"

cp $dst_eval_dir/optional_silence.txt $dst_dir/optional_silence.txt
cp $dst_eval_dir/silence_phones.txt $dst_dir/silence_phones.txt
cp $dst_eval_dir/nonsilence_phones.txt $dst_dir/nonsilence_phones.txt

echo "<SIL> <SIL>" > $dst_dir/lexicon.txt
echo "<UNK> <UNK>" >> $dst_dir/lexicon.txt
# get character-level dictionary
cat $tmpdir/words.train_dev | sed 's/./& /g' | awk '{print $0, "<eow>"}' >> $tmpdir/raw.lexicon.train_dev.txt
paste -d ' ' $tmpdir/words.train_dev $tmpdir/raw.lexicon.train_dev.txt | sort | uniq >> $dst_dir/lexicon.txt

echo "$0: Done"
exit 0
