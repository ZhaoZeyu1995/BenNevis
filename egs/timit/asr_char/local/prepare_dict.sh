#!/usr/bin/env bash
# Prepares two character-based dictionaries for timit,
# the dictionary is built from the train, dev, and test sets.
# Authors:
#   * Zeyu Zhao (The University of Edinburgh) 2024

. ./path.sh || exit 1


dst_dir=data/local/dict
tmpdir=data/local/dict_tmp

[ -d $tmpdir ] && rm -rf $tmpdir
mkdir -p $tmpdir
[ -d $dst_dir ] && rm -rf $dst_dir
mkdir -p $dst_dir

for x in train dev test; do
    [ ! -f data/$x/text ] && echo "$0: expected file data/$x/text to exist" && exit 1;
done

for x in train dev test; do
    cat data/$x/text | cut -d' ' -f2- | tr ' ' '\n' 
done | grep -v -w '<UNK>' | grep -v -w '<SIL>' | sort | uniq > $tmpdir/words.all

# Prepare the dictionary
echo "$0: Preparing the dictionary in $dst_dir"

echo "<SIL>" > $dst_dir/optional_silence.txt
echo "<SIL>" > $dst_dir/silence_phones.txt
echo "<SIL> <SIL>" > $dst_dir/lexicon.txt
echo "<UNK> <UNK>" >> $dst_dir/lexicon.txt
echo "<UNK>" > $dst_dir/nonsilence_phones.txt
# get character-level dictionary
cat $tmpdir/words.all | sed 's/./& /g' | awk '{print $0, "<eow>"}' > $tmpdir/raw.lexicon.txt
paste -d ' ' $tmpdir/words.all $tmpdir/raw.lexicon.txt | sort | uniq >> $dst_dir/lexicon.txt
cat $tmpdir/raw.lexicon.txt | tr ' ' '\n' | sort | uniq | grep -v -w "<SIL>" | awk 'NF>0' >> $dst_dir/nonsilence_phones.txt

echo "$0: Done"
