#!/bin/bash
# This script is used to put a symbol link of the kaldi directory here.
# You can modify the path of the kaldi directory to your own path.
# Usege: ./put_kaldi.sh /your/kaldi/path
# Authors:
#   * Zeyu Zhao (The University of Edinburgh) 2024

if [ $# -ne 1 ]; then
  echo "Usage: $0 /your/kaldi/path"
  exit 1
fi

kaldi_path=$1
if [ ! -d $kaldi_path ]; then
  echo "Error: no such directory $kaldi_path"
  exit 1
fi

ln -s $kaldi_path kaldi
