#!/usr/bin/env python3
'''
Output a word-loop FST text to stdout from a word list file.
The output is normally piped into fstcompile to create a FST binary.
The weight (log-probability) for all transitions is 0.
The weight for the final state is also 0.
The input and output labels are the same, and are the words in the word list,
except for <eps> and disambiguation symbols.

Usage:
    trivial_gfst.py <words> > <Gtxt>
Example:
    trivial_gfst.py words.txt > G.txt
Or, pipe the output into fstcompile:
    trivial_gfst.py words.txt | fstcompile --isymbols=words.txt --osymbols=words.txt > G.fst

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
'''

import sys
import re


with open(sys.argv[1]) as f:
    for line in f:
        word = line.strip().split()[0]
        # skip <eps> and #digit
        if re.match(r'<eps>|#\d+', word):
            continue
        print('0 0 %s %s 0.' % (word, word))
print('0')
