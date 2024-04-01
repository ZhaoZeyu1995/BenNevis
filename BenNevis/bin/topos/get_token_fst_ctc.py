#!/usr/bin/env python3

# Apache 2.0

import sys

fread = open(sys.argv[1], "r")


"""
By default there is an optional silence phone in between words.
To model this word-level silence, we need the special silence token(s).
Here are two possible methods to deal with the word-level silence:
    1. Do not introduce silence tokens at all. In CTC, the blank label can be applied to deal with the silence, and in other topos, we may introduce a <blk>-<eps> self-loop at the start state.
    2. Introduce the silence token(s). This results in an extra token <SIL> in CTC and potentially multiple tokens in other topos. However, in CTC, <SIL> and <blk> may have some function overlap.
        In other topos, multiple silence tokens may be needed to deal with the silence.
After all, I think the first option is better as it is closer to most readily available implementation, where there is no special treatment for silence except the blank label.
For this reason, I set ignore_silence_between_words = True by default.
"""
if len(sys.argv) > 3:
    ignore_silence_between_words = bool(sys.argv[2])
else:
    ignore_silence_between_words = True

print("0 0 <blk> <eps>")

nodeX = 1
nodes = (
    []
)  # list of tuple (phone, nodeX). Note each phone corresponds to one state id, nodeX

for entry in fread.readlines():
    entry = entry.replace("\n", "").strip()
    fields = entry.split(" ")
    phone = fields[0]
    if phone == "<eps>" or phone == "<blk>":
        continue
    if phone == "<SIL>" and ignore_silence_between_words:
        continue
    if phone.startswith("#"):
        print(str(0) + " " + str(0) + " " + phone + " " + phone)
    else:
        print(str(0) + " " + str(nodeX) + " " + phone + " " + phone)
        print(str(nodeX) + " " + str(nodeX) + " " + phone + " <eps>")
        print(str(nodeX) + " " + str(0) + " " + "<blk> <eps>")
        nodes.append((phone, nodeX))
    nodeX += 1
for phone, nodeid in nodes:
    for nid in range(1, len(nodes) + 1):
        if nodeid != nid:
            print("%d %d %s %s" % (nid, nodeid, phone, phone))
    pass

for i in range(len(nodes) + 1):
    print("%d" % (i))

fread.close()
