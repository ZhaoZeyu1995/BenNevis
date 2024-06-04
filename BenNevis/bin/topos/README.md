# Introduction

This directory contains the Python code used for creating FSTs of different topologies.
I know it is a bit confusing as there are so many files in this directory.

However, there is a pattern in the file names that, except CTC topology, each topology
has two different files `get_token_fst_*.py` and `get_token_fst_*_k2.py`.

The first one, `get_token_fst_*.py` accounts for creating the T.fst in a language directory, e.g., `data/lang`.
The second one, however, generates `data/lang/k2/T.fst` which is applied during training as this `T.fst` does
not have `<eps>` in its input symbol table, which is compatible with the output of NN models.

Therefore, there is no need to have a `get_token_fst_ctc_k2.py` as the CTC topology does not have `<eps>` already.

We may get a epsilon-free (on the input side) `T.fst` easily by running a epsilon removal operation on `data/lang/T.fst`,
but this usually results in an FST which is much much much larger than the original one.
By using `get_token_fst_*_k2.py`, we can have a epsilon-free `T.fst` by sacrificing the deterministic property.

# Adding yours

You may have a look at one pair of `get_token_fst_*.py` and `get_token_fst_*_k2.py`, possibly run them
and mimic them to get your own topologies.

At the same time, remember to add them to `BenNevis/run/prepare_topo.sh` and `BenNevis/run/prepare_graph.sh`.
