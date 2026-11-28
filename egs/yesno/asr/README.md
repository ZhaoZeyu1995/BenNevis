# YesNo

This is a demo of how a recipe works in BenNevis.
What you need to do is to run `./run.sh` in this directory.

The `run.sh` script will do,
1. Data preparation
2. Language and dictionary preparation
3. Feature extraction (80-dimensional fbank) and CMVN
4. Topology preparation
5. The language model and the decoding graphs
6. Training with different topologies
7. Prediction
8. Decoding (with `decode-faster` in Kaldi)
9. Alignment with the ground truth
10. Alignment with the decoding results
11. Decoding again with k2 and BenNevis's own search, for comparison

**Stages 1 to 4 and 6 need no Kaldi and no OpenFST**, and that is
verified rather than claimed: they have been run end to end with no
`tools/kaldi` present at all. Everything they use is Python --
`local/prepare_data.py`, `python3 -m BenNevis.lm prepare-lang`,
`run/make_fbank.py`, `run/compute_cmvn_stats.py`,
`run/fix_data_dir.py`, `run/apply_cmvn.py`, `run/prepare_topo.sh` --
except `local/prepare_dict.sh`, which is three `cp`s and a `grep`.

Stage 5 onward is Kaldi's decoding lane: `arpa2fst` for the grammar, the
OpenFST chain for `TLG.fst`, and `decode-faster` itself. Stage 11 is the
replacement for stages 7 and 8 and needs neither.

**The WER figures below are the author's pre-refactor measurement**, on
CUDA at torch 2.1 and on 83-dimensional features -- 80 filterbank
coefficients plus three pitch dimensions. Pitch has since been dropped --
it is a GMM-era addition for tonal languages and yesno is Hebrew -- so
the model's input is 80 wide now and this table describes a configuration
the recipe no longer has.

It is kept because it is the only pre-refactor measurement there is, and
because the comparison against it is informative: dropping pitch cost
**three word errors in total** across all eight topologies.
`tests/golden/wer/yesno.json` holds what the repository produces *now*,
re-anchored to the first run whose preparation was entirely Python, and
records which era each of its numbers belongs to.

The WER(%) results of different topologies are

| Model-Topology     | test_yesno |
| :----------------- | :--------: |
| rnnp-ctc           | 0.0        |
| rnnp-mmi-ctc       | 0.4        |
| rnnp-mmi-ctc-1     | 0.9        |
| rnnp-2state        | 0.0        |
| rnnp-2state-1      | 0.9        |
| rnnp-3state-skip   | 0.0        |
| rnnp-3state-skip-1 | 0.4        |
| rnnp-3state-skip-2 | 0.4        |
