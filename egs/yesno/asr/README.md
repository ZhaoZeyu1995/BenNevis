# YesNo

This is a demo of how a recipe works in BenNevis.
What you need to do is to run `./run.sh` in this directory.

The `run.sh` script will do,
1. Data preparation
2. Language and dictionary preparation 
3. Feature exaction (fbank+pitch in this demo)
4. Topology preparation
5. Decoding graph preparation
6. Training with different topologies
7. Prediction
8. Decoding (with `decode-faster` in Kaldi)
9. Alignment with the ground truth
10. Alignment with the decoding results

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
