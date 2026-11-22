# mini-librispeech

Five hours of training audio and two of development, at 16 kHz, from
[OpenSLR 31](https://www.openslr.org/31/). About 500 MB of downloads
including the language model.

Added for two reasons, both about measurement rather than about the corpus.

**yesno cannot measure improvement.** It sits at nought to two word errors,
which makes it a good regression gate and a useless instrument: nothing that
improves a model can be shown to improve it on yesno.

**yesno's filterbank is degenerate.** Eighty mel bins are asked of 20–4000 Hz
against a 256-point FFT, so fourteen of them span a single FFT bin. That is
the worst possible setting in which to judge a feature reimplementation,
because an ill-conditioned filterbank is exactly where two implementations
disagree most. At 16 kHz the same eighty bins see about three FFT bins each
at worst.

## Two ways to use it

**Stages 0–4 need no training and take minutes.** Download, data preparation,
features, topologies, decoding graphs. This is the path that produces a
16 kHz reference for the Kaldi migration, and it is what the feature and
graph comparisons should be measured against from now on.

```bash
./run.sh --stage 0 --stop_stage 4
```

**Stage 5 onward is hours.** At the throughput a CPU runner manages — about
32× real time for this model — five hours of audio is roughly ten minutes an
epoch, so the twenty epochs configured here take a little over three hours
for one topology. Eighty, as yesno uses, would exceed the six-hour job
ceiling, and all eight topologies would take a day. So this is run
deliberately, one topology at a time, and it is **not** part of the nightly.

```bash
./run.sh --topos ctc --stage 5
```

## What is Kaldi's and what is ours

Data preparation is Kaldi's own, through the `local` symlink into
`egs/mini_librispeech/s5/local`: `download_and_untar.sh`, `download_lm.sh`,
`data_prep.sh`, `prepare_dict.sh` and `format_lms.sh`. There is no reason to
reimplement what Kaldi already does correctly, and doing so would add a
difference nobody asked for to a reference corpus.

Everything from the topologies onward is BenNevis's, exactly as in
`egs/yesno/asr`.

## The model is untuned

`conf/model/rnnp.yaml` is a starting point, not a result. The librispeech
recipe uses 3 × 1024, which is far too slow for the CPU runners this is
exercised on, so this is scaled to 3 × 320 to fit a few hours. Tune it
deliberately and record what the change does to the WER ledger.

There is no entry in `tests/golden/wer/` for this recipe yet, because no
training run has been recorded. Until one has, there is nothing to regress
against and the recipe is a reference corpus rather than a gate.
