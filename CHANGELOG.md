# Changelog

Notable changes to BenNevis. This project follows
[Conventional Commits](https://www.conventionalcommits.org/) and, from
`v0.1.0-pre-refactor` onwards, every change lands via a reviewed pull request.

## Unreleased

### Fixed

- `Trainer._test_epoch` and `Trainer._exit` referenced `self.progress_bar`,
  which was never assigned, so both raised `AttributeError` on rank 0. Testing
  now builds its own progress bar and `_exit` shuts down cleanly.
- Gradient accumulation did not scale the loss by `accum_grad_steps`, which
  inflated the effective learning rate in proportion to the accumulation
  factor. No recipe sets `accum_grad_steps`, so published results are
  unaffected.
- Epoch-based checkpointing raised `TypeError` when saving was driven by
  `save_every_n_steps`, because `save_every_n_epochs` is `None` in that case.
- Removed an `if`/`else` in `fit` whose branches were identical.

### Added

- `CLAUDE.md` recording project conventions: style, commit and pull request
  flow, test tiers, equivalence tolerances and known gotchas.
- This changelog.

## v0.1.0-pre-refactor

Snapshot of the hand-written, Kaldi-backed toolkit as used for the topology
experiments, tagged to mark where the reviewed era begins. Developed between
2023 and 2026 by Zeyu Zhao without the use of AI tools.

Highlights of that era, reconstructed from the commit history rather than
written at the time:

- **Topology zoo** — CTC, 2-state, 3-state-skip and MMI-CTC families, each
  with an OpenFST variant for decoding graphs and an epsilon-free variant for
  training with k2.
- **Core** — `Lang` for lexicon and FST handling, `GraphLoss` for numerator and
  denominator computation in k2, Kaldi-format `Dataset`/`CollateFunc`, and a
  `Trainer` supporting DDP, Weights & Biases logging and top-k checkpointing.
- **Models** — RNN-P, Transformer, Conformer, and finetuning paths for
  wav2vec 2.0 and Whisper encoders.
- **Samplers** — duration-based dynamic batching, synchronised across ranks.
- **Recipes** — `yesno`, TIMIT (phone and character units), and LibriSpeech
  (`asr1` phone, `asr2` BPE, `asr3`).
- **Alignment** — forced alignment against transcripts or decoding results,
  moved to `k2.intersect_dense` to avoid the empty lattices that pruned
  intersection could produce.
