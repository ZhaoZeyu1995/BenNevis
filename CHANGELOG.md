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
- A real test suite. Golden-file and structural-invariant tests for all 15
  topology generators, which pin every topology's arc set exactly and assert
  properties that hold for any phone table: hub connectivity, reachability,
  every phone emittable, blank self-loop present, disambiguation symbols
  preserved in the decoding-graph variants, and no epsilon input labels in the
  `_k2` variants. Plus edge-case tests for the Kaldi-format readers in
  `BenNevis.utils.data`.
- `tests/regenerate_topo_golden.py` to refresh the golden files deliberately.
- `pyproject.toml`: packaging metadata, black and isort configuration, and
  pytest marker definitions for the four test tiers. `k2` is deliberately not
  declared as a dependency, since it ships from its own index against exact
  torch and CUDA versions.
- `.pre-commit-config.yaml`, scoped to the paths under active maintenance.
- GitHub Actions CI: lint, tier-1 tests across Python 3.10 to 3.13, and a
  package build check. Linters are pinned to exact versions matching the
  pre-commit revisions, so a new release cannot turn CI red without a code
  change.

### Changed

- `.flake8` now carries narrow, documented per-file exemptions for legacy
  files instead of leaving the whole package ungated. New code is fully
  gated; each exemption records what has to happen before it can be removed.
- Removed five unused imports flagged by flake8 (`BenNevis/core/losses.py`,
  `BenNevis/core/lang.py`, `BenNevis/bin/topos/get_token_fst_mmictc.py`).
- `BenNevis/utils/nets.py` and the `with` statement in `BenNevis/bin/align.py`
  reformatted by black, the latter into parenthesised context managers now
  that the supported floor is Python 3.9 (verified to parse on 3.9.6).
  Whitespace in `README.md` tidied by pre-commit's first run. No other
  formatting changes.
- Minimum supported Python is now 3.10. 3.8 and 3.9 are both end-of-life and
  no longer available on current GitHub runners; the floor tracks what torch
  and k2 publish wheels for.

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
