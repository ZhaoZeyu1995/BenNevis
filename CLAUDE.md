# CLAUDE.md

Conventions for working in BenNevis. Read this before making changes.

## What this project is

BenNevis is an End-to-End ASR toolkit built on Differentiable Weighted FSTs,
with flexible topology support. [PyTorch](https://github.com/pytorch/pytorch)
is the neural network backend and [k2](https://github.com/k2-fsa/k2) the DWFST
backend. Kaldi currently provides data preparation, feature extraction, graph
compilation and WFST decoding.

The distinctive contribution is the **topology zoo** in `BenNevis/topos/`:
CTC, 2-state, 3-state-skip and MMI-CTC families, each defining a different
token-level state machine. Changes must not quietly alter topology semantics.

## Layout

| Path | Contents |
| --- | --- |
| `BenNevis/core/` | `Lang`, `GraphLoss`, `Dataset`/`CollateFunc`, `Trainer` |
| `BenNevis/models/` | Encoders: rnnp, transformer, conformer, wav2vec2, whisper |
| `BenNevis/samplers/` | Duration-based dynamic batch samplers (DDP-synchronised) |
| `BenNevis/topos/` | Declarative topology specs (`specs/*.yaml`) and the expander |
| `BenNevis/run/` | Training, prediction, decoding and alignment entry points |
| `BenNevis/utils/` | Kaldi-format IO and small helpers |
| `egs/<corpus>/<recipe>/` | Recipes: `run.sh` plus Hydra `conf/` |
| `tests/` | Mirrors the package layout |

## Style

- **Line length 88** (black default). Format with `black` and
  `isort --profile=black`. `.flake8` deliberately allows up to 120: black does
  not split long strings or comments, so flake8's `E501` is set higher to
  tolerate what black leaves alone. Black is the authority on layout.
- **British spelling** in code, comments and docs: "optimise", "normalise",
  "behaviour". Note that existing public API names such as `optimizers` and
  `schedulers` stay as they are — do not rename them for spelling.
- Docstrings follow the existing NumPy-ish style with `Arguments` / `Returns`
  sections. Match the surrounding density; the existing docstrings are good and
  are the reference.
- Type hints on public functions.
- Prefer `subprocess` over `os.popen` for new code.

## Commits and pull requests

- **Nothing lands on `master` directly.** Develop on a branch or a
  `git worktree`, open a PR, review it, then **squash-merge** so `master` gets
  exactly one clean commit per PR.
- **Conventional commits**: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`,
  `chore:`, `perf:`, `ci:`.
- **Every commit on `master` must be in a working state.** Kaldi stays
  functional throughout the Python migration; no PR may leave a recipe broken.
- **History is never rewritten.** The pre-refactor history is an accurate
  research record. `v0.1.0-pre-refactor` marks where the reviewed era begins.
- Long-running work (training runs) belongs in a worktree so the main checkout
  stays usable.

## Tests

Tests are tiered by what they *need*, selected with pytest markers, so CI can
run the cheap ones everywhere and the expensive ones rarely.

| Tier | Marker | Needs | When |
| --- | --- | --- | --- |
| 1 | *(none)* | the standard library, pytest and pyyaml | every push |
| 1b | `torch` | torch (CPU) | every push |
| 2 | `k2` | k2 CPU wheel | every push |
| 3 | `kaldi` | Kaldi build (Docker) | nightly |
| 4 | `gpu` | CUDA, multi-GPU | manual |

`egs/mini_librispeech/asr` exists for the measurements yesno cannot make:
yesno sits at nought to two word errors, so nothing that improves a model can
be shown to improve it, and yesno's filterbank is degenerate so nothing about
features can be judged on it either. Its stages 0–4 need no training and take
minutes, which is the path for feature and graph comparison; stage 5 onward is
about ten minutes an epoch for five hours of audio, so twenty epochs of one
topology is three hours and all eight would be a day. It is therefore **not**
part of the nightly, and has no WER ledger entry until a training run has been
recorded.

Tier 3 runs `egs/yesno/asr` end to end inside the Kaldi image, nightly, from
`.github/workflows/nightly.yml`. It is the **only** coverage the entry points
have — `run/train.py`, `run/predict.py`, `bin/align.py` and the shell
wrappers are reachable from no other tier, which is how the regression in #5
got in. The tests do not run the recipe; the workflow does, and they inspect
what it left behind. Keeping those apart matters, because a test that both
drives and judges an hour-long recipe reports "failed" for a missing binary,
a crashed training run and a diverged WER alike. Point them at a finished run
with `BENNEVIS_YESNO_DIR`; with nothing there they skip rather than fail.

```bash
pytest tests/ -m "not torch and not k2 and not kaldi and not gpu"   # tier 1
pytest tests/ -m "torch"                                            # tier 1b
pytest tests/ -m "k2"                                               # tier 2
```

Tier 1's rule is that it must stay **instant**, not that it must have no
dependencies: a test that needs torch belongs in tier 1b with the `torch`
marker, so the cheap tests are never held up by a 200MB wheel download.
`pyyaml` is allowed in tier 1 because the topology specs are YAML and it is a
small wheel with no transitive dependencies — testing the topologies is
exactly the cheap, high-value work this tier exists for. Judge a new tier 1
dependency by what it costs to install, and keep the list this short.

Linter versions are pinned identically in `.pre-commit-config.yaml` and
`.github/workflows/ci.yml` -- keep the two in step, and treat an upgrade as
its own PR, since black's style evolves between releases. The k2 wheel in the
tier 2 job is pinned to an exact torch version and must be bumped with it.

A test that only asserts a module imports, or that passes whether or not the
code under test raises, is not a test. Pin behaviour: exact arc sets for
topologies, numerical agreement for losses and features.

## Platforms and devices

The library runs on Linux, macOS and Windows; the **recipes do not**. `egs/`
is 53 bash scripts and 5 Perl scripts driving Kaldi and OpenFST, so recipes
are Linux and macOS only. Windows support for them is an *outcome* of the
Pythonisation (goal 2), not a separate piece of work, and is not claimed until
Kaldi has left the dependency graph.

Device selection goes through `BenNevis.utils.platform`, never through
`torch.device(rank)` or a bare `torch.cuda.is_available()` check. The
preference order is CUDA, then MPS, then the CPU, and every choice other than
CUDA warns once about what it costs. An *explicitly requested* device that is
unavailable raises instead of downgrading silently: a run that asked for CUDA
and quietly got the CPU would waste hours before anyone noticed.

These limits were measured, not assumed:

| Constraint | Consequence |
| --- | --- |
| k2 has no MPS backend (`.to("mps")` asserts) | the graph loss runs on the CPU |
| MPS has no float64, and `use_double_scores=True` must stay | likewise |
| DDP cannot wrap an MPS module (`c10d::allgather_` unimplemented) | `Trainer` skips the wrapper on MPS, so MPS is single-process only; asking for several ranks there raises |
| NCCL is CUDA-only | every other device uses gloo, the only backend on Windows |
| Windows symlinks need admin or developer mode | `best.pt` is a symlink where possible and a copy where not |
| NCCL collectives need CUDA tensors; gloo does not | `collective_device` decides where a broadcast tensor lives, which is **not** where the model lives |

Cross-device autograd does work, and `GraphLoss` relies on it: an MPS forward
pass has its loss computed on the CPU and backpropagates correctly to the MPS
parameters, agreeing with an all-CPU run to float32. MPS is for development,
not for speed -- every step copies the log probabilities across devices and
synchronises.

**The checkpoint format does not record whether DDP was used.** `MODEL_STATE`
is written from, and read into, the *unwrapped* module, so DDP's `module.`
key prefix never reaches disc and a four-GPU checkpoint loads into a
single-process CPU or MPS run and back. This is what makes skipping the DDP
wrapper safe. An earlier branch tried conditional wrapping without it,
changed the format silently, and was reverted; do not reintroduce
`self.model.state_dict()` at a save site. A legacy prefixed checkpoint is
still accepted, with the prefix recognised only when *every* key carries it.

**The macOS CI runner has no Metal GPU.** It is a VM, so
`torch.backends.mps.is_built()` is True while
`torch.backends.mps.is_available()` is False. Guard MPS tests on
`is_available()`; `is_built()` will let a test through that then fails with
"no supported Metal GPU was found". The consequence is that nothing which
needs a real MPS device runs in CI, which is why the device *logic* is tested
by faking the availability probes instead -- those tests run on every runner.
Treat the real-device tests as something only a developer on an Apple machine
executes.

**Compare a relative path with `as_posix()`, never `str()`.**
`Path.relative_to(...)` yields the platform separator, so a literal like
`"yesno/asr"` matches nothing on Windows and a manifest keyed on forward
slashes matches nothing either. This has broken tier 1 on Windows three
times: the reference manifest's file list, the recipe-name lookup in the
stage tests, and the parametrise ids that made the first two hard to read.
The same applies to anything written into a committed artefact — a captured
path must be POSIX or it is not portable.

**`bash` exists on a Windows runner and does not work.**
`shutil.which("bash")` finds `C:\Windows\System32\bash.exe`, the WSL
launcher, which is present whether or not a distribution is installed and
exits 1 when one is not. Probing for the executable is therefore not enough:
a test that runs shell must be skipped on `os.name == "nt"`, not on `which`.
The same applies to locating any POSIX tool by name — `which` on Windows
wants a `PATHEXT` extension, and `os.access(..., os.X_OK)` is meaningless
there because there is no execute bit.

CI runs tiers 1, 1b and 2 on `ubuntu-latest`, `macos-latest` and
`windows-latest`. k2 publishes cp312 CPU wheels for all three at the pinned
torch version, so the FST tier is not Linux-only; **check the index before
bumping the pin**, since a newer build may not cover every platform on the
day. Tests must therefore not assume POSIX: no shell pipelines, no `os.popen`,
no hardcoded `/tmp`, and `subprocess` only via `sys.executable` with an
argument list.

## Equivalence and tolerances

The Python migration is gated on reproducing current behaviour, with
tolerances chosen per layer:

- **Features**: `rtol=1e-4` against Kaldi. Bit-exactness is not achievable
  (FFT and accumulation order differ). Check `snip-edges` and `energy-floor`
  defaults too.
  **`--dither=0` is now set in every recipe's `conf/fbank.conf`, and it is a
  precondition rather than a tuning choice.** Kaldi's default of 1.0 adds
  random noise to the waveform; its own generator makes that reproducible
  *for Kaldi*, which is why two captures agreed byte for byte, but no other
  implementation can match it. Measured on yesno: two different dither draws
  differ from each other by 0.045 mean and 2.59 max in the log-mel domain,
  and undithered torchaudio differs from dithered Kaldi by 0.038 mean — the
  same size, concentrated in the quietest bins, which is the signature of
  additive waveform noise. With dither on, features cannot be replaced at
  any tolerance worth the name.
  A capture records `conf/fbank.conf` and `conf/pitch.conf`, and
  `tests/reference/test_against_reference.py` fails if they no longer match
  the recipe — because a reference taken under different settings is a record
  of something the pipeline no longer does, and nothing else would notice.
  **Removing dither cost one word error in total across all eight
  topologies**, and it moved downward, so the baseline survived the change
  intact.
- **`torchaudio.compliance.kaldi.fbank` is close but not equivalent, and the
  cause is narrowed rather than found.** Against undithered Kaldi it agrees
  to 0.065 max and 0.012 mean in the log-mel domain, down from 1.45 and 0.038
  with dither on — so dither was about 95% of the gap. What is left is
  **zero-mean scatter of about 1.2% in linear energy**, and these have been
  ruled out:

  | Hypothesis | Verdict |
  | --- | --- |
  | Configuration | every Kaldi and torchaudio default already agrees — `use_power`, `use_log_fbank`, `htk_compat`, `raw_energy`, `low_freq`, `high_freq`, `povey`, preemphasis, `remove_dc_offset`, `round_to_power_of_two`, `snip_edges`. `energy_floor` differs but only matters with `use_energy=true` |
  | Framing offset | offset 0 agrees twice as well as ±1, so framing is right |
  | The `povey` window | the two definitions agree to 1.3e-16 |
  | Filterbank weights | a fixed weight error would bias each mel bin consistently; per-bin mean differences are ±0.0004 |
  | A normalisation factor | the mean difference of logs is −0.00007, so there is no offset to find |
  | Quiet-bin cancellation | the scatter is flat across frame energies |

  | The filterbank's conditioning | **tested and refuted**, see below |

  **It is not the degenerate filterbank.** That was the natural suspicion and
  it is wrong: measured against mini-librispeech at 16 kHz, where the
  filterbank is well conditioned, the gap is **7.7e-02 max and 1.4e-02
  mean** — the same size as yesno's 6.5e-02 and 1.2e-02 at 8 kHz. So the
  residual is independent of the sample rate and of how many FFT bins a mel
  bin spans, which rules out the one explanation that had looked likely
  enough to justify building a second corpus for.

  What remains is the per-frame spectrum path, and localising it needs one of
  Kaldi's intermediates rather than more guessing from outside.
  `compute-spectrogram-feats` would give the power spectrum before the
  filterbank, and capturing that for a few utterances would say in a single
  run whether the difference is upstream or downstream of the mel
  projection. That is the next cheap step if this is worth pursuing.
- **The yesno filterbank is degenerate, and this is about the recipe rather
  than the migration.** 80 mel bins are requested over 20–4000 Hz, but a
  25 ms frame at 8 kHz is 200 samples, padded to a 256-point FFT, giving 128
  usable bins of about 31 Hz each. So **14 of the 80 mel bins span a single
  FFT bin** and 36 span two or fewer, with triangle weight sums as low as
  0.33. Kaldi's own `feature-fbank.h` says the default of 23 suits 16 kHz and
  that "for 8khz-sampled data, 15 may be better". The 16 kHz recipes are
  fine — 80 bins over 20–8000 Hz against a 512-point FFT is about three FFT
  bins per mel bin at worst. This still matters for what yesno's
  features *are* — many of its 80 dimensions carry nearly the same
  information — but **not** for the fbank comparison: the entry above shows
  16 kHz disagrees by the same amount, so the conditioning is not what the
  implementations differ over. The earlier claim that yesno was therefore the
  worst corpus on which to judge a feature reimplementation was reasonable
  and turned out to be false.
- **Losses**: agreement to ~1e-4 on a fixed batch. This is the fastest and
  most localising signal — prefer it over WER when validating a migration.
- **FSTs are compared as functions, not as bytes.** An LM or lexicon FST is a
  map from word sequences to weights, and two encodings of the same map are
  equally correct — byte equality would additionally require reproducing
  `arpa2fst`'s state numbering, which it assigns in whatever order it meets
  histories while reading the ARPA file, plus the arc order within each of
  249,532 states for mini-librispeech's 3-gram. None of that is meaningful,
  and mimicking it would make the Python harder to read for no gain. So the
  standard for `G.fst`, `L.fst` and `TLG.fst` is: the same weight for every
  word sequence, checked on sampled strings and against the real decoding
  output, with the WER ledger as the backstop.
  **`L.fst` turns out to be byte-exact anyway, and is checked that way.**
  Both choices `G` leaves arbitrary are determined for the lexicon: states
  are allocated in the order `lexicon.txt` lists its words, and
  `prepare_lang.sh` leaves the arcs sorted by output label. So the Python
  build reproduces Kaldi's bytes for yesno exactly. The functional standard
  is there because an FST's encoding is *usually* arbitrary, not because
  bytes are unwelcome where they are not — take them when they are free.
  This holds for `--sil_prob 0.0`, which is what `egs/yesno/asr` uses; a
  non-zero silence probability adds a start state, a silence state and
  weighted entry arcs whose exact shape is `make_lexicon_fst.pl`'s, and the
  builder **raises** rather than guessing at it, since a plausible-looking
  lexicon that decodes differently is the worst outcome available.
  This is a different standard from CMVN's deliberately. There, the object is
  an array and any difference is a real difference, so byte-exactness was
  both achievable and the right test. Byte-exactness is still expected where
  it already holds — the symbol tables, `disambig.int` and the topology FSTs,
  all of which are canonical rather than arbitrary.
- **WER**: the backstop, not the primary tool. Gate on a delta, not equality.

**Kaldi's output is recorded, not re-derived.** `tests/golden/reference/`
holds what Kaldi produced from a real `yesno` run inside the pinned image,
captured by `tests/reference/capture.py` and described by a `manifest.json`
that names the image digest. Each Kaldi stage that moves to Python is
compared against that, in `tests/reference/test_against_reference.py`, at the
tolerance the manifest records for it — so a migration PR is gated in tiers 1
and 2, with no Kaldi installed anywhere.

Recording rather than comparing live is not a convenience. **Kaldi is
bit-rotting**: recent commits cannot fetch the OpenFST version their own
source needs, and the OpenBLAS they pin no longer compiles on current
hardware. The image that works is pinned by digest, and its output is worth
capturing while it can still be produced.

The capture is **never written by hand**. It comes from a nightly run and is
committed from the uploaded artefact. Fabricated reference data is worse than
none: every comparison against it would pass while meaning nothing. Likewise,
a reference edited to make a test pass is no longer a reference — re-capture
instead, and say in the PR what moved.

The WER ledger is `tests/golden/wer/yesno.json`: one `reference_wer` per
topology, taken from `egs/yesno/asr/README.md` and so from the author's own
pre-refactor CUDA runs, and one `max_wer` that is the gate. **Raising
`max_wer` to make a run pass is how a ledger stops meaning anything** — judge
a change against `reference_wer` and tighten `max_wer` deliberately. The
gates start loose because no run has yet been recorded in CI; `test_set` has
232 reference words, so one word error is 0.43%.

The error rate itself is computed by `BenNevis.utils.scoring`, in Python, not
read out of `sclite`. That keeps sctk off the measurement path and makes the
number testable in tier 1. `sclite` is then the *independent oracle*: tier 3
asserts the two agree on real output, which is a stronger check than either
alone and avoids pinning a parser against a hand-made fixture.

## Environment

Locally a venv in `tools/venv` created by `tools/create_env.sh`.

For anything that needs Kaldi there is an image, `docker/Dockerfile`,
published as `ghcr.io/zhaozeyu1995/bennevis-kaldi`. It derives from the Kaldi
project's own CPU image and adds CPU torch and k2 at the same versions as
CI's tier 2 — but **not BenNevis**, which is reached through a mounted
working tree so that the image tests the commit in front of it. The
entrypoint creates the `tools/kaldi` and `tools/venv` links the recipes
expect. `docker/README.md` has the detail; `docker/smoke.sh` checks an image
has everything a recipe invokes.

**Kaldi is borrowed, not built, and deliberately so.** Building it here was
tried and abandoned, and the knot is worth knowing before anyone tries
again: Kaldi fetches OpenFST 1.8.4 from openfst.org, which refuses a CI
runner's egress on both schemes, and falls back to OpenSLR, whose mirror
stops at 1.8.1 — dead for everyone on recent Kaldi. Taking 1.8.1 instead
builds `tools` and then fails in `src`, where `fstext/lattice-weight.h`
compares a pointer with a character, valid only from OpenFST 1.8.2 on. And
OpenBLAS 0.3.13, which Kaldi pins, needs `TARGET` forced to compile on a
current CPU at all. Maintaining that is maintenance of the dependency goal 2
exists to delete; the prebuilt image is the right level of investment for
something on its way out.

The base is pinned by **digest**, not tag: an image that moved underneath the
WER ledger would turn it into noise.

The image is linux/amd64 only, which is what the base publishes. Its Python
is Debian 12's 3.11 rather than the 3.12 the other tiers use, so a failure
seen only in tier 3 may be an interpreter difference — check that early.

torch and k2 appear in both `docker/Dockerfile` and `ci.yml`'s tier 2 job,
and `tests/docker/test_pins.py` fails if the two disagree — a mismatch would
surface as a k2 import error inside a nightly job, a long way from its
cause. The image's dependency list is derived from `pyproject.toml` by
`docker/deps.py` rather than written into the Dockerfile, for the same
reason. The image is built only when `docker/`, `pyproject.toml` or its
workflow changes.

## Gotchas

- **The dynamic batch samplers had three bugs and no tests.** The first real
  tier 3 training run found them in sequence, and `tests/samplers/test_dynamic.py`
  now pins all three. They are worth knowing because two are the kind that
  recur: `num_batches` is only created inside `__iter__`, where the batches
  are formed, so a caller must read it *after* iteration has begun — and
  declaring it alone is not enough, because the count then stays `None` and
  the mid-epoch save schedule silently stops firing. The sampler also called
  `torch.cuda.current_device()` unconditionally, making dynamic batching
  unusable without CUDA. And an utterance longer than `max_sum_dur` **looped
  forever**: the batch was reset but the pointer never advanced. Note
  `max_dur` can exceed `max_sum_dur` in the shipped configs, so that one was
  reachable with real data.
- **Progress bars are off when stderr is not a terminal, and every epoch
  logs a line.** `tqdm` writes a fresh line per update off-terminal, so a
  nightly run produced thousands of near-identical lines with no epoch
  boundaries, and a healthy run was indistinguishable from a hung one — the
  author checked on a run and reported it stuck when it was at epoch 68 of 79
  and converging. Every `tqdm` call passes `disable=None`, which is its
  documented "off unless interactive", and `fit` logs one line per epoch with
  both losses and the step. `tests/core/test_training_logs.py` pins both,
  because neither fails anything when lost.
- **`opt.nickname` is optional now.** `run/train.py`'s `get_opt` attaches it
  via `setattr`, and `Trainer` used to require it, so a `Trainer` built
  anywhere else — a test, a script, a notebook — crashed inside `wandb`
  logging rather than anywhere informative. It falls back to the optimiser's
  class name.
- **Give training and validation separate samplers in a test.** `fit` runs a
  validation sanity check before the first epoch, so a shared sampler is
  already iterated by the time training starts — which hides whether the
  trainer learns the batch count for itself. That masked a real bug until the
  two were separated.
- **Keep `use_double_scores=True`** in `k2` `get_tot_scores`. With `False`,
  NaNs have occurred, particularly in the denominator computation.
- **k2 and torch define the CTC gradient differently with respect to log
  probabilities.** k2 returns the true `d(loss)/d(log_probs)`, which is
  `-gamma` and sums to `-1` per frame. `torch.nn.functional.ctc_loss` returns
  `(p - gamma)`, the gradient with respect to the pre-softmax logits, which
  sums to `0`. Both are invariant under the log_softmax Jacobian, so they
  agree exactly once the log_softmax is in the graph -- which it always is in
  training. Compare the two at the logits, never at the log probabilities, or
  a correct implementation will look broken.
- Do not put the k2 graph loss inside `torch.autocast`. FST intersection and
  score accumulation are numerically delicate; autocast the encoder forward
  and compute the loss in fp32.
- `Trainer` requires an initialised process group, since it reads the rank and
  world size in `__init__` and all-reduces losses. It can be constructed on
  CPU: the tests run it against a real one-rank gloo group backed by a
  `FileStore`, provided by the `process_group` fixture in
  `tests/core/conftest.py`.
- `Trainer` places the model on its own `device`; callers need not, though
  the entry points still do. A `Trainer` whose `device` disagreed with where
  its model lived used to raise a device mismatch from inside the first
  linear layer, far from the cause.
- The **global** rank decides which process writes checkpoints and logs; the
  **device index** is `LOCAL_RANK`, the rank within the node. They coincide on
  one node and diverge across several, so never select a device by global
  rank.
- Topology FSTs and `tokens.txt` must stay consistent — the topology
  determines the neural network's output vocabulary. Both now come from the
  same spec, and `TopologySpec.validate` rejects a spec whose arcs use a token
  its inventory does not list, so the two cannot drift apart silently.
- **`prepare_topo.sh` copies the spec into the language directory** as
  `k2/topo.yaml`, and `Lang.load_topo` expands *that*, not the bundled spec of
  the same name. A language directory therefore stays reproducible when the
  bundled specs change, and the training path needs no OpenFST. The symbol
  tables are read from the directory too, never recomputed: `k2/tokens.txt`
  fixes the network's output indices, and a checkpoint is only valid while
  those indices stay put. `k2/T.fst` is still compiled, but nothing in
  training reads it.
- **The topology FSA is not arc-sorted, and `k2.intersect_dense` documents
  that its graphs must be.** `fstarcsort --sort_type=olabel` in
  `prepare_topo.sh` sorted by *output* label, which does not make a k2 FSA
  `ARC_SORTED` — measured False for all eight topologies, before and after
  the spec rewrite. `k2.compose` arc-sorts its first argument itself, so the
  numerator is fine; `GraphLoss.get_den_graph` passes the topology straight to
  `intersect_dense`, which is the violation. On CPU the denominator scores are
  identical either way (measured on all eight), so this has not been biting,
  but `k2.arc_sort` there would make the precondition hold and wants checking
  on CUDA.
- **A topology's golden file is its specification, not a cache.** The files in
  `tests/golden/topos/` were produced by the fifteen `get_token_fst_*.py`
  scripts that used to define the topologies, and the YAML specs were written
  to reproduce them byte for byte. That provenance is what makes them an
  independent check, and regenerating them from the specs gives it up. Add
  goldens for new topologies; think hard before regenerating existing ones.
- The `k2` variant of every topology must be **epsilon-free on the input
  side**, because its input alphabet is the network's output layer and a
  network cannot emit `<eps>`. This is checked at spec load time.
- **The remaining POSIX-only sites are deliberate.** `Lang.load_lexicon`
  shells out to `fstprint | awk`, and needs the lexicon off Kaldi first,
  which is a larger piece of goal 2. `run/predict.py`'s `cat | awk` merge and
  `run/align.py`'s `gzip -c` pipe are portable in a few lines of Python each,
  but the entry-point scripts have no test coverage, so they wait for tier 3
  rather than being changed blind. `Lang.load_topo` keeps a `fstprint`
  *fallback* for a language directory prepared before `k2/topo.yaml` existed;
  it warns, and no new directory takes that path.
- **Recipe configs rot against torch.** `ReduceLROnPlateau` lost `verbose`,
  and all four recipes passed it, so every one was broken against current
  torch while every test passed — the first real tier 3 run died there.
  `tests/recipes/test_recipe_configs.py` now compares each configured
  optimiser and scheduler against the installed signature, in under a second,
  so this class of rot is caught without a recipe run. It checks that an
  argument *exists*, not that its value is sensible: a nonexistent argument is
  a defect, a wrong learning rate is a research decision.
- **`egs/yesno/asr` assigns every utterance one speaker.** `prepare_data.sh`
  writes `global` for all of them, and Kaldi's `split_scp.pl` refuses to split
  data into more parts than there are speakers, so the recipe could not get
  past feature extraction with its own default of `nj=4`. It is `nj=1` now.
  `run/apply_cmvn.sh` already handled this by clamping `nj` to the speaker
  count; anything new that splits by speaker should do the same.
- **An empty hypothesis used to be scored as its own utterance id.**
  `run/decode_faster.sh` stripped the id with
  `substr($0, index($0, $2))`, which is correct until the decoder emits
  nothing: `$2` is then unset, `index($0, "")` is 0, and `substr($0, 0)`
  returns the whole line. The error *count* is the same either way — one
  substitution plus N−1 deletions against N deletions — so the WER looked
  plausible while the transcription was a file name, and anything reading
  `hyp.wrd.txt` saw a word never spoken. Found by the first complete recipe
  run, whose undertrained model decoded nothing at all.
  `tests/recipes/test_decode_faster.py` extracts the awk from the script and
  runs it, so the test is about the code that ships.
- **The stage-guard convention predates this work.** `egs/timit/*` and
  `egs/librispeech/asr3` already used `stage`/`stop_stage` with the same
  `if [ ${stage} -le N ] && [ ${stop_stage} -ge N ]` form, and asr3 uses
  **stage -1** for an optional download, so a recipe's first stage is not
  necessarily zero. What yesno and mini_librispeech add on top is announcing
  each stage, refusing an inverted range and stopping on a failing one;
  `tests/recipes/test_run_stages.py` holds the list of converted recipes and
  asserts its own membership, so the gap is visible rather than unasserted.
- **Recipes have `--stage` and `--stop_stage`**, numbered 0 to 9 in
  `egs/yesno/asr/run.sh` and announced as they run. Each stage reads what the
  ones before it wrote, so resuming is only safe where the earlier output
  survives — true for a local tree after a completed run, false on a fresh CI
  runner, which is why the scheduled nightly runs everything. A `stage` past
  `stop_stage` is refused rather than exiting 0 having done nothing, which in
  CI is indistinguishable from success.
  `tests/recipes/test_run_stages.py` extracts the guard expressions and
  executes them with the bodies replaced by an echo, so the selection is
  tested without Kaldi.
- **Recipes stop on a failing stage.** `egs/yesno/asr/run.sh` has
  `set -eo pipefail` and an explicit check per stage. Most stages had neither
  while a few had `|| exit 1`, which is why Kaldi printing *"Probably a
  serious error"* went unnoticed — the recipe simply continued. `pipefail`
  matters as much as `errexit`, because nearly everything in a Kaldi recipe
  is a pipeline.
- **A container writing into the workspace writes as root.** The nightly's
  capture step runs in Docker, and anything it creates is root-owned, after
  which a later host-side step cannot write there. That is how the scores
  artefact uploaded empty while the job reported success — the `tar` failed
  with permission denied and a `|| true` swallowed it. Create a directory on
  the runner *before* any container touches it, and do not let a collection
  step hide its own failure: report what it collected, including zero.
- **`make_fbank_pitch.sh` writes compressed archives.** Kaldi's
  `CompressedMatrix` quantises each column to 8 bits under its default
  `kSpeechFeature` scheme, so a 633-frame utterance comes back with under 256
  distinct values per dimension. `kaldiio` decompresses faithfully enough for
  any tolerance comparison, but **not** bit-identically to Kaldi's own
  decompression — which is why the Python CMVN first agreed with
  `apply-cmvn` to 4e-6 rather than exactly, even with Kaldi's algorithm
  reproduced operation for operation. The capture therefore has **Kaldi
  rewrite each archive uncompressed with its own
  `copy-feats --compress=false`** and reads that, so the values are the ones
  Kaldi worked with rather than another decompression of them.
  The **recipe is deliberately unchanged**: making it store features
  uncompressed would alter the inputs to CMVN, hence the statistics, hence
  the WERs, and would invalidate the committed baseline. The round trip is an
  artefact of how the reference is *read*, so the fix belongs in the capture.
  CMVN statistics are read directly, because they are double and never
  compressed — putting them through `copy-feats` would narrow them to float
  and lose the precision the whole normaliser rests on.
  With that in place the Python CMVN is **bit-exact** with `apply-cmvn`, and
  the re-captured features turned out to differ from the old ones by at most
  3.8e-6 — precisely the residual that had been showing up in the output.
- **Bit-exact arithmetic is not the same as a bit-exact pipeline.** The CMVN
  equivalence holds *given identical inputs*. In the live recipe, a Python
  stage would read the compressed `feats.scp` through `kaldiio` and so start
  from values up to 3.8e-6 away from the ones Kaldi used. Replacing a stage
  therefore moves the decompression boundary rather than removing it, and
  the boundary only disappears once feature extraction is Python too and
  owns the storage format. Until then, swapping CMVN into the recipe costs
  about 4e-6 on its inputs — which is inside the documented feature
  tolerance but is **not** nothing, and is a decision rather than a
  detail.
- **Reproduce Kaldi's order of operations, not just its formula.**
  `ApplyCmvn` computes the moments and scale in double, rounds scale and
  offset to float32 as it stores them, then applies them as two separate
  float32 passes rather than one fused affine. Writing the obvious
  `(x - mean) / sqrt(var)` in float64 instead agrees only to about 4e-6.
  Following the original costs nothing and removes a needless source of
  divergence, so there is no reason not to.
- **An exact artefact over a megabyte records its digest and not its bytes.**
  mini-librispeech's `TLG.fst` is **334MB** as text against yesno's 20KB,
  because it carries a 200k-word lexicon and a real 3-gram model; the first
  capture of it came to 606MB, almost all FSTs. Since an exact artefact is
  compared *by digest*, the content is not needed to answer whether a
  replacement reproduces it — only to see how it differs when it does not.
  Those entries are listed in the manifest's `large`, so a deliberately
  absent file can be told from a lost one, and the uploaded workflow artefact
  keeps the bytes for ninety days. Content is only ever needed for a
  *tolerance* comparison, which is why the feature matrices are stored whole
  and the FSTs are not.
  A **symbol table is kept whole however large it is**, because it is an
  *input* to a replacement rather than an output of one: `words.txt` fixes
  the integer labels every FST in the directory is written against, so a
  Python `arpa2fst` cannot be driven at all without it. mini-librispeech's
  is 3.0MB — over the threshold and nothing like the 334MB the threshold
  exists for. Same rule as recording both sides of CMVN.
- **The capture records both sides of CMVN.** `raw/` is `feats.scp` as the
  feature extractor wrote it; `features/` is `feats.cmvn.scp` after
  normalisation. The first captures held only the second, which was no use
  for verifying a replacement for either stage — there was no input to drive
  it with. **Any stage being replaced needs its input captured as well as its
  output**, and that is worth checking before writing the replacement rather
  than after.
- **The reference capture needs `fstprint`, and will not pretend otherwise.**
  It is in the image but only on the PATH a recipe's `path.sh` builds, and
  `tests/reference/capture.py` is invoked directly — so the first two real
  captures contained no FSTs at all and said nothing about it. It now looks
  under `$KALDI_ROOT/tools/openfst/bin` as well, and **fails** rather than
  producing a capture without the lexicon and graph FSTs, since those are
  what a Python `prepare_lang.sh` would be compared against.
  `--allow-no-fstprint` is the deliberate escape.
- **The yesno split is 31/29 on purpose, not 30/30.** The corpus tarball
  holds `README` and `README~` beside the sixty waves, and `prepare_data.sh`
  used to list the directory unfiltered. Under `LC_ALL=C` digits sort before
  `R`, so both landed in the test half of a 62-entry list; feature extraction
  failed on them, Kaldi warned *"Probably a serious error"*, `fix_data_dir.sh`
  dropped them, and the test set was 29 while claiming 31. **A real audio
  failure would have looked identical.** The list is filtered now, and the
  split count is stated as 31 rather than left to halving, because halving 60
  gives 30/30 and moves one utterance across — which would make the published
  WERs in the recipe's README no longer comparable. `tests/golden/wer/yesno.json`
  records the counts and tier 3 asserts them, so the split cannot drift
  unnoticed.
- **A back-off FST's shortest path can be cheaper than the ARPA back-off
  score, and `arpa2fst`'s output does this too.** Backing off early can cost
  less than the correct higher-order arc plus the price of leaving the state
  it leads to. Measured on mini-librispeech's own 3-gram: Kaldi's committed
  `G.fst` scores *OR APPEARED MONGS* at 37.7573 against the back-off rule's
  39.8078, and the Python construction reproduces that same 37.7573. So the
  rule in `BenNevis.lm.arpa` is the oracle for **arc weights**, not for
  **path totals**, and `G.fst` is checked against Kaldi's FST rather than
  against the rule. The one-sided bound does hold unconditionally — the rule
  describes a real path, so the cheapest path cannot cost more. Where the
  two agree is *not* characterisable: on a small model `A B` agrees while
  `A B C` does not, and it turns on the whole path's arithmetic rather than
  on which n-grams are present. Two hypotheses were wrong before this was
  understood: that it needed unrealistic back-off weights (it happens on the
  real pruned 3-gram), and that proper discounting prevents it (it does not).
- **Give an n-gram history a state only if something extends it.** The
  obvious rule — a state per entry below the highest order — gives
  **1,162,307** states for mini-librispeech's 3-gram against `arpa2fst`'s
  249,532. The FST is functionally the same, since the extra states carry
  nothing but a back-off arc, but it is 4.7 times the size and gets composed
  and determinised into a `TLG` that is already 334MB. With the pruning the
  Python build comes to 249,863 states and the same 68,639 final states.
- **OpenFST's text format takes the start state from the first line's
  source.** Kaldi's `G.fst` for mini-librispeech begins at state **2**, not
  0, so reading it as though it began at 0 adds that state's back-off weight
  — 2.2265 nats — to every sentence and makes a correct FST look wrong.
  The same trap in reverse: the *lowest numbered* state is the unigram
  state, so inferring a start state from the arcs silently scores every
  sentence as though it had no beginning. A unigram model cannot tell the
  difference, so yesno agrees either way and every higher order does not.
- **ARPA files mix tabs and spaces within a line.** LibriSpeech's 3-gram
  writes a tab after the log probability and *spaces* between the words, so
  splitting on tab when a tab is present reads `-1.486134\t<UNK> </s>` as a
  malformed two-field bigram. Split on any whitespace: ARPA words cannot
  contain it. Found by the real 13MB model and by nothing smaller.
- **Shell text processing is dialect-dependent, and was relied on.**
  `prepare_topo.sh` selected disambiguation symbols with `grep "#[^\d]"`,
  which picks out `#0` only where `\d` inside a bracket expression means the
  literal characters `\` and `d`. That is POSIX, but under `ugrep` or
  `grep -P` it is a digit class, the pattern stops matching, and the token
  table silently comes out as `#0_0`, `#0_1`. Prefer Python for anything that
  parses a symbol table.
