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
minutes, which is the path for feature and graph comparison; stage 5 onward
is **about twenty minutes an epoch on a CI runner** for five hours of audio,
so twelve epochs of one topology is four hours and all eight would be days.
It is therefore **not** part of the nightly, and has no WER ledger entry
until a training run has been recorded.
The twenty is measured, from run 45202289714: twelve epochs of `ctc` ran
18:05 to 22:08, which is 20m09s each. It had been written here as ten, and
that estimate is what made a 300-minute ceiling look generous — the run hit
it in stage 10 having spent four hours of its five on training. **Validation
loss bottomed at epoch 8** there, 1.8028 against 1.8879 by epoch 12, so the
last four epochs cost eighty minutes and made the model worse: for a run
whose point is to compare two decoders on the same acoustics, eight epochs is
both cheaper and better.
**Twenty minutes is not a stable figure, and the second training run is why.**
Run 45293910683 took **34m30s** an epoch on the same recipe, commit range and
runner label — 71% slower — and was then **SIGKILLed in epoch 4**, exit code
`-9` out of `torchrun` after 213 minutes. Nothing between the two runs touches
training, so this is the runner rather than the code, and the two symptoms
have one likely cause: memory pressure, which both slows a job down and ends
it. Treat the epoch cost as 20 to 35 minutes, and note what that does to the
budget — eight epochs is 160 minutes at the fast end and 276 at the slow one,
so a 350-minute ceiling fits the first comfortably and the second only if
nothing else goes wrong. A run that needs the corpus downloaded as well does
not fit at all.

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

**A package `__init__` that imports a heavy dependency makes every
submodule as expensive as the heaviest one.** `BenNevis/decode/__init__.py`
imported its k2-dependent submodules eagerly, so importing
`BenNevis.decode.pipeline` to format a `.trn` line pulled in k2 and broke
tier 1 and tier 1b on **every platform at once** — nine jobs red for a
module that needs nothing but the standard library. The names now resolve
through a module-level `__getattr__`.
A machine with k2 installed cannot see this, which is why
`tests/decode/test_pipeline.py` runs the import in a subprocess with k2
made unimportable. The same trick checks a tier selection:
`pytest -m "not torch and not k2 and not kaldi and not gpu"` with k2
blocked is the only local test of what tier 1 actually does, and it is
worth running before pushing anything that adds a module.

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

CI runs tiers 1b and 2 on `ubuntu-latest`, `macos-latest` and
`windows-latest`, and tier 1 on Linux and Windows only. k2 publishes cp312
CPU wheels for all three at the pinned torch version, so the FST tier is not
Linux-only; **check the index before bumping the pin**, since a newer build
may not cover every platform on the day. Tests must therefore not assume
POSIX: no shell pipelines, no `os.popen`, no hardcoded `/tmp`, and
`subprocess` only via `sys.executable` with an argument list.

**Tier 1 does not run on macOS, and the reason is allocation rather than
coverage.** On 23 November the macOS pool failed to assign a runner four
times in one day — twice in a single run — each time leaving the job
`cancelled` rather than failed, with `runner_name: ""`, zero steps and
exactly 15m02s of queueing, and each time costing a re-run before a green
PR could merge. Four `macos-latest` tickets in the matrix is four chances
of that per push, and tier 1's was the cheapest to give up: it imports
nothing heavier than the standard library, pytest and pyyaml, while tiers
1b and 2 still cover macOS carrying torch and k2, which is where a
platform difference can actually bite. Windows is **not** interchangeable
with it — `as_posix()` and the `bash` probe below are three real tier 1
breakages that only Windows showed, and no other tier runs there at this
tier's cost. If the pool settles, putting it back is one line; the thing
not to do is make the macOS jobs `continue-on-error`, which would swallow
a real regression along with the allocation faults.

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
  That holds for **mini-librispeech's 33MB `L.fst` too** — 1,539,598 lines
  from a 206,510-entry lexicon, with optional silence and position-dependent
  phones, byte for byte. Four things had to be right and three were not
  guessable:
  **the script is the Python one, not the Perl.**
  `utils/lang/make_lexicon_fst.py` and `make_lexicon_fst.pl` build
  *different* FSTs: the Perl puts `silphone:<eps>` on the start arcs and
  skips the silence split after a silence word, the Python puts
  `<eps>:<eps>` and always splits. Reading the Perl first would have shipped
  the wrong shape; Kaldi's captured output is what settled it, showing
  `0 2 0 0`.
  **Weights are float32.** `BaseFloat` is single precision and `fstprint`
  writes nine significant figures, so `-log(0.5)` is `0.693147182` and not
  `0.6931471805599453`. Computing in double and writing the double is a
  correct lexicon that is not byte-identical. This could not have come up
  earlier: with `--sil_prob 0.0` every weight is zero and `fstprint` omits
  it.
  **Phones are position-dependent unless told otherwise**, and not being
  told is the default — mini-librispeech's lexicon says `AH0` while its
  `phones.txt` holds only `AH0_B`, `AH0_E`, `AH0_I`, `AH0_S`. yesno passes
  `--position-dependent-phones false`, which is exactly why a builder that
  ignored this reproduced yesno's bytes and could not build
  mini-librispeech's at all — a result that looked like success while
  covering one of two cases. The **optional-silence phone stays unsuffixed**
  while the silence *word* takes `SIL_S`.
  The payoff is that a 33MB artefact whose bytes are not committed is gated
  in tier 1 by its digest plus the inputs beside it, which is what the
  digest-only rule and the capture-the-input rule were both for.
  **`add_lex_disambig.pl` has one rule a reimplementation will drop.** The
  empty pronunciation gets a symbol **reserved to it alone**, and every
  per-pronunciation counter then steps *over* that number. Without the
  reservation the symbols still look plausible and the highest comes out too
  low, which puts the wrong symbol on `L_disambig.fst`'s silence arc — four
  stages from where it would be noticed. The algorithm also runs **after**
  position-dependent suffixing, which changes the answer: `x` and `x y` are
  prefix-related while `x_S` and `x_B y_E` are not. It is checkable from
  committed data without rebuilding any FST, because `disambig.txt` is
  `#0 .. #ndisambig` and `ndisambig` is the algorithm's answer plus one for
  silence — so the symbol *count* pins the return value. mini-librispeech's
  `disambig.int` holds 17, so the answer must be 15.
- **`fstaddselfloops` is C++ but its rule is small**, and
  `L_disambig.fst` is byte-exact too. `AddSelfLoops` in
  `fstext/pre-determinize-inl.h` loops **every final state and every state
  with a non-epsilon *output* label on at least one outgoing arc**, with
  weight One. For a lexicon FST that is a single state: every arc carrying
  a word leaves the loop state, and the loop state is the only final one,
  so mini-librispeech's 206,510 entries gain exactly one arc,
  `1 1 347 200004`. Checking that as a *count of one* is far easier than
  checking it as a structure.
  The whole chain is one call, `build_disambig_fst_text`, because the four
  steps are not separable: `make_lexicon_fst.py` needs the silence
  disambiguation symbol's number and that comes out of `add_lex_disambig`.
  A cross-check worth repeating when something here changes:
  `L_disambig` has 1,589,729 lines against `L`'s 1,539,598, and the
  difference of 50,131 is 50,129 disambiguated entries plus one self-loop
  plus the one bridge arc the silence symbol adds.
  **A large FST needs a functional fingerprint, because its digest cannot
  serve.** The two decisions above conflict: an exact artefact over a
  megabyte is recorded by digest alone, and a digest answers only the byte
  question — but `G.fst` is *known* not to be byte-equal, so its digest can
  only ever mismatch and says nothing. On mini-librispeech, the one corpus
  where a graph replacement matters, that left no usable check at all. So
  the capture also records what weights the FST gives a fixed set of word
  sequences, sampled with a fixed seed from `words.txt` so both
  implementations are asked the same questions, and
  `BenNevis.utils.fingerprint` is a few kilobytes against 61MB. This works
  for `G` because a back-off LM FST accepts every sequence of in-vocabulary
  words; `L`, `LG` and `TLG` accept only what their lexicon permits, so a
  sequence sampled from `words.txt` is rejected almost surely and a
  fingerprint of rejections compares nothing.
  **`sample_lexicon_sequences` closes that for the phone-input graphs**,
  and `LG` now has a fingerprint. It draws word sequences from the words
  the *lexicon* pronounces and expands each with the project's own
  `position_dependent`, so the suffixing cannot drift from the lexicon FST
  it probes; position dependence is read off the phone table rather than
  configured, since the two recipes differ. The first pronunciation wins,
  because a word with alternatives would otherwise make the expected phone
  sequence depend on which was picked.
  This is what the big corpus had been missing: the yesno equivalence test
  reads Kaldi's committed `LG.fst.txt` directly, which works only because
  yesno's is 20KB, while mini-librispeech's is 138MB and kept as a digest.
  Verified end to end on yesno, where both sides exist: our Python `LG`
  agrees with Kaldi's on every accepted sequence, **worst gap 0.000896
  against a delta of 0.0009766**, and perturbing our arc weights by 0.05
  fails the gate at 0.249552. Note a sequence the lexicon permits can
  still be rejected by `G` -- yesno's grammar is restrictive and about a
  third of the draws survive -- which is why only the accepted ones are
  recorded.
  **mini-librispeech's gate cannot run in tier 2**, because its ARPA is
  downloaded rather than committed, so our side cannot be rebuilt offline.
  It runs in the nightly instead, as `tests/reference/compare_graphs.py`
  from a step beside the capture: `LG` on lexicon-drawn phone sequences
  and `TLG` on word sequences, both against Kaldi's own graphs dumped
  with `fstprint`. The step is `always()`, because the graphs are built
  in stage 4 and a run that died in training still has graphs worth
  comparing -- which is the cheapest way to get this measurement.
  Two things about it to keep in mind. It is **`continue-on-error` for
  now**, because the cost of `remove_epsilon` on a 43M-arc `TLG` is
  unmeasured; a check that cannot fail the build is a report rather than
  a gate, so turn it blocking once a run has shown what it takes, and
  `--skip-tlg` is the escape if that half proves unaffordable. And it
  **rebuilds the lexicon**, so `--sil-prob 0.5` and
  `--position-dependent-phones` must match the recipe's `prepare_lang.sh`
  call exactly -- the same trap as stage 10, where a mismatch compares
  the Python graph against a Kaldi graph built from a different lexicon
  and both numbers look plausible.
  `tests/recipes/test_nightly_graph_gate.py` reads those flags out of the
  workflow and out of `run.sh` and checks them against each other, so
  neither can drift; its first version compared the graph directory as a
  literal and failed, because the recipe interpolates `${topo}`.
  This is a different standard from CMVN's deliberately. There, the object is
  an array and any difference is a real difference, so byte-exactness was
  both achievable and the right test. Byte-exactness is still expected where
  it already holds — the symbol tables, `disambig.int` and the topology FSTs,
  all of which are canonical rather than arbitrary.
- **WER**: the backstop, not the primary tool. Gate on a delta, not equality.

**One captured artefact is not byte-reproducible, and it was found by
capturing twice.** Two runs of the same commit in the same image produced
`extra_questions.txt` with the same five lines in a *different order* —
Kaldi's `prepare_dict.sh` builds it by iterating a Perl hash, and Perl has
randomised hash order per process since 5.18. Its lines are reproducible
and its bytes are not, so `capture.py` **sorts them before storing and
digesting**, which keeps the stored bytes and the recorded digest in
agreement. That is a normalisation of how the reference is read, like
having Kaldi rewrite the feature archives uncompressed — not an edit to
make a test pass, and the recipe is untouched.
Nothing depended on it: it feeds the decision-tree questions in GMM
training, not the lexicon FSTs. It was fixed anyway because the failure it
causes is a test that fails on a **re-capture** rather than on a change,
which is the hardest kind to read. **Comparing two captures is worth doing
deliberately** — it is the only check that finds this class of thing, and
everything else agreed: 20 of 21 exact digests, all 87 array statistics,
and the `G.fst` fingerprint.
Normalise narrowly. `lexicon.txt`'s order decides `L.fst`'s state
numbering, so sorting *that* would silently change the FST.

**Capture a pipeline's intermediates, not only its ends.**
`L_disambig.fst`'s digest covers four stages, so a mismatch says something
is wrong and nothing about which. With `tmp/lexiconp.txt` and
`tmp/lexiconp_disambig.txt` captured, the two stages that have inputs are
checked **separately and each on Kaldi's own input to it** — so an error in
position-dependent suffixing cannot make the disambiguation look wrong, or
hide a second error that cancels it. Both match entry for entry across
206,510 entries.
Note `tmp/lexiconp.txt` is *not* a copy of `dict/lexiconp.txt`: it is the
position-suffixed version, which is what makes it the boundary between the
two stages.
**yesno's reference predates all of this** and so skips nine checks the
mini-librispeech one runs. Re-capturing it is the cheap way to close that,
and worth doing before relying on either corpus alone.

**A committed reference that nothing reads is not a test.**
`tests/reference/test_corpora.py` discovers every corpus under
`tests/golden/reference/` by its manifest and runs the corpus-agnostic
checks over all of them, because `test_against_reference.py` has yesno's
path as a module constant — so committing mini-librispeech's 9.9MB would
otherwise have added data no test looked at. Discovery rather than a list,
so committing a capture is enough to put it under test.
A related trap, and one that shipped: the `G.fst` gate added in #41 built
its path as `REFERENCE_DIR / "yesno" / ...` while `REFERENCE_DIR` already
ended in `yesno`, so it skipped unconditionally — **a test that could not
fail**. It now asserts that it compared something, which is the only way a
skip-shaped bug announces itself.

**Kaldi's decoding graph weights are quantised to 1/1024, and ours are
not.** `fstminimizeencoded` rounds every weight to a multiple of OpenFST's
`kDelta`: yesno's `G.fst` carries `2.30258512` for `-log(0.1)` and its
`LG.fst` carries `2.30273438`, which is exactly `2358/1024`. So each arc of
a Kaldi-built graph loses up to `kDelta/2`, about 1.5e-4, and a path through
*n* arcs can differ by `n * kDelta / 2`. The k2 graph keeps full precision —
**Kaldi is the less precise side here**, and matching it would mean
deliberately rounding to Kaldi's grid. `QUANTISATION_DELTA` in
`BenNevis.decode.graph` exists to *bound a comparison*, never to reproduce
the rounding. This is the clearest example in the project of an equivalence
gate that would turn an accident into a requirement.

**`TLG` is now gated as a function, and the probe runs on the output
side.** The question a decoder asks is what it costs to recognise a word
sequence, which under a CTC topology is a **minimum over every token
realisation** of those words -- blanks anywhere, labels repeated. So
`prepare_output_probe` inverts the graph and `weight_of_word_sequence`
intersects with the words. Probing one hand-picked token sequence instead
would compare an alignment no decoder need take, and two graphs could
agree on it while differing on the paths that matter. It also sidesteps
the token tables entirely, since `words.txt` is the same on both sides.
Two traps in that, both hit: **`k2.invert`, never `Fsa.invert`**, because
`compose_lg` ends in `remove_epsilon` and the method refuses a ragged
`aux_labels`; and the epsilons must be **removed before intersecting**,
because most of a token graph's arcs emit no word (20 of 29 in yesno's
`TLG`) and as input labels those form cycles, including blank self-loops
-- `k2.shortest_path` requires an acyclic FSA and fails with
`dest_state > state_idx01`, which says nothing about epsilons.
Measured on yesno, where Kaldi's `TLG` is committed: **ours is 11 states
and 33 arcs against Kaldi's 10 and 29**, and the weights agree on every
word sequence, worst gap 0.000746 against a delta of 0.0009766.
Mutation-verified at 0.249701. The cost of `remove_epsilon` on
mini-librispeech's 43,263,880-arc `TLG` is **not yet measured** and may be
the thing that forces a Dijkstra over (state, position) pairs instead,
which tolerates cycles because the weights are non-negative.

**`TLG.fst`'s only consumer is Kaldi's `decode-faster`.** Training never
reads it — that path goes through `k2/topo.yaml` and is already Python. So
porting the graph alone buys nothing: a Python-built text FST still needs
`fstcompile` to become the binary the decoder wants, and Kaldi and OpenFST
both stay. The graph and the decoder have to go together, and doing so drops
`fsttablecompose`, `fstdeterminizestar`, `fstminimizeencoded`,
`fstrmsymbols`, `fstrmepslocal`, `fstarcsort`, `fstcompile` and
`decode-faster` at once, along with the 334MB artefact. `int2sym.pl` and
`sclite` fall out too, because `BenNevis.utils.scoring` is already Python.

**What that chain does is written down in `docs/kaldi-graph-chain.md`, read
off the Kaldi and OpenFST sources rather than inferred.** Read it before
touching `BenNevis/decode/graph.py`. Four things in it are not guessable
from the shell, and the Python currently differs on all four:
`fstdeterminizestar --use-log=true` determinizes **in the log semiring**, so
merged paths are log-added where k2's tropical `determinize` keeps the
cheapest; the *star* is epsilon removal folded into the subset
construction, which k2's has no equivalent for; `fstminimizeencoded`
minimises *and* is where the 1/1024 quantisation enters, and we do not
minimise at all; and the **disambiguation symbols are carried through the
T composition and dropped afterwards**, because they are what makes T∘LG
determinisable — `compose_lg` strips them from LG first.
**Minimisation does not need a deterministic input, which is the opposite
of what the pipeline suggests.** OpenFST's `AcceptorMinimize` uses Revuz
only for `kAcyclic | kIDeterministic` and otherwise falls back to
Hopcroft, which its own comment says "also works for non-deterministic
FSTs"; `Minimize` allows it in any **idempotent** semiring, and tropical
is one. So dropping determinization did *not* put minimisation out of
reach -- the two are separable, and `fstminimizeencoded` is applicable to
the undeterminised LG we ship. It is still not implemented, but **the
arithmetic has changed now that the counts are measured, and the figure
this entry used to give was wrong.** Run 45501684352 recorded Kaldi's
graphs: `LG` 1,383,213 states and **4,367,485 arcs**, `TLG` 3,469,109 and
**12,711,953**. Against our 6,111,314 and 43,263,880 that is **1.40x on
LG and 3.40x on TLG** -- not the "about 5%" claimed here before, which
came from dividing `LG.fst.txt`'s 138,155,979 bytes by a guessed line
length and was **33% too high**. `shapes` was added for exactly this
reason and caught exactly this.
So minimisation buys about **29% on LG** and more on TLG. It remains a
*cost* question rather than a correctness one, since
`intersect_dense_pruned` needs the graph neither deterministic nor
minimal -- but it is a real cost on the graph the decoder loads, and the
1.40x against 3.40x split puts most of it in the **topology
composition** rather than in the lexicon and grammar, which points at the
second pipeline's epsilon-pair merging and minimisation rather than at
determinization. `L_disambig` agrees with Kaldi to the arc, which checks
the counting itself. `fstminimizeencoded` also deliberately avoids the
weight pushing that OpenFST's public `Minimize` does; a replacement must
inherit that. Its quantisation rule is
`floor(w / delta + 0.5) * delta` in float32, which is what puts weights
on the 1/1024 grid and reproduces yesno's 2358/1024 exactly from
-log(0.1).
**That cost has now been paid in full once, and it is large enough to
change the priority.** Run 45511013199 reached stage 10 and decoded
mini-librispeech's `dev_clean_2` with k2 against the 43,263,880-arc
`TLG`: **900 of 1,089 utterances in 4h16m48s**, which is 17.1 seconds an
utterance, and the run was cancelled at the 350-minute ceiling before it
finished. Kaldi's stage 7 did the whole 1,089 with `decode-faster` at
`nj=8`, and stages 6 to 9 together fit in the twelve minutes between the
end of training and the start of stage 10. Building the graph is not the
problem -- `build_decoding_graph` takes 12.2s and the whole stage-10
chain from lexicon to graph takes 35s. The *search* is, and it is being
run against a graph 3.40x the size of the one Kaldi's decoder loads.
So minimisation stops being a tidiness question: **mini-librispeech
cannot be decoded with k2 inside a CI run as things stand**, and that
blocks the comparison stage 10 exists to make. Two things are worth
separating before anything is optimised, because they are not the same
fix: the graph is larger than it needs to be (minimisation, epsilon-pair
merging), and the search is run one utterance at a time where Kaldi's
runs eight jobs. The second is the cheaper to try, and it is already
half done: `decode` takes a batch `(N, T, C)` and
`intersect_dense_pruned` is built for one, but `decode_many` calls it
with `scores.unsqueeze(0)` per utterance, so the batch dimension is
always 1. Neither is measured against the other yet, and the order to
measure them in is batching first, since it needs no new graph
operation.
**`compose_lg` now skips determinization altogether, and that is measured
rather than conceded**: `k2.determinize` costs 3358s on mini-librispeech
and returns 3.4 times the arcs, against 19s for Kaldi's whole first
pipeline, which determinises in the log semiring *and* minimises. Skipping
it costs 1.5s, and `compose_lg` runs in 2.9s.
`intersect_dense_pruned` needs no deterministic graph, so the cost is in
the weights and not the search: Kaldi log-adds paths sharing an input
string, we keep the cheapest. One-best picks the same path; a lattice
would not. Also worth
knowing: `fstrmepslocal` removes *some* epsilons under a guarantee never to
grow the graph, while `k2.remove_epsilon` removes all and can -- and
`fstrmepslocal` is **not only epsilon removal**: it also merges
`(x:eps)(eps:y)` arc pairs into `(x:y)`, which k2 has no counterpart for
and which is part of how the stage shrinks a graph with few input
epsilons to remove. With no flags it is `RemoveEpsLocalSpecial`, which
keeps equivalence in the tropical semiring and stochasticity in the log
sense; the header says this cannot be had by casting, because combining
identical arcs would then break equivalence in the original semiring.
`fsttablecompose` is the one stage that is only an optimisation, so
`k2.compose` is a faithful replacement for it — and at 1.9s on
mini-librispeech it was never the problem.

**An FsaVec's arcs are three levels deep, not two.** `fsa.arcs.shape()` is
FSA, then state, then arc, so `row_splits(1)` gives *state* boundaries per
FSA and `row_splits(2)` gives arc boundaries per state. Slicing per-arc data
such as `aux_labels` with `row_splits(1)` runs one utterance's labels into
the next: a batch of two decoded as `[[1, 2], [2]]` where the answer was
`[[1], [2]]`. A batch of one never shows it, which is why the decoder's
tests batch two.
`aux_labels` is also a plain tensor on some graphs and a `RaggedTensor`
once composition has put several output labels on an arc, and which one
depends on how the graph was built rather than on anything the caller
chose — so both have to be handled.

**The equivalence gates are scaffolding, and retiring them is a decision.**
Everything that compares against `tests/golden/reference/` exists to answer
one question — *does the Python do what Kaldi did?* — and that question
stops being asked, per stage, the moment Kaldi no longer owns that stage.
Keeping those gates past that point is not caution, it is harmful: they pin
the Python to reproducing Kaldi's accidents. `format_weight` narrows to
float32 **only** because Kaldi's `BaseFloat` is single precision; once
nothing compares against Kaldi, computing in double is strictly better and
the byte-exact test would forbid it. The same goes for `arpa2fst`'s state
numbering and `fstprint`'s formatting.

So retirement is per stage, deliberate, and recorded in the PR that does it
— never by deleting a test that has become inconvenient, which is the same
failure as raising `max_wer` to make a run pass. Three things survive the
migration and must not be swept up with the rest:

- **The WER ledger.** It is not an equivalence check against Kaldi's
  *implementation*; it is a regression gate on the system's output. It stays
  permanently, with `reference_wer` re-anchored to the Python baseline once
  that is the baseline.
- **Behaviour tests that happen to have been validated against Kaldi.** The
  back-off rule, "the word rides only the first arc", "`</s>` is a final
  weight and not an arc" — these say *correct*, not *same as Kaldi*, and
  they keep their value when Kaldi is gone. The test to apply is whether a
  test would still mean something if Kaldi had never existed.
- **The captured reference data itself.** Deleting a test is cheap and
  reversible; the capture is not reproducible once the image stops
  building, and Kaldi is bit-rotting. It is the only record of what the
  pre-migration system did, and it costs nothing to keep.

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

**The build keeps no layer cache, and the arithmetic is why.** Measured on
the 21 November build: the layers a cache could save are apt at 16.4s, the
venv at 2.6s, torch and k2 at 27.6s and the remaining dependencies at
11.1s — 58 seconds in total — while `cache-to: type=gha,mode=max` spent
**209.9s** exporting and held **5.22GiB**, which was 88% of everything
cached for this repository against a 10GB budget the nightly's corpus
caches now want. The Dockerfile is single stage, so `mode=max` bought
nothing over `mode=min` and the export covered the whole image including
the 2.1GiB Kaldi base — a base pinned by digest that comes from Docker Hub
in 67.6s with or without a cache. The Actions cache is **scoped per
branch** as well, so a pull request touching `docker/` exported its own
2.61GiB copy that master could never read; that is how two identical
copies came to exist. Dropping it took the build from 6m09s to about
2m39s. `tests/docker/test_build_cache.py` pins it, because an eviction
caused by this would show up as a nightly quietly re-downloading 1.2GiB
rather than as a failure. A registry cache is not forbidden — it does not
touch the Actions budget — but 58 seconds is the number to beat.

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
- **The mini-librispeech trainer leaks memory, and it is not yet known
  what.** Three `ctc` runs were SIGKILLed on a 15.6GiB runner -- epoch 4,
  epoch 4 and epoch 7, so the epoch is not the pattern, the race between
  growth and the epoch count is. The host-side sample from run
  45337445608 is the evidence: the trainer climbs to about 7.4GiB during
  epoch 1, plateaus there for four epochs, then rises again to 9.3GiB and
  is killed with 9MiB of the host left, after which 8.7GiB comes back. A
  plateau that resumes rising is a leak rather than a cache filling.
  **The loss path is measured clean and so is not the cause**: 2,000 steps
  of `GraphLoss` with the real `compile_training_graph`, with fixed and
  with varying batch shapes, hold flat at about 400MiB. `k2.Fsa.to()`
  returns `self` on the same device, so the per-step `.to()` in
  `compile_training_graph` allocates nothing, and `kaldiio`'s `LazyLoader`
  caches no arrays. `k2.arc_sort` on an already-sorted topology is a
  no-op, and rebuilding the denominator `FsaVec` each step costs 13ms for
  a 350-token topology -- 12 seconds an epoch against 34 minutes, so
  caching it is not worth the unbounded cache that dynamic batch sizes
  would need. All of that was measured before being believed, and none of
  it explains the growth.
  What the log could not say is *what* is held, which is why
  `BenNevis.utils.memory` reports the resident size, the live tensor
  storage and **the types that gained instances since the previous
  epoch** once per epoch. The reading is a difference deliberately: a
  trainer legitimately holds a lexicon FST and an optimiser state, so an
  absolute count says nothing and a type that rises every epoch and never
  falls is the answer.
  **That reading is now in, and it says no Python type is growing.** Run
  45511013199, two epochs on a 15.6GiB runner: resident **808MiB,
  7894MiB, 9130MiB**, against live tensor storage of **130MiB, 169MiB,
  169MiB** and a tracked object count of **960,770, 961,347, 961,127** --
  flat within four hundred across 8.3GiB of growth. The named growers are
  three-figure instance counts (`StorageWeakRef` +116, `Tensor` +108,
  `UntypedStorage` +108), which is noise at this scale and not 8GiB of
  anything. So the type-growth column has answered the question it was
  built for, in the negative: the memory is **held below Python**, where
  neither `gc.get_objects()` nor `tracemalloc` can see it, and a fix
  guessed at the Python level would be a fix to the wrong layer.
  What that leaves is the allocator, and the next reading is one
  measurement rather than a theory. glibc serves large blocks from
  per-thread arenas and never returns a free block in a secondary arena
  to the kernel of its own accord, so a workload that allocates and frees
  large arrays across many threads -- k2's intersection and torch's CPU
  kernels both -- can carry a resident size made entirely of memory it
  has already released. `trim_malloc` asks glibc to give it back, the
  per-epoch reading now takes a resident size either side of that, and
  the two cases are distinguishable from one line: a resident size that
  **falls** is a held free heap, and the remedy is `MALLOC_ARENA_MAX` or
  a periodic trim rather than anything in the trainer; one that does
  **not** fall is memory genuinely still live in C++, which localises the
  search to k2 and torch and justifies an allocator-level tool.
  The probe returns **None off Linux**, deliberately: macOS, Windows and
  musl have no `malloc_trim`, and a probe that answered False there would
  read as "the memory is in use" on a platform where the question was
  never put.
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
- **The decoding graph's token table is read, never recomputed — and
  getting that wrong is what the first real k2 decoding run failed on.**
  `prepare_topo.sh` writes `k2/tokens.txt` with **neither `<eps>` nor the
  disambiguation symbols**, and `Lang` takes the network's output size and
  indices from it. Deriving the table instead with `build_tokens`' defaults
  gave ten labels for yesno's CTC topology where the network emits seven,
  **every one shifted by one**: the decoder read `<blk>` as `<eps>` and `Y`
  as `<UNK>`. k2 then failed inside `intersect_dense_pruned` with
  `num_states_final_t <= 1 (2 vs. 1)`, which says nothing about labels.
  The k2 variant's *output* side is `k2/phones.txt`, phones without the
  disambiguation symbols, which is right because `compose_lg` strips them
  from LG before the topology is composed on.
  **The two token tables are not interchangeable, and the ids differ by
  one.** `prepare_topo.sh` writes `tokens.txt` *with* `<eps>` for Kaldi's
  `T.fst` and `k2/tokens.txt` with neither `<eps>` nor the disambiguation
  symbols for ours. Measured on mini-librispeech: `<blk>` is **1** in
  Kaldi's table and **0** in ours, `SIL` 2 and 1, and so on all the way
  down. So **any comparison of `TLG` must resolve token names through each
  side's own table** -- comparing raw ids would line `<blk>` up against
  `SIL` and measure nothing while looking fine. This is the same shift
  described above, approached from the other direction, and all three
  tables are now in the capture so a comparison need not rebuild them.
  **The test could not have caught it**: it generated its own log
  probabilities from the same derived table, so the graph and the acoustics
  were wrong together and agreed. A test that derives its inputs from the
  code under test proves only that the code is self-consistent. The fixture
  now builds a language directory the way `prepare_topo.sh` does and reads
  the table off disc, and one test asserts
  `graph.labels.max() == len(tokens) - 1`, which is the invariant that
  would have failed immediately.
- **`bin/align.py` crashed on every real corpus, and the reduction of
  `aux_labels` is why.** `best_path.aux_labels.numpy()` works while
  `aux_labels` is a plain tensor and raises `AttributeError` once it is a
  `RaggedTensor`, which took out stages 8 and 9 of the first real
  mini-librispeech run, all eight jobs, both before and after decoding.
  yesno's four-word lexicon leaves it dense, so tier 3 — **the only tier
  that reaches `bin/align.py` at all** — stayed green through it.
  **`k2.remove_epsilon` is what makes it ragged, not `k2.compose`**, which
  was measured after assuming otherwise: composing two transducers returns
  a plain tensor, while removing an epsilon-input arc that carries a word
  has to put that word on a surviving arc. A test named for the
  composition had been standing in for this and asserted nothing of the
  kind.
  There are **two** reductions and they are not interchangeable.
  `words_from_best_path` flattens, which answers *which words were said*.
  An alignment reads `labels[i]` and `aux_labels[i]` together to decide
  which frames a word covers, so it needs exactly one entry per arc and a
  length matching `labels` — that is `aux_labels_per_arc`. Two words on
  one arc raises there rather than dropping one, because dropping one
  shifts every word after it and the result looks plausible.
  One asymmetry is left in deliberately: `remove_epsilon` empties the
  final arc's row instead of leaving `-1`, so the ragged path reports `0`
  where the dense path reports `-1`. Normalising would change the ark
  yesno already writes, so both conventions stand and **the CTM from the
  next real run is what should settle it**.
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
- **The topology FSA is not arc-sorted, and `get_den_graph` now sorts it.**
  Fixed, with the premise and the consequence both pinned: one test asserts
  the topology is *not* `ARC_SORTED` on its own, so the fix cannot become
  silently redundant, and one asserts the denominator scores are **exactly**
  unchanged by sorting, so the WER ledger cannot move. **CUDA is still
  unchecked** — there an unsorted graph is undefined rather than merely
  slow, and these tests run on the CPU. The original note follows.
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
- **The remaining POSIX-only sites.** `Lang.load_lexicon` shells out to
  `fstprint | awk`; the `awk` half is now `BenNevis.decode.graph`'s
  `load_openfst_text`, but the `fstprint` half needs the binary `L.fst`, and
  getting off that means a language directory carrying `k2/L.fst.txt` the way
  it already carries `k2/topo.yaml` — which needs the dictionary at
  preparation time, so it is a larger piece of goal 2.
  `run/predict.py`'s `cat | awk` merge **is now Python**
  (`BenNevis.utils.merge`, fifteen tier 1 tests), and replacing it fixed a
  latent bug rather than only porting one: a shell glob is lexicographic, so
  eleven ranks merged as 1, 10, 11, 2, 3 ... Nothing downstream depended on
  it, because both files are keyed by utterance, but it was wrong.
  `bin/align.py`'s `gzip -c` pipe is **still shell**, and deliberately: a
  tested replacement would need `kaldiio`, which tier 1 does not install
  (it installs `pytest` and `pyyaml` only), so the check would be tier 3 —
  and changing it lands in the *next* nightly rather than the one in
  flight. `Lang.load_topo` keeps a `fstprint`
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
- **Stage 10 rebuilds the lexicon, so its settings must match the
  recipe's `prepare_lang.sh` call — and the two recipes differ on both.**
  yesno passes `--position-dependent-phones false --sil_prob 0.0`;
  mini_librispeech passes neither, so `prepare_lang.sh`'s defaults apply
  and stage 10 has to say `--sil_prob 0.5 --position_dependent_phones`
  explicitly. Getting either wrong builds a lexicon that is **wrong rather
  than broken**: the graph composes, the decoder runs, and the
  transcriptions are quietly worse.
  `tests/recipes/test_decode_k2_settings.py` reads both the stage-10 flags
  and the `prepare_lang.sh` line out of each `run.sh` and checks they
  agree, so the table is pinned against the source of truth rather than
  against itself; it also asserts its own list of recipes matches what is
  on disc, so a recipe gaining stage 10 cannot go unchecked.
  Rebuild from **the dictionary `prepare_lang.sh` read**, not from the
  recipe's input copy: yesno's `prepare_dict.sh` copies
  `input/lexicon.txt` to `data/local/dict/lexicon.txt`, so the two agree
  today and would diverge silently the moment a preparation step
  transformed the lexicon.
- **We strip the disambiguation symbols by threshold where Kaldi uses a
  list, and that is an assumption with a silent failure.** `fstrmsymbols`
  takes the explicit `disambig.int` and, with no options, replaces those
  labels with `<eps>` on the **input** side -- a relabelling, not a
  deletion, since `--remove-arcs=true` is a different graph. `compose_lg`
  instead zeroes every label at or above `first_disambig`, which agrees
  only while the `#` symbols are a contiguous suffix of `phones.txt`.
  They are: yesno 8..9 of 10, mini-librispeech 347..363 of 364. Were an
  ordinary phone ever numbered above them it would be mapped to epsilon,
  and the graph would still compose, the decoder would still run and the
  transcriptions would be quietly wrong -- the same "wrong rather than
  broken" shape as the stage-10 lexicon settings. `build_graph` refuses
  such a directory now, and the test is pinned in both directions,
  because a guard that rejected every language directory would also pass
  a one-sided test.
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
- **`grep` in a `pipefail` step fails the step when it matches nothing.**
  The nightly's summary ran `grep -E '\| *Sum/Avg' "$f" | awk ...`, which
  was fine while every results file was sclite's. The k2 decoder's are not —
  they come from `BenNevis.utils.scoring` — so the grep matched nothing, the
  pipeline failed, `bash -e` aborted, and **a successful recipe reported
  failure twice, including a scheduled run**. `awk '/pattern/ {...}'` treats
  no-match as empty output rather than as an error, which is what a
  best-effort parser wants. The step now also says how many files it
  summarised and how many it could not parse, for the same reason the file
  count exists.
  `tests/recipes/test_nightly_scores.py` lifts both parsers out of the YAML
  and runs them under `set -eo pipefail`, the way
  `tests/recipes/test_decode_faster.py` does with `decode_faster.sh`'s awk —
  a parser reimplemented in the test would have passed while the shipped one
  failed.
  **sclite's last three fields are Err, S.Err and a bar**, so `NF-2` is the
  word error rate and `NF-1` is the *sentence* error rate. Reading the wrong
  one gives 3.4 where the answer is 0.4 — a plausible-looking number, and a
  mistake made once while comparing the two decoders by hand.
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
  **`utt2spk` is still missing, which is the fourth time.** Comparing CMVN
  *values* on a corpus needs to know which speaker's statistics apply to
  which utterance, and the capture records the statistics without the
  mapping. mini-librispeech's speaker happens to be the utterance id minus
  its last hyphen-separated field, but that is a LibriSpeech convention
  rather than a guarantee — yesno's is `global` for all sixty. So the
  committed mini-librispeech reference gates its *structure* and its
  digests, not its CMVN arithmetic; that arithmetic is already bit-exact on
  yesno and is independent of the sample rate, so the gap is a confirmation
  deferred rather than a question open. Add `utt2spk` to the next capture.
  The same rule puts **`prepare_lang.sh`'s dictionary in the capture**, under
  `dict/` so an input cannot be mistaken for an output. That gap was found
  by writing a Python `L.fst` and discovering the only lexicon available to
  check it against was yesno's four words.
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
- **`actions/cache` does not save when the job fails.** Its post step is
  `post-if: success()`, so the combined action caches nothing from a run
  that got the corpus and then died in training — and the retry pays for
  the download again, which is the one case a corpus cache is for. The
  split `actions/cache/restore` and `actions/cache/save` pair with
  `if: always()` on the save is the way round it, at the cost of a path
  list and a key written twice; `tests/recipes/test_nightly_cache.py`
  checks the two copies agree, since a pair that drifts restores one set of
  files and stores another and looks exactly like a cache that keeps
  missing. Nothing saves when the runner itself is lost, which is a limit
  of the mechanism rather than something the split pair fixes.
  **It had been written here that this is how run 45202289714 lost 1.2GiB
  of mini-librispeech download. That is wrong**: #57 added these steps at
  18:52 on 23 November and that run was created at 15:10, so it had no
  cache steps at all and there was nothing for a lost runner to take with
  it. The limit is real and the example was invented to illustrate it —
  which is the more useful lesson, since an illustration is exactly the
  part of an entry nobody checks.
  The first run that could use the cache was 45293910683, and it **missed**,
  paying 73 minutes to re-download at 68 KB/s from openslr before saving
  1,647MiB under `mini-librispeech-corpus-v1`. That miss was unavoidable —
  there was no entry yet — but the step still reported **success**, because
  `actions/cache/restore` succeeds on a miss and only `cache-hit` says
  otherwise. So a green restore step is not evidence the corpus came from
  the cache, and the download in the log is the only thing that settles
  it.
- **"The hosted runner lost communication with the server" is an
  annotation, not a log line.** `gh run view --log` does not contain it at
  all; the only place it appears is the check-run's annotations, which is
  the first thing to read when a job fails with nothing in its log —
  `gh api repos/<owner>/<repo>/check-runs/<job-id>/annotations`. The job
  carries the same story in its shape: a lost runner leaves `Run the
  recipe` with `completed_at: null`, every later step never started, and a
  `conclusion` the steps themselves do not account for.
- **It is the *live* job log that is limited to about 2MB, not the finished
  one — and `wget`'s progress dots will spend that limit before a recipe
  starts.** This was recorded wrongly here twice before it was measured
  properly, so the measurements are the entry:

  | Log | Bytes | Lines | Ends at |
  | --- | --- | --- | --- |
  | run 45202289714 attempt 1, runner lost | 2,117,576 | 20,397 | mid-download, 56% |
  | attempt 2 while in progress | 2,117,517 | 20,293 | mid-download, 55% |
  | attempt 2 once completed (cancelled) | 3,009,128 | 28,812 | the end of the job |

  The first two agree to 59 bytes because both are the live buffer. A
  **completed** job — cancelled at its timeout included — serves the
  whole log: that 3MB one holds every stage, all twelve epoch lines and
  24,879 dot lines.

  **A lost runner does not reliably leave even the buffer, and this entry
  used to claim it did.** Run 45474590201 was lost at 3h36m and serves
  *nothing*: `gh run view --log` says `log not
  found` and the job log API returns `BlobNotFound`. An in-progress job
  can be the same — that run returned `BlobNotFound` while it was
  running, where 45202289714 served 2.1MB. So log availability is **not
  guaranteed at any point before a job completes**, and the claim that
  "the buffer is all that survives" was read here as a guarantee and
  acted on. That run cost three and a half hours and produced no readings
  at all, because the memory lines go to the job log and to the tee'd
  artefact and a lost runner takes both. **Anything a run must yield has
  to survive the runner being lost, which in practice means a shorter
  run**, not a better log.

  Two consequences, and the second is the one that was got wrong.
  **A log that stops mid-step says nothing about when the job stopped**, so
  reading that silence as a hang is a mistake — it was offered here as a
  symptom of the lost runner when the sufficient evidence was always the
  annotation, and it also meant that for three hours "still training" could
  not be told from "hung" while a run was in flight. And **the fix for that
  is the meter, not the tee**: `WGETRC` pointing at `.github/wgetrc`
  coarsens the dots to `dot:giga`, a line per 50MiB rather than per 50KiB,
  which turns 24,879 lines into about 25 and so fits a whole recipe inside
  the live buffer, keeping the transfer rate that showed openslr taking 25
  minutes over what usually takes 4. The tee to `artefacts/recipe.log` is
  worth its two lines but was oversold when it landed: a completed log
  already has everything, and a lost runner takes the tee'd file with it.
  What it actually buys is a searchable 90-day artefact that outlives log
  retention.
  Neither `quiet = on` nor `verbose = off` belongs in that file — the first
  drops the errors, the second drops the rate — and
  `tests/recipes/test_nightly_logging.py` forbids both, checks the tee
  lands under `artefacts/`, and checks an `always()` upload collects it.
