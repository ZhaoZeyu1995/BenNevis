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

**yesno's recipe no longer needs Kaldi to train**, verified end to end
with no `tools/kaldi` present -- see the Gotchas entry. mini-librispeech
still does, for its corpus preparation alone.

`egs/mini_librispeech/asr` exists for the measurements yesno cannot make:
yesno sits at nought to two word errors, so nothing that improves a model can
be shown to improve it, and yesno's filterbank is degenerate so nothing about
features can be judged on it either. Its stages 0–4 need no training and take
minutes, which is the path for feature and graph comparison; stage 5 onward
is **about twenty minutes an epoch on a CI runner** for five hours of audio,
so twelve epochs of one topology is four hours and all eight would be days.
It is therefore **not** part of the nightly. **It does have a WER ledger
entry now**, from run 46100161743 -- see the entry below -- so it is a
gate as well as a reference corpus, for `ctc` alone.
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
**The slow end is now the measured norm rather than the bad case, and the
runner's vendor predicts it.** Run 45715720923 took **34m05s** an epoch,
making eight epochs **272 minutes** of a 348-minute budget, and it was an
Intel Xeon 6975P-C; the 34m30s run was an Intel 6973P-C and both fast runs
were AMD EPYC. Two of each is not a law, but when a mini-librispeech run's
first epoch lands near 34 minutes the arithmetic for the rest of the job
should assume it will not recover.
**The vendor part of that is now refuted, and the "assume it will not
recover" part is confirmed.** Run 46100161743 took **36.6 minutes** an
epoch on an **AMD EPYC 7763** -- the same processor as both of the fast
runs, slower than either Intel one. So the heading above was wrong: the
vendor predicts nothing, three readings happened to line up, and the
only reliable figure is the first epoch of the run in front of you. It is
kept rather than deleted because the error is instructive -- four runs
sorted into two groups of two is exactly the amount of evidence that
invites a rule, and the rule was wrong the first time it was tested. The rest of that run is cheap and
stable: stages 0 to 4 cost **2.8 minutes** and stages 6 to 9 **7.8**, so
budget about eleven minutes for everything that is not training or stage
10. What does not fit is stage 10 — see the decode figures below — and
**the answer taken was fewer epochs, not a longer ceiling**: at four
epochs training is 136 minutes at the slow end, which leaves about 200 for
a stage 10 that needs 117.
**Validation loss was still falling at epoch 8, and that does not change
the conclusion.** It read 4.5234, 2.8455, 2.2957, 2.0596, 1.9322, 1.8689,
1.8364, **1.8100**, with the last three improvements 0.063, 0.032 and
0.026 — flattening rather than bottomed, and 1.8100 against the
twelve-epoch run's 1.8028 at the same epoch, so eight epochs reproduces
that run's best model. Four epochs does **not**: 2.0596 is a visibly worse
model, and a four-epoch run buys the decoder comparison at the cost of the
WER being a poor one. That is an acceptable trade only because both
decoders read the same checkpoint — the comparison is between searches,
not between models — and it would be the wrong trade for anything entered
in the WER ledger.
**And the first WER on mini-librispeech is recorded: 49.7%.** Kaldi's
`decode-faster` scored all 1,089 utterances of `dev_clean_2` at 49.7% on
the eight-epoch `ctc` model — 51.0 correct, 24.9 substitutions, **24.1
deletions**, 0.8 insertions, 98.3% sentence error. The deletion rate
thirty times the insertion rate is an undertrained CTC model preferring
blank, which is what five hours of audio and eight epochs of `rnnp` gives.
It was a baseline for the decoder comparison and not a ledger entry; the
entry exists now and came from a later run, and the two are comparable
in one respect worth noting -- **48.8% against this 49.7% is the cost of
dropping pitch on this corpus, and it is negative**. Same recipe, same
eight epochs, same topology, 80 dimensions instead of 83, and
`decode-faster` scored 0.9 points *better*. That is one run against one
run and not a measurement of pitch, but it is the second corpus to say
the tonal-language argument held.

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
- **`torchaudio.compliance.kaldi.fbank` is faithful to Kaldi, and the
  "1.2% residual" this entry used to describe was the reference's 8-bit
  compression.** `make_fbank.sh` writes compressed archives and
  `CompressedMatrix` quantises each column to 8 bits under its default
  `kSpeechFeature` scheme, so the captured features are Kaldi's output
  **put on a grid** -- and anything computed from the waveform was being
  compared against the grid rather than against Kaldi's arithmetic.
  Measured on yesno, by `tools/fbank/probe.py`, which implements Kaldi's
  filterbank from `feature-window.cc` and `mel-computations.cc` directly
  and needs no Kaldi:

  | comparison | mean | note |
  | --- | --- | --- |
  | that implementation vs `torchaudio` | **4e-6** | two independent readings of Kaldi's source agreeing |
  | either vs the capture | **0.0117** | 1.17% in linear energy: the figure that was chased |
  | the capture vs *the nearest grid point to it* | **6e-6** | float32 noise |

  and **53,428 of 53,440 entries have the captured value exactly equal to
  the nearest grid point**; the twelve that do not sit within a rounding
  tie of halfway. Half the median grid step is 0.0244, so uniform
  rounding would average 0.0122 against the 0.0117 observed.
  **The clue was in this file already.** The residual was recorded as
  *independent of the sample rate and of how many FFT bins a mel bin
  spans* -- 7.7e-2 max and 1.4e-2 mean on 16 kHz mini-librispeech against
  6.5e-2 and 1.2e-2 on 8 kHz yesno -- and that was written up as
  refuting the degenerate-filterbank hypothesis. Compression does not
  care about the sample rate or the conditioning, which is exactly why
  the two agreed; the observation was pointing at the answer and was read
  as ruling something out. **A difference that does not move when
  everything about the signal changes is a property of the measurement,
  not of the signal.**
  So the whole hypothesis table below was answering the wrong question,
  and the next step it named -- capturing `compute-spectrogram-feats` to
  localise the difference upstream or downstream of the mel projection --
  **is not needed**. There is nothing to localise. For the record, what it
  ruled out and how, since every line of it is still true of the
  *comparison* even though the comparison was against a grid:

  | Hypothesis | Verdict |
  | --- | --- |
  | Configuration | every Kaldi and torchaudio default already agrees — `use_power`, `use_log_fbank`, `htk_compat`, `raw_energy`, `low_freq`, `high_freq`, `povey`, preemphasis, `remove_dc_offset`, `round_to_power_of_two`, `snip_edges`. `energy_floor` differs but only matters with `use_energy=true` |
  | Framing offset | offset 0 agrees twice as well as ±1, so framing is right |
  | The `povey` window | the two definitions agree to 1.3e-16 |
  | Filterbank weights | a fixed weight error would bias each mel bin consistently; per-bin mean differences are ±0.0004 |
  | A normalisation factor | the mean difference of logs is −0.00007, so there is no offset to find |
  | Quiet-bin cancellation | the scatter is flat across frame energies |
  | The filterbank's conditioning | tested and refuted — and it was the clue, see above |

  **Six hypotheses, each tested properly and each correctly refuted,
  while the quantity being measured was not what anyone thought it was.**
  That is the shape of the error worth remembering, and it is why the
  table is kept rather than deleted. The paragraph that used to follow it
  -- that the per-frame spectrum path was the remaining suspect, and that
  capturing `compute-spectrogram-feats` was the next cheap step -- is
  gone, because there is nothing to localise.
  **The extractor is Python now, and it is the default.**
  `BenNevis.utils.features.fbank` wraps torchaudio's, and
  `run/make_fbank.py` writes an archive for a data directory. It was
  added beside `steps/make_fbank.sh` and left unused for a while,
  because switching changes the features -- Kaldi's are 8-bit compressed
  -- hence the CMVN statistics, hence every WER. **What made the switch
  safe is passing `--compress`**, which puts our archive on the same
  8-bit grid: the grid every number in both ledgers was measured on. So
  the switch is like for like by construction, and writing uncompressed
  is still a measured decision of the same kind as dropping pitch,
  deferred rather than taken. See the stage 2 entry under Gotchas.
  What the module owes is not arithmetic but **the settings**, because
  Kaldi's defaults and torchaudio's differ where it matters:
  `energy_floor` is 0.0 for Kaldi and **1.0** for torchaudio, which would
  floor every mel energy at 1 rather than at float epsilon, and `dither`
  is 1.0 for Kaldi and 0.0 for torchaudio. So it reads the recipe's own
  `conf/fbank.conf` -- the same file Kaldi's extractor reads, so the two
  cannot be given different settings -- takes *Kaldi's* defaults for
  anything the file omits, and **refuses** an option it does not pass on
  rather than ignoring it. A `--dither` above zero is refused outright.
  And the scale is pinned, because it is a plausible mistake with an
  implausible symptom: Kaldi reads a wav and does **not** normalise, so a
  waveform in [-1, 1] gives log energies about `log(32768**2)`, 20.8, too
  small.
  **What the gate needs is an uncompressed reference**, and the capture
  cannot give one while it reads what the recipe wrote -- the recipe must
  not store features uncompressed, because that would alter CMVN's
  inputs and hence the WERs. So the capture should extract features
  uncompressed *itself*, alongside, the same way it already has Kaldi
  rewrite the archives with `copy-feats --compress=false`. **That is now
  the only way it can be had at all**, because the recipe's stage 2 is
  Python, so what the capture reads is no longer Kaldi's output --
  `features_from` records which, and `compare_features.py` is where the
  live comparison went. Meanwhile the
  gate is built the other way round: the closest anything can get to an
  8-bit reference is half a grid step, so asserting closeness at that
  tolerance would pass for an implementation genuinely 0.01 wrong, and
  `tests/utils/test_features.py` asserts instead that **the captured
  value is the nearest grid point to ours** -- 99.97% of entries, a gate
  at roughly 7e-6 out of a reference quantised to 0.05.

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
  **The cost of `remove_epsilon` on a 43M-arc `TLG` is measured now, and
  the answer is that it does not fit.** Run 45554380456 printed
  `preparing the output probes (invert, remove epsilons)` and stopped
  there, inside a minute, with no traceback -- which is what a SIGKILL
  looks like, a Python exception having printed one the time before. So
  `--skip-tlg` is taken rather than held in reserve, and turning the TLG
  half back on wants the Dijkstra over (state, position) pairs instead of
  `remove_epsilon`; it tolerates cycles because the weights are
  non-negative. `tests/recipes/test_nightly_graph_gate.py` pins the flag
  with the reason, so restoring it is a decision.
  **That is done, and the TLG half runs again.** `compare_tlg` inverts
  with `k2.invert` and then *searches* with `weight_of_label_sequence`
  instead of inverting and removing epsilons, so nothing of the 43M-arc
  graph has to be rewritten; the flag stays as an escape, and the test
  now pins its **absence** with the same reason attached, because the
  state to notice is the half being off.
  Two things had to be settled before that was worth dispatching.
  **The probe's own memory had to come down, or it would have been the
  same SIGKILL from the other side.** `graph.arcs.values()[:, :3]
  .tolist()` is a list of three-element lists -- roughly 160 bytes an arc
  once each Python int and each inner list is counted, so about 7GiB for
  43,263,880 arcs, before Kaldi's 12.9M-arc TLG is read beside it on a
  15.6GiB runner. `_arc_arrays` packs the destinations, labels and costs
  into three `array.array`s at four bytes an element, 516MiB in total,
  and they still index to Python ints so the search loop is unchanged.
  Worth remembering as a shape: a probe that avoids an expensive k2
  operation can reintroduce the same limit in Python object overhead.
  **And the two routes are checked against each other on yesno**, where
  both fit: `TestTlgEquivalenceBySearch` asserts the searched weight
  equals the composed one exactly rather than within quantisation, since
  they are one graph's one minimum and any difference would be a defect
  in one of them. That is what licences using the cheap route on
  mini-librispeech, where the expensive one cannot be run to compare.
  The test asserts first that the inverted graphs **still have input
  epsilons**, because a probe that tolerates epsilons can only be shown
  to do so on a graph that has them -- the same lesson as the prefix
  test below.
  One property of the probe is worth knowing before editing it: its
  state space is bounded only because an epsilon arc does not advance
  the position and a label arc advances it only on a match. A mutation
  that advanced on epsilon made the positions unbounded and the search
  **hung** rather than failed, on yesno's 33-arc TLG. So a hang is one
  of this function's failure modes, and a mutation test on it wants a
  timeout.
  **And the verdict is in: our `TLG` is equivalent to Kaldi's too.** Run
  45705231089 compared all 24 word sequences on the output side and
  agreed on every one, within 0.0009765625 per arc, so the **3.36x TLG
  size difference is only a size difference** -- the same conclusion as
  `LG` and the one that closes the correctness question on the whole
  graph chain. Minimisation and epsilon-pair merging are now purely cost
  questions, on both graphs.
  The timings are the other half of what that run measured, and two of
  them are worth keeping:

  | Step | Cost |
  | --- | --- |
  | `k2.invert` on Kaldi's TLG / ours | 0.5s / 4.1s |
  | `build_graph` for our TLG | 38.6s |
  | `fstprint` and parse of Kaldi's TLG | 24.0s |
  | searching 24 sequences, both TLGs | 113.0s |
  | the whole job, `stop_stage=4` | 10m42s |

  Run 45711798591, with the pack hoisted, searched the same 24 sequences
  in **51.8s** and packed both TLG probes in 0.8s, the whole job taking
  8m21s. **That 54% is mostly a faster runner, and the hoist is worth
  about 21s of it.** Every unrelated step moved too -- our TLG built in
  28.0s against 38.6s and Kaldi's loaded in 14.4s against 24.0s -- so
  this runner is some 35% quicker, which puts the like-for-like
  baseline near 73s and the saving near the ~19s that 48 packs at 0.4s
  a graph predicts. Reading the headline as the hoist would have been
  the same error as the 20x claim above, which is only safe because the
  graphs and sequences were held identical for it. The
  `packed both TLG probes` line exists so the next run need not infer
  the pack from a total.

  **`k2.invert` was never the expensive part** -- the entire cost of
  `prepare_output_probe` was `remove_epsilon`, so the half that would
  not fit became a two-minute search by dropping one call.
  **And the memory fix turned out to be a 20x speed fix, which says
  where the old time went.** The LG half fell from 148.9s and 125.7s to
  **6.2s** on identical graphs and sequences, with nothing about the
  search changed. So the old figure was almost entirely `tolist()`
  building millions of Python ints and inner lists, paid afresh on each
  of 48 calls; `frombytes` costs milliseconds. The lesson is the one
  this file keeps relearning from the other direction: a cost attributed
  to an algorithm was in the marshalling around it, and the way to tell
  was to change only the marshalling.
  That also exposed a repeat of the same mistake one level up: the probe
  packed its arrays **per question** rather than per graph, which is 48
  repacks of graphs that do not change. `LabelProbe` holds the pack and
  `weight_of_label_sequence` is now the one-off wrapper, with the count
  pinned by monkeypatching `_arc_arrays` rather than by timing, since
  neither figure is visible on a graph small enough to test with.
  **That crash also hid the LG verdict, which is a defect in the gate
  rather than in the graph.** Mismatches were printed only after both
  halves had run, so a kill in the expensive half took the cheap half's
  result with it. Each half now reports as it finishes.
  **And the first LG verdict was `compared LG on 0 sequences`, because
  Kaldi's LG has input epsilons and ours does not.** `compare_lg` passed
  `treat_epsilons_specially=False`, under which label 0 is an ordinary
  symbol, so a single input epsilon anywhere on a path makes a linear
  probe reject it. Kaldi's LG keeps epsilons by construction --
  `fstrmepslocal` removes only what it can remove without growing the
  graph -- while `compose_lg` ends in `remove_epsilon`, so Kaldi rejected
  every drawn sequence and the gate compared nothing. **yesno could not
  have caught it**: its LG has no input epsilons on the surviving paths,
  so both settings agree there and the gate looked verified on the one
  corpus where both graphs are committed. Verified mechanically instead,
  on a three-arc pair built by hand: with the flag False the probe finds
  no path, with it True it finds one, and a 0.25 perturbation behind the
  epsilon is still caught. The gate's refusal to pass on `compared == 0`
  is what turned this from a green report into a readable one.
  **And fixing that exposed the next layer, which is why the probe is now
  a search rather than a composition.** With the epsilons matched
  properly the intersection succeeds and `k2.shortest_path` then fails
  with `dest_state > state_idx01 (17 vs. 17)`, because an n-gram
  back-off arc is an input epsilon and those form **cycles**. So all
  three obvious routes fail on Kaldi's own LG: epsilons as ordinary
  symbols rejects every path, epsilons matched gives a cyclic lattice,
  and removing them first does not fit the runner.
  `weight_of_label_sequence` is the fourth: a **Dijkstra over
  (state, position) pairs**, which tolerates the cycles because every
  arc of a graph built from `-log` probabilities costs nothing or more,
  so going round a cycle can only make a path dearer. That property is
  the one thing to check before reusing it on a graph from elsewhere.
  It also takes the epsilon label as an argument rather than assuming 0,
  because label 0 is `<blk>` and not `<eps>` on a k2 topology's input
  side -- a probe hardcoding 0 would skip blanks on one graph and
  epsilons on another while looking identical. And it is what the TLG
  half runs on now, without `remove_epsilon` -- the expensive thing
  `--skip-tlg` existed to avoid.
  One test in that set is worth knowing about because it was written
  after a mutation survived. `weight_of_label_sequence` must not accept a
  **prefix** of the sequence, and the first rejection test could not see
  it: on that graph the super-final arc was only reachable once the whole
  sequence had been read, so deleting the `position == length` check
  entirely still passed. The graph that catches it is final both after
  `A` and after `A B`. A rejection test proves nothing unless the thing
  it forbids was reachable.
  **And with all that, the gate has given its verdict: our LG is
  equivalent to Kaldi's on mini-librispeech.** Run 45567166522 compared
  all 24 lexicon-drawn sequences in 148.9s and agreed on every one,
  within 0.0009765625 per arc. That is the measurement the big corpus
  was built for, and what it establishes is that the **1.36x size
  difference is only a size difference** -- the Python `LG` is the same
  function as Kaldi's, undeterminised and unminimised though it is. So
  minimisation is a cost question and nothing is owed on correctness
  here.
  The gate is **blocking** from that run on, and the second half of
  making it so is easy to miss: the command ends in `| tee`, so without
  `set -eo pipefail` the pipeline's status is tee's and removing
  `continue-on-error` alone would have left a gate that still could not
  fail. Same shape as the `G.fst` gate that skipped unconditionally and
  the nightly summary whose `grep` failed a passing run -- a shell
  detail deciding whether a check means anything.
  Run 45701197144 then repeated that verdict with the step able to fail,
  in 8m57s end to end, and it is the run that says what a graph-only
  gate costs: `stop_stage=4` with `topos=ctc` builds the graphs and
  compares them without training anything, so this is cheap enough to
  dispatch against a change to `BenNevis/decode/graph.py` rather than
  waiting for a nightly. Note the gate step is conditioned on the
  mini-librispeech recipe, and the *scheduled* nightly runs yesno -- so
  the gate only ever runs on a dispatch, and making its TLG half
  expensive cannot slow the scheduled run.
  **A stale comment is a defect, and this file's own lesson applies to
  the workflow.** When the gate became blocking, the paragraph above it
  saying "`continue-on-error` deliberately ... **Turn this into a
  blocking gate once the cost is known**" was left in place, directly
  above the new paragraph saying the opposite -- in fact the whole
  comment block was there twice, older copy first. A reader meets the
  stale one first and has no reason to doubt it.
  `test_the_comments_do_not_contradict_the_step` now pins the two claims
  that have a machine-checkable counterpart: a comment promising
  `continue-on-error` while the step carries none, and a comment asking
  for a change that has been made.
  Worth keeping in view: the gate found **three defects in its own
  machinery** before it found anything about the graphs -- the quoted
  shell, the epsilon flag, and `shortest_path` on cycles -- and reported
  `success` through the first two because of `continue-on-error` and the
  tee. A new gate's first few runs are testing the gate.
  And it
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
**12,711,953** -- not the "about 5%" claimed here before, which came from
dividing `LG.fst.txt`'s 138,155,979 bytes by a guessed line length and
was **33% too high**. `shapes` was added for exactly this reason and
caught exactly this.
**Those arc counts are not comparable with k2's as they stand, and the
ratio printed here was wrong for a second time because of it.** A k2 FSA
represents a final state as an arc labelled **-1** into a single
super-final state; OpenFST's text format writes a final state as a line
of its own and counts no arc for it. So the capture's `arcs` for Kaldi
exclude what k2's `num_arcs` includes, and the two must be put on one
footing before being divided. The gate's own output is the arithmetic:
it reported Kaldi's LG as 4,481,905 arcs against the capture's
4,367,485, and the difference is exactly the capture's 114,420
`finals`; the TLG pair is 12,878,151 against 12,711,953, a difference of
exactly its 166,198. Like for like, then, we are **1.36x on LG and 3.36x
on TLG**, not the 1.40x and 3.40x this entry gave. The error is small
and the lesson is not: two numbers for the same graph differing by 2.6%
is a counting convention, not noise, and the way it was caught was by
having both counts in front of each other rather than by either looking
implausible.
So minimisation buys about **27% on LG** and more on TLG. It remains a
*cost* question rather than a correctness one, since
`intersect_dense_pruned` needs the graph neither deterministic nor
minimal -- but it is a real cost on the graph the decoder loads, and the
1.36x against 3.36x split puts most of it in the **topology
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
run against a graph 3.36x the size of the one Kaldi's decoder loads.
So minimisation stops being a tidiness question: **mini-librispeech
cannot be decoded with k2 inside a CI run as things stand**, and that
blocks the comparison stage 10 exists to make. Two things are worth
separating before anything is optimised, because they are not the same
fix: the graph is larger than it needs to be (minimisation, epsilon-pair
merging), and the search is run one utterance at a time where Kaldi's
runs eight jobs. The second is the cheaper to try, and it is already
half done: `decode` takes a batch `(N, T, C)` and
`intersect_dense_pruned` is built for one, but `decode_archive` called
it with `scores.unsqueeze(0)` per utterance, so the batch dimension was
always 1. (It is `decode_archive`, not `decode_many`, which this entry
named for a while.) Neither is measured against the other yet, and the
order to measure them in is batching first, since it needs no new graph
operation.
**The batching is done now, and what it cannot yet claim is the
speedup.** `decode_archive` takes a `batch_size`, pads each batch to its
longest utterance and gives k2 one call per batch; `run/decode_k2.py`
defaults it to **8**, which is both what makes stage 10 plausible and
what `decode_faster.sh` runs at `nj=8`, so the two decoders stay
comparable. The padding is **not searched** -- each utterance carries
its own frame count into k2's segments -- so mixing lengths in a batch
costs tensor memory and no search time. What it costs is the isolation
that was the original reason for one at a time: an utterance exhausting
memory now takes its batch with it, though the shape and token-count
checks still name the offending utterance, because they run before
anything is batched.
The tests pin what a batch can get *wrong*, not what it saves: that the
answers do not depend on how the utterances were grouped, that a short
utterance's padding is not decoded as audio -- mutation-verified by
passing the padded length, which breaks exactly those two tests -- and
that the search is called once per batch rather than once per utterance,
counted rather than timed. **None of that is the measurement, and the
measurement needs a trained model**: stage 10 runs only after stage 5
onward, so the 17.1s an utterance can be compared with a batched figure
only by a full mini-librispeech run of three to five hours. Until that
run exists, treat this as a change of the right shape with an unmeasured
effect -- and note that CI decodes on the **CPU**, where k2's gain from
a batch is a question rather than a given.
**That question is answered now, and the answer is 2.65x.** Run
45715720923 decoded with `batch_size=8` against the same 43,263,880-arc
`TLG` on the same runner class and the same CPU: **600 utterances in
1h04m35s, which is 6.46 seconds each**, against the unbatched 17.1. So a
full 1,089 would take 1h57m where it had taken an extrapolated five
hours, and batching is the cheaper of the two fixes by a wide margin.
Three things it does not settle. **The rate is not flat** — the six
progress lines give 5.12, 4.93, 5.98, 5.81, 9.67 and 7.13 seconds an
utterance, so the last third cost nearly twice the first, and whether
that is utterance length or something accumulating is unmeasured.
**Kaldi is still 19x faster**: its stage 7 did the whole 1,089 in
**6m10s** at `nj=8`, against a graph 3.36x smaller, so minimisation and
epsilon-pair merging remain on the table rather than discharged by this.
And **the comparison stage 10 exists for is still not made** — that run
was cancelled at the ceiling having reached 600 of 1,089, so there is a
Kaldi WER and no k2 WER, which is the second time in a row that the
measurement was lost to training rather than to decoding.
**It was lost a third time, and that is what settled the diagnosis.** Run
45918000128 ran four epochs instead of eight and was **SIGKILLed in stage
10 with 1,000 of 1,089 utterances decoded**, 321 minutes into a 350-minute
budget — so it had the time and not the memory. The decoder's resident size
over the whole stage does **not** trend upward: 354 samples, min 4,236MiB,
median 7,468, p90 10,347, max **14,988**, oscillating between about 5 and
10GiB from the first minute. So that was a per-batch peak rather than a
leak, it came within **231MiB** of death at 05:04 and survived, and at
06:03 it did not. Kaldi scored the same four-epoch checkpoint at **68.1%**
(32.2 correct, 25.1 sub, **42.6 del**, 0.4 ins), against 49.7% at eight
epochs; neither is a ledger entry.

**The 19x was never k2's CPU implementation. It was the graph, and the
answer is a decoder rather than a smaller graph.** Three measurements in
sequence, each of which killed a plausible fix:

- **The two pruning knobs do nothing.** `output_beam` 8 to 2 is worth
  **-0.7%** and `max_active` 10000 to 5000 is worth **9%**, on the real
  43,263,880-arc `TLG` at 1,100 frames. There is no setting of them that
  closes a factor of nineteen, and `output_beam` was the one with a
  reason to work, being k2's lattice beam where `decode-faster` keeps no
  lattice at all.
- **Keeping the language model's back-off arcs as epsilons does nothing
  either.** That was the hypothesis with the best story — `decode-faster`
  has `ProcessNonemitting` and ours cannot, so `compose_lg` must call
  `remove_epsilon`, which expands a back-off arc into one arc per
  reachable word. Built and measured: **43,611,556 arcs against
  43,263,880**, with 352 states still holding 51.5% of them. Costs 12%,
  not 8x. The entry below already said as much and it was still worth
  building to be sure.
- **The out-degree the search actually meets is 168.** Not the graph's own
  mean of 9.31, and not Kaldi's 3.66. Profiling one utterance: a mean of
  2,012 active states a frame and **338,909 candidate arcs**, peaking at
  4.2 million. State 0 carries **208,941 arcs over 106 distinct input
  labels** — 1,950 parallel arcs per label — and **366 states like it hold
  52.5% of the whole graph**. They are the word-boundary states, and the
  cause is `make_lexicon_fst.py`'s flower: a chain per pronunciation, so
  the loop state has one arc per entry. `L_disambig` has a max out-degree
  of 206,525 and `G` of 200,002 before anything is composed. Only
  determinisation fixes that, and `k2.determinize` costs 56 minutes and
  returns 3.4x the arcs. Distinct `(state, label)` pairs number
  **13,846,519 of 43,263,880**, so a determinised graph would be about
  32% of this one — which is Kaldi's 12.88M, confirming the diagnosis from
  the other side.

So `BenNevis/decode/viterbi.py` is Kaldi's `FasterDecoder` written in torch
tensor operations, and **label factoring** gets most of what
determinisation would without a subset construction: where a state has
*k* > 1 arcs sharing an input label, they become one emitting arc into a
new state carrying the group's minimum, plus *k* non-emitting arcs
carrying the residuals. Exact, because `min + (w - min) == w`, and the
surviving token set is identical — a token at the new state costs the
group's minimum, so it survives exactly when a member would have, and the
non-emitting pass then prices each member under the same cutoff. What it
buys is that those *k* arcs are reached only once the label has survived
the beam **with its acoustic score applied**, and at a word boundary a
handful of 106 first phones are acoustically plausible in a given frame.

| search | ms a frame | WER |
| --- | --- | --- |
| `intersect_dense_pruned`, search 20 / output 8 | 8.60 | 41.0% |
| the same, `output_beam` 2 | 8.66 | 41.0% |
| the same, `max_active` 5000 | 7.86 | 39.5% |
| token passing, **no** factoring, beam 16 | 4.90 | 45.0% |
| token passing, factored, beam 20 / max_active 7000 | **0.98** | 41.0% |
| token passing, factored, beam 32 / max_active 5000 | 3.38 | 38.0% |

Candidate arcs a frame: **338,909 before, 12,737 emitting plus 10,742
non-emitting after** — against Kaldi's own ~18,300. So at equal WER this
is **6.4x** k2's search, and at Kaldi's own beam of 32 it is 2.4x faster
*and* more accurate. Memory is the other half: the `SearchGraph` holds
**934MiB** against the source FSA's 1,101, and the only thing allocated
per utterance is the back-pointer table, 91MiB at 1,500 frames and
`max_active` 7000 — where `intersect_dense_pruned` wanted 14,988.

Four things are worth keeping in view.
**The factoring threshold is a memory knob and nothing else.** Thresholds
2, 4, 8 and 16 gave the same WER and the same substitution, deletion and
insertion counts to the error on the real graph; 8 is the default because
at 8 only 100,141 new states cover 55.2% of the arcs and the arc count
grows 0.2%, where at 2 it takes 2,848,972 states to cover 74.9% and costs
6.6%.
**The correctness standard here is an oracle, not an agreement.**
`k2.intersect_dense` is exhaustive and `k2.shortest_path` exact, so on a
graph small enough to intersect the best path is known: 293 random graphs
at three thresholds each agree on **both the cost and the word sequence**.
That is a stronger standard than `test_decoder.py` holds
`intersect_dense_pruned` to, which is only fair, since that module is
being compared with itself.
**`--nj` matters and threads do not.** Four torch threads against one was
measured at **1.44x** on tensors of ten thousand elements, where four
processes are 3.3x — so `decode_shards` forks, which also means the
children share the 934MiB graph by copy-on-write. `share_memory_()` would
work and is deliberately not used: it needs `/dev/shm`, 64MB inside
Docker by default, and the nightly runs the recipe inside Docker. Every
`nj` from 1 to 8 gave byte-identical hypotheses, which is what makes the
flag safe.
**The remaining gap is still the graph, and the next cheap step is
measured.** At beam 32 the non-emitting pass **evaluates 58,234 arcs a
frame and keeps 15,358** — 26.4%. Each group's residuals are
non-negative and already sorted, so a `searchsorted` against
`cutoff - cost` would expand only the prefix within the beam and cut that
3.8x. After that the honest remainder is determinisation: Kaldi evaluates
~18,300 arcs a frame where this evaluates 56,416 at the same beam, which
is the 3x the flower still costs.

**And the comparison stage 10 exists for is finally made.** Run
46049745072, four epochs of `ctc` on master, all stages, success:

| | Kaldi `decode-faster` | Python + token passing |
| --- | --- | --- |
| WER | 67.6% | **65.96%** |
| sub / del / ins | 25.2 / 42.2 / 0.3 | 26.1 / **39.7** / 0.15 |
| utterances | 1,089 | **1,089** |
| wall time at nj=8 | **8m46s** | 29m21s |
| peak decoder RSS | — | **1,818 MiB** |

So on the same checkpoint the Python lane is **1.6 points better**, and
the shape of the difference is the interesting part: it **deletes less**
— 39.7 against 42.2 — and substitutes slightly more. Two plausible
causes, neither measured: our graph keeps full precision where
`fstminimizeencoded` quantises every arc to 1/1024, and the two searches
prune differently. It is **3.35x slower** rather than the 29x the k2
route was, and it peaks at **1,818 MiB against the 14,988 that SIGKILLed
k2** — which is why this run finished at all.
Four things to hold onto. The run was **slow throughout**, 35.5 minutes
an epoch, so the 8m46s is this runner's `decode-faster` and not the
6m10s recorded earlier on a faster one; the ratio is the comparable
figure, not either absolute. **Neither WER is a ledger entry** — four
epochs is a deliberately undertrained model, 42% deletions, and
`tests/golden/wer/` still has no mini-librispeech file. The 67.6% here
against 68.1% on run 45918000128 is the same configuration twice, which
is a useful consistency check on the Kaldi side. And the remaining 3.35x
is still the flower: Kaldi evaluates some 18,300 arcs a frame where this
evaluates more, which only determinisation closes.

**All of that is done, and one of the three was a defect rather than an
optimisation.** Profiling the shipped decoder rather than theorising about
it found, in order of what it was worth:

- **`max_active` was being applied to the wrong set.** `GetCutoff` runs on
  `toks_` at the *start* of a frame, so Kaldi's cap bounds the token list
  **after** `ProcessNonemitting` has added to it. This capped the emitting
  candidates instead, and with factoring that is not a detail: the
  emitting pass left 3,837 tokens, the closure took that to **18,038**,
  and nothing capped them — so `--max-active 5000` was admitting 3.6
  times its own limit. Fixing it took the emitting arcs from 56,984 a
  frame to **13,382**. It also means `max_active` now means what Kaldi's
  does, which matters more than the speed.
- **`GetCutoff` returns two numbers and only one was being used.** The
  other is `adaptive_beam`, which `ProcessEmitting` prunes the next
  frame's candidates with, at `best_new + adaptive_beam`: tighter by
  `beam_delta` when `max_active` bound, looser when `min_active` did.
  Splitting them makes `min_active` inert, which is how this was found.
- **The closure expanded every member of every group it reached.** The
  residuals are non-negative and now stored *sorted*, so one
  `searchsorted` against a state-major quantised key gives the prefix that
  can be within the cutoff: 162,533 arcs a frame became **18,912**. The
  key is quantised coarsely and rounded outwards so the prefix is a
  superset, and the exact float comparison still decides.
- **The second non-emitting round could never add anything.** The states
  factoring introduces stay active, so re-expanding the whole token list
  repeated work guaranteed to lose recombination — 1,670 expansions for
  1,092 frames, of which 578 found nothing. A frontier of what improved
  fixes it, which is what `ProcessNonemitting`'s `queue_` is.
- Two of three `repeat_interleave`s in the ragged expansion are a gather,
  since `base - offset` is one value per token. Worth 10%.

| beam | max_active | ms a frame | s an utterance | WER |
| --- | --- | --- | --- | --- |
| 32 | 5000 | 1.70 | 1.02 | 38.0% |
| **24** | 5000 | **1.00** | 0.60 | **38.0%** |
| 20 | 7000 | 0.69 | 0.41 | 41.0% |
| 16 | 7000 | 0.47 | 0.28 | 46.5% |

So **beam 32 went from 3.38 to 1.70 ms a frame with the WER unchanged**,
and `--nj 8` does the full 1,089 utterances in **six minutes** against
Kaldi's 6m10s on a CI runner — which is the first time the two are the
same order of magnitude. Against the original k2 search it is 8.60 ms a
frame to 1.70 at a better WER.
**Beam 24 looks free and the default has not been changed to it.** It
gives the same WER as 32 at 1.7x the speed and lands at 14,012 candidate
arcs a frame, *below* Kaldi's ~18,300 — but that is measured on
calibrated synthetic acoustics, and tuning a default on those is the
thing this file warns about everywhere else. The real run decides it.
**And a mutation survived the suite here, which is worth recording.**
Rounding the prefix's quantised budget inwards loses arcs on the cutoff,
and every test plus the 293-graph oracle passed with it, because they all
used a beam wide enough that a one-step error was invisible. The test
that catches it sets the beam to *exactly* a member's residual. Same
lesson as the prefix rejection test in the graph probe: a boundary is only
tested at the boundary.
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

**There are two ledgers now, and they are discovered rather than listed.**
`tests/recipes/test_ledger.py` had yesno's path as a module constant, which
is the shape that left mini-librispeech's captured reference unread until
`tests/reference/test_corpora.py` was written — so committing a second
ledger would have added a file no test looked at. It globs
`tests/golden/wer/*.json`, asserts it found some, and runs the checks that
are true of any ledger over all of them; what is true of one corpus alone —
yesno's eight-words-per-utterance arithmetic, mini-librispeech's absent
reference — lives in its own class.

**`tests/golden/wer/mini_librispeech.json` is the first ledger entry this
project has had from a run whose point was the model.** Run 46100161743,
eight epochs of `ctc`, all stages, success in 5h34m of a 5h50m ceiling:

| | WER | sub | del | ins |
| --- | --- | --- | --- | --- |
| BenNevis's search | **46.56%** | 26.6 | 19.1 | 0.8 |
| Kaldi `decode-faster` | 48.8% | 25.7 | 22.2 | 0.9 |

on all 1,089 utterances of `dev_clean_2`, 20,138 reference words, from the
same checkpoint — so **2.24 points** apart on the search alone, and the
difference is mostly deletions, 19.1 against 22.2. That is the third time
the Python search has come out ahead of `decode-faster` by deleting less,
on three different checkpoints, which makes it a property of the two
searches rather than of one model. Wall time was 20m38s at `--nj 8` after
a 55s graph build, against `decode-faster`'s 7m07s: **2.9x**, and peak
memory across all eight shards was 5,321MiB on a 16GiB runner.

Four things about that entry are decisions rather than records.
**The gate is `measured_wer` plus two points, which is wider than the
evidence justifies.** One run gives no estimate of run-to-run spread: the
decode is deterministic given a checkpoint — every `nj` from 1 to 8 gives
byte-identical hypotheses — but the training that produces it is not
bit-reproducible across runners, and this model is undertrained and on a
steep part of its curve. Two points is 403 word errors of 20,138, which
catches a broken feature pipeline or a decoder that deletes everything and
nothing subtler. The second recorded run is what should tighten it, and
`test_ledger.py` caps any gate's allowance at three points so the
widening cannot continue quietly.
**`measured_wer` is the Python search, not Kaldi's.** The ledger is a
regression gate on the system's output, and gating on `decode-faster`
would make it depend on a dependency being deleted. Kaldi's number is
recorded beside it as a cross-check on the whole graph and search chain.
**Only `ctc` is in it**, because at 36 minutes an epoch the other seven
topologies have never been trained on this corpus. A topology absent from
`topologies` is unmeasured, not passing, and the test that the ledger
covers exactly what `run.sh` trains is what keeps those two facts the
same fact.
**The validation loss was still falling** — 4.5318, 2.8998, 2.3510,
2.1049, 1.9717, 1.8920, 1.8555, **1.8279**, improving by 0.080, 0.036 and
0.028 over the last three — so this is a flattening curve rather than a
bottomed one, and the ledger records the whole curve for the next run to
be read against. The thing to change is the model, not the epoch count:
`conf/model/rnnp.yaml` is 3 × 320 where librispeech's is 3 × 1024.
The per-epoch loss is also 0.018 *worse* than the 83-dimensional run's
1.8100 at the same epoch, while the WER came out 0.9 points better — a
reminder that the two do not move together and that the ledger gates the
one that matters.
**That easy win is taken.** The eight shards had finished across a
**five-minute spread**, 08:16:59 to 08:22:03, because `decode_k2.py`
block-split an `output.scp` that is sorted by duration, so the last shard
got every one of the longest utterances and seven shards sat idle at the
end. `split_entries` interleaves now -- `entries[start::count]` -- which
makes each shard a near-uniform sample of the length distribution.
**The reason it had not been done is a comment that was wrong.** The old
`shards` docstring said contiguous blocks were needed "because the blocks
are concatenated in order afterwards and the recipe merges the result
against `ref.wrd.trn` by position". They are not merged by position:
`read_trn` returns a mapping from utterance id and `score` looks every
reference utterance up by id, so the order affects nothing that is
scored. A justification nobody checked had made a free change look
unsafe -- the same shape as the stale `continue-on-error` comment, and
worth the same suspicion.
What the order *does* decide is the bytes of `hyp.wrd.trn`, and the claim
that "every `nj` from 1 to 8 gave byte-identical hypotheses" was true only
by the accident of splitting contiguously. `in_scp_order` restores the
scp's order explicitly before writing, so that property now holds **by
construction**, and the tests assert it across every `nj` rather than at
one. An utterance the decoder returned that is not in the scp is appended
rather than dropped, because `score` fails loudly on a mismatch and an
unexpected extra has to survive to be seen.
The three helpers were lifted out of `main` to be testable at all: nested
in a function that needs k2, a graph and a trained model, the split could
only ever have been exercised by a five-hour run. The balance test is
paired with a **counter-example** that fails for a contiguous split, so
it is known to discriminate rather than merely to pass.

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

**SpecAugment exists now, and it is off by default.**
`BenNevis.core.augment.SpecAugment` masks bands of frequency and spans of
time in a padded batch, and `Trainer` applies it in `_compute_loss` --
which is reached from both `_train_batch` and `_valid_batch`, so the
module being a no-op outside `training` is what keeps the validation loss
comparable across epochs. A validation loss computed on masked features
would wander with the mask draw rather than with the model, and both the
mid-epoch save schedule and the ledger read that loss.
Four decisions in it are not defaults, and each is a failure that would
produce a slightly worse WER and no other symptom:
**time masks are placed inside each utterance's own length**, because a
batch is padded to its longest member and a mask in the padding does
nothing -- the duration samplers group *similar* lengths, not identical
ones, so masking over the padded width weakens the augmentation in
proportion to how ragged each batch is; **the width is capped at a
fraction of the utterance**, since a 100-frame mask removes a sixth of a
600-frame yesno utterance and all of a 60-frame one, and the cap is what
lets one setting serve corpora whose lengths differ by an order of
magnitude; **the mask value is zero because CMVN has already made each
dimension zero-mean**, so zero is the mean and masking to it is masking
to no information -- which also means this only belongs *after*
normalisation; and it **does not write in place**, because the batch it
is handed is the one the DataLoader still owns.
Time warping, the paper's third transform, is deliberately absent: it
needs an interpolation per utterance, it is the smallest of the three
contributions, and espnet leaves it off too.
**None of it is measured on this project yet.** It is off in every
recipe, with the settings written out and commented in
`conf/trainer/default.yaml`, and the WER ledger is how it gets judged --
a 66% WER on five hours of audio is what it exists to move. The tests
pin the four failure modes above rather than the WER, because a WER
cannot be asserted in tier 1b and the failure modes can.

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
- **The mini-librispeech trainer does not leak. glibc was holding the
  heap, and the whole 8GiB came back from one `malloc_trim(0)`.** Three
  `ctc` runs were SIGKILLed on a 15.6GiB runner -- epoch 4, epoch 4 and
  epoch 7, so the epoch was never the pattern, the race between growth
  and the epoch count was -- and the trainer was seen climbing to 7.4GiB
  in epoch 1, plateauing four epochs, then rising to 9.3GiB and dying
  with 9MiB of the host left. Two readings settled it, in this order,
  and the order is the lesson: measure the composition before theorising
  about the cause.
  **First, no Python type is growing.** Run 45511013199, two epochs:
  resident 808MiB, 7894MiB, 9130MiB, against live tensor storage of
  130MiB, 169MiB, 169MiB and a tracked object count of 960,770, 961,347,
  961,127 -- flat within four hundred across 8.3GiB. The named growers
  are three-figure instance counts. So `gc.get_objects()` had answered
  the question it was built for in the negative, and `tracemalloc` would
  have been the same instrument again: both see only what Python
  allocated.
  **Second, the memory was free all along.** Run 45554764613 ended epoch
  1 at 6030MiB and epoch 2 at 6645MiB, and `trim_malloc` took both back
  to the *same* **1058MiB**, releasing 4972MiB and 5588MiB. Identical
  post-trim sizes two epochs apart is the whole proof: there is nothing
  accumulating, and each epoch simply leaves about 5.5GiB of free heap
  that glibc will not return to the kernel on its own -- which it never
  does for a block in a secondary arena, and a workload that allocates
  and frees large arrays across threads, as k2's intersection and torch's
  CPU kernels both do, is what fills those arenas.
  So `fit` now trims once an epoch, which **bounds** the resident size at
  one epoch's allocation instead of letting it accumulate towards the
  runner's limit. Two properties of that are pinned in
  `tests/core/test_training_logs.py` and each is mutation-verified
  separately: the trim happens once an epoch, and it happens on **every
  rank**, because only rank 0 logs and a four-rank run where three never
  gave the heap back would be the same failure with three quarters of the
  evidence missing. The baseline reading before training deliberately
  does not trim -- there is nothing to give back yet, and trimming there
  would make the first epoch's growth unreadable.
  **Eight full epochs have now exercised it, and the bound holds.** Run
  45715720923 read 5054, 8161, 7649, 7596, 7449, 7282, 7113 and 7315MiB
  resident and **1077, 1076, 1074, 1073, 1072, 1073, 1072 and 1072MiB
  after the trim** -- the same figure to within 5MiB across all eight,
  releasing up to 7085MiB at once, with tracked objects flat at 961,478
  from epoch 2 on. The three runs before this fix were SIGKILLed in
  epochs 4, 4 and 7; this one finished training and was cancelled for
  time in stage 10 instead, which is the failure mode it was meant to
  trade down to.
  Three things not to mislearn from this.
  **The loss path was measured clean and that was correct**: 2,000 steps
  of `GraphLoss` with the real `compile_training_graph`, fixed and varying
  batch shapes, flat at about 400MiB; `k2.Fsa.to()` returns `self` on the
  same device; `k2.arc_sort` on a sorted topology is a no-op; `kaldiio`'s
  `LazyLoader` caches no arrays. None of it explained the growth because
  none of it was the growth.
  **The absolute numbers are runner-dependent and the trim's are not.**
  The 8.3GiB run was an Intel Xeon 6973P-C and the 5.5GiB one an AMD EPYC
  7763, both 4 cores and 15.6GiB; the post-trim size was the same twice
  on the one runner, which is the figure that carries the argument.
  **`trim_malloc` returns None off Linux**, because macOS, Windows and
  musl have no `malloc_trim`. A probe answering False there would read as
  "the memory is in use" on a platform where the question was never put.
  If this ever needs to go further, `MALLOC_ARENA_MAX` attacks the cause
  rather than the symptom, but it has to be set before the allocator
  initialises, so it belongs in the launcher and not in `fit`.
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
- **Pitch is gone, and the features are 80 dimensions.**
  `steps/make_fbank_pitch.sh` appended the three dimensions
  `process-kaldi-pitch-feats` produces, making every recipe's input 83
  wide. It was dropped deliberately: pitch is a GMM-era addition for
  **tonal** languages, where F0 carries lexical information, and these
  corpora are English and Hebrew; every modern end-to-end recipe uses
  plain filterbank or the waveform, and this project's own `wav2vec2` and
  `whisper` encoders never touch fbank at all.
  **The practical reason it had to go is that nothing can replace it.**
  `torchaudio.functional.compute_kaldi_pitch` was **removed in torchaudio
  2.1** -- checked against the installed 2.11.1, where it is absent and
  `detect_pitch_frequency` is a different algorithm producing a different
  thing. So a Python feature extractor would have had to reimplement
  `feat/pitch-functions.cc`, NCCF over lag candidates plus a Viterbi,
  with no prospect of bit-exactness. Dropping it is what makes a Python
  fbank the last feature problem rather than the first of two.
  **What this costs is the WER ledger's comparability**, and it is
  recorded rather than absorbed. Every number in
  `tests/golden/wer/yesno.json` and in `egs/yesno/asr/README.md` was
  measured at 83 dimensions; both now say so, and a `features` block
  names which era each belongs to. **`max_wer` was deliberately not
  widened.** If an 80-dimensional run exceeds the gate, that is a real
  finding about pitch and the thing to do is find out why -- widening it
  is how a ledger stops meaning anything.
  **Both captures are re-taken and the cost of dropping pitch is three
  word errors.** Runs 46075407888 (yesno, all stages) and 46092683476
  (mini-librispeech, stages 0 to 4) produced 80-dimensional references.
  Against the 83-dimensional ledger, six of the eight topologies are
  unchanged, `3state-skip-2` is two errors worse and `mmictc` one --
  three in 1,856 reference words, and **every topology stayed inside the
  gate that was deliberately not widened**. That is the measurement the
  decision rested on, and it came out the way the tonal-language argument
  predicted.
  The same run also settles what was still owed on the decoder: all eight
  topologies agree **error for error and error type for error type**
  between `decode-faster` and the Python search, this time on master's
  corrected `max_active` rather than #93's.
  **A fresh capture fired a latent test defect, which is the second
  substring-matching bug of the day.**
  `TestFunctionalEquivalenceOfG` selected its fingerprints with
  `"G.fst" in name` -- and `graph/<topo>/LG.fst.txt` contains `G.fst`, so
  a test for the word-level *grammar* was comparing the Python `G` build
  against every topology's *decoding graph* fingerprint. It passed only
  because yesno is built with `--sil_prob 0.0`, which makes every lexicon
  weight zero, so LG's weights equal G's up to `fstminimizeencoded`'s
  1/1024 rounding -- 6.907755 against a recorded 6.908203, which is
  7074/1024 exactly. The old capture had no LG fingerprints, so nothing
  fired until a new one recorded them. It matches the basename now.
  Together with `--compile-fsts` reading as a binary called `fsts`, that
  is twice in one day: **a filter over names wants the whole name, not a
  substring of it.**
  **The waveforms are not in these captures**, because the runs were
  dispatched from a branch that did not yet have the change recording
  them. So `tests/utils/test_features.py`'s filterbank gate still skips,
  and activates on the next capture rather than this one.
  **The next capture did not provide them either, and the reason is a
  bug that cost a whole run.** `wav.scp` holds paths relative to the
  *recipe* directory — mini-librispeech's entries are
  `flac -c -d -s ./corpus/LibriSpeech/... |` — because every stage of a
  Kaldi recipe runs from there, while `capture.py` is invoked from the
  repository root. So run 46100161743's capture step died on
  `FLAC__STREAM_DECODER_INIT_STATUS_ERROR_OPENING_FILE` after five and a
  half hours of recipe, having been the only new code in the step.
  `capture_waveforms` now takes the recipe directory and reads inside
  it, which is reproducing the recipe's own assumption rather than
  guessing at a path, and one unreadable utterance is logged and skipped
  rather than ending the capture — a gap in a reference is not worth
  losing the rest of a run that has already cost hours.
  **What let it ship is the test fixture, not the review.**
  `tests/reference/test_capture.py`'s synthetic recipe has no `wav.scp`
  at all, so the waveform branch skipped in every one of its ninety-nine
  tests and the suite was green over a function that could not run — the
  same shape as the `G.fst` gate that could not fail and the `idim`
  reader that stopped matching. The fixture writes one now, relative and
  in both the plain and the piped form, and four of the six new tests
  fail if the working directory is taken away. **A capture step is the
  last thing in a five-hour job, so its own tests have to be the kind
  that run in a second.**
  **And it invalidates the captured reference, which is the gate working
  rather than a problem.** The capture records `conf/pitch.conf`, and
  `test_against_reference.py` fails with "recipe/conf/pitch.conf no
  longer exists in the recipe" -- which is exactly what that check is for:
  a reference taken under different settings is a record of something the
  pipeline no longer does. The fix is its own error message's: re-run the
  recipe, re-capture, commit the result. **Not** an edit to the test or
  to the manifest.
- **Three recipes configured an input width their own features did not
  have, and nothing caught it.** `egs/librispeech/asr{1,2,3}` call
  `steps/make_fbank.sh`, which writes 80 filterbank coefficients, while
  their model configuration said `idim: 83` -- a shape mismatch at the
  first forward pass. The only reason nobody met it is that those recipes
  have never run in CI. Note `conf` in asr2 and asr3 is a **symlink** to
  asr1's, so the three share one file and shared one mistake; a loop that
  patches all three double-patches.
  **A model's `idim` is derived from the features, not written out.**
  `odim` was always `null` and filled in from the language directory, so
  a topology change could not desynchronise it, while `idim` was a number
  per recipe. `run/train.py` now measures it with
  `BenNevis.utils.data.feature_dim`, which reads one utterance of
  `feats.cmvn.scp` -- lazily, so a five-hour corpus costs a seek -- and
  every recipe says `idim: null`. An explicit number is still honoured
  and still **checked** against the data, raising rather than failing in
  the first forward pass, so pinning a dimension stays possible and
  cannot be silently wrong.
  Two things about the archive it measures. It is `feats.cmvn.scp`, the
  one `Dataset` reads, not `feats.scp`: those happen to be the same width
  today and would stop being so the moment normalisation changed it. And
  it is skipped entirely when `load_feats` is false, because
  `Wav2Vec2Model` takes the waveform and asking for a feature archive
  would fail a perfectly good run.
  **The test for this skipped everywhere for one commit and passed.** The
  reader matched `idim:\s*(\d+)`, which stopped matching the moment the
  configs became `idim: null`, so all seven parametrised cases skipped
  and the suite was green while checking nothing -- the `G.fst` gate's
  failure in #41, reproduced exactly. There is now a test that the reader
  read something, which is the only shape of assertion that catches it.
- **`prepare_lang.sh`'s tables are Python now, and all nine recorded ones
  are byte-exact first time.** `python3 -m BenNevis.lm prepare-lang` builds
  `phones.txt`, `words.txt`, `oov.{txt,int}`,
  `phones/{silence,disambig,optional_silence,wdisambig_phones,
  wdisambig_words}.int` and `phones/align_lexicon.{txt,int}` from a Kaldi
  dictionary, plus the two lexicon FSTs through the existing builders.
  Checked against mini-librispeech's capture: **nine files identical,
  364 phones and 200,007 words**, and both FSTs still matching their
  recorded digests -- which matters because they are now built against
  *these* tables, so a wrong table would show up as a wrong FST.
  Four of `prepare_lang.sh`'s decisions are not guessable from the shell
  and three of them decide the integers.
  **A silence phone keeps its bare form and a non-silence phone does
  not.** The suffix lists are `"" "" "_B" "_E" "_I" "_S"` for silence and
  `"" "_B" "_E" "_I" "_S"` for non-silence, and the *first field is the
  map's key* -- so silence expands to five versions including the
  unsuffixed one and non-silence to four. That is why
  mini-librispeech's `phones.txt` reads `SIL SIL_B SIL_E SIL_I SIL_S` and
  has no bare `AA`.
  **The order is silence, then non-silence, then disambiguation**, each
  flattened by input line and then by field -- and
  `nonsilence_phones.txt` groups a phone with its stress variants on one
  line, `AA AA0 AA1 AA2`, so the grouping is part of the answer.
  **`words.txt` reserves `#0`, `<s>` and `</s>` after the last word**, and
  `prepare_lang.sh` *exits* if `<s>` or `</s>` is in the lexicon rather
  than renumbering.
  And `align_lexicon.txt` is built from the **position-suffixed** lexicon,
  each line `word word phones...` with the word repeated, plus an
  `<eps> <eps> <silphone>` entry.
  **What is not produced is everything only GMM training reads** --
  `phones/sets.txt`, `roots.txt`, `extra_questions.txt`, `word_boundary.*`
  and `topo`. BenNevis trains no GMM, and writing files nothing reads
  would be four more things to keep correct.
  **Two gaps in the capture, not in the code, and they meet badly.**
  yesno's capture predates the dictionary being recorded, so there is no
  input to rebuild its tables from; mini-librispeech's has the dictionary
  but **not `phones/align_lexicon.int`**. So the one file the *training*
  path reads is ungated on the one corpus that could gate it. Its
  structure is checked instead -- the word repeated, every integer
  resolving through the two byte-exact tables -- and it belongs on the
  list with `utt2spk` for the next capture.
  One test-cost note worth keeping: the first version of the gate ran
  `prepare_lang` in a function-scoped fixture, so mini-librispeech's
  206,510 entries were rebuilt four times and **tier 1 went from ten
  seconds to twenty-six**. It is memoised per corpus now, and skips the
  FSTs because `tests/reference/test_corpora.py` already gates both by
  digest. Memoised rather than a class-scoped fixture, which cannot
  depend on a function-scoped parameter.

- **yesno trains with no Kaldi at all, and that was verified by doing
  it.** Stages 1, 2, 3 and 5 ran end to end against a synthetic
  sixty-wave corpus with no `tools/kaldi` present: two epochs of `ctc`,
  train loss 5.19 to 3.00 and validation 3.64 to 2.26, checkpoints
  written, exit 0. That is the goal-2 milestone for the training half,
  and **running it is what found the three things still in the way** --
  none of which any test had caught, because every test either read a
  script as text or ran one stage in isolation.
  **`utt2dur` was missing, and it is a regression this migration
  introduced.** `BenNevis.core.Dataset` *requires* it -- it filters on
  `min_duration` and `max_duration`, sorts by it, and the duration
  samplers batch on it -- and `make_fbank.sh` wrote it by default,
  through `compute-fbank-feats --write-utt2dur`. `run/make_fbank.py`
  wrote only `utt2num_frames`, so from the stage-2 conversion until this
  fix **the recipes could not start training**, and the suite was green
  throughout: `tests/recipes/test_stage2.py` checked the archive and the
  frame counts, and nothing constructed a `Dataset` from what the stage
  produced. It is written unconditionally now, with no flag, because a
  flag to turn it off is a flag to stop training. Note it is the
  **waveform's** duration: `utils/data/get_utt2dur.sh` has a different
  rule for when there is no audio to measure, `(frames + 1.5) *
  frame_shift`, and the two disagree -- 3.995 against 4.000 on the
  fixture -- so a test asserts the value is *not* the derived one.
  `BenNevis.utils.datadir` had also listed `utt2dur` as a file it
  refuses, which would have made `fix_data_dir.py` reject every real
  data directory the moment the file existed; it is filtered like the
  rest now, and its `$2 > 0` test parses a **float**, because `1.5` is
  not a digit string and `isdigit` would have dropped every utterance.
  **`path.sh` exited when Kaldi was absent, and `exit` in a sourced file
  exits the caller.** So `. ./path.sh` at the top of every `run.sh` made
  the *whole recipe* unrunnable without a Kaldi checkout, including the
  stages that no longer need one. It warns now and sets everything that
  does not need Kaldi; the Kaldi-present behaviour is unchanged, PATH
  order included. The pattern is the one `prepare_graph.sh` and
  `prepare_topo.py` already follow: **the stage that cannot proceed
  complains, not the environment refusing to be set up.**
  **And `run.sh` sourced Kaldi's `utils/parse_options.sh` to read its own
  flags** -- `utils/` being a symlink into the checkout -- so a recipe
  could not parse `--stage` without Kaldi, which is *exactly* the defect
  `prepare_topo.sh` had and was fixed for. `BenNevis/run/parse_options.sh`
  is a replacement with Kaldi's semantics, because seven recipes and four
  `run/*.sh` are written against them: `--name value` with dashes
  becoming underscores, **an unknown option fatal** rather than ignored,
  `--name=value` refused, and a variable currently `true` or `false`
  settable only to `true` or `false`. All nine call sites are converted.
  The unknown-option rule is the one that earns the file: `--stop_stge 4`
  must not quietly run every stage, and a parser that ignored it would
  pass every other test.
  **yesno's data preparation is Python, and the Perl is kept as the
  oracle.** `local/prepare_data.py` replaces `prepare_data.sh` and the
  three Perl scripts beside it, plus `utils/utt2spk_to_spk2utt.pl` --
  which was the last thing stage 1 took from the checkout.
  `tests/recipes/test_yesno_prepare_data.py` drives
  `create_yesno_wav_scp.pl` and `create_yesno_txt.pl` and requires the
  Python to agree **byte for byte** on `wav.scp` and `text` for both
  halves, which is a stronger standard than restating the 31/29 split in
  a constant and asserting the constant. The Perl therefore has no caller
  and is not dead; a test asserts it is still there, with the reason.
  `prepare_data.sh` itself is gone -- its two other outputs,
  `data/local/waves_all.list` and a copy of `task.arpabo`, were read by
  nothing.
  **A recipe can hold Python now, and `egs/` cannot simply be handed to
  the linters.** A recipe directory is full of symlinks: `run` points
  into `BenNevis/`, which isort is deliberately not run over, and
  `utils`, `steps` and `local` point into a Kaldi checkout, whose own
  Python would be linted on any machine that has one. Measured:
  `isort egs/` reports three `BenNevis/run` files and `black egs/` walks
  sixty-four. So the linters name each recipe file, and
  `tests/recipes/test_linted_recipes.py` fails when a recipe gains one
  that is not listed -- the same shape as `test_run_stages.py` asserting
  its own membership.
  Two checks moved rather than being deleted: `test_yesno_split.py` read
  `prepare_data.sh` as text for the `.wav` filter and the stated split
  count, and both are now asserted by **running** the code.
  **What is left of Kaldi on the training path is mini-librispeech's
  corpus preparation and nothing else**: `local/data_prep.sh`,
  `local/prepare_dict.sh` and the two download scripts, all of which are
  Kaldi's own and live in the checkout. yesno's `local/prepare_dict.sh`
  is in-repo and is pure `cp` and `grep`.

- **Stage 1's language directory is Python too, on the two recipes whose
  settings have been checked.** `utils/prepare_lang.sh` is gone from
  yesno and mini-librispeech, replaced by
  `python3 -m BenNevis.lm prepare-lang`, which has been byte-exact on
  nine files since it was written and was waiting only for a call site --
  the same state `run/make_fbank.py` was in.
  **Only those two convert, and that is the point rather than an
  omission.** `prepare-lang` must be given the same `--sil-prob` and
  `--position-dependent-phones` the shell script was, and getting them
  wrong builds a lexicon that is *wrong rather than broken*: the tables
  come out, the FSTs compose, every transcription is quietly worse.
  `egs/librispeech/asr{1,2,3}` and `egs/timit/asr_{char,phn}` have never
  had theirs checked and keep Kaldi's script.
  `tests/recipes/test_stage1_wiring.py` pins the coupling rather than the
  list: **a recipe converted at stage 1 is exactly a recipe converted at
  stage 3**, because both rebuild a lexicon from the dictionary and both
  need those two settings, so converting one end without the other fails.
  **Both settings are now stated explicitly, and that is a real
  improvement rather than tidiness.** While stage 1 was
  `prepare_lang.sh`, mini-librispeech passed neither and got 0.5 and
  position-dependent from Kaldi's defaults -- so its silence probability
  lived in Kaldi rather than in its own recipe, and
  `tests/recipes/test_lexicon_settings.py` had to encode that default to
  check the three stages agreed. The implicit default was the whole shape
  of the trap; the tests now require the values to be present and compare
  them directly.
  **The language model left stage 1, which is what makes the stage
  Kaldi-free at all.** `local/prepare_lm.sh` and `local/format_lms.sh`
  ran there and need `arpa2fst`, `fstcompile`, `fstcompose` and
  `fstinfo` -- for `G.fst`, which **only decoding reads**. So stage 1
  wanted OpenFST for an artefact training never touches. They run at the
  top of stage 4 now, beside the graphs they feed, and the test pins both
  halves: not in stage 1, and *present in stage 4* before
  `prepare_graph.sh`, which is the difference between moving a step and
  deleting it.
  **`prepare-lang` compiles the binary lexicon FSTs when it can, and the
  three-way flag is the established pattern.** `--compile-fsts auto|true|
  false` is `run/prepare_topo.py`'s, for the same reason: nothing in
  BenNevis's lane reads a binary FST -- training reads `k2/L.fst.txt` and
  `run/decode_k2.py` builds its graph in k2 -- while Kaldi's
  `prepare_graph.sh` composes `L_disambig.fst`. So a tree with no OpenFST
  gets a language directory it can train from, and the decoding lane
  fails pointing here. The compiling itself moved into
  `BenNevis.utils.openfst`, shared with `prepare_topo.py`, because two
  implementations of one thing are how the two come to disagree and what
  they would disagree about is the **arc order** of a graph the decoder
  loads -- which composes either way and decodes differently.
  Compiling **our text** rather than rebuilding is safe and is why this
  is cheap: `BenNevis.lm.lexicon` writes *numeric* text that is
  byte-identical to `fstprint`'s output for both corpora, so it compiles
  without symbol tables and gives the same binary as Kaldi's symbolic
  text compiled with them.
  **`prepare_graph.sh`'s input guard could not fail, and that was found
  by reading it.** It said
  `echo "$0: $src/$file is missing!" || exit 1`, where `||` binds to
  `echo`, `echo` always succeeds and the `exit` is unreachable -- so the
  script announced the missing input and carried on to fail inside
  `fsttablecompose` with something unrelated. Same family as the
  nightly's `grep | awk` under `pipefail`, the `| tee` that made a gate
  unable to fail, and the `G.fst` gate that skipped unconditionally.
  It matters more now than it did: a missing `L_disambig.fst` is a
  *reachable* state once stage 1 may skip it, so the guard now exits and
  names the flag that makes it loud.
  `tests/recipes/test_prepare_graph_guard.py` lifts the loop out of the
  script and runs it, five of its seven tests failing against the old
  form.
  **And the capture's `lang/` retires the same way its features did.**
  `produced_by` replaced `features_from` and is a **map from stage to
  implementation** rather than a flag, because the stages convert one at
  a time: `"features"` covers `raw/`, `features/` and `cmvn/`, `"lang"`
  covers `lang/` and `dict/`. `tests/lm/test_prepare_lang.py`'s
  nine-files-byte-exact class skips when the capture's `lang` is ours,
  and `tests/reference/compare_lang.py` is the live replacement: it runs
  Kaldi's `utils/prepare_lang.sh` into a scratch directory with the
  settings **read out of the recipe's own stage-1 call** -- not out of the
  workflow, which would be a copy that drifts -- and compares the tables
  as bytes and both lexicon FSTs against `fstprint`. Equality, since both
  were byte-exact first time.
  The two `tmp/` entries need no such handling and that is worth knowing:
  the Python needs no temporary directory, so a fresh capture simply has
  no `tmp/lexiconp.txt`, and the suffixing and disambiguation tests in
  `tests/reference/test_corpora.py` already skip on absence. **A gate
  that disappears by absence is safer than one that disappears by
  silence.**
  **What is left on the training path is the corpus-specific preparation
  and nothing else**: yesno's `local/prepare_data.sh` with its three Perl
  scripts and Kaldi's `utils/utt2spk_to_spk2utt.pl`, and
  mini-librispeech's `local/data_prep.sh`, `local/prepare_dict.sh` and
  two download scripts, all four of which are Kaldi's own and live in the
  checkout. `local/prepare_dict.sh` for yesno is already pure `cp` and
  `grep`.

- **Stage 2 is Python now, and the whole of it went in one change.**
  `steps/make_fbank.sh`, `steps/compute_cmvn_stats.sh`,
  `utils/fix_data_dir.sh` and the `apply-cmvn` inside
  `run/apply_cmvn.sh` are gone from all **seven** recipes, and with
  `apply_cmvn.sh` reduced to a shim that drops `utils/split_data.sh`,
  `run.pl` and `utils/parse_options.sh` as well. The arithmetic needed
  nothing: `BenNevis.utils.cmvn` has been bit-exact with `apply-cmvn`
  since the first stage of the migration and `run/make_fbank.py` has
  existed since #102. **What was missing was the call site**, which is
  the state this project keeps rediscovering -- a replacement nothing
  calls is not a replacement, and the way that state persists is by
  nobody asserting the call site. `tests/recipes/test_stage2_wiring.py`
  asserts it now, over recipes it discovers rather than lists, and it
  found a defect on its first run: **`run/make_fbank.py` had no execute
  bit**, so every converted recipe would have died in stage 2 after
  downloading the corpus.
  **`--compress` is passed deliberately and is the reason this change
  moves no WER.** `make_fbank.sh` wrote compressed archives, so Kaldi's
  `CompressedMatrix` quantised every feature to 8 bits, and every number
  in both ledgers was measured on quantised features. Writing
  uncompressed would change CMVN's inputs, hence the statistics, hence
  every WER -- a measured decision of the same kind as dropping pitch,
  and not one to take by deleting a flag. So the switch is like for like
  and the uncompressed question stays separate, one variable at a time.
  **`--nj` is gone rather than ignored, and that is measured.** The shell
  split the data by speaker purely so that eight `apply-cmvn` processes
  could run at once. In Python the whole of mini-librispeech's
  `train_clean_5` -- 1,519 utterances, 1.79M frames, 80 dimensions --
  takes **1.2s** to accumulate statistics over and **0.8s** to normalise
  and write, against 3.9s merely to write the compressed archive; the
  filterbank itself is **1ms** for a twelve-second utterance at four
  torch threads. A flag whose best case is a second can only go wrong, so
  a recipe still passing it gets an argument error rather than a silent
  no-op. What is *not* measured is the `flac` decode, which is the real
  cost of stage 2 on mini-librispeech and is where Kaldi's `nj=8` helped;
  expect stage 2 to cost a minute or two more than its 2.8 and read the
  next run for the figure.
  **`fix_data_dir.sh` was read rather than inferred, and two of its rules
  are not guessable.** A speaker absent from `cmvn.scp` loses **all of
  its utterances**, because `filter_speakers` calls its helper with the
  arguments the other way round -- filtering the *speaker list* by
  `cmvn.scp` -- and then rebuilds `utt2spk` from the surviving `spk2utt`.
  And `wav.scp` is utterance-indexed **only when there is no `segments`
  file**; with one it is recording-indexed and filtering it against an
  utterance list would empty it, so `BenNevis.utils.datadir` refuses a
  directory holding `segments`, `vad.scp`, `utt2dur` or the other files
  it does not filter rather than leaving one stale. The third rule is the
  one Kaldi separates into two lists and a reimplementation will merge:
  `vad.scp` is **rewritten but never intersected**, so a stale copy
  cannot delete utterances -- and `feats.cmvn.scp` is ours and is exactly
  that case, written by the next stage and therefore behind the others on
  a re-run. All five of those are mutation-verified.
  `make_fbank.py` also had to learn Kaldi's tolerance: one unreadable
  utterance is logged and skipped, left out of `feats.scp`, and dropped
  from the rest of the directory by `fix_data_dir` afterwards. The first
  version died on it -- the same defect that cost run 46100161743's
  capture, one stage along, and the second time that shape has been
  shipped in a month.
  **The consequence worth understanding is what it does to the captured
  reference.** `raw/`, `features/` and `cmvn/` are stage 2's input,
  output and statistics, so from this change on a fresh capture records
  **ours**, and `tests/utils/test_cmvn.py`'s equivalence class would
  become `apply(raw, stats) == apply(raw, stats)` -- a test that cannot
  fail, which is this repository's most-repeated defect and the exact
  failure the decoding graph's token table had. The manifest now records
  `features_from`, read off the recipe's own `run.sh` so a converted
  recipe cannot be captured as though it had not; the equivalence tests
  skip with that reason when it is not Kaldi's, and
  `tests/reference/test_corpora.py` fails when a committed capture stops
  being Kaldi's. That is the per-stage retirement this file prescribes,
  taken deliberately rather than by deleting a test.
  **So the comparison moved into the nightly**, as
  `tests/reference/compare_features.py`, which runs Kaldi's own
  `compute-fbank-feats`, `compute-cmvn-stats` and `apply-cmvn` into a
  scratch directory on the same `wav.scp` and the same `conf/fbank.conf`
  and compares four things. The one to read first is **the float32 scale
  and offset**, gated at exact equality: the double statistics are *not*
  expected to agree to the bit, because numpy pairwise-sums a matrix
  where Kaldi accumulates frame by frame -- which carries less error, not
  more -- and the normaliser is what decides whether that last-bit
  difference can reach a feature at all. It is `always()` and blocking,
  with `set -eo pipefail` before the `| tee` for the reason the graph
  gate had to learn twice, and it discovers its data sets rather than
  having yesno's and mini-librispeech's names written into the workflow.
  **What is left on the training path is stage 1 and nothing else.**
  `local/data_prep.sh`, `local/prepare_dict.sh`, `utils/prepare_lang.sh`
  and -- for mini-librispeech -- the download scripts, all of which live
  in the Kaldi checkout. `prepare_lang.sh` is the easy one: nine files
  byte-exact and already written, needing only the call site, which is
  the same shape as this change. `local/data_prep.sh` is the one with no
  counterpart.

- **Stage 3 needed Kaldi for an argument parser and an artefact, and does
  not any more.** `prepare_topo.sh` sourced **Kaldi's
  `utils/parse_options.sh`** to read its own flags, and `utils/` is a
  symlink into `tools/kaldi`, so the script could not parse `--topo`
  without a Kaldi checkout; and it ran `fstcompile` unconditionally for
  `T.fst` and `k2/T.fst`, which **nothing in the Python lane reads** --
  `T.fst` is for Kaldi's `prepare_graph.sh`, and `k2/T.fst` only for
  `Lang.load_topo`'s legacy fallback. So a stage whose every output had
  become Python still failed without Kaldi. It is `run/prepare_topo.py`
  now, with the shell file reduced to a shim that `exec`s it and sources
  neither `path.sh` nor `parse_options.sh`; `--compile-fsts auto` writes
  the two FSTs when OpenFST is on PATH and says what it skipped when it is
  not, `true` requires it, and `prepare_graph.sh` passes `true` and fails
  pointing here when `T.fst` is missing.
  **Four of `BenNevis/run/*.sh` still source `utils/parse_options.sh` and
  five source `path.sh`** -- `align.sh`, `decode_faster.sh`,
  `predict.sh` and `prepare_graph.sh`, plus `prepare_nolm.sh` for
  `path.sh` alone. It was six and seven when this entry was written;
  `prepare_topo.sh` and then `apply_cmvn.sh` became shims, and the
  pattern for the rest is set: a Python entry point plus a shim, which
  also makes the thing testable in tier 1 rather than only in tier 3.
  Two things that came out of doing it. The symbol tables are now written
  by two entry points -- this and `python3 -m BenNevis.topos` -- which is
  exactly how two implementations of one thing come to disagree, and what
  they would disagree about is **the network's output indices**, which a
  checkpoint is only valid while they stay put. So
  `tests/recipes/test_prepare_topo.py` asserts every table is
  **byte-identical** to the other CLI's, for five topologies. And the
  graph-chain document's test began demanding that the document explain a
  binary called `fsts`: its extractor matched `fst[a-z]+` anywhere, so the
  flag `--compile-fsts` read as an invocation. A test that scrapes a
  script for command names has to exclude flags.

- **The remaining POSIX-only sites.** `Lang.load_lexicon` **no longer
  shells out, and that was the last thing on the training path that did.**
  This entry used to say that getting off `fstprint` "needs the
  dictionary at preparation time, so it is a larger piece of goal 2". It
  stopped being larger the moment `build_lexicon_fst_text` became
  byte-exact, which it now is for both corpora -- the blocker had been
  discharged by other work and the entry had not noticed. `prepare_topo.sh --lexicon` writes
  `k2/L.fst.txt` with `python3 -m BenNevis.lm lexicon-fst`, and
  `load_lexicon` reads it, falling back to `fstprint` with a warning for a
  directory prepared before this — the same arrangement `load_topo` has
  for `k2/topo.yaml`. Measured on mini-librispeech's 206,510-entry
  lexicon: **1,539,598 lines in 1.3 seconds, byte-identical to the digest
  the capture recorded for Kaldi's `L.fst`**, and 1.2 seconds to read back
  into k2 at 649MiB, once, before it is cached as `k2/L_inv.pt`.
  Three things about it are worth knowing.
  **The flag is optional, and that is a deliberate gap rather than an
  oversight.** Seven recipes call `prepare_topo.sh` and only yesno and
  mini-librispeech have had their settings checked against their own
  `prepare_lang.sh` call; the other five keep the fallback.
  `tests/recipes/test_lexicon_settings.py` pins which recipes are in which
  group, so converting one without listing it fails.
  **Its defaults are the conservative ones, not Kaldi's.**
  `prepare_topo.sh` defaults `--sil-prob` to 0.0 and position-dependence
  to false where `prepare_lang.sh` defaults to 0.5 and true, so a recipe
  that forgets to pass them gets a loud mismatch against its `phones.txt`
  rather than a quietly different lexicon — this is the stage-10 settings
  trap one stage earlier, and with a worse blast radius, since stage 3
  feeds every **training** run rather than one decode.
  **A mutation that dropped the flag from the shell command survived the
  first version of those tests.** Checking that the recipe passes
  `--position-dependent-phones true` says nothing about whether the script
  forwards it, so the test now reads the `BenNevis.lm lexicon-fst`
  invocation itself. Same shape as every other "wrong rather than broken"
  entry here: the recipe looked right from both ends and the middle was
  dropping it.
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
  absent file can be told from a lost one.
  **It had been written here that the uploaded workflow artefact keeps
  the bytes for ninety days. It does not.** `store_exact` returns before
  writing the file, so a `large` artefact's content is never put on disc
  at all and the uploaded artefact has no more of it than the committed
  capture does -- the digest and the shape are the whole record. This was
  found by downloading a reference artefact to inspect Kaldi's `LG.fst`
  and finding `graph/ctc/` held only `tokens.txt` and `k2/`. It is the
  third invented reassurance found in this file, after the cache example
  and the lost-runner log buffer, and the pattern in all three is the
  same: the claim was the part of the entry nobody would check. If the
  bytes are wanted, `fstprint` them in the job that has them.
  Content is only ever needed for a
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
  **A ceiling cancellation has the opposite shape, and that is the cheap
  way to tell them apart.** Run 45715720923 shows it: `Run the recipe`
  `cancelled` with a real `completed_at`, the elapsed time within two
  minutes of the configured ceiling, and **every later step run and
  succeeded** -- the capture, the graph comparison and both uploads. So
  `gh run view --json jobs` settles which of the two happened without
  reading a log or an annotation, and a ceiling cancellation still yields
  its artefacts while a lost runner yields nothing.
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
