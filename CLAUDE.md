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
  (FFT and accumulation order differ). Set `dither=0` on both sides or nothing
  will ever match; check `snip-edges` and `energy-floor` defaults too.
- **Losses**: agreement to ~1e-4 on a fixed batch. This is the fastest and
  most localising signal — prefer it over WER when validating a migration.
- **WER**: the backstop, not the primary tool. Gate on a delta, not equality.

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
- **`egs/yesno/asr` assigns every utterance one speaker.** `prepare_data.sh`
  writes `global` for all of them, and Kaldi's `split_scp.pl` refuses to split
  data into more parts than there are speakers, so the recipe could not get
  past feature extraction with its own default of `nj=4`. It is `nj=1` now.
  `run/apply_cmvn.sh` already handled this by clamping `nj` to the speaker
  count; anything new that splits by speaker should do the same.
- **Shell text processing is dialect-dependent, and was relied on.**
  `prepare_topo.sh` selected disambiguation symbols with `grep "#[^\d]"`,
  which picks out `#0` only where `\d` inside a bracket expression means the
  literal characters `\` and `d`. That is POSIX, but under `ugrep` or
  `grep -P` it is a digit class, the pattern stops matching, and the token
  table silently comes out as `#0_0`, `#0_1`. Prefer Python for anything that
  parses a symbol table.
