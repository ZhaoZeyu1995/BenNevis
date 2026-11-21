# CLAUDE.md

Conventions for working in BenNevis. Read this before making changes.

## What this project is

BenNevis is an End-to-End ASR toolkit built on Differentiable Weighted FSTs,
with flexible topology support. [PyTorch](https://github.com/pytorch/pytorch)
is the neural network backend and [k2](https://github.com/k2-fsa/k2) the DWFST
backend. Kaldi currently provides data preparation, feature extraction, graph
compilation and WFST decoding.

The distinctive contribution is the **topology zoo** in `BenNevis/bin/topos/`:
CTC, 2-state, 3-state-skip and MMI-CTC families, each defining a different
token-level state machine. Changes must not quietly alter topology semantics.

## Layout

| Path | Contents |
| --- | --- |
| `BenNevis/core/` | `Lang`, `GraphLoss`, `Dataset`/`CollateFunc`, `Trainer` |
| `BenNevis/models/` | Encoders: rnnp, transformer, conformer, wav2vec2, whisper |
| `BenNevis/samplers/` | Duration-based dynamic batch samplers (DDP-synchronised) |
| `BenNevis/bin/topos/` | Topology FST generators |
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
| 1 | *(none)* | the standard library and pytest | every push |
| 1b | `torch` | torch (CPU) | every push |
| 2 | `k2` | k2 CPU wheel | every push |
| 3 | `kaldi` | Kaldi build (Docker) | nightly |
| 4 | `gpu` | CUDA, multi-GPU | manual |

```bash
pytest tests/ -m "not torch and not k2 and not kaldi and not gpu"   # tier 1
pytest tests/ -m "torch"                                            # tier 1b
pytest tests/ -m "k2"                                               # tier 2
```

Tier 1 must stay dependency-free beyond pytest; that is what makes it instant.
A test that needs torch belongs in tier 1b with the `torch` marker, so the
dependency-free tests are never held up by a wheel download.

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

## Environment

Currently a venv in `tools/venv` created by `tools/create_env.sh`. `uv` and a
Docker image are planned; Docker is the prerequisite for the Kaldi-dependent
test tier and for cloud runs.

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
  determines the neural network's output vocabulary.
- **The remaining POSIX-only sites are deliberate.** `Lang.load_topo` and
  `Lang.load_lexicon` shell out to `fstprint | awk`; both disappear with the
  declarative topology rewrite, which builds the FSA in Python, so they are
  not worth making portable first. `run/predict.py`'s `cat | awk` merge and
  `run/align.py`'s `gzip -c` pipe are portable in a few lines of Python each,
  but the entry-point scripts have no test coverage, so they wait for tier 3
  rather than being changed blind.
