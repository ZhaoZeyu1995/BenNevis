# A Kaggle GPU smoke test

`smoke.py` answers one question before any work is built on top of it:
**does the BenNevis stack run on a free Kaggle GPU, and does it agree with
the CPU?**

It does not train, does not read a corpus and produces no WER. It takes a
few minutes.

## Why this exists

Training mini-librispeech costs 20 to 35 minutes an epoch on a four-core
CI runner, and stage 10's k2 decode costs 6.46 seconds an utterance, which
has twice exhausted a 350-minute run before the 1,089 utterances of
`dev_clean_2` were finished. Kaggle offers about 30 GPU hours a week on two
T4s, with 12-hour sessions, and it is scriptable. If the stack runs there,
both of those costs change character.

The reason to test before building is that the stack is specific: k2 is
compiled against one exact torch version, and `docker/Dockerfile` and
`ci.yml` pin `k2==1.24.4.dev20261112+cpu.torch2.15.0`. A CUDA wheel exists
at that same k2 build for torch 2.15.0 — `cuda13.2` and `cuda13.4`, cp311
to cp315 — so the GPU path is a wheel swap rather than a version
negotiation. Whether Kaggle's image ships that torch is the thing to find
out, and `smoke.py` reports a mismatch rather than reinstalling torch
underneath it.

## What it checks

| Phase | Question |
| --- | --- |
| environment | what the runner is, the way the nightly's `host.log` does |
| install | does a k2 CUDA wheel exist for *this* image's torch, and install |
| import | does an FSA reach the GPU — the move, not the import |
| loss | **do the GPU and CPU agree** on numerator, denominator, loss and gradient, to 1e-4 |
| pruned search | does `intersect_dense_pruned` run, and pick the same labels |
| arc sorting | **does an unsorted graph change the denominator on CUDA?** |
| multi-GPU | are both T4s visible, and does a two-rank NCCL `all_reduce` work |

Two of those are worth more than the rest. The **loss comparison** is the
gate: a GPU lane that disagreed with the CPU would move the WER ledger for
a reason that is not the model, so a disagreement is a failure and the
speed figure is reported only after. The **arc-sorting** phase settles a
question `CLAUDE.md` has had to leave open — `k2.intersect_dense` requires
arc-sorted graphs, the denominator scores were measured identical either
way on the CPU for all eight topologies, and on CUDA an unsorted graph is
undefined rather than merely slow. No tier of CI runs on CUDA, so this is
the cheapest place that has ever been answerable.

## Running it

Locally, on a CUDA machine, with no Kaggle involved:

```bash
python tools/kaggle/smoke.py
```

On Kaggle, headlessly. Put your API token at `~/.kaggle/kaggle.json`,
replace `YOUR_KAGGLE_USERNAME` in `kernel-metadata.json`, then:

```bash
pip install kaggle
kaggle kernels push -p tools/kaggle
kaggle kernels status  YOUR_KAGGLE_USERNAME/bennevis-k2-gpu-smoke
kaggle kernels output  YOUR_KAGGLE_USERNAME/bennevis-k2-gpu-smoke -p /tmp/out
```

The accelerator must be **GPU T4 x2** for the multi-GPU phase; with one
GPU that phase skips rather than fails. Internet must be on, which needs
phone verification on the account, because the k2 wheel is downloaded at
run time.

It is a **script** kernel rather than a notebook on purpose: the source is
then a reviewable, diffable `.py` in the repository instead of JSON with
outputs baked in, and the same file runs locally.

## What it is not

**Not a gate, and not a source of ledger numbers.** A pull request cannot
block on a Kaggle run, so this is a measurement lane beside CI rather than
part of it; tiers 1 to 4 stay where they are. More importantly, Kaggle runs
its own image rather than one pinned by digest, and the WER ledger's whole
premise is that the environment does not move underneath it — so figures
from here are exploratory until that is thought through properly.

**Kaldi cannot run here**: there is no Docker on Kaggle. That is less of an
obstacle than it looks, because it falls along the seam goal 2 has already
cut — the nightly of 26 November showed the Python and k2 path from
`lexicon.txt` and an ARPA to a scored WER needs no Kaldi binary, matching
`decode-faster` exactly on all eight topologies of yesno. A Kaggle lane
would therefore take `data/` and the graphs as a Kaggle Dataset built by
the GitHub Actions job that still has the image, and run only the Python
stages.

Quotas are a courtesy, not a contract: Kaggle's own FAQ says limits and
hardware vary, and the free accelerator changed from a P100 to two T4s in
September 2026.
