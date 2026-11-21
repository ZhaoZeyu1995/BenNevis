# The Kaldi image

BenNevis still uses Kaldi for data preparation, feature extraction, graph
compilation and WFST decoding, so nothing in `egs/` can run without a Kaldi
build. Compiling Kaldi takes most of an hour, which is why no recipe has ever
been exercised in CI. This image carries that build so it happens once rather
than per run.

```
ghcr.io/zhaozeyu1995/bennevis-kaldi:latest
```

## What is in it

| | |
| --- | --- |
| Kaldi | pinned commit, built `--shared --use-cuda=no --mathlib=OPENBLAS` |
| OpenFST, sctk | from Kaldi's `tools/` |
| Python | 3.12, in a venv at `/opt/venv` |
| torch, k2 | CPU builds, pinned to the same versions as CI's tier 2 |
| BenNevis | **not installed** — see below |

BenNevis itself is deliberately absent. The container runs against a working
tree you mount, which `tools/env.sh` puts on `PYTHONPATH`, so the image tests
the commit in front of it rather than a copy baked in months ago.

The entrypoint creates the two links the recipes expect in that mounted tree,
`tools/kaldi` and `tools/venv`, which `tools/put_kaldi.sh` and
`tools/create_env.sh` would otherwise make. An existing link is left alone,
so your own Kaldi survives.

## Using it

```bash
docker run --rm -it -v "$PWD:/workspace" \
    ghcr.io/zhaozeyu1995/bennevis-kaldi:latest bash

# inside:
cd egs/yesno/asr && ./run.sh
```

To check an image has everything a recipe invokes:

```bash
docker run --rm -v "$PWD:/workspace" \
    ghcr.io/zhaozeyu1995/bennevis-kaldi:latest bash docker/smoke.sh
```

## Building it

```bash
docker build -f docker/Dockerfile -t bennevis-kaldi .
```

The context is the repository root, because the build reads `pyproject.toml`
to find the dependencies to install. Expect somewhere around 45 minutes,
nearly all of it Kaldi's `src`.

CI builds it only when `docker/`, `pyproject.toml` or the workflow itself
changes, and publishes from `master`. `workflow_dispatch` forces a rebuild
without a code change, for a base-image update or a yanked wheel.

## Pins

Four versions are pinned, and three of them move together:

| Pin | Where |
| --- | --- |
| Kaldi commit | `ARG KALDI_COMMIT` |
| OpenFST | `ARG OPENFST_VERSION` + its SHA-256 |
| OpenBLAS | `ARG OPENBLAS_VERSION` |
| torch | `ARG TORCH_VERSION`, and `ci.yml`'s tier 2 |
| k2 | `ARG K2_VERSION`, and `ci.yml`'s tier 2 |
| Python | Ubuntu 24.04's `python3`, which is 3.12 |

OpenFST and OpenBLAS are fetched by the Dockerfile rather than by Kaldi's own
scripts, because neither of Kaldi's routes works from a CI runner: it asks
openfst.org for 1.8.4, which refuses the request, and falls back to openslr,
whose mirror stops at 1.8.1 — so the fallback 404s for everyone on this Kaldi
commit. openslr is reachable, so the image takes 1.8.1 from there, which is
above Kaldi's 1.6.7 floor and leaves the `tools/openfst` symlink the recipes
use unchanged. OpenBLAS is pre-placed too: Kaldi's installer resolves it
through an unauthenticated GitHub API call, rate limited per IP across every
runner sharing one.

The k2 wheel is built against one exact torch build, and the Python version
decides which wheels exist at all. `tests/docker/test_pins.py` checks that
the Dockerfile and `ci.yml` agree, because a mismatch would show up as a k2
import error inside a nightly job, a long way from its cause.

Kaldi is pinned to a full commit rather than a branch on purpose: the
pre-refactor history is a research record, and a floating toolchain would make
an old result impossible to reproduce.

## Platforms

**Linux amd64 only.** Kaldi builds on arm64 and k2 publishes `aarch64`
wheels, but that combination has not been run here, and claiming a platform
nobody has tested is worse than not claiming it. On an Apple Silicon machine
this image runs under emulation, which works and is slow.
