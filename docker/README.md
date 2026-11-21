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
| Kaldi | the Kaldi project's own CPU image, pinned by digest |
| OpenFST, sctk | from that image's `tools/` |
| Python | Debian 12's 3.11, in a venv at `/opt/venv` |
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

## Why Kaldi is borrowed and not built

Building Kaldi here was tried first, and abandoned. The knot is worth
recording, because anyone building recent Kaldi in CI will hit it:

- `tools/Makefile` fetches OpenFST 1.8.4 from openfst.org, which refuses a CI
  runner's egress over HTTP and HTTPS alike, and falls back to OpenSLR, whose
  mirror **stops at 1.8.1**. That fallback is dead for everyone on this
  commit, not only here.
- Taking 1.8.1 from OpenSLR instead gets through the `tools` build and then
  fails in `src`: `fstext/lattice-weight.h` compares a pointer with a
  character, which only type-checks from OpenFST 1.8.2 onwards.
- OpenBLAS 0.3.13, the version Kaldi pins, cannot be compiled on a current
  CPU without forcing `TARGET`, because its architecture probe predates the
  hardware.

All of that is maintenance of the dependency goal 2 exists to delete. The
Kaldi project publishes a prebuilt CPU image, so this one derives from it and
adds only what BenNevis needs. Kaldi becomes a black box we borrow, which is
the right posture for something on its way out.

## Building it

```bash
docker build -f docker/Dockerfile -t bennevis-kaldi .
```

The context is the repository root, because the build reads `pyproject.toml`
to find the dependencies to install. A couple of minutes, plus pulling the
2.45GB base.

CI builds it only when `docker/`, `pyproject.toml` or the workflow itself
changes, and publishes from `master`. `workflow_dispatch` forces a rebuild
without a code change, for a base-image update or a yanked wheel.

## Pins

Four versions are pinned, and three of them move together:

| Pin | Where |
| --- | --- |
| Kaldi | the base image's `@sha256:` digest |
| torch | `ARG TORCH_VERSION`, and `ci.yml`'s tier 2 |
| k2 | `ARG K2_VERSION`, and `ci.yml`'s tier 2 |
| Python | the base image's, which is Debian 12's 3.11 |

The base is pinned by digest, not by tag: a tag can move, and an image that
changed underneath a WER ledger would turn it into noise.


The k2 wheel is built against one exact torch build, and the Python version
decides which wheels exist at all. `tests/docker/test_pins.py` checks that
the Dockerfile and `ci.yml` agree, because a mismatch would show up as a k2
import error inside a nightly job, a long way from its cause.

The Python version differs from the 3.12 the other tiers use, because it is
whatever the base image ships. k2 publishes cp311 wheels at this torch
version and `pyproject.toml`'s floor is 3.10, so it is supported — but a
failure that appears only in tier 3 could be an interpreter difference, and
that is worth checking early.

## Platforms

**Linux amd64 only**, which is what the base image publishes. On an Apple
Silicon machine it runs under emulation, which works and is slow.
