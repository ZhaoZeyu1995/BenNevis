# The Kaldi-dependent test tier, as an image.
#
# BenNevis still leans on Kaldi for data preparation, feature extraction,
# graph compilation and WFST decoding, so the recipes in egs/ cannot run
# without it. That is also the dependency goal 2 exists to remove, which
# decides how this image is built: Kaldi is *borrowed*, not built.
#
# The base is the Kaldi project's own CPU image, pinned by digest. Building
# Kaldi from source here was tried first and is not worth maintaining for a
# dependency on its way out -- for the record, because the knot is real and
# someone will hit it again:
#
#   - Kaldi's tools/Makefile fetches OpenFST 1.8.4 from openfst.org, which
#     refuses a CI runner's egress over HTTP and HTTPS alike, and falls back
#     to openslr, whose mirror stops at 1.8.1. That fallback is dead for
#     everyone building recent Kaldi, not just here.
#   - Taking 1.8.1 from openslr instead gets through the tools build and then
#     fails in src: fstext/lattice-weight.h compares a pointer with a
#     character, which only type-checks from OpenFST 1.8.2 on.
#   - OpenBLAS 0.3.13, which Kaldi pins, cannot be compiled on a current CPU
#     without forcing TARGET, because its getarch probe predates the
#     hardware.
#
# So: Kaldi comes prebuilt, and this file adds only what BenNevis needs on
# top. Linux amd64 only, which is what the base publishes.

# kaldiasr/kaldi:cpu-debian12-openblas-2025-07-28. Pinned by digest because a
# tag can move: an image that changed under us would turn a WER ledger into
# noise, and the pre-refactor results are a research record.
FROM kaldiasr/kaldi:cpu-debian12-openblas-2025-07-28@sha256:335fa60ff1b70d5145dfea83bb6e4cd7b9b8e40bfbf11b8688cd04b358f952f2

LABEL org.opencontainers.image.source=https://github.com/ZhaoZeyu1995/BenNevis
LABEL org.opencontainers.image.description="Kaldi and BenNevis's dependencies, for the recipe test tier"
LABEL org.opencontainers.image.licenses=Apache-2.0

ENV DEBIAN_FRONTEND=noninteractive

# The base already carries g++, gfortran, sox, wget, git, perl and python3.
# These are the rest of what the recipes reach for: `bc` and `gawk` in Kaldi's
# shell utilities, `flac` for corpora that ship it, and python3-venv to keep
# BenNevis's Python separate from Kaldi's.
RUN apt-get update && apt-get install -y --no-install-recommends \
        bc \
        flac \
        gawk \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# These two move together: the k2 wheel is built against one exact torch
# build, and the same pins appear in .github/workflows/ci.yml for the tier 2
# job. tests/docker/test_pins.py fails if the two disagree.
#
# Python is the base image's, which is Debian 12's 3.11 rather than the 3.12
# the rest of CI uses. k2 publishes cp311 wheels at this torch version, and
# pyproject's floor is 3.10, so this is a supported configuration -- but it
# does mean the recipe tier exercises a different interpreter from tiers 1
# to 2, which is worth knowing when a failure appears only here.
ARG TORCH_VERSION=2.15.0
ARG K2_VERSION=1.24.4.dev20261112+cpu.torch2.15.0

# A venv at a fixed path, so tools/env.sh can reach it by symlink and the
# recipes need no modification to run in here.
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "torch==${TORCH_VERSION}" torchaudio \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        "k2==${K2_VERSION}" \
        --find-links https://k2-fsa.github.io/k2/cpu.html

# The remaining dependencies come from pyproject.toml rather than a list
# maintained here, so the two cannot drift. BenNevis itself is deliberately
# not installed: the container runs against a mounted working tree, which
# tools/env.sh puts on PYTHONPATH.
COPY pyproject.toml /tmp/build/pyproject.toml
COPY docker/deps.py /tmp/build/deps.py
RUN python3 /tmp/build/deps.py /tmp/build/pyproject.toml > /tmp/build/requirements.txt \
    && cat /tmp/build/requirements.txt \
    && pip install --no-cache-dir -r /tmp/build/requirements.txt \
    && rm -rf /tmp/build

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV KALDI_ROOT=/opt/kaldi
ENV BENNEVIS_VENV=/opt/venv

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
