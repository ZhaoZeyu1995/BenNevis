# The Kaldi-dependent test tier, as an image.
#
# BenNevis still leans on Kaldi for data preparation, feature extraction,
# graph compilation and WFST decoding, so the recipes in egs/ cannot run
# without a Kaldi build. Compiling Kaldi takes most of an hour, which is why
# nothing in CI has been able to exercise a recipe: this image is built once,
# cached in the registry, and pulled by the nightly job.
#
# Two stages. The first compiles Kaldi; the second takes the result without
# the compilers, the object files and the static archives, which are only
# needed to produce the binaries and not to run them.
#
# Linux amd64 only, deliberately. Kaldi builds on arm64 and k2 publishes
# aarch64 wheels, but neither combination is tested here, and claiming a
# platform nobody has run is worse than not claiming it. See docker/README.md.

# ---------------------------------------------------------------------------
# Stage 1: Kaldi
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS kaldi-builder

# Pinned, not `master`. The pre-refactor history is a research record and so
# is the toolchain that produced it; a floating Kaldi would make an old result
# impossible to reproduce. Bump deliberately, in its own PR.
ARG KALDI_COMMIT=e02e35f0254bb033fab73d1df99fc34123e31d56

ENV DEBIAN_FRONTEND=noninteractive

# `gfortran` and `libtool` are for OpenBLAS, `subversion` for the sctk
# checkout Kaldi's tools Makefile performs, and `zlib1g-dev` for the
# compressed archives Kaldi reads and writes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        automake \
        autoconf \
        bzip2 \
        ca-certificates \
        cmake \
        g++ \
        gawk \
        gfortran \
        git \
        libtool \
        make \
        patch \
        perl \
        pkg-config \
        python3 \
        sox \
        subversion \
        unzip \
        wget \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Fetched by exact commit rather than cloned and checked out, so the layer
# carries one commit instead of the whole history.
RUN git init /opt/kaldi \
    && cd /opt/kaldi \
    && git remote add origin https://github.com/kaldi-asr/kaldi.git \
    && git fetch --depth 1 origin "${KALDI_COMMIT}" \
    && git checkout --detach FETCH_HEAD

WORKDIR /opt/kaldi/tools

# Informational. It reports a missing Intel MKL, which this build does not
# want -- OpenBLAS is used instead -- so its exit status is not a verdict on
# whether the build can proceed.
RUN ./extras/check_dependencies.sh || true

# OpenFST and friends. The single longest step after src itself.
RUN make -j"$(nproc)"

# OpenBLAS, into tools/OpenBLAS/install, which is where src/configure looks
# for it by default when given --mathlib=OPENBLAS.
RUN ./extras/install_openblas.sh

# sclite, for scoring. Built here rather than with src so that a failure in
# it surfaces in two minutes instead of after the forty-minute src build.
RUN make sclite

WORKDIR /opt/kaldi/src

# --shared keeps the binaries linking against .so files, which is what lets
# the second stage drop the static archives. No CUDA: this image exists for
# the recipe tier, which is CPU-only and free to run.
RUN ./configure --shared --use-cuda=no --mathlib=OPENBLAS \
    && make depend -j"$(nproc)" \
    && make -j"$(nproc)"

# What the runtime stage does not need. The intermediates outweigh the
# binaries several times over.
RUN find /opt/kaldi -name '*.o' -delete \
    && find /opt/kaldi -name '*.a' -delete \
    && rm -rf /opt/kaldi/.git /opt/kaldi/tools/OpenBLAS/.git

# ---------------------------------------------------------------------------
# Stage 2: the image the tests run in
# ---------------------------------------------------------------------------
FROM ubuntu:24.04

LABEL org.opencontainers.image.source=https://github.com/ZhaoZeyu1995/BenNevis
LABEL org.opencontainers.image.description="Kaldi and BenNevis's dependencies, for the recipe test tier"
LABEL org.opencontainers.image.licenses=Apache-2.0

ENV DEBIAN_FRONTEND=noninteractive

# Ubuntu 24.04's python3 is 3.12, which is the version the rest of CI pins,
# so no third-party Python is needed. `perl` and `gawk` are not incidental:
# the recipes are driven by Kaldi's Perl and awk scripts. `bc` and `sox` are
# used by Kaldi's data-preparation utilities, `flac` by some corpora.
RUN apt-get update && apt-get install -y --no-install-recommends \
        bc \
        ca-certificates \
        flac \
        gawk \
        gfortran \
        git \
        libgomp1 \
        perl \
        python3 \
        python3-venv \
        sox \
        wget \
        zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=kaldi-builder /opt/kaldi /opt/kaldi

# These three move together. The k2 wheel is built against one exact torch
# build, and the Python version decides which wheels exist at all, so
# changing any of them means changing the others -- and the same pins appear
# in .github/workflows/ci.yml for the tier 2 job. Keep the two in step.
ARG TORCH_VERSION=2.15.0
ARG K2_VERSION=1.24.4.dev20261112+cpu.torch2.15.0

# A venv at a fixed path, so that tools/env.sh can reach it by symlink and
# the recipes need no modification to run in here.
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
