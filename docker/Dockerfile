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

# Two of Kaldi's build inputs are fetched here rather than by its own
# scripts; see the two RUN steps below for why. Both versions are the ones
# the pinned Kaldi asks for, and the build checks that rather than assuming.
# 1.8.1, not the 1.8.4 this Kaldi defaults to, and the reason is reachability
# rather than preference: see the OpenFST step below. Kaldi accepts any
# version from 1.6.7 up and builds a tools/openfst symlink for whichever it
# gets, so nothing downstream notices.
ARG OPENFST_VERSION=1.8.1
ARG OPENFST_SHA256=24fb53b72bb687e3fa8ee96c72a31ff2920d99b980a0a8f61dda426fca6713f0
ARG OPENBLAS_VERSION=0.3.13

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
# whether the build can proceed. The directory is created first because the
# script symlinks a python into it and complains if it is absent.
RUN mkdir -p /opt/kaldi/tools/python \
    && ./extras/check_dependencies.sh || true

# OpenFST, fetched from openslr instead of by Kaldi's Makefile.
#
# Kaldi asks openfst.org for 1.8.4 and falls back to openslr, and on a CI
# runner both fail: openfst.org refuses the request whichever scheme is used
# -- measured, 403 over HTTP and the same refusal over HTTPS -- and openslr's
# mirror stops at 1.8.1, so the fallback 404s. That fallback is dead for
# everyone building this Kaldi commit, not only here.
#
# openslr itself is reachable, which that clean 404 proves, so the route that
# works is openslr at the version it actually carries. Kaldi's floor is 1.6.7
# and OPENFST_VERSION is overridable, so 1.8.1 is a supported choice rather
# than a workaround with consequences; the tools/openfst symlink Kaldi makes
# points at whichever version was built, so path.sh and the recipes are
# unaffected.
#
# Pre-placing the tarball satisfies the Makefile's download target, which has
# no prerequisites, so make skips fetching. Checksummed, because this is a
# build input from neither a distribution repository nor a git host.
RUN grep -q '^OPENFST_VERSION ?=' Makefile \
    && wget -q --tries=3 -O "openfst-${OPENFST_VERSION}.tar.gz" \
        "https://www.openslr.org/resources/2/openfst-${OPENFST_VERSION}.tar.gz" \
    && echo "${OPENFST_SHA256}  openfst-${OPENFST_VERSION}.tar.gz" | sha256sum -c -

# OpenFST and friends. The single longest step after src itself.
# OPENFST_VERSION is passed explicitly: without it make would want 1.8.4 and
# try to download it again.
RUN make -j"$(nproc)" OPENFST_VERSION="${OPENFST_VERSION}"

# OpenBLAS, into tools/OpenBLAS/install, which is where src/configure looks
# for it by default when given --mathlib=OPENBLAS.
#
# Its tarball is pre-placed too, because extras/install_openblas.sh resolves
# the download through an unauthenticated GitHub API call, and that is rate
# limited per IP across every runner sharing it. The URL used here is the one
# the API would have returned, so the archive unpacks to the
# OpenMathLib-OpenBLAS-* directory the script then moves. No checksum: a
# git host's generated archives are not guaranteed byte-stable, and a pin
# that breaks on their side would be worse than none.
#
# TARGET is forced because OpenBLAS 0.3.13 is from December 2020 and does not
# recognise a 2026 CPU: its getarch_2nd probe fails to compile against a
# target its own param.h has never heard of, with a wall of
# SGEMM_DEFAULT_UNROLL_M-undeclared errors. Naming a target skips the probe.
#
# NEHALEM rather than something current, for portability over speed. It needs
# only SSE4.2, so the image runs anywhere, including emulated on an Apple
# Silicon machine where a newer target's AVX2 kernels may not. Nothing here is
# BLAS-bound: torch ships its own kernels and never touches this library, and
# Kaldi's part in this image is feature extraction, graph compilation and
# decoding.
RUN test "$(sed -n 's/^OPENBLAS_VERSION=//p' extras/install_openblas.sh)" \
         = "${OPENBLAS_VERSION}" \
    && mkdir -p /tmp/blas \
    && wget -q --tries=3 -O "/tmp/blas/OpenBLAS-${OPENBLAS_VERSION}.tar.gz" \
        "https://codeload.github.com/OpenMathLib/OpenBLAS/legacy.tar.gz/refs/tags/v${OPENBLAS_VERSION}" \
    && DOWNLOAD_DIR=/tmp/blas TARGET=NEHALEM ./extras/install_openblas.sh \
    && test -f /opt/kaldi/tools/OpenBLAS/install/lib/libopenblas.so \
    && rm -rf /tmp/blas

# sclite, for scoring. `make all` above builds sctk too, so this is an
# explicit check that it is there rather than extra work -- and it runs
# before the forty-minute src build rather than after it.
RUN make sclite OPENFST_VERSION="${OPENFST_VERSION}" \
    && test -x /opt/kaldi/tools/sctk/bin/sclite

WORKDIR /opt/kaldi/src

# --shared keeps the binaries linking against .so files, which is what lets
# the second stage drop the static archives. No CUDA: this image exists for
# the recipe tier, which is CPU-only and free to run.
#
# --openblas-root is given explicitly rather than left to detection, because
# at this commit configure looks for it under tools/extras/OpenBLAS/install
# while extras/install_openblas.sh, run from tools/ as Kaldi's own
# instructions say, puts it in tools/OpenBLAS/install. Saying where it is
# costs nothing and does not depend on which of those is the intended one.
RUN ./configure --shared --use-cuda=no --mathlib=OPENBLAS \
        --openblas-root=/opt/kaldi/tools/OpenBLAS/install \
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
