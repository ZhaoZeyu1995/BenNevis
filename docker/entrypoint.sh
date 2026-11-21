#!/bin/bash
# Wire a mounted BenNevis working tree up to the image's Kaldi and venv.
#
# The recipes reach both through fixed relative paths -- tools/kaldi, created
# by tools/put_kaldi.sh, and tools/venv, created by tools/create_env.sh --
# so rather than teach them about the container, the container creates those
# two links in whatever tree is mounted. Nothing else about the tree is
# touched, and an existing link is left alone so that a developer who has
# their own Kaldi keeps it.
#
# Authors:
#   * Zeyu Zhao (The University of Edinburgh) 2026

set -euo pipefail

workspace="${BENNEVIS_WORKSPACE:-/workspace}"
kaldi_root="${KALDI_ROOT:-/opt/kaldi}"
venv="${BENNEVIS_VENV:-/opt/venv}"

if [ -d "$workspace/BenNevis" ] && [ -d "$workspace/tools" ]; then
    if [ ! -e "$workspace/tools/kaldi" ]; then
        ln -s "$kaldi_root" "$workspace/tools/kaldi"
    fi
    if [ ! -e "$workspace/tools/venv" ]; then
        ln -s "$venv" "$workspace/tools/venv"
    fi
else
    echo "entrypoint: no BenNevis working tree at $workspace;" \
         "mount one with -v \"\$PWD:$workspace\" to run a recipe." >&2
fi

exec "$@"
