#!/bin/bash
# Check that the image contains everything a recipe actually invokes.
#
# The list is not decorative: every entry here appears in egs/yesno/asr/run.sh
# or in one of the scripts it calls, and a missing one fails a recipe part way
# through after minutes of work. Checking them in two seconds is the cheapest
# possible version of that failure.
#
# Usage: docker/smoke.sh
# Authors:
#   * Zeyu Zhao (The University of Edinburgh) 2026

set -uo pipefail

kaldi_root="${KALDI_ROOT:-/opt/kaldi}"

# Kaldi binaries, by the stage that needs them.
binaries=(
    # feature extraction and CMVN
    compute-fbank-feats
    compute-and-process-kaldi-pitch-feats
    compute-cmvn-stats
    apply-cmvn
    paste-feats
    copy-feats
    # language and graph preparation
    arpa2fst
    fstaddselfloops
    fsttablecompose
    fstdeterminizestar
    fstminimizeencoded
    fstrmsymbols
    fstrmepslocal
    # the diagnostics in egs/yesno/asr/local/prepare_lm.sh
    fstisstochastic
    # decoding
    decode-faster
)

# OpenFST tools, used by prepare_topo.sh and Kaldi's prepare_lang.sh.
openfst_binaries=(fstcompile fstprint fstarcsort fstinfo fstcompose)

# Scoring.
sctk_binaries=(sclite)

# Kaldi's shell and Perl helpers, reached through path.sh and the utils and
# steps symlinks in a recipe directory.
scripts=(
    "$kaldi_root/egs/wsj/s5/utils/parse_options.sh"
    "$kaldi_root/egs/wsj/s5/utils/prepare_lang.sh"
    "$kaldi_root/egs/wsj/s5/utils/split_data.sh"
    "$kaldi_root/egs/wsj/s5/utils/split_scp.pl"
    "$kaldi_root/egs/wsj/s5/utils/int2sym.pl"
    "$kaldi_root/egs/wsj/s5/utils/fix_data_dir.sh"
    "$kaldi_root/egs/wsj/s5/utils/utt2spk_to_spk2utt.pl"
    "$kaldi_root/egs/wsj/s5/utils/run.pl"
    "$kaldi_root/egs/wsj/s5/steps/make_fbank_pitch.sh"
    "$kaldi_root/egs/wsj/s5/steps/compute_cmvn_stats.sh"
)

# Python the recipes need. k2 and torch are the point of the image; kaldiio
# reads the feature archives Kaldi writes.
modules=(torch torchaudio k2 kaldiio hydra omegaconf yaml tqdm numpy)

status=0

report() {
    if [ "$2" = "ok" ]; then
        printf '  %-46s ok\n' "$1"
    else
        printf '  %-46s MISSING\n' "$1"
        status=1
    fi
}

export PATH="$kaldi_root/src/featbin:$kaldi_root/src/bin:$kaldi_root/src/fstbin:$PATH"
export PATH="$kaldi_root/src/latbin:$kaldi_root/src/gmmbin:$PATH"
export PATH="$kaldi_root/tools/openfst/bin:$kaldi_root/tools/sctk/bin:$PATH"

echo "Kaldi binaries:"
for name in "${binaries[@]}" "${openfst_binaries[@]}" "${sctk_binaries[@]}"; do
    if command -v "$name" > /dev/null 2>&1; then
        report "$name" ok
    else
        report "$name" missing
    fi
done

echo "Kaldi scripts:"
for path in "${scripts[@]}"; do
    if [ -f "$path" ]; then
        report "$(basename "$path")" ok
    else
        report "$path" missing
    fi
done

echo "Python modules:"
for name in "${modules[@]}"; do
    if python3 -c "import $name" > /dev/null 2>&1; then
        report "$name" ok
    else
        report "$name" missing
    fi
done

echo "Versions:"
python3 - <<'PY'
import platform

print(f"  python  {platform.python_version()}")
for name in ("torch", "k2"):
    try:
        module = __import__(name)
    except Exception as error:  # pragma: no cover - reported, not raised
        print(f"  {name:7s} unavailable: {error}")
        continue
    version = getattr(module, "__version__", None) or getattr(
        module, "__dev_version__", "unknown"
    )
    print(f"  {name:7s} {version}")
PY

if [ "$status" -ne 0 ]; then
    echo "smoke: something the recipes need is missing" >&2
else
    echo "smoke: all present"
fi
exit "$status"
