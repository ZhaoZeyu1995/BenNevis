"""
Kaldi's filterbank, written from its own source, against the capture.

This is the script that settled what CLAUDE.md had down for several
entries as an unexplained "1.2% linear-energy residual" between
`torchaudio.compliance.kaldi.fbank` and Kaldi's `compute-fbank-feats`.

**The residual is the reference's 8-bit compression, not an
implementation difference.** `make_fbank.sh` writes compressed archives,
and `CompressedMatrix` quantises each column to 8 bits under its default
`kSpeechFeature` scheme -- so the captured features are Kaldi's output
*put on a grid*, and anything computed from the waveform is compared
against the grid rather than against Kaldi's arithmetic.

Three measurements, on yesno's `1_0_0_0_0_0_0_0`:

  * this implementation against torchaudio's: **4e-6 mean**, so two
    independent readings of Kaldi's source agree with each other;
  * either against the capture: **0.0117 mean, 1.17% in linear energy**,
    the figure that was being chased;
  * the capture against *the nearest grid point to this implementation*:
    **6e-6 mean**, and for **53,428 of 53,440 entries the captured value
    is exactly that nearest grid point**. The twelve that are not sit
    within a rounding tie of halfway.

Half the median grid step is 0.0244, so uniform rounding would average
0.0122 against the 0.0117 observed. The clue that should have given it
away is in CLAUDE.md already: the residual was **independent of the
sample rate and of how many FFT bins a mel bin spans**. Compression does
not care about either, which is why 8 kHz and 16 kHz disagreed by the
same amount -- recorded there as refuting the degenerate-filterbank
hypothesis, when it was pointing at the answer.

Run it with a yesno checkout downloaded to the path below; it needs no
Kaldi and no OpenFST.
"""

import math
import wave

import numpy as np


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float64), sr


def mel(f):  # MelScale
    return 1127.0 * math.log(1.0 + f / 700.0)


def inv_mel(m):
    return 700.0 * (math.exp(m / 1127.0) - 1.0)


def mel_banks(num_bins, padded, sr, low_freq=20.0, high_freq=0.0):
    """MelBanks::MelBanks. num_fft_bins = padded/2, so Nyquist is ignored."""
    num_fft_bins = padded // 2
    nyquist = 0.5 * sr
    high = high_freq if high_freq > 0.0 else nyquist + high_freq
    fft_bin_width = sr / padded
    mel_low, mel_high = mel(low_freq), mel(high)
    delta = (mel_high - mel_low) / (num_bins + 1)
    weights = np.zeros((num_bins, num_fft_bins))
    for b in range(num_bins):
        left = mel_low + b * delta
        centre = mel_low + (b + 1) * delta
        right = mel_low + (b + 2) * delta
        for i in range(num_fft_bins):
            m = mel(fft_bin_width * i)
            if left < m < right:
                weights[b, i] = (
                    (m - left) / (centre - left)
                    if m <= centre
                    else (right - m) / (right - centre)
                )
    return weights


def povey(n):
    a = 2.0 * math.pi / (n - 1)
    return np.array([(0.5 - 0.5 * math.cos(a * i)) ** 0.85 for i in range(n)])


def fbank(
    sig,
    sr,
    num_bins=80,
    frame_ms=25.0,
    shift_ms=10.0,
    preemph=0.97,
    remove_dc=True,
    dtype=np.float64,
):
    n = int(sr * frame_ms / 1000)
    step = int(sr * shift_ms / 1000)
    padded = 1
    while padded < n:
        padded *= 2
    win = povey(n)
    W = mel_banks(num_bins, padded, sr)
    frames = 1 + (len(sig) - n) // step  # snip_edges
    out = np.empty((frames, num_bins), dtype=np.float32)
    eps = np.finfo(np.float32).eps
    for t in range(frames):
        x = sig[t * step : t * step + n].astype(dtype).copy()
        if remove_dc:
            x -= x.sum() / n
        if preemph:
            x[1:] -= preemph * x[:-1]
            x[0] -= preemph * x[0]
        x *= win
        spec = np.abs(np.fft.rfft(x, padded)) ** 2
        energies = W @ spec[: padded // 2]
        out[t] = np.log(np.maximum(energies, eps))
    return out


if __name__ == "__main__":
    import torch
    import torchaudio.compliance.kaldi as K

    S = (
        "/private/tmp/claude-501/-Users-zeyu-Github-BenNevis/"
        "a9a32353-5a31-47f7-9555-5230745ee00c/scratchpad"
    )
    z = np.load("tests/golden/reference/yesno/raw/test_yesno.npz", allow_pickle=True)
    for utt in list(z.keys())[:3]:
        sig, sr = read_wav(f"{S}/downloads/yesno/waves_yesno/{utt}.wav")
        kaldi = z[utt][:, :80]
        mine = fbank(sig, sr)
        ta = K.fbank(
            torch.from_numpy(sig.astype(np.float32)).unsqueeze(0),
            num_mel_bins=80,
            sample_frequency=sr,
            dither=0.0,
            energy_floor=0.0,
            snip_edges=True,
        ).numpy()
        m = min(len(kaldi), len(mine), len(ta))

        def rep(a, b, label):
            d = np.abs(a[:m] - b[:m])
            rel = np.abs(np.exp(a[:m]) - np.exp(b[:m])) / np.maximum(
                np.exp(a[:m]), 1e-12
            )
            print(
                f"    {label:22s} log-mel max {d.max():.6f} mean {d.mean():.7f} "
                f"| linear rel mean {rel.mean()*100:.3f}%"
            )

        print(f"{utt} ({m} frames)")
        rep(kaldi, mine, "kaldi vs mine")
        rep(kaldi, ta, "kaldi vs torchaudio")
        rep(ta, mine, "torchaudio vs mine")
