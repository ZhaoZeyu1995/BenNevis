"""
Filterbank features in Python, from the recipe's own `conf/fbank.conf`.

This is the replacement for `steps/make_fbank.sh`, and it is a thin
wrapper rather than an implementation because **torchaudio's is already
faithful to Kaldi**. That was not obvious: CLAUDE.md carried an
unexplained "1.2% linear-energy residual" between
`torchaudio.compliance.kaldi.fbank` and `compute-fbank-feats` through six
refuted hypotheses. The residual was the *reference's* 8-bit compression
-- `make_fbank.sh` writes compressed archives and `CompressedMatrix`
quantises each column to 8 bits -- so everything computed from the
waveform was being compared against a grid. `tools/fbank/probe.py`
implements Kaldi's filterbank independently from `feature-window.cc` and
`mel-computations.cc` and agrees with torchaudio to **4e-6**; the capture
agrees with the nearest grid point to either to **6e-6**, for 53,428 of
53,440 entries exactly.

So what this module owes is not arithmetic but **the settings**. Kaldi's
defaults and torchaudio's differ in places that matter, and a feature
extractor that silently used its own would produce features the recipe
never asked for:

| setting | Kaldi | torchaudio | here |
| --- | --- | --- | --- |
| `dither` | 1.0 | 0.0 | **from the conf**, and the recipes set 0 |
| `energy_floor` | 0.0 | 1.0 | **0.0**, Kaldi's |
| `snip_edges` | true | true | true |
| `num_mel_bins` | 23 | 23 | from the conf |
| `sample_frequency` | 16000 | 16000 | from the conf, and checked against the audio |

`--dither=0` is a precondition rather than a preference: Kaldi's default
of 1.0 adds reproducible-for-Kaldi noise to the waveform, which no other
implementation can match, and with it on the features cannot be replaced
at any tolerance worth the name. So a configuration asking for dither is
refused here rather than quietly honoured.

Reading the conf rather than taking arguments is the point: the same file
drives Kaldi's extractor and this one, so the two cannot be given
different settings -- which is the trap stage 10 and stage 3 both have
with the lexicon.

Functions
---------
read_fbank_conf
    A Kaldi `conf/fbank.conf` as keyword arguments.
fbank
    One utterance's filterbank features.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["FBANK_DEFAULTS", "read_fbank_conf", "fbank"]

#: What Kaldi's `compute-fbank-feats` does unless told otherwise, for the
#: settings this module passes on. torchaudio's defaults differ on
#: `energy_floor` (it uses 1.0) and on `dither` (0.0), so neither set can
#: be relied on implicitly.
FBANK_DEFAULTS: Dict[str, Any] = {
    "num_mel_bins": 23,
    "sample_frequency": 16000.0,
    "frame_length": 25.0,
    "frame_shift": 10.0,
    "low_freq": 20.0,
    "high_freq": 0.0,
    "preemphasis_coefficient": 0.97,
    "remove_dc_offset": True,
    "round_to_power_of_two": True,
    "snip_edges": True,
    "use_energy": False,
    "use_power": True,
    "window_type": "povey",
    "dither": 0.0,
    "energy_floor": 0.0,
}

#: How Kaldi spells each option on a command line, which is what a
#: `conf/fbank.conf` holds.
_CONF_NAMES = {
    "num-mel-bins": "num_mel_bins",
    "sample-frequency": "sample_frequency",
    "frame-length": "frame_length",
    "frame-shift": "frame_shift",
    "low-freq": "low_freq",
    "high-freq": "high_freq",
    "preemphasis-coefficient": "preemphasis_coefficient",
    "remove-dc-offset": "remove_dc_offset",
    "round-to-power-of-two": "round_to_power_of_two",
    "snip-edges": "snip_edges",
    "use-energy": "use_energy",
    "use-power": "use_power",
    "window-type": "window_type",
    "dither": "dither",
    "energy-floor": "energy_floor",
}

_BOOLEANS = {"true": True, "false": False}


def _coerce(name: str, raw: str) -> Any:
    default = FBANK_DEFAULTS[name]
    if isinstance(default, bool):
        if raw.lower() not in _BOOLEANS:
            raise ValueError(f"--{name}: {raw!r} is not true or false")
        return _BOOLEANS[raw.lower()]
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def read_fbank_conf(path: Path) -> Dict[str, Any]:
    """
    A Kaldi `conf/fbank.conf` as keyword arguments for `fbank`.

    The file is a list of command-line options, one per line, with `#`
    comments -- which the recipes use heavily, since `--dither=0` carries
    a paragraph explaining itself.

    Arguments
    ---------
    path: Path
        The configuration file.

    Returns
    -------
    Dict[str, Any]
        Kaldi's defaults with the file's settings applied.

    Raises
    ------
    ValueError
        On an option this module does not pass on, rather than ignoring
        it. A silently dropped `--high-freq` would produce features the
        recipe did not ask for, and nothing downstream would notice.
        Also on `--dither` above zero: Kaldi's dither is reproducible for
        Kaldi alone, so features computed with it cannot be compared with
        Kaldi's at any useful tolerance.
    """
    options = dict(FBANK_DEFAULTS)
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        if not text.startswith("--") or "=" not in text:
            raise ValueError(f"{path}:{number}: cannot read {line.strip()!r}")
        flag, raw = text[2:].split("=", 1)
        if flag not in _CONF_NAMES:
            raise ValueError(
                f"{path}:{number}: --{flag} is not an option this extractor "
                f"passes on. Ignoring it would produce features the recipe "
                f"did not ask for; add it to _CONF_NAMES if torchaudio "
                f"supports it."
            )
        name = _CONF_NAMES[flag]
        options[name] = _coerce(name, raw.strip())

    if options["dither"] != 0.0:
        raise ValueError(
            f"{path} asks for --dither={options['dither']}. Kaldi's dither "
            f"is random noise added to the waveform, reproducible by its own "
            f"generator and by nothing else, so features computed with it "
            f"cannot be compared with Kaldi's at any tolerance worth the "
            f"name. Set --dither=0."
        )
    return options


def fbank(samples, sample_rate: Optional[int] = None, **options):
    """
    One utterance's filterbank features.

    Arguments
    ---------
    samples: array-like
        The waveform as **int16-valued** samples, which is the scale
        Kaldi works in: `compute-fbank-feats` reads a wav and does not
        normalise, so a waveform in [-1, 1] would give features about
        `log(32768**2)` too small. A float array is accepted and used as
        it stands; a caller reading audio with `kaldiio` already has the
        right scale.
    sample_rate: Optional[int]
        The audio's rate. Checked against `sample_frequency` when both
        are given, because a mismatch silently changes every filterbank
        edge -- which is the kind of error that shows up as a WER nobody
        can explain.
    **options
        As `read_fbank_conf` returns.

    Returns
    -------
    torch.Tensor
        `(frames, num_mel_bins)`, the log mel energies.

    Raises
    ------
    ValueError
        If `sample_rate` disagrees with the configured
        `sample_frequency`, or if the waveform is not one dimensional.
    """
    import torch
    import torchaudio.compliance.kaldi as kaldi_compliance

    settings = dict(FBANK_DEFAULTS)
    settings.update(options)

    if sample_rate is not None and float(sample_rate) != float(
        settings["sample_frequency"]
    ):
        raise ValueError(
            f"the audio is {sample_rate} Hz but the configuration says "
            f"{settings['sample_frequency']}; every mel bin edge is derived "
            f"from the rate, so this would silently produce different "
            f"features"
        )

    # Copied where it has to be: `kaldiio` hands back a read-only view of
    # the archive and torch warns once per process about non-writable
    # tensors, which is noise in the middle of a feature extraction log.
    # The same reason `run/decode_k2.py` copies its log probabilities.
    if getattr(samples, "flags", None) is not None and not samples.flags.writeable:
        samples = samples.copy()
    waveform = torch.as_tensor(samples, dtype=torch.float32)
    if waveform.dim() == 2 and waveform.shape[0] == 1:
        waveform = waveform[0]
    if waveform.dim() != 1:
        raise ValueError(
            f"expected one channel of samples, got shape " f"{tuple(waveform.shape)}"
        )

    return kaldi_compliance.fbank(waveform.unsqueeze(0), **settings)
