#!/usr/bin/env python3
"""
This module is used to add noise to Librispeech data, where the noise comes from MUSAN dataset.
It takes a kaldi data directory as input and writes out another kaldi data directory with noise added.
The noise added data directory will have the following files:
    * wav.scp
    * utt2spk
    * text
    * spk2gender (if available)
    * utt2dur (if available)
    * utt2num_frames (if available)

All the utterance id will be prefixed with 'noise-SNR-<SNR>-<utterance_id>'.

Note that the output audio files will be in the wav format as the input audio files with the same sampling rate.

Note that this script is particularly useful for small evaluation datasets, as it can generate a large amount of
data with noise added to store in the data directory, which may occupy a large amount of disk space.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""
import torch
import os
import logging
import kaldiio
import torchaudio
import argparse
from tqdm import tqdm
from typing import Callable, Dict, Any, List, Optional
from BenNevis.utils.data import read_keys, read_dict


def addnoise(speech: torch.Tensor, noise: torch.Tensor, SNR: int) -> torch.Tensor:
    """
    Add noise to the speech signal with a given signal-to-noise ratio (SNR) applying torchaudio.functional.add_noise

    Arguments
    ---------
    speech: torch.Tensor
        The speech signal with shape (Ts,)
    noise: torch.Tensor
        The noise signal with shape (Tn,)
    SNR: int
        The signal-to-noise ratio in dB

    Returns
    -------
    noisy_speech: torch.Tensor
        The noisy speech signal with shape (Ts,)
    """
    if speech.shape[0] < noise.shape[0]:
        noise = noise[: speech.shape[0]]
    else:
        noise = torch.cat([noise] * (speech.shape[0] // noise.shape[0] + 1))[
            : speech.shape[0]
        ]
    speech = speech.unsqueeze(0)
    noise = noise.unsqueeze(0)
    noisy_speech = torchaudio.functional.add_noise(speech, noise, torch.tensor([SNR]))
    return noisy_speech.squeeze(0)


def main(args):
    # Check the input data directory
    assert os.path.exists(
        args.data_dir
    ), f"The input data directory {args.data_dir} does not exist."

    # Check the output data directory
    if not os.path.exists(args.output_dir):
        logging.info(
            f"The output data directory {args.output_dir} does not exist, creating it..."
        )
        os.makedirs(os.path.join(args.output_dir, "data", "wavs"), exist_ok=True)
    else:
        logging.info(
            f"The output data directory {args.output_dir} exists, overwriting it..."
        )
        os.makedirs(os.path.join(args.output_dir, "data", "wavs"), exist_ok=True)

    # Check the noise directory
    assert os.path.exists(
        args.musan_dir
    ), f"The noise directory {args.musan_dir} does not exist please download MUSAN dataset from http://www.openslr.org/17/."
    assert os.path.exists(
        os.path.join(args.musan_dir, "noise", "free-sound")
    ), f"The noise directory {args.musan_dir}/noise/free-sound does not exist."

    # Read the data directory
    if not os.path.exists(os.path.join(args.data_dir, "wav.scp")):
        raise ValueError(
            f"The input data directory {args.data_dir} does not contain 'wav.scp' file."
        )
    if not os.path.exists(os.path.join(args.data_dir, "segments")):
        logging.info(
            f"The input data directory {args.data_dir} does not contain 'segments' file, using 'wav.scp' as the utterance id."
        )

    utt2wav = kaldiio.load_scp(os.path.join(args.data_dir, "wav.scp"))

    # Read the musan noise files
    num_noises = 0
    for root, dirs, files in os.walk(
        os.path.join(args.musan_dir, "noise", "free-sound")
    ):
        for file in files:
            if file.endswith(".wav"):
                num_noises += 1

    noise_pointer = 0

    # Add noise to the data
    # Note that the noise file name has the pattern "noise-free-sound-xxxx.wav" where x is a digit.
    output_utt2wav = dict()
    for key, (sr, wav) in tqdm(utt2wav.items()):
        wav = torch.tensor(wav, dtype=torch.float32)
        assert os.path.exists(
            os.path.join(
                args.musan_dir,
                "noise",
                "free-sound",
                "noise-free-sound-%04d.wav" % (noise_pointer),
            )
        ), f"The noise file {args.musan_dir}/noise/free-sound/noise-free-sound-{noise_pointer}.wav does not exist."
        noise_file = os.path.join(
            args.musan_dir,
            "noise",
            "free-sound",
            "noise-free-sound-%04d.wav" % (noise_pointer),
        )
        noise, sr_noise = torchaudio.load(noise_file)
        noise.squeeze_(0)
        if sr_noise != sr:
            noise = torchaudio.functional.resample(noise, sr_noise, sr)

        wav = addnoise(wav, noise, args.snr)
        wav.unsqueeze_(0)

        torchaudio.save(
            os.path.join(
                args.output_dir, "data", "wavs", f"noise-SNR-{args.snr}-{key}.flac"
            ),
            wav,
            sample_rate=sr,
            format="flac",
            bits_per_sample=16,
        )
        output_utt2wav["noise-SNR-%d-%s" % (args.snr, key)] = os.path.abspath(
            os.path.join(
                args.output_dir, "data", "wavs", f"noise-SNR-{args.snr}-{key}.wav"
            )
        )
        noise_pointer = (noise_pointer + 1) % num_noises

    # Write the output data directory
    with open(os.path.join(args.output_dir, "wav.scp"), "w") as f:
        fc = ""
        for key, value in output_utt2wav.items():
            fc += f"{key} flac -c -d -s {value} |\n"
        f.write(fc)

    # Check if other files exist
    if os.path.exists(os.path.join(args.data_dir, "utt2spk")):
        utt2spk = read_dict(os.path.join(args.data_dir, "utt2spk"))
        fc = ""
        with open(os.path.join(args.output_dir, "utt2spk"), "w") as f:
            for key, value in utt2spk.items():
                fc += f"noise-SNR-{args.snr}-{key} noise-SNR-{args.snr}-{value}\n"
            f.write(fc)

    if os.path.exists(os.path.join(args.data_dir, "text")):
        text = read_dict(os.path.join(args.data_dir, "text"))
        fc = ""
        with open(os.path.join(args.output_dir, "text"), "w") as f:
            for key, value in text.items():
                fc += f"noise-SNR-{args.snr}-{key} {value}\n"
            f.write(fc)

    if os.path.exists(os.path.join(args.data_dir, "spk2gender")):
        spk2gender = read_dict(os.path.join(args.data_dir, "spk2gender"))
        fc = ""
        with open(os.path.join(args.output_dir, "spk2gender"), "w") as f:
            for key, value in spk2gender.items():
                fc += f"noise-SNR-{args.snr}-{key} {value}\n"
            f.write(fc)

    if os.path.exists(os.path.join(args.data_dir, "utt2dur")):
        utt2dur = read_dict(os.path.join(args.data_dir, "utt2dur"))
        fc = ""
        with open(os.path.join(args.output_dir, "utt2dur"), "w") as f:
            for key, value in utt2dur.items():
                fc += f"noise-SNR-{args.snr}-{key} {value}\n"
            f.write(fc)

    if os.path.exists(os.path.join(args.data_dir, "utt2num_frames")):
        utt2num_frames = read_dict(os.path.join(args.data_dir, "utt2num_frames"))
        fc = ""
        with open(os.path.join(args.output_dir, "utt2num_frames"), "w") as f:
            for key, value in utt2num_frames.items():
                fc += f"noise-SNR-{args.snr}-{key} {value}\n"
            f.write(fc)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser(description="Add noise to the data directory")
    parser.add_argument("data_dir", help="The input data directory")
    parser.add_argument("output_dir", help="The output data directory")
    parser.add_argument(
        "musan_dir",
        help="The root directory of MUSAN dataset, the script will find the noise files itself from <musan-dir>/noise/free-sound.",
    )
    parser.add_argument(
        "--snr", type=float, default=20, help="The signal-to-noise ratio (SNR) in dB"
    )

    args = parser.parse_args()
    main(args)
