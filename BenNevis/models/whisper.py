"""
This module contains the model for finetuning a pre-trained Whisper model.
Whisper is a transformer-based model for speech recognition, which was trained on large-scale data.
Here, to finetune the model with the graph-based loss function,
we load the pre-trained model and only take the encoder part.
Besides, the original implementation only supports an input length of 3000,
we modify the model to support any input length less than or equal to 10000.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""

import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import whisper


def sinusoids(length, channels, max_timescale=10000):
    """
    Returns sinusoids for positional embedding
    Note: this is copied from the original implementation of the whisper model.
    """
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class WhisperModel(torch.nn.Module):
    def __init__(
        self,
        from_pretrained: str,
        whisper_odim: int,
        fix_conv: bool,
        finetune_last_n_layers: int,
        odim: int,
        max_len: int = 3000,
    ):
        """
        WhisperModel constructor.

        Arguments
        ---------
        from_pretrained : str
            The name of the pre-trained model to load.
            Now we suport "tiny.en", "base.en", "small.en", "medium.en"
        whisper_odim : int
            The output dimension of the whisper model.
        fix_conv : bool
            Whether to fix the convolutional layers of the whisper encoder.
        finetune_last_n_layers : int
            The number of transformer layers to fine-tune at the end of the whisper encoder.
        odim : int
            The output dimension of the model.
        max_len : int
            The maximum length of the input sequence, defaults to 3000.
        """

        super(WhisperModel, self).__init__()

        self.odim = odim
        self.from_pretrained = from_pretrained
        self.fix_conv = fix_conv
        self.whisper_odim = whisper_odim
        self.finetune_last_n_layers = finetune_last_n_layers

        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
        else:
            self.rank = 0

        if self.rank == 0:
            os.makedirs("exp/downloads", exist_ok=True)
            _ = whisper.load_model(from_pretrained, download_root="exp/downloads")
        dist.barrier()
        logging.info(
            f"RANK {self.rank}: Loading the pre-trained model {from_pretrained}"
        )
        whisper_model = whisper.load_model(
            from_pretrained, download_root="exp/downloads"
        )

        self.whisper_enc = whisper_model.encoder
        assert max_len <= 10000, "max_len should be less than or equal to 10000"
        self.whisper_enc.register_buffer(
            "positional_embedding", sinusoids(max_len, whisper_odim)
        )

        self.olayer = nn.Sequential(
            nn.Linear(self.whisper_odim, self.whisper_odim),
            nn.LeakyReLU(),
            nn.Linear(self.whisper_odim, self.whisper_odim),
            nn.LeakyReLU(),
            nn.Linear(self.whisper_odim, self.odim),
        )
        self.freeze_and_init()

    def freeze_and_init(self):
        if self.fix_conv:
            self.whisper_enc.conv1.requires_grad_(False)
            self.whisper_enc.conv2.requires_grad_(False)

        self.whisper_enc.blocks.requires_grad_(False)
        logging.info(
            f"RANK {self.rank}: Fine-tuning the last {self.finetune_last_n_layers} "
            f"transformer layers of the whisper encoder."
        )
        if self.finetune_last_n_layers > 0:
            assert self.finetune_last_n_layers <= len(self.whisper_enc.blocks), (
                f"RANK {self.rank}: finetune_last_n_layers should be less than or equal to "
                f"the number of transformer layers in the whisper encoder"
                f" but got {self.finetune_last_n_layers} and "
                f"{len(self.whisper_enc.blocks)} respectively."
            )
            self.whisper_enc.blocks[-self.finetune_last_n_layers :].requires_grad_(True)

    def forward(self, x, xlens):
        """
        Forward pass of the model.

        Arguments
        ---------
        x: torch.Tensor
            The input tensor with shape (B, T, C), where T is usually 3000 for whisper models
            and C is 80 (80-dim log-mel filterbank features).
        xlens: torch.Tensor
            The length of the input tensor, with shape (B,)

        Returns
        -------
        x: torch.Tensor
            The output tensor with shape (B, T*, odim),
            where T* is usually (T+1)//2 for whisper models, i.e., 1500 for 3000 as input.
        xlens: torch.Tensor
            The length of the output tensor, with shape (B,)
        """
        x = x.permute(0, 2, 1)  # as whisper expects (B, C, T) but outputs (B, T, C)
        x = F.gelu(self.whisper_enc.conv1(x))
        x = F.gelu(self.whisper_enc.conv2(x))
        x = x.permute(0, 2, 1)
        assert x.size(2) == self.whisper_enc.positional_embedding.size(1), (
            f"RANK {self.rank}: The input length {x.size(2)} should be equal to the "
            f"positional embedding length {self.whisper_enc.positional_embedding.size(1)}"
        )
        T = x.size(1)
        x = (x + self.whisper_enc.positional_embedding[:T]).to(x.dtype)

        for block in self.whisper_enc.blocks:
            x = block(x)

        x = self.whisper_enc.ln_post(x)
        x = self.olayer(x)
        x = F.log_softmax(x, dim=-1)

        xlens = (xlens + 1) // 2
        return x, xlens
