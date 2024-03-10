"""
This module contains the model for finetuning a pre-trained Whisper model.
"""

import logging
import os
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import torch.distributed as dist
import whisper


class WhisperModel(torch.nn.Module):
    def __init__(
            self,
            from_pretrained: str,
            whisper_odim: int,
            fix_conv: bool,
            finetune_last_n_layers: int,
            odim: int,
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
            os.makedirs('exp/downloads', exist_ok=True)
            _ = whisper.load_model(from_pretrained, download_root="exp/downloads")
        dist.barrier()
        logging.info(f"RANK {self.rank}: Loading the pre-trained model {from_pretrained}")
        whisper_model = whisper.load_model(from_pretrained, download_root="exp/downloads")

        self.whisper_enc = whisper_model.encoder

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
                f"transformer layers of the whisper encoder.")
        if self.finetune_last_n_layers > 0:
            assert self.finetune_last_n_layers <= len(self.whisper_enc.blocks), \
                (f"RANK {self.rank}: finetune_last_n_layers should be less than or equal to "
                 f"the number of transformer layers in the whisper encoder"
                 f" but got {self.finetune_last_n_layers} and "
                 f"{len(self.whisper_enc.blocks)} respectively.")
            self.whisper_enc.blocks[-self.finetune_last_n_layers:].requires_grad_(True)

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
        x = self.whisper_enc(x)
        xlens = (xlens + 1) // 2
        x = self.olayer(x)
        x = F.log_softmax(x, dim=-1)
        return x, xlens
