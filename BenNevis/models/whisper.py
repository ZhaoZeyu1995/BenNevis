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
from BenNevis.utils.nets import lens2mask
from typing import Optional


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


def qkv_attention(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):
    """
    Function to compute the scaled dot-product attention.
    This function is copied from the original implementation of the whisper model.
    We modified it so that it takes an optional mask tensor as input.
    The mask should have a shape of (B, 1, Tq, Tv), where B is the batch size and
    Tq and Tv are the lengths of the query and value tensors, respectively.
    Obvisouly, Tq and Tv should be the same when we use it as self-attention.

    The mask tensor is added to the qk tensor before applying the softmax function.
    You may also be interested in the function lens2mask() in BenNevis.utils.nets,
    which can be used to create the mask tensor from a length tensor.

    Arguments
    ---------
    q: torch.Tensor
        The query tensor with shape (B, Tq, C), where B is the batch size,
        Tq is the length of the query tensor, and C is the number of channels.
    k: torch.Tensor
        The key tensor with shape (B, Tv, C), where B is the batch size,
        Tv is the length of the value tensor, and C is the number of channels.
    v: torch.Tensor
        The value tensor with shape (B, Tv, Cv), where B is the batch size,
        Tv is the length of the value tensor, and C is the number of channels.
    mask: Optional[torch.Tensor]
        The mask tensor with shape (B, 1, Tq, Tv), where B is the batch size,
        Tq and Tv are the lengths of the query and value tensors, respectively.
        The mask tensor is added to the qk tensor before applying the softmax function.

    Returns
    -------
    torch.Tensor
        The output tensor with shape (B, Tq, Cv).
    torch.Tensor
        The attention logits tensor with shape (B, n_head, Tq, Tv).
    """
    n_batch, n_ctx, n_state = q.shape
    scale = (n_state // self.n_head) ** -0.25
    q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
    k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
    v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

    qk = q @ k
    if mask is not None:
        qk = qk + mask
    qk = qk.float()

    w = F.softmax(qk, dim=-1).to(q.dtype)
    return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


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

        if dist.is_available() and dist.is_initialized():
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
        AttnClass = self.whisper_enc.blocks[0].attn.__class__
        AttnClass.qkv_attention = qkv_attention

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
            where T* is (T+1)//2 for whisper models.
        xlens: torch.Tensor
            The length of the output tensor, with shape (B,)
        """
        x = x.permute(0, 2, 1)  # as whisper expects (B, C, T) but outputs (B, T, C)
        x = F.gelu(self.whisper_enc.conv1(x))
        x = F.gelu(self.whisper_enc.conv2(x))
        xlens = (xlens + 1) // 2
        x = x.permute(0, 2, 1)
        xmasks = lens2mask(xlens, x.size(1))
        assert x.size(2) == self.whisper_enc.positional_embedding.size(1), (
            f"RANK {self.rank}: The input length {x.size(2)} should be equal to the "
            f"positional embedding length {self.whisper_enc.positional_embedding.size(1)}"
        )
        T = x.size(1)
        x = (x + self.whisper_enc.positional_embedding[:T]).to(x.dtype)

        for block in self.whisper_enc.blocks:
            x = block(x, mask=xmasks)

        x = self.whisper_enc.ln_post(x)
        x = self.olayer(x)
        x = F.log_softmax(x, dim=-1)

        return x, xlens
