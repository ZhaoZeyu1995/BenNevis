"""
Network related utility functions.
Authors:
    Zeyu Zhao (The University of Edinburgh) 2024
"""

import math
from typing import Optional

import torch


def sinusoids(
    length: int,
    channels: int,
    max_timescale: int = 10000,
) -> torch.Tensor:
    """
    Build a sinusoidal positional embedding table.

    This follows the Whisper formulation, which spaces the timescales
    logarithmically rather than using the original Transformer's fixed
    geometric progression. It lives here rather than beside a model so that
    any encoder can use it without pulling in that model's dependencies.

    Arguments
    ---------
    length: int
        The number of positions, i.e. the maximum sequence length.
    channels: int
        The embedding dimension. Must be even, as half the channels carry
        sines and half cosines.
    max_timescale: int
        The longest wavelength represented, by default 10000.

    Returns
    -------
    embedding: torch.Tensor
        The positional embedding, of shape `(length, channels)`.
    """
    assert channels % 2 == 0, f"channels must be even, but got {channels}"
    log_timescale_increment = math.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


def make_pad_mask(
    lengths: torch.Tensor,
    maxlen: Optional[int] = None,
) -> torch.Tensor:
    """
    Function to transform a list of lengths to a mask tensor.
    The output mask tensor has a shape of (B, maxlen) and dtype of bool,
    where B is the batch size and maxlen is the maximum length of the
    sequences in the batch. The mask tensor is True for padded elements
    and False for non-padded elements. This is useful for applying torch.masked_fill() to sequences,
    but broadcastability has to be dealt with manually.

    Arguments
    ---------
    lengths: torch.Tensor
        Batch of lengths (B,).
    maxlen: Optional[int]
        Maximum length of the tensor. If None, it will be automatically
        decided by the max value of lengths.

    Returns
    -------
    masks: torch.Tensor
        Mask tensor of shape (B, maxlen).
        Note that the dtype is bool.
    """
    if maxlen is None:
        maxlen = lengths.max().item()

    # Create a range tensor that matches the shape we need
    # torch.arange creates a tensor [0, 1, ..., max_length-1]
    # We then compare this range tensor to each sequence length
    range_tensor = (
        torch.arange(maxlen, device=lengths.device)
        .unsqueeze(0)
        .expand(len(lengths), maxlen)
    )

    # Create the mask by comparing each sequence length to the range tensor
    # The comparison is broadcasted across the batch dimension
    masks = range_tensor >= lengths.unsqueeze(1)

    return masks


def lens2mask(
    lengths: torch.Tensor,
    maxlen: int = None,
    dtype=torch.float32,
):
    """
    This function transforms a length tensor to a mask tensor.

    Arguments
    ---------
    lengths: torch.Tensor
        The length tensor of shape (B,).
    maxlen: int
        The maximum length of the sequences in the batch, defaults to None.
    dtype: torch.dtype
        The dtype of the mask tensor, defaults to torch.float32.

    Returns
    -------
    mask: torch.Tensor
        The mask tensor of shape (B, 1, maxlen, maxlen), where B is the batch size
        and maxlen is the maximum length of the sequences in the batch.
        For the i-th sequence, the value mask[i, 0, j, k] is 0. if j < lengths[i],
        and -inf otherwise.
    """
    if maxlen is None:
        maxlen = lengths.max().item()
    mask = torch.arange(maxlen, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(
        1
    )
    mask = mask.float()
    mask[mask == 0] = torch.finfo(dtype).min
    mask[mask == 1] = 0.0
    mask = mask.unsqueeze(1).unsqueeze(1).expand(-1, 1, maxlen, maxlen)
    return mask
