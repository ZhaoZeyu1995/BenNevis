"""
Network related utility functions.
Authors:
    Zeyu Zhao (The University of Edinburgh) 2024
"""
from typing import Optional
import torch


def make_pad_mask(
    lengths: torch.Tensor,
    maxlen: Optional[int] = None,
) -> torch.Tensor:
    """
    Function to transform a list of lengths to a mask tensor.
    The output mask tensor has a shape of (B, maxlen) and dtype of bool,
    where B is the batch size and maxlen is the maximum length of the
    sequences in the batch. The mask tensor is True for padded elements
    and False for non-padded elements.
    This is useful for applying torch.masked_fill() to sequences,
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
    range_tensor = torch.arange(maxlen).unsqueeze(0).expand(len(lengths), maxlen)

    # Create the mask by comparing each sequence length to the range tensor
    # The comparison is broadcasted across the batch dimension
    masks = range_tensor >= lengths.unsqueeze(1)

    return masks
