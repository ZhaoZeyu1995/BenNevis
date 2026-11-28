"""
SpecAugment: masking spans of the filterbank during training.

The first of the regularisers this project has been missing, and the one
with the most to gain: mini-librispeech is five hours of audio and the
`ctc` baseline sits at 66% WER, which is what an undertrained model on
that much data looks like. SpecAugment (Park et al., 2019) is the
standard answer and costs nothing at inference.

Three things about this implementation are decisions rather than
defaults.

**Time masks are placed within each utterance's own length, not within
the padded batch.** A batch is padded to its longest member, and the
duration-based samplers in `BenNevis/samplers/` deliberately group
utterances of *similar* length -- but not identical, and a mask landing
in the padding is a mask that does nothing. Masking uniformly over the
padded width would quietly weaken the augmentation in proportion to how
ragged the batch is, which is the kind of bug that shows up as "it did
not help much".

**The mask width is capped at a fraction of the utterance.** A 100-frame
time mask on a 600-frame yesno utterance removes a sixth of it; on a
60-frame utterance it would remove everything. `max_time_mask_ratio`
bounds each mask relative to the utterance, which is what espnet's
adaptive variant does and what makes one setting usable across corpora
whose utterances differ by an order of magnitude.

**The mask value is zero because the features are normalised.** CMVN has
already made each dimension zero-mean per speaker, so zero *is* the mean
and masking to it is masking to no information. On un-normalised features
zero would be a loud, atypical value -- so this is only correct
downstream of `apply_cmvn`, and it is where the trainer applies it.

What it does **not** do is time warping, the third transform in the
paper. It is the one that needs an interpolation per utterance, it is
reported as the smallest of the three contributions, and espnet's own
default leaves it off.

Classes
-------
SpecAugment
    Frequency and time masking of a padded feature batch.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

__all__ = ["SpecAugment"]


class SpecAugment(nn.Module):
    """
    Mask spans of frequency and time in a padded feature batch.

    A no-op outside training, so the trainer can call it unconditionally:
    `nn.Module.training` is False under `eval()`, and
    `Trainer._valid_batch` sets that before computing a loss.

    Arguments
    ---------
    num_freq_masks: int
        How many frequency masks to apply per utterance. Zero disables
        frequency masking.
    freq_mask_width: int
        The largest number of consecutive mel channels one mask may
        cover. Each mask's width is drawn uniformly from
        `[0, freq_mask_width]`, as the paper specifies -- so a mask of
        zero width is a legitimate draw and not a bug.
    num_time_masks: int
        How many time masks to apply per utterance.
    time_mask_width: int
        The largest number of consecutive frames one mask may cover,
        before `max_time_mask_ratio` is applied.
    max_time_mask_ratio: float
        The largest fraction of an utterance one time mask may cover.
        Bounds `time_mask_width` per utterance, so one setting works
        across corpora with very different utterance lengths.
    mask_value: float
        What to write into the masked spans. Zero is the mean of
        CMVN-normalised features, which is why it is the default and why
        this belongs after normalisation.

    Raises
    ------
    ValueError
        On a negative count or width, or a ratio outside (0, 1] -- a
        ratio of zero would silently disable time masking while the
        configuration said it was on.
    """

    def __init__(
        self,
        num_freq_masks: int = 2,
        freq_mask_width: int = 27,
        num_time_masks: int = 2,
        time_mask_width: int = 100,
        max_time_mask_ratio: float = 0.2,
        mask_value: float = 0.0,
    ) -> None:
        super().__init__()
        if num_freq_masks < 0 or num_time_masks < 0:
            raise ValueError(
                f"mask counts must not be negative, got "
                f"{num_freq_masks} and {num_time_masks}"
            )
        if freq_mask_width < 0 or time_mask_width < 0:
            raise ValueError(
                f"mask widths must not be negative, got "
                f"{freq_mask_width} and {time_mask_width}"
            )
        if not 0.0 < max_time_mask_ratio <= 1.0:
            raise ValueError(
                f"max_time_mask_ratio must be in (0, 1], got "
                f"{max_time_mask_ratio}; zero would disable time masking "
                f"while the configuration said it was on"
            )
        self.num_freq_masks = num_freq_masks
        self.freq_mask_width = freq_mask_width
        self.num_time_masks = num_time_masks
        self.time_mask_width = time_mask_width
        self.max_time_mask_ratio = max_time_mask_ratio
        self.mask_value = mask_value

    def extra_repr(self) -> str:
        return (
            f"freq={self.num_freq_masks}x{self.freq_mask_width}, "
            f"time={self.num_time_masks}x{self.time_mask_width} "
            f"(<={self.max_time_mask_ratio:g} of the utterance)"
        )

    def forward(
        self, features: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Mask a batch in place of its unmasked self.

        Arguments
        ---------
        features: torch.Tensor
            `(batch, frames, channels)`, padded.
        lengths: Optional[torch.Tensor]
            `(batch,)` valid frame counts. Without them the padded width
            is used, which places time masks in the padding -- so they
            are optional in signature and wanted in practice.

        Returns
        -------
        torch.Tensor
            A new tensor, masked. The input is not modified: it may be
            the batch the caller still holds, and an in-place write would
            make a second epoch over a cached batch see the masks.

        Raises
        ------
        ValueError
            If `features` is not three dimensional, or `lengths` does not
            match its batch.
        """
        if not self.training:
            return features
        if features.dim() != 3:
            raise ValueError(
                f"features must be (batch, frames, channels), got "
                f"{tuple(features.shape)}"
            )
        batch, frames, channels = features.shape
        if lengths is not None and lengths.numel() != batch:
            raise ValueError(
                f"lengths has {lengths.numel()} entries for {batch} utterances"
            )

        out = features.clone()
        if lengths is None:
            valid = torch.full((batch,), frames, device=features.device)
        else:
            valid = lengths.to(features.device).long().clamp(min=1, max=frames)

        for _ in range(self.num_freq_masks):
            self._mask_channels(out, channels)
        for _ in range(self.num_time_masks):
            self._mask_frames(out, valid)
        return out

    def _mask_channels(self, out: torch.Tensor, channels: int) -> None:
        """One frequency mask per utterance, each its own width and place."""
        if self.freq_mask_width <= 0 or channels <= 0:
            return
        batch = out.shape[0]
        width = torch.randint(
            0, min(self.freq_mask_width, channels) + 1, (batch,), device=out.device
        )
        start = (
            torch.rand(batch, device=out.device) * (channels - width).clamp(min=0)
        ).long()
        index = torch.arange(channels, device=out.device).unsqueeze(0)
        covered = (index >= start.unsqueeze(1)) & (index < (start + width).unsqueeze(1))
        out.masked_fill_(covered.unsqueeze(1), self.mask_value)

    def _mask_frames(self, out: torch.Tensor, valid: torch.Tensor) -> None:
        """
        One time mask per utterance, inside that utterance's own frames.

        The cap is per utterance: `max_time_mask_ratio` of *its* length,
        not of the padded batch, so a short utterance is not wiped out and
        a long one is not under-masked.
        """
        if self.time_mask_width <= 0:
            return
        batch, frames, _channels = out.shape
        allowed = torch.minimum(
            torch.full_like(valid, self.time_mask_width),
            (valid.float() * self.max_time_mask_ratio).long(),
        ).clamp(min=0)
        width = (torch.rand(batch, device=out.device) * (allowed + 1).float()).long()
        width = torch.minimum(width, allowed)
        start = (
            torch.rand(batch, device=out.device) * (valid - width).clamp(min=0).float()
        ).long()
        index = torch.arange(frames, device=out.device).unsqueeze(0)
        covered = (index >= start.unsqueeze(1)) & (index < (start + width).unsqueeze(1))
        out.masked_fill_(covered.unsqueeze(2), self.mask_value)
