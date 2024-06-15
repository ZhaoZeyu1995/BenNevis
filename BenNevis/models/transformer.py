"""
This file contains the code for building a TransformerModel based on the whisper implementation,
which means there are many parts of the code that are copied from the whisper implementation.
Note that there are also modifications to the original whisper implementation to make it compatible with the
BenNevis framework.
Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Iterable
from BenNevis.models.whisper import sinusoids


class Conv1d(torch.nn.Conv1d):
    """
    A Conv1d layer that converts the weight and bias to the same dtype as the input tensor.
    Note that this is copied from the whisper implementation.
    """

    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


class LayerNorm(nn.LayerNorm):
    """
    A LayerNorm layer that converts the weight and bias to the same dtype as the input tensor.
    Note that this is copied from the whisper implementation.
    """

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    """
    A Linear layer that converts the weight and bias to the same dtype as the input tensor.
    Note that this is copied from the whisper implementation.
    """

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class MultiHeadAttention(nn.Module):
    """
    A MultiHeadAttention layer that computes the attention weights and values.
    Note that this is copied from the whisper implementation.

    Arguments
    ---------
    n_state : int
        The dimension of the multi-head attention.
    n_head : int
        The number of heads in the multi-head attention.
    """

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ):
        """
        Arguments
        ---------
        x : torch.Tensor
            The input tensor of shape (B, T, n_state).
        mask : torch.Tensor
            The mask tensor of shape larger than or equal to (T, T).

        Returns
        -------
        x : torch.Tensor
            The output tensor of shape (B, T, n_state).
        qk : torch.Tensor
            The attention weights tensor of shape (B, n_head, T, T).
        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):
    """
    A ResidualAttentionBlock layer that computes the attention weights and values.
    Note that this is copied from the whisper implementation.

    Arguments
    ---------
    n_state : int
        The dimension of the multi-head attention.
    n_head : int
        The number of heads in the multi-head attention.
    """

    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ):
        """
        Arguments
        ---------
        x : torch.Tensor
            The input tensor of shape (B, T, n_state).
        mask : torch.Tensor
            The mask tensor of shape larger than or equal to (T, T).

        Returns
        -------
        x : torch.Tensor
            The output tensor of shape (B, T, n_state).
        """
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class Conv1dSubsampling2(nn.Module):
    """
    A Conv1dSubsampling2 layer that subsamples the input tensor.

    Arguments
    ---------
    idim : int
        The input dimension of the tensor.
    odim : int
        The output dimension of the tensor.
    """

    def __init__(self, idim: int, odim: int):
        super().__init__()
        self.conv1 = Conv1d(idim, odim, kernel_size=3, padding=1)
        self.conv2 = Conv1d(odim, odim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor, xlens: Tensor) -> (Tensor, Tensor):
        """
        Arguments
        ---------
        x : torch.Tensor
            The input tensor of shape (B, T, idim).
        xlens : torch.Tensor
            The length of each input sequence in the batch with shape (B,).

        Returns
        -------
        x : torch.Tensor
            The output tensor of shape (B, T*, odim), where T* = (T + 1) // 2 is the subsampled length.
        xlens : torch.Tensor
            The length of each output sequence in the batch with shape (B,).
        """
        x = x.permute(0, 2, 1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        xlens = (xlens + 1) // 2
        return x, xlens


class Conv1dSubsampling3(nn.Module):
    """
    A Conv1dSubsampling3 layer that subsamples the input tensor.

    Arguments
    ---------
    idim : int
        The input dimension of the tensor.
    odim : int
        The output dimension of the tensor.
    """

    def __init__(self, idim: int, odim: int):
        super().__init__()
        self.conv1 = Conv1d(idim, odim, kernel_size=3, padding=1)
        self.conv2 = Conv1d(odim, odim, kernel_size=5, stride=3, padding=1)

    def forward(self, x: Tensor, xlens: Tensor) -> (Tensor, Tensor):
        """
        Arguments
        ---------
        x : torch.Tensor
            The input tensor of shape (B, T, idim).
        xlens : torch.Tensor
            The length of each input sequence in the batch with shape (B,).

        Returns
        -------
        x : torch.Tensor
            The output tensor of shape (B, T*, odim), where T* = (T) // 3 is the subsampled length.
        xlens : torch.Tensor
            The length of each output sequence in the batch with shape (B,).
        """
        x = x.permute(0, 2, 1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        xlens = (xlens) // 3
        return x, xlens


class Conv1dSubsampling4(nn.Module):
    """
    A Conv1dSubsampling4 layer that subsamples the input tensor.

    Arguments
    ---------
    idim : int
        The input dimension of the tensor.
    odim : int
        The output dimension of the tensor.
    """

    def __init__(self, idim: int, odim: int):
        super().__init__()
        self.conv1 = Conv1d(idim, odim, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv1d(odim, odim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor, xlens: Tensor) -> (Tensor, Tensor):
        """
        Arguments
        ---------
        x : torch.Tensor
            The input tensor of shape (B, T, idim).
        xlens : torch.Tensor
            The length of each input sequence in the batch with shape (B,).

        Returns
        -------
        x : torch.Tensor
            The output tensor of shape (B, T*, odim), where T* = ((T + 1) // 2 + 1) // 2 is the subsampled length.
        xlens : torch.Tensor
            The length of each output sequence in the batch with shape (B,).
        """
        x = x.permute(0, 2, 1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        xlens = (xlens + 1) // 2
        xlens = (xlens + 1) // 2
        return x, xlens


class Conv1dSubsampling6(nn.Module):
    """
    A Conv1dSubsampling6 layer that subsamples the input tensor.

    Arguments
    ---------
    idim : int
        The input dimension of the tensor.
    odim : int
        The output dimension of the tensor.
    """

    def __init__(self, idim: int, odim: int):
        super().__init__()
        self.conv1 = Conv1d(idim, odim, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv1d(odim, odim, kernel_size=5, stride=3, padding=1)

    def forward(self, x: Tensor, xlens: Tensor) -> (Tensor, Tensor):
        """
        Arguments
        ---------
        x : torch.Tensor
            The input tensor of shape (B, T, idim).
        xlens : torch.Tensor
            The length of each input sequence in the batch with shape (B,).

        Returns
        -------
        x : torch.Tensor
            The output tensor of shape (B, T*, odim), where T* = (T + 1) // 2 // 3 is the subsampled length.
        xlens : torch.Tensor
        """
        x = x.permute(0, 2, 1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        xlens = (xlens + 1) // 2
        xlens = (xlens) // 3
        return x, xlens


class Conv1dSubsampling8(nn.Module):
    """
    A Conv1dSubsampling8 layer that subsamples the input tensor.

    Arguments
    ---------
    idim : int
        The input dimension of the tensor.
    odim : int
        The output dimension of the tensor.
    """

    def __init__(self, idim: int, odim: int):
        super().__init__()
        self.conv1 = Conv1d(idim, odim, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv1d(odim, odim, kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv1d(odim, odim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor, xlens: Tensor) -> (Tensor, Tensor):
        """
        Arguments
        ---------
        x : torch.Tensor
            The input tensor of shape (B, T, idim).
        xlens : torch.Tensor
            The length of each input sequence in the batch with shape (B,).

        Returns
        -------
        x : torch.Tensor
            The output tensor of shape (B, T*, odim), where T* = (((T + 1) // 2 + 1) // 2 + 1) // 2 is the subsampled length.
        xlens : torch.Tensor
            The length of each output sequence in the batch with shape (B,).
        """
        x = x.permute(0, 2, 1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = x.permute(0, 2, 1)
        xlens = (xlens + 1) // 2
        xlens = (xlens + 1) // 2
        xlens = (xlens + 1) // 2
        return x, xlens


class TransformerModel(torch.nn.Module):
    """
    A TransformerModel layer that computes the attention weights and values.
    Note that this is mostly copied from the whisper implementation with some modifications for the BenNevis framework.

    Arguments
    ---------
    n_mels : int
        The number of mel spectrogram channels.
    n_ctx : int
        The maximum length of the input sequence due to the positional embedding.
    n_state : int
        The dimension of the multi-head attention.
    n_head : int
        The number of heads in the multi-head attention.
    n_layer : int
        The number of transformer layers.
    odim : int
        The output dimension of the tensor.
    input_layer : str
        The type of input layer to use for the Conv1dSubsampling layer.
    """

    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        odim: int,
        input_layer: str,
    ):
        super().__init__()
        if input_layer == "ConvSub2":
            self.conv = Conv1dSubsampling2(n_mels, n_state)
        elif input_layer == "ConvSub3":
            self.conv = Conv1dSubsampling3(n_mels, n_state)
        elif input_layer == "ConvSub4":
            self.conv = Conv1dSubsampling4(n_mels, n_state)
        elif input_layer == "ConvSub6":
            self.conv = Conv1dSubsampling6(n_mels, n_state)
        elif input_layer == "ConvSub8":
            self.conv = Conv1dSubsampling8(n_mels, n_state)
        else:
            raise ValueError(
                f"Unknown input layer: {input_layer}, expected one of ConvSub2, ConvSub3, ConvSub4, ConvSub6, ConvSub8."
            )

        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)
        self.olayer = nn.Sequential(
            nn.Linear(n_state, n_state),
            nn.LeakyReLU(),
            nn.Linear(n_state, n_state),
            nn.LeakyReLU(),
            nn.Linear(n_state, odim),
        )

    def forward(self, x: Tensor, xlens: Tensor) -> (Tensor, Tensor):
        """
        Arguments
        ---------
        x : torch.Tensor
            The input tensor of shape (B, T, n_mels).
        xlens : torch.Tensor
            The length of each input sequence in the batch with shape (B,).

        Returns
        -------
        x : torch.Tensor
            The output tensor of shape (B, T*, odim), where T* is the subsampled length.
        xlens : torch.Tensor
            The length of each output sequence in the batch with shape (B,).
        """
        x, xlens = self.conv(x, xlens)
        assert x.size(2) == self.positional_embedding.size(1), (
            f"RANK {self.rank}: The input length {x.size(2)} should be equal to the "
            f"positional embedding length {self.whisper_enc.positional_embedding.size(1)}"
        )
        T = x.size(1)
        x = (x + self.positional_embedding[:T]).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        x = self.olayer(x)
        x = F.log_softmax(x, dim=-1)
        return x, xlens


if __name__ == "__main__":
    model = MultiHeadAttention(80, 8)
    print(model)
    x = torch.randn(4, 100, 80)
    xlens = torch.tensor([100] * 4)
    y = model(x)
    print(y[0].shape)
    # print(ylens)
