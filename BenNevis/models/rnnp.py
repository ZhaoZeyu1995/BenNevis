"""
This file contains the code for the RNNP model.
Please note that the code here is mostly copied and revised from the ESPnet toolkit.
Kindly refer to the following link for the original code:
    https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/rnn/encoders.py
Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import List, Union, Tuple, Optional
from BenNevis.utils.nets import make_pad_mask


class RNNP(torch.nn.Module):
    """
    RNN with projection layer module

    Arguments
    ----------
    idim: int
        Input dimension
    elayers: int
        Number of encoder layers
    cdim: int
        Number of rnn units (resulted in cdim * 2 if bidirectional)
    hdim: int
        Number of projection units
    subsample: list
        List of subsampling numbers
        The length of this list should be the same as elayers
    dropout: float
        Dropout rate in the RNN layers and after the projection layer
    typ: str
        Model type selected from
        "blstm" for bidirectional LSTM with projection layer
        "bgru" for bidirectional GRU with projection layer
        "lstm" for unidirectional LSTM with projection layer
        "gru" for unidirectional GRU with projection layer
    """

    def __init__(
        self,
        idim: int,
        elayers: int,
        cdim: int,
        hdim: int,
        subsample: List[int],
        dropout: float,
        typ: str = "blstm",
    ):
        super(RNNP, self).__init__()
        assert typ in [
            "blstm",
            "bgru",
            "lstm",
            "gru",
        ], (
            """
        typ must be "blstm", "bgru", "lstm", or "gru" but got %s.
        """
            % typ
        )
        bidir = typ[0] == "b"
        for i in range(elayers):
            if i == 0:
                inputdim = idim
            else:
                inputdim = hdim

            RNN = torch.nn.LSTM if "lstm" in typ else torch.nn.GRU
            rnn = RNN(
                inputdim,
                cdim,
                num_layers=1,
                bidirectional=bidir,
                batch_first=True,
            )
            setattr(self, "%s%d" % ("birnn" if bidir else "rnn", i), rnn)

            # bottleneck layer to merge
            if bidir:
                setattr(self, "bt%d" % i, torch.nn.Linear(2 * cdim, hdim))
            else:
                setattr(self, "bt%d" % i, torch.nn.Linear(cdim, hdim))

        self.elayers = elayers
        self.cdim = cdim
        self.subsample = subsample
        assert (
            len(self.subsample) == self.elayers
        ), "#subsample {} and elayers {} mismatch".format(
            len(self.subsample), self.elayers
        )
        self.typ = typ
        self.bidir = bidir
        self.dropout = dropout

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: Union[List[int], torch.Tensor],
        prev_state: Optional[
            Union[List[torch.Tensor], List[Tuple[torch.Tensor]]]
        ] = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Union[List[torch.Tensor], List[Tuple[torch.Tensor]]]
    ]:
        """
        RNNP forward

        Arguments
        ----------
        xs_pad: torch.Tensor
            Batch of padded input sequences (B, Tmax, idim)
        ilens: Union[List[int], torch.Tensor]
            Batch of lengths of input sequences (B)
        prev_state: torch.Tensor
            Batch of previous RNN states
            If typ is blstm*, list of length elayers of
                [((1, B, cdim * 2), (1, B, cdim * 2))]
            If typ is bgru*, list of length elayers of
                [(1, B, cdim * 2)]
            If typ is lstmp*, list of length elayers of
                [((1, B, cdim), (1, B, cdim))]
            If typ is gru*, list of length elayers of
                [(1, B, cdim)]

        Returns
        -------
        xs_pad: torch.Tensor
            Batch of padded hidden state sequences (B, Tmax*, hdim)
            Sequence length can be shorter than that of input due to subsampling.
        ilens: torch.Tensor
            Batch of lengths of hidden state sequences (B)
        elayer_states: Union[List[torch.Tensor], List[Tuple[torch.Tensor]]]
            List of RNN hidden states
            If typ is blstm*, list of length elayers of
                [((1, B, cdim * 2), (1, B, cdim * 2))]
            If typ is bgru*, list of length elayers of
                [(1, B, cdim * 2)]
            If typ is lstmp*, list of length elayers of
                [((1, B, cdim), (1, B, cdim))]
            If typ is gru*, list of length elayers of
                [(1, B, cdim)]
        """
        elayer_states = []
        for layer in range(self.elayers):
            if not isinstance(ilens, torch.Tensor):
                ilens = torch.tensor(ilens)
            xs_pack = pack_padded_sequence(
                xs_pad,
                ilens.cpu(),
                batch_first=True,
            )
            rnn = getattr(self, ("birnn" if self.bidir else "rnn") + str(layer))
            if self.training:
                rnn.flatten_parameters()
            ys, states = rnn(
                xs_pack,
                hx=None if prev_state is None else prev_state[layer],
            )
            elayer_states.append(states)
            ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
            sub = self.subsample[layer]
            if sub > 1:
                ys_pad = ys_pad[:, ::sub]
                if isinstance(ilens, torch.Tensor):
                    ilens = (ilens + 1) // sub
                else:
                    assert isinstance(
                        ilens, list
                    ), "ilens must be list or Tensor, but got %s" % type(ilens)
                    ilens = torch.tensor([int(i + 1) // sub for i in ilens])
            projection_layer = getattr(self, "bt%d" % layer)
            xs_pad = projection_layer(ys_pad)
            if layer < self.elayers - 1:
                xs_pad = torch.tanh(F.dropout(xs_pad, p=self.dropout))

        return xs_pad, ilens, elayer_states


class RNN(torch.nn.Module):
    """
    RNN module

    Arguments
    ---------
    idim : int
        Input dimension
    elayers : int
        The number of encoder layers
    cdim : int
        Hidden state dimension of LSTM or GRU
    hdim : int
        Hidden state dimension of the final projection layer
    dropout : float
        Dropout rate for the RNN layer
    typ : str
        RNN type can be the following.
        blstm: Bidirectional LSTM
        bgru: Bidirectional GRU
        lstm: Unidirectional LSTM
        gru: Unidirectional GRU
    """

    def __init__(
        self,
        idim: int,
        elayers: int,
        cdim: int,
        hdim: int,
        dropout: float,
        typ="blstm",
    ):
        super(RNN, self).__init__()
        assert typ in [
            "blstm",
            "bgru",
            "lstm",
            "gru",
        ], "typ must be (blstm, bgru, lstm, gru) but got {}".format(typ)
        bidir = typ[0] == "b"
        self.nbrnn = (
            torch.nn.LSTM(
                idim,
                cdim,
                elayers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidir,
            )
            if "lstm" in typ
            else torch.nn.GRU(
                idim,
                cdim,
                elayers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidir,
            )
        )
        if bidir:
            self.l_last = torch.nn.Linear(cdim * 2, hdim)
        else:
            self.l_last = torch.nn.Linear(cdim, hdim)
        self.typ = typ

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: Union[List[int], torch.Tensor],
        prev_state: Optional[Union[torch.Tensor, Tuple[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """
        RNN forward

        Arguments
        ---------
        xs_pad : torch.Tensor
            Batch of padded input sequences (B, Tmax, idim)
        ilens : Union[List[int], torch.Tensor]
            Batch of lengths of input sequences (B)
        prev_state : Union[torch.Tensor, Tuple[torch.Tensor]]
            Batch of previous RNN states
            If typ is blstm, a tuple of two tensors,
                ((elayers, B, 2*cdim), (elayers, B, 2*cdim))
            If typ is bgru, a tensor of shape
                (elayers, B, 2*cdim)
            If typ is lstm, a tuple of two tensors,
                ((elayers, B, cdim), (elayers, B, cdim))
            If typ is gru, a tensor of shape
                (elayers, B, cdim)

        Returns
        -------
        xs_pad : torch.Tensor
            Batch of padded hidden state sequences (B, Tmax, 2*cdim) or (B, Tmax, cdim)
        ilens : torch.Tensor
            Batch of lengths of hidden state sequences (B)
        states : Union[torch.Tensor, Tuple[torch.Tensor]]
            Batch of current RNN states
            If typ is blstm, a tuple of two tensors,
                ((elayers, B, 2*cdim), (elayers, B, 2*cdim))
            If typ is bgru, a tensor of shape
                (elayers, B, 2*cdim)
            If typ is lstm, a tuple of two tensors,
                ((elayers, B, cdim), (elayers, B, cdim))
            If typ is gru, a tensor of shape
                (elayers, B, cdim)
        """
        if not isinstance(ilens, torch.Tensor):
            ilens = torch.tensor(ilens)
        xs_pack = pack_padded_sequence(xs_pad, ilens.cpu(), batch_first=True)
        if self.training:
            self.nbrnn.flatten_parameters()
        ys, states = self.nbrnn(xs_pack, hx=prev_state)
        xs_pad, ilens = pad_packed_sequence(ys, batch_first=True)
        xs_pad = self.l_last(xs_pad)
        return xs_pad, ilens, states


class VGG2L(torch.nn.Module):
    """
    VGG-like module

    Arguments
    ---------
    in_channel : int
        Input channel size
        For mono (1-channel) input, this is 1
    """

    def __init__(self, in_channel=1):
        super(VGG2L, self).__init__()
        # CNN layer (VGG motivated)
        self.conv1_1 = torch.nn.Conv2d(in_channel, 64, 3, stride=1, padding=1)
        self.conv1_2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv2_1 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv2_2 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.in_channel = in_channel

    def forward(self, xs_pad, ilens, prev_state=None):
        """
        VGG2L forward

        Arguments
        ---------
        xs_pad : torch.Tensor
            Batch of padded input sequences (B, Tmax, idim)
        ilens : torch.Tensor
            Batch of lengths of input sequences (B)

        Returns
        -------
        xs_pad : torch.Tensor
            Batch of padded hidden state sequences (B, Tmax // 4, 128 * D // 4)
        ilens : torch.Tensor
            Batch of lengths of hidden state sequences (B)
        """
        xs_pad = xs_pad.view(
            xs_pad.size(0),
            xs_pad.size(1),
            self.in_channel,
            xs_pad.size(2) // self.in_channel,
        ).transpose(1, 2)

        xs_pad = F.relu(self.conv1_1(xs_pad))
        xs_pad = F.relu(self.conv1_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)

        xs_pad = F.relu(self.conv2_1(xs_pad))
        xs_pad = F.relu(self.conv2_2(xs_pad))
        xs_pad = F.max_pool2d(xs_pad, 2, stride=2, ceil_mode=True)
        if torch.is_tensor(ilens):
            ilens = ilens.cpu().numpy()
        else:
            ilens = np.array(ilens, dtype=np.float32)
        ilens = np.array(np.ceil(ilens / 2), dtype=np.int64)
        ilens = np.array(
            np.ceil(np.array(ilens, dtype=np.float32) / 2), dtype=np.int64
        ).tolist()

        xs_pad = xs_pad.transpose(1, 2)
        xs_pad = xs_pad.contiguous().view(
            xs_pad.size(0), xs_pad.size(1), xs_pad.size(2) * xs_pad.size(3)
        )
        return xs_pad, ilens, prev_state


class Encoder(torch.nn.Module):
    """
    Encoder module

    Arguments
    ---------
    etype : str
        Encoder type
        It follows the regular expression "^(vgg)?(b)?(lstm|gru)(p)?$"
        Namely, the optional prefix "vgg", optional bidirectional "b",
        the mandatory "lstm" or "gru" and the optional post fix "p".
    idim : int
        Input dimension
    elayers : int
        Number of encoder layers
    eunits : int
        Number of encoder units
    eprojs : int
        Number of encoder projection units
    odim : int
        Output dimension
    subsample : Union[List[int], Tuple[int], np.ndarray]
        List of subsampling numbers
    dropout : float
        Dropout rate in RNN layers and after the projection layers
    in_channel : int
        Input channel size
        For mono (1-channel) input, this is 1
    """

    def __init__(
        self,
        etype: int,
        idim: int,
        elayers: int,
        eunits: int,
        eprojs: int,
        odim: int,
        subsample: Union[List[int], Tuple[int], np.ndarray],
        dropout: float,
        in_channel: int = 1,
    ):
        super(Encoder, self).__init__()
        typ = etype.lstrip("vgg").rstrip("p")
        assert typ in [
            "lstm",
            "gru",
            "blstm",
            "bgru",
        ], "Error: need to specify an appropriate encoder architecture"

        if etype.startswith("vgg"):
            if etype[-1] == "p":
                self.enc = torch.nn.ModuleList(
                    [
                        VGG2L(in_channel),
                        RNNP(
                            get_vgg2l_odim(idim, in_channel=in_channel),
                            elayers,
                            eunits,
                            eprojs,
                            subsample,
                            dropout,
                            typ=typ,
                        ),
                    ]
                )
                logging.info("Use CNN-VGG + " + typ.upper() + "P for encoder")
                self.layernorm = torch.nn.LayerNorm(eprojs)
                self.olayer = torch.nn.Sequential(
                    torch.nn.Linear(eprojs, eprojs),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(eprojs, odim),
                )
            else:
                self.enc = torch.nn.ModuleList(
                    [
                        VGG2L(in_channel),
                        RNN(
                            get_vgg2l_odim(idim, in_channel=in_channel),
                            elayers,
                            eunits,
                            eprojs,
                            dropout,
                            typ=typ,
                        ),
                    ]
                )
                logging.info("Use CNN-VGG + " + typ.upper() + " for encoder")
                if typ.startswith("b"):
                    self.layernorm = torch.nn.LayerNorm(2 * eunits)
                    self.olayer = torch.nn.Sequential(
                        torch.nn.Linear(2 * eunits, 2 * eunits),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(2 * eunits, odim),
                    )
                else:
                    self.layernorm = torch.nn.LayerNorm(eunits)
                    self.olayer = torch.nn.Sequential(
                        torch.nn.Linear(eunits, eunits),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(eunits, odim),
                    )
            self.conv_subsampling_factor = 4
        else:
            if etype[-1] == "p":
                self.enc = torch.nn.ModuleList(
                    [RNNP(idim, elayers, eunits, eprojs, subsample, dropout, typ=typ)]
                )
                logging.info(typ.upper() + " with every-layer projection for encoder")
                self.layernorm = torch.nn.LayerNorm(eprojs)
                self.olayer = torch.nn.Sequential(
                    torch.nn.Linear(eprojs, eprojs),
                    torch.nn.LeakyReLU(),
                    torch.nn.Linear(eprojs, odim),
                )
            else:
                self.enc = torch.nn.ModuleList(
                    [RNN(idim, elayers, eunits, eprojs, dropout, typ=typ)]
                )
                logging.info(typ.upper() + " without projection for encoder")
                if typ.startswith("b"):
                    self.layernorm = torch.nn.LayerNorm(2 * eunits)
                    self.olayer = torch.nn.Sequential(
                        torch.nn.Linear(2 * eunits, 2 * eunits),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(2 * eunits, odim),
                    )
                else:
                    self.layernorm = torch.nn.LayerNorm(eunits)
                    self.olayer = torch.nn.Sequential(
                        torch.nn.Linear(eunits, eunits),
                        torch.nn.LeakyReLU(),
                        torch.nn.Linear(eunits, odim),
                    )
            self.conv_subsampling_factor = 1

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: Optional[List] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """
        Encoder forward

        Arguments
        ---------
        xs_pad : torch.Tensor
            Batch of padded input sequences (B, Tmax, idim)
        ilens : torch.Tensor
            Batch of lengths of input sequences (B)
        prev_states : Optional[List]
            Batch of previous hidden states for each Module
            Please refer to the method "forward" of the class "RNNP" and "RNN".

        Returns
        -------
        xs_pad : torch.Tensor
            Batch of padded hidden state sequences (B, Tmax, eprojs)
        ilens : torch.Tensor
            Batch of lengths of hidden state sequences (B)
        states : List
            Batch of current encoder hidden states for each module.
            Please refer to the method "forward" of the class "RNNP" and "RNN".
        """
        if prev_states is None:
            prev_states = [None] * len(self.enc)
        assert len(prev_states) == len(self.enc)

        current_states = []
        for module, prev_state in zip(self.enc, prev_states):
            xs_pad, ilens, states = module(xs_pad, ilens, prev_state=prev_state)
            current_states.append(states)

        # make mask to remove bias value in padded part
        mask = make_pad_mask(ilens).unsqueeze(-1).to(xs_pad.device)
        xs_pad = self.layernorm(xs_pad)
        xs_pad = self.olayer(xs_pad)
        xs_pad = F.log_softmax(xs_pad, dim=-1)
        return xs_pad.masked_fill(mask, 0.0), ilens, current_states


def get_vgg2l_odim(
    idim: int,
    in_channel: int = 1,
    out_channel: int = 128,
):
    """
    Return the output size of the VGG frontend.

    Arguments
    ---------
    idim : int
        Input dimension
    in_channel : int
        Input channel size
        For mono (1-channel) input, this is 1
    out_channel : int
        Output channel size of VGG frontend, by default 128

    Returns
    -------
    int
        The output size of the VGG frontend
    """
    idim = idim / in_channel
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 1st max pooling
    idim = np.ceil(np.array(idim, dtype=np.float32) / 2)  # 2nd max pooling
    return int(idim) * out_channel  # numer of channels
