"""
This file contains the code for building Wav2VecModel based on a pre-trained model from torchaudio.
This is especially designed for fine-tuning the pre-trained model on a new dataset.
Currently, we support the following pre-trained models:
    * "WAV2VEC2_BASE"
    * "WAV2VEC2_LARGE"
    * "WAV2VEC2_LARGE_LV60K"
    * "WAV2VEC2_XLSR53"
    * "WAV2VEC2_XLSR_300M"
    * "WAV2VEC2_XLSR_1B"
    * "WAV2VEC2_XLSR_2B"
Note that, "WAV2VEC2_BASE" and "WAV2VEC2_LARGE" are not already wrapped, so we need to wrap them
with the _Wav2Vec2Model class as other models.
Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""

import logging
import os
import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import torch.distributed as dist


class Wav2Vec2Model(torch.nn.Module):
    def __init__(
            self,
            from_pretrained: str,
            wav2vec2_odim: int,
            finetune_last_n_layers: int,
            odim: int,
            ):
        """
        Wav2Vec2Model constructor.

        Arguments
        ---------
        from_pretrained : str
            The name of the pre-trained model to load. This should be one of the models
            listed in torchaudio.pipelines._wav2vec2.impl.__all__.
        wav2vec2_odim : int
            The output dimension of the wav2vec2 model.
        finetune_last_n_layers : int
            The number of transformer layers to fine-tune at the end of the wav2vec2 encoder.
        odim : int
            The output dimension of the model.
        """

        super(Wav2Vec2Model, self).__init__()

        self.odim = odim
        self.from_pretrained = from_pretrained
        self.wav2vec2_odim = wav2vec2_odim
        self.finetune_last_n_layers = finetune_last_n_layers

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        if self.rank == 0:
            bundle = getattr(torchaudio.pipelines, from_pretrained)
            os.makedirs('exp/downloads', exist_ok=True)
            wav2vec2 = bundle.get_model(dl_kwargs={"model_dir": 'exp/downloads'})
        dist.barrier()
        logging.info(f"RANK {self.rank}: Loading the pre-trained model {from_pretrained}")
        bundle = getattr(torchaudio.pipelines, from_pretrained)
        os.makedirs('exp/downloads', exist_ok=True)
        wav2vec2 = bundle.get_model(dl_kwargs={"model_dir": 'exp/downloads'})
        # If the model is not an instance of torchaudio.models.wav2vec2.Wav2Vec2Model, wrap it
        # with the _Wav2Vec2Model class
        # It seems that "WAV2VEC2_BASE" and "WAV2VEC2_LARGE" models are not already wrapped
        if isinstance(wav2vec2, torchaudio.models.wav2vec2.Wav2Vec2Model):
            wav2vec2 = torchaudio.pipelines._wav2vec2.utils._Wav2Vec2Model(
                wav2vec2, normalize_waveform=True, apply_log_softmax=False, append_star=False
            )
        elif isinstance(wav2vec2, torchaudio.pipelines._wav2vec2.utils._Wav2Vec2Model):
            wav2vec2.normalize_waveform = True
            wav2vec2.apply_log_softmax = False
            wav2vec2.append_star = False

        self.wav2vec2 = wav2vec2
        self.wav2vec2.model.aux = None  # get rid of the output linear layer in wav2vec2 model

        self.olayer = nn.Sequential(
            nn.Linear(self.wav2vec2_odim, self.wav2vec2_odim),
            nn.LeakyReLU(),
            nn.Linear(self.wav2vec2_odim, self.wav2vec2_odim),
            nn.LeakyReLU(),
            nn.Linear(self.wav2vec2_odim, self.odim),
        )
        self.freeze_and_init()

    def freeze_and_init(self):
        # By default, only fine-tune the encoder part of the wav2vec2 model
        for para in self.wav2vec2.model.feature_extractor.parameters():
            para.requires_grad = False
        for para in self.wav2vec2.model.encoder.parameters():
            para.requires_grad = False
        logging.info(
                f"RANK {self.rank}: Fine-tuning the last {self.finetune_last_n_layers} "
                f"transformer layers of the wav2vec2 encoder.")
        if self.finetune_last_n_layers > 0:
            assert self.finetune_last_n_layers <= len(self.wav2vec2.model.encoder.transformer.layers), \
                (f"RANK {self.rank}: finetune_last_n_layers should be less than or equal to "
                 f"the number of transformer layers in the wav2vec2 encoder"
                 f" but got {self.finetune_last_n_layers} and "
                 f"{len(self.wav2vec2.model.encoder.transformer.layers)} respectively.")
            self.wav2vec2.model.encoder.transformer.layers[-self.finetune_last_n_layers:].requires_grad_(True)

    def forward(self, x, xlens):
        x, xlens = self.wav2vec2(x, xlens)
        x = self.olayer(x)
        x = F.log_softmax(x, dim=-1)
        return x, xlens
