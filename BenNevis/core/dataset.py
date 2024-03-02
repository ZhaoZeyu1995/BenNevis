"""
This file contains the Dataset class which is responsible for reading the Kaldi data dir and returning the samples.
The Dataset class is a subclass of torch.utils.data.Dataset and it is used by the DataLoader class to load the data.
There is also CollateFunc class which is used by the DataLoader class to collate the samples into a batch.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""
import torch
import os
import logging
import kaldiio
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from typing import Callable, Dict, Any, List, Optional
from BenNevis.utils.data import read_keys, read_dict
from BenNevis.core.lang import Lang


class Dataset(torch.utils.data.Dataset):
    """
    Read a Kaldi data dir, which usually contains
        - wav.scp
        - text
        - spk2utt
        - utt2spk
        - utt2dur
        - utt2num_frames
        - segments (optional)
        - feats.cmvn.scp (optional)

    Arguments
    ---------
    data_dir: str
        Path to data dir
    lang_dir: str
        Path to lang dir
    ratio_th: Optional[float]
        Threshold for filtering utterances based on the ratio of num_frames to num_phones
        The utterances with ratio less than ratio_th will be removed, by default None.
    min_duration: Optional[float]
        Minimum duration of utterances in seconds, by default None.
    max_duration: Optional[float]
        Maximum duration of utterances in seconds, by default None.
    sort: Optional[str]
        Sort the utterances by duration in "ascending" or "descending" order, by default None.
    ctc_target: Optional[bool]
        If True, the target will be a list of token indices for CTC loss, by default False.
        This is useful for training with torch.nn.CTCLoss.
    load_wav: Optional[bool]
        If True, load the wav files, by default False.
    load_feats: Optional[bool]
        If True, load the features, by default False.
    resample_rate: Optional[int]
        The resample rate for the wav files, by default 16000.
    transforms: Optional[Callable]
        A function that takes a sample as input and returns the transformed sample.

    The format of each sample can be found in the __getitem__ method.
    """

    def __init__(self,
                 data_dir: str,
                 lang_dir: str,
                 ratio_th: Optional[float] = None,
                 min_duration: Optional[float] = None,
                 max_duration: Optional[float] = None,
                 sort: Optional[str] = None,
                 ctc_target: Optional[bool] = False,
                 load_wav: Optional[bool] = False,
                 load_feats: Optional[bool] = False,
                 resample_rate: Optional[int] = 16000,
                 transforms: Optional[Callable] = None,
                 ):

        self.data_dir = data_dir
        self.lang_dir = lang_dir
        self.lang = Lang(lang_dir)
        self.ratio_th = ratio_th
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.sort = sort
        self.ctc_target = ctc_target
        self.load_wav = load_wav
        self.load_feats = load_feats
        self.resample_rate = resample_rate
        self.transforms = transforms
        # Check if there is a segment file
        self.segments = None
        if os.path.exists(os.path.join(self.data_dir, 'segments')):
            self.segments = os.path.join(self.data_dir, 'segments')

        self.wav_scp = os.path.join(self.data_dir, 'wav.scp')

        if self.segments:
            self.uttids = read_keys(self.segments)
            logging.info(f"A segments file is found. Loading utterances according to {self.segments}")
        else:
            self.uttids = read_keys(self.wav_scp)
            logging.info(f"No segments file is found. Loading utterances according to {self.wav_scp}")

        original_num_utt = len(self.uttids)

        if load_wav:
            if self.segments:
                self.utt2wav = kaldiio.load_scp(self.wav_scp, segments=self.segments)
            else:
                self.utt2wav = kaldiio.load_scp(self.wav_scp)

        self.utt2spk = read_dict(os.path.join(self.data_dir, 'utt2spk'))
        self.utt2text = read_dict(os.path.join(self.data_dir, 'text'))

        self.utt2dur = read_dict(os.path.join(
            self.data_dir, 'utt2dur'), mapping=float)
        self.utt2num_frames = read_dict(os.path.join(
            self.data_dir, 'utt2num_frames'), mapping=int)

        if load_feats:
            self.dump_feats = os.path.join(self.data_dir, 'feats.cmvn.scp')
            self.utt2feats = kaldiio.load_scp(self.dump_feats)

        if self.min_duration is not None:
            num_short_utt = len([uttid for uttid in self.uttids if self.utt2dur[uttid] < self.min_duration])
            logging.info(f"Filtering utterances with less than {self.min_duration} seconds, {num_short_utt} utterances are removed")
            self.uttids = [uttid for uttid in self.uttids if self.utt2dur[uttid] >= self.min_duration]

        if self.max_duration is not None:
            num_long_utt = len([uttid for uttid in self.uttids if self.utt2dur[uttid] > self.max_duration])
            logging.info(f"Filtering utterances with more than {self.max_duration} seconds, {num_long_utt} utterances are removed")
            self.uttids = [uttid for uttid in self.uttids if self.utt2dur[uttid] <= self.max_duration]

        # Check if the num_frame is enough
        # It is 8.5 when a common experiment setting with a subsampling facotr of 4 and the 2-state topology.
        # This leads to some loss of data by approximately 4% of the training data in WSJ. with BPE 100.
        # However, we should definitely keep ratio_th as None during evaluation.
        if self.ratio_th is not None:
            num_fast_utt = len([uttid for uttid in self.uttids if self.check_ratio(uttid) < self.ratio_th])
            self.uttids = [uttid for uttid in self.uttids if self.check_ratio(uttid) >= self.ratio_th]
            logging.info(f"Filtering utterances with ratio (num_frames (stride of 10ms) / num_phones) less than {self.ratio_th}, {num_fast_utt} utterances are removed.")

        logging.info(f"Original number of utterances: {original_num_utt}. Current number of utterances: {len(self.uttids)} after filtering")

        if self.sort is not None:
            assert self.sort in ["ascending", "descending"], "sort must be ascending or descending"
            if self.sort == "ascending":
                logging.info("Sorting utterances by ascending order of duration")
                self.uttids = sorted(self.uttids, key=lambda x: self.utt2dur[x])
            else:
                logging.info("Sorting utterances by descending order of duration")
                self.uttids = sorted(self.uttids, key=lambda x: self.utt2dur[x], reverse=True)

    def __len__(self):
        return len(self.uttids)

    def check_ratio(self, uttid: str) -> float:
        """
        Check the ratio of the number of frames to the number of phones

        Arguments
        ---------
        uttid: str
            The utterance id

        Returns
        -------
        ratio: float
            The ratio of the number of frames to the number of phones
        """
        words = self.utt2text[uttid].split(' ')
        word_ids = [self.lang.word2idx[word] if word in self.lang.word2idx else self.lang.word2idx["<UNK>"] for word in words]
        phone_ids = self.lang.wids2pids([word_ids])[0]
        num_frame = self.utt2num_frames[uttid]
        ratio = num_frame / len(phone_ids)
        return ratio

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Return a data sample from the dataset

        Arguments
        ---------
        idx: int
            The index of the data sample

        Returns
        -------
        sample: Dict[str, Any]
            A dictionary containing the data sample
            It has the following key-value pairs:
                - name: str (utterance id)
                - spk: str (speaker id)
                - dur: float (duration of the utterance)
                - num_frame: int (number of frames in the utterance with a stride of 10ms)
                - text: str (transcription of the utterance)
                - word_ids: List[int] (word ids of the transcription)
                - target: torch.Tensor (phone-level sequence of the transcription)
                - target_length: int (length of the target phone-level sequence)
                if self.load_feats:
                    - feats: torch.Tensor (feature matrix of the utterance)
                    - feats_len: int (length of the feature matrix)
                if self.load_wav:
                    - wav: torch.Tensor (waveform of the utterance)
                    - wav_len: int (length of the waveform)
                if self.ctc_target:
                    - target_ctc: torch.Tensor (CTC target (token-level) of the transcription)
        """
        uttid = self.uttids[idx]
        spk = self.utt2spk[uttid]
        dur = self.utt2dur[uttid]
        num_frame = self.utt2num_frames[uttid]
        text = self.utt2text[uttid]
        words = text.split(' ')
        word_ids = [self.lang.word2idx[word] if word in self.lang.word2idx else self.lang.word2idx["<UNK>"] for word in words]
        pids = self.lang.wids2pids([word_ids])[0]

        tids = []  # for ctc only
        if self.ctc_target:
            for pid in pids:
                assert self.lang.idx2phone[pid] in self.lang.token2idx, \
                        "Cannot find the token %s from the token list, please make sure you are using CTC topo" % (
                    self.lang.idx2phone[pid])
                tids.append(self.lang.token2idx[self.lang.idx2phone[pid]])

        sample = {
            'target_length': len(pids),
            'target': torch.tensor(pids, dtype=torch.int64),
            'name': uttid,
            'spk': spk,
            'dur': dur,
            'num_frame': num_frame,
            'word_ids': word_ids,
            'text': text
        }

        if self.load_wav:
            rate, wav = self.utt2wav[uttid]
            wav = torch.tensor(wav, dtype=torch.float32)
            if rate != self.resample_rate:
                wav = torchaudio.functional.resample(wav, rate, self.resample_rate)

            wav = (wav - wav.mean()) / (torch.sqrt(wav.var()) + 1e-5)  # normalize
            sample['wav'] = wav
            sample['wav_len'] = len(wav)

        if self.load_feats:
            feats = torch.tensor(self.utt2feats[uttid], dtype=torch.float32)
            sample['feats'] = feats
            sample['feats_len'] = feats.shape[0]

        if self.ctc_target:
            sample['target_ctc'] = torch.tensor(tids, dtype=torch.int64)

        if self.transforms:
            return self.transforms(sample)
        else:
            return sample


class CollateFunc:
    """
    A callable class to collate a list of data samples into a batch

    Arguments
    ---------
    load_wav: bool
        Whether to load the waveform, by default is False.
    load_feats: bool
        Whether to load the feature matrix, by default is False.
    ctc_target: bool
        Whether to load the CTC target, by default is False.
    sort: str
        The order to sort the utterances, either "ascending" or "descending", by default is "descending".
    """
    def __init__(self,
                 load_wav: bool = False,
                 load_feats: bool = False,
                 ctc_target: bool = False,
                 sort: str = "descending",
                 ):
        assert sort in ["ascending", "descending", None], "sort must be either 'ascending' or 'descending'"
        self.sort = sort
        self.load_wav = load_wav
        self.load_feats = load_feats
        self.ctc_target = ctc_target

    def __call__(self, list_of_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a list of data samples into a batch.
        If sort is set to "ascending" or "descending", the utterances will be sorted by their durations.

        Arguments
        ---------
        list_of_samples: List[Dict[str, Any]]
            A list of data samples

        Returns
        -------
        batch: Dict[str, Any]
            A dictionary containing the batch
            It has the following key-value pairs:
                - names: List[str] (utterance ids)
                - spks: List[str] (speaker ids)
                - durs: List[float] (durations of the utterances)
                - num_frames: List[int] (number of frames in the utterances)
                - texts: List[str] (transcriptions of the utterances)
                - word_ids: List[List[int]] (word ids of the transcriptions)
                - targets: torch.Tensor (phone-level sequences of the transcriptions)
                - target_lengths: torch.Tensor (lengths of the target phone-level sequences)
                - batch_size: int (the number of utterances in the batch)
                if load_feats:
                    - feats: torch.Tensor (feature matrices of the utterances)
                    - feats_lens: List[int] (lengths of the feature matrices)
                if load_wav:
                    - wavs: torch.Tensor (waveforms of the utterances)
                    - wav_lens: List[int] (lengths of the waveforms)
                if ctc_target:
                    - targets_ctc: torch.Tensor (CTC targets (token-level) of the transcriptions)
        """
        # sort the utterances by their durations
        if self.sort is not None:
            list_of_samples = sorted(list_of_samples, key=lambda x: x["dur"], reverse=self.sort == "descending")

        batch_targets = [sample['target'] for sample in list_of_samples]
        batch_target_lengths = [sample['target_length'] for sample in list_of_samples]
        batch_names = [sample['name'] for sample in list_of_samples]
        batch_spks = [sample['spk'] for sample in list_of_samples]
        batch_durs = [sample['dur'] for sample in list_of_samples]
        batch_num_frames = [sample['num_frame'] for sample in list_of_samples]
        batch_texts = [sample['text'] for sample in list_of_samples]
        batch_word_ids = [sample['word_ids'] for sample in list_of_samples]

        batch = {
            'targets': pad_sequence(batch_targets, batch_first=True),
            'target_lengths': torch.tensor(batch_target_lengths, dtype=torch.int32),
            'names': batch_names,
            'spks': batch_spks,
            'durs': batch_durs,
            'num_frames': batch_num_frames,
            'texts': batch_texts,
            'word_ids': batch_word_ids,
            'batch_size': len(list_of_samples),
        }

        if self.load_wav:
            assert all('wav' in sample for sample in list_of_samples), "wav is not available in the samples"
            assert all('wav_len' in sample for sample in list_of_samples), "wav_len is not available in the samples"
            batch_wavs = [sample['wav'] for sample in list_of_samples]
            batch_wav_lens = [sample['wav_len'] for sample in list_of_samples]
            batch['wav'] = pad_sequence(batch_wavs, batch_first=True)
            batch['wav_len'] = torch.tensor(batch_wav_lens, dtype=torch.int32)

        if self.load_feats:
            assert all('feats' in sample for sample in list_of_samples), "feats is not available in the samples"
            assert all('feats_len' in sample for sample in list_of_samples), "feats_len is not available in the samples"
            batch_feats = [sample['feats'] for sample in list_of_samples]
            batch_feats_lens = [sample['feats_len'] for sample in list_of_samples]
            batch['feats'] = pad_sequence(batch_feats, batch_first=True)
            batch['feats_len'] = torch.tensor(batch_feats_lens, dtype=torch.int32)

        if self.ctc_target:
            assert all('target_ctc' in sample for sample in list_of_samples), "target_ctc is not available in the samples"
            batch_targets_ctc = [sample['target_ctc'] for sample in list_of_samples]
            batch['target_ctc'] = pad_sequence(batch_targets_ctc, batch_first=True)

        return batch
