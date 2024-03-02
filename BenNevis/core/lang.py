"""
This file contains the language specific functions and classes.
The basic purpose is to read a language directory and load several
FSTs so that we can use them to calculate loss values.

The structure of the language directory is quite similar to Kaldi's.
Namely, it contains the following files:
    - disambig.int (created by BenNevis)
    - L.fst
    - L_disambig.fst
    - oov.int
    - oov.txt
    - phones.txt
    - T.fst
    - tokens.txt (created by BenNevis)
    - tokens_disambig.txt (created by BenNevis)
    - topo
    - words.txt
and two directories:
    - k2
    - phones
where the `phones` directory is the same as Kaldi's `phones` directory,
and the `k2` directory contains the FSTs that are used to calculate
loss values with k2.
Usually the `k2` directory contains the following files:
    - k2/phones.txt
    - k2/tokens.txt
    - k2/T.fst
    - k2/L_inv.pt
The `L_inv.pt` file is created after loading `L.fst` and is used to
calculate loss values with k2.

Note that there are some redundant files in the language directory, but
they make it easier to read the language directory, load and process the FSTs.

To be able to run the code here, you will need to install openfst, and
make sure that path/to/openfst/bin is in your PATH.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""
import os
import logging
import k2
import torch
from typing import List, Union
from BenNevis.utils.data import read_keys, read_dict


class Lang(object):
    """
    This class is used to load the language directory and process the FSTs
    so that we can use them to calculate loss values.

    Arguments
    ---------
    lang_dir: str
        The language directory. It should contain the following files:
            - disambig.int (created by BenNevis) containing the indexes of the disambig symbols in tokens_disambig.txt
            - L.fst
            - L_disambig.fst
            - oov.int
            - oov.txt
            - phones.txt
            - T.fst
            - tokens.txt (created by BenNevis) containing the input symbol table for T.fst
            - tokens_disambig.txt (created by BenNevis)
            - topo
            - words.txt
        and two directories:
            - k2 (created by BenNevis)
            - phones
        where the `phones` directory is the same as Kaldi's `phones` directory,
        and the `k2` directory contains the FSTs that are used to calculate loss.
        Usually the `k2` directory contains the following files:
            - k2/phones.txt
            - k2/tokens.txt
            - k2/T.fst
            - k2/L_inv.pt (created after loading L.fst)
        A language directory is usually created by Kaldi scripts and BenNevis's scripts.
    load_topo: bool
        Whether or not load lang_dir/k2/T.fst. Default is False.
    load_lexicon: bool
        Whether or not load lang_dir/k2/L_inv.pt. Generate one from lang_dir/L.fst if there is not. Default is False.
    """
    def __init__(self,
                 lang_dir: str,
                 load_topo: bool = False,
                 load_lexicon: bool = False,
                 ):
        self._lang_dir = lang_dir
        # with <eps> and disambig
        self.words = read_keys(os.path.join(lang_dir, 'words.txt'))
        # with <eps> and disambig
        self.phones = read_keys(os.path.join(lang_dir, 'phones.txt'))
        # no <eps> no disambig, to match the NN outputs
        self.tokens = read_keys(os.path.join(lang_dir, 'k2', 'tokens.txt'))

        self.num_nn_output = len(self.tokens)

        self.word2idx = {word: idx for idx, word in enumerate(self.words)}
        self.idx2word = {idx: word for idx, word in enumerate(self.words)}

        self.phone2idx = {phone: idx for idx, phone in enumerate(self.phones)}
        self.idx2phone = {idx: phone for idx, phone in enumerate(self.phones)}

        self.token2idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx2token = {idx: token for idx, token in enumerate(self.tokens)}

        self.load_align_lexicon()

        if load_topo:
            self.load_topo()

        if load_lexicon:
            self.load_lexicon()

    def load_align_lexicon(self):
        """
        Load the lexicon mapping from {lang_dir}/phones/align_lexicon.int,
        where each line has the pattern:
            word_id word_id phone_id1 phone_id2 ...
        and the indexes can be accepted by self.idx2phone and self.idx2word.

        self.lexicon: Dict[int, List[int]]
            The lexicon mapping from word_id to phone_ids.
        """

        self.lexicon = read_dict(os.path.join(self._lang_dir, 'phones', 'align_lexicon.int'),
                                 value_start=2,
                                 key_mapping=int,
                                 mapping=lambda x: list(map(int, x.split())),
                                 )

    def load_topo(self):
        """
        Load {lang_dir}/k2/T.fst and process it to get the self.topo.

        self.topo: k2.Fsa
            The topology FSA.
        """
        assert os.path.exists(os.path.join(self._lang_dir, 'k2', 'T.fst')), \
            f'{self._lang_dir}/k2/T.fst does not exist'

        logging.info(f'Loading and processing topo from {self._lang_dir}/k2/T.fst')

        cmd = (
            f"""fstprint {self._lang_dir}/k2/T.fst | """
            f"""awk -F '\t' '{{if (NF==4) {{print $0 FS "0.0"; }} else {{print $0;}}}}'"""
        )
        openfst_txt = os.popen(cmd).read()
        self.topo = k2.Fsa.from_openfst(openfst_txt, acceptor=False)
        logging.info(f'Finished loading topo from {self._lang_dir}/k2/T.fst')

    def load_lexicon(self):
        """
        Load lang_dir/k2/L_inv.pt.
        Generate one based on {lang_dir}/L.fst if there is not one already.

        self.L: k2.Fsa
            The lexicon FSA.
        self.L_inv: k2.Fsa
            The inverse of the lexicon FSA.
        """
        L_inv_fst = os.path.join(self._lang_dir, 'k2', 'L_inv.pt')
        if os.path.exists(L_inv_fst):
            logging.info(f'Loading L_inv from {self._lang_dir}/k2/L_inv.pt')
            self.L_inv = k2.arc_sort(k2.Fsa.from_dict(torch.load(L_inv_fst)))
            self.L = k2.arc_sort(self.L_inv.invert())
        else:
            logging.info(
                f'Loading {self._lang_dir}/L.fst and generating {self._lang_dir}/k2/L_inv.pt.')

            cmd = (
                f"""fstprint {self._lang_dir}/L.fst | """
                f"""awk -F '\t' '{{if (NF==4) {{print $0 FS "0.0"; }} else {{print $0;}}}}'"""
            )
            openfst_txt = os.popen(cmd).read()
            self.L = k2.arc_sort(k2.Fsa.from_openfst(
                openfst_txt, acceptor=False))
            self.L_inv = k2.arc_sort(self.L.invert())
            torch.save(self.L_inv.as_dict(), L_inv_fst)

    def compile_training_graph(self,
                               word_ids_list: List[List[int]],
                               device: torch.device,
                               ) -> k2.Fsa:
        """
        Compile the training graph according to the word_ids_list.

        Arguments
        ---------
        word_ids_list: List[List[int]]
            A list of list of word_ids, where each list of word_ids is a transcription for an utterance.
        device: torch.device
            The device to put the training graph on.
            Usually it is the same device as neural network outputs.

        Returns
        -------
        training_graph: k2.Fsa
            The training graph for the whole batch of samples.
        """
        self.topo = self.topo.to(device)
        self.L_inv = self.L_inv.to(device)

        word_fsa = k2.linear_fsa(word_ids_list, device=device)
        word_fsa_with_self_loop = k2.add_epsilon_self_loops(word_fsa)
        fsa = k2.intersect(self.L_inv, word_fsa_with_self_loop,
                           treat_epsilons_specially=False)

        trans_fsa = k2.arc_sort(fsa.invert())  # trans_fsa: phones -> words
        trans_fsa_with_self_loop = k2.arc_sort(
            k2.remove_epsilon_and_add_self_loops(trans_fsa))

        training_graph = k2.compose(
            self.topo, trans_fsa_with_self_loop, treat_epsilons_specially=False)

        return training_graph

    def wids2pids(self, word_ids_list: List[List[int]]) -> List[List[int]]:
        """
        Transform a batch of word_id lists to a batch of phone_id lists
        according to the self.lexicon.

        Arguments
        ---------
        word_ids_list: List[List[int]]
            A list of list of word_ids, where each list of word_ids is a transcription for an utterance.

        Returns
        -------
        phone_ids_list: List[List[int]]
            A list of list of phone_ids, where each list of phone_ids represents a transcription for an utterance.
        """
        phone_ids_list = []
        for wids in word_ids_list:
            pids = []
            for wid in wids:
                pids.extend(self.lexicon[wid])
            phone_ids_list.append(pids)
        return phone_ids_list
