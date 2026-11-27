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
    - k2/topo.yaml
    - k2/L_inv.pt
The `L_inv.pt` file is created after loading `L.fst` and is used to
calculate loss values with k2.

Note that there are some redundant files in the language directory, but
they make it easier to read the language directory, load and process the FSTs.

The topology FSA is built from `k2/topo.yaml`, the declarative specification
that `prepare_topo.sh` copies in, so the training path needs no OpenFST.
`k2/T.fst` is kept because the decoding graph is still compiled with Kaldi,
and a language directory prepared before the specification was copied in is
still read through `fstprint` as a fallback. Loading the lexicon does still
need openfst on PATH when `k2/L_inv.pt` has not been generated yet, because
`L.fst` comes from Kaldi.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""

import os
import logging
import warnings
import k2
import torch
from pathlib import Path
from typing import List
from BenNevis.topos import build as topo_build
from BenNevis.topos import spec as topo_spec
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
            - k2/topo.yaml (created by BenNevis) the topology specification
            - k2/L_inv.pt (created after loading L.fst)
        A language directory is usually created by Kaldi scripts and BenNevis's scripts.
    load_topo: bool
        Whether or not to build the topology FSA, from lang_dir/k2/topo.yaml
        where it exists and from lang_dir/k2/T.fst otherwise. Default is
        False.
    load_lexicon: bool
        Whether or not load lang_dir/k2/L_inv.pt. Generate one from lang_dir/L.fst if there is not. Default is False.
    """

    def __init__(
        self,
        lang_dir: str,
        load_topo: bool = False,
        load_lexicon: bool = False,
    ):
        self._lang_dir = lang_dir
        # with <eps> and disambig
        self.words = read_keys(os.path.join(lang_dir, "words.txt"))
        # with <eps> and disambig
        self.phones = read_keys(os.path.join(lang_dir, "phones.txt"))
        # no <eps> no disambig, to match the NN outputs
        self.tokens = read_keys(os.path.join(lang_dir, "k2", "tokens.txt"))

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

        self.lexicon = read_dict(
            os.path.join(self._lang_dir, "phones", "align_lexicon.int"),
            value_start=2,
            key_mapping=int,
            mapping=lambda x: list(map(int, x.split())),
        )

    def load_topo(self):
        """
        Build self.topo, the topology FSA used for training.

        Preferred route: expand {lang_dir}/k2/topo.yaml, the topology
        specification that `prepare_topo.sh` copies into the language
        directory, against {lang_dir}/k2/phones.txt and
        {lang_dir}/k2/tokens.txt. This is pure Python and needs no OpenFST.

        Fallback route: a language directory prepared before the
        specification was copied in has only the compiled
        {lang_dir}/k2/T.fst, which can be read back only through `fstprint`.
        Such a directory keeps working, with a warning, so that existing
        experiments do not have to be re-prepared.

        The two routes produce the same machine. The compiled FST was built
        from the same specification, and the `fstcompile`/`fstarcsort`/
        `fstprint` round trip only renumbered the labels through the same
        symbol tables and reordered arcs within each state -- and `k2.compose`
        arc-sorts its first argument itself, so that ordering never reached
        the loss.

        self.topo: k2.Fsa
            The topology FSA.
        """
        spec_path = os.path.join(self._lang_dir, "k2", "topo.yaml")
        if os.path.exists(spec_path):
            self.topo = self._topo_from_spec(spec_path)
            return

        fst_path = os.path.join(self._lang_dir, "k2", "T.fst")
        assert os.path.exists(fst_path), (
            f"neither {spec_path} nor {fst_path} exists, so there is nothing "
            f"to build the topology from; re-run prepare_topo.sh"
        )
        warnings.warn(
            f"{spec_path} does not exist, so the topology is being read back "
            f"from the compiled {fst_path} with `fstprint`. That needs OpenFST "
            f"on PATH and a POSIX shell. Re-run prepare_topo.sh to have the "
            f"specification copied into the language directory, after which "
            f"the topology is built in Python.",
            stacklevel=2,
        )
        logging.info(f"Loading and processing topo from {fst_path}")
        cmd = (
            f"""fstprint {fst_path} | """
            f"""awk -F '\t' '{{if (NF==4) {{print $0 FS "0.0"; }} else {{print $0;}}}}'"""
        )
        openfst_txt = os.popen(cmd).read()
        self.topo = k2.Fsa.from_openfst(openfst_txt, acceptor=False)
        logging.info(f"Finished loading topo from {fst_path}")

    def _topo_from_spec(self, spec_path: str) -> k2.Fsa:
        """
        Expand a topology specification into the training FSA.

        The symbol tables are read from the language directory rather than
        derived from the specification again. `k2/tokens.txt` fixes the neural
        network's output indices, and a checkpoint trained against it stays
        valid only while those indices do not move, so the table on disc is
        the authority.

        Arguments
        ---------
        spec_path: str
            The copy of the specification in the language directory.

        Returns
        -------
        k2.Fsa
            The topology FSA, with input labels in `k2/tokens.txt` and output
            labels in `k2/phones.txt`.
        """
        logging.info(f"Building topo from {spec_path}")
        topology = topo_spec.load_spec(Path(spec_path))
        k2_dir = Path(self._lang_dir) / "k2"
        openfst_txt = topo_build.build_openfst_int_text(
            topology,
            "k2",
            topo_build.read_phone_table(k2_dir / "phones.txt"),
            isymbols=topo_build.read_symbol_table(k2_dir / "tokens.txt"),
            osymbols=topo_build.read_symbol_table(k2_dir / "phones.txt"),
        )
        topo = k2.Fsa.from_openfst(openfst_txt, acceptor=False)
        logging.info(
            f"Built topo {topology.name!r} with {topo.shape[0]} states and "
            f"{topo.num_arcs} arcs"
        )
        return topo

    def load_lexicon(self):
        """
        The lexicon FSA, and its inverse, which training composes against.

        Three sources, in order, and the middle one is why the training
        path no longer needs OpenFST:

        1. `k2/L_inv.pt`, the cache this writes on first use;
        2. **`k2/L.fst.txt`**, written by `prepare_topo.sh --lexicon` from
           the dictionary, in Python;
        3. `fstprint {lang_dir}/L.fst`, which needs OpenFST on PATH and a
           POSIX shell, and warns.

        The third was the only route until `k2/L.fst.txt` existed, and
        `compile_training_graph` genuinely needs what it returns -- so a
        first training run required OpenFST and Kaldi's compiled `L.fst`
        however much else had been ported. It is kept for a language
        directory prepared before this, which is the same arrangement
        `load_topo` has for `k2/topo.yaml`.

        **Rebuilding rather than converting is safe because it is
        byte-exact**: `build_lexicon_fst_text` reproduces Kaldi's `L.fst`
        for both corpora, and `tests/reference/test_corpora.py` gates that
        by digest. The text is therefore Kaldi's lexicon FST, computed in
        Python.

        self.L: k2.Fsa
            The lexicon FSA.
        self.L_inv: k2.Fsa
            The inverse of the lexicon FSA.
        """
        L_inv_fst = os.path.join(self._lang_dir, "k2", "L_inv.pt")
        if os.path.exists(L_inv_fst):
            logging.info(f"Loading L_inv from {self._lang_dir}/k2/L_inv.pt")
            self.L_inv = k2.arc_sort(k2.Fsa.from_dict(torch.load(L_inv_fst)))
            self.L = k2.arc_sort(self.L_inv.invert())
            return

        text_path = os.path.join(self._lang_dir, "k2", "L.fst.txt")
        if os.path.exists(text_path):
            logging.info(f"Loading the lexicon from {text_path}")
            with open(text_path, encoding="utf-8") as handle:
                openfst_txt = handle.read()
        else:
            logging.warning(
                f"{self._lang_dir}/k2/L.fst.txt is missing, so the lexicon "
                f"is being read from the compiled {self._lang_dir}/L.fst "
                f"with `fstprint`. That needs OpenFST on PATH and a POSIX "
                f"shell, and is the last thing on the training path that "
                f"does. Re-run prepare_topo.sh with --lexicon to have the "
                f"directory carry the text instead."
            )
            cmd = (
                f"""fstprint {self._lang_dir}/L.fst | """
                f"""awk -F '\t' '{{if (NF==4) {{print $0 FS "0.0"; }} else {{print $0;}}}}'"""
            )
            openfst_txt = os.popen(cmd).read()
            if not openfst_txt.strip():
                raise RuntimeError(
                    f"`fstprint {self._lang_dir}/L.fst` produced nothing. "
                    f"Either OpenFST is not on PATH or the FST is missing; "
                    f"an empty lexicon would train against an empty graph "
                    f"rather than fail here."
                )
        # `from_openfst` wants five fields an arc, and both `fstprint` and
        # the Python builder omit a weight that is the semiring One -- which
        # for `--sil_prob 0.0` is every weight in the file.
        # `load_openfst_text` fills those in and arc-sorts; it also handles
        # the one-field final-state line, which the `awk` this replaces did
        # not.
        from BenNevis.decode.graph import load_openfst_text

        self.L = load_openfst_text(openfst_txt, acceptor=False)
        self.L_inv = k2.arc_sort(self.L.invert())
        torch.save(self.L_inv.as_dict(), L_inv_fst)

    def compile_training_graph(
        self,
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
        fsa = k2.intersect(
            self.L_inv, word_fsa_with_self_loop, treat_epsilons_specially=False
        )

        trans_fsa = k2.arc_sort(fsa.invert())  # trans_fsa: phones -> words
        trans_fsa_with_self_loop = k2.arc_sort(
            k2.remove_epsilon_and_add_self_loops(trans_fsa)
        )

        training_graph = k2.compose(
            self.topo, trans_fsa_with_self_loop, treat_epsilons_specially=False
        )

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
