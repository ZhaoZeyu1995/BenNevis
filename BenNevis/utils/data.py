"""
Data-loading related utility functions.
Authors:
    Zeyu Zhao (The University of Edinburgh) 2024
"""

from typing import Any, Callable, Dict, List, Optional, Union


def read_dict(
    path: str,
    value_start: int = 1,
    key_mapping: Optional[Callable] = None,
    mapping: Optional[Callable] = None,
) -> Dict[str, Union[str, Any]]:
    """
    Read a dict file, e.g, wav.scp, text, with the following pattern in each line:
        <key> <value1> <value2> ...
    Return a dictionary mapping from <key> and all the <value>s
    By default, <value>s are concatenated into a string, but can be transformed by mapping (None, by default).
    The mapping function should take a string as input and return the desired type.

    Arguments
    ---------
    path: str
        Path to the file
    value_start: int
        The index of the first value, by default 1.
        You can set it to a larger number to skip some values.
    key_mapping: Optional[Callable]
        A function to transform a key string to the desired type, by default None
    mapping: Optional[Callable]
        A function to transform a string to the desired type, by default None
    """
    temp = dict()
    with open(path) as f:
        for line in f:
            lc = line.strip().split()
            key = lc[0] if key_mapping is None else key_mapping(lc[0])
            assert value_start < len(lc), (
                "Expect at least %d elements per line but got %d"
                % (value_start + 1, len(lc))
                + "Line content: "
                + line
            )
            temp[key] = (
                " ".join(lc[value_start:])
                if mapping is None
                else mapping(" ".join(lc[value_start:]))
            )
    return temp


def read_keys(
    path: str,
) -> List[str]:
    """
    Read a file with the pattern '<key> <value>'
    Return the keys in a list

    Arguments
    ---------
    path: str
        Path to the file

    Returns
    -------
    keys: list
        List of keys
    """
    with open(path) as f:
        keys = []
        for line in f:
            key = line.strip().split()[0]
            keys.append(key)
    return keys


#: The feature archive `Dataset` reads when `load_feats` is set. Named here
#: because `feature_dim` has to agree with it or it would measure a
#: different file from the one training consumes.
FEATURES_SCP = "feats.cmvn.scp"


def feature_dim(data_dir: str, scp: str = FEATURES_SCP) -> int:
    """
    How many coefficients a data directory's features have.

    Exists so that a model's `idim` can be **derived rather than written
    out**. `odim` already is -- `run/train.py` fills it in from the
    language directory, so a topology change cannot desynchronise it --
    while `idim` was a number in each recipe's `conf/model/*.yaml`, and
    three recipes had it wrong: `egs/librispeech/asr{1,2,3}` said 83 while
    running `steps/make_fbank.sh`, which writes 80. That is a shape
    mismatch at the first forward pass, and nothing caught it because
    those recipes have never run in CI.

    Reads one utterance. `kaldiio.load_scp` is lazy, so this is a seek and
    a single matrix rather than a pass over the archive.

    Arguments
    ---------
    data_dir: str
        A Kaldi data directory.
    scp: str
        Which archive to measure, defaulting to the one `Dataset` reads.

    Returns
    -------
    int
        The number of coefficients per frame.

    Raises
    ------
    FileNotFoundError
        If the archive is not there, naming the stage that writes it --
        this is what a run that skipped feature extraction hits, and the
        message is the difference between a five-second fix and reading a
        traceback out of `kaldiio`.
    ValueError
        If the archive is empty, or an utterance is not a matrix. An empty
        archive would otherwise give a model with a zero-wide input layer
        and a run that trains on nothing.
    """
    import os

    import kaldiio

    path = os.path.join(data_dir, scp)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} does not exist, so the feature dimension cannot be "
            f"measured. It is written by the feature stage -- "
            f"steps/make_fbank.sh followed by run/apply_cmvn.sh -- so run "
            f"that stage, or set the model's idim explicitly."
        )
    for uttid, array in kaldiio.load_scp_sequential(path):
        if getattr(array, "ndim", 0) != 2:
            raise ValueError(
                f"{path}: {uttid} has shape "
                f"{getattr(array, 'shape', None)}, which is not a "
                f"(frames, coefficients) matrix"
            )
        return int(array.shape[1])
    raise ValueError(
        f"{path} is empty, so there is no feature dimension to measure. A "
        f"model built from it would have a zero-wide input layer."
    )
