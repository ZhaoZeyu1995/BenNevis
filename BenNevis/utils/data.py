"""
Data-loading related utility functions.
Authors:
    Zeyu Zhao (The University of Edinburgh) 2024
"""
from typing import List, Optional, Callable, Dict, Any, Union


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
