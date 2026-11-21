"""
Cepstral mean and variance normalisation, in Python.

The first stage of the Kaldi removal, chosen because it is the simplest: two
binaries, `compute-cmvn-stats` and `apply-cmvn`, doing arithmetic that fits
on a page. If the pattern of "replace a stage, compare against the captured
reference at that layer's tolerance" does not work here, it will not work
for the harder stages either.

Kaldi's statistics layout is reproduced exactly, because the files have to
stay interchangeable while the migration is half done:

    stats[0, 0:D]  sum of each dimension over the frames
    stats[0, D]    the number of frames
    stats[1, 0:D]  sum of squares of each dimension
    stats[1, D]    unused, zero

So a statistics matrix is `(2, D + 1)` for `D`-dimensional features, and the
frame count lives in the first row's last column rather than anywhere
sensible. That layout is verifiable against the captured reference: its
`stats[0, D]` is 18996, which is exactly the sum of the per-utterance frame
counts recorded beside it.

Normalisation is per speaker, keyed through `utt2spk`, which is what makes
it worth computing statistics separately from applying them: every utterance
of a speaker is normalised by that speaker's statistics, not its own.

Functions
---------
compute_stats
    Accumulate the statistics for one speaker's features.
accumulate
    Combine statistics, for a speaker whose features arrive in pieces.
mean_and_variance
    The normaliser a statistics matrix implies.
apply
    Normalise features with a statistics matrix.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

from typing import Iterable, Tuple

__all__ = [
    "VAR_FLOOR",
    "compute_stats",
    "accumulate",
    "mean_and_variance",
    "apply",
]

#: Floor on a variance before it is inverted. A dimension that never varies
#: -- a constant feature, or a speaker with one frame -- would otherwise
#: divide by zero. Kaldi floors it too; the exact constant it uses has not
#: been confirmed against this implementation, because no dimension in the
#: yesno reference comes near it (the smallest variance there is of order
#: one). Anything that relies on the floor is therefore not yet verified
#: against Kaldi, and this is the place to look if a corpus with a constant
#: dimension ever disagrees.
VAR_FLOOR = 1.0e-20


def _as_matrix(features):
    """
    A two-dimensional float64 view of one utterance's features.

    float64 because the statistics are sums over tens of thousands of
    frames: accumulating those in float32 loses precision where it is
    cheapest to keep.
    """
    import numpy as np

    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"expected features shaped (frames, dim), got {matrix.shape}")
    return matrix


def compute_stats(features: Iterable) -> "object":
    """
    Accumulate CMVN statistics over one or more utterances.

    Arguments
    ---------
    features: Iterable
        The feature matrices to accumulate, each `(frames, dim)`. A single
        matrix is accepted as well as a sequence of them.

    Returns
    -------
    numpy.ndarray
        The statistics, shaped `(2, dim + 1)` in Kaldi's layout.

    Raises
    ------
    ValueError
        If nothing was given, if the matrices disagree about the dimension,
        or if any is not two-dimensional.
    """
    import numpy as np

    matrices = [features] if hasattr(features, "ndim") else list(features)
    if not matrices:
        raise ValueError("no features to accumulate statistics over")

    matrices = [_as_matrix(matrix) for matrix in matrices]
    dims = {matrix.shape[1] for matrix in matrices}
    if len(dims) != 1:
        raise ValueError(f"features disagree about the dimension: {sorted(dims)}")
    dim = dims.pop()

    stats = np.zeros((2, dim + 1), dtype=np.float64)
    for matrix in matrices:
        stats[0, :dim] += matrix.sum(axis=0)
        stats[1, :dim] += (matrix**2).sum(axis=0)
        stats[0, dim] += matrix.shape[0]
    return stats


def accumulate(parts: Iterable) -> "object":
    """
    Add statistics matrices together.

    Used where a speaker's utterances are processed in pieces: the layout is
    additive, so combining is addition, with the count in `stats[0, D]`
    adding up with everything else.

    Arguments
    ---------
    parts: Iterable
        Statistics matrices, all of the same shape.

    Returns
    -------
    numpy.ndarray
        Their sum.

    Raises
    ------
    ValueError
        If nothing was given or the shapes disagree.
    """
    import numpy as np

    matrices = [np.asarray(part, dtype=np.float64) for part in parts]
    if not matrices:
        raise ValueError("no statistics to accumulate")
    shapes = {matrix.shape for matrix in matrices}
    if len(shapes) != 1:
        raise ValueError(f"statistics shapes disagree: {sorted(shapes)}")
    total = np.zeros(matrices[0].shape, dtype=np.float64)
    for matrix in matrices:
        total += matrix
    return total


def mean_and_variance(stats, var_floor: float = VAR_FLOOR) -> Tuple:
    """
    The mean and variance a statistics matrix implies.

    Arguments
    ---------
    stats: numpy.ndarray
        Statistics shaped `(2, dim + 1)`.
    var_floor: float
        Lower bound applied to each variance before it is used.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        The per-dimension mean and the floored per-dimension variance.

    Raises
    ------
    ValueError
        If the statistics are the wrong shape, or record no frames -- which
        would otherwise divide by zero and produce a normaliser of NaNs that
        silently ruins every feature it touches.
    """
    import numpy as np

    stats = np.asarray(stats, dtype=np.float64)
    if stats.ndim != 2 or stats.shape[0] != 2 or stats.shape[1] < 2:
        raise ValueError(f"expected statistics shaped (2, dim + 1), got {stats.shape}")

    dim = stats.shape[1] - 1
    count = stats[0, dim]
    if count <= 0:
        raise ValueError(
            f"the statistics record {count} frames, so there is no mean to "
            f"normalise by"
        )

    mean = stats[0, :dim] / count
    variance = stats[1, :dim] / count - mean**2
    return mean, np.maximum(variance, var_floor)


def apply(
    features,
    stats,
    *,
    norm_means: bool = True,
    norm_vars: bool = True,
    var_floor: float = VAR_FLOOR,
) -> "object":
    """
    Normalise features with a statistics matrix.

    Arguments
    ---------
    features: numpy.ndarray
        One utterance, shaped `(frames, dim)`.
    stats: numpy.ndarray
        The speaker's statistics, shaped `(2, dim + 1)`.
    norm_means: bool
        Whether to subtract the mean. True matches Kaldi's default.
    norm_vars: bool
        Whether to divide by the standard deviation. True is what
        `run/apply_cmvn.sh` passes; Kaldi's own default is False, so this
        argument is named rather than positional to stop the two being
        confused.
    var_floor: float
        Passed through to `mean_and_variance`.

    Returns
    -------
    numpy.ndarray
        The normalised features, float32, which is what Kaldi writes and so
        what the rest of the pipeline expects to read.

    Raises
    ------
    ValueError
        If the features and statistics disagree about the dimension.
    """
    import numpy as np

    matrix = _as_matrix(features)
    mean, variance = mean_and_variance(stats, var_floor=var_floor)

    if matrix.shape[1] != mean.shape[0]:
        raise ValueError(
            f"features have dimension {matrix.shape[1]} but the statistics "
            f"describe {mean.shape[0]}"
        )

    out = matrix
    if norm_means:
        out = out - mean
    if norm_vars:
        out = out / np.sqrt(variance)
    # float32 on the way out: Kaldi's archives hold float32, so returning
    # float64 would make a byte comparison against them fail for a reason
    # that has nothing to do with the arithmetic.
    return out.astype(np.float32)
