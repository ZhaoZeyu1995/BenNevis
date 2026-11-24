"""
Reference data: capturing what Kaldi produces, and comparing against it.

Kaldi is being removed, and every stage that leaves has to be shown to
reproduce what the stage it replaces produced. That needs Kaldi's output
recorded, not Kaldi available: recorded output can be committed, compared
against in the cheap test tiers, and still be there when Kaldi can no longer
be built at all -- which is not hypothetical, since recent Kaldi cannot fetch
the OpenFST version its own source requires.

So the reference is a directory of captured artefacts plus a manifest saying
what each one is and how closely a replacement has to match it. Two kinds:

- **Exact.** Symbol tables, FST text, integer lists, transcriptions. Recorded
  as bytes and compared as bytes, because there is no reason for a correct
  reimplementation to differ by even a character.
- **Approximate.** Feature matrices and CMVN statistics. Bit-exactness is not
  achievable here: FFT and accumulation order differ between
  implementations. Recorded as arrays with a stated tolerance, which for
  features is `rtol=1e-4` (see "Equivalence and tolerances" in CLAUDE.md).

Capturing every feature matrix in full would be 12MB of float32 for yesno and
buy nothing over a handful, so a few utterances are kept whole for a genuine
elementwise comparison and the rest are kept as per-utterance statistics.
Statistics rather than checksums: a checksum of floats cannot be compared at a
tolerance, so it would answer the wrong question.

The manifest records which image produced the data. A reference whose
provenance is unknown is not a reference.

Functions
---------
sha256_file, sha256_bytes
    Digests, for the exact artefacts.
array_stats
    The per-utterance summary kept for arrays not stored whole.
compare_exact
    Compare bytes against a recorded digest.
compare_array
    Compare an array against a recorded one, at a tolerance.
compare_stats
    Compare an array's statistics against recorded ones, at a tolerance.
Manifest
    Read, write and query a reference manifest.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "MANIFEST_NAME",
    "FEATURE_TOLERANCE",
    "EXACT_TOLERANCE",
    "Tolerance",
    "Difference",
    "Manifest",
    "sha256_bytes",
    "sha256_file",
    "array_stats",
    "compare_exact",
    "compare_array",
    "compare_stats",
]

MANIFEST_NAME = "manifest.json"

#: The statistics kept for an array that is not stored whole, in this order.
STAT_KEYS = ("frames", "dim", "mean", "std", "min", "max", "first_row_sum")


@dataclass(frozen=True)
class Tolerance:
    """
    How closely a replacement has to match the reference.

    Attributes
    ----------
    rtol: float
        Relative tolerance.
    atol: float
        Absolute tolerance, which is what decides agreement near zero.
    """

    rtol: float = 0.0
    atol: float = 0.0

    @property
    def is_exact(self) -> bool:
        """Whether this tolerance admits no difference at all."""
        return self.rtol == 0.0 and self.atol == 0.0


#: Features and anything derived from them. Bit-exactness is not achievable:
#: FFT and accumulation order differ. See CLAUDE.md.
FEATURE_TOLERANCE = Tolerance(rtol=1e-4, atol=1e-5)

#: Symbol tables, FST text, transcriptions. No reason to differ at all.
EXACT_TOLERANCE = Tolerance()


@dataclass(frozen=True)
class Difference:
    """
    One way in which something failed to match its reference.

    Attributes
    ----------
    name: str
        What was compared, as the manifest names it.
    detail: str
        A sentence saying what differed, with the numbers in it. Written to
        be read in a test failure, so it carries the measured values rather
        than only the verdict.
    """

    name: str
    detail: str

    def __str__(self) -> str:
        return f"{self.name}: {self.detail}"


def sha256_bytes(payload: bytes) -> str:
    """
    The SHA-256 of some bytes.

    Arguments
    ---------
    payload: bytes
        The content.

    Returns
    -------
    str
        The digest as lowercase hexadecimal.
    """
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """
    The SHA-256 of a file, read in chunks.

    Arguments
    ---------
    path: Path
        The file.

    Returns
    -------
    str
        The digest as lowercase hexadecimal.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_stats(array) -> Dict[str, float]:
    """
    The summary kept for a feature matrix that is not stored whole.

    Chosen so that a wrong implementation is unlikely to match: the moments
    catch a scaling or offset error, the extremes catch clipping, the shape
    catches a framing or dimension error, and the first row's sum catches a
    time reversal or an off-by-one in the frame offset, which every
    whole-matrix moment is blind to.

    Arguments
    ---------
    array: numpy.ndarray
        A two-dimensional array, frames by dimensions.

    Returns
    -------
    Dict[str, float]
        The statistics, keyed as STAT_KEYS.
    """
    import numpy as np

    values = np.asarray(array, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {values.shape}")
    return {
        "frames": int(values.shape[0]),
        "dim": int(values.shape[1]),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "first_row_sum": float(values[0].sum()),
    }


def compare_exact(
    name: str, payload: bytes, expected_digest: str
) -> Optional[Difference]:
    """
    Compare bytes against a recorded digest.

    Arguments
    ---------
    name: str
        What is being compared, for the message.
    payload: bytes
        The content produced now.
    expected_digest: str
        The digest recorded in the manifest.

    Returns
    -------
    Optional[Difference]
        None if they match, otherwise the difference.
    """
    actual = sha256_bytes(payload)
    if actual == expected_digest:
        return None
    return Difference(
        name,
        f"content differs from the reference: sha256 {actual[:12]} rather "
        f"than {expected_digest[:12]} ({len(payload)} bytes produced)",
    )


def compare_array(
    name: str, array, reference, tolerance: Tolerance
) -> Optional[Difference]:
    """
    Compare an array against a recorded one, elementwise, at a tolerance.

    Arguments
    ---------
    name: str
        What is being compared, for the message.
    array: numpy.ndarray
        The array produced now.
    reference: numpy.ndarray
        The recorded array.
    tolerance: Tolerance
        How closely they must agree.

    Returns
    -------
    Optional[Difference]
        None if they agree, otherwise the difference, carrying the worst
        offending element so that a near miss and a wholesale mismatch do
        not read the same.
    """
    import numpy as np

    got = np.asarray(array, dtype=np.float64)
    want = np.asarray(reference, dtype=np.float64)

    if got.shape != want.shape:
        return Difference(
            name, f"shape {got.shape} rather than the reference's {want.shape}"
        )

    if got.size == 0:
        return None

    difference = np.abs(got - want)
    allowed = tolerance.atol + tolerance.rtol * np.abs(want)
    over = difference > allowed
    if not over.any():
        return None

    worst = int(np.argmax(difference - allowed))
    raw = np.unravel_index(worst, got.shape)
    # Plain ints, so the message reads (1, 2) rather than
    # (np.int64(1), np.int64(2)) in the test failure it exists to explain.
    index = tuple(int(axis) for axis in raw)
    return Difference(
        name,
        f"{int(over.sum())} of {got.size} elements exceed rtol="
        f"{tolerance.rtol:g} atol={tolerance.atol:g}; worst at {index}, "
        f"{got[index]:.8g} rather than {want[index]:.8g} "
        f"(absolute {difference[index]:.3g}, allowed {allowed[index]:.3g})",
    )


def compare_stats(
    name: str,
    stats: Dict[str, float],
    reference: Dict[str, float],
    tolerance: Tolerance,
) -> List[Difference]:
    """
    Compare an array's statistics against recorded ones.

    The counts, `frames` and `dim`, are compared exactly whatever the
    tolerance: a different number of frames is a framing bug, not numerical
    noise, and letting a tolerance absorb it would hide the clearest signal
    there is.

    Arguments
    ---------
    name: str
        What is being compared, for the messages.
    stats: Dict[str, float]
        The statistics produced now, as `array_stats` returns.
    reference: Dict[str, float]
        The recorded statistics.
    tolerance: Tolerance
        How closely the real-valued statistics must agree.

    Returns
    -------
    List[Difference]
        One entry per statistic that differs; empty if all agree.
    """
    differences: List[Difference] = []

    for key in ("frames", "dim"):
        if key in reference and int(stats.get(key, -1)) != int(reference[key]):
            differences.append(
                Difference(
                    f"{name}.{key}",
                    f"{stats.get(key)} rather than the reference's "
                    f"{reference[key]}; a count, so compared exactly",
                )
            )

    for key in STAT_KEYS:
        if key in ("frames", "dim") or key not in reference:
            continue
        if key not in stats:
            differences.append(Difference(f"{name}.{key}", "not produced"))
            continue
        got = float(stats[key])
        want = float(reference[key])
        allowed = tolerance.atol + tolerance.rtol * abs(want)
        if abs(got - want) > allowed:
            differences.append(
                Difference(
                    f"{name}.{key}",
                    f"{got:.8g} rather than {want:.8g} (absolute "
                    f"{abs(got - want):.3g}, allowed {allowed:.3g})",
                )
            )

    return differences


@dataclass
class Manifest:
    """
    What a reference capture contains and how exactly it must be matched.

    Attributes
    ----------
    corpus: str
        Which recipe the capture is of, such as "yesno".
    captured: str
        The date of capture, ISO format.
    image: str
        The container image that produced it, including its digest. A
        reference whose provenance is unknown is not a reference.
    recipe: str
        The recipe directory, relative to the repository root.
    notes: List[str]
        Anything a reader needs to interpret the data.
    exact: Dict[str, str]
        Relative path to SHA-256, for artefacts compared as bytes.
    arrays: Dict[str, Dict[str, Any]]
        Name to a record with `path`, `shape` and `tolerance`, for arrays
        stored whole.
    stats: Dict[str, Dict[str, float]]
        Name to statistics, for arrays not stored whole.
    large: Dict[str, int]
        Relative path to size in bytes, for exact artefacts whose digest was
        recorded without their content. An exact artefact is compared by
        digest, so the comparison still works; what is lost is the ability to
        diff a mismatch against the committed reference. Listed separately so
        a missing file can be told apart from a forgotten one.
    shapes: Dict[str, Dict[str, int]]
        Relative path to the FST's size as `states`, `arcs` and `finals`.
        Recorded for every FST text captured, large or small, because a
        size comparison against a Python replacement otherwise has to be
        inferred from the byte count -- 138,155,979 bytes of `LG.fst.txt`
        divided by a guessed bytes-per-arc, which is how the figure
        "about 5.8M arcs" entered `docs/kaldi-graph-chain.md`. The
        question it answers is what Kaldi's determinize-and-minimise
        actually buys over the composition we ship, and that deserves a
        count rather than an estimate.
    functional: Dict[str, Dict[str, float]]
        Relative path to a map from a word sequence to the weight the FST
        gives it. The standard for an FST is functional equivalence rather
        than byte equality, but a digest supports only the latter -- so a
        large FST, whose bytes are not kept, would be checkable by exactly
        the test its standard rejects. `G.fst` is already known not to be
        byte-equal, because `arpa2fst` numbers its states in whatever order
        it meets histories, so its digest can never match and would say
        nothing. This is the fingerprint that makes the right comparison
        possible on a corpus too large to store whole.
    """

    corpus: str = ""
    captured: str = ""
    image: str = ""
    recipe: str = ""
    notes: List[str] = field(default_factory=list)
    exact: Dict[str, str] = field(default_factory=dict)
    arrays: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    large: Dict[str, int] = field(default_factory=dict)
    shapes: Dict[str, Dict[str, int]] = field(default_factory=dict)
    functional: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def write(self, path: Path) -> None:
        """
        Write the manifest as formatted JSON.

        Sorted and indented so that a capture appears in review as a
        readable diff rather than one long line.

        Arguments
        ---------
        path: Path
            Where to write it.
        """
        payload = asdict(self)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    @classmethod
    def read(cls, path: Path) -> "Manifest":
        """
        Read a manifest.

        Arguments
        ---------
        path: Path
            The manifest file.

        Returns
        -------
        Manifest
            The parsed manifest.

        Raises
        ------
        ValueError
            If a key the manifest format does not define is present, which
            means the file was written by something else or by a later
            version.
        """
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected a JSON object")
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"{path}: unrecognised keys {unknown}")
        return cls(**payload)

    def tolerance_for(self, name: str) -> Tolerance:
        """
        The tolerance recorded for one array, defaulting to exact.

        Arguments
        ---------
        name: str
            The array's name in the manifest.

        Returns
        -------
        Tolerance
            Its tolerance.
        """
        record = self.arrays.get(name, {})
        recorded = record.get("tolerance") or {}
        return Tolerance(
            rtol=float(recorded.get("rtol", 0.0)),
            atol=float(recorded.get("atol", 0.0)),
        )

    def names(self) -> Tuple[Sequence[str], Sequence[str], Sequence[str]]:
        """
        What the capture holds, as three sorted lists.

        Returns
        -------
        Tuple[Sequence[str], Sequence[str], Sequence[str]]
            The exact paths, the whole arrays and the statistics.
        """
        return sorted(self.exact), sorted(self.arrays), sorted(self.stats)

    def is_empty(self) -> bool:
        """
        Whether the manifest records nothing at all.

        A capture that has not been taken yet is empty rather than absent,
        so that the tests comparing against it can say so and skip rather
        than fail on a missing file.
        """
        return not (self.exact or self.arrays or self.stats)
