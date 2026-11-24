"""
Memory indicators for a long training run.

A `ctc` run on mini-librispeech was SIGKILLed three times on a 15.6GiB CI
runner, and the host-side sample taken every thirty seconds showed the
trainer climbing from 1.3GiB in epoch 1 to 9.3GiB by epoch 7, where the
runner ran out. That sample says *how much* and *when*; it cannot say
*what*, and without that the only remedy is to train fewer epochs.

What is missing is the composition. The loss path was ruled out locally
-- two thousand steps of `GraphLoss`, with fixed and with varying batch
shapes, held flat at about 400MiB -- so whatever accumulates needs the
real data path to show itself, and the real data path costs three and a
half hours a look. That is the case for instrumenting rather than
guessing again.

The reading is deliberately a *difference*. An absolute count says
little: a trainer legitimately holds a lexicon FST, a topology, cached
features and an optimiser state. A type whose instance count rises every
epoch and never falls is a leak, and naming that type is the whole
question -- `k2.Fsa` would point at the graph, `torch.Tensor` at a
retained autograd graph, `dict` or `list` at something the trainer
appends to.

Taking the reading costs memory itself: `gc.get_objects()` builds a list
of every tracked object, which on a large run is millions of references.
So the resident size is read *before* that list exists, and the reading
is taken once an epoch rather than per step.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import gc
import os
import subprocess
import sys
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple

__all__ = [
    "rss_bytes",
    "count_live_objects",
    "tensor_bytes",
    "MemoryReporter",
]


def rss_bytes() -> Optional[int]:
    """
    The process's current resident size in bytes, or None if unavailable.

    The *current* size, not the peak: `resource.getrusage`'s `ru_maxrss`
    never decreases, so it cannot show a plateau or a release and is the
    wrong tool for watching growth. Note also that `ru_maxrss` is bytes on
    macOS and kilobytes on Linux, a unit difference that has already been
    misreported once in this project.

    Returns
    -------
    rss: Optional[int]
        Resident size in bytes, or None where it cannot be read.
    """
    # Linux: field 2 of /proc/self/statm is the resident set in pages.
    try:
        with open("/proc/self/statm", "r") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        pass
    # Windows: the working set is the analogue of the resident set, read
    # through psapi with ctypes so that this needs no new dependency --
    # tier 1 installs pytest and pyyaml only, and an indicator that is
    # blank on a developer's own machine is an indicator they will not
    # use.
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            # The signatures are declared rather than left to ctypes'
            # defaults, and that is the whole of why a first attempt here
            # returned nothing. `GetCurrentProcess` hands back the
            # pseudo-handle -1; with the default `c_int` return type that
            # becomes a 32-bit -1, and passing it where a 64-bit HANDLE is
            # wanted truncates it, so the call fails and reports no error
            # anyone sees.
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.GetCurrentProcess.argtypes = []
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_Counters),
                wintypes.DWORD,
            ]

            counters = _Counters()
            counters.cb = ctypes.sizeof(_Counters)
            if psapi.GetProcessMemoryInfo(
                kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            ):
                return int(counters.WorkingSetSize)
        except Exception:
            pass
        return None
    # macOS and other POSIX systems: ps reports kilobytes.
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        if out:
            return int(out.split()[0]) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def _scan() -> Tuple[Dict[str, int], Optional[int]]:
    """
    Walk the live objects once, counting types and tensor storage.

    One pass rather than two, because the walk is the expensive part and
    `gc.get_objects()` allocates a list of every tracked object -- taking
    it twice doubles both the time and the memory the reading itself
    costs.

    Inspection is done with warnings suppressed. Touching an attribute of
    a deprecated alias during the walk raises `FutureWarning` from inside
    torch, which has nothing to do with the run being measured and would
    otherwise print once per reading.

    Returns
    -------
    counts: Dict[str, int]
        Type name to number of live instances.
    total: Optional[int]
        Bytes held by distinct live tensor storages, or None without
        torch.
    """
    try:
        import torch

        tensor_type = torch.Tensor
    except ImportError:
        tensor_type = None

    counts: Counter = Counter()
    seen = set()
    total = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for obj in gc.get_objects():
            try:
                counts[type(obj).__name__] += 1
            except Exception:  # pragma: no cover - a broken __name__
                counts["<unknown>"] += 1
                continue
            if tensor_type is None:
                continue
            try:
                if not isinstance(obj, tensor_type):
                    continue
                storage = obj.untyped_storage()
                key = (storage.data_ptr(), storage.device)
            except Exception:
                # A tensor can raise on inspection -- a meta tensor, or
                # one whose storage has been freed. Skipping it is right:
                # it holds no resident memory to account for.
                continue
            if key[0] and key not in seen:
                seen.add(key)
                total += storage.nbytes()
    return dict(counts), (None if tensor_type is None else total)


def count_live_objects() -> Dict[str, int]:
    """
    Count the live garbage-collected objects by type name.

    Only objects the collector tracks are counted, which excludes plain
    integers and strings but includes every container, every instance of a
    class, and so every `k2.Fsa` and `torch.Tensor`.

    Returns
    -------
    counts: Dict[str, int]
        Type name to number of live instances.
    """
    return _scan()[0]


def tensor_bytes() -> Optional[int]:
    """
    The total storage of every live torch tensor, in bytes.

    Distinct storages are counted once, because a view and the tensor it
    views share their memory and adding both would double-count it. This
    is the figure that separates "the trainer is holding tensors" from
    "the allocator is holding memory the trainer has released".

    Returns
    -------
    total: Optional[int]
        Bytes held by live tensor storages, or None without torch.
    """
    return _scan()[1]


class MemoryReporter:
    """
    Report resident size and what grew since the previous reading.

    Arguments
    ---------
    top: int
        How many of the fastest-growing types to name, by default 6.

    Notes
    -----
    The first reading has nothing to compare against and so names no
    growers; it establishes the baseline. Keep one reporter for the run
    rather than one per epoch, or every reading is a first reading.
    """

    def __init__(self, top: int = 6):
        self.top = top
        self._previous: Optional[Dict[str, int]] = None

    def growth(self, counts: Dict[str, int]) -> List[Tuple[str, int]]:
        """
        The types that gained the most instances since the last reading.

        Arguments
        ---------
        counts: Dict[str, int]
            The current counts, as `count_live_objects` returns them.

        Returns
        -------
        growers: List[Tuple[str, int]]
            Type name and increase, largest first. Empty on the first
            reading, and types that did not grow are left out -- a leak is
            a rise that does not come back, so a fall is not interesting
            and a flat count is the normal case.
        """
        previous = self._previous
        self._previous = dict(counts)
        if previous is None:
            return []
        risen = [
            (name, count - previous.get(name, 0))
            for name, count in counts.items()
            if count - previous.get(name, 0) > 0
        ]
        risen.sort(key=lambda item: (-item[1], item[0]))
        return risen[: self.top]

    def report(self, label: str = "") -> str:
        """
        Take a reading and format it as one line for the log.

        Arguments
        ---------
        label: str
            Prefix identifying the reading, such as `epoch 3`.

        Returns
        -------
        line: str
            A single line: resident size, live tensor storage, and the
            types that grew since the previous reading.
        """
        # Read before `gc.get_objects()` exists, since that list is itself
        # millions of references on a large run.
        rss = rss_bytes()
        counts, tensors = _scan()
        growth = self.growth(counts)

        parts = [f"memory{' ' + label if label else ''}:"]
        parts.append(
            f"rss {rss / 2 ** 20:.0f}MiB" if rss is not None else "rss unavailable"
        )
        if tensors is not None:
            parts.append(f"tensors {tensors / 2 ** 20:.0f}MiB")
        parts.append(f"objects {sum(counts.values())}")
        if growth:
            grew = ", ".join(f"{name} +{count}" for name, count in growth)
            parts.append(f"grew: {grew}")
        return " ".join(parts)
