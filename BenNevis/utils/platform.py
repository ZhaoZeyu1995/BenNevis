"""
Platform and device detection, so that BenNevis runs on Linux, macOS and
Windows and on whatever accelerator is actually present.

The toolkit was written for Linux with CUDA, which is still the only
configuration that supports the whole feature set. The helpers here make the
other configurations usable rather than pretending they are equivalent: each
one that costs performance or capability says so once, through `warnings.warn`,
instead of failing deep inside a training step.

The constraints encoded here were measured, not assumed:

- **k2 has no MPS backend.** Moving an FSA to an MPS device raises an
  `AssertionError`, so the graph loss cannot run there.
- **MPS has no float64.** `use_double_scores=True` must stay (see the gotcha in
  CLAUDE.md), and MPS cannot represent the scores, so the graph loss has to be
  computed on the CPU even when the network runs on MPS. Cross-device autograd
  handles this: the gradient flows back to the MPS tensors correctly.
- **MPS cannot take part in DDP.** The `c10d::allgather_` operator is not
  implemented for it, so an MPS module cannot be wrapped, whatever the backend.
- **NCCL is CUDA-only.** Every other device has to use gloo, which is also the
  only backend available on Windows.

Functions
---------
available_accelerators
    List the accelerator kinds torch can see.
resolve_device
    Choose the device to train or predict on, warning about the trade-offs.
resolve_dist_backend
    Choose the `torch.distributed` backend matching a device.
supports_ddp
    Whether a device can be wrapped in `DistributedDataParallel`.
graph_loss_device
    Where the k2 graph loss must be computed for a given network device.
supports_symlink
    Whether this platform can create symbolic links.
link_or_copy
    Create a symlink, falling back to a copy where links are unavailable.
describe_platform
    A one-line description of the platform, for logs and bug reports.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""

import logging
import os
import platform
import shutil
import warnings
from typing import List, Optional

import torch

__all__ = [
    "available_accelerators",
    "resolve_device",
    "resolve_dist_backend",
    "supports_ddp",
    "graph_loss_device",
    "supports_symlink",
    "link_or_copy",
    "describe_platform",
]


def _mps_is_available() -> bool:
    """
    Whether torch can see a usable MPS device.

    `torch.backends.mps` exists on every platform in the supported torch
    versions, but is accessed defensively here so that a build without it
    reports no MPS rather than raising.

    Returns
    -------
    bool
        True if an MPS device is present and usable.
    """
    backends = getattr(torch.backends, "mps", None)
    if backends is None:
        return False
    try:
        return bool(backends.is_available())
    except Exception:  # pragma: no cover - defensive, build-dependent
        return False


def available_accelerators() -> List[str]:
    """
    List the accelerator kinds torch can see on this machine.

    Returns
    -------
    List[str]
        A list containing any of "cuda" and "mps", in order of preference.
        Empty if only the CPU is available.
    """
    accelerators = []
    if torch.cuda.is_available():
        accelerators.append("cuda")
    if _mps_is_available():
        accelerators.append("mps")
    return accelerators


def resolve_device(
    requested: Optional[str] = None,
    *,
    local_rank: int = 0,
    allow_mps: bool = True,
    warn: bool = True,
) -> torch.device:
    """
    Choose the device to place the model and batches on.

    The preference order is CUDA, then MPS, then the CPU. Anything other than
    CUDA costs either performance or capability, and each such choice emits a
    warning so that a slow or restricted run is never silent.

    Arguments
    ---------
    requested: Optional[str]
        An explicit device string such as "cuda", "cuda:1", "mps" or "cpu".
        When given it is honoured if torch can provide it, and an unavailable
        request is an error rather than a silent downgrade -- a run asked for
        CUDA that quietly lands on the CPU would waste hours before anyone
        noticed. None means detect.
    local_rank: int
        The device index to use for CUDA, which is the rank within the node
        rather than the global rank. Ignored for the other device types.
    allow_mps: bool
        Whether MPS may be selected by detection. False excludes it, for a
        caller that must have a device able to take part in DDP -- MPS cannot.
        An explicit `requested="mps"` is honoured regardless.
    warn: bool
        Whether to emit the warnings. False is for callers that have already
        reported the platform, such as tests.

    Returns
    -------
    torch.device
        The device to use.

    Raises
    ------
    RuntimeError
        If `requested` names a device type torch cannot provide.
    """
    if requested is not None:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Device 'cuda' was requested but torch reports no CUDA device. "
                f"This platform offers: {available_accelerators() or ['cpu']}."
            )
        if device.type == "mps" and not _mps_is_available():
            raise RuntimeError(
                "Device 'mps' was requested but torch reports no MPS device. "
                f"This platform offers: {available_accelerators() or ['cpu']}."
            )
        if warn:
            _warn_about(device)
        return device

    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)

    if _mps_is_available():
        if allow_mps:
            device = torch.device("mps")
            if warn:
                _warn_about(device)
            return device
        if warn:
            warnings.warn(
                "An MPS device is available but was excluded, because the "
                "caller requires a device that can take part in "
                "DistributedDataParallel and MPS cannot (c10d::allgather_ is "
                "not implemented for it). Falling back to the CPU, which "
                "will be very slow.",
                stacklevel=3,
            )

    device = torch.device("cpu")
    if warn:
        _warn_about(device)
    return device


def _warn_about(device: torch.device) -> None:
    """
    Emit the warning appropriate to a chosen device, if any.

    CUDA is the configuration the toolkit was written for and warrants no
    warning. The others each lose something worth saying out loud.

    Arguments
    ---------
    device: torch.device
        The device that was chosen.
    """
    if device.type == "cpu":
        warnings.warn(
            "No accelerator was found, so training will run on the CPU. This "
            "is extremely slow and is intended for development and testing "
            "rather than for real training runs.",
            stacklevel=3,
        )
    elif device.type == "mps":
        warnings.warn(
            "Using the MPS device. k2 has no MPS backend and MPS cannot "
            "represent float64, so the graph loss will be computed on the "
            "CPU; every step therefore copies the log probabilities across "
            "devices and synchronises. Distributed training is also "
            "unavailable on MPS. Expect this to be useful for development "
            "rather than fast.",
            stacklevel=3,
        )


def resolve_dist_backend(device: torch.device) -> str:
    """
    Choose the `torch.distributed` backend matching a device.

    Arguments
    ---------
    device: torch.device
        The device the model will be placed on.

    Returns
    -------
    str
        "nccl" for CUDA, "gloo" otherwise. NCCL is CUDA-only, and gloo is the
        only backend available on macOS and Windows.
    """
    return "nccl" if device.type == "cuda" else "gloo"


def supports_ddp(device: torch.device) -> bool:
    """
    Whether a device can be wrapped in `DistributedDataParallel`.

    Arguments
    ---------
    device: torch.device
        The device the model will be placed on.

    Returns
    -------
    bool
        False for MPS, which does not implement the collective operators DDP
        needs; True otherwise.
    """
    return device.type != "mps"


def graph_loss_device(device: torch.device) -> torch.device:
    """
    Where the k2 graph loss must be computed, given the network's device.

    k2 supports the CPU and CUDA only, and the double-precision scores the
    loss relies on cannot be represented on MPS. Cross-device autograd means
    computing the loss on the CPU while the network runs on MPS is correct:
    the gradient flows back to the MPS tensors.

    Arguments
    ---------
    device: torch.device
        The device the network's outputs are on.

    Returns
    -------
    torch.device
        The device to compute the graph loss on, which is `device` itself for
        the CPU and CUDA, and the CPU for MPS.
    """
    return torch.device("cpu") if device.type == "mps" else device


def supports_symlink() -> bool:
    """
    Whether this platform can create symbolic links.

    POSIX always can. Windows needs either administrator rights or developer
    mode, so the capability is probed rather than inferred from the platform
    name.

    Returns
    -------
    bool
        True if `os.symlink` is usable here.
    """
    return hasattr(os, "symlink") and os.name != "nt"


def link_or_copy(target: str, link_path: str) -> None:
    """
    Point `link_path` at `target`, by symlink where possible and by copy where
    not.

    Used for the `best.pt` checkpoint pointer. A copy costs disc space but
    behaves identically for every reader, which matters more than saving the
    bytes: a checkpoint directory that works the same way everywhere is worth
    one duplicated file.

    Arguments
    ---------
    target: str
        The path the link should resolve to. A relative target is interpreted
        relative to the directory containing `link_path`, as `os.symlink`
        does.
    link_path: str
        The path of the link to create. An existing file or link there is
        replaced.
    """
    if os.path.islink(link_path) or os.path.exists(link_path):
        os.remove(link_path)

    if os.path.isabs(target):
        resolved_target = target
    else:
        resolved_target = os.path.join(os.path.dirname(link_path), target)

    if supports_symlink():
        os.symlink(target, link_path)
        return

    logging.debug(
        f"Symbolic links are unavailable on this platform; copying "
        f"{resolved_target} to {link_path} instead."
    )
    shutil.copyfile(resolved_target, link_path)


def describe_platform() -> str:
    """
    A one-line description of the platform, for logs and bug reports.

    Returns
    -------
    str
        The operating system, machine architecture, Python and torch versions,
        and the accelerators torch can see.
    """
    accelerators = available_accelerators() or ["cpu only"]
    return (
        f"{platform.system()} {platform.machine()} | "
        f"Python {platform.python_version()} | "
        f"torch {torch.__version__} | "
        f"accelerators: {', '.join(accelerators)}"
    )
