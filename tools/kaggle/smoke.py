"""
Does the BenNevis stack run on a free Kaggle GPU at all?

This answers that in one short run, before any plumbing is built around it.
It is a *smoke test* in the strict sense: it does not train, does not touch
a corpus and does not produce a WER. It asks the questions whose answers
decide whether a Kaggle lane is worth building:

1. does a k2 CUDA wheel exist for the torch Kaggle ships, and does it
   install?
2. does k2 import with CUDA support, and does a GPU FSA operation run?
3. **does the graph loss agree between the GPU and the CPU?** This is the
   one that matters. A GPU lane that disagreed with the CPU would move the
   WER ledger for a reason that is not the model, so the test compares
   numerator, denominator and loss and fails on a disagreement rather than
   reporting a speed-up.
4. does `intersect_dense_pruned` -- the decoding kernel, and the thing
   that has twice run out of CI time at 6.46s an utterance -- run on the
   GPU and pick the same best path?
5. is a BenNevis topology arc-sorted, and **does sorting change the
   denominator scores on CUDA?** `CLAUDE.md` records that it does not on
   the CPU and that "CUDA is still unchecked -- there an unsorted graph is
   undefined rather than merely slow". This is the cheapest place that
   question has ever been answerable.
6. are both T4s visible, and is NCCL usable across them? That is what a
   multi-rank DDP run would need, and nothing in CI can exercise it.

It is written as a Kaggle **script** kernel rather than a notebook: the
source is then a reviewable, diffable `.py` in the repository instead of
JSON with outputs baked in, and the same file runs locally with
`python tools/kaggle/smoke.py`. `kernel-metadata.json` beside it carries
the GPU and internet flags.

Nothing here is a gate. A PR cannot block on a Kaggle run, and Kaggle runs
its own image rather than one pinned by digest, so treat every figure it
prints as exploratory. See `tools/kaggle/README.md`.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import os
import platform
import re
import subprocess
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

# The k2 build and torch version CI and docker/Dockerfile pin. A CUDA wheel
# exists at this exact k2 build for torch 2.15.0 (cuda 13.2 and 13.4,
# cp311 to cp315), so the GPU path is a wheel swap rather than a version
# negotiation -- the same k2 source, compiled for CUDA.
PINNED_K2_BUILD = "1.24.4.dev20261112"
PINNED_TORCH = "2.15.0"
K2_CUDA_INDEX = "https://k2-fsa.github.io/k2/cuda.html"

# The losses tolerance from CLAUDE.md's "Equivalence and tolerances". The
# graph loss is the fastest and most localising signal there is, so it is
# what a device comparison should be judged on.
LOSS_RTOL = 1e-4

# A phone table small enough to read and real enough to expand a topology
# against, matching tests/core/conftest.py so a failure here can be
# compared with a tier 2 failure.
PHONES = ["<eps>", "a", "b", "c"]

results: List[Tuple[str, str, str]] = []


def record(phase: str, status: str, detail: str = "") -> None:
    """
    Note one phase's outcome and print it as it happens.

    Printing immediately matters more here than it would locally: a Kaggle
    session can be cut off at its 12-hour limit or lose its accelerator,
    and a phase that only reported at the end would report nothing.

    Arguments
    ---------
    phase: str
        What was being checked.
    status: str
        "PASS", "FAIL" or "SKIP".
    detail: str
        The measurement, or the reason for a failure or a skip.
    """
    results.append((phase, status, detail))
    print(f"[{status:4}] {phase}" + (f" -- {detail}" if detail else ""), flush=True)


def shell(command: Sequence[str]) -> str:
    """
    Run a command and return its output, or a readable failure.

    Arguments
    ---------
    command: Sequence[str]
        The argument list. Never a shell string: Kaggle's image is Linux
        but the recipes' POSIX assumptions are exactly what goal 2 is
        removing, so nothing new should add one.

    Returns
    -------
    str
        The combined output, stripped, or a message naming the failure.
    """
    try:
        done = subprocess.run(
            list(command), capture_output=True, text=True, timeout=600
        )
    except FileNotFoundError:
        return f"({command[0]} not found)"
    except subprocess.TimeoutExpired:
        return f"({command[0]} timed out)"
    return (done.stdout + done.stderr).strip()


def phase_environment() -> None:
    """
    Report what the runner is, the way the nightly's host.log does.

    A figure from a Kaggle run is worth nothing without this: the free
    accelerator changed from a P100 to two T4s in September 2026, and the
    quota is explicitly not contractual.
    """
    print("=" * 72, flush=True)
    print(f"python   {sys.version.split()[0]}  ({platform.platform()})", flush=True)
    smi = shell(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    print(f"nvidia-smi\n{smi}", flush=True)
    print("=" * 72, flush=True)
    if "not found" in smi or not smi:
        record("a GPU is attached", "FAIL", "nvidia-smi reports nothing")
        return
    record("a GPU is attached", "PASS", smi.replace("\n", " | "))


def wheel_candidates(torch_version: str, cuda_version: str) -> List[str]:
    """
    The k2 CUDA wheels published for one torch and CUDA pair.

    Scraped from k2's own index with the standard library rather than by
    guessing a filename, because the CUDA minor version k2 builds against
    moves and a guess fails as "no matching distribution" -- which reads
    like a broken network rather than a missing build.

    Arguments
    ---------
    torch_version: str
        As `torch.__version__` reports it, less any local part.
    cuda_version: str
        As `torch.version.cuda` reports it, for example "13.2".

    Returns
    -------
    List[str]
        Full version strings, newest build first. Empty if the index could
        not be read or publishes nothing for this pair.
    """
    try:
        with urllib.request.urlopen(K2_CUDA_INDEX, timeout=60) as response:
            page = response.read().decode("utf-8", "replace")
    except Exception as error:  # noqa: BLE001 -- any failure means "unknown"
        print(f"could not read {K2_CUDA_INDEX}: {error}", flush=True)
        return []
    pattern = re.compile(
        r"k2-(\d[\w.]*\+cuda"
        + re.escape(cuda_version)
        + r"\.torch"
        + re.escape(torch_version)
        + r")-cp"
    )
    found = sorted(set(pattern.findall(page)), reverse=True)
    return found


def phase_install() -> Optional[str]:
    """
    Install a k2 CUDA wheel that matches the torch already present.

    **Kaggle's torch is not reinstalled**, even when it differs from the
    pinned 2.15.0. Replacing it is a multi-gigabyte download that can
    leave the image's CUDA setup inconsistent, and a smoke test whose
    first act is to rebuild the environment is no longer testing the
    environment. A mismatch is reported loudly instead, because a figure
    measured against a different torch is not comparable with CI's.

    Returns
    -------
    Optional[str]
        The installed k2 version, or None if nothing could be installed.
    """
    try:
        import torch
    except ImportError:
        record("torch is importable", "FAIL", "no torch in the image")
        return None

    torch_version = torch.__version__.split("+")[0]
    cuda_version = torch.version.cuda
    record(
        "torch is importable",
        "PASS",
        f"torch {torch.__version__}, built against CUDA {cuda_version}",
    )
    if torch_version != PINNED_TORCH:
        record(
            "torch matches the pinned version",
            "FAIL",
            f"image has {torch_version}, CI and the image pin {PINNED_TORCH} -- "
            f"numbers from this run are not comparable with CI's, and the "
            f"pins in docker/Dockerfile and ci.yml would have to move together",
        )
    else:
        record("torch matches the pinned version", "PASS", PINNED_TORCH)
    if cuda_version is None:
        record("torch is a CUDA build", "FAIL", "this is a CPU-only wheel")
        return None

    candidates = wheel_candidates(torch_version, cuda_version)
    preferred = [v for v in candidates if v.startswith(PINNED_K2_BUILD)]
    ordered = preferred + [v for v in candidates if v not in preferred]
    if not ordered:
        record(
            "a k2 CUDA wheel exists for this torch",
            "FAIL",
            f"k2 publishes none for torch {torch_version} + CUDA {cuda_version}; "
            f"see {K2_CUDA_INDEX}",
        )
        return None
    record(
        "a k2 CUDA wheel exists for this torch",
        "PASS",
        f"{len(ordered)} candidate(s), trying {ordered[0]}"
        + ("" if preferred else " (NOT the pinned k2 build)"),
    )

    for version in ordered[:3]:
        print(f"installing k2=={version} ...", flush=True)
        output = shell(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                f"k2=={version}",
                "--find-links",
                K2_CUDA_INDEX,
            ]
        )
        try:
            import k2  # noqa: F401

            record("the k2 CUDA wheel installs", "PASS", version)
            return version
        except ImportError:
            print(output[-2000:], flush=True)
            continue
    record("the k2 CUDA wheel installs", "FAIL", "no candidate imported")
    return None


def phase_import() -> bool:
    """
    Import k2 and confirm it can place an FSA on the GPU.

    An import that succeeds proves less than it looks: a CPU-only k2 built
    against the right torch imports perfectly and then fails on the first
    `.to("cuda")`. So the check is the move, not the import.

    Returns
    -------
    bool
        Whether k2 is usable on the GPU.
    """
    try:
        import k2
        import torch
    except ImportError as error:
        record("k2 imports", "FAIL", str(error))
        return False
    record("k2 imports", "PASS", f"k2 {k2.__version__}")

    if not torch.cuda.is_available():
        record("torch sees the GPU", "FAIL", "torch.cuda.is_available() is False")
        return False
    record(
        "torch sees the GPU",
        "PASS",
        f"{torch.cuda.device_count()} device(s): "
        + ", ".join(
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ),
    )
    try:
        fsa = k2.ctc_topo(3, device=torch.device("cuda", 0))
        _ = fsa.num_arcs
        record(
            "a k2 FSA lives on the GPU",
            "PASS",
            f"ctc_topo: {fsa.shape[0]} states, {fsa.num_arcs} arcs on {fsa.device}",
        )
    except Exception as error:  # noqa: BLE001
        record("a k2 FSA lives on the GPU", "FAIL", f"{type(error).__name__}: {error}")
        return False
    return True


def loss_on(device: str, seed: int = 0) -> Dict[str, object]:
    """
    One CTC graph loss, computed end to end on one device.

    Deliberately the same shape as `BenNevis.core.GraphLoss`: a numerator
    built from the token sequences, a denominator from the topology, both
    intersected with the same dense vector, and `use_double_scores=True`,
    which `CLAUDE.md` requires because `False` has produced NaNs.

    Arguments
    ---------
    device: str
        "cpu" or "cuda".
    seed: int
        Fixes the log probabilities so the two devices are asked the same
        question. Without this the comparison measures nothing.

    Returns
    -------
    Dict[str, object]
        The numerator, denominator and loss tensors, on the CPU in double
        precision so they can be compared, and the elapsed seconds.
    """
    import k2
    import torch

    torch.manual_seed(seed)
    frames, tokens, batch = 60, 4, 3
    sequences = [[1, 2, 3], [2, 2], [3, 1, 2, 1]]

    log_probs = (
        torch.randn(batch, frames, tokens, generator=None)
        .log_softmax(-1)
        .to(device)
        .requires_grad_(True)
    )
    segments = torch.tensor([[i, 0, frames] for i in range(batch)], dtype=torch.int32)
    dense = k2.DenseFsaVec(log_probs, segments)

    numerator = k2.ctc_graph(sequences, device=torch.device(device))
    # The denominator is one copy of the topology per batch item, sorted
    # once and then replicated -- exactly what GraphLoss.get_den_graph
    # does. A single Fsa here silently returns one score for the whole
    # batch instead of one each, which looks like a working test.
    topology = k2.arc_sort(k2.ctc_topo(tokens - 1, device=torch.device(device)))
    denominator = k2.create_fsa_vec([topology for _ in range(batch)])

    started = time.perf_counter()
    num_lattice = k2.intersect_dense(k2.arc_sort(numerator), dense, output_beam=10.0)
    den_lattice = k2.intersect_dense(denominator, dense, output_beam=10.0)
    num = num_lattice.get_tot_scores(use_double_scores=True, log_semiring=True)
    den = den_lattice.get_tot_scores(use_double_scores=True, log_semiring=True)
    # A full CTC topology accepts every token sequence, so in the log
    # semiring its total over normalised log probabilities is log 1 = 0
    # for every utterance. That makes `den` a check that normalisation
    # holds rather than a discriminating device comparison, so the
    # Viterbi score is taken as well: tropical rather than log, a real
    # number, and a different k2 kernel.
    den_max = den_lattice.get_tot_scores(use_double_scores=True, log_semiring=False)
    loss = (den - num).sum()
    loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    return {
        "num": num.detach().double().cpu(),
        "den": den.detach().double().cpu(),
        "den_max": den_max.detach().double().cpu(),
        "loss": loss.detach().double().cpu(),
        "grad": log_probs.grad.detach().double().cpu(),
        "seconds": elapsed,
    }


def phase_loss_agrees() -> None:
    """
    Compare the graph loss between the devices, and only then the speed.

    The gradient is compared as well as the scores. `CLAUDE.md` records
    that k2 returns the true `d(loss)/d(log_probs)`, summing to -1 a
    frame, so the gradient is a sharper check than the loss: two devices
    could agree on a total while disagreeing on where it came from.
    """
    import torch

    try:
        cpu = loss_on("cpu")
    except Exception as error:  # noqa: BLE001
        record(
            "the loss computes on the CPU", "FAIL", f"{type(error).__name__}: {error}"
        )
        return
    record("the loss computes on the CPU", "PASS", f"{cpu['seconds']:.3f}s")

    try:
        gpu = loss_on("cuda")
    except Exception as error:  # noqa: BLE001
        record(
            "the loss computes on the GPU", "FAIL", f"{type(error).__name__}: {error}"
        )
        return
    record("the loss computes on the GPU", "PASS", f"{gpu['seconds']:.3f}s")

    zero = cpu["den"].abs().max().item()
    record(
        "the denominator totals zero in the log semiring",
        "PASS" if zero < 1e-4 else "FAIL",
        f"worst |den| is {zero:.3e}; a full CTC topology accepts everything, so "
        f"anything else means the log probabilities are not normalised",
    )

    for name in ("num", "den", "den_max", "loss", "grad"):
        a, b = cpu[name], gpu[name]
        if torch.isnan(b).any():
            record(f"the GPU {name} is finite", "FAIL", "it holds NaN")
            continue
        gap = (a - b).abs().max().item()
        scale = max(a.abs().max().item(), 1.0)
        if gap <= LOSS_RTOL * scale:
            record(
                f"the GPU and CPU {name} agree",
                "PASS",
                f"worst gap {gap:.3e} against a tolerance of {LOSS_RTOL * scale:.3e}",
            )
        else:
            record(
                f"the GPU and CPU {name} agree",
                "FAIL",
                f"worst gap {gap:.3e} exceeds {LOSS_RTOL * scale:.3e} -- a GPU "
                f"lane disagreeing with the CPU would move the WER ledger for "
                f"a reason that is not the model",
            )

    ratio = cpu["seconds"] / gpu["seconds"] if gpu["seconds"] else float("nan")
    record(
        "the GPU is faster on this shape",
        "PASS" if ratio > 1 else "FAIL",
        f"{ratio:.2f}x -- on a batch this small the kernel launch dominates, so "
        f"this is a sanity figure and not a training estimate",
    )


def phase_pruned_search() -> None:
    """
    Run the decoding kernel on the GPU and check it picks the same path.

    `intersect_dense_pruned` is what stage 10 spends its time in, and it
    is the reason a GPU lane is interesting at all: 6.46s an utterance on
    four CPU cores has twice exhausted a 350-minute CI run before the
    1,089 utterances of `dev_clean_2` were finished.
    """
    import k2
    import torch

    frames, tokens, batch = 40, 4, 2
    torch.manual_seed(1)
    probs = torch.randn(batch, frames, tokens).log_softmax(-1)
    segments = torch.tensor([[i, 0, frames] for i in range(batch)], dtype=torch.int32)

    paths = {}
    for device in ("cpu", "cuda"):
        try:
            graph = k2.arc_sort(k2.ctc_topo(tokens - 1, device=torch.device(device)))
            dense = k2.DenseFsaVec(probs.to(device), segments)
            started = time.perf_counter()
            lattice = k2.intersect_dense_pruned(
                graph,
                dense,
                search_beam=20.0,
                output_beam=8.0,
                min_active_states=30,
                max_active_states=10000,
            )
            best = k2.shortest_path(lattice, use_double_scores=True)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            paths[device] = (
                [int(x) for x in best.labels.cpu().tolist() if x > 0],
                elapsed,
            )
            record(
                f"intersect_dense_pruned runs on the {device.upper()}",
                "PASS",
                f"{elapsed:.3f}s, best path has {len(paths[device][0])} labels",
            )
        except Exception as error:  # noqa: BLE001
            record(
                f"intersect_dense_pruned runs on the {device.upper()}",
                "FAIL",
                f"{type(error).__name__}: {error}",
            )

    if len(paths) == 2:
        if paths["cpu"][0] == paths["cuda"][0]:
            record(
                "both devices decode the same labels",
                "PASS",
                f"{len(paths['cpu'][0])} labels, identical",
            )
        else:
            record(
                "both devices decode the same labels",
                "FAIL",
                f"CPU {paths['cpu'][0][:12]} against GPU {paths['cuda'][0][:12]} -- "
                f"pruning is approximate, so a small difference may be the beam "
                f"rather than a defect, but it must be understood before trusting "
                f"a GPU WER",
            )


def unsorted_copy(text: str) -> str:
    """
    The same FST with each state's arcs in reverse order.

    Built by rewriting the OpenFST text rather than by permuting a k2
    tensor, so only documented API is involved and the result is still
    exactly the same FST as a function.

    Arguments
    ---------
    text: str
        OpenFST text with integer labels, as `build_openfst_int_text`
        writes it: arc lines grouped by source state, then final states.

    Returns
    -------
    str
        The same lines, with each source state's arcs reversed.
    """
    groups: Dict[str, List[str]] = {}
    order: List[str] = []
    finals: List[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) < 4:
            finals.append(line)
            continue
        if fields[0] not in groups:
            groups[fields[0]] = []
            order.append(fields[0])
        groups[fields[0]].append(line)
    out: List[str] = []
    for state in order:
        out.extend(reversed(groups[state]))
    out.extend(finals)
    return "\n".join(out) + "\n"


def phase_arc_sorting() -> None:
    """
    Settle the arc-sorting question that `CLAUDE.md` leaves open on CUDA.

    `k2.intersect_dense` documents that its graphs must be arc-sorted, and
    `prepare_topo.sh`'s `fstarcsort --sort_type=olabel` does not make a k2
    FSA `ARC_SORTED`. On the CPU the denominator scores were measured
    identical either way for all eight topologies; on CUDA an unsorted
    graph is undefined rather than merely slow, and no tier of CI can run
    there. So this compares a BenNevis topology against a deliberately
    unsorted copy of the same FST, on the GPU.

    It needs the repository, and clones it if this is a bare Kaggle
    session.
    """
    import torch

    try:
        from BenNevis.topos import build as topo_build
        from BenNevis.topos import spec as topo_spec
    except ImportError:
        target = "/kaggle/working/BenNevis"
        if not os.path.isdir(target):
            print(f"cloning BenNevis into {target} ...", flush=True)
            print(
                shell(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        "https://github.com/ZhaoZeyu1995/BenNevis",
                        target,
                    ]
                )[-800:],
                flush=True,
            )
        sys.path.insert(0, target)
        try:
            from BenNevis.topos import build as topo_build
            from BenNevis.topos import spec as topo_spec
        except ImportError as error:
            record("the BenNevis topologies load", "SKIP", f"no repository: {error}")
            return

    import k2

    try:
        spec = topo_spec.load("ctc")
        tokens = topo_build.build_tokens(
            spec, PHONES, with_eps=False, with_disambig=False
        )
        phones = topo_build.build_phone_symbols(PHONES, with_disambig=False)
        text = topo_build.build_openfst_int_text(
            spec,
            "k2",
            PHONES,
            isymbols=topo_build.symbol_ids(tokens),
            osymbols=topo_build.symbol_ids(phones),
        )
    except Exception as error:  # noqa: BLE001
        record(
            "the BenNevis topologies load", "FAIL", f"{type(error).__name__}: {error}"
        )
        return
    record(
        "the BenNevis topologies load",
        "PASS",
        f"ctc over {len(tokens)} tokens: {tokens}",
    )

    device = torch.device("cuda", 0)
    plain = k2.Fsa.from_openfst(text, acceptor=False).to(device)
    reversed_arcs = k2.Fsa.from_openfst(unsorted_copy(text), acceptor=False).to(device)

    sorted_flag = k2.fsa_properties.ARC_SORTED
    record(
        "the topology is not arc-sorted on its own",
        "PASS" if not (plain.properties & sorted_flag) else "FAIL",
        f"properties as built: ARC_SORTED is "
        f"{bool(plain.properties & sorted_flag)} -- the premise of "
        f"get_den_graph's k2.arc_sort call",
    )

    frames, batch = 50, 2
    torch.manual_seed(2)
    probs = torch.randn(batch, frames, len(tokens)).log_softmax(-1).to(device)
    segments = torch.tensor([[i, 0, frames] for i in range(batch)], dtype=torch.int32)

    scores = {}
    for name, graph in (
        ("sorted", k2.arc_sort(plain)),
        ("unsorted", reversed_arcs),
        ("unsorted then sorted", k2.arc_sort(reversed_arcs)),
    ):
        try:
            dense = k2.DenseFsaVec(probs, segments)
            vec = k2.create_fsa_vec([graph for _ in range(batch)])
            lattice = k2.intersect_dense(vec, dense, output_beam=10.0)
            # Tropical, not log. The log-semiring total of a full CTC
            # topology over normalised log probabilities is zero for
            # every utterance whatever the arc order, so comparing it
            # across sortings would compare 0 with 0 and detect nothing
            # -- which is exactly the shape of test this file exists to
            # avoid. The Viterbi score is a real number that a wrong
            # intersection would move.
            scores[name] = (
                lattice.get_tot_scores(use_double_scores=True, log_semiring=False)
                .double()
                .cpu()
            )
            record(
                f"the {name} denominator computes on the GPU",
                "PASS",
                f"Viterbi scores {[round(float(x), 4) for x in scores[name]]}",
            )
        except Exception as error:  # noqa: BLE001
            record(
                f"the {name} denominator computes on the GPU",
                "FAIL",
                f"{type(error).__name__}: {error}",
            )

    if "sorted" in scores and "unsorted then sorted" in scores:
        gap = (scores["sorted"] - scores["unsorted then sorted"]).abs().max().item()
        record(
            "arc_sort makes arc order irrelevant on CUDA",
            "PASS" if gap == 0.0 else "FAIL",
            f"worst gap {gap:.3e} between two sortings of the same FST; anything "
            f"but zero means arc_sort is not canonicalising",
        )
    if "sorted" in scores and "unsorted" in scores:
        gap = (scores["sorted"] - scores["unsorted"]).abs().max().item()
        record(
            "an UNSORTED graph gives the same scores on CUDA",
            "PASS" if gap == 0.0 else "FAIL",
            f"worst gap {gap:.3e} -- this is the question CLAUDE.md leaves open. "
            f"A gap confirms that get_den_graph's arc_sort is load-bearing on "
            f"CUDA where it was only tidiness on the CPU; zero says CUDA "
            f"tolerates it too, and neither result makes the sort removable",
        )


def phase_multi_gpu() -> None:
    """
    Check both T4s are usable and that NCCL works across them.

    `Trainer` all-reduces losses and selects its device by `LOCAL_RANK`, and
    NCCL is CUDA-only, so every other device uses gloo. Nothing in CI has
    ever run a two-rank CUDA group; two free T4s is the first opportunity.
    """
    import torch

    count = torch.cuda.device_count()
    if count < 2:
        record(
            "two GPUs are attached",
            "SKIP",
            f"only {count} -- select 'GPU T4 x2' as the accelerator",
        )
        return
    record("two GPUs are attached", "PASS", f"{count} devices")

    if not torch.distributed.is_nccl_available():
        record("NCCL is available", "FAIL", "torch reports no NCCL")
        return
    record("NCCL is available", "PASS", f"version {torch.cuda.nccl.version()}")

    script = (
        "import os, torch, torch.distributed as dist\n"
        "dist.init_process_group('nccl')\n"
        "rank = dist.get_rank()\n"
        "torch.cuda.set_device(rank)\n"
        "t = torch.full((4,), float(rank + 1), device='cuda')\n"
        "dist.all_reduce(t)\n"
        "assert t[0].item() == 3.0, t[0].item()\n"
        "if rank == 0:\n"
        "    print('nccl all_reduce ok', t.tolist())\n"
        "dist.destroy_process_group()\n"
    )
    path = "/tmp/bennevis_nccl_probe.py"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(script)
    output = shell(
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", path]
    )
    if "nccl all_reduce ok" in output:
        record(
            "a two-rank NCCL all_reduce works",
            "PASS",
            "DDP across both T4s is reachable, which no CI tier can test",
        )
    else:
        record("a two-rank NCCL all_reduce works", "FAIL", output[-600:])


def main() -> int:
    """
    Run every phase and summarise.

    Each phase is independent and reports its own outcome, so one failure
    does not hide the rest -- the same reason the nightly's graph gate now
    reports each half as it finishes rather than at the end.

    Returns
    -------
    int
        0 if nothing failed, 1 otherwise.
    """
    phase_environment()
    if phase_install() is None:
        print("\nno usable k2; the later phases cannot run", flush=True)
    elif phase_import():
        phase_loss_agrees()
        phase_pruned_search()
        phase_arc_sorting()
        phase_multi_gpu()

    print("\n" + "=" * 72, flush=True)
    failed = [r for r in results if r[1] == "FAIL"]
    skipped = [r for r in results if r[1] == "SKIP"]
    print(
        f"{len(results) - len(failed) - len(skipped)} passed, "
        f"{len(failed)} failed, {len(skipped)} skipped",
        flush=True,
    )
    for phase, _status, detail in failed:
        print(f"  FAIL  {phase}: {detail}", flush=True)
    print("=" * 72, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
