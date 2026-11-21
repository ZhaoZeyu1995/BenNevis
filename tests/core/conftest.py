"""
Fixtures for the k2-dependent core tests.

Building a usable `Lang` without a Kaldi or OpenFST installation needs two
tricks, both worth knowing about:

- `Lang.load_topo` shells out to `fstprint` to turn a binary `k2/T.fst` into
  text, so a topology cannot be loaded without OpenFST on PATH. These tests
  construct `lang.topo` directly instead. (Removing that shell-out is one of
  the motivations for replacing the topology generators with declarative
  specs built straight into k2.)
- `Lang.load_lexicon` reads `k2/L_inv.pt` with `torch.load` when it exists and
  only falls back to `fstprint` on `L.fst` otherwise, so writing the `.pt`
  avoids OpenFST entirely.

k2 is imported inside the fixtures rather than at module scope, so collecting
the dependency-free tiers does not require it.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TOPO_DIR = REPO_ROOT / "BenNevis" / "bin" / "topos"

# A deliberately tiny symbol set with a one-to-one word-to-phone lexicon, so
# that a phone sequence and a word sequence are interchangeable and the loss
# can be compared against torch's CTC implementation directly.
PHONES = "<eps> 0\na 1\nb 2\nc 3\n"
TOKENS = "<blk> 0\na 1\nb 2\nc 3\n"  # no <eps>: matches the network's outputs
WORDS = "<eps> 0\nA 1\nB 2\nC 3\n"
ALIGN_LEXICON = "1 1 1\n2 2 2\n3 3 3\n"  # word_id word_id phone_id...

PHONE_IDS = {"<eps>": 0, "a": 1, "b": 2, "c": 3}
TOKEN_IDS = {"<blk>": 0, "a": 1, "b": 2, "c": 3}
NUM_TOKENS = 4


def openfst_text_to_int(text: str, isym: dict, osym: dict) -> str:
    """
    Rewrite OpenFST text with symbol names into one with integer ids.

    The topology generators print symbol names, which the real pipeline feeds
    to `fstcompile --isymbols=... --osymbols=...`. `k2.Fsa.from_openfst`
    requires integers, so this performs the same mapping in Python and lets a
    generator's output reach k2 without OpenFST.

    Arguments
    ---------
    text: str
        The generator's output.
    isym: dict
        Input symbol table, mapping name to id.
    osym: dict
        Output symbol table, mapping name to id.

    Returns
    -------
    text: str
        The same FST with integer labels and an explicit zero weight.
    """
    lines = []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) == 1:  # final state
            lines.append(f"{fields[0]} 0.0")
        elif len(fields) == 4:  # src dst ilabel olabel
            src, dst, ilabel, olabel = fields
            lines.append(f"{src} {dst} {isym[ilabel]} {osym[olabel]} 0.0")
        else:
            raise AssertionError(f"unexpected FST line: {line!r}")
    return "\n".join(lines) + "\n"


def generate_topo(name: str, phones_path: Path) -> str:
    """Run one topology generator and return its OpenFST text."""
    return subprocess.run(
        [sys.executable, str(TOPO_DIR / f"get_token_fst_{name}.py"), str(phones_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture
def lang_dir(tmp_path):
    """
    Write a minimal but complete lang directory and return its path.

    Contains everything `Lang` reads, including a prebuilt `k2/L_inv.pt` so
    that the lexicon loads without OpenFST.
    """
    import k2
    import torch

    (tmp_path / "k2").mkdir()
    (tmp_path / "phones").mkdir()
    (tmp_path / "phones.txt").write_text(PHONES)
    (tmp_path / "k2" / "phones.txt").write_text(PHONES)
    (tmp_path / "k2" / "tokens.txt").write_text(TOKENS)
    (tmp_path / "words.txt").write_text(WORDS)
    (tmp_path / "phones" / "align_lexicon.int").write_text(ALIGN_LEXICON)

    # L_inv: words on the input side, phones as aux labels, as Lang expects
    # after inverting Kaldi's phones-to-words L.fst.
    arcs = "\n".join(f"0 0 {i} {i} 0.0" for i in (1, 2, 3)) + "\n0 0.0\n"
    l_inv = k2.arc_sort(k2.Fsa.from_openfst(arcs, acceptor=False))
    torch.save(l_inv.as_dict(), os.path.join(tmp_path, "k2", "L_inv.pt"))
    return tmp_path


@pytest.fixture
def lang(lang_dir):
    """
    A `Lang` with a CTC topology, built from `k2.ctc_topo`.

    `load_topo=False` because loading one from disk requires OpenFST; the
    topology is assigned directly instead.
    """
    import k2

    from BenNevis.core.lang import Lang

    obj = Lang(str(lang_dir), load_topo=False, load_lexicon=True)
    obj.topo = k2.ctc_topo(NUM_TOKENS - 1)
    return obj


@pytest.fixture
def generated_ctc_topo(lang_dir):
    """
    The CTC topology as produced by the project's own generator.

    Used to check the generator against `k2.ctc_topo`, which is the oracle the
    declarative rewrite will have to satisfy.
    """
    import k2

    text = generate_topo("ctc", lang_dir / "phones.txt")
    return k2.arc_sort(
        k2.Fsa.from_openfst(
            openfst_text_to_int(text, TOKEN_IDS, PHONE_IDS), acceptor=False
        )
    )


@pytest.fixture(scope="session")
def process_group(tmp_path_factory):
    """
    A real single-process gloo process group, for the duration of the session.

    `Trainer` requires an initialised process group: it reads the rank and
    world size in `__init__` and all-reduces losses. Rather than stub those
    out, the tests run against a genuine one-rank group, so the distributed
    code paths are the ones actually exercised.

    A FileStore is used rather than TCP so that nothing depends on a free
    port, which is a common source of flakiness in CI.
    """
    import torch.distributed as dist

    if dist.is_initialized():  # pragma: no cover - defensive
        yield
        return

    store_path = tmp_path_factory.mktemp("pg") / "store"
    store = dist.FileStore(str(store_path), 1)
    dist.init_process_group(backend="gloo", store=store, rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


class StubLoss:
    """
    Stands in for GraphLoss without needing k2.

    Returns a differentiable scalar derived from the log probabilities, so
    gradients flow and accumulation behaviour can be observed, while keeping
    these tests in the torch tier rather than the k2 tier.
    """

    def __init__(self, reduction: str = "sum"):
        self.reduction = reduction
        self.calls = 0

    def __call__(self, log_probs, log_prob_lens, word_ids, target_lengths):
        self.calls += 1
        loss = -log_probs.sum()
        if self.reduction == "mean":
            loss = loss / target_lengths.sum()
        return loss


@pytest.fixture
def model_factory():
    """
    Build a model with the interface Trainer expects: it is called as
    `model(inputs, input_lens)` and returns `(log_probs, log_prob_lens)`.
    """
    import torch

    class TinyModel(torch.nn.Module):
        def __init__(self, idim=4, odim=3):
            super().__init__()
            self.proj = torch.nn.Linear(idim, odim)

        def forward(self, inputs, input_lens):
            return self.proj(inputs).log_softmax(-1), input_lens

    return TinyModel


@pytest.fixture
def make_batch():
    """Build a batch shaped as CollateFunc produces one."""
    import torch

    def build(batch_size=2, frames=5, idim=4, seed=0):
        torch.manual_seed(seed)
        return {
            "feats": torch.randn(batch_size, frames, idim),
            "feats_lens": torch.full((batch_size,), frames, dtype=torch.int32),
            "target_lengths": torch.full((batch_size,), 3, dtype=torch.int32),
            "word_ids": [[1, 2, 3]] * batch_size,
            "batch_size": batch_size,
        }

    return build


@pytest.fixture
def trainer_factory(process_group, model_factory, tmp_path):
    """
    Build a real `Trainer` on CPU with sensible defaults for a test.

    Deliberately a real Trainer, not a stub: the point of making the device
    and DDP wrapping explicit was to allow exactly this.
    """
    import torch

    from BenNevis.core.trainer import Trainer

    def build(**kwargs):
        model = kwargs.pop("model", None) or model_factory()
        opts = kwargs.pop("optimizers", None)
        if opts is None:
            opts = [torch.optim.SGD(model.parameters(), lr=0.1)]
        kwargs.setdefault("loss_func", StubLoss("mean"))
        kwargs.setdefault("exp_dir", str(tmp_path / "exp"))
        return Trainer(
            model=model,
            optimizers=opts,
            schedulers=kwargs.pop("schedulers", []),
            **kwargs,
        )

    return build
