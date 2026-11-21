"""
Declarative topology specifications.

A topology used to be a Python script that printed OpenFST text. There were
fifteen of them, they shared most of their logic by copy and paste, and adding
one meant writing another script. A topology is really a small amount of data:
how many states each phone gets, which arcs connect them, and how the token
inventory is derived from the phone inventory. This module reads that data from
a YAML file instead.

Each file describes one topology *family* and holds two variants:

- `lang`, which produces `T.fst`. Its input symbols include `<eps>` and the
  disambiguation symbols, so it composes with Kaldi's `L_disambig.fst`.
- `k2`, which produces `k2/T.fst`. Its input symbols are exactly the neural
  network's outputs: no `<eps>`, no disambiguation symbols. It is
  non-deterministic where the `lang` variant is epsilon-ful, which is the
  trade the old `*_k2.py` scripts made and is still the right one -- removing
  the epsilons with `fstrmepsilon` instead produces a far larger FST.

States are referred to symbolically. `hub` is state 0, the state every phone
leaves from and returns to, and `s0`, `s1`, ... are the states belonging to
the phone currently being expanded. The expansion assigns each phone a
contiguous block of `stride` state ids, so `s0` for the second phone in a
2-state topology is state 3.

Labels are templates. `{phone}` is the phone being expanded, and in
`inter_phone_arcs` `{dst_phone}` is the phone owning the destination block.
Anything else, such as `<eps>` or `<blk>`, is a literal.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SPEC_DIR = Path(__file__).resolve().parent / "specs"

EPS = "<eps>"
BLANK = "<blk>"
HUB = "hub"

#: How a variant treats the disambiguation symbols (`#0`, `#1`, ...).
#: `self_loop` gives each one a `#N:#N` self-loop on the hub, which is what
#: lets `T.fst` compose with `L_disambig.fst`. `skip` omits them, which is
#: what the network-facing variant needs, since they are not network outputs.
DISAMBIG_SELF_LOOP = "self_loop"
DISAMBIG_SKIP = "skip"

#: Which states are final. `hub` makes only state 0 final. `hub_and_bases`
#: additionally makes the first state of every phone block final, which CTC
#: needs so that an utterance may end mid-token.
FINALS_HUB = "hub"
FINALS_HUB_AND_BASES = "hub_and_bases"

_STATE_RE = re.compile(r"^s(\d+)$")


def _parse_state(value: str, stride: int, where: str) -> Optional[int]:
    """
    Turn a symbolic state reference into an offset within a phone block.

    Arguments
    ---------
    value: str
        Either "hub" or "sN".
    stride: int
        How many states a phone block has, used to reject out-of-range
        references at load time rather than producing a silently wrong FST.
    where: str
        A description of the arc, for the error message.

    Returns
    -------
    Optional[int]
        None for the hub, otherwise the offset within the block.
    """
    if value == HUB:
        return None
    match = _STATE_RE.match(value)
    if match is None:
        raise ValueError(
            f"{where}: state reference {value!r} is neither 'hub' nor 'sN'"
        )
    offset = int(match.group(1))
    if offset >= stride:
        raise ValueError(
            f"{where}: state reference {value!r} is outside the phone block, "
            f"which has stride {stride} (so s0..s{stride - 1})"
        )
    return offset


@dataclass(frozen=True)
class Arc:
    """
    One arc of a topology, with states given as offsets within a phone block.

    Attributes
    ----------
    src: Optional[int]
        The source state: None for the hub, otherwise an offset.
    dst: Optional[int]
        The destination state, likewise.
    ilabel: str
        The input label template.
    olabel: str
        The output label template.
    """

    src: Optional[int]
    dst: Optional[int]
    ilabel: str
    olabel: str

    @classmethod
    def from_list(cls, raw: Sequence[str], stride: int, where: str) -> "Arc":
        """
        Build an arc from its YAML form, `[src, dst, ilabel, olabel]`.

        Arguments
        ---------
        raw: Sequence[str]
            The four fields as written in the spec.
        stride: int
            The variant's stride, for validating state references.
        where: str
            A description used in error messages.

        Returns
        -------
        Arc
            The parsed arc.
        """
        if len(raw) != 4:
            raise ValueError(
                f"{where}: an arc needs exactly 4 fields "
                f"[src, dst, ilabel, olabel], got {len(raw)}: {raw!r}"
            )
        src, dst, ilabel, olabel = raw
        return cls(
            src=_parse_state(str(src), stride, where),
            dst=_parse_state(str(dst), stride, where),
            ilabel=str(ilabel),
            olabel=str(olabel),
        )


@dataclass(frozen=True)
class Variant:
    """
    One of the two FSTs a topology family produces.

    Attributes
    ----------
    stride: int
        How many state ids each phone entry consumes. This is not always the
        number of states a phone actually uses: `3state-skip` labels three
        tokens but occupies only two states, so its stride is 2.
    arcs: Tuple[Arc, ...]
        The per-phone arcs, emitted in this order for each phone in turn.
    disambig: str
        DISAMBIG_SELF_LOOP or DISAMBIG_SKIP.
    disambig_advances: bool
        Whether a disambiguation entry consumes `stride` state ids even though
        it adds no states. This only matters for a phone table whose
        disambiguation symbols are not last, which is not how Kaldi writes
        them -- but the original scripts differed on it, so it is recorded
        rather than guessed.
    finals: str
        FINALS_HUB or FINALS_HUB_AND_BASES.
    inter_phone_arcs: Tuple[Arc, ...]
        Arcs between *different* phone blocks, emitted after all phone blocks,
        for every ordered pair of distinct blocks, destination-major. CTC uses
        these to allow a direct transition from one token to another without
        passing through the hub.
    """

    stride: int
    arcs: Tuple[Arc, ...]
    disambig: str = DISAMBIG_SELF_LOOP
    disambig_advances: bool = True
    finals: str = FINALS_HUB
    inter_phone_arcs: Tuple[Arc, ...] = ()


@dataclass(frozen=True)
class TopologySpec:
    """
    A topology family: a token-derivation rule plus the two FST variants.

    Attributes
    ----------
    name: str
        The topology name, as `prepare_topo.sh --topo` takes it.
    summary: str
        A human-readable sketch of the state machine, carried over from the
        docstrings of the scripts this replaces.
    tokens_per_phone: Tuple[str, ...]
        Templates for the tokens one phone contributes. CTC's is `("{phone}",)`
        -- the phone is its own token -- while the multi-state topologies use
        `("{phone}_0", "{phone}_1")` and so on. This determines the neural
        network's output vocabulary, so it must agree with the arcs' input
        labels; `validate` checks that it does.
    blank: bool
        Whether the hub carries a `<blk>:<eps>` self-loop. Always true for the
        topologies in this repository; present because a topology without a
        blank is a coherent thing to want.
    skip_phones: Tuple[str, ...]
        Phone-table entries that contribute nothing and consume no states.
    variants: Dict[str, Variant]
        Keyed by "lang" and "k2".
    """

    name: str
    summary: str
    tokens_per_phone: Tuple[str, ...]
    variants: Dict[str, Variant]
    blank: bool = True
    skip_phones: Tuple[str, ...] = (EPS, "<SIL>")
    _source: Optional[Path] = field(default=None, compare=False, repr=False)

    @property
    def source(self) -> Path:
        """
        The file this spec was loaded from.

        Returns
        -------
        Path
            The spec file.

        Raises
        ------
        ValueError
            If the spec was built in memory rather than loaded from a file.
        """
        if self._source is None:
            raise ValueError(
                f"topology {self.name!r} was not loaded from a file, so it has "
                f"no source path"
            )
        return self._source

    def variant(self, which: str) -> Variant:
        """
        Look up a variant by name, with a legible error if it is missing.

        Arguments
        ---------
        which: str
            "lang" or "k2".

        Returns
        -------
        Variant
            The requested variant.
        """
        try:
            return self.variants[which]
        except KeyError:
            raise KeyError(
                f"topology {self.name!r} has no {which!r} variant; it defines "
                f"{sorted(self.variants)}"
            ) from None

    def validate(self) -> None:
        """
        Check the spec is internally consistent.

        Catches the mistakes that would otherwise produce a topology whose
        token inventory and FST disagree -- which trains a network with the
        wrong number of outputs, and fails far from the cause.
        """
        if not self.variants:
            raise ValueError(f"topology {self.name!r} defines no variants")
        if not self.tokens_per_phone:
            raise ValueError(
                f"topology {self.name!r} derives no tokens from a phone, so "
                f"the network would have nothing to predict"
            )

        allowed = set(self.tokens_per_phone) | {EPS, BLANK}
        for which, variant in self.variants.items():
            where = f"topology {self.name!r}, variant {which!r}"
            if variant.stride < 1:
                raise ValueError(f"{where}: stride must be at least 1")
            if variant.disambig not in (DISAMBIG_SELF_LOOP, DISAMBIG_SKIP):
                raise ValueError(
                    f"{where}: disambig must be {DISAMBIG_SELF_LOOP!r} or "
                    f"{DISAMBIG_SKIP!r}, not {variant.disambig!r}"
                )
            if variant.finals not in (FINALS_HUB, FINALS_HUB_AND_BASES):
                raise ValueError(
                    f"{where}: finals must be {FINALS_HUB!r} or "
                    f"{FINALS_HUB_AND_BASES!r}, not {variant.finals!r}"
                )
            if not variant.arcs:
                raise ValueError(f"{where}: no per-phone arcs")

            for arc in variant.arcs:
                if arc.ilabel not in allowed:
                    raise ValueError(
                        f"{where}: input label {arc.ilabel!r} is not one of "
                        f"the tokens this topology derives from a phone "
                        f"({sorted(allowed)}). The token table and the FST "
                        f"would disagree."
                    )
            if which == "k2":
                eps_inputs = [a for a in variant.arcs if a.ilabel == EPS]
                if eps_inputs:
                    raise ValueError(
                        f"{where}: the k2 variant must be epsilon-free on the "
                        f"input side, because its input symbols are the "
                        f"network's outputs, but {len(eps_inputs)} arc(s) have "
                        f"an <eps> input label"
                    )

    def tokens_for_phone(self, phone: str) -> List[str]:
        """
        The tokens one phone contributes, in order.

        Arguments
        ---------
        phone: str
            The phone.

        Returns
        -------
        List[str]
            The token names.
        """
        return [template.format(phone=phone) for template in self.tokens_per_phone]


def _require(mapping: dict, key: str, where: str):
    """Fetch a required key, naming the file in the error."""
    if key not in mapping:
        raise ValueError(f"{where}: missing required key {key!r}")
    return mapping[key]


def load_spec(path: Path) -> TopologySpec:
    """
    Load and validate one topology spec from a YAML file.

    Arguments
    ---------
    path: Path
        The spec file.

    Returns
    -------
    TopologySpec
        The parsed, validated spec.
    """
    import yaml

    where = str(path)
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected a mapping at the top level")

    variants: Dict[str, Variant] = {}
    raw_variants = _require(raw, "variants", where)
    for which, body in raw_variants.items():
        vwhere = f"{where} [{which}]"
        stride = int(_require(body, "stride", vwhere))
        if stride < 1:
            raise ValueError(f"{vwhere}: stride must be at least 1, got {stride}")
        variants[which] = Variant(
            stride=stride,
            arcs=tuple(
                Arc.from_list(arc, stride, vwhere)
                for arc in _require(body, "arcs", vwhere)
            ),
            disambig=body.get("disambig", DISAMBIG_SELF_LOOP),
            disambig_advances=bool(body.get("disambig_advances", True)),
            finals=body.get("finals", FINALS_HUB),
            inter_phone_arcs=tuple(
                Arc.from_list(arc, stride, f"{vwhere} inter_phone_arcs")
                for arc in body.get("inter_phone_arcs", [])
            ),
        )

    spec = TopologySpec(
        name=str(_require(raw, "name", where)),
        summary=str(raw.get("summary", "")).rstrip(),
        tokens_per_phone=tuple(_require(raw, "tokens_per_phone", where)),
        variants=variants,
        blank=bool(raw.get("blank", True)),
        skip_phones=tuple(raw.get("skip_phones", [EPS, "<SIL>"])),
        _source=path,
    )
    spec.validate()
    return spec


def spec_path(name: str) -> Path:
    """
    The spec file for a topology name.

    Arguments
    ---------
    name: str
        The topology name, such as "ctc" or "3state-skip-1".

    Returns
    -------
    Path
        The file's path, which may not exist.
    """
    return SPEC_DIR / f"{name}.yaml"


def load(name: str) -> TopologySpec:
    """
    Load a topology by name from the bundled specs.

    Arguments
    ---------
    name: str
        The topology name.

    Returns
    -------
    TopologySpec
        The parsed, validated spec.
    """
    path = spec_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"no topology named {name!r}; available: {', '.join(available())}"
        )
    return load_spec(path)


def available() -> List[str]:
    """
    The names of all bundled topologies, sorted.

    Returns
    -------
    List[str]
        The topology names.
    """
    return sorted(p.stem for p in SPEC_DIR.glob("*.yaml"))
