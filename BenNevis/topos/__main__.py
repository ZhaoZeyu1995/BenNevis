"""
Command line interface to the topology specifications.

Replaces the fifteen `get_token_fst_*.py` scripts with one entry point that
takes the topology by name:

    python -m BenNevis.topos fst ctc data/lang/phones.txt
    python -m BenNevis.topos fst 2state --variant k2 data/lang/k2/phones.txt
    python -m BenNevis.topos tokens 2state data/lang/phones.txt
    python -m BenNevis.topos list

`fst` writes OpenFST text on stdout, exactly as the old scripts did, so it
drops into the existing `fstcompile` pipelines unchanged.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from BenNevis.topos import build, spec


def _add_phones(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "phones",
        type=Path,
        help="A Kaldi-style phones.txt. Only the symbol column is read.",
    )


def main(argv=None) -> int:
    """
    Run the CLI.

    Arguments
    ---------
    argv: Optional[List[str]]
        The arguments, defaulting to sys.argv[1:].

    Returns
    -------
    int
        The process exit status.
    """
    parser = argparse.ArgumentParser(
        prog="python -m BenNevis.topos",
        description="Expand a declarative topology into an FST or a symbol table.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fst = sub.add_parser("fst", help="print the topology FST as OpenFST text")
    fst.add_argument("topo", help="topology name, e.g. ctc or 3state-skip-1")
    _add_phones(fst)
    fst.add_argument(
        "--variant",
        default="lang",
        choices=["lang", "k2"],
        help=(
            "lang (default) for T.fst, whose input symbols include <eps> and "
            "the disambiguation symbols; k2 for k2/T.fst, whose input symbols "
            "are the network's outputs"
        ),
    )

    tokens = sub.add_parser("tokens", help="print a token symbol table")
    tokens.add_argument("topo", help="topology name")
    _add_phones(tokens)
    tokens.add_argument(
        "--no-eps",
        action="store_false",
        dest="with_eps",
        help="omit <eps>, as k2/tokens.txt does",
    )
    tokens.add_argument(
        "--no-disambig",
        action="store_false",
        dest="with_disambig",
        help="omit the disambiguation symbols, as tokens.txt does",
    )

    phones = sub.add_parser("phones", help="print a phone symbol table")
    _add_phones(phones)
    phones.add_argument(
        "--no-disambig",
        action="store_false",
        dest="with_disambig",
        help="omit the disambiguation symbols, as k2/phones.txt does",
    )

    disambig = sub.add_parser(
        "disambig",
        help="print the ids of the disambiguation symbols in tokens_disambig.txt",
    )
    disambig.add_argument("topo", help="topology name")
    _add_phones(disambig)

    sub.add_parser("list", help="list the available topologies")

    described = sub.add_parser("describe", help="print a topology's summary")
    described.add_argument("topo", help="topology name")

    args = parser.parse_args(argv)

    if args.command == "list":
        for name in spec.available():
            sys.stdout.write(f"{name}\n")
        return 0

    if args.command == "describe":
        topology = spec.load(args.topo)
        sys.stdout.write(f"{topology.name}\n\n{topology.summary}\n\n")
        sys.stdout.write(f"tokens per phone: {', '.join(topology.tokens_per_phone)}\n")
        sys.stdout.write(f"variants: {', '.join(sorted(topology.variants))}\n")
        return 0

    if args.command == "phones":
        symbols = build.read_phone_table(args.phones)
        sys.stdout.write(
            build.write_symbol_table(
                build.build_phone_symbols(symbols, with_disambig=args.with_disambig)
            )
        )
        return 0

    topology = spec.load(args.topo)
    symbols = build.read_phone_table(args.phones)

    if args.command == "fst":
        sys.stdout.write(build.build_openfst_text(topology, args.variant, symbols))
        return 0

    if args.command == "disambig":
        ids = build.symbol_ids(build.build_tokens(topology, symbols))
        for phone in symbols:
            if build.is_disambig(phone):
                sys.stdout.write(f"{ids[phone]}\n")
        return 0

    sys.stdout.write(
        build.write_symbol_table(
            build.build_tokens(
                topology,
                symbols,
                with_eps=args.with_eps,
                with_disambig=args.with_disambig,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
