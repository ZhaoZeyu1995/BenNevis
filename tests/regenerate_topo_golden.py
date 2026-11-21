#!/usr/bin/env python3
"""
Regenerate the golden topology FSTs in tests/golden/topos/.

A golden file is the specification of a topology, not a cache. Run this only
when a topology's output is *intended* to change, and review the resulting
diff as carefully as you would review the change that caused it.

A note on provenance. These files were originally produced by the fifteen
`get_token_fst_*.py` scripts that used to define the topologies, and the
declarative specs in BenNevis/topos/specs/ were written to reproduce them byte
for byte. That is what makes them an independent check on the rewrite.
Regenerating them from the specs gives that up: from then on the specs and the
goldens can only agree. So prefer adding a golden for a *new* topology over
regenerating the existing ones, and if you must regenerate, read every line of
the diff.

Usage:
    python tests/regenerate_topo_golden.py

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from pathlib import Path

from BenNevis.topos import build, spec

TESTS_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = TESTS_DIR / "golden" / "topos"
PHONES = TESTS_DIR / "fixtures" / "phones.txt"


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    phones = build.read_phone_table(PHONES)
    for name in spec.available():
        topology = spec.load(name)
        for variant in sorted(topology.variants):
            label = name if variant == "lang" else f"{name}_{variant}"
            text = build.build_openfst_text(topology, variant, phones)
            (GOLDEN_DIR / f"{label}.fst.txt").write_text(text)
            print(f"{label}: {len(text.splitlines())} lines")


if __name__ == "__main__":
    main()
