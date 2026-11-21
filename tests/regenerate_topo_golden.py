#!/usr/bin/env python3
"""
Regenerate the golden topology FSTs in tests/golden/topos/.

The generators in BenNevis/bin/topos/ emit OpenFST text for a given phone
symbol table. Their output defines the semantics of each topology, so it is
captured here and asserted by tests/bin/test_topos.py.

Run this only when a topology's output is *intended* to change, and review
the resulting diff as carefully as you would review the generator change:
a golden file is the specification of a topology, not a cache.

Usage:
    python tests/regenerate_topo_golden.py

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import subprocess
import sys
from pathlib import Path
from typing import List

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
TOPO_DIR = REPO_ROOT / "BenNevis" / "bin" / "topos"
GOLDEN_DIR = TESTS_DIR / "golden" / "topos"
PHONES = TESTS_DIR / "fixtures" / "phones.txt"


def generator_paths() -> List[Path]:
    """Return every topology generator, sorted by name."""
    return sorted(TOPO_DIR.glob("get_token_fst_*.py"))


def run_generator(script: Path) -> str:
    """Run one generator against the fixture phone table and return its stdout."""
    result = subprocess.run(
        [sys.executable, str(script), str(PHONES)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for script in generator_paths():
        name = script.stem.replace("get_token_fst_", "")
        out = run_generator(script)
        (GOLDEN_DIR / f"{name}.fst.txt").write_text(out)
        print(f"{name}: {len(out.splitlines())} lines")


if __name__ == "__main__":
    main()
