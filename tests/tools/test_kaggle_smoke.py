"""
Tests for the one non-trivial pure function in `tools/kaggle/smoke.py`.

`unsorted_copy` exists so the smoke test can ask whether CUDA tolerates an
unsorted graph -- a question `CLAUDE.md` has had to leave open, because no
tier of CI runs on a GPU. The answer is only worth having if the
"unsorted" FST really is the same FST with its arcs reordered, and nothing
else: a reversal that dropped a final-state line, or that reordered arcs
across states, would produce a *different* FST and the comparison would
then measure that difference instead of the arc order.

Those are text properties, so they are checked here in tier 1 rather than
on the GPU where the mistake would be invisible. The rest of `smoke.py`
needs a CUDA device and is not testable anywhere in this repository.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import sys
from pathlib import Path

import pytest

KAGGLE_DIR = Path(__file__).resolve().parents[2] / "tools" / "kaggle"

if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))

smoke = pytest.importorskip("smoke")


# Two states with several arcs each, then two final states. The arc order
# within a state is what must reverse; everything else must not move.
FST = (
    "0 1 1 1 0.0\n"
    "0 2 2 2 0.0\n"
    "0 3 3 3 0.0\n"
    "1 1 1 0 0.0\n"
    "1 2 2 2 0.0\n"
    "2\n"
    "1\n"
)


def arc_lines(text: str):
    """
    The arc lines of an OpenFST text, in order.

    Arguments
    ---------
    text: str
        OpenFST text.

    Returns
    -------
    list
        The lines with four or more fields.
    """
    return [line for line in text.splitlines() if len(line.split()) >= 4]


def final_lines(text: str):
    """
    The final-state lines of an OpenFST text, in order.

    Arguments
    ---------
    text: str
        OpenFST text.

    Returns
    -------
    list
        The lines with fewer than four fields.
    """
    return [
        line for line in text.splitlines() if line.strip() and len(line.split()) < 4
    ]


class TestUnsortedCopy:
    """`unsorted_copy` reorders arcs within a state and changes nothing else."""

    def test_it_keeps_exactly_the_same_lines(self):
        """No line is added, dropped or rewritten -- only moved."""
        out = smoke.unsorted_copy(FST)
        assert sorted(out.splitlines()) == sorted(
            line for line in FST.splitlines() if line.strip()
        )

    def test_it_reverses_the_arcs_of_each_state(self):
        """State 0's three arcs and state 1's two come back in reverse."""
        out = arc_lines(smoke.unsorted_copy(FST))
        assert out[:3] == ["0 3 3 3 0.0", "0 2 2 2 0.0", "0 1 1 1 0.0"]
        assert out[3:] == ["1 2 2 2 0.0", "1 1 1 0 0.0"]

    def test_it_does_not_move_arcs_between_states(self):
        """Each state's arcs stay contiguous and in the original state order."""
        sources = [line.split()[0] for line in arc_lines(smoke.unsorted_copy(FST))]
        assert sources == ["0", "0", "0", "1", "1"]

    def test_the_final_states_stay_at_the_end_in_order(self):
        """
        `k2.Fsa.from_openfst` reads the final states from the trailing
        lines, so a reversal that interleaved them with arcs would change
        which states are final -- a different FST, not a reordered one.
        """
        out = smoke.unsorted_copy(FST)
        assert final_lines(out) == ["2", "1"]
        assert out.splitlines()[-2:] == ["2", "1"]

    def test_it_is_an_involution_on_this_fst(self):
        """Reversing twice restores the original, which pins the grouping."""
        once = smoke.unsorted_copy(FST)
        twice = smoke.unsorted_copy(once)
        assert twice == "".join(
            f"{line}\n" for line in FST.splitlines() if line.strip()
        )

    def test_it_tolerates_blank_lines(self):
        """A trailing or stray newline must not become an arc or a final state."""
        out = smoke.unsorted_copy("0 1 1 1 0.0\n\n1\n\n")
        assert out == "0 1 1 1 0.0\n1\n"

    def test_a_single_arc_state_is_unchanged(self):
        """
        The degenerate case matters: if the only state had one arc the
        "unsorted" copy would equal the sorted one and the GPU comparison
        would silently compare a graph with itself.
        """
        assert smoke.unsorted_copy("0 1 1 1 0.0\n1\n") == "0 1 1 1 0.0\n1\n"


class TestTheKernelIsConfiguredForWhatItChecks:
    """The metadata must ask Kaggle for what the script needs."""

    def test_the_metadata_requests_a_gpu_and_internet(self):
        """
        Without a GPU every phase after the first skips, and without
        internet the k2 CUDA wheel cannot be downloaded -- in both cases
        the run reports a tidy failure that says nothing about the stack.
        """
        import json

        meta = json.loads((KAGGLE_DIR / "kernel-metadata.json").read_text())
        assert meta["enable_gpu"] is True
        assert meta["enable_internet"] is True
        assert meta["kernel_type"] == "script"
        assert meta["code_file"] == "smoke.py"

    def test_the_pinned_versions_match_the_image_and_ci(self):
        """
        `smoke.py` reports a mismatch against the pinned torch rather than
        reinstalling it, so the pin it compares against has to be the one
        `docker/Dockerfile` uses. If they drift, the smoke test would call
        a correct image wrong.
        """
        dockerfile = (KAGGLE_DIR.parents[1] / "docker" / "Dockerfile").read_text()
        assert f"torch{smoke.PINNED_TORCH}" in dockerfile
        assert smoke.PINNED_K2_BUILD in dockerfile
