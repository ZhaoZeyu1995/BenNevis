"""
Print the dependency list the Docker image should install.

The image needs BenNevis's dependencies without BenNevis itself: the package
is not baked in, because the container runs against a mounted working tree.
That makes `pyproject.toml` the one place dependencies are declared, rather
than a list in the Dockerfile that silently drifts from it.

`torch` and `torchaudio` are excluded. They are pinned to an exact version in
the Dockerfile, because the k2 wheel is built against one specific torch
build and installing an unpinned torch afterwards would quietly replace it.

Usage
-----
    python3 docker/deps.py pyproject.toml

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

#: Installed separately, pinned to the version the k2 wheel was built for.
PINNED_ELSEWHERE = ("torch", "torchaudio")

#: Needed to run the tests inside the container but not to use the library,
#: so it is not a runtime dependency of the package.
EXTRA = ("pytest",)


def requirement_name(requirement: str) -> str:
    """
    The bare distribution name of a PEP 508 requirement string.

    Arguments
    ---------
    requirement: str
        A requirement as `pyproject.toml` lists it, such as `"torch"` or
        `"pytest>=7.0"`.

    Returns
    -------
    str
        The distribution name, lowercased, with any version specifier,
        extras or environment marker removed.
    """
    name = requirement.strip()
    for separator in ("[", ";", "=", "<", ">", "!", "~", " "):
        name = name.split(separator)[0]
    return name.strip().lower()


def dependencies(pyproject: Path) -> List[str]:
    """
    The dependencies to install in the image, in declaration order.

    Arguments
    ---------
    pyproject: Path
        The project's `pyproject.toml`.

    Returns
    -------
    List[str]
        Requirement strings, with the separately pinned ones removed and the
        test runner appended.
    """
    # tomllib is standard from 3.11; the image's Python is newer than that,
    # and so is the floor in pyproject.toml.
    import tomllib

    with open(pyproject, "rb") as handle:
        data = tomllib.load(handle)

    declared = data.get("project", {}).get("dependencies", [])
    kept = [
        requirement
        for requirement in declared
        if requirement_name(requirement) not in PINNED_ELSEWHERE
    ]
    return kept + list(EXTRA)


def main(argv=None) -> int:
    """
    Print one requirement per line.

    Arguments
    ---------
    argv: Optional[List[str]]
        The arguments, defaulting to sys.argv[1:].

    Returns
    -------
    int
        The process exit status.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        sys.stderr.write("usage: python3 docker/deps.py <pyproject.toml>\n")
        return 2
    for requirement in dependencies(Path(argv[0])):
        sys.stdout.write(f"{requirement}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
