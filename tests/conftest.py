"""Pytest configuration for optional k2-dependent tests."""

import importlib.util

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip tests marked 'k2' when k2 is not available."""
    del config  # unused
    has_k2 = importlib.util.find_spec("k2") is not None
    if has_k2:
        return

    skip_k2 = pytest.mark.skip(reason="k2 module not available")
    for item in items:
        if "k2" in item.keywords:
            item.add_marker(skip_k2)
