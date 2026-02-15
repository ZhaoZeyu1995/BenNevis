"""Tests for BenNevis package initialization."""

import importlib.util

import pytest

import BenNevis  # noqa: E402


def test_version():
    """Test that version is defined."""
    assert hasattr(BenNevis, "__version__")
    assert isinstance(BenNevis.__version__, str)
    assert BenNevis.__version__ == "0.1.0"


def test_author():
    """Test that author is defined."""
    assert hasattr(BenNevis, "__author__")
    assert BenNevis.__author__ == "Zeyu Zhao"


def test_exports():
    """Test that main classes are exported when k2 is available."""
    has_k2 = importlib.util.find_spec("k2") is not None
    exported = ("Trainer", "Dataset", "CollateFunc", "Lang", "GraphLoss")

    if has_k2:
        for name in exported:
            assert hasattr(BenNevis, name)
    else:
        for name in exported:
            with pytest.raises(ModuleNotFoundError):
                getattr(BenNevis, name)


def test_all_list():
    """Test that __all__ is properly defined."""
    assert hasattr(BenNevis, "__all__")
    assert "Trainer" in BenNevis.__all__
    assert "Dataset" in BenNevis.__all__
    assert "CollateFunc" in BenNevis.__all__
    assert "Lang" in BenNevis.__all__
    assert "GraphLoss" in BenNevis.__all__
