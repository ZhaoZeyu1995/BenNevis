"""Tests for utility functions."""

import pytest

# Skip all tests if k2 is not available (e.g., in CI without CUDA)
pytest.importorskip("k2", reason="k2 module not available (requires CUDA)")


class TestRandomUtils:
    """Tests for random utility functions."""

    def test_random_import(self):
        """Test that random utils can be imported."""
        from BenNevis.utils import random

        assert random is not None

    def test_setup_seed_exists(self):
        """Test that setup_seed function exists."""
        from BenNevis.utils.random import setup_seed

        assert callable(setup_seed)


class TestMiscUtils:
    """Tests for misc utility functions."""

    def test_misc_import(self):
        """Test that misc utils can be imported."""
        from BenNevis.utils import misc

        assert misc is not None

    def test_dynamic_import_exists(self):
        """Test that dynamic_import function exists."""
        from BenNevis.utils.misc import dynamic_import

        assert callable(dynamic_import)
