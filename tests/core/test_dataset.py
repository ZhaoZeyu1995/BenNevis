"""Tests for Dataset class."""

import pytest

# Skip all tests if k2 is not available (e.g., in CI without CUDA)
pytest.importorskip("k2", reason="k2 module not available (requires CUDA)")


class TestDatasetBasic:
    """Basic tests for Dataset class."""

    def test_dataset_import(self):
        """Test that Dataset can be imported."""
        from BenNevis.core.dataset import CollateFunc, Dataset

        assert Dataset is not None
        assert CollateFunc is not None

    def test_dataset_is_torch_dataset(self):
        """Test that Dataset is a PyTorch Dataset."""
        import torch.utils.data

        from BenNevis.core.dataset import Dataset

        assert issubclass(Dataset, torch.utils.data.Dataset)
