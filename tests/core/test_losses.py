"""Tests for GraphLoss class."""


class TestGraphLossBasic:
    """Basic tests for GraphLoss class."""

    def test_graphloss_import(self):
        """Test that GraphLoss can be imported."""
        from BenNevis.core.losses import GraphLoss

        assert GraphLoss is not None

    def test_graphloss_is_module(self):
        """Test that GraphLoss is a PyTorch Module."""
        import torch.nn as nn

        from BenNevis.core.losses import GraphLoss

        assert issubclass(GraphLoss, nn.Module)
