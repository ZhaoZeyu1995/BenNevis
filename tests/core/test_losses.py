"""Tests for GraphLoss class."""

import pytest


class TestGraphLossBasic:
    """Basic tests for GraphLoss class."""

    def test_graphloss_import(self):
        """Test that GraphLoss can be imported."""
        from BenNevis.core.losses import GraphLoss
        assert GraphLoss is not None

    def test_graphloss_is_module(self):
        """Test that GraphLoss is a PyTorch Module."""
        from BenNevis.core.losses import GraphLoss
        import torch.nn as nn
        
        assert issubclass(GraphLoss, nn.Module)
