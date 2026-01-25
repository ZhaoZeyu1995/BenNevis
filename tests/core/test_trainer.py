"""Tests for Trainer class."""

import pytest


class TestTrainerBasic:
    """Basic tests for Trainer class."""

    def test_trainer_import(self):
        """Test that Trainer can be imported."""
        from BenNevis.core.trainer import Trainer
        assert Trainer is not None
        
    # Note: Cannot test initialization or methods without distributed setup
    # Trainer requires torch.distributed.init_process_group()
    # which is typically done via torchrun in production
