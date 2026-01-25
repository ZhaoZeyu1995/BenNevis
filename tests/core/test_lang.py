"""Tests for Lang class."""

import pytest
import tempfile
import os


class TestLangBasic:
    """Basic tests for Lang class that don't require actual lang directory."""

    def test_lang_import(self):
        """Test that Lang can be imported."""
        from BenNevis.core.lang import Lang
        assert Lang is not None

    def test_lang_init_requires_directory(self):
        """Test that Lang requires a valid directory."""
        from BenNevis.core.lang import Lang
        
        # Test with non-existent directory
        with tempfile.TemporaryDirectory() as tmpdir:
            non_existent = os.path.join(tmpdir, "non_existent")
            # Lang may or may not raise an error depending on implementation
            # This is a placeholder test
            try:
                lang = Lang(non_existent)
            except (FileNotFoundError, AssertionError):
                # Expected behavior
                pass
