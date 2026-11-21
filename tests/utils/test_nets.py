"""
Tests for BenNevis.utils.nets.

`sinusoids` used to live in BenNevis/models/whisper.py, which imports the
`whisper` package at module scope. models/transformer.py imported it from
there, so the Transformer model could not be used without openai-whisper
installed -- for the sake of one positional-embedding function that has
nothing to do with Whisper. It now lives here, and
test_transformer_does_not_require_the_whisper_package pins that.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import pytest

torch = pytest.importorskip("torch")

from BenNevis.utils.nets import sinusoids  # noqa: E402

pytestmark = pytest.mark.torch


class TestSinusoids:
    """Shape and value properties of the positional embedding table."""

    @pytest.mark.parametrize(
        "length,channels",
        [(4, 8), (1500, 512), (3000, 1024), (10000, 768)],
    )
    def test_shape(self, length, channels):
        assert sinusoids(length, channels).shape == (length, channels)

    def test_odd_channels_rejected(self):
        """Half the channels carry sines and half cosines, so even is required."""
        with pytest.raises(AssertionError):
            sinusoids(10, 7)

    def test_values_are_bounded(self):
        out = sinusoids(100, 64)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_first_position_is_sin0_then_cos0(self):
        """At position 0 every sine is 0 and every cosine is 1."""
        out = sinusoids(8, 16)
        assert torch.allclose(out[0, :8], torch.zeros(8))
        assert torch.allclose(out[0, 8:], torch.ones(8))

    def test_halves_are_sin_and_cos_of_the_same_argument(self):
        """sin^2 + cos^2 == 1 channel-wise, which pins the layout convention."""
        out = sinusoids(32, 32)
        half = out.shape[1] // 2
        total = out[:, :half] ** 2 + out[:, half:] ** 2
        assert torch.allclose(total, torch.ones_like(total), atol=1e-6)

    def test_positions_are_distinct(self):
        """A positional embedding that repeats cannot encode position."""
        out = sinusoids(64, 64)
        assert len({tuple(row.tolist()) for row in out}) == 64

    def test_deterministic(self):
        assert torch.equal(sinusoids(16, 16), sinusoids(16, 16))

    def test_longer_table_extends_rather_than_rescales(self):
        """
        Whisper's encoder is given a longer table to accept inputs beyond its
        native length, which only works if row i does not depend on `length`.
        """
        short, long = sinusoids(100, 64), sinusoids(10000, 64)
        assert torch.equal(short, long[:100])


def test_transformer_does_not_require_the_whisper_package(import_without):
    """
    The Transformer model has no relationship to Whisper and must not inherit
    a dependency on it for the sake of one positional-embedding function.
    """
    import_without("BenNevis.models.transformer", "whisper")
