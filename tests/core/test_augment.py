"""
SpecAugment, and the four ways a mask can be quietly wrong.

Tier 1b: it needs torch and nothing else.

Augmentation is unusually easy to get wrong without noticing, because
every failure mode produces a model that trains to a *slightly* worse
WER and nothing else. The four pinned here are the ones that produce no
error and no visible symptom:

* a mask that lands in the **padding**, which does nothing at all. The
  samplers group utterances of similar but not identical length, so this
  weakens the augmentation in proportion to how ragged each batch is;
* a mask applied during **validation**, which makes the validation loss
  wander with the mask draw rather than with the model -- and the
  mid-epoch save schedule and the WER ledger both read that loss;
* a mask that covers the **whole** utterance, which trains the model to
  predict from nothing;
* an **in-place** write, which would leave the caller holding masked
  features and a second reader seeing them.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import pytest

torch = pytest.importorskip("torch")

from BenNevis.core.augment import SpecAugment  # noqa: E402

pytestmark = pytest.mark.torch


def ones(batch=4, frames=200, channels=80):
    return torch.ones(batch, frames, channels)


def freq_only(**kwargs):
    settings = dict(
        num_freq_masks=1, freq_mask_width=10, num_time_masks=0, time_mask_width=0
    )
    settings.update(kwargs)
    return SpecAugment(**settings).train()


def time_only(**kwargs):
    settings = dict(
        num_freq_masks=0, freq_mask_width=0, num_time_masks=1, time_mask_width=20
    )
    settings.update(kwargs)
    return SpecAugment(**settings).train()


class TestItMasksAtAll:
    def test_something_is_masked_over_many_draws(self):
        """
        A width of zero is a legitimate draw -- the paper specifies
        uniform over `[0, W]` -- so one call proving nothing is expected
        and many calls proving nothing is not.
        """
        augment = freq_only()
        masked = sum(int((augment(ones(batch=1)) == 0.0).any()) for _ in range(50))
        assert masked > 25, f"only {masked} of 50 draws masked anything"

    def test_the_mask_value_is_written(self):
        augment = freq_only(mask_value=-7.5)
        for _ in range(50):
            out = augment(ones(batch=1))
            if (out == -7.5).any():
                return
        pytest.fail("no draw wrote the mask value")

    def test_shape_and_dtype_survive(self):
        out = SpecAugment().train()(ones())
        assert out.shape == (4, 200, 80)
        assert out.dtype == torch.float32

    def test_zero_masks_change_nothing(self):
        augment = SpecAugment(num_freq_masks=0, num_time_masks=0).train()
        features = ones()
        assert torch.equal(augment(features), features)


class TestItIsOffOutsideTraining:
    def test_eval_returns_the_input_unchanged(self):
        """
        The validation loss must depend on the model, not on the mask
        draw. `Trainer._valid_batch` calls `model.eval()` before
        computing a loss, so this gate is what keeps the loss comparable
        across epochs -- and the mid-epoch save schedule reads it.
        """
        augment = SpecAugment(
            num_freq_masks=4, freq_mask_width=40, num_time_masks=4, time_mask_width=100
        ).eval()
        features = ones()
        out = augment(features)
        assert out is features, "eval should return the very same tensor"

    def test_training_mode_does_mask(self):
        augment = SpecAugment(
            num_freq_masks=4, freq_mask_width=40, num_time_masks=4, time_mask_width=100
        ).train()
        assert (augment(ones()) == 0.0).any()


class TestTimeMasksRespectTheLengths:
    def test_nothing_inside_the_padding_is_touched(self):
        """
        The one that would be invisible: a batch padded to 200 frames
        whose utterances are 20 frames long. A mask drawn over the padded
        width would mostly land where nothing is read.
        """
        features = ones(batch=8, frames=200)
        lengths = torch.full((8,), 20, dtype=torch.int32)
        augment = time_only(num_time_masks=4, time_mask_width=100)
        for _ in range(40):
            out = augment(features, lengths)
            # Padding must be exactly as it came in.
            assert torch.equal(out[:, 20:, :], features[:, 20:, :])

    def test_a_mask_lands_inside_the_valid_frames(self):
        features = ones(batch=1, frames=400)
        lengths = torch.tensor([100], dtype=torch.int32)
        augment = time_only(num_time_masks=3, time_mask_width=30)
        touched = False
        for _ in range(60):
            out = augment(features, lengths)
            if (out[0, :100] == 0.0).any():
                touched = True
                break
        assert touched, "no draw masked anything inside the valid frames"

    def test_without_lengths_the_padded_width_is_used(self):
        """
        Documented behaviour rather than desirable behaviour: the
        trainer always passes lengths, and this records what happens if a
        caller does not.
        """
        augment = time_only(num_time_masks=4, time_mask_width=50)
        hit = any(
            bool((augment(ones(batch=1, frames=200))[0, 150:] == 0.0).any())
            for _ in range(60)
        )
        assert hit

    def test_a_length_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="entries for"):
            time_only()(ones(batch=4), torch.tensor([10, 10], dtype=torch.int32))


class TestTheRatioCap:
    def test_a_short_utterance_is_never_wiped_out(self):
        """
        A 100-frame time mask on a 60-frame utterance would remove it
        entirely. The cap is a fraction of *that* utterance.
        """
        features = ones(batch=1, frames=60)
        lengths = torch.tensor([60], dtype=torch.int32)
        augment = time_only(
            num_time_masks=1, time_mask_width=100, max_time_mask_ratio=0.2
        )
        for _ in range(100):
            out = augment(features, lengths)
            masked = int((out[0, :60, 0] == 0.0).sum())
            assert masked <= 12, f"{masked} of 60 frames masked, cap is 20%"

    def test_the_cap_scales_with_the_utterance(self):
        """
        The same setting on a long utterance must allow a wider mask, or
        the cap would be a fixed width in disguise.
        """
        augment = time_only(
            num_time_masks=1, time_mask_width=100, max_time_mask_ratio=0.2
        )
        widest = 0
        for _ in range(200):
            out = augment(ones(batch=1, frames=500), torch.tensor([500]))
            widest = max(widest, int((out[0, :, 0] == 0.0).sum()))
        assert 40 < widest <= 100, widest

    def test_a_width_wider_than_the_cap_is_bounded_by_it(self):
        augment = time_only(
            num_time_masks=1, time_mask_width=1000, max_time_mask_ratio=0.1
        )
        for _ in range(100):
            out = augment(ones(batch=1, frames=300), torch.tensor([300]))
            assert int((out[0, :, 0] == 0.0).sum()) <= 30


class TestFrequencyMasks:
    def test_a_mask_is_contiguous_in_channels(self):
        augment = freq_only(num_freq_masks=1, freq_mask_width=20)
        for _ in range(60):
            out = augment(ones(batch=1, frames=5))
            zeros = (out[0, 0] == 0.0).nonzero().flatten()
            if zeros.numel() > 1:
                assert torch.equal(
                    zeros, torch.arange(int(zeros[0]), int(zeros[-1]) + 1)
                ), "a frequency mask is not contiguous"
                return
        pytest.fail("no draw produced a mask of more than one channel")

    def test_the_same_channels_are_masked_at_every_frame(self):
        """
        A frequency mask removes a band for the whole utterance; masking
        a different band per frame would be a different transform and
        much weaker.
        """
        augment = freq_only(num_freq_masks=1, freq_mask_width=20)
        for _ in range(60):
            out = augment(ones(batch=1, frames=20))
            per_frame = out[0] == 0.0
            if per_frame.any():
                assert torch.equal(
                    per_frame[0].expand(20, -1), per_frame
                ), "the masked band varies by frame"
                return
        pytest.fail("no draw masked anything")

    def test_a_mask_never_exceeds_the_channel_count(self):
        augment = freq_only(num_freq_masks=2, freq_mask_width=200)
        out = augment(ones(batch=2, frames=4, channels=8))
        assert out.shape[-1] == 8


class TestItDoesNotWriteInPlace:
    def test_the_input_is_untouched(self):
        """
        An in-place write would leave the caller holding masked features,
        and `Trainer._compute_loss` is handed the batch the DataLoader
        still owns.
        """
        augment = SpecAugment(
            num_freq_masks=4, freq_mask_width=40, num_time_masks=4, time_mask_width=100
        ).train()
        features = ones()
        reference = features.clone()
        augment(features, torch.full((4,), 200, dtype=torch.int32))
        assert torch.equal(features, reference)


class TestItRefusesNonsense:
    @pytest.mark.parametrize(
        "options, message",
        [
            (dict(num_freq_masks=-1), "must not be negative"),
            (dict(num_time_masks=-1), "must not be negative"),
            (dict(freq_mask_width=-1), "must not be negative"),
            (dict(time_mask_width=-1), "must not be negative"),
            (dict(max_time_mask_ratio=0.0), "in \\(0, 1\\]"),
            (dict(max_time_mask_ratio=1.5), "in \\(0, 1\\]"),
        ],
    )
    def test_bad_settings(self, options, message):
        with pytest.raises(ValueError, match=message):
            SpecAugment(**options)

    def test_a_two_dimensional_batch_is_refused(self):
        with pytest.raises(ValueError, match="batch, frames, channels"):
            SpecAugment().train()(torch.ones(10, 80))


class TestTheTrainerAppliesItOnlyWhereItShould:
    """
    Read out of `trainer.py` rather than executed: reaching
    `_compute_loss` needs a process group, a language directory, a model
    and a loss. What matters is *where* the call sits, and that is
    checkable as text.
    """

    @staticmethod
    def source():
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent.parent
        return (root / "BenNevis" / "core" / "trainer.py").read_text(encoding="utf-8")

    def test_it_is_applied_before_the_model(self):
        text = self.source()
        call = text.index("self.spec_augment(inputs, input_lens)")
        forward = text.index("outputs = self.model(inputs, input_lens)")
        assert call < forward

    def test_it_is_applied_only_to_features(self):
        """
        `Wav2Vec2Model` is given a waveform, where masking mel channels
        is meaningless.
        """
        assert 'self.spec_augment is not None and "feats" in batch' in self.source()

    def test_it_is_off_by_default(self):
        """
        Every recipe trained without it before, so a run that does not
        ask for it must behave as it did.
        """
        assert "spec_augment: Optional[Dict[str, Any]] = None" in self.source()
        assert "self.spec_augment = None" in self.source()
