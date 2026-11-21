"""
Tests for Trainer.

Trainer needs an initialised process group, so these run against a real
one-rank gloo group rather than stubbing the distributed calls out: the code
paths exercised are the ones that run in production. Before this was possible
Trainer was untestable, because `__init__` chose its device with
`torch.device(rank)` -- which resolves to CUDA on a training node but MPS on
an Apple machine -- and passed `device_ids` to DDP, which a CPU module
rejects.

The gradient accumulation tests are the valuable ones: that code was silently
wrong until recently, and nothing could have caught it.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import os

import pytest

torch = pytest.importorskip("torch")

from BenNevis.utils.platform import supports_symlink  # noqa: E402

pytestmark = pytest.mark.torch


class TestConstruction:
    """What Trainer decides at construction time."""

    def test_builds_on_cpu(self, trainer_factory):
        t = trainer_factory()
        assert t.device == torch.device("cpu")
        assert t.world_size == 1
        assert t.gpu_id == 0

    def test_device_is_explicit_not_inferred_from_rank(self, trainer_factory):
        """
        `torch.device(0)` resolves to the default accelerator, which is MPS on
        Apple machines, so the device must be chosen explicitly. Pinned by
        delegating to the resolver rather than by naming a device, since what
        is correct here depends on the machine.
        """
        from BenNevis.utils.platform import resolve_device

        t = trainer_factory(device=None)
        assert t.device == resolve_device(local_rank=0, warn=False)

    def test_detection_selects_mps_when_it_is_the_only_accelerator(
        self, trainer_factory, monkeypatch
    ):
        """
        MPS used to be excluded from detection, because DDP wrapped the model
        unconditionally and cannot wrap an MPS module. Now that wrapping is
        conditional, MPS is selectable.
        """
        import BenNevis.utils.platform as bp

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(bp, "_mps_is_available", lambda: True)
        if not bp._mps_is_available():  # pragma: no cover - sanity
            pytest.skip("monkeypatch did not take")
        if not torch.backends.mps.is_built():
            pytest.skip("this torch build cannot place a module on MPS")

        t = trainer_factory(device=None)
        assert t.device.type == "mps"

    def test_detection_indexes_cuda_by_local_rank(self, trainer_factory, monkeypatch):
        """
        The device index is LOCAL_RANK, not the global rank. Checked through
        the resolver rather than by building a Trainer, since this runner has
        no CUDA device to place a model on.
        """
        from BenNevis.utils.platform import resolve_device

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        device = resolve_device(local_rank=5, allow_mps=False, warn=False)
        assert (device.type, device.index) == ("cuda", 5)

    def test_device_can_be_overridden(self, trainer_factory):
        t = trainer_factory(device=torch.device("cpu"))
        assert t.device == torch.device("cpu")

    def test_the_model_is_moved_to_the_chosen_device(
        self, trainer_factory, model_factory
    ):
        """
        Trainer used to leave this to the caller. The entry points do it, so
        nothing failed in practice -- but a Trainer whose `device` disagreed
        with where its model lived raised a device mismatch from inside the
        first linear layer, far from the cause. Checked by spying on `.to`,
        since a CPU-only runner has no second device to move between.
        """
        model = model_factory()
        seen = []
        original_to = model.to

        def spy(*args, **kwargs):
            seen.append(args[0] if args else kwargs.get("device"))
            return original_to(*args, **kwargs)

        model.to = spy
        trainer_factory(model=model, device=torch.device("cpu"))

        assert seen == [torch.device("cpu")]

    def test_model_is_wrapped_in_ddp_even_at_world_size_one(self, trainer_factory):
        """
        Wrapping unconditionally keeps checkpoints interchangeable between
        single- and multi-process runs. Skipping DDP at world_size 1 was tried
        on an earlier branch and reverted.
        """
        t = trainer_factory()
        assert type(t.model).__name__ == "DistributedDataParallel"

    def test_the_wrapped_model_carries_the_ddp_prefix(self, trainer_factory):
        """
        DDP adds `module.` to every key. This is about the live model, not the
        on-disk format: checkpoints are written from the unwrapped module, so
        the prefix never reaches them.
        """
        t = trainer_factory()
        assert all(k.startswith("module.") for k in t.model.state_dict())

    def test_the_on_disk_format_has_no_ddp_prefix(self, trainer_factory):
        """
        Pins the on-disk format, which no longer records whether DDP was used.
        This is what makes conditional wrapping safe, and is exactly what an
        earlier attempt at it got wrong.
        """
        t = trainer_factory()
        assert not any(k.startswith("module.") for k in t._unwrapped_model.state_dict())


class TestSaveEveryNEpochsValidation:
    """The accepted values are documented; this pins them."""

    @pytest.mark.parametrize("value", [1, 2, 10, 0.5, 0.25, 1.0, 2.0])
    def test_accepted(self, trainer_factory, value):
        t = trainer_factory(save_every_n_epochs=value)
        assert t.save_every_n_epochs == value

    @pytest.mark.parametrize("value", [1.5, 2.5, -1, -0.5, "1"])
    def test_rejected(self, trainer_factory, value):
        with pytest.raises(ValueError):
            trainer_factory(save_every_n_epochs=value)

    def test_falsy_value_falls_back_to_step_based_saving(self, trainer_factory):
        t = trainer_factory(save_every_n_epochs=0, save_every_n_steps=100)
        assert t.save_every_n_epochs is None
        assert t.save_every_n_steps == 100

    def test_neither_set_defaults_to_every_epoch(self, trainer_factory):
        t = trainer_factory(save_every_n_epochs=None, save_every_n_steps=None)
        assert t.save_every_n_epochs == 1


class TestGradientAccumulation:
    """
    The loss must be divided by accum_grad_steps, or the effective learning
    rate scales with the accumulation factor.
    """

    def test_accumulating_the_same_batch_matches_a_single_step(
        self, trainer_factory, make_batch
    ):
        """
        Feeding one batch twice with accum_grad_steps=2 must leave the same
        gradient as feeding it once with accum_grad_steps=1. Without the
        division the accumulated gradient would be twice as large.
        """
        batch = make_batch()

        single = trainer_factory(accum_grad_steps=1)
        single.step = 0
        for opt in single.optimizers:
            opt.zero_grad(set_to_none=True)
        single_grads = self._grads_after(single, [batch])

        accum = trainer_factory(accum_grad_steps=2)
        accum.model.load_state_dict(single_grads["state"])
        accum.step = 0
        for opt in accum.optimizers:
            opt.zero_grad(set_to_none=True)
        accum_grads = self._grads_after(accum, [batch, batch], steps=[0])

        for a, b in zip(single_grads["grads"], accum_grads["grads"]):
            assert torch.allclose(a, b, rtol=1e-5, atol=1e-7)

    @staticmethod
    def _grads_after(trainer, batches, steps=None):
        """
        Run batches through _train_batch, capturing gradients before the
        optimiser consumes them.
        """
        state = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        captured = None
        for i, batch in enumerate(batches):
            trainer.step = i
            is_last = i == len(batches) - 1
            if is_last:
                # Capture before the step zeroes the gradients: patch the
                # optimiser step to snapshot first.
                grads = {}

                def snapshot(orig=trainer.optimizers[0].step):
                    grads["g"] = [
                        p.grad.clone()
                        for p in trainer.model.parameters()
                        if p.grad is not None
                    ]
                    return orig()

                trainer.optimizers[0].step = snapshot
            trainer._train_batch(batch)
            if is_last:
                captured = grads.get("g")
        assert captured, "the optimiser never stepped, so nothing was captured"
        return {"grads": captured, "state": state}

    def test_optimiser_steps_only_on_the_final_micro_batch(
        self, trainer_factory, make_batch
    ):
        batch = make_batch()
        t = trainer_factory(accum_grad_steps=3)
        calls = []
        original = t.optimizers[0].step
        t.optimizers[0].step = lambda *a, **k: (calls.append(1), original())[1]

        for step in range(6):
            t.step = step
            t._train_batch(batch)

        assert len(calls) == 2, "expected one step per three micro-batches"

    def test_single_step_accumulation_steps_every_batch(
        self, trainer_factory, make_batch
    ):
        batch = make_batch()
        t = trainer_factory(accum_grad_steps=1)
        calls = []
        original = t.optimizers[0].step
        t.optimizers[0].step = lambda *a, **k: (calls.append(1), original())[1]

        for step in range(4):
            t.step = step
            t._train_batch(batch)

        assert len(calls) == 4


class TestGradientClipping:
    """Clipping is applied when configured and skipped when not."""

    def test_clipping_is_applied_when_a_norm_is_set(
        self, trainer_factory, make_batch, monkeypatch
    ):
        seen = []
        monkeypatch.setattr(
            torch.nn.utils,
            "clip_grad_norm_",
            lambda params, norm: seen.append(norm),
        )
        t = trainer_factory(grad_max_norm=5.0)
        t.step = 0

        t._train_batch(make_batch())

        assert seen == [5.0]

    def test_clipping_is_skipped_by_default(
        self, trainer_factory, make_batch, monkeypatch
    ):
        """grad_max_norm defaults to None, so no recipe clips today."""
        seen = []
        monkeypatch.setattr(
            torch.nn.utils,
            "clip_grad_norm_",
            lambda params, norm: seen.append(norm),
        )
        t = trainer_factory()
        t.step = 0

        t._train_batch(make_batch())

        assert seen == []


class TestBatchSteps:
    """_train_batch and _valid_batch report an averaged loss."""

    def test_train_batch_returns_a_finite_loss(self, trainer_factory, make_batch):
        t = trainer_factory()
        t.step = 0
        value = t._train_batch(make_batch())
        assert isinstance(value, float)
        assert value == value  # not NaN

    def test_valid_batch_leaves_no_gradients(self, trainer_factory, make_batch):
        t = trainer_factory()
        t._valid_batch(make_batch())
        assert all(p.grad is None for p in t.model.parameters())

    def test_valid_batch_puts_the_model_in_eval_mode(self, trainer_factory, make_batch):
        t = trainer_factory()
        t._valid_batch(make_batch())
        assert not t.model.training

    def test_missing_inputs_are_reported(self, trainer_factory, make_batch):
        t = trainer_factory()
        batch = make_batch()
        del batch["feats"]
        with pytest.raises(ValueError, match="feats"):
            t._valid_batch(batch)


class TestCheckpointing:
    """_save_checkpoint's retention and best-pointer behaviour."""

    @pytest.fixture
    def ready(self, trainer_factory, tmp_path):
        """A Trainer with the state fit() would have set up."""

        def build(save_top_k=2):
            t = trainer_factory(save_top_k=save_top_k)
            t.ckpt_dir = str(tmp_path / "ckpts")
            os.makedirs(t.ckpt_dir, exist_ok=True)
            t.top_k = []
            t.epoch = 0
            t.step = 0
            from omegaconf import OmegaConf

            t.config = OmegaConf.create({"trainer": {"max_epochs": 1}})
            return t

        return build

    def test_saves_a_file_named_as_documented(self, ready):
        t = ready()
        t.epoch, t.step = 2, 50
        t._save_checkpoint(1.2345)
        assert os.path.exists(
            os.path.join(t.ckpt_dir, "epoch_2_step_50_val_loss_1.2345.pt")
        )

    def test_keeps_only_the_best_k(self, ready):
        t = ready(save_top_k=2)
        for i, loss in enumerate([3.0, 1.0, 2.0, 4.0]):
            t.epoch, t.step = i, i * 10
            t._save_checkpoint(loss)

        saved = [f for f in os.listdir(t.ckpt_dir) if f.endswith(".pt")]
        assert len([f for f in saved if f != "best.pt"]) == 2

    def test_best_pointer_resolves_to_the_lowest_loss(self, ready):
        t = ready(save_top_k=3)
        for i, loss in enumerate([3.0, 1.0, 2.0]):
            t.epoch, t.step = i, i * 10
            t._save_checkpoint(loss)

        # Asserted through the contents rather than os.readlink, because
        # best.pt is a copy rather than a symlink on platforms without them.
        # This is also the stronger assertion: it checks what a reader of
        # best.pt actually gets.
        best = torch.load(
            os.path.join(t.ckpt_dir, "best.pt"),
            map_location="cpu",
            weights_only=True,
        )
        assert best["VALID_LOSS"] == pytest.approx(1.0)

    @pytest.mark.skipif(
        not supports_symlink(), reason="platform does not support symlinks"
    )
    def test_best_is_a_symlink_where_the_platform_allows_one(self, ready):
        t = ready(save_top_k=3)
        for i, loss in enumerate([3.0, 1.0, 2.0]):
            t.epoch, t.step = i, i * 10
            t._save_checkpoint(loss)

        target = os.readlink(os.path.join(t.ckpt_dir, "best.pt"))
        assert "val_loss_1.0000" in target

    def test_round_trips_through_load(self, ready):
        t = ready()
        t.epoch, t.step = 7, 700
        t._save_checkpoint(0.5)
        path = os.path.join(t.ckpt_dir, "epoch_7_step_700_val_loss_0.5000.pt")

        t.epoch, t.step = 0, 0
        t._load_checkpoint(path)

        assert t.epoch == 7
        assert t.step == 700
        assert t.ckpt_config.trainer.max_epochs == 1


class TestConditionalDdp:
    """
    DDP is applied where it can be and skipped where it cannot.

    The conditional itself is tested by faking `supports_ddp`, so the logic is
    covered on every platform rather than only on an Apple machine.
    """

    @pytest.fixture
    def no_ddp(self, monkeypatch):
        """Make the trainer believe the device cannot take part in DDP."""
        import BenNevis.core.trainer as bt

        monkeypatch.setattr(bt, "supports_ddp", lambda device: False)

    def test_the_model_is_left_unwrapped_when_the_device_cannot_ddp(
        self, no_ddp, trainer_factory
    ):
        t = trainer_factory()
        assert type(t.model).__name__ != "DistributedDataParallel"
        assert t._unwrapped_model is t.model

    def test_an_unwrappable_device_with_several_ranks_is_an_error(
        self, no_ddp, trainer_factory, monkeypatch
    ):
        """
        Falling back to a single process silently would produce a run that
        looks distributed, trains on one shard of the data and reports a
        world_size it is not using.
        """
        import torch.distributed as dist

        monkeypatch.setattr(dist, "get_world_size", lambda *a, **k: 2)

        with pytest.raises(RuntimeError, match="cannot take part"):
            trainer_factory()

    def test_wrapping_still_happens_at_world_size_one_where_supported(
        self, trainer_factory
    ):
        """
        A single-process CPU run goes through the same code path as a
        distributed one, so the path is exercised by the ordinary tests.
        """
        t = trainer_factory()
        assert type(t.model).__name__ == "DistributedDataParallel"

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="no MPS device here"
    )
    def test_a_real_mps_module_is_not_wrapped(self, trainer_factory):
        t = trainer_factory(device=torch.device("mps"))
        assert t.device.type == "mps"
        assert type(t.model).__name__ != "DistributedDataParallel"


class TestCheckpointFormatIsWrapperAgnostic:
    """
    The property that makes conditional DDP safe: the on-disk format does not
    record whether the run that wrote it used DDP.

    An earlier attempt at conditional wrapping was reverted because it changed
    this format silently. These tests exist so that cannot recur.
    """

    @pytest.fixture
    def no_ddp(self, monkeypatch):
        import BenNevis.core.trainer as bt

        monkeypatch.setattr(bt, "supports_ddp", lambda device: False)

    @staticmethod
    def _weights(trainer):
        return {k: v.clone() for k, v in trainer._unwrapped_model.state_dict().items()}

    def test_a_ddp_checkpoint_loads_into_an_unwrapped_run(
        self, trainer_factory, monkeypatch, tmp_path
    ):
        wrapped = trainer_factory()
        expected = self._weights(wrapped)
        path = str(tmp_path / "ckpt.pt")
        torch.save({"MODEL_STATE": wrapped._unwrapped_model.state_dict()}, path)

        import BenNevis.core.trainer as bt

        monkeypatch.setattr(bt, "supports_ddp", lambda device: False)
        plain = trainer_factory()
        plain._load_weights(path)

        for key, value in expected.items():
            assert torch.equal(plain._unwrapped_model.state_dict()[key], value)

    def test_an_unwrapped_checkpoint_loads_into_a_ddp_run(
        self, trainer_factory, no_ddp, tmp_path, monkeypatch
    ):
        plain = trainer_factory()
        expected = self._weights(plain)
        path = str(tmp_path / "ckpt.pt")
        torch.save({"MODEL_STATE": plain._unwrapped_model.state_dict()}, path)

        import BenNevis.core.trainer as bt

        monkeypatch.setattr(bt, "supports_ddp", lambda device: True)
        wrapped = trainer_factory()
        wrapped._load_weights(path)

        for key, value in expected.items():
            assert torch.equal(wrapped._unwrapped_model.state_dict()[key], value)

    def test_a_legacy_prefixed_checkpoint_still_loads(self, trainer_factory, tmp_path):
        """
        Checkpoints written before this change stored the wrapper's keys. No
        such checkpoint is known to exist, but tolerating them costs one dict
        comprehension and removes a trap.
        """
        t = trainer_factory()
        legacy = {f"module.{k}": v for k, v in t._unwrapped_model.state_dict().items()}
        path = str(tmp_path / "legacy.pt")
        torch.save({"MODEL_STATE": legacy}, path)

        t._load_weights(path)  # must not raise

    def test_a_saved_checkpoint_round_trips_both_ways(
        self, trainer_factory, tmp_path, monkeypatch
    ):
        """End to end through _save_checkpoint, not a hand-built dict."""
        from omegaconf import OmegaConf

        t = trainer_factory()
        t.ckpt_dir = str(tmp_path / "ckpts")
        os.makedirs(t.ckpt_dir, exist_ok=True)
        t.top_k, t.epoch, t.step = [], 1, 10
        t.config = OmegaConf.create({"trainer": {"max_epochs": 1}})
        t._save_checkpoint(0.25)
        saved = os.path.join(t.ckpt_dir, "epoch_1_step_10_val_loss_0.2500.pt")

        stored = torch.load(saved, map_location="cpu", weights_only=True)
        assert not any(k.startswith("module.") for k in stored["MODEL_STATE"])

        import BenNevis.core.trainer as bt

        monkeypatch.setattr(bt, "supports_ddp", lambda device: False)
        plain = trainer_factory()
        plain._load_weights(saved)


class TestStripDdpPrefix:
    """The prefix is recognised by DDP's signature, not by guessing."""

    def test_a_fully_prefixed_dict_is_stripped(self):
        from BenNevis.core.trainer import Trainer

        out = Trainer._strip_ddp_prefix({"module.a": 1, "module.b": 2})
        assert out == {"a": 1, "b": 2}

    def test_an_unprefixed_dict_is_untouched(self):
        from BenNevis.core.trainer import Trainer

        assert Trainer._strip_ddp_prefix({"a": 1}) == {"a": 1}

    def test_a_partial_match_is_left_alone(self):
        """
        A model with a submodule genuinely named `module` alongside others
        must not be mangled. Only an all-keys match is DDP's signature.
        """
        from BenNevis.core.trainer import Trainer

        state = {"module.weight": 1, "encoder.weight": 2}
        assert Trainer._strip_ddp_prefix(state) == state

    def test_an_empty_dict_is_handled(self):
        from BenNevis.core.trainer import Trainer

        assert Trainer._strip_ddp_prefix({}) == {}
