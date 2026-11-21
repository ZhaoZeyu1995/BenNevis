"""
An end-to-end training step with the network on MPS and the graph loss on the
CPU.

Everything else about MPS support is covered by unit tests that fake the
availability probes, so they run everywhere. This file is the integration: a
real `Trainer`, a real `GraphLoss`, a real optimiser step, on a real MPS
device. It therefore runs only on an Apple machine with k2 installed, and
skips elsewhere -- which is the whole reason the rest is tested by faking.

What it pins is the arrangement the unit tests cannot: that the DDP wrapper
being absent, the loss hopping to the CPU and the gradient hopping back all
compose into a step that actually changes the weights.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("k2")

pytestmark = [
    pytest.mark.torch,
    pytest.mark.k2,
    pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="no MPS device here"
    ),
]


@pytest.fixture
def mps_trainer(trainer_factory, model_factory, lang):
    """A Trainer on MPS with a real GraphLoss."""
    from BenNevis.core.losses import GraphLoss

    model = model_factory()
    trainer = trainer_factory(
        model=model,
        device=torch.device("mps"),
        loss_func=GraphLoss(lang, use_den=False, reduction="sum"),
    )
    # State that fit() would have set up.
    trainer.step = 0
    trainer.epoch = 0
    return trainer


class TestMpsTrainingStep:
    def test_the_model_is_on_mps_and_unwrapped(self, mps_trainer):
        assert mps_trainer.device.type == "mps"
        assert type(mps_trainer.model).__name__ != "DistributedDataParallel"
        param = next(mps_trainer.model.parameters())
        assert param.device.type == "mps"

    def test_a_training_step_returns_a_finite_loss(self, mps_trainer, make_batch):
        loss = mps_trainer._train_batch(make_batch())
        assert isinstance(loss, float)
        assert loss == loss  # not NaN
        assert abs(loss) != float("inf")

    def test_a_training_step_changes_the_weights(self, mps_trainer, make_batch):
        """
        The point of the whole arrangement: the gradient has to make it from a
        CPU-side FST intersection back to parameters living on MPS.
        """
        before = [p.detach().clone() for p in mps_trainer.model.parameters()]

        mps_trainer._train_batch(make_batch())

        after = list(mps_trainer.model.parameters())
        assert any(
            not torch.equal(b, a.detach()) for b, a in zip(before, after)
        ), "no parameter changed, so the gradient did not reach the model"

    def test_several_steps_stay_finite(self, mps_trainer, make_batch):
        losses = [mps_trainer._train_batch(make_batch(seed=seed)) for seed in range(3)]
        assert all(loss == loss and abs(loss) != float("inf") for loss in losses)

    def test_a_checkpoint_from_an_mps_run_loads_on_the_cpu(
        self, mps_trainer, make_batch, trainer_factory, model_factory, tmp_path
    ):
        """
        An MPS run is unwrapped and a CPU run is DDP-wrapped, so this crosses
        both the device and the wrapper boundary at once.
        """
        import os

        from omegaconf import OmegaConf

        mps_trainer._train_batch(make_batch())
        mps_trainer.ckpt_dir = str(tmp_path / "ckpts")
        os.makedirs(mps_trainer.ckpt_dir, exist_ok=True)
        mps_trainer.top_k, mps_trainer.epoch, mps_trainer.step = [], 0, 1
        mps_trainer.config = OmegaConf.create({"trainer": {"max_epochs": 1}})
        mps_trainer._save_checkpoint(0.5)
        saved = os.path.join(mps_trainer.ckpt_dir, "epoch_0_step_1_val_loss_0.5000.pt")

        on_cpu = trainer_factory(model=model_factory(), device=torch.device("cpu"))
        on_cpu._load_weights(saved)

        expected = mps_trainer._unwrapped_model.state_dict()
        got = on_cpu._unwrapped_model.state_dict()
        for key in expected:
            assert torch.allclose(got[key], expected[key].cpu(), atol=1e-6)
