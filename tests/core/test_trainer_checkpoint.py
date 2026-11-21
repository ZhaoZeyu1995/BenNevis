"""
Tests for Trainer's checkpoint loading.

torch changed `torch.load`'s default to `weights_only=True` in 2.6, which
refuses to unpickle arbitrary Python objects. BenNevis used to store the live
Hydra configuration object in the checkpoint, so loading one raised
UnpicklingError on any current torch -- breaking resuming and prediction.
The configuration is now stored as a YAML string instead.

`Trainer` cannot be constructed here: its `__init__` calls `dist.get_rank()`
and wraps the model in DDP with explicit `device_ids`, so it needs an
initialised process group and CUDA. The methods under test are therefore
called unbound, against a stub carrying only the attributes they touch. That
exercises the real implementation rather than a reimplementation of it.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import pytest

torch = pytest.importorskip("torch")
omegaconf = pytest.importorskip("omegaconf")

from omegaconf import OmegaConf  # noqa: E402

from BenNevis.core.trainer import Trainer  # noqa: E402

pytestmark = pytest.mark.torch


class _TrainerStub:
    """
    The minimum surface Trainer's checkpoint methods touch.

    Deliberately not a Trainer: constructing one requires a process group and
    CUDA. The methods under test are borrowed from the real class, so this
    exercises the real implementation; only the state it reads is stubbed.
    Holding that state explicitly also documents what those methods depend on.
    """

    # Borrowed, not reimplemented.
    _torch_load = Trainer._torch_load
    _load_checkpoint = Trainer._load_checkpoint
    _load_weights = Trainer._load_weights
    _load_model_state = Trainer._load_model_state
    _unwrapped_model = Trainer._unwrapped_model
    # Re-wrapped: accessing a staticmethod on the class unwraps it to a
    # plain function, which would then rebind as an instance method here.
    _strip_ddp_prefix = staticmethod(Trainer._strip_ddp_prefix)

    def __init__(self):
        self.gpu_id = 0
        self.device = torch.device("cpu")
        self.model = torch.nn.Linear(4, 3)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.optimizers = [opt]
        self.schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(opt)]
        self.config = OmegaConf.create(
            {
                "model": {"name": "RNNPModel", "kwargs": {"odim": 42}},
                "trainer": {"max_epochs": 10},
            }
        )


@pytest.fixture
def stub():
    return _TrainerStub()


def snapshot(stub, config):
    """Build a checkpoint shaped exactly as _save_checkpoint writes one."""
    snap = {
        "CONFIG": config,
        "MODEL_STATE": stub.model.state_dict(),
        "EPOCH": 3,
        "STEP": 120,
        "VALID_LOSS": 1.2345,
    }
    for i, opt in enumerate(stub.optimizers):
        snap[f"OPTIMIZER_{i}"] = opt.state_dict()
    for i, sched in enumerate(stub.schedulers):
        snap[f"SCHEDULER_{i}"] = sched.state_dict()
    return snap


class TestTorchLoad:
    """_torch_load must use the safe path and fail legibly when it cannot."""

    def test_loads_a_checkpoint_whose_config_is_yaml(self, stub, tmp_path):
        path = tmp_path / "ckpt.pt"
        torch.save(snapshot(stub, OmegaConf.to_yaml(stub.config)), path)

        ckpt = stub._torch_load(str(path))

        assert isinstance(ckpt["CONFIG"], str)
        assert ckpt["EPOCH"] == 3
        assert ckpt["STEP"] == 120

    def test_optimiser_and_scheduler_state_survive_the_safe_path(self, stub, tmp_path):
        """
        The point of storing the config as text: everything else in the
        checkpoint is already safe-loadable, so nothing has to be given up.
        """
        path = tmp_path / "ckpt.pt"
        torch.save(snapshot(stub, OmegaConf.to_yaml(stub.config)), path)

        ckpt = stub._torch_load(str(path))

        assert "param_groups" in ckpt["OPTIMIZER_0"]
        assert ckpt["OPTIMIZER_0"]["param_groups"][0]["lr"] == pytest.approx(1e-3)
        assert "best" in ckpt["SCHEDULER_0"]

    def test_pickled_config_raises_a_message_naming_the_cause(self, stub, tmp_path):
        """
        A checkpoint holding the live config object cannot load safely. The
        error must explain that, not surface a bare UnpicklingError.
        """
        path = tmp_path / "legacy.pt"
        torch.save(snapshot(stub, stub.config), path)

        with pytest.raises(RuntimeError, match="weights_only=True"):
            stub._torch_load(str(path))

    def test_missing_file_also_reports_through_the_wrapper(self, stub, tmp_path):
        with pytest.raises(RuntimeError):
            stub._torch_load(str(tmp_path / "absent.pt"))


class TestLoadCheckpoint:
    """_load_checkpoint restores training state without clobbering the config."""

    def test_restores_epoch_step_and_loss(self, stub, tmp_path):
        path = tmp_path / "ckpt.pt"
        torch.save(snapshot(stub, OmegaConf.to_yaml(stub.config)), path)

        stub._load_checkpoint(str(path))

        assert stub.epoch == 3
        assert stub.step == 120
        assert stub.valid_loss == pytest.approx(1.2345)

    def test_checkpoint_config_is_parsed_back_into_a_config(self, stub, tmp_path):
        path = tmp_path / "ckpt.pt"
        torch.save(snapshot(stub, OmegaConf.to_yaml(stub.config)), path)

        stub._load_checkpoint(str(path))

        assert stub.ckpt_config.model.kwargs.odim == 42
        assert stub.ckpt_config.trainer.max_epochs == 10

    def test_live_config_is_not_overwritten_by_the_checkpoint(self, stub, tmp_path):
        """
        Resuming with a changed setting must keep the change. The checkpoint's
        configuration is provenance, not the configuration to run with.
        """
        path = tmp_path / "ckpt.pt"
        torch.save(snapshot(stub, OmegaConf.to_yaml(stub.config)), path)
        stub.config = OmegaConf.create(
            {
                "model": {"name": "RNNPModel", "kwargs": {"odim": 42}},
                "trainer": {"max_epochs": 999},
            }
        )

        stub._load_checkpoint(str(path))

        assert stub.config.trainer.max_epochs == 999, "live config was clobbered"
        assert stub.ckpt_config.trainer.max_epochs == 10, "provenance was lost"


class TestLoadWeights:
    """_load_weights shares the same loader, so it gets the same safety."""

    def test_loads_model_state_only(self, stub, tmp_path):
        path = tmp_path / "ckpt.pt"
        torch.save(snapshot(stub, OmegaConf.to_yaml(stub.config)), path)
        with torch.no_grad():
            stub.model.weight.fill_(99.0)

        stub._load_weights(str(path))

        assert not torch.allclose(
            stub.model.weight, torch.full_like(stub.model.weight, 99.0)
        ), "weights were not replaced"
        assert not hasattr(stub, "epoch"), "_load_weights must not restore progress"


def test_trainer_does_not_require_k2(import_without):
    """
    Trainer only mentions GraphLoss in a type annotation, so k2 must not be
    needed to import it. Keeping it out means the trainer can be tested
    without the k2 wheel, which is tied to an exact torch version.
    """
    import_without("BenNevis.core.trainer", "k2")
