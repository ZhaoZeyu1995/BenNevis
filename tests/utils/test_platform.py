"""
Tests for BenNevis.utils.platform.

Device detection is awkward to test because the answer depends on the machine
the tests run on. Every test here therefore fakes the availability probes, so
that all four platform configurations -- CUDA, MPS, CPU-only, and an explicit
request -- are exercised on every runner. The handful of tests that do depend
on the real machine are marked as such and assert only invariants that hold
everywhere.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""

import os
import warnings

import pytest

torch = pytest.importorskip("torch")

from BenNevis.utils import platform as bp  # noqa: E402

pytestmark = pytest.mark.torch


@pytest.fixture
def fake_platform(monkeypatch):
    """
    Fake the accelerator availability probes.

    Returns a function taking `cuda` and `mps` booleans, so each test states
    the configuration it is testing rather than inheriting the host's.
    """

    def _set(cuda: bool = False, mps: bool = False):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(bp, "_mps_is_available", lambda: mps)

    return _set


class TestAvailableAccelerators:
    """available_accelerators reports what torch can see, in preference order."""

    def test_cuda_and_mps_are_ordered_with_cuda_first(self, fake_platform):
        fake_platform(cuda=True, mps=True)
        assert bp.available_accelerators() == ["cuda", "mps"]

    def test_cpu_only_is_an_empty_list(self, fake_platform):
        fake_platform(cuda=False, mps=False)
        assert bp.available_accelerators() == []

    def test_mps_alone(self, fake_platform):
        fake_platform(cuda=False, mps=True)
        assert bp.available_accelerators() == ["mps"]


class TestResolveDeviceDetection:
    """Detection order and the local_rank handling."""

    def test_cuda_is_preferred_and_indexed_by_local_rank(self, fake_platform):
        fake_platform(cuda=True, mps=True)
        device = bp.resolve_device(local_rank=3, warn=False)
        assert device.type == "cuda"
        assert device.index == 3

    def test_mps_is_chosen_when_cuda_is_absent_and_allowed(self, fake_platform):
        fake_platform(cuda=False, mps=True)
        assert bp.resolve_device(warn=False).type == "mps"

    def test_mps_is_skipped_when_not_allowed(self, fake_platform):
        fake_platform(cuda=False, mps=True)
        assert bp.resolve_device(allow_mps=False, warn=False).type == "cpu"

    def test_cpu_is_the_fallback(self, fake_platform):
        fake_platform(cuda=False, mps=False)
        assert bp.resolve_device(warn=False).type == "cpu"

    def test_local_rank_does_not_leak_into_non_cuda_devices(self, fake_platform):
        fake_platform(cuda=False, mps=False)
        assert bp.resolve_device(local_rank=2, warn=False).index is None


class TestResolveDeviceRequested:
    """An explicit request is honoured, or refused -- never downgraded."""

    def test_explicit_cpu_is_honoured_even_with_cuda_present(self, fake_platform):
        fake_platform(cuda=True, mps=False)
        assert bp.resolve_device("cpu", warn=False).type == "cpu"

    def test_explicit_cuda_index_is_preserved(self, fake_platform):
        fake_platform(cuda=True, mps=False)
        device = bp.resolve_device("cuda:2", local_rank=0, warn=False)
        assert (device.type, device.index) == ("cuda", 2)

    def test_requesting_absent_cuda_raises(self, fake_platform):
        fake_platform(cuda=False, mps=True)
        # A run that asked for CUDA and quietly got the CPU would waste hours
        # before anyone noticed, so this must be an error.
        with pytest.raises(RuntimeError, match="no CUDA device"):
            bp.resolve_device("cuda", warn=False)

    def test_requesting_absent_mps_raises(self, fake_platform):
        fake_platform(cuda=False, mps=False)
        with pytest.raises(RuntimeError, match="no MPS device"):
            bp.resolve_device("mps", warn=False)

    def test_explicit_mps_overrides_allow_mps_false(self, fake_platform):
        fake_platform(cuda=False, mps=True)
        assert bp.resolve_device("mps", allow_mps=False, warn=False).type == "mps"


class TestResolveDeviceWarnings:
    """Every configuration that costs something must say so."""

    def test_cpu_fallback_warns_about_speed(self, fake_platform):
        fake_platform(cuda=False, mps=False)
        with pytest.warns(UserWarning, match="extremely slow"):
            bp.resolve_device()

    def test_mps_warns_about_the_cpu_graph_loss(self, fake_platform):
        fake_platform(cuda=False, mps=True)
        with pytest.warns(UserWarning, match="graph loss will be computed on the"):
            bp.resolve_device()

    def test_skipping_an_available_mps_explains_why(self, fake_platform):
        fake_platform(cuda=False, mps=True)
        with pytest.warns(UserWarning, match="DistributedDataParallel"):
            bp.resolve_device(allow_mps=False)

    def test_cuda_is_the_supported_path_and_warns_about_nothing(self, fake_platform):
        fake_platform(cuda=True, mps=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert bp.resolve_device().type == "cuda"

    def test_warn_false_silences_everything(self, fake_platform):
        fake_platform(cuda=False, mps=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert bp.resolve_device(warn=False).type == "cpu"


class TestResolveDistBackend:
    """NCCL is CUDA-only; gloo is the only option everywhere else."""

    @pytest.mark.parametrize(
        "device_str, expected",
        [
            ("cuda", "nccl"),
            ("cuda:1", "nccl"),
            ("cpu", "gloo"),
            ("mps", "gloo"),
        ],
    )
    def test_backend_matches_the_device(self, device_str, expected):
        assert bp.resolve_dist_backend(torch.device(device_str)) == expected


class TestSupportsDdp:
    """MPS does not implement the collectives DDP needs."""

    @pytest.mark.parametrize(
        "device_str, expected",
        [("cuda", True), ("cuda:0", True), ("cpu", True), ("mps", False)],
    )
    def test_ddp_support_by_device(self, device_str, expected):
        assert bp.supports_ddp(torch.device(device_str)) is expected


class TestGraphLossDevice:
    """k2 supports the CPU and CUDA only, and MPS has no float64."""

    def test_mps_sends_the_loss_to_the_cpu(self):
        assert bp.graph_loss_device(torch.device("mps")).type == "cpu"

    @pytest.mark.parametrize("device_str", ["cpu", "cuda", "cuda:2"])
    def test_other_devices_compute_the_loss_in_place(self, device_str):
        device = torch.device(device_str)
        assert bp.graph_loss_device(device) == device


class TestLinkOrCopy:
    """best.pt must end up readable on every platform."""

    def test_the_pointer_resolves_to_the_target_contents(self, tmp_path):
        target = tmp_path / "ckpt-1.pt"
        target.write_bytes(b"weights")
        link = tmp_path / "best.pt"

        bp.link_or_copy("ckpt-1.pt", str(link))

        assert link.exists()
        assert link.read_bytes() == b"weights"

    def test_an_existing_pointer_is_replaced(self, tmp_path):
        first = tmp_path / "ckpt-1.pt"
        first.write_bytes(b"old")
        second = tmp_path / "ckpt-2.pt"
        second.write_bytes(b"new")
        link = tmp_path / "best.pt"

        bp.link_or_copy("ckpt-1.pt", str(link))
        bp.link_or_copy("ckpt-2.pt", str(link))

        assert link.read_bytes() == b"new"

    def test_a_dangling_symlink_is_replaced(self, tmp_path):
        if not bp.supports_symlink():
            pytest.skip("platform has no symlinks, so none can dangle")
        link = tmp_path / "best.pt"
        os.symlink("gone.pt", link)
        assert not link.exists()  # dangling: exists() follows the link

        target = tmp_path / "ckpt-1.pt"
        target.write_bytes(b"weights")
        bp.link_or_copy("ckpt-1.pt", str(link))

        assert link.read_bytes() == b"weights"

    def test_the_copy_fallback_produces_a_real_file(self, tmp_path, monkeypatch):
        # Forced, so the Windows path is covered on POSIX runners too.
        monkeypatch.setattr(bp, "supports_symlink", lambda: False)
        target = tmp_path / "ckpt-1.pt"
        target.write_bytes(b"weights")
        link = tmp_path / "best.pt"

        bp.link_or_copy("ckpt-1.pt", str(link))

        assert not os.path.islink(link)
        assert link.read_bytes() == b"weights"

    def test_an_absolute_target_is_not_rebased(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bp, "supports_symlink", lambda: False)
        target = tmp_path / "elsewhere" / "ckpt-1.pt"
        target.parent.mkdir()
        target.write_bytes(b"weights")
        link = tmp_path / "best.pt"

        bp.link_or_copy(str(target), str(link))

        assert link.read_bytes() == b"weights"


class TestDescribePlatform:
    """The log line that makes a bug report answerable."""

    def test_it_names_the_system_python_and_torch(self):
        description = bp.describe_platform()
        assert torch.__version__ in description
        assert "Python" in description
        assert "accelerators" in description

    def test_cpu_only_is_stated_rather_than_left_blank(self, fake_platform):
        fake_platform(cuda=False, mps=False)
        assert "cpu only" in bp.describe_platform()


class TestRealMachine:
    """Invariants that must hold on whatever runner this is."""

    def test_the_detected_device_is_one_torch_can_allocate_on(self):
        device = bp.resolve_device(warn=False)
        tensor = torch.zeros(2, device=device)
        assert tensor.device.type == device.type

    def test_the_backend_for_the_detected_device_is_a_known_one(self):
        backend = bp.resolve_dist_backend(bp.resolve_device(warn=False))
        assert backend in ("nccl", "gloo")

    def test_symlink_support_matches_the_platform_family(self):
        assert bp.supports_symlink() == (os.name != "nt")
