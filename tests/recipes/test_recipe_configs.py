"""
Check every recipe's optimiser configuration against the installed torch.

A recipe config names a class and a dictionary of keyword arguments, and
`run/train.py` passes the second to the first. Nothing verifies the two agree
until a training run starts, which for a real corpus is a long way in. When
torch removes a parameter the config becomes a crash that only a full recipe
run can find: `ReduceLROnPlateau` lost `verbose`, and all four recipes in
this repository carried it, so every one of them was broken against current
torch while every test passed.

This reads the configs and compares them with the signatures of the classes
they name. It is a second, not a recipe, and it would have caught that.

What it cannot check is whether a value is sensible -- only that the argument
exists. That is the right boundary: a wrong learning rate is a research
decision, a nonexistent argument is a defect.

Marked `torch`, since the whole point is to compare against the installed
torch rather than against a list written down here.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import inspect
from pathlib import Path

import pytest

pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

pytestmark = pytest.mark.torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EGS = REPO_ROOT / "egs"


def opt_configs():
    """Every optimiser config file in every recipe, as (id, path) pairs."""
    paths = sorted(EGS.glob("*/*/conf/opts/*.yaml"))
    return [pytest.param(path, id=str(path.relative_to(EGS))) for path in paths]


CONFIGS = opt_configs()


def load(path: Path):
    """The parsed config, which is a list of optimiser entries."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(module_name: str, class_name: str):
    """
    Import a class the way `BenNevis.utils.misc.dynamic_import` does.

    Imported here rather than called, because the point is to check the
    config against torch rather than to exercise the import helper.
    """
    import importlib

    module = importlib.import_module(module_name)
    assert hasattr(module, class_name), f"{module_name} has no {class_name}"
    return getattr(module, class_name)


def accepted(cls):
    """
    The keyword arguments a class's constructor accepts.

    Returns None when the signature takes arbitrary keywords, in which case
    there is nothing to check.
    """
    signature = inspect.signature(cls.__init__)
    parameters = signature.parameters
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return None
    return {
        name
        for name, parameter in parameters.items()
        if name != "self"
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


def check(cls, kwargs, where):
    """Assert every configured keyword exists on the class."""
    allowed = accepted(cls)
    if allowed is None:
        return
    unknown = sorted(set(kwargs or {}) - allowed)
    assert not unknown, (
        f"{where}: {cls.__module__}.{cls.__qualname__} does not accept "
        f"{unknown}. The installed torch is {__import__('torch').__version__}; "
        f"it accepts {sorted(allowed)}. A removed parameter here is a crash "
        f"that only a full recipe run would otherwise find."
    )


class TestOptimisers:
    @pytest.mark.parametrize("path", CONFIGS)
    def test_the_config_is_a_list_of_entries(self, path):
        config = load(path)
        assert isinstance(config, list), f"{path}: expected a list"
        assert config, f"{path}: no optimisers configured"

    @pytest.mark.parametrize("path", CONFIGS)
    def test_every_optimiser_exists_and_accepts_its_kwargs(self, path):
        for index, entry in enumerate(load(path)):
            where = f"{path.relative_to(REPO_ROOT)} [{index}]"
            for key in ("module", "name", "nickname"):
                assert key in entry, f"{where}: missing {key!r}"
            cls = resolve(entry["module"], entry["name"])
            check(cls, entry.get("kwargs"), where)

    @pytest.mark.parametrize("path", CONFIGS)
    def test_every_scheduler_exists_and_accepts_its_kwargs(self, path):
        """
        The one that matters: ReduceLROnPlateau's `verbose` was removed from
        torch, and every recipe passed it.
        """
        for index, entry in enumerate(load(path)):
            for position, scheduler in enumerate(entry.get("lr_schedulers", [])):
                where = (
                    f"{path.relative_to(REPO_ROOT)} "
                    f"[{index}].lr_schedulers[{position}]"
                )
                for key in ("module", "name"):
                    assert key in scheduler, f"{where}: missing {key!r}"
                cls = resolve(scheduler["module"], scheduler["name"])
                check(cls, scheduler.get("kwargs"), where)


class TestTheCheckItself:
    """
    The check has to reject something, or it is decoration.
    """

    def test_a_removed_parameter_is_caught(self):
        import torch

        with pytest.raises(AssertionError, match="does not accept"):
            check(
                torch.optim.lr_scheduler.ReduceLROnPlateau,
                {"mode": "min", "verbose": True},
                "synthetic",
            )

    def test_a_valid_parameter_passes(self):
        import torch

        check(
            torch.optim.lr_scheduler.ReduceLROnPlateau,
            {"mode": "min", "factor": 0.5, "patience": 4},
            "synthetic",
        )

    def test_a_class_taking_arbitrary_keywords_is_not_second_guessed(self):
        class Anything:
            def __init__(self, required, **kwargs):
                pass

        assert accepted(Anything) is None
        check(Anything, {"whatever": 1}, "synthetic")

    def test_the_message_names_the_torch_version(self):
        import torch

        with pytest.raises(AssertionError, match=torch.__version__):
            check(
                torch.optim.lr_scheduler.ReduceLROnPlateau,
                {"verbose": True},
                "synthetic",
            )
