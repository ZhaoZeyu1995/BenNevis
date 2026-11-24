"""
Tests that a training run says where it is.

A nightly recipe run is ninety minutes inside a container, and the only
window onto it is the job log. Until this was fixed that log was useless:
`tqdm` writes a fresh line on every update when stderr is not a terminal,
so eighty epochs produced thousands of near-identical lines with no epoch
boundaries in them, and a healthy run looked exactly like a hung one. The
author of this repository went to check on a run and reported it stuck; it
was at epoch 68 of 79 and converging.

Two things are pinned here, because both are easy to lose and neither fails
anything when lost:

1. Progress bars disable themselves when stderr is not a terminal.
2. Each epoch logs one line, with the losses and the step count.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import logging
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.torch

TRAINER = (
    Path(__file__).resolve().parent.parent.parent / "BenNevis" / "core" / "trainer.py"
)


class TestProgressBarsAreQuietOffTerminal:
    def test_every_bar_disables_itself_when_not_a_terminal(self):
        """
        `disable=None` is tqdm's documented way of saying "off unless stderr
        is a terminal". `False` would keep the bar; omitting it defaults to
        `False`, which is what made the logs unreadable.
        """
        text = TRAINER.read_text()
        calls = len(re.findall(r"\btqdm\(", text))
        disabled = len(re.findall(r"disable=None", text))
        assert calls > 0
        assert disabled == calls, (
            f"{calls} tqdm call site(s) but {disabled} with disable=None; a "
            f"bar without it writes a line per update into the CI log"
        )

    def test_a_bar_really_is_disabled_off_terminal(self, monkeypatch):
        """
        Not just that the argument is passed, but that tqdm honours it --
        checked against the installed tqdm rather than assumed.
        """
        import io

        from tqdm import tqdm

        stream = io.StringIO()  # not a tty
        bar = tqdm(total=10, disable=None, file=stream)
        for _ in range(10):
            bar.update(1)
        bar.close()
        assert stream.getvalue() == "", "tqdm wrote to a non-terminal stream"


class TestEachEpochIsLogged:
    def test_fit_logs_one_line_per_epoch(
        self, process_group, dataset_for_logs, tmp_path, monkeypatch, caplog
    ):
        """
        The line a reader needs. Without it, a run with the bars disabled
        says nothing at all between starting and finishing.
        """
        from omegaconf import OmegaConf

        monkeypatch.setenv("WANDB_MODE", "disabled")
        trainer, loader, valid = dataset_for_logs(
            tmp_path,
            max_epochs=3,
            config=OmegaConf.create({"logger": {"project": None, "name": "t"}}),
        )

        with caplog.at_level(logging.INFO):
            trainer.fit(loader, valid)

        lines = [
            record.message
            for record in caplog.records
            if record.message.startswith("Epoch ")
        ]
        assert len(lines) == 3, lines
        assert lines[0].startswith("Epoch 1/3:")
        assert lines[-1].startswith("Epoch 3/3:")

    def test_the_line_carries_the_losses_and_the_step(
        self, process_group, dataset_for_logs, tmp_path, monkeypatch, caplog
    ):
        from omegaconf import OmegaConf

        monkeypatch.setenv("WANDB_MODE", "disabled")
        trainer, loader, valid = dataset_for_logs(
            tmp_path,
            max_epochs=1,
            config=OmegaConf.create({"logger": {"project": None, "name": "t"}}),
        )
        with caplog.at_level(logging.INFO):
            trainer.fit(loader, valid)

        line = next(r.message for r in caplog.records if r.message.startswith("Epoch "))
        assert "train loss" in line
        assert "valid loss" in line
        assert "step" in line
        # A number for each, so the line is informative rather than decorative.
        assert len(re.findall(r"\d+\.\d{6}", line)) == 2


class TestMemoryIsLoggedEachEpoch:
    """
    The reading that three SIGKILLs went without.

    A `ctc` run on mini-librispeech was killed three times on a 15.6GiB
    runner with the trainer at 9.3GiB, and the log said nothing about what
    was holding it -- so the next run had to be spent on measuring rather
    than on the WER it was dispatched for. These pin the line, because an
    indicator nothing asserts is one that disappears in a refactor and is
    missed only on the next long run.
    """

    def test_fit_logs_a_memory_line_per_epoch_and_a_baseline(
        self, process_group, dataset_for_logs, tmp_path, monkeypatch, caplog
    ):
        from omegaconf import OmegaConf

        monkeypatch.setenv("WANDB_MODE", "disabled")
        trainer, loader, valid = dataset_for_logs(
            tmp_path,
            max_epochs=2,
            config=OmegaConf.create({"logger": {"project": None, "name": "t"}}),
        )

        with caplog.at_level(logging.INFO):
            trainer.fit(loader, valid)

        lines = [
            record.message
            for record in caplog.records
            if record.message.startswith("memory")
        ]
        # One before training plus one per epoch: the baseline is what the
        # first epoch's growth is measured against.
        assert len(lines) == 3, lines
        assert "before training" in lines[0]
        assert "after epoch 1" in lines[1]
        assert "after epoch 2" in lines[2]

    def test_the_line_carries_a_resident_size(
        self, process_group, dataset_for_logs, tmp_path, monkeypatch, caplog
    ):
        """A reading without a number is decoration."""
        from omegaconf import OmegaConf

        monkeypatch.setenv("WANDB_MODE", "disabled")
        trainer, loader, valid = dataset_for_logs(
            tmp_path,
            max_epochs=1,
            config=OmegaConf.create({"logger": {"project": None, "name": "t"}}),
        )
        with caplog.at_level(logging.INFO):
            trainer.fit(loader, valid)

        line = next(
            r.message for r in caplog.records if r.message.startswith("memory after")
        )
        assert re.search(r"rss \d+MiB", line), line
        assert re.search(r"objects \d+", line), line
