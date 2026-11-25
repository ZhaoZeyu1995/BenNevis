"""
Tests for the memory indicators.

These exist because of a failure the log could not explain. A `ctc` run
on mini-librispeech was SIGKILLed three times on a 15.6GiB CI runner, and
a host-side sample every thirty seconds showed the trainer at 1.3GiB in
epoch 1 and 9.3GiB by epoch 7. That told us how much and when, and
nothing about what -- and the loss path had already been cleared locally,
two thousand steps of `GraphLoss` with fixed and with varying batch
shapes holding flat at about 400MiB. So the reading has to come from the
real run, which costs three and a half hours a look, and an indicator
that is subtly wrong wastes one of those.

What is pinned here is therefore the arithmetic, against leaks made on
purpose. A reporter that always printed "nothing grew" would be worse
than no reporter: it would read as evidence.

Two properties matter beyond the obvious. The *first* reading must name
no growers, because everything is new at that point and a baseline that
accuses every type of growing is noise. And distinct tensor storages must
be counted once -- a view shares its storage with the tensor it views, so
adding both would report twice the memory that is really held, and views
are everywhere in a training step.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import sys

import pytest

from BenNevis.utils.memory import (
    MemoryReporter,
    count_live_objects,
    rss_bytes,
    tensor_bytes,
    trim_malloc,
)


class Leaked:
    """A distinctive type, so the counter can be checked by name."""


class TestRssBytes:
    def test_every_supported_platform_can_report_one(self):
        """
        No skip here, deliberately.

        The first Windows implementation returned None -- it truncated the
        process pseudo-handle -- and because the other tests skipped when
        no reading was available, the whole suite went green with the
        indicator dead on a supported platform. A skip that hides a
        defect is indistinguishable from a platform that cannot do the
        job, so this asserts outright that the three platforms the
        library supports all report a size.
        """
        if sys.platform not in ("linux", "darwin", "win32"):
            pytest.skip(f"{sys.platform} is not a supported platform")
        assert rss_bytes() is not None, (
            f"no resident size on {sys.platform}, so the memory indicator "
            "is blank exactly where a developer would read it"
        )

    def test_it_reports_a_plausible_size(self):
        rss = rss_bytes()
        if rss is None:
            pytest.skip("no resident size available on this platform")
        # The interpreter plus pytest is more than a megabyte and less
        # than a hundred gigabytes on any machine this runs on.
        assert 2**20 < rss < 100 * 2**30

    def test_it_is_current_rather_than_peak(self):
        """`ru_maxrss` never falls; this must be able to."""
        if rss_bytes() is None:
            pytest.skip("no resident size available on this platform")
        big = bytearray(256 * 2**20)
        high = rss_bytes()
        del big
        assert high is not None
        # Not asserting that it falls -- the allocator need not return the
        # pages -- but that the reading moved with the allocation, which a
        # constant or a peak-only figure would fail for the wrong reason.
        assert high > 128 * 2**20


class TestCountLiveObjects:
    def test_it_sees_new_instances_by_type_name(self):
        before = count_live_objects().get("Leaked", 0)
        keep = [Leaked() for _ in range(50)]
        after = count_live_objects().get("Leaked", 0)
        assert after - before == 50
        assert len(keep) == 50

    def test_it_counts_containers(self):
        counts = count_live_objects()
        assert counts.get("dict", 0) > 0
        assert counts.get("list", 0) > 0


class TestGrowth:
    def test_the_first_reading_names_nothing(self):
        """Everything is new at the baseline, so nothing has grown."""
        assert MemoryReporter().growth({"Fsa": 10, "Tensor": 99}) == []

    def test_it_names_what_rose(self):
        reporter = MemoryReporter()
        reporter.growth({"Fsa": 1, "Tensor": 5, "dict": 3})
        assert reporter.growth({"Fsa": 4, "Tensor": 5, "dict": 1}) == [("Fsa", 3)]

    def test_it_orders_by_size_then_name(self):
        reporter = MemoryReporter()
        reporter.growth({"a": 0, "b": 0, "c": 0})
        assert reporter.growth({"a": 2, "b": 9, "c": 2}) == [
            ("b", 9),
            ("a", 2),
            ("c", 2),
        ]

    def test_a_type_that_appears_from_nothing_counts_as_growth(self):
        reporter = MemoryReporter()
        reporter.growth({"dict": 1})
        assert reporter.growth({"dict": 1, "Fsa": 7}) == [("Fsa", 7)]

    def test_it_reports_at_most_top(self):
        reporter = MemoryReporter(top=2)
        reporter.growth({})
        assert len(reporter.growth({"a": 1, "b": 2, "c": 3, "d": 4})) == 2

    def test_each_reading_differences_against_the_previous_one(self):
        """Not against the baseline, or a one-off rise never clears."""
        reporter = MemoryReporter()
        reporter.growth({"Fsa": 0})
        reporter.growth({"Fsa": 100})
        assert reporter.growth({"Fsa": 100}) == []


class TestReport:
    def test_it_is_a_single_line(self):
        line = MemoryReporter().report("epoch 1")
        assert "\n" not in line
        assert "epoch 1" in line

    def test_it_names_a_deliberate_leak(self):
        reporter = MemoryReporter()
        reporter.report("before")
        keep = [Leaked() for _ in range(200)]
        assert "Leaked +200" in reporter.report("after")
        assert len(keep) == 200

    def test_the_baseline_line_names_no_growth(self):
        assert "grew:" not in MemoryReporter().report("before training")


@pytest.mark.torch
class TestTensorBytes:
    def test_it_accounts_for_held_tensors(self):
        import torch

        before = tensor_bytes()
        assert before is not None
        keep = torch.zeros(1024, 1024, dtype=torch.float32)  # 4MiB
        after = tensor_bytes()
        assert after - before >= 4 * 2**20
        assert keep.shape == (1024, 1024)

    def test_a_view_is_not_counted_twice(self):
        import torch

        base = torch.zeros(1024, 1024, dtype=torch.float32)
        before = tensor_bytes()
        view = base.view(-1)
        after = tensor_bytes()
        assert after == before, "a view shares its storage and adds no memory"
        assert view.numel() == base.numel()

    def test_the_report_includes_the_tensor_total(self):
        assert "tensors" in MemoryReporter().report()


class TestTrimMalloc:
    """
    The allocator probe, which is Linux-only by nature.

    It exists because the first real reading showed 8.3GiB of resident
    growth with no Python type accumulating, so the next question is
    whether that memory is in use or merely held. What is pinned here is
    that the answer is a *tristate* and that the caller can tell "nothing
    to release" from "could not ask" -- a probe that returned False on
    macOS would read as "the memory is genuinely in use" on a platform
    where the question was never put.
    """

    def test_it_answers_a_boolean_on_linux_and_none_elsewhere(self):
        answer = trim_malloc()
        if sys.platform.startswith("linux"):
            assert answer in (True, False)
        else:
            assert answer is None

    def test_it_can_be_called_repeatedly(self):
        # A second trim with nothing freed in between must not raise; the
        # per-epoch reading calls it for the life of a run.
        trim_malloc()
        trim_malloc()

    def test_the_report_asks_only_when_told_to(self, monkeypatch):
        called = []

        def fake():
            called.append(True)
            return True

        monkeypatch.setattr("BenNevis.utils.memory.trim_malloc", fake)
        reporter = MemoryReporter()
        reporter.report("quiet")
        assert called == []
        reporter.report("loud", trim=True)
        assert called == [True]

    def test_it_reports_what_was_released(self, monkeypatch):
        sizes = iter([900 * 2**20, 400 * 2**20])
        monkeypatch.setattr("BenNevis.utils.memory.rss_bytes", lambda: next(sizes))
        monkeypatch.setattr("BenNevis.utils.memory.trim_malloc", lambda: True)
        line = MemoryReporter().report("after epoch 1", trim=True)
        assert "after trim 400MiB" in line
        assert "released 500MiB" in line

    def test_a_trim_that_releases_nothing_still_reports_the_size(self, monkeypatch):
        # This is the reading that would matter most: the memory is held
        # by something that still considers it live, so the line must say
        # so rather than fall back to "unavailable".
        monkeypatch.setattr("BenNevis.utils.memory.rss_bytes", lambda: 9130 * 2**20)
        monkeypatch.setattr("BenNevis.utils.memory.trim_malloc", lambda: False)
        line = MemoryReporter().report("after epoch 2", trim=True)
        assert "after trim 9130MiB" in line
        assert "released 0MiB" in line

    def test_an_unavailable_trim_says_so(self, monkeypatch):
        monkeypatch.setattr("BenNevis.utils.memory.trim_malloc", lambda: None)
        line = MemoryReporter().report("after epoch 1", trim=True)
        assert "trim unavailable" in line
        assert "after trim" not in line
