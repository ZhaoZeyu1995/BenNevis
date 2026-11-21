"""
Tests for the duration-based dynamic batch sampler.

The sampler had no tests at all, and the first real training run in CI died
in it: `Trainer` read `sampler.num_batches` before the loop started, and the
attribute is only created inside `__iter__`, where the batches are formed.
Nothing caught it because nothing had ever run a training epoch with this
sampler -- which needs only a process group and a dataset with durations, so
it was always reachable from this tier.

The sampler agrees batch boundaries across ranks with a broadcast, so it
needs a real process group; the session-scoped `process_group` fixture
provides a one-rank gloo one.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2026
"""

import pytest

torch = pytest.importorskip("torch")

from BenNevis.samplers.dynamic import DistributedSyncDynamicBatchSampler  # noqa: E402

pytestmark = pytest.mark.torch


class DurationDataset(torch.utils.data.Dataset):
    """
    The least a dataset can be and still drive the sampler.

    The sampler reads `uttids`, `utt2dur` and `sort`, and takes the length;
    everything else about `Dataset` is irrelevant to batching.
    """

    def __init__(self, durations, sort="ascending"):
        self.uttids = [f"utt{index:03d}" for index in range(len(durations))]
        self.utt2dur = dict(zip(self.uttids, durations))
        self.sort = sort
        if sort == "ascending":
            order = sorted(self.uttids, key=lambda u: self.utt2dur[u])
        elif sort == "descending":
            order = sorted(self.uttids, key=lambda u: self.utt2dur[u], reverse=True)
        else:
            order = list(self.uttids)
        self.uttids = order
        self.utt2dur = {uttid: self.utt2dur[uttid] for uttid in order}

    def __len__(self):
        return len(self.uttids)

    def __getitem__(self, index):
        uttid = self.uttids[index]
        return {"uttid": uttid, "dur": self.utt2dur[uttid]}


def collate(items):
    """A batch shaped as Trainer expects one."""
    size = len(items)
    return {
        "feats": torch.zeros(size, 5, 4),
        "feats_lens": torch.full((size,), 5, dtype=torch.int32),
        "target_lengths": torch.full((size,), 2, dtype=torch.int32),
        "word_ids": [[1, 2]] * size,
        "batch_size": size,
    }


@pytest.fixture
def dataset():
    # Durations chosen so that max_sum_dur=10 cannot fit everything in one
    # batch, and so the batch sizes are not all equal.
    return DurationDataset([1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def make_sampler(dataset, **kwargs):
    kwargs.setdefault("shuffle", False)
    kwargs.setdefault("max_sum_dur", 10.0)
    return DistributedSyncDynamicBatchSampler(dataset, **kwargs)


class TestNumBatches:
    def test_is_none_before_iteration(self, process_group, dataset):
        """
        The attribute exists from construction, so a caller gets None rather
        than AttributeError -- which is what broke the first CI training run.
        """
        sampler = make_sampler(dataset)
        assert sampler.num_batches is None

    def test_is_set_once_iteration_begins(self, process_group, dataset):
        sampler = make_sampler(dataset)
        iterator = iter(sampler)
        assert sampler.num_batches is not None
        assert sampler.num_batches > 0
        assert sampler.num_batches == len(list(iterator))

    def test_matches_the_batches_produced(self, process_group, dataset):
        sampler = make_sampler(dataset)
        batches = list(sampler)
        assert sampler.num_batches == len(batches)

    def test_is_refreshed_on_a_later_epoch(self, process_group, dataset):
        sampler = make_sampler(dataset)
        list(sampler)
        first = sampler.num_batches
        sampler.set_epoch(1)
        second_batches = list(sampler)
        assert sampler.num_batches == len(second_batches)
        assert first > 0


class TestBatching:
    def test_every_index_appears_exactly_once(self, process_group, dataset):
        sampler = make_sampler(dataset)
        flat = [index for batch in sampler for index in batch]
        assert sorted(flat) == list(range(len(dataset)))

    def test_no_batch_is_empty(self, process_group, dataset):
        for batch in make_sampler(dataset):
            assert batch

    def test_a_batch_respects_the_duration_budget(self, process_group):
        """
        The point of the sampler: a batch's total duration stays within the
        budget, except where one sample alone exceeds it.
        """
        data = DurationDataset([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
        sampler = make_sampler(data, max_sum_dur=4.0)
        for batch in sampler:
            total = sum(data.utt2dur[data.uttids[index]] for index in batch)
            assert total <= 4.0 or len(batch) == 1

    def test_a_single_oversized_sample_still_gets_a_batch(self, process_group):
        """
        Otherwise it would be silently dropped, and a long utterance would
        vanish from training without a word.
        """
        data = DurationDataset([0.5, 0.5, 99.0])
        sampler = make_sampler(data, max_sum_dur=2.0)
        flat = [index for batch in sampler for index in batch]
        assert sorted(flat) == [0, 1, 2]

    def test_the_longest_utterance_is_in_the_first_batch_when_ascending(
        self, process_group, dataset
    ):
        """
        With an ascending sort the sampler moves the last batch to the front,
        and that batch holds the longest utterances. Deliberate: it makes the
        largest allocation happen on step one, so a run that will exhaust
        memory does so immediately rather than an hour in.

        It is the longest *utterance* that lands first, not the largest total
        duration -- a batch of many short utterances can sum to more.
        """
        sampler = make_sampler(dataset)
        batches = list(sampler)

        longest = max(dataset.utt2dur.values())
        first = [dataset.utt2dur[dataset.uttids[i]] for i in batches[0]]
        assert max(first) == longest

    def test_len_is_the_sample_count_not_the_batch_count(self, process_group, dataset):
        """
        Surprising, and relied upon: `Trainer` uses len(sampler) for the
        progress bar's total, which counts samples.
        """
        sampler = make_sampler(dataset)
        assert len(sampler) == len(dataset)


class TestWithTheTrainer:
    """
    A real training epoch driven by this sampler.

    These are the tests whose absence let the bug through. Before this, no
    test had ever run a whole epoch -- only single batches through
    `_train_batch` -- so the one line that reads the sampler's batch count
    was never executed. None of it needs more than a process group, so it
    belonged in this tier all along.
    """

    @staticmethod
    def build(dataset, tmp_path, **kwargs):
        """A Trainer and a loader wired as a recipe wires them."""
        from BenNevis.core.trainer import Trainer
        from tests.core.conftest import StubLoss

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(4, 3)

            def forward(self, inputs, input_lens):
                return self.proj(inputs).log_softmax(-1), input_lens

        # Separate samplers for training and validation. Sharing one would
        # have `fit`'s validation sanity check iterate it before the first
        # training epoch, which sets its batch count as a side effect and
        # hides whether the trainer learns the count for itself.
        sampler = make_sampler(dataset)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate,
            num_workers=0,
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=make_sampler(dataset),
            collate_fn=collate,
            num_workers=0,
        )
        model = Model()
        optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
        # `Trainer` logs `opt.nickname`, which run/train.py's get_opt
        # attaches with setattr. A Trainer built without it crashes inside
        # logging rather than anywhere informative.
        optimiser.nickname = "test.SGD"
        trainer = Trainer(
            model=model,
            optimizers=[optimiser],
            schedulers=[],
            loss_func=StubLoss("mean"),
            exp_dir=str(tmp_path / "exp"),
            device=torch.device("cpu"),
            **kwargs,
        )
        return trainer, loader, valid_loader, sampler

    def test_fit_runs_with_this_sampler(
        self, process_group, dataset, tmp_path, monkeypatch
    ):
        """
        The whole path a recipe takes: `fit` wires the sampler up, runs an
        epoch, validates and checkpoints. The failure was an AttributeError
        before the first batch, so completing this is the assertion.

        `save_every_n_epochs` is set so the mid-epoch save schedule divides
        by the batch count on every step -- the arithmetic that a missing
        count would break.
        """
        from omegaconf import OmegaConf

        monkeypatch.setenv("WANDB_MODE", "disabled")

        trainer, loader, valid, sampler = self.build(
            dataset,
            tmp_path,
            max_epochs=1,
            save_every_n_epochs=1.0,
            config=OmegaConf.create({"logger": {"project": None, "name": "test"}}),
        )

        trainer.fit(loader, valid)

        assert sampler.num_batches is not None
        assert sampler.num_batches > 0

    def test_two_epochs_run(self, process_group, dataset, tmp_path, monkeypatch):
        """
        More than one, because the sampler recomputes its batches every epoch
        and `set_epoch` reshuffles them. A count cached from the first epoch
        would be wrong for the second.
        """
        from omegaconf import OmegaConf

        monkeypatch.setenv("WANDB_MODE", "disabled")

        trainer, loader, valid, sampler = self.build(
            dataset,
            tmp_path,
            max_epochs=2,
            save_every_n_epochs=1.0,
            config=OmegaConf.create({"logger": {"project": None, "name": "t"}}),
        )

        trainer.fit(loader, valid)

        assert trainer.epoch == 2
        assert sampler.num_batches == len(list(sampler))

    def test_the_mid_epoch_save_schedule_fires(
        self, process_group, dataset, tmp_path, monkeypatch
    ):
        """
        The test that distinguishes the two halves of the fix.

        Declaring `num_batches` on the sampler stops the AttributeError on
        its own, but leaves the count None for every dynamic-batching run --
        and the mid-epoch save schedule divides by it, so it would simply
        never fire again. Nothing would crash; checkpointing would quietly
        become once-per-epoch.

        `save_every_n_epochs=0.5` asks for a save half way through, which can
        only happen if the trainer learned the real batch count. `fit` also
        saves at the end of each epoch, so the signal is two saves rather
        than one.
        """
        from omegaconf import OmegaConf

        monkeypatch.setenv("WANDB_MODE", "disabled")

        trainer, loader, valid, sampler = self.build(
            dataset,
            tmp_path,
            max_epochs=1,
            save_every_n_epochs=0.5,
            config=OmegaConf.create({"logger": {"project": None, "name": "t"}}),
        )

        saves = []
        original = trainer._save_checkpoint
        monkeypatch.setattr(
            trainer,
            "_save_checkpoint",
            lambda *a, **k: (saves.append(1), original(*a, **k))[1],
        )

        trainer.fit(loader, valid)

        assert sampler.num_batches and sampler.num_batches > 1
        assert len(saves) >= 2, (
            f"only {len(saves)} checkpoint save(s); the mid-epoch schedule "
            f"did not fire, so the trainer never learned the batch count"
        )

    def test_a_checkpoint_is_written(
        self, process_group, dataset, tmp_path, monkeypatch
    ):
        """
        The mid-epoch save schedule divides by the batch count, so a run that
        saves at all is a run where that arithmetic worked.
        """
        from omegaconf import OmegaConf

        monkeypatch.setenv("WANDB_MODE", "disabled")

        trainer, loader, valid, _ = self.build(
            dataset,
            tmp_path,
            max_epochs=1,
            save_every_n_epochs=1.0,
            config=OmegaConf.create({"logger": {"project": None, "name": "t"}}),
        )

        trainer.fit(loader, valid)

        checkpoints = list((tmp_path / "exp" / "checkpoints").glob("*.pt"))
        assert checkpoints, "no checkpoint was written"

    def test_the_sampler_is_one_fit_accepts(self, process_group, dataset):
        """
        `fit` raises for anything that is neither a DistributedSampler nor
        this one, so the two have to stay in step.
        """
        from BenNevis.samplers.dynamic import DistributedSyncDynamicBatchSampler

        sampler = make_sampler(dataset)
        assert isinstance(sampler, DistributedSyncDynamicBatchSampler)
