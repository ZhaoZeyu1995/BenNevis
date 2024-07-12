"""
This file contains several (currently two) distributed dynamic batching samplers.
They are named **distributed** because they are designed to apply in a distributed setting (e.g., DDP).
Compared to `DistributedSampler` in Pytorch, these samplers are designed to batch data with variable batch sizes.
There is a main argument with respect to dynamic batching:
    * max_sum_dur: the maximum sum of duration of all samples in a batch.
To make dynamic batching more efficient, we assume that the dataset has been sorted based on the `dur` (duration).
For example, in ASR, we want to group the samples with similar duration together,
and we start from those samples with the smallest duration.
In this case, through out an epoch, the batch size on each process will descend.

Here is how `DistributedSyncDynamicBatchSampler` works:
1. At the beginning of each epoch, we split the dataset indices into `num_replicas` splits,
where the sorting property is kept, and each process will get one split only.
2. On the main process (rank 0), we calculate the batch sizes based on the `max_sum_dur`.
3. We broadcast the batch sizes to other processes.
4. All the processes will have the same batch sizes in this epoch.
Note that we cannot guarantee that all the processes strictly have a total duration
lower than `max_sum_dur` in each batch with this sampler.
However, as the dataset has been sorted, we assume that this should hold and
will not cause troubles in most cases.
In this way, we only need to broadcast the batch sizes from rank 0 at the beginning of each epoch, and as we said,
all the processes will have the same batch sizes in an epoch.

The other way is to use `DistributedDynamicBatchSampler`.
The main difference is that, each process will have do the dynamic batch individually.
Thus, if we need to know the total batch size sometimes, we will have to do `all_reduce`,
which may introduce a inter-device (e.g., GPUs) communication overhead.
Thus, currently, we recommend using `DistributedSyncDynamicBatchSampler` if possible.

Note that both of the samplers are designed to be used with
`torch.utils.data.DataLoader` as the `batch_sampler` argument.
A small difference is that `len(batch_sampler)` will return the number of total samples
on each process, but not the number of batches with a `BatchSampler` as usual.

Authors:
    * Zeyu Zhao (The University of Edinburgh) 2024
"""

import math
from typing import Optional, Iterator, List
import torch
import numpy as np
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
import logging


class DistributedDynamicBatchSampler(Sampler[List[int]]):
    """
    Distributed Sampler for dynamic batching.
    This sampler is used to batch data with variable batch sizes.
    There is a main argument with respect to dynamic batching:
        * max_sum_dur: the maximum sum of duration of all samples in a batch.

    This sampler assumes that the dataset has been sorted with respect to `dur` (duration).
    This is suitable for training ASR models as we want to group the samples with similar duration together,
    and to start from those with smallest duration (if we sort the dataset by duration ascendingly).

    Note that in this sampler, each process will do the dynamic batching individually,
    which may probably result in different batch sizes on processes.
    This is, in my opinion, not preferred, as we may need to know the total batch size during training,
    and we will have to do `all_reduce`, which may result in a performance drop.
    Thus, `DistributedSyncDynamicBatchSampler` is recommended, if possible.

    Arguments
    ---------
    dataset: Dataset
        The dataset to sample from.
    num_replicas: int, optional (default = None)
        Number of processes participating in the training, if None, it will be set to `dist.get_world_size()`.
    rank: int, optional (default = None)
        Rank of the current process, if None, it will be set to `dist.get_rank()`.
    shuffle: bool, optional (default = True)
        Whether to shuffle the indices or not.
    seed: int, optional (default = 0)
        Random seed to use for shuffling.
    max_sum_dur: float, optional (default = 100.0)
        The maximum sum of duration of all samples in a batch
    drop_last: bool, optional (default = False)
        Whether to drop the last incomplete batch or not.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        max_sum_dur: float = 100.0,
        drop_last: bool = False,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.dataset_length = len(self.dataset)
        self.max_sum_dur = max_sum_dur

        assert self.dataset.sort in [
            "ascending",
            "descending",
        ], "dataset should be sorted in ascending or descending order"
        self.indices = list(range(self.dataset_length))

        logging.debug("Rank %d is creating the indices", self.rank)
        if not self.drop_last and self.total_size > len(self.indices):
            padding_size = self.total_size - len(self.indices)
            if padding_size <= len(self.indices):
                # add the last padding_size elements as self.indices has been sorted
                self.indices += self.indices[-padding_size:]
            else:
                # Usually, it means that the dataset is too small, and we need to duplicate the indices
                self.indices += (
                    self.indices * math.ceil(padding_size / len(self.indices))
                )[:padding_size]
        else:
            self.indices = self.indices[: self.total_size]
        logging.debug("Rank %d has created the indices", self.rank)

        assert (
            len(self.indices) == self.total_size
        ), f"{len(self.indices)} vs {self.total_size}"

    def __iter__(self) -> Iterator[List[int]]:
        """
        Create an iterator that returns the indices of the samples in the dataset.
        Note that each process will do the dynamic batching individually,
        which may probably result in different batch sizes on different processes.
        """

        indices = self.indices.copy()
        indices = np.array(indices).reshape(self.num_samples, self.num_replicas).T
        if self.shuffle:
            np.random.seed(self.seed + self.epoch)
            indices = np.random.permutation(indices)
        indices = indices[self.num_replicas - self.rank - 1].tolist()
        assert len(indices) == self.num_samples, f"{len(indices)} vs {self.num_samples}"

        logging.debug(
            f"Rank {self.rank}: len(indices) {len(indices)}, len(self.dataset) {len(self.dataset)}"
        )
        start = 0
        pointer = 0
        accumulate = 0
        batch_size = 0
        while pointer < len(indices):
            logging.debug(
                f"Rank {dist.get_rank()} pointer {pointer} accumulate {accumulate} batch_size {batch_size}"
            )
            uttid = self.dataset.uttids[indices[pointer]]
            dur = self.dataset.utt2dur[uttid]
            if accumulate + dur <= self.max_sum_dur:
                accumulate += dur
                batch_size += 1
                pointer += 1
            else:
                yield indices[start:pointer]
                start = pointer
                accumulate = 0
                batch_size = 0
        if batch_size > 0:
            yield indices[start:pointer]

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DistributedSyncDynamicBatchSampler(Sampler[List[int]]):
    """
    Distributed Sampler for dynamic batching.
    This sampler is used to batch data with variable batch sizes.
    There is a main argument with respect to dynamic batching:
        * max_sum_dur: the maximum sum of duration of all samples in a batch.

    This sampler assumes that the dataset has been sorted with respect to `dur` (duration).
    This is suitable for training ASR models as we want to group the samples with similar duration together,
    and to start from those samples with the smallest duration (if we sort the dataset by duration ascendingly).

    Note that in this sampler, at the beginning of each epoch,
    the main process with rank 0 will do the dynamic batching first.
    Then, the indices will be broadcasted to all other processes.
    This is to ensure that all processes will have the batch partition,
    so that they will always have the same batch sizes and roughly finish computation at the same time.
    Also, there is no need to do any GPU synchronization during one epoch with this sampler.

    Arguments
    ---------
    dataset: Dataset
        The dataset to sample from.
    num_replicas: int, optional (default = None)
        Number of processes participating in the training, if None, it will be set to `dist.get_world_size()`.
    rank: int, optional (default = None)
        Rank of the current process, if None, it will be set to `dist.get_rank()`.
    shuffle: bool, optional (default = True)
        Whether to shuffle the indices or not.
    seed: int, optional (default = 0)
        Random seed to use for shuffling.
    max_sum_dur: float, optional (default = 100.0)
        The maximum sum duration of all samples in a batch.
    drop_last: bool, optional (default = False)
        Whether to drop the last incomplete batch or not.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        max_sum_dur: float = 1000.0,
        drop_last: bool = False,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        logging.debug(f"RANK: {rank}, creating sampler")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.dataset_length = len(self.dataset)
        self.max_sum_dur = max_sum_dur

        assert self.dataset.sort in [
            "ascending",
            "descending",
        ], "dataset should be sorted in ascending or descending order"
        self.indices = list(range(self.dataset_length))

        logging.debug("Rank %d is creating the indices", self.rank)
        if not self.drop_last and self.total_size > len(self.indices):
            padding_size = self.total_size - len(self.indices)
            if padding_size <= len(self.indices):
                # add the last padding_size elements as self.indices has been sorted
                self.indices += self.indices[-padding_size:]
            else:
                # Usually, this should not happen, but just in case
                # It means that the dataset is too small
                self.indices += (
                    self.indices * math.ceil(padding_size / len(self.indices))
                )[:padding_size]
        else:
            self.indices = self.indices[: self.total_size]
        logging.debug("Rank %d has created the indices", self.rank)

        assert (
            len(self.indices) == self.total_size
        ), f"{len(self.indices)} vs {self.total_size}"

    def __iter__(self) -> Iterator[List[int]]:
        """
        Returns an iterator of lists of indices.
        Each list of indices denotes a batch of samples.
        We will do the dynamic batching on the master process and,
        broadcast batch_sizes (List[int]) to other processes.

        At the beginning of each epoch, we shuffle the indices and then
        divide the indices into num_replicas splits, and each process will
        get one split only.

        Finally, `batch_sizes` contains N+1 elements, where N is the number of batches
        for each process.
        indices[batch_sizes[i], batch_sizes[i+1]] is the indices for the i-th batch, i=0,1,...,N-1.
        """

        indices = self.indices.copy()
        indices = np.array(indices).reshape(self.num_samples, self.num_replicas).T
        if self.shuffle:
            np.random.seed(self.seed + self.epoch)
            indices = np.random.permutation(indices)
        indices = indices[self.num_replicas - self.rank - 1].tolist()
        assert len(indices) == self.num_samples, f"{len(indices)} vs {self.num_samples}"
        device = torch.cuda.current_device()

        logging.debug(
            f"Rank {self.rank}: len(indices) {len(indices)}, len(self.dataset) {len(self.dataset)}"
        )
        if self.rank == 0:
            batch_sizes = []
            pointer = 0
            accumulate = 0
            batch_size = 0
            while pointer < len(indices):
                uttid = self.dataset.uttids[indices[pointer]]
                dur = self.dataset.utt2dur[uttid]
                if accumulate + dur <= self.max_sum_dur:
                    accumulate += dur
                    batch_size += 1
                    pointer += 1
                else:
                    batch_sizes.append(batch_size)
                    accumulate = 0
                    batch_size = 0
            if batch_size > 0:
                batch_sizes.append(batch_size)
            batch_sizes_len = len(batch_sizes)
            batch_sizes = torch.tensor(batch_sizes, dtype=torch.int32, device=device)
            batch_sizes_len = torch.tensor(
                [batch_sizes_len], dtype=torch.int32, device=device
            )
        else:
            batch_sizes = torch.empty(1, dtype=torch.int32, device=device)
            batch_sizes_len = torch.empty(1, dtype=torch.int32, device=device)

        logging.debug(
            f"Rank {dist.get_rank()} batch_sizes {batch_sizes} batch_sizes_len {batch_sizes_len}"
        )

        dist.broadcast(batch_sizes_len, 0)
        logging.debug(f"Rank {dist.get_rank()} batch_sizes_len {batch_sizes_len}")
        if self.rank != 0:
            batch_sizes.resize_(batch_sizes_len.tolist())
            logging.debug(
                f"Rank {dist.get_rank()} batch_sizes.size() {batch_sizes.size()}"
            )
        dist.broadcast(batch_sizes, 0)
        logging.debug(f"Rank {dist.get_rank()} batch_sizes {batch_sizes}")

        torch.cumsum(batch_sizes, 0, out=batch_sizes)
        logging.debug(f"Rank {dist.get_rank()} torch.cumsum(batch_sizes) {batch_sizes}")
        assert batch_sizes[-1] == len(
            indices
        ), "Rank {} batch_sizes[-1] {} vs len(self.indices) {}".format(
            self.rank, batch_sizes[-1], len(indices)
        )
        batch_sizes = batch_sizes.cpu().tolist()
        batch_sizes = [0] + batch_sizes
        indices = [
            indices[batch_sizes[i] : batch_sizes[i + 1]]
            for i in range(len(batch_sizes) - 1)
        ]
        # if dataset is sorted in ascending order, we put the last element of indices at the beginning
        # this is more efficient for GPU memory caching
        if self.dataset.sort == "ascending":
            indices = [indices[-1]] + indices[:-1]
        self.num_batches = len(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
