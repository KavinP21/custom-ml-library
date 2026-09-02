"""A deliberately small dataset and minibatch API."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterator
from typing import Any

import numpy as np

from .tensor import Tensor, stack


class Dataset:
    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, *tensors: Tensor):
        if not tensors or any(len(t) != len(tensors[0]) for t in tensors):
            raise ValueError("TensorDataset needs equally-sized tensors")
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(t[index] for t in self.tensors)

    def __len__(self):
        return len(self.tensors[0])


def default_collate(items):
    first = items[0]
    if isinstance(first, Tensor):
        return stack(items)
    if isinstance(first, tuple):
        return tuple(default_collate(list(group)) for group in zip(*items))
    if isinstance(first, list):
        return [default_collate(list(group)) for group in zip(*items)]
    if isinstance(first, dict):
        return {key: default_collate([item[key] for item in items]) for key in first}
    return Tensor(np.stack(items))


class DataLoader:
    """Single-process minibatching with deterministic optional shuffling."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        collate_fn=default_collate,
        seed: int | None = None,
        sampler=None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sampler is not None and shuffle:
            raise ValueError("sampler and shuffle cannot be combined")
        self.dataset, self.batch_size, self.shuffle = dataset, batch_size, shuffle
        self.drop_last, self.collate_fn = drop_last, collate_fn
        self._rng = np.random.default_rng(seed)
        self.sampler = sampler

    def __iter__(self) -> Iterator[Any]:
        if self.sampler is None:
            indices = np.arange(len(self.dataset))
        else:
            values = [operator.index(value) for value in self.sampler]
            if any(value < 0 or value >= len(self.dataset) for value in values):
                raise IndexError("sampler index is outside the dataset")
            indices = np.asarray(values, dtype=np.int64)
        if self.shuffle:
            self._rng.shuffle(indices)
        stop = len(indices) - (len(indices) % self.batch_size if self.drop_last else 0)
        for start in range(0, stop, self.batch_size):
            batch = indices[start : start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                break
            yield self.collate_fn([self.dataset[int(i)] for i in batch])

    def __len__(self):
        fn = math.floor if self.drop_last else math.ceil
        size = len(self.dataset) if self.sampler is None else len(self.sampler)
        return fn(size / self.batch_size)


class DistributedSampler:
    """Equal-length rank partitions, with deterministic epoch shuffling.

    By default, padding repeats a few examples so every rank has the same
    number of samples. drop_last instead discards the uneven tail.
    """

    def __init__(self, dataset, num_replicas, rank, *, shuffle=True, seed=0, drop_last=False):
        if (
            not isinstance(rank, int)
            or not isinstance(num_replicas, int)
            or not 0 <= rank < num_replicas
        ):
            raise ValueError("require 0 <= rank < num_replicas")
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.dataset, self.num_replicas, self.rank = dataset, num_replicas, rank
        self.shuffle, self.seed, self.drop_last = shuffle, seed, drop_last
        self.epoch = 0

    def __len__(self):
        operation = math.floor if self.drop_last else math.ceil
        return operation(len(self.dataset) / self.num_replicas)

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        self.epoch = epoch

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            indices = np.random.default_rng(self.seed + self.epoch).permutation(indices).tolist()
        total = len(self) * self.num_replicas
        if self.drop_last:
            indices = indices[:total]
        elif indices:
            indices = (indices * math.ceil(total / len(indices)))[:total]
        return iter(indices[self.rank : total : self.num_replicas])
