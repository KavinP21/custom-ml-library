"""A deliberately small dataset and minibatch API."""

from __future__ import annotations

import math
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
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset, self.batch_size, self.shuffle = dataset, batch_size, shuffle
        self.drop_last, self.collate_fn = drop_last, collate_fn
        self._rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[Any]:
        indices = np.arange(len(self.dataset))
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
        return fn(len(self.dataset) / self.batch_size)
