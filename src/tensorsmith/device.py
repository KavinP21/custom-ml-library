"""CPU array helpers."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class Device:
    type: str = "cpu"
    index: int | None = None

    def __post_init__(self):
        if self.type != "cpu" or self.index is not None:
            raise ValueError("this milestone supports CPU only")

    def __str__(self):
        return self.type


DeviceLike = str | Device | None


def device(value: DeviceLike = None):
    return value if isinstance(value, Device) else Device("cpu" if value is None else value)


def is_available(value):
    return str(value) == "cpu" or value is None


def require_available(value):
    return device(value)


def xp_for(value):
    device(value)
    return np


def backend_dtype(dtype, value):
    device(value)
    return dtype


def array(data: Any, value, dtype=None):
    device(value)
    return np.asarray(getattr(data, "_data", data), dtype=dtype)


def asnumpy(data):
    return np.asarray(data)


def copy_array(data, value):
    device(value)
    return data.copy()


def add_at(target, key, source, value):
    device(value)
    np.add.at(target, key, source)
    return target


def assign_add(target, key, source, value):
    device(value)
    target[key] += source
    return target


def assign(target, key, source):
    target[key] = source
    return target


def seed(value):
    np.random.seed(value)


def synchronize(value):
    device(value)


def evaluate(*values):
    pass


def random_uniform(shape, value):
    device(value)
    return np.random.random(shape).astype("float32")


def available_devices():
    return [Device("cpu")]


def normalize_shape(shape):
    return (shape,) if isinstance(shape, int) else tuple(shape)
