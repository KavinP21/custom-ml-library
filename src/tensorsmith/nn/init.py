"""Parameter initializers."""

from __future__ import annotations

import math

import numpy as np

from ..device import array


def _replace(tensor, values):
    tensor._data = array(values.astype(np.float32), tensor.device)
    return tensor


def zeros_(tensor):
    return _replace(tensor, np.zeros(tensor.shape))


def ones_(tensor):
    return _replace(tensor, np.ones(tensor.shape))


def constant_(tensor, value: float):
    return _replace(tensor, np.full(tensor.shape, value))


def uniform_(tensor, a: float = 0.0, b: float = 1.0):
    return _replace(tensor, np.random.uniform(a, b, tensor.shape))


def normal_(tensor, mean: float = 0.0, std: float = 1.0):
    return _replace(tensor, np.random.normal(mean, std, tensor.shape))


def _fan_in_out(tensor):
    if tensor.ndim < 2:
        raise ValueError("fan in/out requires a tensor with at least 2 dimensions")
    receptive = math.prod(tensor.shape[2:]) if tensor.ndim > 2 else 1
    return tensor.shape[1] * receptive, tensor.shape[0] * receptive


def xavier_uniform_(tensor, gain: float = 1.0):
    fan_in, fan_out = _fan_in_out(tensor)
    bound = gain * math.sqrt(6 / (fan_in + fan_out))
    return uniform_(tensor, -bound, bound)


def kaiming_uniform_(tensor, a: float = 0.0):
    fan_in, _ = _fan_in_out(tensor)
    gain = math.sqrt(2 / (1 + a * a))
    bound = math.sqrt(3) * gain / math.sqrt(fan_in)
    return uniform_(tensor, -bound, bound)
