"""Custom ML Library: explicit automatic differentiation."""

from .device import Device, available_devices, device, evaluate, is_available, seed, synchronize
from .tensor import (
    Tensor,
    arange,
    empty,
    enable_grad,
    is_grad_enabled,
    no_grad,
    ones,
    ones_like,
    rand,
    randn,
    tensor,
    zeros,
    zeros_like,
)

__version__ = "0.1.0"
__all__ = [
    "Tensor",
    "arange",
    "empty",
    "enable_grad",
    "is_grad_enabled",
    "no_grad",
    "ones",
    "ones_like",
    "rand",
    "randn",
    "tensor",
    "zeros",
    "zeros_like",
    "Device",
    "available_devices",
    "device",
    "evaluate",
    "is_available",
    "seed",
    "synchronize",
]
