"""Device discovery and backend-neutral array helpers.

TensorSmith deliberately keeps this layer small. NumPy, CuPy, and MLX execute
primitive kernels; TensorSmith owns graph construction and differentiation.
"""

from __future__ import annotations

import importlib.util
import platform
import subprocess
import sys
from collections.abc import Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np

# Checkpointing replays native device draws without private vendor RNG-state
# APIs. Normal execution does not retain draws; context is task-local.
_random_tape = ContextVar("tensorsmith_random_tape", default=None)


@dataclass(frozen=True)
class Device:
    """A compute device, e.g. ``cpu``, ``cuda:0``, or ``metal``."""

    type: str = "cpu"
    index: int | None = None

    def __post_init__(self) -> None:
        kind = self.type.lower()
        if kind == "mps":
            kind = "metal"
        if kind not in {"cpu", "cuda", "metal"}:
            raise ValueError(f"unknown device {self.type!r}; expected cpu, cuda, or metal")
        if kind != "cuda" and self.index is not None:
            raise ValueError(f"{kind} does not accept a device index")
        if kind == "cuda" and self.index is not None and self.index < 0:
            raise ValueError("CUDA device index must be non-negative")
        object.__setattr__(self, "type", kind)

    def __str__(self) -> str:
        return f"cuda:{self.index}" if self.type == "cuda" and self.index is not None else self.type


DeviceLike = str | Device | None


def device(value: DeviceLike = None) -> Device:
    if value is None:
        return Device("cpu")
    if isinstance(value, Device):
        return value
    value = value.lower().strip()
    if value.startswith("cuda:"):
        return Device("cuda", int(value.split(":", 1)[1]))
    return Device(value)


@cache
def _is_available(dev: Device) -> bool:
    if dev.type == "cpu":
        return True
    if dev.type == "cuda":
        if importlib.util.find_spec("cupy") is None:
            return False
        try:
            import cupy as cp

            return cp.cuda.runtime.getDeviceCount() > (dev.index or 0)
        except (ImportError, RuntimeError, OSError):
            return False
    if not (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx") is not None
    ):
        return False
    # Some Metal loaders terminate the process instead of raising when MLX is
    # installed in a headless/sandboxed environment. Probe in a child process
    # once so device discovery remains safe and truthful.
    probe = "import mlx.core as mx; mx.eval(mx.array([0.0]))"
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def is_available(value: DeviceLike) -> bool:
    return _is_available(device(value))


def require_available(value: DeviceLike) -> Device:
    dev = device(value)
    if is_available(dev):
        return dev
    hint = {
        "cuda": "install TensorSmith with `pip install -e '.[cuda]'` and use an NVIDIA GPU",
        "metal": "install TensorSmith with `pip install -e '.[metal]'` on Apple silicon",
    }.get(dev.type, "")
    raise RuntimeError(f"device {dev} is unavailable" + (f"; {hint}" if hint else ""))


def xp_for(value: DeviceLike):
    dev = require_available(value)
    if dev.type == "cpu":
        return np
    if dev.type == "cuda":
        import cupy as cp

        cp.cuda.Device(dev.index or 0).use()
        return cp
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    return mx


def backend_dtype(dtype: Any, value: DeviceLike):
    """Translate portable dtype names to a backend's native dtype token."""
    dev = device(value)
    if dtype is None or dev.type != "metal":
        return dtype
    import mlx.core as mx

    if type(dtype).__module__.startswith("mlx"):
        name = str(dtype).split(".")[-1]
    else:
        name = dtype if isinstance(dtype, str) else str(np.dtype(dtype))
    aliases = {"bool": "bool_", "long": "int64", "float": "float32"}
    name = aliases.get(name, name)
    try:
        return getattr(mx, name)
    except AttributeError as error:
        raise TypeError(f"dtype {dtype!r} is not supported on Metal") from error


def array(data: Any, value: DeviceLike, dtype: Any = None):
    dev = require_available(value)
    if hasattr(data, "_data"):
        data = data._data
    source_module = type(data).__module__
    if dev.type == "cpu":
        if source_module.startswith("cupy"):
            data = data.get()
        elif source_module.startswith("mlx"):
            data = np.asarray(data)
        return np.asarray(data, dtype=dtype)
    if dev.type == "cuda":
        import cupy as cp

        with cp.cuda.Device(dev.index or 0):
            if source_module.startswith("cupy"):
                return cp.asarray(data, dtype=dtype)
            return cp.asarray(asnumpy(data), dtype=dtype)
    import mlx.core as mx

    if source_module.startswith("mlx"):
        return data if dtype is None else data.astype(backend_dtype(dtype, dev))
    host = asnumpy(data)
    return mx.array(host) if dtype is None else mx.array(host, dtype=backend_dtype(dtype, dev))


def asnumpy(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data
    module = type(data).__module__
    if module.startswith("cupy"):
        return data.get()
    if module.startswith("mlx"):
        import mlx.core as mx

        mx.eval(data)
        return np.asarray(data)
    return np.asarray(data)


def copy_array(data: Any, value: DeviceLike):
    dev = device(value)
    if dev.type == "metal":
        return data + xp_for(dev).zeros_like(data)
    return data.copy()


def add_at(target: Any, key: Any, source: Any, value: DeviceLike):
    """Backend-neutral indexed accumulation used by slicing and convolution."""
    dev = device(value)
    if dev.type in {"cpu", "cuda"}:
        xp_for(dev).add.at(target, key, source)
        return target
    try:
        return target.at[key].add(source)
    except (AttributeError, TypeError):  # compatibility with older MLX releases
        host = asnumpy(target).copy()
        np.add.at(host, key, asnumpy(source))
        return array(host, dev)


def assign_add(target: Any, key: Any, source: Any, value: DeviceLike):
    """Add a dense slice, functionally on MLX and in-place elsewhere."""
    dev = device(value)
    if dev.type == "metal":
        try:
            return target.at[key].add(source)
        except (AttributeError, TypeError):
            host = asnumpy(target).copy()
            host[key] += asnumpy(source)
            return array(host, dev)
    target[key] += source
    return target


def seed(value: int) -> None:
    """Seed host-side initialization (and native GPU RNGs when present)."""
    np.random.seed(value)
    if is_available("cuda"):
        import cupy as cp

        cp.random.seed(value)
    if is_available("metal"):
        import mlx.core as mx

        mx.random.seed(value)


def synchronize(value: DeviceLike) -> None:
    dev = require_available(value)
    if dev.type == "cuda":
        xp_for(dev).cuda.runtime.deviceSynchronize()
    elif dev.type == "metal":
        import mlx.core as mx

        mx.synchronize()


def evaluate(*values: Any) -> None:
    """Materialize lazy arrays, including nested tensors, gradients, and state.

    Unlike ``synchronize``, this submits MLX graphs for execution. Timing and
    training loops should evaluate outputs AND gradients/updated parameters.
    """
    arrays = []
    # A recursively self-referencing closure forms a Python reference cycle
    # that retains `arrays` (potentially gigabytes of completed GPU logits)
    # until cyclic GC. Iterative traversal releases all outputs on return.
    pending = list(reversed(values))
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(reversed(list(value.values())))
        elif isinstance(value, (list, tuple)):
            pending.extend(reversed(value))
        else:
            raw = getattr(value, "_data", value)
            if type(raw).__module__.startswith("mlx"):
                arrays.append(raw)
    if arrays:
        import mlx.core as mx

        mx.eval(*arrays)


def random_uniform(shape: tuple[int, ...], value: DeviceLike):
    """Generate random values on the selected device, without a host transfer."""
    dev = device(value)
    tape = _random_tape.get()
    if tape is not None and tape["mode"] == "replay":
        position = tape["position"]
        if position >= len(tape["draws"]):
            raise RuntimeError("checkpoint recomputation changed the number of random draws")
        expected_shape, expected_device, raw = tape["draws"][position]
        if tuple(shape) != expected_shape or dev != expected_device:
            raise RuntimeError("checkpoint recomputation changed a random draw's shape/device")
        tape["position"] += 1
        # Restore/advance the host RNG consistently with ts.rand/randn calls.
        return np.random.random(shape).astype("float32") if dev.type == "cpu" else raw
    xp = xp_for(dev)
    if dev.type == "metal":
        raw = xp.random.uniform(shape=shape)
    else:
        raw = xp.random.random(shape).astype("float32")
    if tape is not None:
        tape["draws"].append((tuple(shape), dev, None if dev.type == "cpu" else raw))
    return raw


def assign(target: Any, key: Any, source: Any):
    """Overwrite an indexed slice on NumPy, CuPy, or MLX."""
    target[key] = source
    return target


def available_devices() -> list[Device]:
    result = [Device("cpu")]
    if is_available("cuda"):
        import cupy as cp

        result.extend(Device("cuda", i) for i in range(cp.cuda.runtime.getDeviceCount()))
    if is_available("metal"):
        result.append(Device("metal"))
    return result


def normalize_shape(shape: int | Iterable[int]) -> tuple[int, ...]:
    return (shape,) if isinstance(shape, int) else tuple(shape)
