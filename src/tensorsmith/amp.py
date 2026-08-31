"""Operator-scoped mixed precision and dynamic loss scaling."""

from __future__ import annotations

import math
from contextlib import ContextDecorator
from contextvars import ContextVar

import numpy as np

from .device import asnumpy, xp_for
from .device import device as parse_device

_autocast = ContextVar("tensorsmith_autocast", default=None)


class autocast(ContextDecorator):
    """Cast FP32 matmul/linear/convolution inputs to FP16 in this context.

    Model parameters remain FP32, and casts preserve their backward paths.
    Float64 operations are left untouched. CPU, CUDA and Metal share the same
    explicit policy; BF16 and PyTorch's full operator-policy table are not provided.
    Use a separate context instance in each thread/task.
    """

    def __init__(self, device_type="cuda", *, dtype="float16", enabled=True):
        self.device_type = parse_device(device_type).type
        try:
            self.dtype = np.dtype(dtype).name
        except TypeError as error:
            raise ValueError("autocast currently supports dtype='float16' only") from error
        if self.dtype != "float16":
            raise ValueError("autocast currently supports dtype='float16' only")
        self.enabled = bool(enabled)
        self._tokens = []

    def __enter__(self):
        state = dict(_autocast.get() or {})
        state[self.device_type] = (self.enabled, self.dtype)
        self._tokens.append(_autocast.set(state))
        return self

    def __exit__(self, *exc):
        _autocast.reset(self._tokens.pop())
        return False

    def _recreate_cm(self):
        return type(self)(self.device_type, dtype=self.dtype, enabled=self.enabled)


def is_autocast_enabled(device_type="cuda"):
    entry = (_autocast.get() or {}).get(parse_device(device_type).type)
    return bool(entry and entry[0])


def get_autocast_dtype(device_type="cuda"):
    entry = (_autocast.get() or {}).get(parse_device(device_type).type)
    return entry[1] if entry is not None else None


def _cast_inputs(*values):
    tensors = [value for value in values if value is not None]
    if not tensors:
        return values
    dev = tensors[0].device
    entry = (_autocast.get() or {}).get(dev.type)
    if not entry or not entry[0] or any(value.device != dev for value in tensors):
        return values
    if any(str(value.dtype).split(".")[-1] not in {"float16", "float32"} for value in tensors):
        return values
    return tuple(
        value.astype(entry[1]) if value is not None and "float32" in str(value.dtype) else value
        for value in values
    )


class GradScaler:
    """Dynamic loss scaling with overflow detection and skipped optimizer steps.

    Use model.to(dtype='float16'), scale(loss).backward(), unscale_(optimizer)
    before clipping, step(optimizer), then update(). Adam stores FP32 master
    weights/moments for low-precision parameters. Finite checking synchronizes
    one scalar per device; this is correctness-first, not a fused AMP engine.
    """

    def __init__(
        self,
        init_scale=65536.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True,
    ):
        if (
            not math.isfinite(init_scale)
            or init_scale <= 0
            or not math.isfinite(growth_factor)
            or growth_factor <= 1
        ):
            raise ValueError("invalid scale/growth factor")
        if not 0 < backoff_factor < 1 or growth_interval <= 0:
            raise ValueError("invalid backoff factor/growth interval")
        self.enabled = enabled
        self._scale = float(init_scale) if enabled else 1.0
        self.growth_factor, self.backoff_factor = growth_factor, backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0
        self._optimizers = {}

    def get_scale(self):
        return self._scale

    def scale(self, loss):
        return loss * self._scale if self.enabled else loss

    def unscale_(self, optimizer):
        """Unscale once and check finiteness; return True when clipping is safe."""
        if id(optimizer) in self._optimizers:
            raise RuntimeError("gradients already unscaled; call update() between steps")
        finite_by_device = {}
        found = False
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter._grad is None:
                    continue
                found = True
                xp = xp_for(parameter.device)
                grad = parameter._grad
                # Divide in FP32; inverse scale can underflow if computed in FP16.
                compute_dtype = xp.float32 if "float16" in str(grad.dtype) else grad.dtype
                grad = grad.astype(compute_dtype) / self._scale
                parameter._grad = grad.astype(parameter.dtype)
                finite = xp.all(xp.isfinite(parameter._grad))
                previous = finite_by_device.get(parameter.device)
                finite_by_device[parameter.device] = (
                    finite if previous is None else previous & finite
                )
        if not found:
            raise RuntimeError("no gradients to unscale")
        overflow = not all(bool(asnumpy(value)) for value in finite_by_device.values())
        self._optimizers[id(optimizer)] = {"overflow": overflow, "stepped": False}
        return not overflow

    def step(self, optimizer):
        if id(optimizer) not in self._optimizers:
            self.unscale_(optimizer)
        state = self._optimizers[id(optimizer)]
        if state["stepped"]:
            raise RuntimeError("optimizer already stepped; call update()")
        state["stepped"] = True
        if not state["overflow"]:
            optimizer.step()
            return True
        return False

    def update(self):
        if not self._optimizers or not all(s["stepped"] for s in self._optimizers.values()):
            raise RuntimeError("call step() before update()")
        if self.enabled:
            if any(state["overflow"] for state in self._optimizers.values()):
                self._scale = max(self._scale * self.backoff_factor, 1e-12)
                self._growth_tracker = 0
            else:
                self._growth_tracker += 1
                if self._growth_tracker >= self.growth_interval:
                    self._scale = min(self._scale * self.growth_factor, 2.0**100)
                    self._growth_tracker = 0
        self._optimizers.clear()

    def state_dict(self):
        return {
            "scale": self._scale,
            "growth_tracker": self._growth_tracker,
            "growth_factor": self.growth_factor,
            "backoff_factor": self.backoff_factor,
            "growth_interval": self.growth_interval,
            "enabled": self.enabled,
        }

    def load_state_dict(self, state):
        restored = GradScaler(
            state["scale"],
            state["growth_factor"],
            state["backoff_factor"],
            state["growth_interval"],
            state["enabled"],
        )
        tracker = state["growth_tracker"]
        if not 0 <= tracker < restored.growth_interval:
            raise ValueError("invalid growth tracker")
        restored._growth_tracker = tracker
        self.__dict__.update(restored.__dict__)
