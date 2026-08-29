"""Explicit low-precision training support; no implicit/autocast dtype policy."""

from __future__ import annotations

import math

from .device import asnumpy, xp_for


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
