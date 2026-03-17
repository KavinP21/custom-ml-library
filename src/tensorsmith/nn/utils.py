from __future__ import annotations

import math

from ..device import xp_for


def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0) -> float:
    if math.isnan(max_norm) or math.isnan(norm_type) or max_norm < 0 or norm_type <= 0:
        raise ValueError("max_norm must be nonnegative and norm_type positive")
    parameters = [p for p in parameters if p._grad is not None]
    if not parameters:
        return 0.0
    # One host synchronization per device, not one per parameter. In lazy
    # backends the latter serializes otherwise independent reduction kernels.
    reductions = {}
    for parameter in parameters:
        xp = xp_for(parameter.device)
        gradient = parameter._grad
        if "float16" in str(gradient.dtype):
            gradient = gradient.astype(xp.float32)
        value = (
            xp.max(xp.abs(gradient))
            if norm_type == float("inf")
            else (xp.abs(gradient) ** norm_type).sum()
        )
        reductions.setdefault(parameter.device, []).append(value)
    device_totals = []
    for device, values in reductions.items():
        xp = xp_for(device)
        stacked = xp.stack(values)
        device_totals.append(float(xp.max(stacked) if norm_type == float("inf") else stacked.sum()))
    total = (
        max(device_totals) if norm_type == float("inf") else sum(device_totals) ** (1 / norm_type)
    )
    if not math.isfinite(total):
        raise FloatingPointError("non-finite gradient norm; unscale/check overflow before clipping")
    coefficient = min(1.0, max_norm / (total + 1e-6))
    for parameter in parameters:
        parameter._grad = parameter._grad * coefficient
    return total


def clip_grad_value_(parameters, clip_value: float) -> None:
    if math.isnan(clip_value) or clip_value < 0:
        raise ValueError("clip_value must be nonnegative")
    for parameter in parameters:
        if parameter._grad is not None:
            xp = xp_for(parameter.device)
            parameter._grad = xp.minimum(xp.maximum(parameter._grad, -clip_value), clip_value)
