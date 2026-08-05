from __future__ import annotations

import math

from ..device import xp_for
from .optimizer import Optimizer, _validate_lr


class Adam(Optimizer):
    """Adam with coupled L2 decay (use :class:`AdamW` for decoupled decay)."""

    decoupled_weight_decay = False

    def _validate_param_group(self, group):
        super()._validate_param_group(group)
        betas = group["betas"]
        if len(betas) != 2 or any(not math.isfinite(v) or not 0 <= v < 1 for v in betas):
            raise ValueError("betas must be finite values in [0, 1)")
        if any(not math.isfinite(group[k]) or group[k] < 0 for k in ("eps", "weight_decay")):
            raise ValueError("eps and weight_decay must be finite and non-negative")
        if not isinstance(group["amsgrad"], bool):
            raise TypeError("amsgrad must be boolean")

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        _validate_lr(lr)
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("betas must be in [0, 1)")
        if eps < 0 or weight_decay < 0:
            raise ValueError("eps and weight_decay must be non-negative")
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "amsgrad": amsgrad,
            },
        )

    def step(self) -> None:
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter._grad is None:
                    continue
                xp = xp_for(parameter.device)
                low_precision = "float16" in str(parameter.dtype)
                compute_dtype = xp.float32 if low_precision else parameter.dtype
                grad = parameter._grad.astype(compute_dtype)
                if group["weight_decay"] and not self.decoupled_weight_decay:
                    grad = grad + group["weight_decay"] * parameter._data.astype(compute_dtype)
                if id(parameter) not in self.state:
                    self.state[id(parameter)] = {
                        "step": 0,
                        "exp_avg": xp.zeros(parameter.shape, dtype=compute_dtype),
                        "exp_avg_sq": xp.zeros(parameter.shape, dtype=compute_dtype),
                    }
                state = self.state[id(parameter)]
                state["step"] += 1
                if low_precision and "master_weight" not in state:
                    state["master_weight"] = parameter._data.astype(xp.float32)
                state["exp_avg"] = beta1 * state["exp_avg"] + (1 - beta1) * grad
                state["exp_avg_sq"] = beta2 * state["exp_avg_sq"] + (1 - beta2) * grad * grad
                if group["amsgrad"]:
                    if "max_exp_avg_sq" not in state:
                        state["max_exp_avg_sq"] = xp.zeros(parameter.shape, dtype=compute_dtype)
                    previous = state["max_exp_avg_sq"]
                    state["max_exp_avg_sq"] = xp.maximum(previous, state["exp_avg_sq"])
                    denominator_sq = state["max_exp_avg_sq"]
                else:
                    denominator_sq = state["exp_avg_sq"]
                step = state["step"]
                corrected_first = state["exp_avg"] / (1 - beta1**step)
                corrected_second = denominator_sq / (1 - beta2**step)
                weight = state["master_weight"] if low_precision else parameter._data
                if group["weight_decay"] and self.decoupled_weight_decay:
                    weight = weight * (1 - group["lr"] * group["weight_decay"])
                weight = weight - group["lr"] * corrected_first / (
                    xp.sqrt(corrected_second) + group["eps"]
                )
                if low_precision:
                    state["master_weight"] = weight
                parameter._data = weight.astype(parameter.dtype)


class AdamW(Adam):
    """Adam with decoupled weight decay."""

    decoupled_weight_decay = True
