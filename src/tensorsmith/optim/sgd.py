from __future__ import annotations

from .optimizer import Optimizer, _validate_lr


class SGD(Optimizer):
    """Stochastic gradient descent with momentum, dampening, and Nesterov momentum."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        _validate_lr(lr)
        if momentum < 0 or dampening < 0 or weight_decay < 0:
            raise ValueError("momentum, dampening, and weight_decay must be non-negative")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and zero dampening")
        super().__init__(
            params,
            {
                "lr": lr,
                "momentum": momentum,
                "dampening": dampening,
                "weight_decay": weight_decay,
                "nesterov": nesterov,
            },
        )

    def step(self) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter._grad is None:
                    continue
                grad = parameter._grad
                if group["weight_decay"]:
                    grad = grad + group["weight_decay"] * parameter._data
                if group["momentum"]:
                    state = self.state.setdefault(id(parameter), {})
                    if "momentum_buffer" not in state:
                        buffer = grad
                    else:
                        buffer = state["momentum_buffer"] * group["momentum"] + grad * (
                            1 - group["dampening"]
                        )
                    state["momentum_buffer"] = buffer
                    grad = grad + group["momentum"] * buffer if group["nesterov"] else buffer
                parameter._data = parameter._data - group["lr"] * grad
