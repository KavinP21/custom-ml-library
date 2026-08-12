from __future__ import annotations

import math
from copy import deepcopy

from .optimizer import _validate_lr


class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_epoch = -1

    def get_lr(self) -> list[float]:
        raise NotImplementedError

    def _sync_groups(self):
        """Track appended groups without resetting existing schedule progress."""
        groups = self.optimizer.param_groups
        if len(groups) < len(self.base_lrs):
            raise ValueError("parameter groups cannot be removed from an active scheduler")
        added = [group["lr"] for group in groups[len(self.base_lrs) :]]
        for lr in added:
            _validate_lr(lr)
            if hasattr(self, "min_lr") and lr < self.min_lr:
                raise ValueError("a new group's learning rate is below min_lr")
        self.base_lrs.extend(added)

    def step(self) -> None:
        self._sync_groups()
        self.last_epoch += 1
        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group["lr"] = lr

    def get_last_lr(self) -> list[float]:
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        self._sync_groups()
        return {k: deepcopy(v) for k, v in self.__dict__.items() if k != "optimizer"}

    def load_state_dict(self, state):
        if set(state) != set(self.state_dict()) or len(state["base_lrs"]) != len(
            self.optimizer.param_groups
        ):
            raise ValueError("scheduler state does not match this scheduler")
        self.__dict__.update(deepcopy(state))
        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group["lr"] = lr


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.step_size, self.gamma = step_size, gamma
        super().__init__(optimizer)

    def get_lr(self):
        return [
            lr * self.gamma ** (max(self.last_epoch, 0) // self.step_size) for lr in self.base_lrs
        ]


class ExponentialLR(LRScheduler):
    def __init__(self, optimizer, gamma: float):
        self.gamma = gamma
        super().__init__(optimizer)

    def get_lr(self):
        return [lr * self.gamma ** max(self.last_epoch, 0) for lr in self.base_lrs]


class CosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, t_max: int, eta_min: float = 0.0):
        if t_max <= 0:
            raise ValueError("t_max must be positive")
        self.t_max, self.eta_min = t_max, eta_min
        super().__init__(optimizer)

    def get_lr(self):
        phase = min(max(self.last_epoch, 0), self.t_max)
        return [
            self.eta_min + (lr - self.eta_min) * (1 + math.cos(math.pi * phase / self.t_max)) / 2
            for lr in self.base_lrs
        ]


class LinearLR(LRScheduler):
    def __init__(
        self, optimizer, start_factor: float = 1 / 3, end_factor: float = 1.0, total_iters: int = 5
    ):
        if total_iters <= 0 or start_factor < 0 or end_factor < 0:
            raise ValueError("factors must be non-negative and total_iters positive")
        self.start_factor, self.end_factor, self.total_iters = start_factor, end_factor, total_iters
        super().__init__(optimizer)

    def get_lr(self):
        progress = min(max(self.last_epoch, 0) / self.total_iters, 1.0)
        factor = self.start_factor + progress * (self.end_factor - self.start_factor)
        return [lr * factor for lr in self.base_lrs]


class WarmupCosineLR(LRScheduler):
    """Step-wise warmup then cosine decay; call step AFTER optimizer.step.

    Initialization sets the first update's rate. last_epoch is the number of
    completed updates; the minimum rate is held after total_steps updates.
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr=0.0):
        if warmup_steps < 0 or total_steps <= warmup_steps or min_lr < 0:
            raise ValueError("require 0 <= warmup_steps < total_steps and min_lr >= 0")
        super().__init__(optimizer)
        if any(min_lr > base for base in self.base_lrs):
            raise ValueError("min_lr cannot exceed a base learning rate")
        self.warmup_steps, self.total_steps, self.min_lr = warmup_steps, total_steps, min_lr
        self.last_epoch = 0
        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group["lr"] = lr

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base * (self.last_epoch + 1) / self.warmup_steps for base in self.base_lrs]
        progress = min(
            1.0, (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        )
        return [
            self.min_lr + (base - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
            for base in self.base_lrs
        ]
