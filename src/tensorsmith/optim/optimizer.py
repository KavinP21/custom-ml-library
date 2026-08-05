"""Optimizer base class."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from ..device import array, asnumpy
from ..nn.module import Parameter


class Optimizer:
    def __init__(self, params: Iterable[Parameter], defaults: dict[str, Any]):
        entries = list(params)
        if not entries:
            raise ValueError("optimizer received an empty parameter list")
        self.defaults = deepcopy(defaults)
        self.param_groups = []
        self.state: dict[int, dict[str, Any]] = {}
        if isinstance(entries[0], Mapping):
            if any(not isinstance(entry, Mapping) for entry in entries):
                raise TypeError("do not mix parameter groups and individual parameters")
            for entry in entries:
                self.add_param_group(entry)
        else:
            self.add_param_group({"params": entries})

    def _validate_param_group(self, group):
        _validate_lr(group["lr"])

    def add_param_group(self, group: Mapping[str, Any]) -> None:
        """Add parameters with per-group options, e.g. when unfreezing a layer.

        Parameters may occur only once across the optimizer. New groups inherit
        constructor defaults; the input dictionary is never mutated.
        """
        if not isinstance(group, Mapping) or "params" not in group:
            raise TypeError("a parameter group must be a mapping containing params")
        raw = group["params"]
        parameters = [raw] if isinstance(raw, Parameter) else list(raw)
        if not parameters:
            raise ValueError("a parameter group cannot be empty")
        if any(not isinstance(p, Parameter) for p in parameters):
            raise TypeError("optimizers expect Parameter objects")
        existing = {id(p) for current in self.param_groups for p in current["params"]}
        ids = {id(p) for p in parameters}
        if len(ids) != len(parameters) or existing & ids:
            raise ValueError("a parameter appears more than once")
        options = {
            **deepcopy(self.defaults),
            **{k: deepcopy(v) for k, v in group.items() if k != "params"},
        }
        candidate = {**options, "params": parameters}
        self._validate_param_group(candidate)
        self.param_groups.append(candidate)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for parameter in group["params"]:
                parameter.zero_grad(set_to_none)

    def step(self) -> None:
        raise NotImplementedError

    def state_dict(self) -> dict[str, Any]:
        packed_state = {}
        parameter_ids = {}
        index = 0
        for group in self.param_groups:
            for parameter in group["params"]:
                parameter_ids[id(parameter)] = index
                if id(parameter) in self.state:
                    packed_state[index] = {
                        key: asnumpy(value).copy() if hasattr(value, "shape") else deepcopy(value)
                        for key, value in self.state[id(parameter)].items()
                    }
                index += 1
        groups = [
            {
                key: ([parameter_ids[id(p)] for p in value] if key == "params" else value)
                for key, value in group.items()
            }
            for group in self.param_groups
        ]
        return {"state": packed_state, "param_groups": groups}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore portable optimizer state by parameter order, not object IDs.

        Construct/load the model on its final device/dtype before restoring the
        optimizer. Array state retains its saved precision (FP32 Adam masters).
        """
        groups = state["param_groups"]
        if len(groups) != len(self.param_groups):
            raise ValueError("optimizer parameter-group count mismatch")
        restored, new_groups, seen = {}, [], set()
        for incoming, current in zip(groups, self.param_groups):
            if len(incoming["params"]) != len(current["params"]):
                raise ValueError("optimizer parameter count mismatch")
            indices = incoming["params"]
            if (
                any(not isinstance(i, int) or i < 0 for i in indices)
                or len(set(indices)) != len(indices)
                or seen & set(indices)
            ):
                raise ValueError("optimizer checkpoint contains invalid parameter indices")
            seen.update(indices)
            candidate = {
                **deepcopy(self.defaults),
                **deepcopy(incoming),
                "params": current["params"],
            }
            self._validate_param_group(candidate)
            new_groups.append(candidate)
            for index, parameter in zip(incoming["params"], current["params"]):
                values = {}
                for key, value in state["state"].get(index, {}).items():
                    if hasattr(value, "shape"):
                        if tuple(value.shape) != parameter.shape:
                            raise ValueError(f"optimizer state {key} shape mismatch")
                        values[key] = array(asnumpy(value).copy(), parameter.device)
                    else:
                        values[key] = deepcopy(value)
                if values:
                    restored[id(parameter)] = values
        self.param_groups, self.state = new_groups, restored


def _validate_lr(lr: float) -> None:
    if lr < 0 or not math.isfinite(lr):
        raise ValueError(f"invalid learning rate: {lr}")
