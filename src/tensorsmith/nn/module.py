"""Module and parameter registration."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from typing import Any

from ..device import DeviceLike, array, device
from ..tensor import Tensor


class Parameter(Tensor):
    """A trainable tensor discovered automatically by :class:`Module`."""

    def __init__(self, data: Any, requires_grad: bool = True, **kwargs: Any) -> None:
        super().__init__(data, requires_grad=requires_grad, **kwargs)


def _walk_value(value: Any, prefix: str, seen: set[int]):
    if isinstance(value, Parameter):
        if id(value) not in seen:
            seen.add(id(value))
            yield prefix, value
    elif isinstance(value, Module):
        yield from value.named_parameters(prefix, seen)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_value(item, f"{prefix}.{index}" if prefix else str(index), seen)
    elif isinstance(value, dict):
        for name, item in value.items():
            yield from _walk_value(item, f"{prefix}.{name}" if prefix else str(name), seen)


class Module:
    """Base class for stateful neural-network components."""

    def __init__(self) -> None:
        self.training = True
        self._buffers: dict[str, Tensor] = {}

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.forward(*args, **kwargs)

    def named_parameters(
        self, prefix: str = "", _seen: set[int] | None = None
    ) -> Iterator[tuple[str, Parameter]]:
        seen = set() if _seen is None else _seen
        for name, value in self.__dict__.items():
            if name in {"training", "_buffers"}:
                continue
            full = f"{prefix}.{name}" if prefix else name
            yield from _walk_value(value, full, seen)

    def parameters(self) -> Iterator[Parameter]:
        for _, parameter in self.named_parameters():
            yield parameter

    def children(self) -> Iterator[Module]:
        seen: set[int] = set()

        def visit(value):
            if isinstance(value, Module) and id(value) not in seen:
                seen.add(id(value))
                yield value
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from visit(item)
            elif isinstance(value, dict):
                for item in value.values():
                    yield from visit(item)

        for name, value in self.__dict__.items():
            if name not in {"training", "_buffers"}:
                yield from visit(value)

    def modules(self) -> Iterator[Module]:
        yield self
        for child in self.children():
            yield from child.modules()

    def train(self, mode: bool = True) -> Module:
        self.training = mode
        for child in self.children():
            child.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    def register_buffer(self, name: str, value: Tensor) -> None:
        if not name.isidentifier():
            raise ValueError(f"invalid buffer name {name!r}")
        self._buffers[name] = value
        setattr(self, name, value)

    def to(self, target: DeviceLike = None, dtype: Any = None) -> Module:
        target_dev = None if target is None else device(target)
        for parameter in self.parameters():
            moved = parameter.to(target_dev, dtype)
            parameter._data = moved._data
            parameter.device = moved.device
            parameter._grad = None
        for module in self.modules():
            for name, value in list(module._buffers.items()):
                moved = value.to(target_dev, dtype)
                module._buffers[name] = moved
                setattr(module, name, moved)
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters():
            parameter.zero_grad(set_to_none)

    def state_dict(self) -> OrderedDict[str, Any]:
        state = OrderedDict(
            (name, parameter.numpy()) for name, parameter in self.named_parameters()
        )
        for module_prefix, module in self.named_modules():
            for name, value in module._buffers.items():
                full = f"{module_prefix}.{name}" if module_prefix else name
                state[full] = value.numpy()
        return state

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
        yield prefix, self
        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                child_prefix = f"{prefix}.{name}" if prefix else name
                yield from value.named_modules(child_prefix)
            elif isinstance(value, (list, tuple)):
                for i, child in enumerate(value):
                    if isinstance(child, Module):
                        child_prefix = f"{prefix}.{name}.{i}" if prefix else f"{name}.{i}"
                        yield from child.named_modules(child_prefix)
            elif isinstance(value, dict):
                for key, child in value.items():
                    if isinstance(child, Module):
                        child_prefix = f"{prefix}.{name}.{key}" if prefix else f"{name}.{key}"
                        yield from child.named_modules(child_prefix)

    def load_state_dict(self, state: Mapping[str, Any], strict: bool = True) -> None:
        destinations: dict[str, Tensor] = dict(self.named_parameters())
        for module_prefix, module in self.named_modules():
            for name, value in module._buffers.items():
                destinations[f"{module_prefix}.{name}" if module_prefix else name] = value
        missing = set(destinations) - set(state)
        unexpected = set(state) - set(destinations)
        if strict and (missing or unexpected):
            raise KeyError(
                f"missing keys: {sorted(missing)}; unexpected keys: {sorted(unexpected)}"
            )
        for name in destinations.keys() & state.keys():
            destination = destinations[name]
            incoming = array(state[name], destination.device, destination.dtype)
            if tuple(incoming.shape) != destination.shape:
                raise ValueError(
                    f"shape mismatch for {name}: {incoming.shape} vs {destination.shape}"
                )
            destination._data = incoming

    def __repr__(self) -> str:
        children = [
            (name, value) for name, value in self.__dict__.items() if isinstance(value, Module)
        ]
        if not children:
            return f"{type(self).__name__}()"
        body = "\n".join(f"  ({name}): {value!r}" for name, value in children)
        return f"{type(self).__name__}(\n{body}\n)"
