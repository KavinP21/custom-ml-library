"""Dynamic reverse-mode automatic differentiation."""

from __future__ import annotations


import math


from collections.abc import Callable, Sequence


from contextlib import ContextDecorator


from contextvars import ContextVar


from typing import Any


import numpy as np


from .device import Device, DeviceLike, array, asnumpy, backend_dtype, copy_array, xp_for


from .device import device as parse_device


_grad_enabled: ContextVar[bool] = ContextVar("tensorsmith_grad_enabled", default=True)


class no_grad(ContextDecorator):
    """Disable graph recording within a block (thread/task local)."""

    def __enter__(self):
        self._token = _grad_enabled.set(False)
        return self

    def __exit__(self, *exc):
        _grad_enabled.reset(self._token)
        return False


class enable_grad(ContextDecorator):
    def __enter__(self):
        self._token = _grad_enabled.set(True)
        return self

    def __exit__(self, *exc):
        _grad_enabled.reset(self._token)
        return False


def is_grad_enabled() -> bool:
    return _grad_enabled.get()


def _is_float_dtype(dtype: Any) -> bool:
    return "float" in str(dtype) or "bfloat" in str(dtype)


def _sum_to_shape(grad: Any, shape: tuple[int, ...], dev: Device):
    """Reverse NumPy broadcasting by summing broadcast dimensions."""
    xp = xp_for(dev)
    while grad.ndim > len(shape):
        grad = xp.sum(grad, axis=0)
    axes = tuple(
        i for i, (got, wanted) in enumerate(zip(grad.shape, shape)) if wanted == 1 and got != 1
    )
    if axes:
        grad = xp.sum(grad, axis=axes, keepdims=True)
    return grad


def _normalize_axis(axis: int | Sequence[int] | None, ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    axes = tuple(a + ndim if a < 0 else a for a in axes)
    if any(a < 0 or a >= ndim for a in axes):
        raise ValueError(f"axis {axis} is out of range for {ndim} dimensions")
    if len(set(axes)) != len(axes):
        raise ValueError(f"duplicate axis in {axis}")
    return axes


Backward = Callable[[Any], tuple[Any | None, ...]]


class Tensor:
    """An n-dimensional array that can participate in a dynamic autograd graph."""

    __array_priority__ = 1000

    def __init__(
        self,
        data: Any,
        *,
        requires_grad: bool = False,
        device: DeviceLike = None,
        dtype: Any = None,
        _copy: bool = False,
    ) -> None:
        if dtype is None and not isinstance(data, Tensor) and not hasattr(data, "dtype"):
            inferred = np.asarray(data).dtype
            if inferred.kind == "f":
                dtype = "float32"
            elif inferred.kind in "iu":
                dtype = "int64"
        if isinstance(data, Tensor):
            source = data
            target = source.device if device is None else parse_device(device)
            raw = array(source._data, target, dtype)
            if _copy or target == source.device:
                raw = copy_array(raw, target)
        else:
            target = parse_device(device)
            raw = array(data, target, dtype)
        if requires_grad and not _is_float_dtype(raw.dtype):
            raise TypeError("only floating-point tensors can require gradients")
        self._data = raw
        self.requires_grad = bool(requires_grad)
        self.device = target
        self._grad: Any | None = None
        self._parents: tuple[Tensor, ...] = ()
        self._backward: Backward | None = None
        self._op = ""
        self._graph_freed = False

    @classmethod
    def _from_op(
        cls, data: Any, parents: tuple[Tensor, ...], backward: Backward, op: str
    ) -> Tensor:
        obj = cls.__new__(cls)
        obj._data = data
        obj.device = parents[0].device
        obj.requires_grad = is_grad_enabled() and any(p.requires_grad for p in parents)
        obj._grad = None
        obj._parents = parents if obj.requires_grad else ()
        obj._backward = backward if obj.requires_grad else None
        obj._op = op if obj.requires_grad else ""
        obj._graph_freed = False
        return obj

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._data.shape)

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    def numel(self) -> int:
        return self.size

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def grad(self) -> Tensor | None:
        return None if self._grad is None else Tensor(self._grad, device=self.device, _copy=False)

    @grad.setter
    def grad(self, value: Tensor | Any | None) -> None:
        self._grad = None if value is None else array(value, self.device)

    @property
    def data(self) -> Tensor:
        return self.detach()

    @data.setter
    def data(self, value: Tensor | Any) -> None:
        raw = array(value, self.device)
        if tuple(raw.shape) != self.shape:
            raise ValueError(
                f"cannot assign data of shape {raw.shape} to tensor of shape {self.shape}"
            )
        self._data = raw

    def numpy(self) -> np.ndarray:
        return asnumpy(self._data).copy()

    def item(self):
        if self.size != 1:
            raise ValueError("item() requires a one-element tensor")
        return self.numpy().item()

    def tolist(self):
        return self.numpy().tolist()

    def detach(self) -> Tensor:
        return Tensor(self._data, device=self.device, _copy=False)

    def clone(self) -> Tensor:
        result = Tensor._from_op(
            copy_array(self._data, self.device), (self,), lambda g: (g,), "clone"
        )
        return result

    def requires_grad_(self, mode: bool = True) -> Tensor:
        if mode and not _is_float_dtype(self.dtype):
            raise TypeError("only floating-point tensors can require gradients")
        self.requires_grad = mode
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._grad = None if set_to_none else xp_for(self.device).zeros_like(self._data)

    def to(self, target: DeviceLike = None, dtype: Any = None) -> Tensor:
        target_dev = self.device if target is None else parse_device(target)
        if target_dev == self.device and (dtype is None or str(dtype) == str(self.dtype)):
            return self
        # Device transfer is intentionally a graph boundary, as in most frameworks.
        return Tensor(self._data, requires_grad=self.requires_grad, device=target_dev, dtype=dtype)

    def cpu(self) -> Tensor:
        return self.to("cpu")

    def astype(self, dtype: Any) -> Tensor:
        data = self._data.astype(backend_dtype(dtype, self.device))
        if not _is_float_dtype(data.dtype):
            return Tensor(data, device=self.device)
        return Tensor._from_op(data, (self,), lambda g: (g.astype(self.dtype),), "cast")

    def float(self) -> Tensor:
        return self.astype("float32")

    def long(self) -> Tensor:
        return self.astype("int64")

    def backward(self, gradient: Tensor | Any | None = None, *, retain_graph: bool = False) -> None:
        """Run a complete reverse-mode pass from this tensor through its tape."""
        if not self.requires_grad:
            raise RuntimeError("cannot call backward() on a tensor that does not require gradients")
        if self._graph_freed:
            raise RuntimeError(
                "this autograd tape has already been freed; pass retain_graph=True to the first backward()"
            )
        xp = xp_for(self.device)
        if gradient is None:
            if self.size != 1:
                raise RuntimeError("gradient must be supplied for non-scalar outputs")
            seed = xp.ones_like(self._data)
        else:
            seed = array(gradient, self.device)
            if tuple(seed.shape) != self.shape:
                raise ValueError(
                    f"gradient shape {seed.shape} does not match output shape {self.shape}"
                )

        _run_backward(self, seed, retain_graph=retain_graph, accumulate=True)

    def _coerce(self, other: Tensor | Any) -> Tensor:
        if isinstance(other, Tensor):
            if other.device != self.device:
                raise ValueError(
                    f"tensors are on different devices: {self.device} and {other.device}"
                )
            return other
        # Python numeric scalars are weakly typed: x / 2 and x ** 2 must
        # preserve a floating x's dtype. Wrapping 2 as a strong int64 array
        # silently promoted NumPy float32 reductions/BatchNorm to float64.
        scalar_dtype = self.dtype if np.isscalar(other) and _is_float_dtype(self.dtype) else None
        return Tensor(other, device=self.device, dtype=scalar_dtype)

    def __add__(self, other: Tensor | Any) -> Tensor:
        other = self._coerce(other)
        data = self._data + other._data
        return Tensor._from_op(
            data,
            (self, other),
            lambda g: (
                _sum_to_shape(g, self.shape, self.device),
                _sum_to_shape(g, other.shape, self.device),
            ),
            "add",
        )

    __radd__ = __add__

    def __neg__(self) -> Tensor:
        return Tensor._from_op(-self._data, (self,), lambda g: (-g,), "neg")

    def __sub__(self, other: Tensor | Any) -> Tensor:
        return self + -self._coerce(other)

    def __rsub__(self, other: Tensor | Any) -> Tensor:
        return self._coerce(other) - self

    def __mul__(self, other: Tensor | Any) -> Tensor:
        other = self._coerce(other)
        data = self._data * other._data
        return Tensor._from_op(
            data,
            (self, other),
            lambda g: (
                _sum_to_shape(g * other._data, self.shape, self.device),
                _sum_to_shape(g * self._data, other.shape, self.device),
            ),
            "mul",
        )

    __rmul__ = __mul__

    def __truediv__(self, other: Tensor | Any) -> Tensor:
        other = self._coerce(other)
        data = self._data / other._data
        return Tensor._from_op(
            data,
            (self, other),
            lambda g: (
                _sum_to_shape(g / other._data, self.shape, self.device),
                _sum_to_shape(-g * self._data / (other._data**2), other.shape, self.device),
            ),
            "div",
        )

    def __rtruediv__(self, other: Tensor | Any) -> Tensor:
        return self._coerce(other) / self

    def __pow__(self, exponent: Tensor | float | int) -> Tensor:
        exponent = self._coerce(exponent)
        xp = xp_for(self.device)
        data = self._data**exponent._data

        def backward(g):
            ga = g * exponent._data * (self._data ** (exponent._data - 1))
            gb = g * data * xp.log(self._data) if exponent.requires_grad else None
            return (
                _sum_to_shape(ga, self.shape, self.device),
                None if gb is None else _sum_to_shape(gb, exponent.shape, self.device),
            )

        return Tensor._from_op(data, (self, exponent), backward, "pow")

    def __matmul__(self, other: Tensor | Any) -> Tensor:
        other = self._coerce(other)
        xp = xp_for(self.device)
        data = xp.matmul(self._data, other._data)

        def backward(g):
            a, b = self._data, other._data
            a1, b1 = a.ndim == 1, b.ndim == 1
            aa = xp.expand_dims(a, -2) if a1 else a
            bb = xp.expand_dims(b, -1) if b1 else b
            gg = g
            if a1:
                gg = xp.expand_dims(gg, -2)
            if b1:
                gg = xp.expand_dims(gg, -1)
            ga = xp.matmul(gg, xp.swapaxes(bb, -1, -2))
            gb = xp.matmul(xp.swapaxes(aa, -1, -2), gg)
            if a1:
                ga = xp.squeeze(ga, axis=-2)
            if b1:
                gb = xp.squeeze(gb, axis=-1)
            return _sum_to_shape(ga, self.shape, self.device), _sum_to_shape(
                gb, other.shape, self.device
            )

        return Tensor._from_op(data, (self, other), backward, "matmul")

    def __rmatmul__(self, other: Tensor | Any) -> Tensor:
        return self._coerce(other) @ self

    def sum(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> Tensor:
        xp = xp_for(self.device)
        axes = _normalize_axis(axis, self.ndim)
        data = xp.sum(self._data, axis=None if axis is None else axes, keepdims=keepdims)

        def backward(g):
            if not keepdims:
                for ax in sorted(axes):
                    g = xp.expand_dims(g, ax)
            return (xp.broadcast_to(g, self.shape),)

        return Tensor._from_op(data, (self,), backward, "sum")

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of a scalar tensor")
        return self.shape[0]

    def __repr__(self) -> str:
        suffix = f", device='{self.device}'" if self.device.type != "cpu" else ""
        grad = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.numpy()!r}{suffix}{grad})"


def _run_backward(root, seed, *, retain_graph=False, accumulate=False, requested=()):
    """One first-order VJP engine for backward(), grad(), and recomputation."""
    topo, visited = [], set()
    stack = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        ident = id(node)
        if expanded:
            topo.append(node)
            continue
        if ident in visited:
            continue
        if node._graph_freed:
            raise RuntimeError(
                "autograd tape already freed; use retain_graph=True on the first pass"
            )
        visited.add(ident)
        stack.append((node, True))
        stack.extend((parent, False) for parent in node._parents)
    requested = set(requested)
    result, pending = {}, {id(root): seed}
    for node in reversed(topo):
        contribution = pending.pop(id(node), None)
        if contribution is None:
            continue
        # VJPs into an explicitly mixed-dtype operation must return the
        # destination tensor's gradient precision, not the output's dtype.
        if node.requires_grad and contribution.dtype != node.dtype:
            contribution = contribution.astype(node.dtype)
        if id(node) in requested:
            result[id(node)] = contribution
        if accumulate and node.requires_grad:
            node._grad = contribution if node._grad is None else node._grad + contribution
        if node._backward is not None:
            parent_grads = node._backward(contribution)
            if len(parent_grads) != len(node._parents):
                raise RuntimeError(f"internal error: {node._op} returned wrong number of gradients")
            for parent, parent_grad in zip(node._parents, parent_grads):
                if parent_grad is None or not parent.requires_grad:
                    continue
                key = id(parent)
                pending[key] = parent_grad if key not in pending else pending[key] + parent_grad
    if not retain_graph:
        for node in topo:
            if node._backward is not None:
                node._parents = ()
                node._backward = None
                node._graph_freed = True
    return result


def tensor(data: Any, **kwargs: Any) -> Tensor:
    return Tensor(data, **kwargs)


def _shape_args(shape: tuple[Any, ...]) -> tuple[int, ...]:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(shape[0])
    return tuple(shape)


def zeros(
    *shape: int, device: DeviceLike = None, dtype: Any = "float32", requires_grad=False
) -> Tensor:
    dev = parse_device(device)
    return Tensor(
        xp_for(dev).zeros(_shape_args(shape), dtype=backend_dtype(dtype, dev)),
        device=dev,
        requires_grad=requires_grad,
    )


def ones(
    *shape: int, device: DeviceLike = None, dtype: Any = "float32", requires_grad=False
) -> Tensor:
    dev = parse_device(device)
    return Tensor(
        xp_for(dev).ones(_shape_args(shape), dtype=backend_dtype(dtype, dev)),
        device=dev,
        requires_grad=requires_grad,
    )


def empty(
    *shape: int, device: DeviceLike = None, dtype: Any = "float32", requires_grad=False
) -> Tensor:
    dev = parse_device(device)
    return Tensor(
        xp_for(dev).empty(_shape_args(shape), dtype=backend_dtype(dtype, dev)),
        device=dev,
        requires_grad=requires_grad,
    )


def randn(
    *shape: int, device: DeviceLike = None, dtype: Any = "float32", requires_grad=False
) -> Tensor:
    # Host RNG gives exactly reproducible initialization across every backend.
    data = np.random.standard_normal(_shape_args(shape)).astype(dtype)
    return Tensor(data, device=device, requires_grad=requires_grad)


def rand(
    *shape: int, device: DeviceLike = None, dtype: Any = "float32", requires_grad=False
) -> Tensor:
    data = np.random.random(_shape_args(shape)).astype(dtype)
    return Tensor(data, device=device, requires_grad=requires_grad)


def arange(*args: int, device: DeviceLike = None, dtype: Any = None) -> Tensor:
    dev = parse_device(device)
    kwargs = {} if dtype is None else {"dtype": backend_dtype(dtype, dev)}
    return Tensor(xp_for(dev).arange(*args, **kwargs), device=dev)


def zeros_like(value: Tensor, *, requires_grad: bool = False) -> Tensor:
    return Tensor(
        xp_for(value.device).zeros_like(value._data),
        device=value.device,
        requires_grad=requires_grad,
    )


def ones_like(value: Tensor, *, requires_grad: bool = False) -> Tensor:
    return Tensor(
        xp_for(value.device).ones_like(value._data),
        device=value.device,
        requires_grad=requires_grad,
    )
