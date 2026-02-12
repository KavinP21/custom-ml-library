"""Dynamic reverse-mode automatic differentiation."""

from __future__ import annotations


import math


from collections.abc import Callable, Sequence


from contextlib import ContextDecorator


from contextvars import ContextVar


from itertools import pairwise


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

    def __rpow__(self, base: Tensor | float | int) -> Tensor:
        return self._coerce(base) ** self

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

    def mean(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> Tensor:
        axes = _normalize_axis(axis, self.ndim)
        count = math.prod(self.shape[a] for a in axes)
        if "float16" in str(self.dtype):
            xp = xp_for(self.device)
            # Half sums/denominators can overflow even for a finite mean.
            output = xp.mean(self._data.astype(xp.float32), axis=axes, keepdims=keepdims).astype(
                self.dtype
            )

            def backward(g):
                g = g.astype(xp.float32) / count
                if not keepdims:
                    for ax in sorted(axes):
                        g = xp.expand_dims(g, ax)
                return (xp.broadcast_to(g, self.shape).astype(self.dtype),)

            return Tensor._from_op(output, (self,), backward, "mean")
        return self.sum(axis, keepdims) / count

    def var(
        self,
        axis: int | Sequence[int] | None = None,
        keepdims: bool = False,
        correction: int = 1,
    ) -> Tensor:
        axes = _normalize_axis(axis, self.ndim)
        count = math.prod(self.shape[a] for a in axes)
        if count - correction <= 0:
            raise ValueError("variance degrees of freedom must be positive")
        if "float16" in str(self.dtype):
            xp = xp_for(self.device)
            working = self._data.astype(xp.float32)
            centered = working - xp.mean(working, axis=axes, keepdims=True)
            output = (
                xp.sum(centered * centered, axis=axes, keepdims=keepdims) / (count - correction)
            ).astype(self.dtype)

            def backward(g):
                g = g.astype(xp.float32)
                if not keepdims:
                    for ax in sorted(axes):
                        g = xp.expand_dims(g, ax)
                return ((2 * centered * g / (count - correction)).astype(self.dtype),)

            return Tensor._from_op(output, (self,), backward, "var")
        centered = self - self.mean(axis, keepdims=True)
        result = (centered * centered).sum(axis, keepdims=keepdims) / (count - correction)
        return result

    def std(self, axis=None, keepdims: bool = False, correction: int = 1) -> Tensor:
        return self.var(axis, keepdims, correction).sqrt()

    def max(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> Tensor:
        xp = xp_for(self.device)
        axes = _normalize_axis(axis, self.ndim)
        reduced = xp.max(self._data, axis=None if axis is None else axes, keepdims=True)
        data = reduced if keepdims else xp.squeeze(reduced, axis=axes)

        def backward(g):
            if not keepdims:
                for ax in sorted(axes):
                    g = xp.expand_dims(g, ax)
            mask = self._data == reduced
            ties = xp.sum(mask, axis=axes, keepdims=True)
            return (mask * xp.broadcast_to(g, self.shape) / ties,)

        return Tensor._from_op(data, (self,), backward, "max")

    def min(self, axis: int | Sequence[int] | None = None, keepdims: bool = False) -> Tensor:
        return -(-self).max(axis, keepdims)

    def reshape(self, *shape: int | tuple[int, ...]) -> Tensor:
        final = (
            tuple(shape[0])
            if len(shape) == 1 and isinstance(shape[0], (tuple, list))
            else tuple(shape)
        )
        data = self._data.reshape(final)
        return Tensor._from_op(data, (self,), lambda g: (g.reshape(self.shape),), "reshape")

    view = reshape

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Tensor:
        start_dim %= self.ndim
        end_dim %= self.ndim
        if end_dim < start_dim:
            raise ValueError("end_dim must be >= start_dim")
        merged = math.prod(self.shape[start_dim : end_dim + 1])
        return self.reshape(self.shape[:start_dim] + (merged,) + self.shape[end_dim + 1 :])

    def transpose(self, dim0: int, dim1: int) -> Tensor:
        axes = list(range(self.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return self.permute(*axes)

    def permute(self, *dims: int | tuple[int, ...]) -> Tensor:
        dims = (
            tuple(dims[0]) if len(dims) == 1 and isinstance(dims[0], (tuple, list)) else tuple(dims)
        )
        if sorted(d % self.ndim for d in dims) != list(range(self.ndim)):
            raise ValueError(f"{dims} is not a permutation of {self.ndim} dimensions")
        dims = tuple(d % self.ndim for d in dims)
        inverse = tuple(dims.index(i) for i in range(self.ndim))
        xp = xp_for(self.device)
        return Tensor._from_op(
            xp.transpose(self._data, dims),
            (self,),
            lambda g: (xp.transpose(g, inverse),),
            "permute",
        )

    @property
    def T(self) -> Tensor:
        return self.permute(*reversed(range(self.ndim)))

    def unsqueeze(self, dim: int) -> Tensor:
        dim = dim + self.ndim + 1 if dim < 0 else dim
        if dim < 0 or dim > self.ndim:
            raise ValueError(f"dimension {dim} is out of range for unsqueeze")
        return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])

    def squeeze(self, dim: int | None = None) -> Tensor:
        if dim is None:
            shape = tuple(s for s in self.shape if s != 1)
        else:
            dim %= self.ndim
            if self.shape[dim] != 1:
                return self
            shape = self.shape[:dim] + self.shape[dim + 1 :]
        return self.reshape(shape)

    def broadcast_to(self, shape: Sequence[int]) -> Tensor:
        xp = xp_for(self.device)
        shape = tuple(shape)
        return Tensor._from_op(
            xp.broadcast_to(self._data, shape),
            (self,),
            lambda g: (_sum_to_shape(g, self.shape, self.device),),
            "broadcast",
        )

    def __getitem__(self, key: Any) -> Tensor:
        from .device import add_at

        if isinstance(key, Tensor):
            if key.device != self.device:
                raise ValueError("index tensor must be on the same device")
            key = key._data
        elif isinstance(key, tuple):
            normalized = []
            for item in key:
                if isinstance(item, Tensor):
                    if item.device != self.device:
                        raise ValueError("index tensor must be on the same device")
                    item = item._data
                normalized.append(item)
            key = tuple(normalized)
        data = self._data[key]

        def backward(g):
            base = xp_for(self.device).zeros_like(self._data)
            return (add_at(base, key, g, self.device),)

        return Tensor._from_op(data, (self,), backward, "slice")

    def exp(self) -> Tensor:
        xp = xp_for(self.device)
        data = xp.exp(self._data)
        return Tensor._from_op(data, (self,), lambda g: (g * data,), "exp")

    def log(self) -> Tensor:
        xp = xp_for(self.device)
        return Tensor._from_op(xp.log(self._data), (self,), lambda g: (g / self._data,), "log")

    def log1p(self) -> Tensor:
        return (self + 1).log()

    def sqrt(self) -> Tensor:
        return self**0.5

    def tanh(self) -> Tensor:
        xp = xp_for(self.device)
        data = xp.tanh(self._data)
        return Tensor._from_op(data, (self,), lambda g: (g * (1 - data**2),), "tanh")

    def sin(self) -> Tensor:
        xp = xp_for(self.device)
        return Tensor._from_op(
            xp.sin(self._data), (self,), lambda g: (g * xp.cos(self._data),), "sin"
        )

    def cos(self) -> Tensor:
        xp = xp_for(self.device)
        return Tensor._from_op(
            xp.cos(self._data), (self,), lambda g: (-g * xp.sin(self._data),), "cos"
        )

    def sigmoid(self) -> Tensor:
        xp = xp_for(self.device)
        # Split form avoids overflow on either half of the number line.
        positive = 1 / (1 + xp.exp(-xp.maximum(self._data, 0)))
        negative_exp = xp.exp(xp.minimum(self._data, 0))
        negative = negative_exp / (1 + negative_exp)
        data = xp.where(self._data >= 0, positive, negative)
        return Tensor._from_op(data, (self,), lambda g: (g * data * (1 - data),), "sigmoid")

    def relu(self) -> Tensor:
        xp = xp_for(self.device)
        data = xp.maximum(self._data, 0)
        return Tensor._from_op(data, (self,), lambda g: (g * (self._data > 0),), "relu")

    def abs(self) -> Tensor:
        xp = xp_for(self.device)
        data = xp.abs(self._data)
        return Tensor._from_op(data, (self,), lambda g: (g * xp.sign(self._data),), "abs")

    __abs__ = abs

    def clip(self, minimum: float | None = None, maximum: float | None = None) -> Tensor:
        xp = xp_for(self.device)
        data = self._data
        mask = xp.ones_like(data, dtype=backend_dtype(bool, self.device))
        if minimum is not None:
            data = xp.maximum(data, minimum)
            mask = mask & (self._data >= minimum)
        if maximum is not None:
            data = xp.minimum(data, maximum)
            mask = mask & (self._data <= maximum)
        return Tensor._from_op(data, (self,), lambda g: (g * mask,), "clip")

    def softmax(self, dim: int = -1) -> Tensor:
        shifted = self - self.max(dim, keepdims=True).detach()
        exps = shifted.exp()
        return exps / exps.sum(dim, keepdims=True)

    def log_softmax(self, dim: int = -1) -> Tensor:
        shifted = self - self.max(dim, keepdims=True).detach()
        return shifted - shifted.exp().sum(dim, keepdims=True).log()

    def argmax(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        xp = xp_for(self.device)
        try:
            data = xp.argmax(self._data, axis=axis, keepdims=keepdims)
        except TypeError:
            data = xp.argmax(self._data, axis=axis)
            if keepdims and axis is not None:
                data = xp.expand_dims(data, axis)
        return Tensor(data, device=self.device)

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


def grad(
    output: Tensor,
    inputs: Tensor | Sequence[Tensor],
    grad_outputs=None,
    *,
    retain_graph=False,
    allow_unused=False,
) -> tuple[Tensor | None, ...]:
    """Return first-order gradients without modifying any .grad buffers.

    One output tensor; supply an upstream gradient for nonscalar outputs.
    No create_graph/higher-order differentiation is implemented.
    """
    inputs = (inputs,) if isinstance(inputs, Tensor) else tuple(inputs)
    if not output.requires_grad or any(not value.requires_grad for value in inputs):
        raise RuntimeError("output and requested inputs must require gradients")
    if any(value.device != output.device for value in inputs):
        raise ValueError("grad inputs and output must be on the same device")
    if grad_outputs is None:
        if output.size != 1:
            raise RuntimeError("supply grad_outputs for a nonscalar output")
        seed = xp_for(output.device).ones_like(output._data)
    else:
        seed = array(grad_outputs, output.device)
        if tuple(seed.shape) != output.shape:
            raise ValueError("upstream gradient shape mismatch")
    values = _run_backward(
        output, seed, retain_graph=retain_graph, requested=(id(x) for x in inputs)
    )
    if not allow_unused and any(id(x) not in values for x in inputs):
        raise RuntimeError("a requested input is not used by the output")
    return tuple(
        Tensor(values[id(x)], device=x.device) if id(x) in values else None for x in inputs
    )


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


def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    if not tensors:
        raise ValueError("cat expects at least one tensor")
    first = tensors[0]
    if any(t.device != first.device for t in tensors):
        raise ValueError("all tensors must be on the same device")
    xp = xp_for(first.device)
    data = xp.concatenate([t._data for t in tensors], axis=dim)
    offsets = np.cumsum([0] + [t.shape[dim] for t in tensors])

    def backward(g):
        grads = []
        for start, stop in pairwise(offsets):
            key = [slice(None)] * g.ndim
            key[dim] = slice(int(start), int(stop))
            grads.append(g[tuple(key)])
        return tuple(grads)

    return Tensor._from_op(data, tuple(tensors), backward, "cat")


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    return cat([t.unsqueeze(dim) for t in tensors], dim=dim)


def maximum(left: Tensor | Any, right: Tensor | Any) -> Tensor:
    left = (
        left
        if isinstance(left, Tensor)
        else Tensor(left, device=right.device if isinstance(right, Tensor) else None)
    )
    right = left._coerce(right)
    xp = xp_for(left.device)
    data = xp.maximum(left._data, right._data)

    def backward(g):
        left_wins = left._data > right._data
        ties = left._data == right._data
        gl = g * (left_wins + 0.5 * ties)
        gr = g * ((right._data > left._data) + 0.5 * ties)
        return _sum_to_shape(gl, left.shape, left.device), _sum_to_shape(
            gr, right.shape, right.device
        )

    return Tensor._from_op(data, (left, right), backward, "maximum")


def minimum(left: Tensor | Any, right: Tensor | Any) -> Tensor:
    return -maximum(
        -left if isinstance(left, Tensor) else -np.asarray(left),
        -right if isinstance(right, Tensor) else -np.asarray(right),
    )


def where(condition: Tensor | Any, left: Tensor | Any, right: Tensor | Any) -> Tensor:
    reference = left if isinstance(left, Tensor) else right if isinstance(right, Tensor) else None
    if reference is None:
        raise TypeError("where requires at least one Tensor result")
    left = reference._coerce(left)
    right = reference._coerce(right)
    if isinstance(condition, Tensor):
        if condition.device != reference.device:
            raise ValueError("condition must be on the same device")
        raw_condition = condition._data
    else:
        raw_condition = array(condition, reference.device)
    xp = xp_for(reference.device)
    data = xp.where(raw_condition, left._data, right._data)
    return Tensor._from_op(
        data,
        (left, right),
        lambda g: (
            _sum_to_shape(xp.where(raw_condition, g, 0), left.shape, left.device),
            _sum_to_shape(xp.where(raw_condition, 0, g), right.shape, right.device),
        ),
        "where",
    )
