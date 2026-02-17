"""Stateless neural-network operators."""

from __future__ import annotations


from collections.abc import Sequence


from ..device import xp_for


from ..tensor import Tensor, _sum_to_shape


def _single(value: int | Sequence[int]) -> tuple[int]:
    return (value,) if isinstance(value, int) else tuple(value)


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else tuple(value)


def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Dense projection as one tape node, using a flattened backend GEMM."""
    if input.ndim < 1 or weight.ndim != 2 or input.shape[-1] != weight.shape[1]:
        raise ValueError("linear expects [...,in_features] and [out_features,in_features]")
    parents = (input, weight) if bias is None else (input, weight, bias)
    if any(p.device != input.device for p in parents):
        raise ValueError("linear tensors must be on the same device")
    if bias is not None and bias.shape != (weight.shape[0],):
        raise ValueError("linear bias must have shape [out_features]")
    xp = xp_for(input.device)
    x = input._data.reshape(-1, weight.shape[1])
    output = xp.matmul(x, weight._data.T)
    if bias is not None:
        output = output + bias._data
    output = output.reshape(input.shape[:-1] + (weight.shape[0],))

    def backward(g):
        flat = g.reshape(-1, weight.shape[0])
        gradients = [
            xp.matmul(flat, weight._data).reshape(input.shape).astype(input.dtype),
            xp.matmul(flat.T, x).astype(weight.dtype),
        ]
        if bias is not None:
            gradients.append(_sum_to_shape(g, bias.shape, input.device).astype(bias.dtype))
        return tuple(gradients)

    return Tensor._from_op(output, parents, backward, "linear")


def relu(input: Tensor) -> Tensor:
    return input.relu()


def leaky_relu(input: Tensor, negative_slope: float = 0.01) -> Tensor:
    return input.relu() - negative_slope * (-input).relu()


def sigmoid(input: Tensor) -> Tensor:
    return input.sigmoid()


def tanh(input: Tensor) -> Tensor:
    return input.tanh()


__all__ = ["leaky_relu", "linear", "relu", "sigmoid", "tanh"]
