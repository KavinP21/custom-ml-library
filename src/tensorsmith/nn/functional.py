"""Stateless neural-network operators."""

from __future__ import annotations


from collections.abc import Sequence


import numpy as np


from ..device import add_at, array, asnumpy, xp_for


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


def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    loss = (input - target) ** 2
    return _reduce_loss(loss, reduction)


def binary_cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    eps = 1e-7
    loss = -(
        target * input.clip(eps, 1 - eps).log() + (1 - target) * (1 - input).clip(eps, 1).log()
    )
    return _reduce_loss(loss, reduction)


def binary_cross_entropy_with_logits(
    input: Tensor, target: Tensor, reduction: str = "mean"
) -> Tensor:
    # max(x, 0) - x*y + log(1 + exp(-abs(x)))
    positive = input.relu()
    loss = positive - input * target + (-abs(input)).exp().log1p()
    return _reduce_loss(loss, reduction)


def cross_entropy(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    *,
    axis: int = 1,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Stable indexed cross-entropy; O(tokens * classes), never O(classes²).

    ``axis=-1`` accepts language-model logits [batch, time, vocabulary].
    Ignored labels contribute zero; mean reduction divides by valid tokens.
    """
    if input.ndim < 2:
        raise ValueError("cross_entropy expects logits with at least two dimensions")
    if input.device != target.device or "float" not in str(input.dtype):
        raise ValueError("cross_entropy needs floating logits and targets on the same device")
    if not 0 <= label_smoothing <= 1:
        raise ValueError("label_smoothing must be in [0, 1]")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if target.requires_grad:
        raise ValueError("class targets cannot require gradients")
    if "int" not in str(target.dtype):
        raise TypeError("class targets must have an integer dtype")
    if not -input.ndim <= axis < input.ndim:
        raise ValueError("class axis is out of range")
    axis %= input.ndim
    classes = input.shape[axis]
    if classes == 0:
        raise ValueError("cross_entropy needs at least one class")
    expected = input.shape[:axis] + input.shape[axis + 1 :]
    labels = asnumpy(target._data)
    if labels.shape != expected:
        raise ValueError(f"target shape {labels.shape} does not match logits shape {input.shape}")
    valid = labels != ignore_index
    if np.any(((labels < 0) | (labels >= classes)) & valid):
        raise ValueError("target class index is out of range")
    xp = xp_for(input.device)
    permutation = tuple(i for i in range(input.ndim) if i != axis) + (axis,)
    inverse = tuple(permutation.index(i) for i in range(input.ndim))
    moved_shape = expected + (classes,)
    logits = xp.transpose(input._data, permutation).reshape(-1, classes)
    if "float16" in str(logits.dtype):
        logits = logits.astype(xp.float32)
    indices = array(np.where(valid, labels, 0).reshape(-1).astype(np.int32), input.device)
    active = array(valid.reshape(-1), input.device)
    rows = xp.arange(logits.shape[0])
    shifted = logits - xp.max(logits, axis=-1, keepdims=True)
    exponentials = xp.exp(shifted)
    denominator = xp.sum(exponentials, axis=-1, keepdims=True)
    log_probs = shifted - xp.log(denominator)
    losses = -(1 - label_smoothing) * log_probs[rows, indices]
    if label_smoothing:
        losses = losses - label_smoothing * xp.mean(log_probs, axis=-1)
    losses = xp.where(active, losses, 0)
    count = max(int(valid.sum()), 1)
    if reduction == "none":
        output = losses.reshape(expected)
    else:
        output = xp.sum(losses) / (count if reduction == "mean" else 1)
    probabilities = exponentials / denominator

    def backward(g):
        gradient = probabilities - label_smoothing / classes
        gradient = add_at(gradient, (rows, indices), -(1 - label_smoothing), input.device)
        gradient = xp.where(active[:, None], gradient, 0)
        if reduction == "none":
            gradient = gradient * g.reshape(-1, 1)
        else:
            gradient = gradient * g / (count if reduction == "mean" else 1)
        gradient = xp.transpose(gradient.reshape(moved_shape), inverse)
        return (gradient.astype(input.dtype),)

    return Tensor._from_op(output, (input,), backward, "cross_entropy")


def _reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


__all__ = [
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "cross_entropy",
    "leaky_relu",
    "linear",
    "mse_loss",
    "relu",
    "sigmoid",
    "tanh",
]
