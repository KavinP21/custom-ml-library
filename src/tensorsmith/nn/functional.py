"""Stateless neural-network operators."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..amp import _cast_inputs
from ..device import add_at, array, asnumpy, assign_add, random_uniform, xp_for
from ..tensor import Tensor, _sum_to_shape
from .attention import rotary_embedding, scaled_dot_product_attention
from .primitives import (
    bias_gelu,
    layer_norm,
    residual_layer_norm,
    residual_rms_norm,
    rms_norm,
    silu,
)
from .primitives import gelu as _gelu

__all__ = [
    "avg_pool1d",
    "avg_pool2d",
    "bias_gelu",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
    "conv1d",
    "conv2d",
    "cross_entropy",
    "dropout",
    "embedding",
    "gelu",
    "layer_norm",
    "leaky_relu",
    "linear",
    "log_softmax",
    "max_pool1d",
    "max_pool2d",
    "mse_loss",
    "relu",
    "residual_layer_norm",
    "residual_rms_norm",
    "rms_norm",
    "rotary_embedding",
    "scaled_dot_product_attention",
    "sigmoid",
    "silu",
    "softmax",
    "tanh",
]


def _single(value: int | Sequence[int]) -> tuple[int]:
    return (value,) if isinstance(value, int) else tuple(value)


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    return (value, value) if isinstance(value, int) else tuple(value)


def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Dense projection as one tape node, using a flattened backend GEMM."""
    input, weight, bias = _cast_inputs(input, weight, bias)
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


def gelu(input: Tensor) -> Tensor:
    return _gelu(input)


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    return input.softmax(dim)


def log_softmax(input: Tensor, dim: int = -1) -> Tensor:
    return input.log_softmax(dim)


def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    if not 0 <= p < 1:
        raise ValueError("dropout probability must be in [0, 1)")
    if not training or p == 0:
        return input
    mask = (random_uniform(input.shape, input.device) >= p).astype(input.dtype) / (1 - p)
    return Tensor._from_op(input._data * mask, (input,), lambda g: (g * mask,), "dropout")


def embedding(indices: Tensor, weight: Tensor) -> Tensor:
    if indices.requires_grad:
        raise ValueError("embedding indices cannot require gradients")
    if indices.device != weight.device:
        raise ValueError("indices and embedding weight must be on the same device")
    raw_indices = indices._data
    data = weight._data[raw_indices]

    def backward(g):
        grad_weight = xp_for(weight.device).zeros_like(weight._data)
        return (add_at(grad_weight, raw_indices, g, weight.device),)

    return Tensor._from_op(data, (weight,), backward, "embedding")


def conv1d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tensor:
    """NCHW-style 1-D cross-correlation lowered to batched contractions."""
    input, weight, bias = _cast_inputs(input, weight, bias)
    if input.ndim != 3 or weight.ndim != 3:
        raise ValueError("conv1d expects input [N,C,L] and weight [O,C/groups,K]")
    if input.device != weight.device or (bias is not None and bias.device != input.device):
        raise ValueError("conv1d tensors must be on the same device")
    n, channels, length = input.shape
    out_channels, channels_per_group, kernel = weight.shape
    if channels % groups or out_channels % groups or channels_per_group != channels // groups:
        raise ValueError("channels must be divisible by groups and agree with weight shape")
    output_length = (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    if output_length <= 0:
        raise ValueError("kernel is larger than the padded input")
    xp = xp_for(input.device)
    padded = xp.pad(input._data, ((0, 0), (0, 0), (padding, padding)))
    columns = xp.stack(
        [
            padded[:, :, k * dilation : k * dilation + output_length * stride : stride]
            for k in range(kernel)
        ],
        axis=2,
    )
    cpg, opg = channels // groups, out_channels // groups
    grouped_columns = columns.reshape(n, groups, cpg, kernel, output_length)
    grouped_weight = weight._data.reshape(groups, opg, cpg, kernel)
    output = xp.einsum("ngckl,gock->ngol", grouped_columns, grouped_weight).reshape(
        n, out_channels, output_length
    )
    if bias is not None:
        output = output + bias._data.reshape(1, -1, 1)

    def backward(g):
        grouped_grad = g.reshape(n, groups, opg, output_length)
        grad_weight = xp.einsum("ngol,ngckl->gock", grouped_grad, grouped_columns).reshape(
            weight.shape
        )
        grad_columns = xp.einsum("ngol,gock->ngckl", grouped_grad, grouped_weight)
        grad_padded = xp.zeros_like(padded)
        for k in range(kernel):
            key = (
                slice(None),
                slice(None),
                slice(k * dilation, k * dilation + output_length * stride, stride),
            )
            source = grad_columns[:, :, :, k, :].reshape(n, channels, output_length)
            grad_padded = assign_add(grad_padded, key, source, input.device)
        grad_input = grad_padded[:, :, padding : padding + length] if padding else grad_padded
        result = [grad_input, grad_weight]
        if bias is not None:
            result.append(xp.sum(g, axis=(0, 2)))
        return tuple(result)

    parents = (input, weight) if bias is None else (input, weight, bias)
    return Tensor._from_op(output, parents, backward, "conv1d")


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    groups: int = 1,
) -> Tensor:
    """2-D cross-correlation using im2col views and one batched contraction."""
    input, weight, bias = _cast_inputs(input, weight, bias)
    if input.ndim != 4 or weight.ndim != 4:
        raise ValueError("conv2d expects input [N,C,H,W] and weight [O,C/groups,KH,KW]")
    if input.device != weight.device or (bias is not None and bias.device != input.device):
        raise ValueError("conv2d tensors must be on the same device")
    sh, sw = _pair(stride)
    ph, pw = _pair(padding)
    dh, dw = _pair(dilation)
    n, channels, height, width = input.shape
    out_channels, channels_per_group, kh, kw = weight.shape
    if channels % groups or out_channels % groups or channels_per_group != channels // groups:
        raise ValueError("channels must be divisible by groups and agree with weight shape")
    oh = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("kernel is larger than the padded input")
    xp = xp_for(input.device)
    padded = xp.pad(input._data, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    windows = []
    for i in range(kh):
        row = []
        for j in range(kw):
            row.append(
                padded[
                    :,
                    :,
                    i * dh : i * dh + oh * sh : sh,
                    j * dw : j * dw + ow * sw : sw,
                ]
            )
        windows.append(xp.stack(row, axis=2))
    columns = xp.stack(windows, axis=2)  # [N,C,KH,KW,OH,OW]
    cpg, opg = channels // groups, out_channels // groups
    grouped_columns = columns.reshape(n, groups, cpg, kh, kw, oh, ow)
    grouped_weight = weight._data.reshape(groups, opg, cpg, kh, kw)
    output = xp.einsum("ngcijhw,gocij->ngohw", grouped_columns, grouped_weight).reshape(
        n, out_channels, oh, ow
    )
    if bias is not None:
        output = output + bias._data.reshape(1, -1, 1, 1)

    def backward(g):
        grouped_grad = g.reshape(n, groups, opg, oh, ow)
        grad_weight = xp.einsum("ngohw,ngcijhw->gocij", grouped_grad, grouped_columns).reshape(
            weight.shape
        )
        grad_columns = xp.einsum("ngohw,gocij->ngcijhw", grouped_grad, grouped_weight)
        grad_padded = xp.zeros_like(padded)
        for i in range(kh):
            for j in range(kw):
                key = (
                    slice(None),
                    slice(None),
                    slice(i * dh, i * dh + oh * sh, sh),
                    slice(j * dw, j * dw + ow * sw, sw),
                )
                source = grad_columns[:, :, :, i, j, :, :].reshape(n, channels, oh, ow)
                grad_padded = assign_add(grad_padded, key, source, input.device)
        grad_input = grad_padded[:, :, ph : ph + height, pw : pw + width]
        result = [grad_input, grad_weight]
        if bias is not None:
            result.append(xp.sum(g, axis=(0, 2, 3)))
        return tuple(result)

    parents = (input, weight) if bias is None else (input, weight, bias)
    return Tensor._from_op(output, parents, backward, "conv2d")


def _pool2d(
    input: Tensor,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] | None,
    padding: int | Sequence[int],
    mode: str,
) -> Tensor:
    if input.ndim != 4:
        raise ValueError("pool2d expects input [N,C,H,W]")
    kh, kw = _pair(kernel_size)
    sh, sw = _pair(kernel_size if stride is None else stride)
    ph, pw = _pair(padding)
    n, channels, height, width = input.shape
    oh = (height + 2 * ph - kh) // sh + 1
    ow = (width + 2 * pw - kw) // sw + 1
    xp = xp_for(input.device)
    fill = -float("inf") if mode == "max" else 0.0
    padded = xp.pad(input._data, ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=fill)
    rows = []
    for i in range(kh):
        rows.append(
            xp.stack(
                [padded[:, :, i : i + oh * sh : sh, j : j + ow * sw : sw] for j in range(kw)],
                axis=2,
            )
        )
    windows = xp.stack(rows, axis=2)  # [N,C,KH,KW,OH,OW]
    if mode == "max":
        data = xp.max(windows, axis=(2, 3))
    else:
        data = xp.mean(windows, axis=(2, 3))

    def backward(g):
        if mode == "max":
            # MaxPool (unlike amax reduction) routes ties to the first
            # row-major winner, matching PyTorch's pooling contract.
            winners = xp.argmax(windows.reshape(n, channels, kh * kw, oh, ow), axis=2)
            locations = xp.arange(kh * kw).reshape(1, 1, kh * kw, 1, 1)
            scale = (locations == winners[:, :, None]).reshape(windows.shape).astype(g.dtype)
        else:
            scale = xp.ones_like(windows) / (kh * kw)
        source_windows = scale * xp.expand_dims(xp.expand_dims(g, 2), 3)
        grad_padded = xp.zeros_like(padded)
        for i in range(kh):
            for j in range(kw):
                key = (
                    slice(None),
                    slice(None),
                    slice(i, i + oh * sh, sh),
                    slice(j, j + ow * sw, sw),
                )
                grad_padded = assign_add(grad_padded, key, source_windows[:, :, i, j], input.device)
        return (grad_padded[:, :, ph : ph + height, pw : pw + width],)

    return Tensor._from_op(data, (input,), backward, f"{mode}_pool2d")


def max_pool2d(input: Tensor, kernel_size, stride=None, padding=0) -> Tensor:
    return _pool2d(input, kernel_size, stride, padding, "max")


def avg_pool2d(input: Tensor, kernel_size, stride=None, padding=0) -> Tensor:
    return _pool2d(input, kernel_size, stride, padding, "avg")


def _pool1d(input: Tensor, kernel_size: int, stride: int | None, padding: int, mode: str) -> Tensor:
    if input.ndim != 3:
        raise ValueError("pool1d expects input [N,C,L]")
    # Reuse the verified 2-D implementation with a singleton spatial dimension.
    result = _pool2d(
        input.unsqueeze(2), (1, kernel_size), (1, stride or kernel_size), (0, padding), mode
    )
    return result.squeeze(2)


def max_pool1d(
    input: Tensor, kernel_size: int, stride: int | None = None, padding: int = 0
) -> Tensor:
    return _pool1d(input, kernel_size, stride, padding, "max")


def avg_pool1d(
    input: Tensor, kernel_size: int, stride: int | None = None, padding: int = 0
) -> Tensor:
    return _pool1d(input, kernel_size, stride, padding, "avg")


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
