"""Single-tape-node activations and normalization with explicit VJPs.

These avoid Python autograd subgraphs. They use backend array kernels, not a
claim of universally fused GPU execution. MLX fast normalization is used when
its semantics match; every derivative still belongs to TensorSmith.
"""

from __future__ import annotations

import math

from ..device import xp_for
from ..tensor import Tensor, _sum_to_shape, is_grad_enabled


def _compute_array(tensor: Tensor):
    raw = tensor._data
    return raw.astype(xp_for(tensor.device).float32) if "float16" in str(raw.dtype) else raw


def gelu(input: Tensor) -> Tensor:
    return bias_gelu(input)


def bias_gelu(input: Tensor, bias: Tensor | None = None) -> Tensor:
    """Tanh-approximate GELU, optionally with a broadcast bias, in one tape node."""
    xp = xp_for(input.device)
    if bias is not None and bias.device != input.device:
        raise ValueError("input and bias must be on the same device")
    x = _compute_array(input)
    if bias is not None:
        x = x + bias._data
    factor = math.sqrt(2 / math.pi)
    tangent = xp.tanh(factor * (x + 0.044715 * x**3))
    output = (0.5 * x * (1 + tangent)).astype(input.dtype)

    def backward(g):
        derivative = 0.5 * (1 + tangent) + 0.5 * x * (1 - tangent**2) * factor * (
            1 + 3 * 0.044715 * x**2
        )
        dx = g * derivative
        gradients = [_sum_to_shape(dx, input.shape, input.device).astype(input.dtype)]
        if bias is not None:
            gradients.append(_sum_to_shape(dx, bias.shape, bias.device).astype(bias.dtype))
        return tuple(gradients)

    return Tensor._from_op(
        output, (input,) if bias is None else (input, bias), backward, "bias_gelu"
    )


def silu(input: Tensor) -> Tensor:
    xp = xp_for(input.device)
    x = _compute_array(input)
    # Stable sigmoid; avoid overflow for extreme negative inputs.
    exponent = xp.exp(-xp.abs(x))
    sigmoid = xp.where(x >= 0, 1 / (1 + exponent), exponent / (1 + exponent))
    output = (x * sigmoid).astype(input.dtype)
    return Tensor._from_op(
        output,
        (input,),
        lambda g: ((g * (sigmoid + x * sigmoid * (1 - sigmoid))).astype(input.dtype),),
        "silu",
    )


def _norm(input, weight, bias, eps, rms, residual=None):
    if eps <= 0 or not math.isfinite(eps):
        raise ValueError("eps must be positive")
    if input.shape[-weight.ndim :] != weight.shape:
        raise ValueError("normalization weight must match the trailing input dimensions")
    parents = [input]
    if residual is not None:
        if residual.shape != input.shape:
            raise ValueError("residual and input must have equal shapes")
        parents.append(residual)
    parents.append(weight)
    if bias is not None:
        if bias.shape != weight.shape:
            raise ValueError("normalization bias and weight must have equal shapes")
        parents.append(bias)
    if any(p.device != input.device for p in parents):
        raise ValueError("normalization tensors must be on the same device")
    xp = xp_for(input.device)
    x = _compute_array(input)
    if residual is not None:
        x = x + _compute_array(residual)
    native = input.device.type == "metal" and weight.ndim == 1
    if native:
        if rms:
            output = xp.fast.rms_norm(x, weight._data.astype(x.dtype), eps).astype(input.dtype)
        else:
            output = xp.fast.layer_norm(
                x,
                weight._data.astype(x.dtype),
                None if bias is None else bias._data.astype(x.dtype),
                eps,
            ).astype(input.dtype)
        if not is_grad_enabled() or not any(p.requires_grad for p in parents):
            # Inference needs no duplicate saved-statistics expression graph.
            return Tensor(output, device=input.device)
    axes = tuple(range(input.ndim - weight.ndim, input.ndim))
    centered = x if rms else x - xp.mean(x, axis=axes, keepdims=True)
    reciprocal_std = 1 / xp.sqrt(xp.mean(centered**2, axis=axes, keepdims=True) + eps)
    normalized = centered * reciprocal_std
    if not native:
        output = normalized * weight._data
        if bias is not None:
            output = output + bias._data
        output = output.astype(input.dtype)

    def backward(g):
        weighted = g * weight._data
        projected = normalized * xp.mean(weighted * normalized, axis=axes, keepdims=True)
        dx = reciprocal_std * (weighted - projected)
        if not rms:
            dx = dx - reciprocal_std * xp.mean(weighted, axis=axes, keepdims=True)
        result = [dx.astype(input.dtype)]
        if residual is not None:
            result.append(dx.astype(residual.dtype))
        result.append(
            _sum_to_shape(g * normalized, weight.shape, input.device).astype(weight.dtype)
        )
        if bias is not None:
            result.append(_sum_to_shape(g, bias.shape, input.device).astype(bias.dtype))
        return tuple(result)

    name = "rms_norm" if rms else "layer_norm"
    return Tensor._from_op(
        output, tuple(parents), backward, "residual_" + name if residual is not None else name
    )


def layer_norm(input: Tensor, weight: Tensor, bias: Tensor | None = None, eps: float = 1e-5):
    return _norm(input, weight, bias, eps, False)


def rms_norm(input: Tensor, weight: Tensor, eps: float = 1e-6):
    return _norm(input, weight, None, eps, True)


def residual_layer_norm(
    input: Tensor, residual: Tensor, weight: Tensor, bias: Tensor | None = None, eps: float = 1e-5
):
    return _norm(input, weight, bias, eps, False, residual)


def residual_rms_norm(input: Tensor, residual: Tensor, weight: Tensor, eps: float = 1e-6):
    return _norm(input, weight, None, eps, True, residual)
