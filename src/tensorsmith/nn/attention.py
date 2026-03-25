"""Scaled dot-product attention and rotary positions, with custom autograd.

``streaming`` is blockwise exact attention, not a claim of FlashAttention CUDA
kernel parity. ``native`` uses MLX's optimized Metal attention forward and our
recomputing VJP; CPU/CUDA use native array matmul primitives.
"""

from __future__ import annotations

import math

from ..device import assign_add, random_uniform, xp_for
from ..tensor import Tensor, is_grad_enabled


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    *,
    scale: float | None = None,
    causal_offset: int = 0,
    backend: str = "auto",
    block_size: int = 64,
    enable_gqa: bool = False,
) -> Tensor:
    """Attention for [batch, heads, sequence, channels] tensors.

    Boolean masks use True=allowed; floating masks are added to scores. Fully
    masked rows return zero with zero gradients. ``causal_offset`` is the
    absolute position of the first query in a cached key sequence.
    """
    if any(t.ndim != 4 for t in (query, key, value)):
        raise ValueError("attention expects [batch, heads, sequence, channels]")
    if query.device != key.device or query.device != value.device:
        raise ValueError("attention tensors must be on the same device")
    if query.dtype != key.dtype or query.dtype != value.dtype or "float" not in str(query.dtype):
        raise TypeError("attention inputs must have the same floating-point dtype")
    if query.shape[0] != key.shape[0] or key.shape[:3] != value.shape[:3]:
        raise ValueError("attention batch/head dimensions and key/value lengths must agree")
    repetitions = query.shape[1] // key.shape[1] if key.shape[1] else 0
    if query.shape[1] != key.shape[1] and (
        not enable_gqa or not repetitions or query.shape[1] % key.shape[1]
    ):
        raise ValueError(
            "grouped-query attention requires enable_gqa and query heads divisible by KV heads"
        )
    if query.shape[-1] != key.shape[-1] or min(*query.shape, *key.shape, *value.shape) <= 0:
        raise ValueError("query/key head dimensions must match and sequences must be nonempty")
    if not 0 <= dropout_p < 1 or block_size <= 0 or causal_offset < 0:
        raise ValueError("invalid dropout probability, block size, or causal offset")
    if backend not in {"auto", "dense", "streaming", "native"}:
        raise ValueError("attention backend must be auto, dense, streaming, or native")
    xp = xp_for(query.device)
    b, h, qlen, dim = query.shape
    klen = key.shape[2]
    multiplier = 1 / math.sqrt(dim) if scale is None else scale
    if not math.isfinite(multiplier):
        raise ValueError("attention scale must be finite")
    compute_dtype = xp.float32 if "float16" in str(query.dtype) else query.dtype
    mask = None
    boolean_mask = False
    if attn_mask is not None:
        if not any(name in str(attn_mask.dtype) for name in ("bool", "float")):
            raise TypeError("attention masks must be boolean or floating point")
        if attn_mask.requires_grad or attn_mask.device != query.device:
            raise ValueError("attention masks must be non-differentiable and on the input device")
        try:
            mask = xp.broadcast_to(attn_mask._data, (b, h, qlen, klen))
        except (ValueError, RuntimeError) as error:
            raise ValueError("attention mask is not broadcastable to [B,H,Q,K]") from error
        boolean_mask = "bool" in str(attn_mask.dtype)

    selected = backend
    if selected == "auto":
        if query.device.type == "metal" and not dropout_p and value.shape[-1] == dim:
            selected = "native"
        else:
            selected = "streaming" if qlen * klen > 1024 * 1024 and not dropout_p else "dense"
    if selected in {"native", "streaming"} and dropout_p:
        raise ValueError("attention dropout currently requires the dense backend")
    if selected == "native" and (query.device.type != "metal" or value.shape[-1] != dim):
        raise ValueError("native attention requires Metal and equal query/value head dimensions")
    recording = is_grad_enabled() and any(t.requires_grad for t in (query, key, value))
    q = k = v = None
    if selected != "native" or recording:
        q = query._data.astype(compute_dtype)
        k = key._data.astype(compute_dtype)
        v = value._data.astype(compute_dtype)
        if repetitions > 1:
            # Native forward consumes compact KV groups, no inference expansion.
            def repeat(raw):
                return xp.broadcast_to(
                    raw[:, :, None], (b, key.shape[1], repetitions, raw.shape[2], raw.shape[3])
                ).reshape(b, h, raw.shape[2], raw.shape[3])

            k, v = repeat(k), repeat(v)

    def scores(qblock, kblock, qi, ki):
        result = xp.matmul(qblock, xp.swapaxes(kblock, -1, -2)) * multiplier
        nq, nk = qblock.shape[2], kblock.shape[2]
        if mask is not None:
            part = mask[:, :, qi : qi + nq, ki : ki + nk]
            result = xp.where(part, result, -float("inf")) if boolean_mask else result + part
        if is_causal:
            allowed = (
                xp.arange(qi, qi + nq)[:, None] + causal_offset >= xp.arange(ki, ki + nk)[None, :]
            )
            result = xp.where(allowed, result, -float("inf"))
        return result

    def normalization(qblock, qi):
        shape = qblock.shape[:3] + (1,)
        maximum = xp.full(shape, -float("inf"), dtype=compute_dtype)
        total = xp.zeros(shape, dtype=compute_dtype)
        for ki in range(0, klen, block_size):
            logits = scores(qblock, k[:, :, ki : ki + block_size], qi, ki)
            next_max = xp.maximum(maximum, xp.max(logits, axis=-1, keepdims=True))
            safe_max = xp.where(xp.isfinite(next_max), next_max, 0)
            correction = xp.exp(maximum - safe_max)
            total = total * correction + xp.sum(xp.exp(logits - safe_max), axis=-1, keepdims=True)
            maximum = next_max
        safe_total = xp.where(total > 0, total, 1)
        return xp.where(total > 0, maximum + xp.log(safe_total), -float("inf"))

    saved_probabilities = None
    dropout_mask = None
    saved_lse = None
    if selected == "dense":
        logits = scores(q, k, 0, 0)
        maximum = xp.max(logits, axis=-1, keepdims=True)
        maximum = xp.where(xp.isfinite(maximum), maximum, 0)
        exponentials = xp.exp(logits - maximum)
        total = xp.sum(exponentials, axis=-1, keepdims=True)
        probabilities = exponentials / xp.where(total > 0, total, 1)
        saved_probabilities = probabilities
        if dropout_p:
            dropout_mask = (random_uniform(probabilities.shape, query.device) >= dropout_p).astype(
                compute_dtype
            ) / (1 - dropout_p)
            probabilities = probabilities * dropout_mask
        output = xp.matmul(probabilities, v)
    elif selected == "native":
        native_mask = None
        if mask is None and is_causal and causal_offset == 0 and qlen == klen:
            native_mask = "causal"
        elif mask is not None or (is_causal and not (qlen == 1 and causal_offset >= klen - 1)):
            # Mask construction is O(Q*K), but has no batch/head duplication for
            # the common causal-only case; decoding uses no mask allocation.
            allowed = xp.arange(qlen)[:, None] + causal_offset >= xp.arange(klen)[None, :]
            if mask is None:
                native_mask = allowed
            elif boolean_mask:
                native_mask = mask & allowed if is_causal else mask
            else:
                native_mask = xp.where(allowed, mask, -float("inf")) if is_causal else mask
        output = xp.fast.scaled_dot_product_attention(
            query._data, key._data, value._data, scale=multiplier, mask=native_mask
        )
        if native_mask is not None and not isinstance(native_mask, str):
            # Vendor kernels need not define fully masked rows as zero (some
            # return a mean or NaN). Enforce our contract before the VJP.
            allowed_rows = xp.any(
                native_mask if "bool" in str(native_mask.dtype) else xp.isfinite(native_mask),
                axis=-1,
                keepdims=True,
            )
            output = xp.where(allowed_rows, output, 0)
        if not recording:
            return Tensor(output, device=query.device)
        output = output.astype(compute_dtype)
    else:
        outputs, normalizers = [], []
        for qi in range(0, qlen, block_size):
            qb = q[:, :, qi : qi + block_size]
            shape = qb.shape[:3] + (1,)
            maximum = xp.full(shape, -float("inf"), dtype=compute_dtype)
            total = xp.zeros(shape, dtype=compute_dtype)
            accumulator = xp.zeros(qb.shape[:3] + (v.shape[-1],), dtype=compute_dtype)
            for ki in range(0, klen, block_size):
                logits = scores(qb, k[:, :, ki : ki + block_size], qi, ki)
                next_max = xp.maximum(maximum, xp.max(logits, axis=-1, keepdims=True))
                safe_max = xp.where(xp.isfinite(next_max), next_max, 0)
                correction = xp.exp(maximum - safe_max)
                probabilities = xp.exp(logits - safe_max)
                accumulator = accumulator * correction + xp.matmul(
                    probabilities, v[:, :, ki : ki + block_size]
                )
                total = total * correction + xp.sum(probabilities, axis=-1, keepdims=True)
                maximum = next_max
            safe_total = xp.where(total > 0, total, 1)
            outputs.append(accumulator / safe_total)
            normalizers.append(xp.where(total > 0, maximum + xp.log(safe_total), -float("inf")))
        output = xp.concatenate(outputs, axis=2)
        saved_lse = xp.concatenate(normalizers, axis=2)

    def backward(upstream):
        g = upstream.astype(compute_dtype)
        p = saved_probabilities
        if selected == "native" and qlen * klen <= 1024 * 1024:
            # For small workloads, dense recomputation is faster than Python
            # block dispatch and still avoids saving probabilities in forward.
            logits = scores(q, k, 0, 0)
            maximum = xp.max(logits, axis=-1, keepdims=True)
            exponentials = xp.exp(logits - xp.where(xp.isfinite(maximum), maximum, 0))
            total = xp.sum(exponentials, axis=-1, keepdims=True)
            p = exponentials / xp.where(total > 0, total, 1)
        if p is not None:
            effective_p = p if dropout_mask is None else p * dropout_mask
            dv = xp.matmul(xp.swapaxes(effective_p, -1, -2), g)
            dp = xp.matmul(g, xp.swapaxes(v, -1, -2))
            if dropout_mask is not None:
                dp = dp * dropout_mask
            ds = p * (dp - xp.sum(dp * p, axis=-1, keepdims=True)) * multiplier
            dq = xp.matmul(ds, k)
            dk = xp.matmul(xp.swapaxes(ds, -1, -2), q)
        else:
            dq_blocks = []
            dk, dv = xp.zeros_like(k), xp.zeros_like(v)
            for qi in range(0, qlen, block_size):
                qb = q[:, :, qi : qi + block_size]
                gb = g[:, :, qi : qi + block_size]
                lse = (
                    normalization(qb, qi)
                    if saved_lse is None
                    else saved_lse[:, :, qi : qi + block_size]
                )
                safe_lse = xp.where(xp.isfinite(lse), lse, 0)
                delta = xp.sum(gb * output[:, :, qi : qi + block_size], axis=-1, keepdims=True)
                dq_block = xp.zeros_like(qb)
                for ki in range(0, klen, block_size):
                    kb, vb = k[:, :, ki : ki + block_size], v[:, :, ki : ki + block_size]
                    logits = scores(qb, kb, qi, ki)
                    p = xp.exp(logits - safe_lse)
                    ds = p * (xp.matmul(gb, xp.swapaxes(vb, -1, -2)) - delta) * multiplier
                    dq_block = dq_block + xp.matmul(ds, kb)
                    index = (slice(None), slice(None), slice(ki, ki + kb.shape[2]), slice(None))
                    dk = assign_add(dk, index, xp.matmul(xp.swapaxes(ds, -1, -2), qb), query.device)
                    dv = assign_add(dv, index, xp.matmul(xp.swapaxes(p, -1, -2), gb), query.device)
                dq_blocks.append(dq_block)
            dq = xp.concatenate(dq_blocks, axis=2)
        if repetitions > 1:
            dk = dk.reshape(b, key.shape[1], repetitions, klen, dim).sum(axis=2)
            dv = dv.reshape(b, key.shape[1], repetitions, klen, value.shape[-1]).sum(axis=2)
        return dq.astype(query.dtype), dk.astype(key.dtype), dv.astype(value.dtype)

    return Tensor._from_op(
        output.astype(query.dtype), (query, key, value), backward, "sdpa_" + selected
    )


def rotary_embedding(
    input: Tensor, offset: int = 0, base: float = 10000.0, rotary_dim: int | None = None
) -> Tensor:
    """Interleaved-pair RoPE for [..., sequence, channels], with inverse-rotation VJP."""
    if input.ndim < 2:
        raise ValueError("RoPE requires at least sequence and channel dimensions")
    dim = input.shape[-1] if rotary_dim is None else rotary_dim
    if (
        dim <= 0
        or dim % 2
        or dim > input.shape[-1]
        or offset < 0
        or base <= 0
        or not math.isfinite(base)
    ):
        raise ValueError("RoPE requires a positive even rotary dimension and valid positions/base")
    xp = xp_for(input.device)
    frequencies = base ** (-xp.arange(0, dim, 2).astype(xp.float32) / dim)
    angles = (
        xp.arange(offset, offset + input.shape[-2]).astype(xp.float32)[:, None]
        * frequencies[None, :]
    )
    cosine = xp.stack((xp.cos(angles), xp.cos(angles)), axis=-1).reshape(input.shape[-2], dim)
    sine = xp.stack((xp.sin(angles), xp.sin(angles)), axis=-1).reshape(input.shape[-2], dim)
    cosine, sine = cosine.astype(input.dtype), sine.astype(input.dtype)

    def rotate(raw):
        return xp.stack((-raw[..., 1::2], raw[..., 0::2]), axis=-1).reshape(raw.shape)

    part = input._data[..., :dim]
    result = part * cosine + rotate(part) * sine
    if dim < input.shape[-1]:
        result = xp.concatenate((result, input._data[..., dim:]), axis=-1)

    def backward(g):
        front = g[..., :dim]
        dx = front * cosine - rotate(front) * sine
        if dim < input.shape[-1]:
            dx = xp.concatenate((dx, g[..., dim:]), axis=-1)
        return (dx,)

    return Tensor._from_op(result, (input,), backward, "rope")
