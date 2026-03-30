"""Bounded, preallocated inference caches, including per-token int8 KV storage."""

from __future__ import annotations

import math

from ..device import array, assign, backend_dtype, xp_for
from ..device import device as parse_device
from ..tensor import Tensor, is_grad_enabled


class KVCache:
    """Preallocated [B,H,capacity,D] keys/values for autoregressive inference.

    Quantized storage uses symmetric int8 with one FP32 scale per token/head.
    Attention currently dequantizes the active prefix; this saves persistent
    storage, not a promise of faster decode or fused int8 attention.
    Cache mutation is rejected when it would silently sever an autograd graph.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        *,
        device=None,
        dtype="float32",
        quantized: bool = False,
        value_dim=None,
    ):
        if min(batch_size, num_heads, head_dim, max_seq_len) <= 0:
            raise ValueError("cache dimensions and capacity must be positive")
        self.batch_size, self.num_heads, self.head_dim = batch_size, num_heads, head_dim
        self.value_dim = head_dim if value_dim is None else value_dim
        if self.value_dim <= 0:
            raise ValueError("value_dim must be positive")
        self.max_seq_len = max_seq_len
        self.device = parse_device(device)
        self.dtype = backend_dtype(dtype, self.device)
        if "float" not in str(self.dtype):
            raise TypeError("cache compute dtype must be floating point")
        self.quantized = quantized
        self.length = 0
        xp = xp_for(self.device)
        storage_dtype = xp.int8 if quantized else self.dtype
        shape = (batch_size, num_heads, max_seq_len)
        self._keys = xp.zeros(shape + (head_dim,), dtype=storage_dtype)
        self._values = xp.zeros(shape + (self.value_dim,), dtype=storage_dtype)
        self._key_scales = xp.ones(shape + (1,), dtype=xp.float32) if quantized else None
        self._value_scales = xp.ones(shape + (1,), dtype=xp.float32) if quantized else None

    def append(self, key: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
        """Append a chunk after validating all shapes and capacity."""
        if key.device != self.device or value.device != self.device:
            raise ValueError("cache and key/value tensors must be on the same device")
        if is_grad_enabled() and (key.requires_grad or value.requires_grad):
            raise RuntimeError("KVCache is inference-only; use no_grad() for differentiable inputs")
        prefix = (self.batch_size, self.num_heads)
        if key.ndim != 4 or key.shape[:2] != prefix or key.shape[-1] != self.head_dim:
            raise ValueError("key shape does not match cache configuration")
        if value.shape != key.shape[:3] + (self.value_dim,):
            raise ValueError("value shape does not match cache configuration")
        stop = self.length + key.shape[2]
        if key.shape[2] == 0 or stop > self.max_seq_len:
            raise ValueError(f"cache capacity {self.max_seq_len} exceeded or empty append")
        index = (slice(None), slice(None), slice(self.length, stop), slice(None))
        xp = xp_for(self.device)
        keys, values = key._data.astype(self.dtype), value._data.astype(self.dtype)
        if self.quantized:

            def quantize(raw):
                raw = raw.astype(xp.float32)
                scale = xp.maximum(xp.max(xp.abs(raw), axis=-1, keepdims=True) / 127, 1e-8)
                compressed = xp.round(raw / scale).astype(xp.int8)
                return compressed, scale

            keys, key_scales = quantize(keys)
            values, value_scales = quantize(values)
            self._key_scales = assign(self._key_scales, index, key_scales)
            self._value_scales = assign(self._value_scales, index, value_scales)
        self._keys = assign(self._keys, index, keys)
        self._values = assign(self._values, index, values)
        self.length = stop
        return self.get()

    def get(self) -> tuple[Tensor, Tensor]:
        keys = self._keys[:, :, : self.length]
        values = self._values[:, :, : self.length]
        if self.quantized:
            xp = xp_for(self.device)
            keys = (keys.astype(xp.float32) * self._key_scales[:, :, : self.length]).astype(
                self.dtype
            )
            values = (values.astype(xp.float32) * self._value_scales[:, :, : self.length]).astype(
                self.dtype
            )
        return Tensor(keys, device=self.device), Tensor(values, device=self.device)

    def reset(self) -> None:
        """Reuse the allocation; future appends overwrite old entries."""
        self.length = 0

    def truncate(self, length: int) -> None:
        if not 0 <= length <= self.length:
            raise ValueError("truncate length must be between zero and the current length")
        self.length = length

    def reorder(self, indices) -> None:
        """Reorder or duplicate batch rows for fixed-width beam search."""
        from ..device import asnumpy

        host = asnumpy(indices._data if isinstance(indices, Tensor) else indices)
        if host.shape != (self.batch_size,) or host.dtype.kind not in "iu":
            raise ValueError("beam indices must be one integer per cache batch row")
        if (host < 0).any() or (host >= self.batch_size).any():
            raise ValueError("beam index out of range")
        order = array(host.astype("int32"), self.device)
        self._keys, self._values = self._keys[order], self._values[order]
        if self.quantized:
            self._key_scales = self._key_scales[order]
            self._value_scales = self._value_scales[order]

    @property
    def storage(self) -> tuple:
        """Raw persistent arrays, for explicit lazy-device evaluation/checkpointing."""
        return self._keys, self._values, self._key_scales, self._value_scales

    @property
    def memory_bytes(self) -> int:
        """Allocated persistent payload, excluding temporary dequantized arrays."""
        arrays = [self._keys, self._values]
        if self.quantized:
            arrays += [self._key_scales, self._value_scales]
        return sum(math.prod(a.shape) * a.itemsize for a in arrays)

    def __len__(self):
        return self.length

    def __repr__(self):
        return f"KVCache(length={self.length}, capacity={self.max_seq_len}, quantized={self.quantized}, device='{self.device}')"
