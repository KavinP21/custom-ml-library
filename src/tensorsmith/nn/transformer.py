"""Usable decoder-only transformers, GQA, RoPE, and cached token generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..device import asnumpy, evaluate, random_uniform, xp_for
from ..tensor import Tensor, cat, is_grad_enabled, no_grad
from . import functional as F
from .attention import rotary_embedding, scaled_dot_product_attention
from .cache import KVCache
from .checkpoint import checkpoint
from .layers import Dropout, Embedding, Linear
from .module import Module, Parameter


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6, device=None):
        super().__init__()
        if dim <= 0:
            raise ValueError("normalization dimension must be positive")
        self.eps = eps
        self.weight = Parameter(Tensor(np.ones(dim, np.float32), device=device))

    def forward(self, input: Tensor) -> Tensor:
        return F.rms_norm(input, self.weight, self.eps)


class SiLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.silu(input)


class SwiGLU(Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0, device=None):
        super().__init__()
        self.gate = Linear(dim, hidden_dim, bias=False, device=device)
        self.up = Linear(dim, hidden_dim, bias=False, device=device)
        self.down = Linear(hidden_dim, dim, bias=False, device=device)
        self.dropout = Dropout(dropout)

    def forward(self, input: Tensor) -> Tensor:
        return self.dropout(self.down(F.silu(self.gate(input)) * self.up(input)))


class MultiHeadAttention(Module):
    """Self/cross attention with optional grouped-query heads and rotary positions."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
        rotary: bool = True,
        backend: str = "auto",
        device=None,
    ):
        super().__init__()
        kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if (
            min(embed_dim, num_heads, kv_heads) <= 0
            or embed_dim % num_heads
            or num_heads % kv_heads
        ):
            raise ValueError(
                "embed_dim must divide into heads and query heads must divide into KV groups"
            )
        self.embed_dim, self.num_heads, self.num_kv_heads = embed_dim, num_heads, kv_heads
        self.head_dim = embed_dim // num_heads
        if rotary and self.head_dim % 2:
            raise ValueError("rotary attention requires an even head dimension")
        if not 0 <= dropout < 1 or backend not in {"auto", "dense", "streaming", "native"}:
            raise ValueError("invalid attention dropout/backend")
        self.rotary, self.backend, self.dropout_p = rotary, backend, dropout
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias, device=device)
        self.k_proj = Linear(embed_dim, kv_heads * self.head_dim, bias=bias, device=device)
        self.v_proj = Linear(embed_dim, kv_heads * self.head_dim, bias=bias, device=device)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, device=device)

    def forward(
        self,
        input: Tensor,
        context: Tensor | None = None,
        *,
        attn_mask=None,
        is_causal: bool = True,
        cache: KVCache | None = None,
    ) -> Tensor:
        if input.ndim != 3 or input.shape[-1] != self.embed_dim:
            raise ValueError("attention input must be [batch, sequence, embed_dim]")
        source = input if context is None else context
        if (
            source.ndim != 3
            or source.shape[0] != input.shape[0]
            or source.shape[-1] != self.embed_dim
        ):
            raise ValueError("attention context must be [batch, sequence, embed_dim]")
        if cache is not None and (self.training or is_grad_enabled() or context is not None):
            raise RuntimeError(
                "cached self-attention requires eval() and no_grad(); cross-attention caching is not supported"
            )
        batch, steps, _ = input.shape
        offset = 0 if cache is None else cache.length
        if self.backend == "native" and input.device.type != "metal":
            raise ValueError("native attention requires Metal")
        if attn_mask is not None:
            if attn_mask.requires_grad or attn_mask.device != input.device:
                raise ValueError(
                    "attention mask must be non-differentiable and on the input device"
                )
            try:
                expected_mask_shape = (batch, self.num_heads, steps, offset + source.shape[1])
                if np.broadcast_shapes(attn_mask.shape, expected_mask_shape) != expected_mask_shape:
                    raise ValueError("mask expands attention dimensions")
                if not any(name in str(attn_mask.dtype) for name in ("bool", "float")):
                    raise TypeError("attention masks must be boolean or floating point")
            except ValueError as error:
                raise ValueError("attention mask is not broadcastable to [B,H,Q,K]") from error
        q = (
            self.q_proj(input)
            .reshape(batch, steps, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_proj(source)
            .reshape(batch, source.shape[1], self.num_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(source)
            .reshape(batch, source.shape[1], self.num_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        if self.rotary:
            q = rotary_embedding(q, offset)
            k = rotary_embedding(k, offset if context is None else 0)
        if cache is not None:
            k, v = cache.append(k, v)
        attended = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask,
            self.dropout_p if self.training else 0.0,
            is_causal,
            causal_offset=offset,
            backend=self.backend,
            enable_gqa=self.num_kv_heads != self.num_heads,
        )
        return self.out_proj(attended.permute(0, 2, 1, 3).reshape(batch, steps, self.embed_dim))


class TransformerDecoderBlock(Module):
    """Pre-RMSNorm decoder block with GQA/RoPE attention and a SwiGLU MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: int | None = None,
        *,
        num_kv_heads=None,
        dropout=0.0,
        attention_backend="auto",
        device=None,
    ):
        super().__init__()
        self.attention_norm = RMSNorm(dim, device=device)
        self.attention = MultiHeadAttention(
            dim,
            num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            backend=attention_backend,
            device=device,
        )
        self.residual_dropout = Dropout(dropout)
        self.ffn_norm = RMSNorm(dim, device=device)
        self.ffn = SwiGLU(dim, hidden_dim or 4 * dim, dropout, device)

    def forward(self, input: Tensor, *, cache=None, attn_mask=None) -> Tensor:
        hidden = input + self.residual_dropout(
            self.attention(self.attention_norm(input), cache=cache, attn_mask=attn_mask)
        )
        return hidden + self.ffn(self.ffn_norm(hidden))


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    hidden_dim: int | None = None
    num_kv_heads: int | None = None
    max_seq_len: int = 1024
    dropout: float = 0.0
    tie_weights: bool = True
    attention_backend: str = "auto"
    activation_checkpointing: bool = False

    def __post_init__(self):
        if min(self.vocab_size, self.dim, self.num_heads, self.num_layers, self.max_seq_len) <= 0:
            raise ValueError("transformer dimensions must be positive")
        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        kv = self.num_heads if self.num_kv_heads is None else self.num_kv_heads
        if (
            kv <= 0
            or self.dim % self.num_heads
            or self.num_heads % kv
            or (self.dim // self.num_heads) % 2
        ):
            raise ValueError("invalid attention head configuration")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.attention_backend not in {"auto", "dense", "streaming", "native"}:
            raise ValueError("invalid attention backend")


class TransformerLM(Module):
    """A decoder-only language model with [B,T,V] logits and cached generation."""

    def __init__(self, config: TransformerConfig, device=None):
        super().__init__()
        self.config = config
        self.token_embedding = Embedding(config.vocab_size, config.dim, device=device)
        self.embedding_dropout = Dropout(config.dropout)
        self.blocks = [
            TransformerDecoderBlock(
                config.dim,
                config.num_heads,
                config.hidden_dim,
                num_kv_heads=config.num_kv_heads,
                dropout=config.dropout,
                attention_backend=config.attention_backend,
                device=device,
            )
            for _ in range(config.num_layers)
        ]
        self.final_norm = RMSNorm(config.dim, device=device)
        self.output_weight = (
            self.token_embedding.weight
            if config.tie_weights
            else Parameter(
                Tensor(
                    np.random.normal(0, 0.02, (config.vocab_size, config.dim)).astype(np.float32),
                    device=device,
                )
            )
        )
        # Embedding's generic unit variance is inappropriate for residual stacks.
        from .init import normal_

        normal_(self.token_embedding.weight, std=0.02)

    def forward(
        self,
        tokens: Tensor,
        *,
        caches: list[KVCache] | None = None,
        attn_mask=None,
        logits_to_keep: int = 0,
    ) -> Tensor:
        if tokens.ndim != 2 or "int" not in str(tokens.dtype) or tokens.shape[1] == 0:
            raise ValueError("tokens must be a nonempty integer [batch, time] tensor")
        if not isinstance(logits_to_keep, int) or not 0 <= logits_to_keep <= tokens.shape[1]:
            raise ValueError("logits_to_keep must be between zero (all) and the input length")
        offset = 0
        if caches is not None:
            if len(caches) != len(self.blocks) or len({cache.length for cache in caches}) != 1:
                raise ValueError("provide one cache per block, all at the same sequence position")
            if self.training or is_grad_enabled():
                raise RuntimeError("cached decoding requires eval() and no_grad()")
            offset = caches[0].length
            for block, cache in zip(self.blocks, caches):
                if cache.batch_size != tokens.shape[0] or cache.device != tokens.device:
                    raise ValueError("cache batch/device does not match tokens")
                if offset + tokens.shape[1] > cache.max_seq_len:
                    raise ValueError("cache capacity exceeded")
                attention = block.attention
                if (
                    cache.num_heads != attention.num_kv_heads
                    or cache.head_dim != attention.head_dim
                    or cache.value_dim != attention.head_dim
                    or cache.dtype != self.token_embedding.weight.dtype
                ):
                    raise ValueError("cache head configuration/dtype does not match model")
        if offset + tokens.shape[1] > self.config.max_seq_len:
            raise ValueError("model context length exceeded")
        hidden = self.embedding_dropout(self.token_embedding(tokens))
        try:
            for i, block in enumerate(self.blocks):
                if self.config.activation_checkpointing and self.training and is_grad_enabled():
                    hidden = checkpoint(block, hidden, attn_mask=attn_mask)
                else:
                    hidden = block(
                        hidden, cache=None if caches is None else caches[i], attn_mask=attn_mask
                    )
            if logits_to_keep:
                hidden = hidden[:, -logits_to_keep:, :]
            return F.linear(self.final_norm(hidden), self.output_weight)
        except Exception:
            if caches is not None:
                for cache in caches:
                    cache.truncate(offset)
            raise

    def loss(
        self, tokens: Tensor, targets: Tensor, *, ignore_index=-100, label_smoothing=0.0
    ) -> Tensor:
        return F.cross_entropy(
            self(tokens),
            targets,
            axis=-1,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )

    def new_cache(
        self, batch_size: int, max_seq_len: int | None = None, *, quantized=False
    ) -> list[KVCache]:
        capacity = self.config.max_seq_len if max_seq_len is None else max_seq_len
        if capacity <= 0 or capacity > self.config.max_seq_len:
            raise ValueError("cache capacity must fit the model context")
        return [
            KVCache(
                batch_size,
                block.attention.num_kv_heads,
                block.attention.head_dim,
                capacity,
                device=self.token_embedding.weight.device,
                dtype=self.token_embedding.weight.dtype,
                quantized=quantized,
            )
            for block in self.blocks
        ]

    @no_grad()
    def generate(
        self,
        prompt: Tensor,
        max_new_tokens: int,
        *,
        temperature=1.0,
        top_k: int | None = None,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
        use_cache: bool = True,
        quantized_cache: bool = False,
    ) -> Tensor:
        """Batched greedy/top-k/nucleus sampling; returns prompt plus generated tokens."""
        if (
            prompt.ndim != 2
            or prompt.shape[1] == 0
            or max_new_tokens < 0
            or "int" not in str(prompt.dtype)
        ):
            raise ValueError("invalid prompt or generation length")
        if prompt.shape[1] + max_new_tokens > self.config.max_seq_len:
            raise ValueError("generation exceeds the configured context length")
        if temperature < 0 or not np.isfinite(temperature) or not 0 < top_p <= 1:
            raise ValueError("temperature must be finite/nonnegative and top_p in (0,1]")
        if top_k is not None and not 1 <= top_k <= self.config.vocab_size:
            raise ValueError("top_k must be within the vocabulary")
        if eos_token_id is not None and not 0 <= eos_token_id < self.config.vocab_size:
            raise ValueError("eos_token_id out of range")
        modes = [(module, module.training) for module in self.modules()]
        try:
            self.eval()
            caches = (
                self.new_cache(
                    prompt.shape[0], prompt.shape[1] + max_new_tokens, quantized=quantized_cache
                )
                if use_cache and max_new_tokens
                else None
            )
            result, current = prompt, prompt
            xp = xp_for(prompt.device)
            finished = xp.zeros((prompt.shape[0],), dtype=xp.bool_)
            for _ in range(max_new_tokens):
                logits = self(current, caches=caches, logits_to_keep=1)[:, -1, :]
                next_token = sample_logits(logits, temperature, top_k, top_p)
                if eos_token_id is not None:
                    next_token = Tensor(
                        xp.where(finished, eos_token_id, next_token._data), device=prompt.device
                    )
                    finished = finished | (next_token._data == eos_token_id)
                result = cat((result, next_token.unsqueeze(1)), dim=1)
                evaluate(result, [cache.storage for cache in caches] if caches else [])
                if eos_token_id is not None and bool(asnumpy(xp.all(finished))):
                    break
                current = next_token.unsqueeze(1) if caches is not None else result
        finally:
            for module, mode in modes:
                module.training = mode
        return result


def sample_logits(
    logits: Tensor, temperature: float = 1.0, top_k: int | None = None, top_p: float = 1.0
) -> Tensor:
    """Device-resident sampling from [batch, vocabulary] logits."""
    if logits.ndim != 2 or temperature < 0 or not np.isfinite(temperature) or not 0 < top_p <= 1:
        raise ValueError("invalid sampling configuration")
    vocab = logits.shape[-1]
    if top_k is not None and not 1 <= top_k <= vocab:
        raise ValueError("top_k must be within the vocabulary")
    xp = xp_for(logits.device)
    if temperature == 0:
        return Tensor(xp.argmax(logits._data, axis=-1), device=logits.device)
    scores = logits._data.astype(xp.float32) / temperature
    if top_k is not None:
        # Limit normalization/nucleus/CDF work to K candidates rather than V.
        # Backend argpartition implementations may still sort internally.
        candidates = xp.argpartition(-scores, kth=top_k - 1, axis=-1)[:, :top_k]
        candidate_scores = xp.take_along_axis(scores, candidates, axis=-1)
        ranking = xp.argsort(-candidate_scores, axis=-1)
        order = xp.take_along_axis(candidates, ranking, axis=-1)
    else:
        order = xp.argsort(-scores, axis=-1)
    sorted_scores = xp.take_along_axis(scores, order, axis=-1)
    probabilities = xp.exp(sorted_scores - xp.max(sorted_scores, axis=-1, keepdims=True))
    probabilities = probabilities / xp.sum(probabilities, axis=-1, keepdims=True)
    if top_p < 1:
        keep = xp.cumsum(probabilities, axis=-1) - probabilities < top_p
        probabilities = xp.where(keep, probabilities, 0)
        probabilities = probabilities / xp.sum(probabilities, axis=-1, keepdims=True)
    uniform = random_uniform((logits.shape[0], 1), logits.device)
    cumulative = xp.cumsum(probabilities, axis=-1)
    cumulative = cumulative / cumulative[:, -1:]
    chosen = xp.sum(cumulative < uniform, axis=-1)
    chosen = xp.minimum(chosen, order.shape[-1] - 1)
    return Tensor(order[xp.arange(logits.shape[0]), chosen], device=logits.device)
