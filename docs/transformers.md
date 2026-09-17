# Transformer training and inference

`TransformerLM` is a decoder-only model: token embedding → pre-RMSNorm decoder blocks with
GQA/RoPE attention and SwiGLU MLPs → final RMSNorm → optionally tied vocabulary projection.
Inputs are integer `[batch,time]` tensors; logits are `[batch,time,vocabulary]`.
This is a configurable small research model, not an out-of-the-box pretrained Llama/GPT loader.

## Training

```python
import tensorsmith as ts
from tensorsmith import nn

model = nn.TransformerLM(
    nn.TransformerConfig(
        vocab_size=256,
        dim=128,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        hidden_dim=384,
        max_seq_len=512,
        activation_checkpointing=False,
    ),
    device="cpu",
)
optimizer = ts.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = ts.optim.WarmupCosineLR(optimizer, warmup_steps=100, total_steps=1000)
tokens = ts.tensor([[1, 2, 3, 4]])
targets = ts.tensor([[2, 3, 4, 5]])
optimizer.zero_grad()
loss = model.loss(tokens, targets, ignore_index=-100, label_smoothing=0.1)
loss.backward()
nn.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
scheduler.step()  # AFTER the optimizer; constructor sets first warmup LR
ts.evaluate(loss, list(model.parameters()), optimizer.state)
```

`model.loss` uses class axis `-1`. `nn.CrossEntropyLoss` defaults to class axis `1` for compatibility
with the existing classifier API; instantiate `CrossEntropyLoss(axis=-1)` for LM logits.
Mean loss divides by valid tokens, not padded sequence length. An all-ignored batch yields zero loss
and zero gradients. Cross-entropy saves `[tokens,vocabulary]` probabilities, never `eye(vocabulary)`.
Label range/shape validation reads labels on the host: robust, but an O(tokens) synchronization cost.

For gradient accumulation, zero once, backward on each `microbatch_loss / accumulation_steps`,
then clip/step once. Equal microbatch sizes/valid-token counts are assumed by that averaging rule;
otherwise weight each microbatch by its fraction of the total valid tokens.

## Attention contracts

`nn.scaled_dot_product_attention(q,k,v,...)` uses `[B,H,T,D]`. Boolean masks mean `True=allowed`
(not the inverted convention used by some APIs). Float masks add to scores; use `-inf` to disallow.
Masks must be nondifferentiable, on the input device, and broadcastable to `[B,Hq,Q,K]`.
For padding, a boolean key-valid mask shaped `[B,1,1,K]` works; combine with `is_causal=True`.
Fully masked rows have zero outputs and gradients. Attention dropout is active whenever `dropout_p`
is positive; `MultiHeadAttention` automatically disables it in eval mode.

- `dense`: standard matmul/softmax attention, saved probabilities, dropout supported.
- `streaming`: exact online-softmax blocks, recomputing VJP; avoids a saved full attention matrix.
  FP rounding can differ. `block_size=64` by default; no dropout yet. Python block dispatch can be
  slower than dense kernels, particularly for small sequences.
- `native`: Metal vendor attention forward, TensorSmith recomputing VJP. Small backward workloads
  use dense recomputation; larger ones are blockwise. No dropout. This is not a custom FlashAttention
  CUDA kernel or a fully fused training backward.
- `auto`: native on Metal when head dimensions/dropout permit; otherwise dense, switching to
  streaming above `Q*K > 1024*1024` when dropout is zero. This is a documented heuristic, not
  device-specific autotuning. Override and benchmark it for your workload.

`enable_gqa=True` allows query heads to be an integer multiple of KV heads. Metal native forward
consumes grouped heads directly; dense/recomputing paths expand raw KV arrays and sum gradients
back into groups. Persistent KV cache size follows **KV** heads, not query heads.
Interleaved-pair RoPE supports offset/partial rotary dimensions. It is not automatically interchangeable
with rotate-half pretrained checkpoint conventions without converting projection weights.

`MultiHeadAttention(x, context=other, is_causal=False)` also supports cross-attention. Cross-attention
KV caching is not implemented. Inputs/masks and every parameter must use compatible placement/dtypes.

## Manual KV cache

```python
model.eval()
with ts.no_grad():
    caches = model.new_cache(batch_size=1, max_seq_len=128)
    prefill = model(tokens, caches=caches, logits_to_keep=1)
    one_token = ts.tensor([[5]])
    logits = model(one_token, caches=caches, logits_to_keep=1)
    ts.evaluate(logits, [cache.storage for cache in caches])
```

Each layer owns fixed-capacity `[B,Hkv,capacity,D]` K/V arrays. RoPE starts at the cache's current
length; causal query positions use the same offset, including multi-token chunks. Append validates
shape/capacity before writing. A failed model forward rolls back all layer lengths. Mutation requires
`eval()` and `no_grad()`; silently detaching a training graph would give incorrect gradients.

`reset()` reuses capacity; `truncate(n)` rolls the logical history back; `reorder(indices)` reorders
or duplicates fixed-width batch rows for beam-search building blocks. These are not a complete beam
search or paged-attention serving implementation. Do not share a mutable cache across concurrent calls.
`memory_bytes` reports persistent payload only, not temporary arrays/runtime overhead.

`model.new_cache(..., quantized=True)` stores symmetric int8 K/V plus FP32 per-token/head scales.
It reduces persistent storage at the cost of approximate logits. Attention dequantizes the active
prefix: no fused int8 kernel or automatic decoding speedup is claimed. Quantization is not QAT and
does not quantize model weights.

`logits_to_keep=1` projects only the last hidden position to vocabulary logits, avoiding a needless
`[B,T,V]` allocation during prefill/full-prefix generation. Default `0` returns all positions for training.

## Generation

```python
output = model.generate(tokens, max_new_tokens=32, temperature=0.8, top_k=40, top_p=0.9)
greedy = model.generate(tokens, 32, temperature=0)
```

Generation includes the prompt in its return value. It uses device-resident sampling and per-layer
caches by default, switches to eval/no-grad, then restores each module's prior training mode.
`use_cache=False` is a correctness/debugging baseline. `quantized_cache=True` opts into approximate
int8 storage. `eos_token_id` stops once every batch row has finished (finished rows emit EOS until
then). EOS detection reads one scalar; outputs/cache arrays are evaluated per token. Context overflow
raises rather than silently dropping history. There is no sliding-window, continuous batching or
speculative decoding yet.

## Explicit FP16 training

Build/move the model **before** constructing/restoring the optimizer:

```python
model.to(dtype="float16")
optimizer = ts.optim.AdamW(model.parameters(), lr=0.001)
scaler = ts.amp.GradScaler(init_scale=128)
optimizer.zero_grad()
loss = model.loss(tokens, targets)
scaler.scale(loss).backward()
finite = scaler.unscale_(optimizer)
if finite:
    nn.clip_grad_norm_(model.parameters(), 1.0)
applied = scaler.step(optimizer)  # False on overflow; parameters/moments unchanged
scaler.update()
ts.evaluate(loss, list(model.parameters()), optimizer.state)
```

Attention math, norm statistics, and cross-entropy promote FP16 to FP32 where needed; Adam maintains
FP32 master weights/moments to preserve small updates. This is explicit low-precision support, not
automatic autocast or a comprehensive BF16/FP8 policy. Gradient/finite checks may synchronize.
Only advance the learning-rate scheduler when `applied` is true. Unscale once, before clipping;
call `update()` after each scaler step.

## Activation checkpointing and pure gradients

`TransformerConfig(activation_checkpointing=True)` recomputes each decoder block during backward.
For standalone modules use `nn.checkpoint(module, x)`. For a closure capturing trainable parameters,
pass `parameters=module.parameters()` explicitly. The callable returns one Tensor and must be pure.
Training BatchNorm is rejected because recomputation would update running statistics twice; parameters,
module modes, caches, input contents and control flow must not change between forward/backward.

Checkpointing uses TensorSmith's own `grad()` engine without modifying nested `.grad` buffers.
NumPy randomness is restored; TensorSmith native dropout/uniform GPU draws are replayed from saved
draw arrays, so replay does not consume future RNG values. Those stochastic arrays still occupy memory;
external Python/library RNGs are not preserved. Metal checkpoints evaluate the forward boundary to
release the lazy backend's interior dependencies; this can add synchronization overhead.

`ts.grad(output, inputs, grad_outputs=None, retain_graph=False, allow_unused=False)` returns a tuple
of first-order gradients without touching existing `.grad` buffers. Unused inputs return `None` when
allowed. No higher-order/create-graph support.

## Save/resume

```python
ts.save(
    {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": 10,
    },
    "training.npz",
)
state = ts.load("training.npz")
model.load_state_dict(state["model"])
optimizer.load_state_dict(state["optimizer"])
scheduler.load_state_dict(state["scheduler"])
scaler.load_state_dict(state["scaler"])
```

State is portable NPZ arrays plus a JSON tree, never executable pickle. Saving uses atomic replacement.
Legacy flat NPZ weights still load. Optimizer restoration uses parameter order and checks shapes;
model loading preserves destination dtype. Save configuration, vocabulary, RNGs and data-loader
position too for an application-level reproducible resume. The offline example saves NumPy batch RNG,
configuration/vocabulary/corpus hash/dtype and rejects incompatible resumes; its scheduler horizon
is restored, not silently extended by a larger `--steps` argument.
