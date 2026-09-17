# Architecture

## The core invariant

Every `Tensor` owns a backend array, a device, an optional accumulated gradient, and—only when needed—a
small piece of the dynamic autograd tape:

```text
Tensor
├── _data: NumPy / CuPy / MLX array
├── requires_grad: bool
├── _grad: backend array | None
├── _parents: tuple[Tensor, ...]
└── _backward(upstream) -> tuple[parent gradients, ...]
```

An operation such as `c = a * b` creates `c._data` (eagerly computed on NumPy/CuPy, deferred on MLX).
If gradient recording is enabled and
either input needs a gradient, it also records `(a, b)` and a closure implementing
`(dc/da, dc/db) = (upstream*b, upstream*a)`. This is reverse-mode automatic differentiation expressed
as vector-Jacobian products (VJPs); the framework never materializes a full Jacobian.

## Backward pass

For `loss.backward()`, TensorSmith:

1. Iteratively depth-first traverses the reachable graph to produce a topological order. The
   iterative implementation avoids Python's recursion limit on deep networks.
2. Seeds the scalar loss with one, or accepts an explicit upstream gradient for non-scalars.
3. Visits nodes in reverse topological order and calls each VJP closure.
4. Adds contributions in a temporary map when multiple graph branches meet.
5. Accumulates the resulting value into each tensor's public `.grad`.

The temporary map matters. Reusing a parameter's historical `.grad` while propagating would make a
second backward pass incorrectly re-propagate old gradients. TensorSmith separates gradients for the
current traversal from persistent leaf accumulation.

The same `_run_backward` traversal powers `ts.grad`, which returns requested gradients without
modifying buffers. Tape lifetime is explicit: ordinary passes release VJP closures/parents, and
reusing freed interiors raises; `retain_graph=True` keeps a tape for another first-order pass.
Higher-order derivatives are not recorded.

### Broadcasting

Forward broadcasting can insert leading dimensions or expand dimensions of length one. Its reverse is
`_sum_to_shape`: remove extra leading axes by summing, then sum every expanded singleton axis while
retaining its dimension. All binary VJPs pass through this operation.

### Non-smooth operations

- ReLU uses zero at exactly zero.
- `max` divides the upstream gradient equally among ties.
- elementwise `maximum` divides ties equally between operands.
- indexed reads scatter-add on backward, so repeated indices correctly accumulate.

These choices are explicit and testable rather than accidental consequences of implementation.

## Backend boundary

`device.py` is the only backend-selection layer. NumPy is always present. CuPy mirrors NumPy while
launching CUDA kernels. MLX provides lazy Metal arrays on Apple silicon; TensorSmith uses functional
indexed updates for MLX and mutable updates for NumPy/CuPy. Initializers use a host RNG so a seed gives
the same initial values across devices.

Moving a tensor across devices is a graph boundary. This avoids hidden cross-device copies during a
backward pass and makes placement errors immediate. Optimizer state is allocated next to its parameter.

`evaluate()` recursively materializes MLX arrays inside outputs, gradients, and optimizer/cache state;
`synchronize()` waits for submitted work. Neither a loss-only evaluation nor a queue barrier alone
proves a full backward/update has completed. This distinction is tested by the benchmark guards.

## Transformer operators

Dense projection, tanh-approximate bias-GELU, SiLU, normalization, cross-entropy and attention each
record one custom tape node rather than a large Python subgraph. This reduces tape overhead; it does
not imply every underlying array operation becomes one GPU kernel. Metal uses vendor fast norm and
attention **forward** kernels when available; every gradient formula remains TensorSmith code.

For attention, `P = softmax(scale * QKᵀ + mask)` and `O = PV`. The dense VJP is:

```text
dV = Pᵀ dO
dP = dO Vᵀ
dS = P * (dP - sum(P * dP, last_axis))
dQ = scale * dS K
dK = scale * dSᵀ Q
```

Dropout modifies effective probabilities for dV and masks dP. Boolean/additive/causal masks are
applied before normalization; all-masked rows return zero, including the vendor path.
GQA shares K/V heads across groups and sums group contributions on backward.

Streaming attention maintains a row maximum `m`, softmax denominator `l`, and weighted accumulator
`a` while visiting key blocks. For the new block maximum `m'`, rescale old values by `exp(m-m')`, add
`exp(scores-m')` contributions, then return `a/l`. It saves per-row log-normalizers, not the full
probability matrix; backward recomputes probabilities blockwise and uses `sum(dO*O)` for the softmax
projection. This is an explainable exact algorithm, not a hardware-tuned FlashAttention implementation.
Explicit user masks may still allocate quadratic storage; streaming also recomputes more kernels.

The decoder model owns embeddings, RMSNorms, GQA/RoPE projections, SwiGLU and weight tying.
Persistent inference caches own compact KV-head storage and position offsets. They are outside
autograd deliberately: training through overwritten history would be wrong. Capacity/schema checks
and rollback of logical lengths make inference failures recoverable.

## Recomputing activations

`nn.checkpoint` runs a pure module without recording its interior tape, materializes the lazy output,
then records input/parameter dependencies as one node. Its VJP replays the module with fresh input
leaves and uses `grad()` to return contributions without double-accumulating captured parameters.
NumPy RNG state is preserved, and native GPU uniform/dropout draws are retained/replayed. This avoids
private vendor RNG APIs but keeps stochastic draw memory; it is not a zero-overhead checkpoint engine.
Nested checkpoints flatten during recomputation. Stateful running-stat/cache updates are not safe.

## Low precision and resume

FP16 attention math, norm statistics and cross-entropy use FP32 accumulation. Adam holds FP32 master
weights and moments, casting only exposed low-precision parameters after updates. `GradScaler`
unscales/checks gradients before clipping, skips both weights and moments on overflow, then changes
the scale. This explicit policy is smaller than a complete autocast/mixed-dtype dispatch system.

Model state is host arrays; optimizer state maps parameters to stable positional indices and preserves
moment/master precision. Nested checkpoints store only arrays and a JSON container tree in NPZ,
with `allow_pickle=False` on load and atomic replacement on save. Application configuration,
vocabulary and RNG/data position belong in that same tree for meaningful reproducible resume.

## Convolution and pooling

Convolutions are cross-correlations, matching mainstream deep-learning APIs. Rather than looping over
individual output scalars, TensorSmith gathers strided/dilated windows into an `im2col`-style tensor and
uses one backend `einsum` contraction. Groups become an explicit dimension:

```text
input windows: [N, groups, Cin/group, Kh, Kw, Oh, Ow]
weights:       [groups, Cout/group, Cin/group, Kh, Kw]
output:        [N, groups, Cout/group, Oh, Ow]
```

The backward pass contracts the same tensors for weight and window gradients, then scatter-adds window
gradients into the padded input. The approach is compact, vectorized, supports groups/stride/padding/
dilation, and maps to accelerator kernels. Its tradeoff is the temporary window tensor; a production
compiler would replace it with fused direct-convolution kernels selected by an autotuner.

Pooling uses the same window geometry. Average pooling distributes gradient uniformly. Max pooling
builds a mask and divides among ties before scatter-add, preserving the gradient sum.

## Module system

`Parameter` is a trainable `Tensor`. `Module.named_parameters()` recursively walks attributes,
containers, and child modules while deduplicating shared parameters by identity. This avoids metaclass
magic and keeps registration easy to explain. Buffers such as batch-normalization statistics are
tracked separately: they move and serialize with a model but never reach an optimizer.

State dictionaries contain plain NumPy arrays and serialize through `np.savez` with pickling disabled.
That makes weight files inspectable and avoids arbitrary-code execution during loading.

## Complexity and tradeoffs

| Operation | Time | Important storage |
|---|---:|---:|
| Elementwise | O(n) | O(n) output |
| Matmul `[m,k]@[k,n]` | O(mkn) | O(mn) output |
| Autograd traversal | O(nodes + edges) | O(nodes) |
| Conv2d | O(N·Cout·Cin/groups·Kh·Kw·Oh·Ow) | im2col windows |
| Adam(W) | O(parameters) | two state arrays per parameter |

The Python graph layer introduces dispatch overhead for tiny operations. Large matrix and convolution
work remains in optimized backend kernels. A natural next step is graph capture plus elementwise fusion,
but that is kept outside the core so the current execution model remains transparent.
