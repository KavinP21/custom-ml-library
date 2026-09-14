# Changes

## 0.3.0

- CUDA 12/13 installation extras, with optional CUDA 13 toolkit components.
- Per-group optimizer settings and appended parameter groups for layer unfreezing.
  Schedulers follow appended groups; SGD momentum owns independent gradient storage.
- Custom first-order backward functions, saved forward snapshots, removable
  tensor-gradient hooks and module pre/post-forward hooks.
- FP16 autocast for matmul, linear and convolution operators while preserving FP32
  parameters. Activation recomputation restores precision/RNG policy and defers
  parameter hooks to the outer backward pass.
- Pickle-free TCP collectives, replica schema checks, bucketed gradient averaging,
  explicit distributed synchronization and equal-length distributed samplers.
- CUDA Graph inference replay with storage/mode guards, plus CuPy/NVRTC user kernels.
  Actual NVIDIA execution is covered by a separate hardware validation workflow.

The tensor engine remains first-order. Distributed communication stages through
host memory and does not provide NCCL, sharding or elastic membership. CUDA Graphs
are inference-only; NVRTC compilation is supplied by CuPy.

## 0.2.0

- Correct lazy Metal benchmark timing: materialize losses, gradients, parameters, optimizer/cache
  state before synchronization. Separate warmup/inference/training/prefill/fixed-prefix decode,
  include median/mean/p95/raw samples/environment JSON and optional same-shape PyTorch baselines.
- Indexed stable cross-entropy with arbitrary class axis, ignore index and label smoothing;
  removes the previous vocabulary-squared identity/one-hot allocation.
- Dense/online-softmax/native-Metal attention with explicit Q/K/V VJPs, causal offsets, masks,
  fully-masked row semantics, GQA and interleaved RoPE.
- RMSNorm/SiLU/SwiGLU, single-node projection/bias-GELU/residual normalization, device-native dropout.
- Decoder-only TransformerLM/config/blocks, tied weights, last-position-only projection,
  cache-aware chunked prefill and batched greedy/top-k/nucleus/EOS generation.
- Preallocated float or per-token/head int8 KV storage, reset/truncate/reorder, capacity checks
  and rollback of layer cache lengths on failed inference.
- Pure first-order grad() and activation checkpoint recomputation with TensorSmith dropout replay.
- FP32 Adam master weights/moments for FP16, dynamic GradScaler/overflow skips, warmup-cosine,
  portable optimizer/scheduler resume, and atomic nested pickle-free training checkpoints.
- Offline character-LM example, transformer/performance guides, finite-difference/integration/
  real-Metal parity tests and an optional independent PyTorch whole-transformer oracle.

Limitations of 0.2: no distributed/sharded training, autocast, custom fused CUDA attention/backward,
pretrained checkpoint/tokenizer interoperability, or production paged serving. Local CUDA hardware
validation remains outstanding; this development machine is Apple silicon.
