# Completed-work benchmarks

The old Metal benchmark synchronized a queue but did not materialize lazy backward expressions.
Its numbers could describe graph construction, not completed GPU work. **Do not reuse those numbers.**
The new shared harness evaluates returned outputs/gradients/updated state and synchronizes before
stopping each sample timer; a regression test enforces that ordering.

## Reproduce

```bash
python benchmarks/bench_ops.py --device cpu --size 512 --iterations 50 --warmup 10 --json ops.json
python benchmarks/bench_transformer.py --device metal --iterations 50 --warmup 10 --json lm.json
python benchmarks/bench_transformer.py --device metal --seq-len 512 --dim 128 --vocab 1024 \
    --iterations 50 --warmup 10 --torch --json lm-long.json
```

`--torch` needs a separate optional PyTorch installation; it is not a library dependency. It compares
same-input, same-FP32-shape causal attention on CPU/CUDA or Metal vs PyTorch MPS, checks gradients
before timing, and uses that runtime's own synchronization. It does **not** compare different LM
architectures and present that as a framework speed ratio. PyTorch uses its normal SDPA dispatch;
this is an eager end-to-end baseline, not a best-possible compiled/tuned configuration.

`--activation-checkpointing` changes LM training to recomputation. `--quantized-cache` opts into
approximate int8 inference storage and reports persistent cache bytes and maximum logit error against
full-precision decoding. It need not be faster. Published reference cases leave both disabled.

## Timing contract

- Warmup excluded (kernel/runtime/compiler/cache initialization is not steady-state latency).
- FP32, identical shapes/dtypes/inputs for attention reference comparisons. Inference uses no-grad;
  forward/backward uses the same mean-squared output loss and gradients for all Q/K/V.
- Wall clock includes Python dispatch, tape/gradient allocation, explicit evaluation and synchronization.
  These are **not** kernel-only timings. Gradient zeroing is included; input/model creation excluded.
- Training includes cross-entropy, complete backward, AdamW update and updated moments/parameters.
  Parameters evolve over warmup/samples: this is throughput measurement, not a quality comparison.
- Prefill includes filling all layer caches and only the last vocabulary-logit projection. Cached
  decode restores a fixed prefix length before each one-token append; full-prefix decode uses the
  same prefix+token and also projects only the last position. Correctness is checked before timing.
- Each JSON includes all latency samples, mean/median/interpolated p95, arguments, package versions,
  hardware/runtime metadata, thread environment and PyTorch thread count. Throughput is work units
  divided by median latency; for LM cases those units are processed/generated tokens.
- Run scripts serially without other heavy work. CPU BLAS threading and GPU scheduling/thermal state
  affect results. No isolated machine/fixed-clock guarantee; tails are shown rather than hidden.

## Local evidence

Reference hardware: Apple M3 Max, 128 GiB unified memory, macOS 26.5, Python 3.12.10, NumPy 2.5.3,
MLX 0.32.2, PyTorch 2.14.0. Reports use 50 measured iterations after 10 warmups; thread settings
are in the JSON. This is one machine, two small synthetic cases, not frontier-model scale or an
accuracy/data-loading/serving evaluation.

Raw reports: [CPU short](../benchmarks/results/m3max-cpu-small.json),
[CPU longer](../benchmarks/results/m3max-cpu-long.json),
[Metal short](../benchmarks/results/m3max-metal-small.json),
[Metal longer](../benchmarks/results/m3max-metal-long.json).

Short: batch 1, sequence 128, dim 64, heads 4/KV heads 2, layers 2, vocabulary 256.
Longer: batch 1, sequence 512, dim 128, heads 4/KV heads 2, layers 2, vocabulary 1024.
Standalone attention uses equal query/KV heads; the decoder uses GQA.

Selected medians in milliseconds (all p95 values/raw samples remain in the JSON):

| Workload | CPU short | CPU longer | Metal short | Metal longer |
|---|---:|---:|---:|---:|
| LM training, including AdamW | 3.584 | 29.296 | 6.097 | 7.949 |
| One-token decode, cached | 0.475 | 0.576 | 1.670 | 1.715 |
| One-token decode, full prefix | 1.552 | 17.470 | 1.913 | 3.455 |
| TensorSmith attention inference (CPU dense / Metal native) | 0.292 | 5.128 | 0.486 | 0.718 |
| PyTorch SDPA inference | 0.088 | 0.381 | 0.559 | 0.385 |

For the longer case, cache decoding is about **30× faster on CPU** and **2× faster on Metal** than
recomputing the prefix with the same model. Metal training is about 3.7× faster than CPU for that
particular model, but the tiny model trains faster on CPU. Short Metal attention inference narrowly
beats this PyTorch MPS run; longer Metal attention does not. These are local shape-specific outcomes,
not a claim about framework superiority or independent full-LM PyTorch throughput.

The original matmul case has separate [CPU](../benchmarks/results/m3max-ops-cpu.json) and
[Metal](../benchmarks/results/m3max-ops-metal.json) reports, also with equivalent PyTorch gradients.

The raw data is the authority. On CPU, PyTorch's optimized SDPA is faster than TensorSmith's
array-composed attention. On Metal, native forward avoids the Python/blockwise overhead of streaming
attention, but performance relative to PyTorch changes with shape and scheduling; it is not a blanket
win. Cached decoding avoids old-token projections/MLPs/queries and benefits longer prefixes much more
than the tiny context, where dispatch/synchronization dominate. Int8 storage only promises a smaller
persistent payload. Streaming attention trades saved attention state for extra recomputation/dispatch;
especially on Metal, the Python block loop can be much slower.

## Next performance work

1. Validate CUDA numerical parity and profiles on an actual NVIDIA GPU. No NVIDIA hardware was
   available locally, so no CUDA speed claim is made.
2. Optimize the measured attention VJP/dispatch bottleneck: custom fused CUDA/Metal backward and
   better block scheduling, with mask/dropout/GQA parity and memory profiles.
3. Graph capture/fusion and automatic mixed precision, evaluated at realistic batch/context sizes.
4. Pretrained-checkpoint/tokenizer conversion and useful held-out quality workloads; then
   distributed/sharded training and production serving as separate engineering efforts.

Frontier-lab value here is the tested systems reasoning and readable implementation—not an assertion
that a compact custom framework already competes with their production training stacks.
