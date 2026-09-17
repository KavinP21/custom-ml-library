# Local real-workload evaluation

Completed on September 17, 2026. Both full TensorSmith training runs were repeated after fixing
the memory-lifetime issue; final CPU/Metal acceptance, repeated timings and the consistency audit
all passed. Historical results and failures are retained separately.

## Scope and hardware

Executed locally on an Apple M3 Max with 128 GiB unified memory, macOS 26.5, Python 3.12.10,
NumPy 2.5.3, MLX 0.32.2 and PyTorch 2.14.0. TensorSmith owns the autograd tape and derivatives;
native NumPy/MLX kernels execute its array operations. PyTorch is only an independent benchmark
dependency, not the implementation of TensorSmith's training.

These are complete real-data research-task evaluations, **not MLPerf certification** or evidence
of frontier-scale convergence. See the [reproduction protocol](task-benchmarks.md) and
[campaign status/logs](../benchmarks/results/campaign-status.json).

## Held-out quality

| Workload | TensorSmith | Matched PyTorch | Train-fitted prior baseline |
| --- | ---: | ---: | ---: |
| CIFAR-10 test accuracy, higher is better | 90.26% | 89.99% | 10.00% |
| WikiText-2 test perplexity, lower is better | 133.56 | 134.08 | 914.42 |

CIFAR uses the full 45,000-image training partition for 30 epochs: 1,350,000 training examples,
5,000 validation images selected only from the original training split, and all 10,000 untouched
test images. Architecture: a 272,474-parameter, 20-layer residual CNN with projection shortcuts.
Training includes seeded random crops/flips, BatchNorm, momentum SGD and warmup/cosine decay.

WikiText uses all training text for ten epochs: 20,886,080 usable next-token training targets
and all 245,552 usable test targets. Architecture: a 7,963,968-parameter four-layer transformer
with GQA, RoPE, RMSNorm, SwiGLU and tied embeddings, trained with clipped AdamW and warmup/cosine.
The vocabulary is fitted only on training text. Perplexity is specific to this word-level,
fixed-stream/reset-context protocol, not directly comparable to BPE/sliding-window leaderboards.

Architectures, data hashes, validation indices, seed, CPU-created initial weights, batch sizes,
epoch budgets and optimization schedules match across engines. Checkpoints are selected by
validation NLL, never test results. Single quality seed (42): small differences do **not** establish
statistical quality superiority. Baselines use exactly the same scored test targets.

Raw reports: [TensorSmith vision](../benchmarks/results/cifar10-tensorsmith-metal-seed42.json),
[PyTorch vision](../benchmarks/results/cifar10-torch-metal-seed42.json),
[TensorSmith language](../benchmarks/results/wikitext2-tensorsmith-metal-seed42.json),
[PyTorch language](../benchmarks/results/wikitext2-torch-metal-seed42.json),
[train-fitted baselines](../benchmarks/results/train-fitted-baselines.json).

## Isolated completed-work speed

Both engines load the same trained TensorSmith checkpoint. Values below are the median of
three independent run medians, in milliseconds; parentheses show the range of those medians.

| Completed real-data workload | TensorSmith ms | Eager PyTorch ms |
| --- | ---: | ---: |
| Transformer training update, 16 x 128 tokens | 51.12 (50.19–51.65) | 58.61 (58.18–59.11) |
| Transformer full-vocabulary inference, 16 x 128 tokens | 8.32 (8.18–8.36) | 9.30 (9.23–9.34) |
| Residual CNN training update, 64 images | 45.39 (44.29–45.95) | 10.46 (10.28–10.93) |
| Residual CNN inference, 64 images | 13.02 (12.99–13.05) | 2.41 (2.37–2.49) |

On this configuration TensorSmith has 14.7% higher transformer training throughput and 11.7%
higher inference throughput than the eager/default-optimizer reference. **CNN training is 4.34x
slower and inference 5.39x slower.** This is not an across-the-board performance win.

An additional three fresh-process paired runs tested PyTorch's available **native fused AdamW**,
again alternating engine order, after the CPU-heavy suite finished:

| Additional transformer pair | TensorSmith ms | PyTorch with fused AdamW ms |
| --- | ---: | ---: |
| Complete training update | 44.85 (44.56–44.96) | 53.95 (53.74–54.48) |
| Full-vocabulary inference, original weights restored | 7.58 (7.53–7.67) | 8.33 (8.30–8.40) |

This paired set gives 20.3% higher training throughput and 9.9% higher inference throughput
for TensorSmith. It is a separate timing set, not a controlled optimizer-only ablation against
the earlier set: desktop/thermal state can differ. Both Torch variants are uncompiled FP32;
these results do not exhaust compiler, precision or backend tuning possibilities.

TensorSmith fixed-prefix decoding (512 real tokens, one additional token, same last-logit output)
measured 2.22 ms cached versus 3.12 ms uncached in the primary three-repeat set, about 1.40x faster.
Run medians span 2.22–2.41 ms cached and 3.02–3.14 ms uncached. This is not a growing-cache serving
benchmark or a comparison against PyTorch KV caching.

Each engine/task runs in three fresh processes, alternating engine order, with ten warmup
iterations followed by fifty recorded samples per phase. Complete updates include input transfer,
forward/loss/backward, clipping for the language model, optimizer updates and GPU synchronization.
Inference inputs are resident; every requested full-vocabulary/class logit is materialized.
Data loading and CPU augmentation are outside isolated timing, but included in full-training reports.
Raw samples, per-run medians/p95 and within-run bootstrap intervals are retained. Intervals do not
capture thermal drift or establish general superiority on other hardware/models.
The [audited summary](../benchmarks/results/audited-campaign-summary.json) contains each run's
key-phase median/p95 and memory counters, including the additional fused pairs; the underlying JSON files
retain all raw samples and bootstrap intervals. The read-only audit checks full epoch/target
budgets, matching data/model/hyperparameters and initial-weight hashes, validation-only selection,
acceptance coverage, all twelve primary reports and all six additional pair reports.

## Independent correctness and stateful paths

All 20 post-fix Metal-campaign acceptance groups and all 16 CPU groups passed, and the native
unit suite passed 81 tests plus 20 subtests. The sandboxed CPU-only rerun passed 80 tests,
with one Metal test explicitly skipped. The suite checks complete trained-model outputs and every
parameter-gradient element against independently differentiated PyTorch, including running BN
buffers. Optimizer arithmetic and moments are then compared using identical gradients, isolating
Adam's near-zero sign sensitivity from accepted floating-point derivative differences. Actual
full training uses each engine's own gradients throughout.
The wheel also built successfully: installed runtime files match the source, the framework
imports without PyTorch, and the installed wheel passed the same 80-test CPU suite (one Metal skip).

| Independent Metal/PyTorch derivative comparison | Gradient elements per batch | Real batches | Largest per-parameter relative L2 error |
| --- | ---: | ---: | ---: |
| Trained residual CNN | 272,474 | 3 | 5.34e-6 |
| Trained four-layer transformer | 7,963,968 | 3 | 3.76e-5 |
| Full 2,048-token transformer backward | 7,963,968 | 1 | 1.40e-4 |
| 129M-parameter transformer, 512 tokens | 129,385,728 | 1 | 4.37e-6 |

Every gradient element was compared, not only these norm summaries. Whole-model FP64 directional
checks run on CPU explicitly, including in the Metal campaign; this is not Metal FP64 support.

Coverage includes full 2,048-token transformer backward, a 129,385,728-parameter real-text
transformer and all its gradients, FP64 whole-model directional differences, stochastic activation
recomputation/RNG continuation, microbatch accumulation, attention implementations, portable
model/optimizer/scheduler continuation, real DataLoader behavior, composed 1D/2D convolutions/pools,
GELU/LayerNorm, and scaled FP16 training and large FP16 BatchNorm reductions.

Cache checks compare 640 real tokens in unequal chunks against full-context logits, quantized-cache
NLL, duplicate/reordered beams and 64 greedy tokens versus uncached generation. Metal checks forbid
floating/bulk compatibility fallbacks in native device helpers; intentional input transfer,
label/scalar checks and explicit oracle exports remain allowed.
On the two 640-token real sequences, float-cache logits had maximum absolute error 1.43e-5,
all 64 greedy tokens matched uncached generation, and int8-cache absolute NLL drift was 0.000589.
Persistent KV storage decreased from 3,145,728 to 884,736 bytes (3.56x smaller, cache only).
FP16 held-out NLL drift was 0.000029 across 2,048 scored targets; eight scaled training updates
completed without a skipped update. These are bounded subset checks, not full-corpus quantized quality claims.

Raw acceptance: [Metal](../benchmarks/results/real-task-acceptance-metal.json),
[CPU](../benchmarks/results/real-task-acceptance-cpu.json).

## Bugs found, fixed and retested

- Python scalar promotion caused CPU float32 precision/state drift; scalar coercion and mixed-dtype
  gradient destinations now preserve the intended precision.
- BatchNorm running variance needed the unbiased correction; singleton training statistics reject.
- Large FP16 means/variances and BN statistics overflowed; accumulation and statistics use FP32.
- Max-pooling ties needed first-winner rather than split gradients.
- A recursive `evaluate()` closure retained GPU outputs until cyclic garbage collection. Iterative
  traversal releases them immediately. The real-text regression completes 32 full-logit forwards
  with cyclic GC disabled and requires bounded active allocations, excluding allocator cache.

The first Metal acceptance attempt also exposed a harness error: its own intentional RNG export
was mistakenly rejected by the host-fallback guard. The export was corrected without weakening
that guard. [The failed attempt](../benchmarks/results/real-task-acceptance-metal-attempt1.json)
and [pre-memory-fix reports](../benchmarks/results/pre-memory-fix) remain available. Older training
times/peaks are historical, not post-fix performance claims; the first language run also overlapped
background CPU diagnostics, so its time is not a controlled memory-fix speed ablation.

Process RSS, MLX peak-active allocations, MLX cache and Torch MPS end-of-phase current/driver
allocations are different measures. They must not be compared as interchangeable GPU peaks.
The refreshed full language run recorded 1,515,114,268 bytes of MLX peak-active allocations
(1.41 GiB), versus the historical 26,755,881,000-byte peak. Completed inference's active memory
is now about 32 MB rather than the historical multi-gigabyte retention; the GC-disabled real-text
regression recorded exactly zero growth across 32 forwards. Neither comparison is a controlled
speed ablation. Refreshed full training/evaluation took 579.8 seconds for language and 1,065.7
seconds for vision, excluding initial data/model setup but including validation/checkpoint work.

## Limits

No NVIDIA hardware is present: CUDA is unverified here. Two real tasks and a large functional
stress case cannot prove every operator configuration, dtype, mask or boundary case. The 129M
model receives five updates, **not** a convergence budget or quality claim; extending the trained
small model to 2,048 tokens is a derivative stress test, not long-context quality evidence.
This evaluation did not assess distributed training, billion-parameter convergence, pretrained interoperability,
production serving evaluation or full original-paper convergence budget in this campaign.

Dataset sources: [official CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html),
[University of Toronto's hosted copy](https://huggingface.co/datasets/uoft-cs/cifar10),
[official PyTorch word-language-model example](https://github.com/pytorch/examples/tree/main/word_language_model)
and [Salesforce WikiText](https://huggingface.co/datasets/Salesforce/wikitext).
Pinned revisions and SHA-256 hashes are in the raw reports.
