# Real-data task benchmark protocol

These evaluations complement, not replace, numerical/unit tests and operator benchmarks.
They train full models on established datasets and score held-out splits. They are **not
MLPerf-certified**, frontier-scale training demonstrations, or state-of-the-art accuracy claims.

## Reproduce

Install `pip install -e '.[metal,dev,benchmark]'` on Apple silicon (or choose the CPU/CUDA extra).
PyTorch, Pillow, and Arrow are benchmark-only dependencies, never framework dependencies.

```bash
python benchmarks/task_data.py --task all --cifar-source huggingface

python benchmarks/bench_tasks.py --task wikitext2 --engine tensorsmith --device metal \
  --epochs 10 --batch-size 16 --eval-batch-size 16 --output benchmarks/results/wikitext2-tensorsmith-metal-seed42.json \
  --checkpoint benchmarks/checkpoints/wikitext2-tensorsmith-metal-seed42.npz

python benchmarks/bench_tasks.py --task cifar10 --engine tensorsmith --device metal \
  --epochs 30 --batch-size 64 --eval-batch-size 64 --lr 0.1 --min-lr 0.001 \
  --weight-decay 0.0005 --warmup-steps 100 --output benchmarks/results/cifar10-tensorsmith-metal-seed42.json \
  --checkpoint benchmarks/checkpoints/cifar10-tensorsmith-metal-seed42.npz
```

Run matched references by changing `--engine` to `torch`, keeping **every** model/data/optimization
argument identical, and assigning a different checkpoint/report filename. Execute GPU jobs serially.
`--max-steps` is an explicit diagnostic cap: reports from capped runs are not full-data results.

After training:

```bash
python benchmarks/verify_tasks.py --device metal \
  --language-checkpoint benchmarks/checkpoints/wikitext2-tensorsmith-metal-seed42.npz \
  --vision-checkpoint benchmarks/checkpoints/cifar10-tensorsmith-metal-seed42.npz \
  --output benchmarks/results/real-task-acceptance-metal.json

python benchmarks/bench_task_speed.py --engine tensorsmith --device metal --batch-size 16 \
  --checkpoint benchmarks/checkpoints/wikitext2-tensorsmith-metal-seed42.npz \
  --output benchmarks/results/wikitext2-isolated-speed-tensorsmith-metal.json
```

Repeat acceptance on `--device cpu`. Run isolated speed for each engine in a separate process,
with no other GPU task, and repeat independently before drawing strong speed conclusions.
`run_campaign.py` automates the two full matched references, a separately labeled five-update
129M-parameter real-text stress run, CPU/Metal acceptance, the native unit suite, and three
fresh-process timing replicates with alternating engine order. It assumes the full TensorSmith
training checkpoints above exist; `--wait-for REPORT.json` waits for the current GPU job first.
After the complete campaign, `python benchmarks/audit_campaign.py` performs a read-only consistency
audit: uncapped epoch/target budgets, validation-only selection, matching data/model arguments,
available initial-weight hashes, acceptance coverage and all twelve fresh-process timing reports.
`--start-at JOBNAME` resumes after a failure without hiding earlier attempt logs/status.
After a runtime fix, archive old results/checkpoints and use `--refresh-tensorsmith
--start-at tensorsmith-wikitext2-refresh` to repeat both full training runs and subsequent checks
without rerunning already-completed matched references. Historical versions must remain identifiable.
`python benchmarks/bench_baselines.py` scores train-fitted unigram/class-prior baselines on exactly
the same usable held-out targets; these are useful checks that the networks learned beyond priors.

## Datasets and held-out quality

- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): all 50,000 original training and 10,000
  test images. A seeded, stratified 5,000-image validation subset comes **only** from training,
  leaving 45,000 actual training images. Random padded crop/horizontal flip apply only at training;
  all splits use documented channel normalization. A 20-layer, 16/32/64-channel residual CNN uses
  batch normalization and projection shortcuts (a specified variant, not paper-identical ResNet20).
  Report full test top-1 accuracy and mean NLL, selecting the checkpoint by validation NLL only.
- [WikiText-2](https://huggingface.co/datasets/Salesforce/wikitext), tokenized distribution from
  [PyTorch's official word-language-model example](https://github.com/pytorch/examples/tree/main/word_language_model):
  all train/validation/test text. Whitespace words plus `<eos>` for every line, **training-only**
  vocabulary, out-of-vocabulary words mapped to `<unk>`. Contiguous fixed streams, next-token
  prediction, reset attention context per segment. Full usable validation/test targets, including
  short final segments; at most batch-size-minus-one trailing tokens and the first token of each
  stream lack a scored predecessor. Mean token NLL and `exp(NLL)` perplexity. This is not directly
  comparable to raw/BPE, sliding-window, or all-split-vocabulary leaderboard perplexities.
  The default model has four decoder layers, dimension 192, six query/two KV heads, SwiGLU hidden
  size 512, RoPE, RMSNorm, and tied embedding/output weights. It is trained from scratch.

File hashes, immutable source revisions, split sizes, validation-index hash, arguments, source
digest, environment, epoch curves and all timing samples are retained in JSON. Public dataset
licenses remain their owners' licenses. Dataset downloads/checkpoints are ignored, not vendored.

## Checks beyond learning curves

`verify_tasks.py` records each independent pass/failure and exits nonzero on any failure:

- Full residual network and transformer outputs/losses against independently differentiated
  PyTorch; **every element of every parameter gradient**, plus actual SGD/AdamW updates.
  Optimizer arithmetic is isolated using identical gradients after the independent derivative
  checks: Adam's first update is sign-sensitive near zero, so allowable floating-point gradient
  noise must not be confused with an incorrect optimizer. Full task runs still use each engine's
  own natural gradients, without this controlled substitution.
- Residual-network batch-normalization running buffers against the reference.
- Whole-model FP64 central directional differences on CPU, three directions/three step sizes,
  over the full trained architectures, independently of PyTorch autograd.
- Recomputed activations and microbatch accumulation versus ordinary transformer gradients.
- Dense/streaming/native-auto attention derivatives over real language batches; stochastic
  checkpoint recomputation with dropout and exact continuation of the device RNG.
- Repeated completed full-vocabulary Metal inference with cyclic garbage collection disabled;
  active output memory must not accumulate (allocator cache is accounted separately).
- Optional `--long-context` checks a complete 2,048-real-token transformer backward pass against
  the independent oracle. The context capacity is extended for this functional stress test;
  it is not a long-context quality claim about a model trained with 128-token segments.
- Safe disk checkpoint restore followed by real-batch continuation; model, buffers, optimizer
  moments and scheduler state must agree with uninterrupted execution.
- 640 real-text tokens in unequal cached chunks, exact-cache full logits, int8-cache NLL drift,
  beam reordering/duplication, and 64 generated greedy tokens versus uncached generation.
- FP16 held-out NLL drift and multiple scaled/clipped/AdamW training updates with finite weights.
- FP16 BatchNorm on real 64-image batches (65,536 statistics values/channel), including input
  and affine gradients and unbiased running variance; means/variances/statistics accumulate in FP32.
- Grouped/dilated Conv1d, sigmoid/tanh, max/average pooling and linear projection composing over
  real image signals, with an independent output and input/weight-gradient oracle.
- A second real-image classifier combines grouped/dilated Conv2d, both 2D pools, fused bias/GELU,
  LayerNorm and classification; input and every weight gradient are independently checked.
  Max-pooling ties select the first window element, matching the PyTorch pooling convention.
- Real CIFAR images through `TensorDataset`/`DataLoader`: deterministic shuffle, aligned inputs,
  all indices visited exactly once, the short final batch preserved and inference NLL unchanged.

Metal acceptance forbids bulk host compatibility fallbacks in native indexed-update helpers,
including convolution backward and caches. Intentional input transfer, integer-label validation,
scalar overflow checks and explicit output/gradient export for the independent oracle are allowed.

Thresholds live in the checked-in test code. Passing tolerances demonstrates bounded numerical
agreement, not bitwise identity. Quantized caches and FP16 intentionally receive different,
quality-oriented acceptance limits from FP32 exact-cache checks.

## Performance interpretation

Training times include completed forward/loss/backward/optimizer updates and synchronization,
not just construction of lazy expressions. The training-run curves include CPU augmentation and
input transfer; the isolated timing uses a preprocessed real batch and includes transfer during
training, but resident inputs during inference. Warmup updates execute normally but are excluded
from timing statistics. Medians, p95, raw samples and a within-run bootstrap interval are retained.
The interval does not capture thermal drift, correlated samples, or between-run variation.

Process peak resident memory includes data/CPU/runtime/unified allocations. MLX allocator peak
is separately labeled and is **not** interchangeable with process RSS or a Torch MPS peak.
The isolated report records MLX active/cache/phase-peak and Torch MPS end-of-phase allocator/driver
values with this distinction; optimizer state and gradients are released before inference.
Models/data/weights/seeds and optimization schedules are matched; backend kernel implementations
are deliberately allowed to differ. Dropout defaults to zero, avoiding unmatched RNG masks in the
direct numerical comparisons. Changing precision, dropout, sequence length or batch size creates
a different benchmark configuration and must be reported.
The matched PyTorch reference is eager/uncompiled and language AdamW uses `foreach=False`
(also the installed MPS default). An additional isolated `--engine torch --torch-fused-optimizer`
language run checks native fused AdamW as a separately labeled optimized baseline; it does not
silently fall back if the installed hardware/runtime lacks support. Neither baseline exhausts
possible PyTorch compilation/precision/tuning configurations.

## What this still cannot establish

There is no NVIDIA GPU on this Mac, so these runs cannot validate CUDA. This protocol does not assess distributed
training, billion-parameter convergence, pretrained-checkpoint ecosystem, production serving,
custom-kernel compiler, or MLPerf compliance in this protocol. Not every operator configuration,
mask shape, boundary case or numerical regime is covered by two tasks. The separate test suite
remains necessary. Single-seed task results do not establish seed-independent accuracy, and a
specified finite epoch budget is not training to the original papers' full convergence budgets.
