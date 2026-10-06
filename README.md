# Custom ML Library

[![Tests](https://github.com/KavinP21/custom-ml-library/actions/workflows/ci.yml/badge.svg)](https://github.com/KavinP21/custom-ml-library/actions/workflows/ci.yml)

A machine-learning framework with explicit reverse-mode automatic
differentiation. It implements computation graphs, backward rules, neural-network
layers, optimizers, and a decoder transformer. NumPy, CuPy, and MLX provide the
array operations; PyTorch is used only for independent comparisons.

## Training and extension APIs

- FP16 autocast for matrix multiplication, linear layers and convolutions, with
  FP32 parameters and dynamic loss scaling.
- Optimizer parameter groups, layer unfreezing and resumable learning-rate schedules.
- Custom backward functions, removable tensor-gradient hooks and module forward hooks.
- Synchronous data-parallel training with bucketed gradient averaging, replica
  checks and deterministic distributed sampling.
- Fixed-shape CUDA Graph inference replay and user CUDA kernels compiled through
  CuPy/NVRTC.

See [the advanced training guide](docs/advanced-training.md) for examples and
contracts. Run the two-process CPU example with
`python examples/train_distributed.py`, or the AMP example with
`python examples/train_mixed_precision.py --device cpu`.

The distributed tests execute real worker processes and compare their updates
against a global-batch reference. CUDA Graph/kernel execution needs NVIDIA
hardware; the GPU validation workflow runs those tests separately.

## Recorded results

Local experiments completed on an Apple M3 Max. These are
single-seed results under matched architectures and training protocols.

| Held-out task | This library | Matched PyTorch |
| --- | ---: | ---: |
| CIFAR-10 accuracy, 272K-parameter residual CNN | 90.26% | 89.99% |
| WikiText-2 perplexity, 7.96M-parameter transformer | 133.56 | 134.08 |

## Run on CPU

Requires Python 3.10 or later and NumPy.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/train_mlp.py
```

The example trains a small XOR classifier. A minimal backward pass is:

```python
import tensorsmith as ts

x = ts.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x * x).sum()
loss.backward()
print(x.grad.tolist())  # [2.0, 4.0, 6.0]
```

For Apple silicon, install `.[metal]`; for CUDA 12 use `.[cuda12]`, and for
CUDA 13 use `.[cuda13]`. See [GPU setup](docs/gpu.md) for toolkit options.
CUDA adapters are implemented, but NVIDIA hardware validation remains
pending. Accelerator availability is checked before use.

## How it works

Each differentiable operation records its parents and a vector-Jacobian-product
function. Backward visits the reachable graph in reverse topological order and
accumulates gradients where branches meet.

```text
Model, optimizer, data loader
             |
    Tensor and autograd graph
             |
     NumPy / CuPy / MLX
```

Start with [the tensor engine](src/tensorsmith/tensor.py),
[its gradient checks](tests/test_autograd.py), and the
[architecture guide](docs/architecture.md). The
[transformer guide](docs/transformers.md) covers attention, KV caches,
generation, and checkpointing.

## Verification

```bash
python -m pip install -e '.[dev]'
ruff check src tests examples benchmarks
ruff format --check src tests examples benchmarks
pytest
```

Tests cover numerical gradients, repeated indices, grouped convolutions,
attention, cache equivalence, checkpoint replay, optimizer continuation, and
half-precision training. A separate CI job installs PyTorch to execute the
independent reference tests. The [benchmark protocol](docs/task-benchmarks.md)
explains how to reproduce the recorded real-data experiments.

## Scope

The engine implements first-order derivatives. Distributed reduction currently
uses TCP with host staging, rather than GPU-native NCCL collectives or sharded
training. CUDA Graphs replay captured inference operations; they do not optimize
or compile arbitrary Python graphs. CUDA C++ kernels are compiled by CuPy/NVRTC.
Higher-order derivatives and an optimizing graph compiler remain unimplemented.


MIT licensed.
