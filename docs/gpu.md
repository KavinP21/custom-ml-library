# GPU backends

## NVIDIA CUDA

TensorSmith uses CuPy arrays and accepts `cuda` or `cuda:N` device names. Install the CuPy wheel matching
the installed CUDA major version. Choose one extra in a clean environment:

```bash
pip install -e '.[cuda12]'      # CUDA 12; the legacy [cuda] extra is equivalent
pip install -e '.[cuda13]'      # CUDA 13, using CuPy 14
```

If a system CUDA 13 toolkit is not installed, `.[cuda13-ctk]` also installs the
CUDA components supplied through CuPy's `ctk` extra. A compatible NVIDIA driver
and supported GPU are still required. Do not install CUDA 12 and CUDA 13 CuPy
wheels together: both provide the same `cupy` module. Switching major versions
requires a separate environment or removing the old wheel first.

These installation paths follow the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).
The adapter uses the same array API for both versions. Package configuration
does not establish execution or performance on NVIDIA hardware; run the checks
below and the parity suite on that hardware.

Verify both runtime discovery and a backward pass:

```python
import tensorsmith as ts

assert ts.is_available("cuda")
x = ts.randn(1024, 1024, device="cuda", requires_grad=True)
loss = (x @ x).mean()
loss.backward()
ts.evaluate(loss, x.grad)
ts.synchronize("cuda")
print(x.grad.device)
```

TensorSmith does not silently move mixed-device operands. This prevents an unnoticed PCIe transfer from
turning into a performance bug.

## Apple Metal

The Metal backend uses MLX and is available on Apple silicon:

```bash
pip install -e '.[metal]'
```

```python
import tensorsmith as ts

assert ts.is_available("metal")
model = ts.nn.Linear(4096, 4096, device="metal")
x = ts.randn(32, 4096, device="metal", requires_grad=True)
loss = model(x).mean()
loss.backward()
ts.evaluate(loss, x.grad, [p.grad for p in model.parameters()])
ts.synchronize("metal")
```

MLX evaluates lazily. A synchronization barrier alone does **not** submit unevaluated expressions.
Use `ts.evaluate(loss, gradients, parameters, optimizer.state)` to submit all the work being measured,
then `ts.synchronize("metal")` before stopping the timer. Evaluating only the loss excludes lazy
backward/optimizer work. The benchmark scripts do both automatically, inside each timed sample.

For long training loops, evaluate each optimizer update even when you do not print a loss; otherwise
deferred graphs can accumulate. `generate()` evaluates its outputs and cache state each token.

Metal uses native optimized attention and normalization forward paths when supported; TensorSmith
implements the backward rules itself. Device draw masks for dropout and sampling are generated
on-device. Cache updates on MLX are functional backend updates: preallocated logical capacity does
not guarantee zero temporary allocations or in-place donation at the kernel level.

## Correct benchmarking

- Warm up kernels before timing.
- Materialize all outputs, gradients, and updated optimizer/cache state; synchronize before stopping
  each latency sample's timer. CUDA synchronization covers all streams on the selected device.
- Separate model/device-transfer time from compute time.
- Compare equal dtypes, shapes, batch sizes, and gradient behavior.
- Report median and tail latency, not only the fastest run.
- Run equal shapes/dtypes and gradient semantics, with inference/training reported separately.
- Do not run competing benchmarks concurrently. Record thread/runtime/hardware settings.

See [the completed-work benchmark guide](performance.md). CUDA support uses CuPy's array kernels;
CUDA attention currently uses array matmul/softmax primitives, not TensorSmith-authored fused kernels.
Local GPU results are Metal-only until an NVIDIA runner executes the same parity tests/benchmarks.
