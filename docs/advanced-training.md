# Training and extension APIs

These APIs build on the tensor engine's explicit first-order VJPs. They are
independent of PyTorch at runtime. PyTorch is an optional correctness reference.

## Parameter groups and unfreezing

Different layers can use different learning rates, decay or momentum settings:

```python
backbone = ts.nn.Sequential(ts.nn.Linear(8, 32), ts.nn.GELU())
head = ts.nn.Linear(32, 4)
model = ts.nn.Sequential(backbone, head)
optimizer = ts.optim.AdamW([
    {"params": list(backbone.parameters()), "lr": 1e-4, "weight_decay": 0},
    {"params": list(head.parameters()), "lr": 3e-4},
], weight_decay=0.01)
```

Pass iterables of `Parameter` objects. A parameter may appear only once in the
optimizer. `add_param_group()` inherits constructor defaults and validates the
new options before changing optimizer state. Schedulers preserve their existing
progress and adopt a new group's supplied rate as its base rate. Save model,
optimizer and scheduler states together for continuation.

## Automatic mixed precision

```python
optimizer = ts.optim.AdamW(model.parameters(), lr=1e-3)
scaler = ts.amp.GradScaler(init_scale=128)

optimizer.zero_grad()
with ts.amp.autocast("cuda", dtype="float16"):
    logits = model(inputs)
    loss = ts.nn.CrossEntropyLoss()(logits, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
ts.evaluate(loss, list(model.parameters()), optimizer.state)
```

Autocast selects FP16 for matmul, linear and convolution inputs. Parameters stay
FP32; their cast nodes return FP32 gradients to the original parameters. Float64
operations are preserved. Nested `enabled=False` contexts disable the policy,
and thread/task contexts are isolated. CPU and Metal use the same explicit policy.
BF16 and a comprehensive operator-policy table are not implemented. FP16 on CPU
is useful for correctness checks and is not a CPU speed guarantee.

Checkpoint recomputation restores the forward autocast and RNG policy. A
parameter gradient hook is applied once to the final accumulated contribution.

## Custom backward functions and hooks

```python
class Square(ts.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, gradient):
        return 2 * ctx.saved_tensors[0] * gradient

y = Square.apply(x)
handle = x.register_hook(lambda gradient: gradient * 0.5)
loss = y.sum()
loss.backward()
handle.remove()
```

Forward returns one tensor. Backward returns one tensor or `None` per positional
tensor input; keyword arguments carry non-tensor settings. Saved tensors are
detached copies, which trades memory for stable forward values without version
counters. This does not enable higher-order differentiation.

Module `register_forward_pre_hook()` and `register_forward_hook()` can observe or
replace arguments and results. `with_kwargs=True` exposes keyword arguments.
Handles are removable/context-managed; hooks are not model parameters or checkpoint
state. Gradient replacements must preserve shape, dtype and device.

## Data-parallel training

Construct a `TCPProcessGroup` with the same endpoint and world size on each rank,
then wrap the model:

```python
ddp = ts.distributed.DistributedDataParallel(model, group)
optimizer = ts.optim.SGD(ddp.parameters(), lr=0.01)

optimizer.zero_grad()
loss = criterion(ddp(local_inputs), local_targets)
loss.backward()
ddp.sync_gradients()
optimizer.step()
```

Construction broadcasts parameters/buffers from rank zero and checks model and
wrapper settings across ranks. `sync_gradients()` averages gradients in
dtype-homogeneous buckets. Use equal local batch sizes and local mean losses;
unequal batches need explicit loss weighting. `DistributedSampler` supplies
equal-length rank partitions, padding or dropping an uneven tail. Call
`sampler.set_epoch(epoch)` when shuffling.

With AMP, synchronize scaled gradients **before** `GradScaler.step()` or
`unscale_()` so overflow propagates across ranks. Scaler settings must match.
Set `find_unused_parameters=True` for locally unused parameters; globally unused
parameters keep `grad=None`. Changing model structure requires a new wrapper.

The transport has deadlines, packet-size limits and pickle-free numeric frames.
It uses a rank-zero coordinator and host staging, including for CUDA/Metal
gradients. It is a small-job correctness baseline, not NCCL or an elastic/sharded
trainer. Peers must be trusted on a private interface; the protocol has no
authentication or encryption. BatchNorm statistics remain local, as with ordinary
DDP; this is not synchronized BatchNorm.

Run `python examples/train_distributed.py --steps 40` for a two-process CPU demo.

## CUDA inference graphs and user kernels

```python
model = model.to("cuda").eval()
graph = ts.cuda.CUDAGraph().capture(model, example_input)
prediction = graph.replay(next_input)
```

Shapes, dtypes, devices, model modes and parameter/buffer storage must stay fixed.
Replay copies new input values into captured storage; outputs are reused and
should be cloned if retained. Python forward hooks, training capture and
host-dependent control flow are unsupported. Optimizer steps that replace weight
arrays require recapture. Replay establishes stream dependencies for its returned
output.

`ts.cuda.RawKernel(code, entry)` compiles CUDA C++ through CuPy/NVRTC. Pass contiguous
CUDA tensors and explicitly typed scalar arguments such as `np.int32(n)`. Supply
the derivative through `autograd.Function`; compilation does not infer a backward
rule. This is a user-kernel interface, not an optimizing graph compiler.

The tests in `tests/test_cuda_features.py` compare graph replay with eager output
and check a user kernel plus its backward rule on an actual NVIDIA device. They
are explicitly skipped without that device. CUDA 13 package configuration and
CPU guard tests do not establish GPU correctness or throughput. See
[GPU setup](gpu.md) for the hardware validation path.
