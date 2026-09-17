# Contributing

Keep TensorSmith's core small enough to explain. New differentiable operations should include:

- a shape/dtype/device contract;
- an explicit VJP that returns one entry per recorded parent;
- central finite-difference checks in float64, including broadcasting or repeated inputs;
- a backend test on every available accelerator;
- documentation of any deliberate subgradient choice.

Run the full suite with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Performance changes should include warm-up, explicit evaluation of lazy outputs/gradients/updated state,
device synchronization, representative tensor sizes, and separate inference and forward/backward
measurements. Capture raw JSON samples and equal-shape baseline checks; do not benchmark concurrently.
Never hide a host fallback: correctness fallbacks must remain
obvious in the backend layer and should have a benchmark before optimization.
