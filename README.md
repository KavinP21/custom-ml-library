# Custom ML Library

A small machine-learning framework with explicit reverse-mode derivatives.
The Python package is called `tensorsmith`.

Record operation parents and backward functions, traverse the graph in reverse order, and accumulate gradients across branches. Cover scalar outputs, gradient accumulation and graph lifetime.

## Run

```bash
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests -v
```

Runtime dependencies: NumPy. Optional accelerator and reference dependencies
are introduced in later milestones. No accelerator performance is inferred
from these CPU checks.
