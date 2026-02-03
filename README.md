# Custom ML Library

A small machine-learning framework with explicit reverse-mode derivatives.
The Python package is called `tensorsmith`.

Reduce broadcast gradients to the input shape and add reshape, transpose and dimension operations. Cover reduction gradients and dtype-preserving scalar arithmetic.

## Run

```bash
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests -v
```

Runtime dependencies: NumPy. Optional accelerator and reference dependencies
are introduced in later milestones. No accelerator performance is inferred
from these CPU checks.
