# Custom ML Library

A small machine-learning framework with explicit reverse-mode derivatives.
The Python package is called `tensorsmith`.

Support momentum and Nesterov updates, tensor datasets and minibatch iteration. Include a linear-regression convergence check.

## Run

```bash
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests -v
```

Runtime dependencies: NumPy. Optional accelerator and reference dependencies
are introduced in later milestones. No accelerator performance is inferred
from these CPU checks.
