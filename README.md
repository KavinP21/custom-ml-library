# Custom ML Library

A small machine-learning framework with explicit reverse-mode derivatives.
The Python package is called `tensorsmith`.

Save portable state in pickle-free NPZ files and add gradient clipping utilities. Include MLP and CNN training examples.

## Run

```bash
python -m pip install -e .
PYTHONPATH=src python -m unittest discover -s tests -v
```

Runtime dependencies: NumPy. Optional accelerator and reference dependencies
are introduced in later milestones. No accelerator performance is inferred
from these CPU checks.
