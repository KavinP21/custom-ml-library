# Custom ML Library

A small machine-learning framework with explicit reverse-mode derivatives.
The Python package is called `tensorsmith`.

## Implemented

- Tensor arithmetic, broadcasting and explicit VJPs.
- Neural-network layers, optimizers and decoder transformers.

## Run on CPU

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/train_mlp.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

For CUDA 12 use `.[cuda12]`; for CUDA 13 use `.[cuda13]`. Install one
matching CuPy wheel in an environment with a compatible NVIDIA driver/GPU.
For Apple silicon use `.[metal]`.
