"""Common neural-network modules."""

from __future__ import annotations


import math


import numpy as np


from ..device import DeviceLike


from ..tensor import Tensor


from . import functional as F


from .module import Module, Parameter


def _uniform(shape, bound: float, device: DeviceLike) -> Tensor:
    return Tensor(np.random.uniform(-bound, bound, shape).astype(np.float32), device=device)


class Linear(Module):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device: DeviceLike = None
    ):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        bound = 1 / math.sqrt(in_features)
        self.weight = Parameter(_uniform((out_features, in_features), bound, device))
        self.bias = Parameter(_uniform((out_features,), bound, device)) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def __repr__(self):
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"


class ReLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.relu()


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input, self.negative_slope)


class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.sigmoid()


class Tanh(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.tanh()


class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim, self.end_dim = start_dim, end_dim

    def forward(self, input: Tensor) -> Tensor:
        return input.flatten(self.start_dim, self.end_dim)


class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()
        self.layers = list(modules)

    def forward(self, input: Tensor) -> Tensor:
        for layer in self.layers:
            input = layer(input)
        return input

    def __getitem__(self, index):
        return self.layers[index]

    def __len__(self):
        return len(self.layers)

    def __repr__(self):
        body = "\n".join(f"  ({i}): {layer!r}" for i, layer in enumerate(self.layers))
        return f"Sequential(\n{body}\n)"
