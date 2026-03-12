"""Common neural-network modules."""

from __future__ import annotations


import math


from collections.abc import Sequence


import numpy as np


from ..device import DeviceLike, xp_for


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


class Conv1d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        device: DeviceLike = None,
    ):
        super().__init__()
        if in_channels % groups or out_channels % groups:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        self.in_channels, self.out_channels, self.kernel_size = (
            in_channels,
            out_channels,
            kernel_size,
        )
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
        bound = 1 / math.sqrt((in_channels // groups) * kernel_size)
        self.weight = Parameter(
            _uniform((out_channels, in_channels // groups, kernel_size), bound, device)
        )
        self.bias = Parameter(_uniform((out_channels,), bound, device)) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        return F.conv1d(
            input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] = 0,
        dilation: int | Sequence[int] = 1,
        groups: int = 1,
        bias: bool = True,
        device: DeviceLike = None,
    ):
        super().__init__()
        if in_channels % groups or out_channels % groups:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        kernel = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.in_channels, self.out_channels, self.kernel_size = in_channels, out_channels, kernel
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dilation, groups
        bound = 1 / math.sqrt((in_channels // groups) * math.prod(kernel))
        self.weight = Parameter(
            _uniform((out_channels, in_channels // groups, *kernel), bound, device)
        )
        self.bias = Parameter(_uniform((out_channels,), bound, device)) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(
            input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: DeviceLike = None):
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        values = np.random.normal(0, 1, (num_embeddings, embedding_dim)).astype(np.float32)
        self.weight = Parameter(Tensor(values, device=device))

    def forward(self, indices: Tensor) -> Tensor:
        return F.embedding(indices, self.weight)


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


class GELU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input)


class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim, self.end_dim = start_dim, end_dim

    def forward(self, input: Tensor) -> Tensor:
        return input.flatten(self.start_dim, self.end_dim)


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        return F.dropout(input, self.p, self.training)


class LayerNorm(Module):
    def __init__(
        self, normalized_shape: int | Sequence[int], eps: float = 1e-5, device: DeviceLike = None
    ):
        super().__init__()
        self.normalized_shape = (
            (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        )
        self.eps = eps
        self.weight = Parameter(Tensor(np.ones(self.normalized_shape, np.float32), device=device))
        self.bias = Parameter(Tensor(np.zeros(self.normalized_shape, np.float32), device=device))

    def forward(self, input: Tensor) -> Tensor:
        if input.shape[-len(self.normalized_shape) :] != self.normalized_shape:
            raise ValueError("input's trailing shape does not match normalized_shape")
        return F.layer_norm(input, self.weight, self.bias, self.eps)


class _BatchNorm(Module):
    expected_ndim: int

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device: DeviceLike = None,
    ):
        super().__init__()
        self.num_features, self.eps, self.momentum = num_features, eps, momentum
        self.affine, self.track_running_stats = affine, track_running_stats
        self.weight = (
            Parameter(Tensor(np.ones(num_features, np.float32), device=device)) if affine else None
        )
        self.bias = (
            Parameter(Tensor(np.zeros(num_features, np.float32), device=device)) if affine else None
        )
        if track_running_stats:
            self.register_buffer(
                "running_mean", Tensor(np.zeros(num_features, np.float32), device=device)
            )
            self.register_buffer(
                "running_var", Tensor(np.ones(num_features, np.float32), device=device)
            )

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim != self.expected_ndim or input.shape[1] != self.num_features:
            raise ValueError(
                f"expected a {self.expected_ndim}D input with {self.num_features} channels"
            )
        reduction_axes = (0,) + tuple(range(2, input.ndim))
        broadcast = (1, self.num_features) + (1,) * (input.ndim - 2)
        original_input = input
        low_precision = "float16" in str(input.dtype)
        if low_precision:
            xp = xp_for(input.device)
            input = Tensor._from_op(
                input._data.astype(xp.float32),
                (input,),
                lambda g: (g.astype(original_input.dtype),),
                "batch_norm_promote",
            )
            if self.track_running_stats:
                running_dtypes = (self.running_mean.dtype, self.running_var.dtype)
        if self.training or not self.track_running_stats:
            sample_count = math.prod(input.shape[axis] for axis in reduction_axes)
            if sample_count <= 1:
                raise ValueError(
                    "batch normalization requires more than one value per channel when training"
                )
            mean = input.mean(reduction_axes, keepdims=True)
            variance = ((input - mean) ** 2).mean(reduction_axes, keepdims=True)
            if self.track_running_stats:
                self.running_mean._data = (
                    1 - self.momentum
                ) * self.running_mean._data + self.momentum * mean.detach()._data.reshape(
                    self.num_features
                )
                self.running_var._data = (
                    1 - self.momentum
                ) * self.running_var._data + self.momentum * (
                    sample_count / (sample_count - 1)
                ) * variance.detach()._data.reshape(self.num_features)
                if low_precision:
                    self.running_mean._data = self.running_mean._data.astype(running_dtypes[0])
                    self.running_var._data = self.running_var._data.astype(running_dtypes[1])
        else:
            mean = self.running_mean.reshape(broadcast)
            variance = self.running_var.reshape(broadcast)
            if low_precision:
                mean = Tensor(mean._data.astype(xp.float32), device=input.device)
                variance = Tensor(variance._data.astype(xp.float32), device=input.device)
        output = (input - mean) / (variance + self.eps).sqrt()
        if self.affine:
            output = output * self.weight.reshape(broadcast) + self.bias.reshape(broadcast)
        if low_precision:
            output = Tensor._from_op(
                output._data.astype(original_input.dtype),
                (output,),
                lambda g: (g.astype(xp.float32),),
                "batch_norm_cast",
            )
        return output


class BatchNorm1d(_BatchNorm):
    expected_ndim = 3

    def forward(self, input: Tensor) -> Tensor:
        if input.ndim == 2:
            output = super().forward(input.unsqueeze(-1))
            return output.squeeze(-1)
        return super().forward(input)


class BatchNorm2d(_BatchNorm):
    expected_ndim = 4


class MaxPool1d(Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 0):
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

    def forward(self, input):
        return F.max_pool1d(input, self.kernel_size, self.stride, self.padding)


class AvgPool1d(MaxPool1d):
    def forward(self, input):
        return F.avg_pool1d(input, self.kernel_size, self.stride, self.padding)


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding

    def forward(self, input):
        return F.max_pool2d(input, self.kernel_size, self.stride, self.padding)


class AvgPool2d(MaxPool2d):
    def forward(self, input):
        return F.avg_pool2d(input, self.kernel_size, self.stride, self.padding)


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
