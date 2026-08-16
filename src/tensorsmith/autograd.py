"""User-defined first-order backward functions."""

from __future__ import annotations

from .tensor import Tensor, no_grad


class FunctionContext:
    """Per-call state for a custom operation.

    Saved tensors are detached copies: backward sees the original forward
    values even if the caller later mutates an input. Copying has a memory cost;
    this engine does not use PyTorch's version-counter/view machinery.
    """

    def __init__(self, inputs):
        self.needs_input_grad = tuple(value.requires_grad for value in inputs)
        self._saved_tensors = ()

    def save_for_backward(self, *values: Tensor):
        if any(not isinstance(value, Tensor) for value in values):
            raise TypeError("save_for_backward expects Tensor objects")
        with no_grad():
            self._saved_tensors = tuple(value.detach().clone() for value in values)

    @property
    def saved_tensors(self):
        return self._saved_tensors


class Function:
    """Subclass and implement forward(ctx, ...) and backward(ctx, gradient).

    Positional inputs must be tensors on one device. Non-tensor settings may
    be keyword arguments. Forward returns one tensor; backward returns one
    tensor or None per positional input. Derivatives remain first-order.
    """

    @staticmethod
    def forward(ctx, *inputs, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, gradient):
        raise NotImplementedError

    @classmethod
    def apply(cls, *inputs: Tensor, **kwargs):
        if not inputs or any(not isinstance(value, Tensor) for value in inputs):
            raise TypeError("custom Function positional inputs must be tensors")
        device = inputs[0].device
        if any(value.device != device for value in inputs):
            raise ValueError("custom Function inputs must share a device")
        ctx = FunctionContext(inputs)
        with no_grad():
            output = cls.forward(ctx, *inputs, **kwargs)
        if not isinstance(output, Tensor) or output.device != device:
            raise TypeError("custom Function forward must return one tensor on the input device")
        if "float" not in str(output.dtype):
            raise TypeError("custom Function output must have a floating dtype")

        def backward(raw_gradient):
            with no_grad():
                values = cls.backward(ctx, Tensor(raw_gradient, device=device))
            if len(inputs) == 1 and (values is None or isinstance(values, Tensor)):
                values = (values,)
            if not isinstance(values, (tuple, list)) or len(values) != len(inputs):
                raise RuntimeError("custom backward must return one gradient per tensor input")
            result = []
            for value, parent in zip(values, inputs):
                if value is None:
                    result.append(None)
                elif not isinstance(value, Tensor):
                    raise TypeError("custom backward gradients must be tensors or None")
                elif value.device != parent.device or value.shape != parent.shape:
                    raise ValueError("custom backward gradient shape/device mismatch")
                else:
                    result.append(value._data.astype(parent.dtype))
            return tuple(result)

        return Tensor._from_op(output._data, tuple(inputs), backward, cls.__name__)
