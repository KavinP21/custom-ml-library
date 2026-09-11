"""CUDA stream capture and user kernels backed by CuPy/NVRTC.

CuPy is imported only when CUDA execution is requested. These APIs do not
provide an optimizing graph compiler or replace the backend's compiler.
"""

from __future__ import annotations

import numpy as np

from .device import device, xp_for
from .nn.module import Module
from .tensor import Tensor, no_grad


class CUDAGraph:
    """Capture an eval-mode module with fixed CUDA tensor inputs.

    Replay reuses input/output storage and executes GPU work, not Python hooks.
    Shapes, dtypes, module modes and parameter/buffer storage must remain fixed.
    Outputs are overwritten by subsequent replays; clone retained predictions.
    Training capture and host-dependent control flow are not supported.
    """

    def __init__(self):
        self._graph = None

    @staticmethod
    def _signature(model):
        parameters = [
            (name, id(p), id(p._data), p.shape, str(p.dtype))
            for name, p in model.named_parameters()
        ]
        buffers = [
            (prefix, name, id(value), id(value._data), value.shape, str(value.dtype))
            for prefix, child in model.named_modules()
            for name, value in child._buffers.items()
        ]
        modes = [(id(child), child.training) for child in model.modules()]
        return parameters, buffers, modes

    @staticmethod
    def _check_model(model):
        if any(child.training for child in model.modules()):
            raise ValueError("CUDA graph inference requires model.eval()")
        if any(child._forward_hooks or child._forward_pre_hooks for child in model.modules()):
            raise ValueError("Python forward hooks are not executed by CUDA graph replay")

    def capture(self, model, *inputs, warmup=3):
        if (
            not isinstance(model, Module)
            or not inputs
            or any(not isinstance(value, Tensor) for value in inputs)
        ):
            raise TypeError("capture requires a Module and tensor inputs")
        dev = inputs[0].device
        if dev.type != "cuda" or any(value.device != dev for value in inputs):
            raise ValueError("CUDA graph inputs must be CUDA tensors on one device")
        if not isinstance(warmup, int) or warmup < 1:
            raise ValueError("warmup must be a positive integer")
        self._check_model(model)
        values = list(model.parameters()) + [
            value for _, child in model.named_modules() for value in child._buffers.values()
        ]
        if any(value.device != dev for value in values):
            raise ValueError("model parameters and buffers must share the input CUDA device")
        xp = xp_for(dev)
        with xp.cuda.Device(dev.index or 0), no_grad():
            xp.cuda.runtime.deviceSynchronize()
            stream = xp.cuda.Stream(non_blocking=True)
            with stream:
                static_inputs = tuple(value.detach().clone() for value in inputs)
                for _ in range(warmup):
                    result = model(*static_inputs)
                    del result
            stream.synchronize()
            signature = self._signature(model)
            with stream:
                stream.begin_capture()
                try:
                    output = model(*static_inputs)
                except BaseException:
                    # End an invalidated capture before propagating the original error.
                    try:
                        stream.end_capture()
                    except RuntimeError:
                        pass
                    raise
                graph = stream.end_capture()
            if not isinstance(output, Tensor) or output.device != dev:
                raise TypeError("captured module must return one CUDA tensor")
            self._model, self._device = model, dev
            self._signature_at_capture = signature
            self._inputs, self.output = static_inputs, output
            self._stream, self._graph = stream, graph
        return self

    def replay(self, *inputs):
        if self._graph is None:
            raise RuntimeError("capture a CUDA graph before replay")
        self._check_model(self._model)
        if self._signature(self._model) != self._signature_at_capture:
            raise RuntimeError("captured model storage or modes changed; recapture the graph")
        if inputs and (
            len(inputs) != len(self._inputs)
            or any(
                not isinstance(value, Tensor)
                or value.shape != saved.shape
                or value.dtype != saved.dtype
                or value.device != saved.device
                for value, saved in zip(inputs, self._inputs)
            )
        ):
            raise ValueError("replay inputs must match captured shapes, dtypes and device")
        xp = xp_for(self._device)
        with xp.cuda.Device(self._device.index or 0):
            caller = xp.cuda.get_current_stream()
            self._stream.wait_event(caller.record())
            with self._stream:
                for incoming, saved in zip(inputs, self._inputs):
                    xp.copyto(saved._data, incoming._data)
                self._graph.launch(stream=self._stream)
            caller.wait_event(self._stream.record())
        return self.output

    __call__ = replay


class RawKernel:
    """Compile a CUDA C++ kernel through CuPy's NVRTC interface.

    Tensor arguments must be contiguous CUDA arrays on the chosen device.
    Scalar argument types must match the CUDA signature: use NumPy scalar types
    for explicit widths. Backward rules are supplied separately with autograd.Function.
    """

    def __init__(self, code, name, *, device_name="cuda", options=("--std=c++17",)):
        self.device = device(device_name)
        if self.device.type != "cuda":
            raise ValueError("RawKernel requires a CUDA device")
        if not isinstance(code, str) or not isinstance(name, str) or not code or not name:
            raise ValueError("kernel source and entry name must be nonempty strings")
        self.code, self.name, self.options = code, name, tuple(options)
        self._kernel = None

    def _get_kernel(self):
        xp = xp_for(self.device)
        if self._kernel is None:
            with xp.cuda.Device(self.device.index or 0):
                self._kernel = xp.RawKernel(self.code, self.name, options=self.options)
        return xp, self._kernel

    def compile(self, *, log_stream=None):
        xp, kernel = self._get_kernel()
        with xp.cuda.Device(self.device.index or 0):
            kernel.compile(log_stream=log_stream)
        return self

    def __call__(self, grid, block, args, *, shared_mem=0):
        if any(
            not isinstance(dims, tuple)
            or not 1 <= len(dims) <= 3
            or any(not isinstance(n, (int, np.integer)) or n <= 0 for n in dims)
            for dims in (grid, block)
        ):
            raise ValueError("grid and block must contain one to three positive integer dimensions")
        if not isinstance(shared_mem, int) or shared_mem < 0:
            raise ValueError("shared_mem must be non-negative")
        xp, kernel = self._get_kernel()
        prepared = []
        for value in args:
            if isinstance(value, Tensor):
                if value.device.type != "cuda" or (value.device.index or 0) != (
                    self.device.index or 0
                ):
                    raise ValueError("kernel tensor is on a different device")
                value = value._data
            if isinstance(value, xp.ndarray):
                if value.device.id != (self.device.index or 0) or not value.flags.c_contiguous:
                    raise ValueError("kernel arrays must be contiguous and on the kernel device")
            elif not isinstance(value, (int, float, complex, bool, np.number)):
                raise TypeError("kernel arguments must be CUDA tensors/arrays or numeric scalars")
            prepared.append(value)
        with xp.cuda.Device(self.device.index or 0):
            kernel(grid, block, tuple(prepared), shared_mem=shared_mem)
