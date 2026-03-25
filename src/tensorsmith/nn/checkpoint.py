"""Activation recomputation owned by TensorSmith, not backend autodiff."""

from __future__ import annotations

import numpy as np

from ..device import _random_tape, evaluate
from ..tensor import Tensor, enable_grad, grad, is_grad_enabled, no_grad
from .module import Module


def checkpoint(function, *inputs: Tensor, parameters=(), **kwargs) -> Tensor:
    """Trade saved activations for one recomputation during backward.

    Modules discover parameters; closures must provide parameters=.... Callable
    must be pure: no cache mutation, running-stat updates, device transfers, or
    parameter/mode changes between forward/backward. TensorSmith dropout/device
    uniform draws are replayed exactly; GPU draw arrays stay saved (not a
    constant-memory stochastic RNG engine). NumPy RNG state is preserved too;
    external-library randomness is not. Returns one Tensor, not a container.
    """
    if any(not isinstance(x, Tensor) for x in inputs) or not inputs:
        raise TypeError("checkpoint requires at least one Tensor input")
    if not is_grad_enabled():
        return function(*inputs, **kwargs)
    if isinstance(function, Module):
        if any(m.training and type(m).__name__.startswith("BatchNorm") for m in function.modules()):
            raise ValueError(
                "checkpointing training BatchNorm would update running statistics twice"
            )
        parameters = tuple(function.parameters()) + tuple(parameters)
    else:
        parameters = tuple(parameters)
    if any(not isinstance(p, Tensor) for p in parameters):
        raise TypeError("checkpoint parameters must be Tensors")
    parents = tuple({id(x): x for x in (*inputs, *parameters)}.values())
    if (
        not is_grad_enabled()
        or not any(x.requires_grad for x in parents)
        or _random_tape.get() is not None
    ):
        return function(*inputs, **kwargs)
    if any(x.device != inputs[0].device for x in parents):
        raise ValueError("checkpoint inputs/parameters must share a device")
    host_state = np.random.get_state()
    parameter_arrays = [(p, p._data) for p in parameters]
    modes = [(m, m.training) for m in function.modules()] if isinstance(function, Module) else []
    recording = {"mode": "record", "draws": [], "position": 0}
    token = _random_tape.set(recording)
    try:
        with no_grad():
            output = function(*inputs, **kwargs)
    finally:
        _random_tape.reset(token)
    if (
        not isinstance(output, Tensor)
        or output.device != inputs[0].device
        or "float" not in str(output.dtype)
    ):
        raise TypeError("checkpoint callable must return one floating Tensor on the input device")
    # Materializing also releases dependencies in the lazy backend graph.
    evaluate(output, [draw[2] for draw in recording["draws"]])

    def backward(upstream):
        if any(p._data is not raw for p, raw in parameter_arrays) or any(
            m.training != mode for m, mode in modes
        ):
            raise RuntimeError("checkpoint parameters or module modes changed before backward")
        clones = {
            id(x): Tensor(x._data, device=x.device, requires_grad=x.requires_grad) for x in inputs
        }
        replay_inputs = tuple(clones[id(x)] for x in inputs)
        differentiable = tuple(
            {id(x): x for x in (*replay_inputs, *parameters) if x.requires_grad}.values()
        )
        current_host_state = np.random.get_state()
        np.random.set_state(host_state)
        replay = {"mode": "replay", "draws": recording["draws"], "position": 0}
        rng_token = _random_tape.set(replay)
        try:
            with enable_grad():
                result = function(*replay_inputs, **kwargs)
                if result.shape != output.shape:
                    raise RuntimeError("checkpoint recomputation changed the output shape")
                derivatives = (
                    grad(result, differentiable, upstream, allow_unused=True)
                    if result.requires_grad
                    else (None,) * len(differentiable)
                )
            if replay["position"] != len(replay["draws"]):
                raise RuntimeError("checkpoint recomputation skipped random draws")
        finally:
            _random_tape.reset(rng_token)
            np.random.set_state(current_host_state)
        gradients = {id(x): dx for x, dx in zip(differentiable, derivatives)}
        returned = []
        for parent in parents:
            terms = []
            if id(parent) in clones and id(clones[id(parent)]) in gradients:
                terms.append(gradients[id(clones[id(parent)])])
            if id(parent) in gradients:
                terms.append(gradients[id(parent)])
            terms = [x._data for x in terms if x is not None]
            returned.append(sum(terms[1:], terms[0]) if terms else None)
        return tuple(returned)

    return Tensor._from_op(output._data, parents, backward, "checkpoint")
