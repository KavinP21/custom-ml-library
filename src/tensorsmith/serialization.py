from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .device import asnumpy

_METADATA = "__tensorsmith_tree_v1__"


def save(state: dict, path: str | Path) -> None:
    """Atomically save nested model/optimizer/scheduler state without pickle.

    Supports dicts with scalar keys, lists/tuples, scalar values, and arrays.
    No arbitrary objects are deserialized or executed. Existing checkpoints
    remain intact if serialization fails before replacement.
    """
    arrays = {}

    def encode(value):
        if isinstance(value, dict):
            return {"type": "dict", "items": [[encode(k), encode(v)] for k, v in value.items()]}
        if isinstance(value, (list, tuple)):
            return {
                "type": "tuple" if isinstance(value, tuple) else "list",
                "items": [encode(v) for v in value],
            }
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (bool, int, float, str)):
            return {"type": "scalar", "value": value}
        if hasattr(value, "shape"):
            raw = asnumpy(getattr(value, "_data", value))
            if raw.dtype.hasobject:
                raise TypeError("object arrays are not safe checkpoint values")
            name = f"array_{len(arrays)}"
            arrays[name] = raw
            return {"type": "array", "name": name}
        raise TypeError(f"unsupported checkpoint value: {type(value).__name__}")

    tree = encode(state)
    arrays[_METADATA] = np.array(json.dumps(tree))
    destination = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(path: str | Path) -> dict:
    """Load a pickle-free tree or legacy flat NPZ model dictionary."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if _METADATA not in archive.files:
            return {key: archive[key] for key in archive.files}
        tree = json.loads(str(archive[_METADATA].item()))

        def decode(node):
            kind = node["type"]
            if kind == "scalar":
                return node["value"]
            if kind == "array":
                return archive[node["name"]].copy()
            if kind == "dict":
                return {decode(k): decode(v) for k, v in node["items"]}
            if kind in {"tuple", "list"}:
                values = [decode(item) for item in node["items"]]
                return tuple(values) if kind == "tuple" else values
            raise ValueError(f"unsupported checkpoint node: {kind}")

        return decode(tree)
