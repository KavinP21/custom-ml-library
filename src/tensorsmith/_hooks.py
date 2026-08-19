"""Removable hook handles without keeping the owning object alive."""

from itertools import count
from weakref import ref

_ids = count()


class RemovableHandle:
    def __init__(self, owner, attribute):
        self.id = next(_ids)
        self._owner = ref(owner)
        self._attribute = attribute

    def remove(self):
        owner = self._owner()
        if owner is not None:
            getattr(owner, self._attribute).pop(self.id, None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


def add_hook(owner, attribute, value):
    handle = RemovableHandle(owner, attribute)
    getattr(owner, attribute)[handle.id] = value
    return handle
