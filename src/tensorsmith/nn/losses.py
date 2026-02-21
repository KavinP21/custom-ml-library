from __future__ import annotations

from . import functional as F
from .module import Module


class _Loss(Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction


class MSELoss(_Loss):
    def forward(self, input, target):
        return F.mse_loss(input, target, self.reduction)


class CrossEntropyLoss(_Loss):
    def __init__(self, reduction="mean", *, axis=1, ignore_index=-100, label_smoothing=0.0):
        super().__init__(reduction)
        self.axis = axis
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        return F.cross_entropy(
            input,
            target,
            self.reduction,
            axis=self.axis,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


class BCELoss(_Loss):
    def forward(self, input, target):
        return F.binary_cross_entropy(input, target, self.reduction)


class BCEWithLogitsLoss(_Loss):
    def forward(self, input, target):
        return F.binary_cross_entropy_with_logits(input, target, self.reduction)
