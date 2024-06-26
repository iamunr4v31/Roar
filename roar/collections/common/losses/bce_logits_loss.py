from typing import List

import torch
from torch import nn

from roar.core.classes import Serialization, Typing, typecheck
from roar.core.neural_types import (
    LabelsType,
    LogitsType,
    LossType,
    MaskType,
    NeuralType,
)

__all__ = ["BCEWithLogitsLoss"]


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, Serialization, Typing):
    """
    BCEWithLogitsLoss

    https://pytorch.org/docs/1.9.1/generated/torch.nn.BCEWithLogitsLoss.html
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return {
            "logits": NeuralType(
                ["B"] + ["ANY"] * (self._logits_dim - 1), LogitsType()
            ),
            "labels": [
                NeuralType(["B"] + ["ANY"] * (self._logits_dim - 2), LabelsType())
            ],
            "loss_mask": NeuralType(
                ["B"] + ["ANY"] * (self._logits_dim - 2), MaskType(), optional=True
            ),
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(
        self,
        logits_ndim: int = 2,
        weight: torch.Tensor = None,
        reduction: str = "mean",
        pos_weight: torch.Tensor = None,
    ):
        """
        Args:
            logits_ndim: number of dimensions (or rank) of the logits tensor
            weight: list of rescaling weight given to each class
            reduction: type of the reduction over the batch
            pos_weight: weight given to positive samples
        """
        if pos_weight is not None and not torch.is_tensor(pos_weight):
            pos_weight = torch.FloatTensor(pos_weight)

        super().__init__(weight=weight, pos_weight=pos_weight, reduction=reduction)
        self._logits_dim = logits_ndim

    @typecheck()
    def forward(self, logits: float, labels: List[int], loss_mask: torch.Tensor = None):
        """
        Args:
            logits: output of the classifier
            labels: ground truth labels
        """
        labels = torch.stack(labels)
        labels = labels.t().float()

        return super().forward(logits, labels)
