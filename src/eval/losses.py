"""
losses
======

Implementation of loss functions for training Deep Learning models.
"""

__version__ = "1.1.0"

import torch
import torch.nn as nn

__all__ = ["QLIKELoss"]


class QLIKELoss(nn.Module):
    """PyTorch implementation of the Smoothed Quasi-Likelihood (QLIKE) loss."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        y_pred_eps = y_pred + self.eps
        y_true_eps = y_true + self.eps
        
        ratio = y_true_eps / y_pred_eps
        loss = ratio - torch.log(ratio) - 1
        
        return torch.mean(loss)