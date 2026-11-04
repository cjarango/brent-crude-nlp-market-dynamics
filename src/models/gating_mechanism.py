"""
gating_mechanism
================

Dynamic gating network for the MS-WT-LSTM.

Takes the concatenated hidden states of the J+1 LSTMs and produces a
probability vector over the J+1 wavelet components, using a funnel-shaped
MLP followed by a Softmax.
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["GatingMechanism"]


class GatingMechanism(nn.Module):
    """Funnel-shaped gating network with a Softmax output.

    Args:
        input_dim: Input dimension, typically (J + 1) * hidden_size.
        output_dim: Number of outputs, typically J + 1.
        initial_dim: Width of the first hidden layer. Must be >= 16.
        num_hidden_layers: Number of hidden layers. Must be >= 2.
        dropout: Dropout rate after each hidden ReLU. Must be in [0, 0.5].
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        initial_dim: int = 64,
        num_hidden_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if num_hidden_layers < 2:
            raise ValueError(
                f"num_hidden_layers must be >= 2, got {num_hidden_layers}."
            )
        if initial_dim < 16:
            raise ValueError(f"initial_dim must be >= 16, got {initial_dim}.")
        if not (0.0 <= dropout <= 0.5):
            raise ValueError(f"dropout must be in [0.0, 0.5], got {dropout}.")

        layers: list[nn.Module] = []
        in_features = input_dim
        current_dim = initial_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_features, current_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_features = current_dim
            current_dim = max(16, current_dim // 2)

        layers.append(nn.Linear(in_features, output_dim))
        layers.append(nn.Softmax(dim=-1))

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)