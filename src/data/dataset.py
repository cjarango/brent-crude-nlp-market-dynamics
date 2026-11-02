"""
dataset
=======

PyTorch Dataset implementations for time series forecasting.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["TimeSeriesDataset"]


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset for one-step volatility forecasting.

    Given aligned ``returns`` and ``targets`` arrays of length ``N``, each item
    ``i`` returns the window ``returns[i : i + look_back]`` as input and
    ``targets[i + look_back]`` as the scalar target. The number of valid items
    is ``N - look_back``.

    Attributes:
        returns (torch.Tensor): Return series as ``float32`` tensor, shape ``(N,)``.
        targets (torch.Tensor): RS variance series as ``float32`` tensor, shape ``(N,)``.
        look_back (int): Number of past returns in each input window.
    """

    def __init__(
        self,
        returns: np.ndarray,
        targets: np.ndarray,
        look_back: int,
    ) -> None:
        """Initializes the dataset.

        Args:
            returns (np.ndarray): Log-return series of shape ``(N,)``.
            targets (np.ndarray): Rogers-Satchell variance series of shape ``(N,)``,
                aligned index-by-index with ``returns``.
            look_back (int): Number of past returns used as input context.

        Raises:
            ValueError: If ``look_back`` is not positive, if the arrays have
                different lengths, or if there are not enough samples to
                produce at least one window.
        """
        if look_back <= 0:
            raise ValueError(f"look_back must be positive, got {look_back}.")
        if len(returns) != len(targets):
            raise ValueError(
                f"returns and targets must have the same length, "
                f"got {len(returns)} and {len(targets)}."
            )
        if len(returns) <= look_back:
            raise ValueError(
                f"Need len(returns) > look_back to build at least one window, "
                f"got len(returns)={len(returns)} and look_back={look_back}."
            )

        self.returns = torch.as_tensor(returns, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.look_back = look_back

    def __len__(self) -> int:
        """Returns the number of valid sliding windows."""
        return len(self.returns) - self.look_back

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the input window and target for index ``idx``.

        Args:
            idx (int): Index in ``[0, len(self) - 1]``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``x`` of shape ``(look_back, 1)``
            and ``y`` of shape ``(1,)``.
        """
        x = self.returns[idx : idx + self.look_back].unsqueeze(-1)
        y = self.targets[idx + self.look_back].unsqueeze(0)
        return x, y