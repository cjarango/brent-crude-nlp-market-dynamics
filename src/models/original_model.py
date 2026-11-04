"""
original_model
==============

MS-WT-LSTM: Multi-Scale Wavelet Transform LSTM with dynamic gating.

DWT mode: 'symmetric' (PyWavelets default), matching the EDA pipeline.
`look_back` must be even so the symmetric reconstruction is exact.
"""
from __future__ import annotations

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gating_mechanism import GatingMechanism

__all__ = [
    "MSWTNetwork",
    "min_window_length",
    "validate_even_look_back",
]


_DWT_MODE = "symmetric"


def min_window_length(wavelet: str, J: int) -> int:
    filter_len = pywt.Wavelet(wavelet).dec_len
    return (filter_len - 1) * (2 ** J)


def validate_even_look_back(look_back: int) -> None:
    if look_back % 2 != 0:
        raise ValueError(
            f"look_back must be even to avoid truncation with the symmetric "
            f"DWT reconstruction. Got {look_back}."
        )


def _dwt_window(window_np: np.ndarray, wavelet: str, J: int) -> np.ndarray:
    """Decompose and reconstruct a single window (mirrors EDA logic)."""
    coeffs = pywt.wavedec(window_np, wavelet=wavelet, level=J, mode=_DWT_MODE)
    n = len(window_np)
    bands: list[np.ndarray] = []
    for k in range(len(coeffs)):
        isolated = [np.zeros_like(c) for c in coeffs]
        isolated[k] = coeffs[k]
        rec = pywt.waverec(isolated, wavelet=wavelet, mode=_DWT_MODE)
        bands.append(rec[:n])
    return np.stack(bands, axis=-1)


class MSWTNetwork(nn.Module):
    """MS-WT-LSTM.

    If `precomputed=True`, forward expects (batch, look_back, J+1) bands.
    If `precomputed=False` (default), forward expects (batch, look_back, 1)
    raw returns and computes per-window DWT internally (legacy path).
    """

    def __init__(
        self,
        J: int,
        look_back: int,
        wavelet: str,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        gating_dim: int,
        gating_hidden_layers: int,
        precomputed: bool = False,
    ) -> None:
        super().__init__()

        validate_even_look_back(look_back)

        required = min_window_length(wavelet, J)
        if look_back < required:
            raise ValueError(
                f"look_back={look_back} is insufficient for wavelet='{wavelet}' "
                f"with J={J}: at least {required} samples are required."
            )

        self.J = J
        self.wavelet = wavelet
        self.look_back = look_back
        self.precomputed = precomputed

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstms = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=1,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=lstm_dropout,
                )
                for _ in range(J + 1)
            ]
        )
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, 1) for _ in range(J + 1)]
        )
        self.gating = GatingMechanism(
            input_dim=(J + 1) * hidden_size,
            output_dim=J + 1,
            initial_dim=gating_dim,
            num_hidden_layers=gating_hidden_layers,
            dropout=dropout,
        )
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.precomputed:
            bands = x   # (batch, look_back, J+1)
        else:
            batch = x.shape[0]
            x_np = x.detach().cpu().numpy()
            bands_np = np.stack(
                [_dwt_window(x_np[b, :, 0], self.wavelet, self.J)
                 for b in range(batch)]
            )
            bands = torch.tensor(bands_np, dtype=torch.float32, device=x.device)

        h_states, v_scalars = [], []
        for k, lstm in enumerate(self.lstms):
            _, (h_n, _) = lstm(bands[:, :, k : k + 1])
            h_last = h_n[-1]
            h_states.append(h_last)
            v_scalars.append(self.heads[k](h_last))

        alpha = self.gating(torch.cat(h_states, dim=-1))
        v = torch.cat(v_scalars, dim=-1)
        return F.softplus(torch.sum(alpha * v, dim=-1, keepdim=True)) + self.eps