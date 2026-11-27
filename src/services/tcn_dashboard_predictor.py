from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tcn_wrapper import TCNModel


class TCNDashboardPredictor:
  def __init__(self, bundle_path: Path | None = None) -> None:
    path = bundle_path or (PROJECT_ROOT / "models" / "tcn_dashboard.pt")
    self.device = torch.device("cpu")
    ckpt = torch.load(path, map_location=self.device, weights_only=False)

    self.model = TCNModel(**ckpt["params"])
    self.model.net.load_state_dict(ckpt["state_dict"])
    self.model.net.to(self.device)
    self.model.net.eval()

    self.look_back: int = int(ckpt["look_back"])
    self.mean: float = float(ckpt["scaler_mean"])
    self.scale: float = float(ckpt["scaler_scale"])
    self.target_scale: float = float(ckpt["target_scale"])
    self.quantiles: dict[str, list[float]] = ckpt["fhs_quantiles"]

  def predict_next(
      self,
      recent_returns: np.ndarray,
      last_rs_var: float,
      last_close_price: float | None = None,
      alpha: float = 0.05,
  ) -> dict:
    arr = np.asarray(recent_returns, dtype=np.float64)
    if len(arr) < self.look_back:
      raise ValueError(f"Se requieren al menos {self.look_back} retornos; recibidos {len(arr)}.")

    window_scaled = ((arr[-self.look_back :] - self.mean) / self.scale).reshape(
        1, self.look_back, 1
    )
    x_tensor = torch.tensor(window_scaled, dtype=torch.float32, device=self.device)

    with torch.inference_mode():
      raw_pred = float(self.model.net(x_tensor).squeeze().cpu().numpy())

    pred_var = max(raw_pred / self.target_scale, 1e-12)
    pred_vol = float(np.sqrt(pred_var))

    last_var = float(max(last_rs_var, 1e-12))
    last_vol = float(np.sqrt(last_var))

    diff_vol = pred_vol - last_vol
    pct_vol = (diff_vol / last_vol) * 100.0 if last_vol > 0 else 0.0
    direction = "SUBE" if diff_vol > 0 else ("BAJA" if diff_vol < 0 else "SE MANTIENE")

    key = f"{alpha:.2f}"
    if key in self.quantiles:
      q_low, q_high = self.quantiles[key]
    else:
      q_low, q_high = self.quantiles["0.05"]

    r_low = float(q_low * pred_vol)
    r_high = float(q_high * pred_vol)

    out = {
        "direction": direction,
        "last_vol": last_vol,
        "pred_vol": pred_vol,
        "diff_vol": diff_vol,
        "pct_change": pct_vol,
        "last_var": last_var,
        "pred_var": pred_var,
        "return_band": [r_low, r_high],
    }

    if last_close_price is not None:
      p0 = float(last_close_price)
      out["price_band"] = [float(p0 * np.exp(r_low)), float(p0 * np.exp(r_high))]

    return out