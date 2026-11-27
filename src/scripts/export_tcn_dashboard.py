from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

# Tres niveles hacia arriba desde src/scripts/export_tcn_dashboard.py -> raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tcn_wrapper import TCNModel

warnings.filterwarnings("ignore")


def _fitted_scaler(scaler: StandardScaler | None, y_train: np.ndarray) -> StandardScaler:
  if isinstance(scaler, StandardScaler):
    try:
      check_is_fitted(scaler)
      return scaler
    except NotFittedError:
      pass
  s = StandardScaler()
  s.fit(y_train.reshape(-1, 1))
  return s


def main() -> None:
  models_dir = PROJECT_ROOT / "models"
  data_dir = PROJECT_ROOT / "data" / "processed"
  src_ckpt = models_dir / "best_tcn_model.pt"

  ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
  if isinstance(ckpt, TCNModel):
    model = ckpt
    state_dict = model.net.state_dict()
    params = getattr(model, "params", None)
  elif isinstance(ckpt, dict):
    params = ckpt.get("params") or ckpt.get("hyperparams") or ckpt.get("config")
    if params is None:
      raise KeyError("Falta 'params' en best_tcn_model.pt para instanciar TCNModel.")
    model = TCNModel(**params)
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict")
    if state_dict is None:
      raise KeyError("Falta 'model_state_dict' o 'state_dict' en best_tcn_model.pt.")
    model.net.load_state_dict(state_dict)
  else:
    raise TypeError(f"Checkpoint no soportado: {type(ckpt)}")

  df_train = pd.read_parquet(data_dir / "brent_train.parquet")
  y_train = df_train["log_returns"].to_numpy(dtype=np.float64)
  rs_train = df_train["rs_daily"].to_numpy(dtype=np.float64)

  raw_scaler = ckpt.get("scaler") if isinstance(ckpt, dict) else getattr(model, "scaler", None)
  scaler = _fitted_scaler(raw_scaler, y_train)

  raw_tscale = (
      ckpt.get("target_scale") if isinstance(ckpt, dict) else getattr(model, "target_scale", None)
  )
  target_scale = float(raw_tscale) if raw_tscale is not None else float(1.0 / np.mean(rs_train))

  look_back = int(
      getattr(model, "context_length", getattr(model, "look_back", params["look_back"]))
  )

  model.net.eval()
  y_scaled = scaler.transform(y_train.reshape(-1, 1)).flatten()
  windows = np.lib.stride_tricks.sliding_window_view(y_scaled, look_back)[:-1]
  with torch.inference_mode():
    preds_tr = model.net(torch.tensor(windows[:, :, None], dtype=torch.float32)).squeeze(-1).numpy()

  sigma_tr = np.sqrt(np.maximum(preds_tr / target_scale, 1e-12))
  z_train = y_train[look_back:] / sigma_tr
  z_train = z_train[np.isfinite(z_train)]

  bundle = {
      "params": params,
      "state_dict": state_dict,
      "look_back": look_back,
      "scaler_mean": float(scaler.mean_[0]),
      "scaler_scale": float(scaler.scale_[0]),
      "target_scale": target_scale,
      "fhs_quantiles": {
          "0.05": [float(np.quantile(z_train, 0.025)), float(np.quantile(z_train, 0.975))],
          "0.10": [float(np.quantile(z_train, 0.050)), float(np.quantile(z_train, 0.950))],
      },
  }

  out_pt = models_dir / "tcn_dashboard.pt"
  torch.save(bundle, out_pt)
  (models_dir / "tcn_dashboard_meta.json").write_text(
      json.dumps(
          {k: v for k, v in bundle.items() if k not in {"params", "state_dict"}},
          indent=2,
      )
  )
  print(f"Listo: {out_pt}")


if __name__ == "__main__":
  main()