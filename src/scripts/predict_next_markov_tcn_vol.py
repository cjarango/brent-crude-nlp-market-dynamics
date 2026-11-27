from __future__ import annotations

import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tcn_wrapper import TCNModel

warnings.filterwarnings("ignore")

SEED = 42


def set_seed(seed: int = SEED) -> None:
  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def resolve_tcn_checkpoint(models_dir: Path) -> Path:
  candidates = [
      models_dir / "best_tcn_model.pt",
      models_dir / "best_tcn_model.pth",
      models_dir / "best_tcn_model",
  ]
  for path in candidates:
    if path.exists():
      return path

  for pattern in ("*best_tcn_model*", "*tcn*"):
    matches = sorted(
        models_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if matches:
      return matches[0]

  raise FileNotFoundError(f"No se encontró best_tcn_model en {models_dir}.")


def _ensure_fitted_scaler(
    scaler: StandardScaler | None,
    train_returns: np.ndarray,
) -> StandardScaler:
  if isinstance(scaler, StandardScaler):
    try:
      check_is_fitted(scaler)
      return scaler
    except NotFittedError:
      pass
  fitted = StandardScaler()
  fitted.fit(train_returns.reshape(-1, 1))
  return fitted


def load_tcn_checkpoint(
    ckpt_path: Path,
    train_returns: np.ndarray,
    train_rs: np.ndarray,
    device: torch.device,
) -> tuple[TCNModel, StandardScaler, float]:
  checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

  if isinstance(checkpoint, TCNModel):
    model = checkpoint
    model.device = device
    model.net.to(device)
  elif isinstance(checkpoint, dict):
    params = (
        checkpoint.get("params")
        or checkpoint.get("hyperparams")
        or checkpoint.get("config")
        or checkpoint.get("model_params")
    )
    if params is None:
      raise KeyError(
          "El checkpoint es un dict pero no incluye 'params'/'hyperparams'/'config' para"
          " instanciar TCNModel."
      )
    model = TCNModel(**params)
    model.device = device
    model.net.to(device)

    state_dict = (
        checkpoint.get("model_state_dict")
        or checkpoint.get("state_dict")
        or checkpoint.get("net_state_dict")
    )
    if state_dict is None:
      raise KeyError("No se encontró 'model_state_dict' ni 'state_dict' en el checkpoint.")
    model.net.load_state_dict(state_dict)
  else:
    raise TypeError(f"Tipo de checkpoint no soportado: {type(checkpoint)}")

  raw_scaler = checkpoint.get("scaler") if isinstance(checkpoint, dict) else None
  if raw_scaler is None:
    raw_scaler = getattr(model, "scaler", None)
  scaler = _ensure_fitted_scaler(raw_scaler, train_returns)

  target_scale = checkpoint.get("target_scale") if isinstance(checkpoint, dict) else None
  if target_scale is None:
    target_scale = getattr(model, "target_scale", None)
  if target_scale is None:
    rs_mean = float(np.mean(train_rs))
    target_scale = 1.0 / rs_mean if rs_mean > 0 else 1.0

  model.net.eval()
  return model, scaler, float(target_scale)


def compute_fhs_quantiles_from_train(
    model: TCNModel,
    scaler: StandardScaler,
    target_scale: float,
    y_train: np.ndarray,
    rs_train: np.ndarray,
    alpha: float = 0.05,
    mu: float = 0.0,
    batch_size: int = 256,
) -> tuple[float, float]:
  """Estima cuantiles FHS (q_low, q_high) en train sin fuga de información."""
  look_back = int(
      getattr(model, "context_length", getattr(model, "look_back", model.params["look_back"]))
  )
  y_scaled = scaler.transform(y_train.reshape(-1, 1)).flatten()

  n_samples = len(y_scaled) - look_back
  if n_samples <= 0:
    raise ValueError("Insuficientes observaciones en train para construir ventanas.")

  windows = np.lib.stride_tricks.sliding_window_view(y_scaled, look_back)[:-1]
  x_train = torch.tensor(windows[:, :, None], dtype=torch.float32, device=model.device)

  preds_list: list[np.ndarray] = []
  with torch.no_grad():
    for start in range(0, len(x_train), batch_size):
      batch_out = model.net(x_train[start : start + batch_size]).squeeze(-1).cpu().numpy()
      preds_list.append(np.atleast_1d(batch_out))

  pred_var_train = np.maximum(np.concatenate(preds_list) / target_scale, 1e-12)
  sigma_train = np.sqrt(pred_var_train)
  r_train_aligned = y_train[look_back:]

  z_train = (r_train_aligned - mu) / sigma_train
  z_train = z_train[np.isfinite(z_train)]

  q_low = float(np.quantile(z_train, alpha / 2.0))
  q_high = float(np.quantile(z_train, 1.0 - alpha / 2.0))
  return q_low, q_high


def main() -> None:
  set_seed(SEED)

  data_dir = PROJECT_ROOT / "data" / "processed"
  models_dir = PROJECT_ROOT / "models"
  ckpt_path = resolve_tcn_checkpoint(models_dir)

  df_train = pd.read_parquet(data_dir / "brent_train.parquet")
  df_test = pd.read_parquet(data_dir / "brent_test.parquet")

  df_train.index = pd.to_datetime(df_train.index)
  df_test.index = pd.to_datetime(df_test.index)

  y_train = df_train["log_returns"].to_numpy(dtype=np.float64)
  rs_train = df_train["rs_daily"].to_numpy(dtype=np.float64)

  y_test = df_test["log_returns"].to_numpy(dtype=np.float64)
  rs_test = df_test["rs_daily"].to_numpy(dtype=np.float64)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model, scaler, target_scale = load_tcn_checkpoint(
      ckpt_path=ckpt_path,
      train_returns=y_train,
      train_rs=rs_train,
      device=device,
  )

  look_back = int(
      getattr(model, "context_length", getattr(model, "look_back", model.params["look_back"]))
  )
  if len(y_test) < look_back:
    raise ValueError(f"Se requieren al menos {look_back} datos en test, hay {len(y_test)}.")

  last_window_raw = y_test[-look_back:].reshape(-1, 1)
  last_window_scaled = scaler.transform(last_window_raw).reshape(1, look_back, -1)
  x_next = torch.tensor(last_window_scaled, dtype=torch.float32, device=device)

  with torch.no_grad():
    raw_pred_var = float(model.net(x_next).squeeze().cpu().numpy())

  pred_var_next = max(raw_pred_var / target_scale, 1e-12)
  pred_vol_next = float(np.sqrt(pred_var_next))

  last_test_var = float(rs_test[-1])
  last_test_vol = float(np.sqrt(max(last_test_var, 1e-12)))

  diff_vol = pred_vol_next - last_test_vol
  pct_vol = (diff_vol / last_test_vol) * 100.0 if last_test_vol > 0 else np.nan
  direccion = "SUBE" if diff_vol > 0 else ("BAJA" if diff_vol < 0 else "SE MANTIENE")

  diff_var = pred_var_next - last_test_var
  pct_var = (diff_var / last_test_var) * 100.0 if last_test_var > 0 else np.nan

  alpha = 0.05
  q_low, q_high = compute_fhs_quantiles_from_train(
      model=model,
      scaler=scaler,
      target_scale=target_scale,
      y_train=y_train,
      rs_train=rs_train,
      alpha=alpha,
      mu=0.0,
  )

  r_low = q_low * pred_vol_next
  r_high = q_high * pred_vol_next

  print(f"Checkpoint        : {ckpt_path}")
  print(f"Última fecha test : {df_test.index[-1].date()}")
  print(f"Volatilidad cierre: {last_test_vol:.6f}  (Varianza: {last_test_var:.8f})")
  print(f"Volatilidad sig.  : {pred_vol_next:.6f}  (Varianza: {pred_var_next:.8f})")
  print(f"Resultado         : {direccion} -> {diff_vol:+.6f} ({pct_vol:+.2f}%)")
  print(f"Cambio varianza   : {diff_var:+.8f} ({pct_var:+.2f}%)")
  print(f"Intervalo retorno ({int((1-alpha)*100)}% FHS): [{r_low:+.6f}, {r_high:+.6f}]")

  close_col = next(
      (c for c in df_test.columns if c.lower() in {"close", "adj_close", "adj close", "price"}),
      None,
  )
  if close_col is not None:
    last_price = float(df_test[close_col].iloc[-1])
    p_low = last_price * np.exp(r_low)
    p_high = last_price * np.exp(r_high)
    print(f"Último precio test: {last_price:.4f}")
    print(f"Intervalo precio  ({int((1-alpha)*100)}% FHS): [{p_low:.4f}, {p_high:.4f}]")


if __name__ == "__main__":
  main()