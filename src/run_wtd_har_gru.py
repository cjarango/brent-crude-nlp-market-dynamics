import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

from src.eval.scorer import VolatilityScorer
from src.models.wtd_har_gru_wrapper import WTDHARGRUWrapper
from src.trainer.walk_forward import WalkForwardTrainer

warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 70)
    print(f"ENTORNO DETECTADO: Usando GPU -> {gpu_name}")
    print("=" * 70)
else:
    print("=" * 70)
    print("ENTORNO DETECTADO: Usando CPU (No se detectó GPU)")
    print("=" * 70)

# 2. AHORA USAMOS PROJECT_ROOT PARA LAS RUTAS
OUT_PATH = PROJECT_ROOT / "data" / "processed"
RESULTS_PATH = PROJECT_ROOT / "results" / "WTD_HAR_GRU_FULL"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

print("Cargando datos...")
df_train = pd.read_parquet(OUT_PATH / "brent_train.parquet")
y_train = df_train["log_returns"].values.astype("float64")
rs_train = df_train["rs_daily"].values.astype("float64")

def wtd_har_gru_search_space(trial: optuna.Trial) -> dict:
    return {
        "lag_weekly": trial.suggest_categorical("lag_weekly", [5, 7]),
        "lag_monthly": trial.suggest_categorical("lag_monthly", [22, 30]),
        "look_back": trial.suggest_categorical("look_back", [7, 14, 21, 28]),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]), 
        "num_layers": trial.suggest_int("num_layers", 1, 2),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),                    
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True), 
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "max_epochs": 200,
        "patience": 20,
        "retrain_freq": 1,
        "target_scale": 1e4, 
        "seed": 42
    }

if __name__ == "__main__":
    current_warmup = 1210
    print(f"Iniciando WTD-HAR-GRU | Ejecución Única | warmup={current_warmup}")

    trainer = WalkForwardTrainer(
        model_class=WTDHARGRUWrapper,
        scorer=VolatilityScorer(),
        output_dir=RESULTS_PATH,
        n_splits=5,
        warmup=current_warmup,
        embargo=1,
        suggest_params_fn=wtd_har_gru_search_space,
        n_trials=50,
        early_stop_rounds=15
    )

    t0 = time.time()
    trainer.run(y_train, rs_train)
    print(f"Tiempo total: {(time.time() - t0) / 60:.1f} min")