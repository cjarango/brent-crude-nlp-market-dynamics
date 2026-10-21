import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from src.eval.scorer import VolatilityScorer
from src.models.base import BaseVolatilityModel
from tqdm.auto import tqdm


class WalkForwardTrainer:
    """Orchestrates the walk-forward validation process with checkpointing and grid search."""

    def __init__(
        self, 
        model_class: type[BaseVolatilityModel], 
        param_grid: dict[str, list],
        scorer: VolatilityScorer, 
        output_dir: str | Path,
        n_splits: int = 5,
        warmup: int = 1200
    ) -> None:
        self.model_class = model_class
        self.param_grid = list(ParameterGrid(param_grid))
        self.scorer = scorer
        self.output_dir = Path(output_dir)
        self.n_splits = n_splits
        self.warmup = warmup
        
        if not self.output_dir.exists() or not self.output_dir.is_dir():
            raise FileNotFoundError(
                f"Output directory '{self.output_dir}' does not exist. Please create it first."
            )
            
        self.csv_path = self.output_dir / "results.csv"
        self.json_path = self.output_dir / "checkpoint.json"

    def _save_checkpoint(self, status: str, current_params: dict, elapsed_time: float, failed_combinations: int) -> None:
        """Saves the current overall state, history and metadata to a JSON file."""
        
        history = {}
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r') as f:
                    existing_data = json.load(f)
                    history = existing_data.get("results_history", {})
            except Exception:  # noqa: BLE001, S110
                pass

        if self.csv_path.exists():
            try:
                df = pd.read_csv(self.csv_path)
                for _, row in df.iterrows():
                    param_dict = {k: v for k, v in row.items() if k not in ['fold', 'train_size', 'val_size', 'qlike', 'mse', 'rmse', 'mae']}
                    param_key = str(param_dict)
                    
                    if param_key not in history:
                        history[param_key] = {"folds": {}}
                        
                    history[param_key]["folds"][str(int(row['fold']))] = {
                        "qlike": row['qlike'],
                        "mse": row['mse'],
                        "rmse": row['rmse'],
                        "mae": row['mae']
                    }
            except Exception:  # noqa: BLE001, S110
                pass

        checkpoint_data = {
            "model_name": self.model_class.__name__,
            "status": status,
            "total_combinations": len(self.param_grid),
            "failed_combinations": failed_combinations,
            "current_params": current_params,
            "n_splits": self.n_splits,
            "warmup": self.warmup,
            "total_time_seconds": round(elapsed_time, 2),
            "timestamp": datetime.now().isoformat(),
            "results_history": history
        }
        
        with open(self.json_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=4)

    def _load_completed_folds(self, params: dict) -> int:
        """Reads the results CSV to see how many folds are already done for a specific combination."""
        if not self.csv_path.exists():
            return 0
        
        try:
            df_existing = pd.read_csv(self.csv_path)
            if df_existing.empty:
                return 0
                
            for key, value in params.items():
                df_existing = df_existing[df_existing[key] == value]
                
            if df_existing.empty:
                return 0
                
            return int(df_existing['fold'].max())
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"Could not read existing CSV: {e}. Starting from scratch.")
            return 0

    def run(self, y: np.ndarray, rs: np.ndarray) -> None:
        """Runs the walk-forward validation over the entire grid with progress bars and resume support."""
        y_warm = y[self.warmup:]
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        folds = list(tscv.split(y_warm))
        
        start_global_time = time.time()
        failed_combinations = 0
        
        pbar = tqdm(self.param_grid, desc="Grid Search", unit="config")
        
        for params in pbar:
            pbar.set_postfix_str(f"params={params}")
            
            completed_folds = self._load_completed_folds(params)
            if completed_folds >= self.n_splits:
                continue
                
            model = self.model_class(params)
            
            folds_to_process = folds[completed_folds:]
            has_failed = False
            
            for i, (train_idx, val_idx) in enumerate(folds_to_process):
                fold_idx = completed_folds + i + 1

                val_start = self.warmup + val_idx[0]
                val_end = self.warmup + val_idx[-1] + 1
                
                y_train = y[:val_start]
                y_val = y[val_start:val_end]
                rs_val = rs[val_start:val_end]

                context = model.context_length
                y_seq = y[val_start - context : val_end]

                metrics = {"qlike": np.nan, "mse": np.nan, "rmse": np.nan, "mae": np.nan}
                
                try:
                    model.fit(y_train)
                    
                    if not model.is_valid:
                        has_failed = True
                    else:
                        preds = model.forecast(y_seq)
                        metrics = self.scorer.calculate_all(rs_val, preds)
                        
                except Exception:  # noqa: BLE001
                    has_failed = True

                fold_result = {
                    **params,
                    "fold": fold_idx,
                    "train_size": len(y_train),
                    "val_size": len(y_val),
                    **metrics
                }
                
                df_fold = pd.DataFrame([fold_result])
                file_exists = self.csv_path.exists()
                df_fold.to_csv(self.csv_path, mode='a', header=not file_exists, index=False)
                
                self._save_checkpoint("running", params, time.time() - start_global_time, failed_combinations)

            if has_failed:
                failed_combinations += 1

        total_time = time.time() - start_global_time
        self._save_checkpoint("completed", {}, total_time, failed_combinations)
        pbar.close()