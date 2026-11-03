"""
walk_forward
============

Orchestrates Walk-Forward Validation, Optuna hyperparameter search,
and MLflow experiment tracking for all volatility models.
"""

import time
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
from src.eval.scorer import VolatilityScorer
from src.models.base import BaseVolatilityModel
from src.utils.logger import setup_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)

__all__ = ["WalkForwardTrainer"]


class WalkForwardTrainer:
    """Generic orchestrator for Walk-Forward Validation and Optuna."""

    def __init__(
        self,
        model_class: type[BaseVolatilityModel],
        scorer: VolatilityScorer,
        output_dir: str | Path,
        n_splits: int = 5,
        warmup: int = 1200,
        embargo: int = 1,
        n_trials: int = 30,
    ) -> None:
        """Initializes the trainer."""
        if embargo < 1:
            raise ValueError(
                f"Embargo must be >= 1 to prevent data leakage, got {embargo}."
            )

        self.model_class = model_class
        self.scorer = scorer
        self.output_dir = Path(output_dir)
        self.n_splits = n_splits
        self.warmup = warmup
        self.embargo = embargo
        self.n_trials = n_trials

        self.logger = setup_logger(f"Trainer_{model_class.__name__}")

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / f"{self.model_class.__name__}_results.csv"

    def _get_folds(self, n_samples: int) -> list[tuple[int, int, int, int]]:
        """Generates expanding-window folds with embargo."""
        val_size = (n_samples - self.warmup) // self.n_splits
        folds = []
        for i in range(self.n_splits):
            val_start = self.warmup + i * val_size
            val_end = val_start + val_size if i < self.n_splits - 1 else n_samples
            train_end = val_start - self.embargo
            folds.append((0, train_end, val_start, val_end))
        return folds

    def _load_completed_folds(self, params: dict) -> int:
        """Returns the number of folds already completed for a given config."""
        if not self.csv_path.exists():
            return 0
        try:
            df = pd.read_csv(self.csv_path)
            if df.empty:
                return 0
            for key, value in params.items():
                df = df[df[key] == value]
            return int(df["fold"].max()) if not df.empty else 0
        except Exception:  # noqa: BLE001
            return 0

    def objective(self, trial: optuna.Trial, y: np.ndarray, rs: np.ndarray) -> float:
        """Evaluates one Optuna trial via expanding-window walk-forward."""
        params = self.model_class.suggest_params(trial)
        folds = self._get_folds(len(y))
        completed_folds = self._load_completed_folds(params)

        if completed_folds >= self.n_splits:
            df = pd.read_csv(self.csv_path)
            for k, v in params.items():
                df = df[df[k] == v]
            return float(df["qlike"].mean())

        fold_qlikes = []
        all_converged = True

        for i, (train_start, train_end, val_start, val_end) in enumerate(
            folds[completed_folds:]
        ):
            fold_idx = completed_folds + i + 1

            y_train = y[train_start:train_end]
            rs_train = rs[train_start:train_end]
            y_val = y[val_start:val_end]
            rs_val = rs[val_start:val_end]

            model = self.model_class(**params)
            
            y_seq = y_val
            rs_seq = rs_val

            qlike_val = 1e10  
            converged = True
            epochs_run = 0
            train_time = 0.0

            try:
                start_fold_time = time.time()
                model.fit(y_train, rs_train)
                train_time = time.time() - start_fold_time

                diag = model.get_diagnostics()
                converged = diag.get("converged", True)
                epochs_run = diag.get("epochs_run", 0)

                if not converged:
                    self.logger.warning(f"Fold {fold_idx} did NOT converge with {params}")

                if model.is_valid:
                    preds = model.forecast(y_seq, rs_seq)
                    qlike_val = self.scorer.qlike(rs_val, preds)
                    
                    if not np.isfinite(qlike_val):
                        self.logger.error(f"Fold {fold_idx} produced non-finite QLIKE: {qlike_val}")
                        qlike_val = 1e10
                        converged = False
                else:
                    self.logger.error(f"Fold {fold_idx} model invalid with {params}")
                    converged = False
                    
            except Exception as e:  # noqa: BLE001
                self.logger.error(f"Exception in Fold {fold_idx} with {params}: {e}")
                converged = False
                qlike_val = 1e10
                all_converged = False

            if not converged:
                all_converged = False

            fold_qlikes.append(qlike_val)

            fold_result = {
                **params,
                "fold": fold_idx,
                "train_size": len(y_train),
                "val_size": len(y_val),
                "converged": converged,
                "epochs_run": epochs_run,
                "train_time_sec": round(train_time, 2),
                "qlike": qlike_val,
            }
            pd.DataFrame([fold_result]).to_csv(
                self.csv_path, mode="a", header=not self.csv_path.exists(), index=False
            )

            mlflow.log_metric(f"fold_{fold_idx}_qlike", qlike_val, step=fold_idx)
            mlflow.log_metric(f"fold_{fold_idx}_converged", int(converged), step=fold_idx)
            mlflow.log_metric(f"fold_{fold_idx}_epochs", epochs_run, step=fold_idx)
            mlflow.log_metric(f"fold_{fold_idx}_train_time", train_time, step=fold_idx)

            trial.report(np.mean(fold_qlikes), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        trial.set_user_attr("converged", all_converged)
        return float(np.mean(fold_qlikes))

    def run(self, y: np.ndarray, rs: np.ndarray) -> None:
        """Runs the full Optuna search and logs the best trial to MLflow."""
        mlflow.set_experiment("Brent_Volatility_Benchmark")

        with mlflow.start_run(run_name=self.model_class.__name__):
            study_name = f"{self.model_class.__name__}_study"
            storage_url = f"sqlite:///{self.output_dir.resolve()}/optuna_study.db"
            
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                load_if_exists=True,
                direction="minimize",
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=2,
                ),
            )
            
            remaining_trials = max(0, self.n_trials - len(study.trials))
            self.logger.info(
                f"Resuming study '{study_name}'. "
                f"Trials completed so far: {len(study.trials)}. "
                f"Running {remaining_trials} more."
            )
            
            if remaining_trials > 0:
                study.optimize(
                    lambda trial: self.objective(trial, y, rs),
                    n_trials=remaining_trials,
                    catch=(Exception,),
                    show_progress_bar=True
                )

            valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            if not valid_trials:
                self.logger.error(
                    f"[{self.model_class.__name__}] No trials completed successfully. "
                    f"Check the CSV at {self.csv_path} for errors."
                )
                return

            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_qlike", study.best_value)
            mlflow.log_metric("total_trials", len(study.trials))

            best_trial = study.best_trial
            mlflow.log_metric(
                "best_trial_converged",
                int(best_trial.user_attrs.get("converged", True)),
            )

            self.logger.warning(
                f"[{self.model_class.__name__}] Optimization finished. "
                f"Best QLIKE: {study.best_value:.4f} | Params: {study.best_params}"
            )