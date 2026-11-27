from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch,
    lilliefors
)
from statsmodels.tsa.stattools import bds
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = ROOT / "results"
DATA_PATH = ROOT / "data" / "processed"


MODELS = {
    "garch": "Modelo GARCH",
    "egarch": "Modelo EGARCH",
    "har": "Modelo HAR",
    "wtd_har": "Modelo WTD-HAR",
    "gru": "Gated Recurrent Unit",
    "tcn": "Temporal Convolutional Network",
    "bigru_tcn": "Modelo BiGRU-TCN",
    "causal_transformer": "Causal Transformer",
    "conv_transformer": "Convolutional Transformer",
    "band_causal_tcn": "Band Causal TCN",
    "markov_tcn": "Markov TCN"
}


COMMON_PARAMS = {
    "max_epochs": 200,
    "patience": 20,
    "retrain_freq": 1,
    "seed": 42
}


MODEL_CONFIG = {

    "gru": {
        "execution": "exec_2",
        "short_name": "GRU",
        "description": (
            "Arquitectura neuronal recurrente basada en "
            "mecanismos de actualización y reinicio que "
            "permiten controlar la propagación de información "
            "a través de las dependencias temporales."
        ),
        "params": {
            "look_back": 7,
            "hidden_size": 128,
            "num_layers": 1,
            "dropout": 0.299,
            "learning_rate": 0.0065,
            "weight_decay": 6.5e-4
        }
    },

    "tcn": {
        "execution": "exec_6",
        "short_name": "TCN",
        "description": (
            "Arquitectura convolucional temporal basada en "
            "convoluciones causales y dilatadas, que permite "
            "capturar dependencias históricas mediante un "
            "campo receptivo temporal expansivo."
        ),
        "params": {
            "look_back": 14,
            "kernel_size": 3,
            "num_convs": 2,
            "num_levels": 5,
            "channels": 64,
            "dropout": 0.232,
            "learning_rate": 0.0072,
            "weight_decay": 6.0e-4
        }
    },

    "bigru_tcn": {
        "execution": "exec_1",
        "short_name": "BiGRU-TCN",
        "description": (
            "Arquitectura híbrida que combina un bloque "
            "recurrente bidireccional GRU con una red "
            "convolucional temporal y capas densas para "
            "integrar representaciones de la dinámica histórica."
        ),
        "params": {
            "look_back": 7,
            "bigru_hidden": 32,
            "tcn_channels": 32,
            "tcn_levels": 3,
            "kernel_size": 2,
            "num_convs": 1,
            "dense_units": 256,
            "num_dense_layers": 2,
            "dropout": 0.176,
            "learning_rate": 0.0004,
            "weight_decay": 2.0e-5
        }
    },

    "causal_transformer": {
        "execution": "exec_2",
        "short_name": "Causal Transformer",
        "description": (
            "Arquitectura basada en un codificador Transformer "
            "con mecanismos de autoatención causal, conexiones "
            "residuales y prenormalización, diseñada para "
            "modelar dependencias temporales sin utilizar "
            "información futura."
        ),
        "params": {
            "look_back": 7,
            "d_model": 32,
            "nhead": 2,
            "num_layers": 2,
            "dim_feedforward": 64,
            "dropout": 0.214,
            "activation": "relu",
            "learning_rate": 0.0071,
            "weight_decay": 6.0e-4
        }
    },

    "conv_transformer": {
        "execution": "exec_1",
        "short_name": "Convolutional Transformer",
        "description": (
            "Arquitectura híbrida que integra convoluciones "
            "causales para la extracción de patrones locales "
            "con mecanismos de autoatención para representar "
            "dependencias temporales dentro de la ventana "
            "histórica."
        ),
        "params": {
            "look_back": 21,
            "d_model": 32,
            "nhead": 4,
            "num_layers": 1,
            "dim_feedforward": 64,
            "conv_kernel_size": 3,
            "dropout": 0.208,
            "activation": "relu",
            "learning_rate": 0.0003,
            "weight_decay": 9.7e-4
        }
    },

    "band_causal_tcn": {
        "execution": "exec_1",
        "short_name": "Band Causal TCN",
        "description": (
            "Arquitectura convolucional multiescala que "
            "combina una descomposición wavelet causal "
            "con bloques convolucionales temporales dilatados "
            "para representar componentes de frecuencia "
            "y dependencias históricas de la volatilidad."
        ),
        "params": {
            "look_back": 7,
            "num_levels": 4,
            "channels": 128,
            "kernel_size": 2,
            "num_convs": 3,
            "dropout": 0.248,
            "learning_rate": 0.0010,
            "weight_decay": 9.2e-4,
            "J": 2,
            "wavelet": "db4",
            "band_kernel_size": 1,
            "batch_size": 128
        }
    }
}


ECONOMETRIC = {"garch", "egarch", "har", "wtd_har"}

MODEL_CONFIG.update({

    "garch": {
        "execution": "",
        "short_name": "GARCH",
        "description": (
            "Modelo de heterocedasticidad condicional "
            "autorregresiva generalizada que representa "
            "la persistencia de la varianza mediante "
            "innovaciones y varianzas pasadas."
        ),
        "params": {
            "p": 1,
            "q": 3,
            "dist": "normal"
        }
    },

    "egarch": {
        "execution": "",
        "short_name": "EGARCH",
        "description": (
            "Modelo exponencial de heterocedasticidad "
            "condicional autorregresiva que permite "
            "representar respuestas asim\u00e9tricas de "
            "la volatilidad ante las innovaciones."
        ),
        "params": {
            "p": 1,
            "q": 3,
            "dist": "t"
        }
    },

    "har": {
        "execution": "exec_6",
        "short_name": "HAR",
        "description": (
            "Modelo autorregresivo heterog\u00e9neo que "
            "combina componentes diarios, semanales "
            "y mensuales de la varianza realizada "
            "para representar su persistencia temporal."
        ),
        "params": {
            "lag_weekly": 5,
            "lag_monthly": 22
        }
    },

    "wtd_har": {
        "execution": "exec_6",
        "short_name": "WTD-HAR",
        "description": (
            "Modelo autorregresivo heterog\u00e9neo que "
            "incorpora una descomposici\u00f3n wavelet "
            "para representar la din\u00e1mica de la "
            "varianza en distintas escalas temporales."
        ),
        "params": {
            "lag_weekly": 5,
            "lag_monthly": 22,
            "wtd_level": 3
        }
    },

    "markov_tcn": {
        "execution": "exec_7",
        "short_name": "Markov TCN",
        "description": (
            "Arquitectura convolucional temporal que "
            "incorpora reg\u00edmenes de Markov para "
            "representar cambios en la din\u00e1mica "
            "y los estados de volatilidad del mercado."
        ),
        "params": {
            "look_back": 14,
            "kernel_size": 3,
            "num_convs": 2,
            "num_levels": 5,
            "channels": 64,
            "dropout": 0.232,
            "learning_rate": 0.0072,
            "weight_decay": 6.0e-4,
            "ma_window": 22,
            "ar_order": 2,
            "k_regimes": 2
        }
    }
})


PARAM_LABELS = {
    "p": "Orden autorregresivo",
    "q": "Orden de media m\u00f3vil",
    "dist": "Distribuci\u00f3n",
    "lag_weekly": "Ventana semanal",
    "lag_monthly": "Ventana mensual",
    "wtd_level": "Nivel de descomposici\u00f3n wavelet",
    "ma_window": "Ventana de media m\u00f3vil",
    "ar_order": "Orden autorregresivo",
    "k_regimes": "N\u00famero de reg\u00edmenes",
    "look_back": "Ventana de contexto",
    "hidden_size": "Unidades ocultas",
    "num_layers": "Número de capas",
    "dropout": "Dropout",
    "learning_rate": "Tasa de aprendizaje",
    "weight_decay": "Weight decay",
    "kernel_size": "Tamaño del kernel",
    "num_convs": "Convoluciones por bloque",
    "num_levels": "Niveles convolucionales",
    "channels": "Canales",
    "bigru_hidden": "Unidades ocultas BiGRU",
    "tcn_channels": "Canales TCN",
    "tcn_levels": "Niveles TCN",
    "dense_units": "Unidades densas",
    "num_dense_layers": "Capas densas",
    "d_model": "Dimensión del modelo",
    "nhead": "Cabezas de atención",
    "dim_feedforward": "Dimensión feed-forward",
    "activation": "Función de activación",
    "conv_kernel_size": "Kernel convolucional",
    "J": "Nivel de descomposición wavelet",
    "wavelet": "Familia wavelet",
    "band_kernel_size": "Kernel por banda",
    "batch_size": "Tamaño del lote",
    "max_epochs": "Máximo de épocas",
    "patience": "Paciencia",
    "retrain_freq": "Frecuencia de actualización",
    "seed": "Semilla"
}


def normalize_model(model):

    aliases = {
        "convolutional_transformer": "conv_transformer",
        "bigru_tsn": "bigru_tcn",
        "bandcausal_tsn": "band_causal_tcn",
        "bandcausal_tcn": "band_causal_tcn"
    }

    return aliases.get(model, model)


def model_directory(model):

    model = normalize_model(model)

    if model not in MODEL_CONFIG:
        raise ValueError(
            f"No hay resultados integrados para {model}."
        )

    path = (
        RESULTS_PATH
        / model
        / MODEL_CONFIG[model]["execution"]
    )

    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró la carpeta {path}"
        )

    return path


@lru_cache(maxsize=2)
def load_partition(partition):

    if partition not in ("train", "test"):
        raise ValueError("Partición no válida.")

    path = DATA_PATH / f"brent_{partition}.parquet"

    df = pd.read_parquet(path).copy()

    df.index = pd.to_datetime(df.index)

    if not df.index.is_unique:
        raise ValueError(
            "El parquet contiene fechas duplicadas."
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "El parquet no está ordenado cronológicamente."
        )

    return df


def load_predictions(model, partition):

    model = normalize_model(model)

    directory = model_directory(model)

    if model in ECONOMETRIC:
        if partition == "train":
            path = RESULTS_PATH / model / f"{model}_train_predictions.csv"

            if not path.is_file():
                path = RESULTS_PATH / model / f"best_{model}_train_predictions.csv"

            if not path.is_file():
                return None
        else:
            path = RESULTS_PATH / model / f"best_{model}_predictions.csv"
    else:
        path = directory / f"{model}_{partition}_predictions.csv"

    df = pd.read_csv(path)

    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    required = {"rs_true", "y_pred"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"Columnas faltantes en {path.name}: "
            f"{required - set(df.columns)}"
        )

    original = load_partition(partition)

    if "date" in df.columns:

        df["date"] = pd.to_datetime(df["date"])

        if df["date"].duplicated().any():
            raise ValueError(
                "Las predicciones contienen fechas duplicadas."
            )

        aligned = original.reindex(
            pd.DatetimeIndex(df["date"])
        )

    else:

        look_back = (
            {
                "garch": 0,
                "egarch": 0,
                "har": 22,
                "wtd_har": 22
            }[model]
            if model in ECONOMETRIC
            else MODEL_CONFIG[model]["params"]["look_back"]
        )

        expected = (
            original.iloc[look_back:]
            if partition == "train"
            else original
        )

        if len(df) != len(expected):
            raise ValueError(
                f"Longitud inesperada en {path.name}: "
                f"{len(df)} frente a {len(expected)}."
            )

        aligned = expected.copy()

        df["date"] = aligned.index

    if len(df) != len(aligned):
        raise ValueError(
            "Las predicciones no coinciden con la partición."
        )

    if aligned["rs_daily"].isna().any():
        raise ValueError(
            "No fue posible recuperar todas las fechas."
        )

    if not np.allclose(
        df["rs_true"].to_numpy(dtype=float),
        aligned["rs_daily"].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-12
    ):
        raise ValueError(
            f"Las varianzas observadas de {path.name} "
            "no coinciden con el parquet original."
        )

    returns = aligned["log_returns"].to_numpy(dtype=float)

    if "y_true" in df.columns:

        if not np.allclose(
            df["y_true"].to_numpy(dtype=float),
            returns,
            rtol=1e-6,
            atol=1e-12
        ):
            raise ValueError(
                "Los retornos no coinciden con el parquet."
            )

    df["y_true"] = returns

    df = df.sort_values("date").reset_index(drop=True)

    df["vol_realized"] = np.sqrt(
        df["rs_true"].clip(lower=0)
    )

    df["vol_predicted"] = np.sqrt(
        df["y_pred"].clip(lower=0)
    )

    return df


def load_history(model):

    model = normalize_model(model)

    path = (
        model_directory(model)
        / f"{model}_training_history.csv"
    )

    history = pd.read_csv(path)

    required = {
        "epoch",
        "train_loss",
        "val_loss"
    }

    if not required.issubset(history.columns):
        raise ValueError(
            "El historial no contiene todas las columnas."
        )

    history = history.sort_values("epoch")

    return history.reset_index(drop=True)


def load_optimization(model):

    directory = model_directory(model)

    if model in ECONOMETRIC:
        files = sorted(
            (RESULTS_PATH / model).glob("exec_*/*_results.csv")
        )

        if not files:
            raise FileNotFoundError(
                "No hay resultados de optimizacion."
            )

        return pd.concat(
            [
                pd.read_csv(file).assign(
                    execution=file.parent.name
                )
                for file in files
            ],
            ignore_index=True
        )

    files = list(directory.glob("*_results.csv"))

    if len(files) != 1:
        raise ValueError(
            "No se pudo identificar un CSV único de Optuna."
        )

    return pd.read_csv(files[0])


def standardized_residuals(df, model):

    model = normalize_model(model)

    returns = df["y_true"].to_numpy(dtype=float)
    variance = df["y_pred"].to_numpy(dtype=float)

    residuals = np.full(
        len(df),
        np.nan,
        dtype=float
    )

    if model in ("gru", "tcn"):

        mask = (
            np.isfinite(returns)
            & np.isfinite(variance)
            & (variance > 0)
        )

        residuals[mask] = (
            returns[mask]
            / np.sqrt(variance[mask])
        )

    else:

        mask = (
            np.isfinite(returns)
            & np.isfinite(variance)
        )

        residuals[mask] = (
            returns[mask]
            / np.sqrt(
                np.maximum(
                    variance[mask],
                    1e-12
                )
            )
        )

    return residuals


def residual_diagnostics(residuals):

    residuals = np.asarray(
        residuals,
        dtype=float
    )

    residuals = residuals[
        np.isfinite(residuals)
    ]

    lags = (10, 20, 30)
    alpha = 0.05

    rows = []

    lb = acorr_ljungbox(
        residuals,
        lags=list(lags),
        return_df=True
    )

    for lag in lags:

        rows.append({
            "test": f"Ljung-Box (Rezago {lag})",
            "stat": float(lb.loc[lag, "lb_stat"]),
            "p_raw": float(lb.loc[lag, "lb_pvalue"]),
            "family": "LB"
        })

    for lag in lags:

        arch = het_arch(
            residuals,
            nlags=lag
        )

        rows.append({
            "test": f"ARCH-LM (Rezago {lag})",
            "stat": float(arch[0]),
            "p_raw": float(arch[1]),
            "family": "ARCH"
        })

    jb_stat, jb_p = stats.jarque_bera(residuals)

    rows.append({
        "test": "Jarque-Bera",
        "stat": float(jb_stat),
        "p_raw": float(jb_p),
        "family": "Normality"
    })

    lf_stat, lf_p = lilliefors(
        residuals,
        dist="norm",
        pvalmethod="table"
    )

    rows.append({
        "test": "Lilliefors",
        "stat": float(lf_stat),
        "p_raw": float(lf_p),
        "family": "Normality"
    })

    bds_stat, bds_p = bds(
        residuals,
        max_dim=2
    )

    rows.append({
        "test": "BDS (Dimensión 2)",
        "stat": float(np.asarray(bds_stat).item()),
        "p_raw": float(np.asarray(bds_p).item()),
        "family": "BDS"
    })

    result = pd.DataFrame(rows)

    result["p_adj"] = np.nan

    for family in ("LB", "ARCH"):

        mask = result["family"] == family

        adjusted = multipletests(
            result.loc[mask, "p_raw"],
            method="bonferroni"
        )[1]

        result.loc[mask, "p_adj"] = adjusted

    conclusions = []

    for _, row in result.iterrows():

        p = (
            row["p_adj"]
            if pd.notna(row["p_adj"])
            else row["p_raw"]
        )

        family = row["family"]

        if not np.isfinite(p):

            conclusion = "No disponible"

        elif family == "LB":

            conclusion = (
                "Sin Autocorrelación"
                if p > alpha
                else "Autocorrelación Significativa"
            )

        elif family == "ARCH":

            conclusion = (
                "Sin Efectos ARCH"
                if p > alpha
                else "Efectos ARCH Significativos"
            )

        elif family == "BDS":

            conclusion = (
                "Sin Estructura No Lineal"
                if p > alpha
                else "Estructura No Lineal Significativa"
            )

        else:

            conclusion = (
                "Residuos Normales"
                if p > alpha
                else "Residuos No Normales"
            )

        conclusions.append(conclusion)

    result["conclusion"] = conclusions

    return result[
        [
            "test",
            "stat",
            "p_raw",
            "p_adj",
            "conclusion"
        ]
    ].rename(
        columns={
            "test": "Prueba Estadística",
            "stat": "Estadístico",
            "p_raw": "p-valor",
            "p_adj": "p-adj",
            "conclusion": "Conclusión (α = 0.05)"
        }
    )


def calculate_metrics(df):

    actual = df["rs_true"].to_numpy(dtype=float)
    predicted = df["y_pred"].to_numpy(dtype=float)

    if (
        not np.isfinite(actual).all()
        or not np.isfinite(predicted).all()
    ):
        raise ValueError(
            "Las predicciones contienen valores no finitos."
        )

    if np.any(actual <= 0) or np.any(predicted <= 0):
        raise ValueError(
            "Las métricas requieren varianzas positivas."
        )

    eps = 1e-8

    ratio = (
        (actual + eps)
        / (predicted + eps)
    )

    error = actual - predicted

    sigma_true = np.sqrt(actual)
    sigma_pred = np.sqrt(predicted)

    def pinball(tau):

        quantile = stats.norm.ppf(
            tau,
            loc=0,
            scale=sigma_pred
        )

        diff = sigma_true - quantile

        return float(
            np.mean(
                np.maximum(
                    tau * diff,
                    (tau - 1) * diff
                )
            )
        )

    nodes, weights = (
        np.polynomial.legendre.leggauss(20)
    )

    taus = (nodes + 1) / 2

    crps_values = np.zeros(len(actual))

    for tau, weight in zip(taus, weights):

        quantile = stats.norm.ppf(
            tau,
            loc=0,
            scale=sigma_pred
        )

        diff = sigma_true - quantile

        loss = np.maximum(
            tau * diff,
            (tau - 1) * diff
        )

        crps_values += weight * loss * 0.5

    return {
        "QLIKE": float(
            np.mean(ratio - np.log(ratio) - 1)
        ),
        "CRPS": float(
            np.mean(crps_values * 2)
        ),
        "Pinball (τ = 0.05)": pinball(0.05),
        "Pinball (τ = 0.95)": pinball(0.95),
        "MSE": float(
            np.mean(error ** 2)
        ),
        "RMSE": float(
            np.sqrt(np.mean(error ** 2))
        ),
        "MAE": float(
            np.mean(np.abs(error))
        ),
        "HMSE": float(
            np.mean((ratio - 1) ** 2)
        ),
        "HMAE": float(
            np.mean(np.abs(ratio - 1))
        )
    }


def parameter_table(model):

    model = normalize_model(model)

    params = {
        **MODEL_CONFIG[model]["params"],
        **({} if model in ECONOMETRIC else COMMON_PARAMS)
    }

    return pd.DataFrame([
        {
            "Hiperparámetro": PARAM_LABELS.get(
                key,
                key
            ),
            "Identificador": key,
            "Valor": str(value)
        }
        for key, value in params.items()
    ])


def evaluate_partition(model, partition):

    df = load_predictions(
        model,
        partition
    )

    if df is None:
        return None

    df["std_residual"] = standardized_residuals(
        df,
        model
    )

    metrics = calculate_metrics(df)

    diagnostics = residual_diagnostics(
        df["std_residual"].to_numpy()
    )

    return {
        "predictions": df,
        "metrics": metrics,
        "diagnostics": diagnostics
    }


@lru_cache(maxsize=11)
def load_model_results(model):

    model = normalize_model(model)

    if model not in MODEL_CONFIG:
        raise ValueError(
            "Los resultados de este modelo "
            "todavía no están integrados."
        )

    history = None if model in ECONOMETRIC else load_history(model)

    train = evaluate_partition(
        model,
        "train"
    )

    test = evaluate_partition(
        model,
        "test"
    )

    comparison = pd.DataFrame([
        {
            "M\u00e9trica": metric,
            "Entrenamiento": (
                train["metrics"][metric]
                if train is not None
                else np.nan
            ),
            "Prueba": test["metrics"][metric],
            "Diferencia": (
                test["metrics"][metric] - train["metrics"][metric]
                if train is not None
                else np.nan
            )
        }
        for metric in test["metrics"]
    ])

    best_epoch = None if history is None else int(
        history.loc[
            history["val_loss"].idxmin(),
            "epoch"
        ]
    )

    return {
        "model": model,
        "name": MODELS[model],
        "config": MODEL_CONFIG[model],
        "parameters": parameter_table(model),
        "history": history,
        "optimization": load_optimization(model),
        "best_epoch": best_epoch,
        "train": train,
        "test": test,
        "comparison": comparison
    }