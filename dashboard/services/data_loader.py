from functools import lru_cache

import numpy as np
import pandas as pd

from config import DATA_DIR


RAW_PATH = DATA_DIR / "raw" / "brent_raw.parquet"

TRAIN_PATH = DATA_DIR / "processed" / "brent_train.parquet"

TEST_PATH = DATA_DIR / "processed" / "brent_test.parquet"

OHLCV_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume"
]

PROCESSED_COLUMNS = OHLCV_COLUMNS + [
    "log_returns",
    "rs_daily"
]


def read_dataset(path, required_columns):

    if not path.exists():
        raise FileNotFoundError(
            f"Conjunto de datos no encontrado: {path}"
        )

    df = pd.read_parquet(path)

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Columnas faltantes: {sorted(missing)}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="raise")

    if df.index.hasnans:
        raise ValueError(
            f"Fechas inválidas en {path.name}"
        )

    df = df.sort_index()

    if df.index.has_duplicates:
        raise ValueError(
            f"Fechas duplicadas en {path.name}"
        )

    df.index.name = "Date"

    return df


@lru_cache(maxsize=1)
def _load_raw():

    return read_dataset(
        RAW_PATH,
        OHLCV_COLUMNS
    )


@lru_cache(maxsize=1)
def _load_train():

    return read_dataset(
        TRAIN_PATH,
        PROCESSED_COLUMNS
    )


@lru_cache(maxsize=1)
def _load_test():

    return read_dataset(
        TEST_PATH,
        PROCESSED_COLUMNS
    )


@lru_cache(maxsize=1)
def _load_processed():

    train = _load_train().assign(partition="Train")
    test = _load_test().assign(partition="Test")

    if train.index.max() >= test.index.min():
        raise ValueError(
            "Los periodos de entrenamiento y prueba se superponen."
        )

    return pd.concat(
        [train, test]
    ).sort_index()


def load_raw():

    return _load_raw().copy()


def load_train():

    return _load_train().copy()


def load_test():

    return _load_test().copy()


def load_processed():

    return _load_processed().copy()


def get_market_snapshot():

    df = _load_processed()

    latest = df.iloc[-1]

    variance = float(latest["rs_daily"])

    return {
        "date": df.index[-1],
        "close": float(latest["close"]),
        "log_return": float(latest["log_returns"]),
        "variance": variance,
        "volatility": (
            float(np.sqrt(variance))
            if variance >= 0
            else np.nan
        ),
        "volume": float(latest["volume"])
    }


def get_dataset_summary():

    raw = _load_raw()
    train = _load_train()
    test = _load_test()
    processed = _load_processed()

    return {
        "raw_observations": len(raw),
        "processed_observations": len(processed),
        "train_observations": len(train),
        "test_observations": len(test),
        "raw_features": len(raw.columns),
        "processed_features": len(train.columns),
        "start_date": processed.index.min(),
        "end_date": processed.index.max(),
        "train_start": train.index.min(),
        "train_end": train.index.max(),
        "test_start": test.index.min(),
        "test_end": test.index.max(),
        "train_percentage": len(train) / len(processed) * 100,
        "test_percentage": len(test) / len(processed) * 100
    }


def filter_data(
    partition="All",
    start_date=None,
    end_date=None
):

    datasets = {
        "All": load_processed,
        "Train": load_train,
        "Test": load_test,
        "Raw": load_raw
    }

    if partition not in datasets:
        raise ValueError(
            f"Partición inválida: {partition}"
        )

    df = datasets[partition]()

    if start_date is not None:
        df = df.loc[df.index >= pd.Timestamp(start_date)]

    if end_date is not None:
        df = df.loc[df.index <= pd.Timestamp(end_date)]

    return df