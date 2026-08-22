from pathlib import Path

import pandas as pd
import yfinance as yf


def fetch_raw_brent_data(
    ticker: str = "BZ=F",
    start_year: int = 2014,
    end_date: str | None = None,
    output_dir: str = "data/raw",
) -> None:
    """Fetches raw historical Brent Crude futures data from Yahoo Finance,

    standardizes column names, casts all numerical features to float32, and
    saves the dataset as a Parquet file.

    Args:
        ticker: Yahoo Finance ticker symbol for Brent Crude futures.
        start_year: Starting year for the historical data download.
        end_date: End date for the data download (YYYY-MM-DD format). Defaults
          to the current date if None.
        output_dir: Local directory path where the Parquet file will be saved.

    Raises:
        ValueError: If Yahoo Finance returns no data for the specified ticker.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    start_date = f"{start_year}-01-01"
    df = yf.download(ticker, start=start_date, end=end_date, progress=True)

    if df is None or df.empty:
        raise ValueError(f"Failed to fetch data for ticker '{ticker}'.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    df.columns.name = None

    for col in df.columns:
        df[col] = df[col].astype("float32")

    df.to_parquet(output_path / "brent_raw.parquet")


if __name__ == "__main__":
    fetch_raw_brent_data()