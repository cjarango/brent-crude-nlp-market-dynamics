# Brent Crude Volatility & Geopolitical Risk Forecasting

A multimodal Deep Learning framework integrating econometric baseline models with NLP-extracted geopolitical news signals to model and forecast Brent crude oil price dynamics, returns, and market risk.

## Overview

This project explores the integration of quantitative econometric models (e.g., GARCH) with natural language processing (NLP) to forecast the volatility and market dynamics of Brent crude oil. By combining historical financial time series with contextual embeddings extracted from global geopolitical news, the framework aims to capture non-linear market shocks and improve risk modeling performance.

## Objectives

- **Baseline Econometric Modeling:** Implement conditional volatility models (GARCH/EGARCH) as primary quantitative priors.
- **Geopolitical Signal Extraction:** Process global news headlines using domain-specific NLP models (e.g., FinBERT) to generate daily contextual embeddings.
- **Multimodal Fusion:** Train temporal Deep Learning architectures (LSTM / GRU / TCN) that fuse financial features with text embeddings.
- **Risk Assessment:** Evaluate predictive performance on realized volatility and return dynamics using financial loss functions (e.g., QLIKE, MSE) and directional accuracy metrics.

## Data Sources

- **Financial Time Series:** Daily Brent Crude Oil futures (`BZ=F`) retrieved from Yahoo Finance via `yfinance`.
  - **Timeframe:** 10-year historical window (2014 – Present).
  - **Features:** Open, High, Low, Close, Adjusted Close, Volume, Daily Log Returns, and Realized Volatility proxies.
- **Geopolitical News:** Global news headlines and geopolitical event signals sourced from open repositories (GDELT / HuggingFace Financial Datasets).

## Installation & Setup

To set up the development environment locally using Conda:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/tu-usuario/brent-crude-volatility-geopolitical-risk.git
    cd brent-crude-volatility-geopolitical-risk
    ```

2. **Recreate the Conda environment:**
    ```bash
    conda env create -f environment.yml
    ```

3. **Activate the environment:**
    ```bash
    conda activate brent-dl
    ```

## Project Structure

```text
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   └── utils/
├── environment.yml
├── LICENSE
└── README.md
```

## License

Distributed under the MIT License. See `LICENSE` for more information.
