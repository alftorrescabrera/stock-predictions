# Stock Prediction Project

## Project Overview
This project predicts next-day stock direction (UP/DOWN) and next-day return using historical price data and market news. It combines price features with local news embeddings and FinBERT sentiment analysis, and uses a chronological split to reduce leakage.

## Problem Statement
- Classification: predict whether the stock goes UP or DOWN the next day.
- Regression: predict the next-day return.

## Data Description
- `price.csv` columns: date, ticker, open, high, low, close, volume
- `news.csv` columns: datetime, ticker, headline, summary

## Pipeline Summary
1. Load and validate data.
2. Clean and embed news text with SentenceTransformer (all-MiniLM-L6-v2).
3. Compute FinBERT sentiment and aggregate news by date + ticker.
4. Shift news features forward by one day to avoid leakage.
5. Engineer price features and targets.
6. Merge price + news features.
7. Expand embeddings and optionally apply PCA (fit on train only).
8. Train baselines and XGBoost models.
9. Evaluate metrics and save artifacts.

## How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## If You Add New Data
1. Replace or append rows in `data/price.csv` and/or `data/news.csv`.
2. In Streamlit, click **Recompute Embeddings (force)** to rebuild the cache.
3. Click **Run Full Pipeline** to retrain models and refresh metrics.

The cached embeddings are stored in `data_processed/news_embeddings.parquet`.

## Embeddings Must Be Computed
- If `data_processed/news_embeddings.parquet` does not exist, the app will compute embeddings on first run.
- This requires access to download `all-MiniLM-L6-v2` and `ProsusAI/finbert` from Hugging Face.
- If you cannot access Hugging Face, generate the embeddings once on a machine with access, then copy the
    `data_processed/news_embeddings.parquet` file into this project.

## Troubleshooting
**Hugging Face downloads blocked (VPN/offline)**
- First run needs access to download `all-MiniLM-L6-v2` and `ProsusAI/finbert`.
- If you cannot access Hugging Face, run once on a machine with access to generate
    `data_processed/news_embeddings.parquet`, then reuse it offline.
- Optional: set `LOCAL_FILES_ONLY=True` in `src/config.py` to force cached-only usage.
- Optional: set `HF_HUB_DISABLE_XET=1` to avoid Xet download issues.

**XGBoost error: libxgboost.dylib / libomp missing (macOS)**
- Install OpenMP runtime: `conda install -c conda-forge libomp` or `brew install libomp`.

**Non-conda environments (pip/venv)**
- macOS: `brew install libomp`
- Windows: `pip install --force-reinstall xgboost`

**nbformat missing (notebook rendering error)**
- `pip install "nbformat>=4.2.0"`

**torchvision missing**
- `pip install torchvision`

**Streamlit “Event loop is closed” after interrupt**
- This occurs when stopping downloads or terminating the app mid-run.
- Restart Streamlit after downloads complete.

## Project Structure
```
stock_prediction_project/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   ├── price.csv
│   └── news.csv
├── data_processed/
│   ├── processed_dataset.parquet
│   └── news_embeddings.parquet
├── models/
│   ├── classifier.pkl
│   ├── regressor.pkl
│   ├── pca.pkl
│   └── feature_columns.pkl
├── notebooks/
│   └── analysis.ipynb
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_loader.py
    ├── data_validator.py
    ├── text_cleaner.py
    ├── news_processor.py
    ├── feature_engineering.py
    ├── model_trainer.py
    ├── evaluator.py
    ├── baseline_models.py
    ├── model_persistence.py
    └── facade.py
```

## Models Used
- Baselines: Logistic Regression, Random Forest, Naive Direction
- Main models: XGBoost Classifier and XGBoost Regressor

## Metrics
- Classification: accuracy, precision, recall, F1, ROC-AUC
- Regression: MAE, RMSE, R2

## Design Decisions
- Local embeddings with SentenceTransformer (no API keys required).
- FinBERT sentiment analysis via Hugging Face.
- Chronological split to avoid leakage.
- News features shifted by one day to prevent using future information.

## Limitations
- Stock prediction is noisy and uncertain.
- Sentiment does not guarantee price movement.
- Embeddings capture semantics, not causality.
- This is not financial advice.

