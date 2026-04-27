"""Project configuration and constants."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = BASE_DIR / "data_processed"
MODELS_DIR = BASE_DIR / "models"

PRICE_FILE = DATA_DIR / "price.csv"
NEWS_FILE = DATA_DIR / "news.csv"
NEWS_CACHE_FILE = PROCESSED_DIR / "news_embeddings.parquet"
PROCESSED_DATA_FILE = PROCESSED_DIR / "processed_dataset.parquet"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SENTIMENT_MODEL_NAME = "ProsusAI/finbert"
LOCAL_FILES_ONLY = False
DISABLE_HF_XET = True

USE_PCA = True
PCA_COMPONENTS = 50
RANDOM_STATE = 42

LOW_ROW_THRESHOLD = 30
