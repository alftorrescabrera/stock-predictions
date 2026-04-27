"""Data loading utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataLoader:
    """Load price and news data from CSV files."""

    price_path: Path
    news_path: Path

    price_columns: List[str] = None
    news_columns: List[str] = None

    def __post_init__(self) -> None:
        """Initialize default expected columns when not provided."""
        self.price_columns = self.price_columns or [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        self.news_columns = self.news_columns or [
            "datetime",
            "ticker",
            "headline",
            "summary",
        ]

    def _validate_columns(self, df: pd.DataFrame, expected: List[str], name: str) -> None:
        """Ensure expected columns are present in the dataset."""
        missing = [col for col in expected if col not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

    def load_price(self) -> pd.DataFrame:
        """Load and normalize the price dataset."""
        logger.info("Loading price data from %s", self.price_path)
        df = pd.read_csv(self.price_path)
        self._validate_columns(df, self.price_columns, "price.csv")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
        return df

    def load_news(self) -> pd.DataFrame:
        """Load and normalize the news dataset."""
        logger.info("Loading news data from %s", self.news_path)
        df = pd.read_csv(self.news_path)
        self._validate_columns(df, self.news_columns, "news.csv")
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["date"] = df["datetime"].dt.date
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(["ticker", "datetime"]).reset_index(drop=True)
        return df

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load price and news data together."""
        price_df = self.load_price()
        news_df = self.load_news()
        return {"price": price_df, "news": news_df}
