"""Data validation utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .config import LOW_ROW_THRESHOLD

logger = logging.getLogger(__name__)


@dataclass
class DataValidator:
    """Run data quality checks for price and news datasets."""

    low_row_threshold: int = LOW_ROW_THRESHOLD

    def validate(self, price_df: pd.DataFrame, news_df: pd.DataFrame) -> Dict[str, object]:
        """Return a validation report for missing values, duplicates, and coverage."""
        report: Dict[str, object] = {}

        report["price_missing_values"] = price_df.isna().sum().to_dict()
        report["news_missing_values"] = news_df.isna().sum().to_dict()

        report["price_duplicate_rows"] = int(price_df.duplicated().sum())
        report["news_duplicate_rows"] = int(news_df.duplicated().sum())

        report["price_invalid_dates"] = int(price_df["date"].isna().sum())
        report["news_invalid_datetimes"] = int(news_df["datetime"].isna().sum())

        valid_news = news_df.dropna(subset=["ticker", "date"]).copy()
        grouped_counts = valid_news.groupby(["ticker", "date"]).size()
        report["news_row_count"] = int(len(valid_news))
        report["news_group_total_rows"] = int(grouped_counts.sum())
        report["news_group_count_mismatch"] = int(len(valid_news) - grouped_counts.sum())
        report["news_group_count_summary"] = grouped_counts.describe().to_dict()
        report["news_group_count_sample"] = grouped_counts.sort_values(ascending=False).head(5).to_dict()

        price_counts = price_df.groupby("ticker").size()
        low_row_tickers = price_counts[price_counts < self.low_row_threshold].to_dict()
        report["tickers_with_few_rows"] = low_row_tickers

        price_tickers = set(price_df["ticker"].unique())
        news_tickers = set(news_df["ticker"].unique())
        report["news_tickers_without_price"] = sorted(list(news_tickers - price_tickers))

        logger.info("Validation report generated")
        return report
