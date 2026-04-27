"""Feature engineering for price and news data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineer:
    """Generate price features and merge news features."""

    def add_price_features(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Create return, rolling, and price-shape features plus targets."""
        df = price_df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

        df["return"] = df.groupby("ticker")["close"].pct_change()
        df["return_lag_1"] = df.groupby("ticker")["return"].shift(1)
        df["return_lag_2"] = df.groupby("ticker")["return"].shift(2)
        df["return_lag_3"] = df.groupby("ticker")["return"].shift(3)

        df["ma_5"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(5).mean())
        df["ma_10"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(10).mean())
        df["ma_20"] = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20).mean())

        df["volatility_5"] = df.groupby("ticker")["return"].transform(lambda x: x.rolling(5).std())
        df["volatility_10"] = df.groupby("ticker")["return"].transform(lambda x: x.rolling(10).std())

        df["volume_change"] = df.groupby("ticker")["volume"].pct_change()
        df["volume_ma_5"] = df.groupby("ticker")["volume"].transform(lambda x: x.rolling(5).mean())

        df["high_low_range"] = df["high"] - df["low"]
        df["close_open_body"] = df["close"] - df["open"]

        range_safe = df["high_low_range"].replace(0, np.nan)
        df["body_ratio"] = (df["close_open_body"] / range_safe).fillna(0.0)
        df["close_to_high"] = (df["close"] / df["high"] - 1).replace([np.inf, -np.inf], 0).fillna(0)
        df["close_to_low"] = (df["close"] / df["low"] - 1).replace([np.inf, -np.inf], 0).fillna(0)

        df["day_of_week"] = df["date"].dt.dayofweek

        df["target_next_return"] = df.groupby("ticker")["return"].shift(-1)
        df["target_direction"] = (df["target_next_return"] > 0).astype(int)

        return df

    def merge_news_features(
        self, price_df: pd.DataFrame, news_agg_df: pd.DataFrame, embedding_dim: int
    ) -> pd.DataFrame:
        """Join aggregated news features to price data and fill missing values."""
        df = price_df.merge(news_agg_df, on=["ticker", "date"], how="left")

        zero_embedding = [0.0] * embedding_dim
        df["news_count"] = df["news_count"].fillna(0).astype(int)

        for col in [
            "positive_count",
            "negative_count",
            "neutral_count",
            "mean_sentiment_score",
            "max_sentiment_score",
            "mean_sentiment_numeric",
            "abs_mean_sentiment",
            "latest_sentiment_numeric",
        ]:
            df[col] = df[col].fillna(0)

        df["weighted_mean_embedding"] = df["weighted_mean_embedding"].apply(
            lambda x: zero_embedding if isinstance(x, float) or x is None else x
        )
        df["latest_news_embedding"] = df["latest_news_embedding"].apply(
            lambda x: zero_embedding if isinstance(x, float) or x is None else x
        )

        # Keep vector columns for inspection alongside expanded features.
        df["embedding"] = df["latest_news_embedding"]
        df["weighted_embedding"] = df["weighted_mean_embedding"]

        return df

    @staticmethod
    def expand_embeddings(df: pd.DataFrame, embedding_dim: int) -> pd.DataFrame:
        """Expand embedding vectors into numeric columns."""
        df = df.copy()
        weighted_cols = [f"weighted_emb_{i}" for i in range(embedding_dim)]
        latest_cols = [f"latest_emb_{i}" for i in range(embedding_dim)]

        weighted_matrix = np.array(df["weighted_mean_embedding"].tolist(), dtype=float)
        latest_matrix = np.array(df["latest_news_embedding"].tolist(), dtype=float)

        weighted_df = pd.DataFrame(weighted_matrix, columns=weighted_cols, index=df.index)
        latest_df = pd.DataFrame(latest_matrix, columns=latest_cols, index=df.index)

        df = pd.concat(
            [
                df.drop(["weighted_mean_embedding", "latest_news_embedding"], axis=1),
                weighted_df,
                latest_df,
            ],
            axis=1,
        )
        return df

    @staticmethod
    def get_feature_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
        """Return feature column names after excluding targets and keys."""
        return [col for col in df.columns if col not in exclude]
