"""News processing: embeddings, sentiment, and aggregation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from .text_cleaner import TextCleaner

logger = logging.getLogger(__name__)


@dataclass
class NewsProcessor:
    """Process news with local embeddings and FinBERT sentiment."""

    embedding_model_name: str
    sentiment_model_name: str
    cache_path: Path
    local_files_only: bool = False
    disable_hf_xet: bool = True

    def _load_models(self) -> None:
        """Load embedding and sentiment models, honoring offline settings."""
        if self.disable_hf_xet:
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        logger.info("Loading embedding model: %s", self.embedding_model_name)
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name, local_files_only=self.local_files_only
        )
        logger.info("Loading sentiment model: %s", self.sentiment_model_name)
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.sentiment_model_name,
            tokenizer=self.sentiment_model_name,
            local_files_only=self.local_files_only,
        )

    @staticmethod
    def _sentiment_numeric(label: str) -> int:
        """Map sentiment labels to numeric values."""
        mapping = {"positive": 1, "neutral": 0, "negative": -1}
        return mapping.get(label.lower(), 0)

    def process_news(self, news_df: pd.DataFrame, force_recompute: bool = False) -> pd.DataFrame:
        """Clean text, compute embeddings and sentiment, and cache results."""
        if self.cache_path.exists() and not force_recompute:
            logger.info("Loading cached news embeddings from %s", self.cache_path)
            cached = pd.read_parquet(self.cache_path)
            return cached

        if self.local_files_only and not self.cache_path.exists():
            raise RuntimeError(
                "Local-only mode is enabled and no cached embeddings were found. "
                "Disable local_files_only or pre-download the models and cache."
            )

        self._load_models()
        cleaner = TextCleaner()

        logger.info("Cleaning news text")
        news_df = news_df.copy()
        news_df["clean_text"] = news_df.apply(
            lambda row: cleaner.clean(row.get("headline"), row.get("summary")), axis=1
        )

        logger.info("Generating embeddings")
        texts = news_df["clean_text"].tolist()
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        news_df["embedding"] = [emb.tolist() for emb in embeddings]

        logger.info("Running sentiment analysis")
        sentiments = self.sentiment_pipeline(texts, batch_size=32, truncation=True)
        labels = [s["label"].lower() for s in sentiments]
        scores = [float(s["score"]) for s in sentiments]

        news_df["sentiment_label"] = labels
        news_df["sentiment_score"] = scores
        news_df["sentiment_numeric"] = [self._sentiment_numeric(l) for l in labels]
        news_df["sentiment_weight"] = [
            1.0 if l == "neutral" else 1.0 + score
            for l, score in zip(labels, scores)
        ]

        logger.info("Saving news embeddings cache to %s", self.cache_path)
        news_df.to_parquet(self.cache_path, index=False)
        return news_df

    def aggregate_daily(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate news features to daily ticker-level summaries."""
        logger.info("Aggregating news by date and ticker")

        def _aggregate(group: pd.DataFrame) -> pd.Series:
            """Aggregate a single ticker-date group into summary features."""
            embeddings = np.array(group["embedding"].tolist(), dtype=float)
            weights = np.array(group["sentiment_weight"].tolist(), dtype=float)
            if weights.sum() > 0:
                weighted_mean = np.average(embeddings, axis=0, weights=weights)
            else:
                weighted_mean = embeddings.mean(axis=0)

            latest_row = group.sort_values("datetime").iloc[-1]

            return pd.Series(
                {
                    "weighted_mean_embedding": weighted_mean.tolist(),
                    "latest_news_embedding": np.array(latest_row["embedding"], dtype=float).tolist(),
                    "news_count": int(len(group)),
                    "positive_count": int((group["sentiment_label"] == "positive").sum()),
                    "negative_count": int((group["sentiment_label"] == "negative").sum()),
                    "neutral_count": int((group["sentiment_label"] == "neutral").sum()),
                    "mean_sentiment_score": float(group["sentiment_score"].mean()),
                    "max_sentiment_score": float(group["sentiment_score"].max()),
                    "mean_sentiment_numeric": float(group["sentiment_numeric"].mean()),
                    "abs_mean_sentiment": float(group["sentiment_numeric"].abs().mean()),
                    "latest_sentiment_numeric": int(latest_row["sentiment_numeric"]),
                }
            )

        agg_df = news_df.groupby(["ticker", "date"]).apply(_aggregate).reset_index()

        # Shift news features forward by one day to prevent leakage
        agg_df["date"] = pd.to_datetime(agg_df["date"]) + pd.Timedelta(days=1)
        return agg_df

    @staticmethod
    def embedding_dim(news_df: pd.DataFrame) -> int:
        """Return embedding dimensionality from the first row."""
        first = news_df["embedding"].iloc[0]
        return len(first)
