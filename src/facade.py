"""Facade pattern orchestration for the pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .baseline_models import BaselineModels
from .config import (
    MODELS_DIR,
    NEWS_CACHE_FILE,
    PCA_COMPONENTS,
    PROCESSED_DATA_FILE,
    RANDOM_STATE,
    SENTIMENT_MODEL_NAME,
    USE_PCA,
    EMBEDDING_MODEL_NAME,
)
from .data_loader import DataLoader
from .data_validator import DataValidator
from .evaluator import Evaluator
from .feature_engineering import FeatureEngineer
from .model_persistence import (
    load_feature_columns,
    load_model,
    save_feature_columns,
    save_model,
)
from .model_trainer import ModelTrainer
from .news_processor import NewsProcessor

logger = logging.getLogger(__name__)


@dataclass
class StockPredictionFacade:
    """Facade to orchestrate the end-to-end pipeline."""

    data_loader: DataLoader
    data_validator: DataValidator
    news_processor: NewsProcessor
    feature_engineer: FeatureEngineer
    model_trainer: ModelTrainer
    evaluator: Evaluator
    baseline_models: BaselineModels
    use_pca: bool = USE_PCA
    pca_components: int = PCA_COMPONENTS
    random_state: int = RANDOM_STATE

    price_df: Optional[pd.DataFrame] = None
    news_df: Optional[pd.DataFrame] = None
    processed_df: Optional[pd.DataFrame] = None

    train_df: Optional[pd.DataFrame] = None
    val_df: Optional[pd.DataFrame] = None
    test_df: Optional[pd.DataFrame] = None

    pca: Optional[PCA] = None
    feature_columns: List[str] = field(default_factory=list)
    base_feature_columns: List[str] = field(default_factory=list)
    embedding_columns: List[str] = field(default_factory=list)
    non_embedding_columns: List[str] = field(default_factory=list)

    classifier: Optional[object] = None
    regressor: Optional[object] = None

    metrics: Dict[str, object] = field(default_factory=dict)
    baseline_metrics: Dict[str, object] = field(default_factory=dict)
    metrics_by_ticker: Dict[str, Dict[str, object]] = field(default_factory=dict)

    def load_data(self) -> None:
        """Load price and news data into memory."""
        data = self.data_loader.load_data()
        self.price_df = data["price"]
        self.news_df = data["news"]

    def validate_data(self) -> Dict[str, object]:
        """Run validation checks on loaded datasets."""
        if self.price_df is None or self.news_df is None:
            self.load_data()
        return self.data_validator.validate(self.price_df, self.news_df)

    def prepare_features(self, force_recompute_news: bool = False) -> None:
        """Process news, engineer features, and persist the dataset."""
        if self.price_df is None or self.news_df is None:
            self.load_data()

        processed_news = self.news_processor.process_news(self.news_df, force_recompute=force_recompute_news)
        embedding_dim = self.news_processor.embedding_dim(processed_news)
        news_agg = self.news_processor.aggregate_daily(processed_news)

        price_features = self.feature_engineer.add_price_features(self.price_df)
        merged = self.feature_engineer.merge_news_features(price_features, news_agg, embedding_dim)
        merged = self.feature_engineer.expand_embeddings(merged, embedding_dim)
        merged.to_parquet(PROCESSED_DATA_FILE, index=False)
        self.processed_df = merged

    @staticmethod
    def train_validation_test_split(
        df: pd.DataFrame, train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data by chronological date into train/val/test."""
        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError("Train/val/test sizes must sum to 1.0")

        df = df.sort_values("date")
        unique_dates = df["date"].drop_duplicates().sort_values().to_list()
        n = len(unique_dates)
        train_end = int(n * train_size)
        val_end = train_end + int(n * val_size)

        train_dates = unique_dates[:train_end]
        val_dates = unique_dates[train_end:val_end]
        test_dates = unique_dates[val_end:]

        train_df = df[df["date"].isin(train_dates)].reset_index(drop=True)
        val_df = df[df["date"].isin(val_dates)].reset_index(drop=True)
        test_df = df[df["date"].isin(test_dates)].reset_index(drop=True)

        return train_df, val_df, test_df

    def _apply_pca(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, emb_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit PCA on train embeddings and transform all splits."""
        self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
        X_train_emb = self.pca.fit_transform(X_train[emb_cols])
        X_val_emb = self.pca.transform(X_val[emb_cols])
        X_test_emb = self.pca.transform(X_test[emb_cols])

        pca_cols = [f"pca_{i}" for i in range(self.pca_components)]
        X_train = X_train.drop(columns=emb_cols).reset_index(drop=True)
        X_val = X_val.drop(columns=emb_cols).reset_index(drop=True)
        X_test = X_test.drop(columns=emb_cols).reset_index(drop=True)

        X_train = pd.concat([X_train, pd.DataFrame(X_train_emb, columns=pca_cols)], axis=1)
        X_val = pd.concat([X_val, pd.DataFrame(X_val_emb, columns=pca_cols)], axis=1)
        X_test = pd.concat([X_test, pd.DataFrame(X_test_emb, columns=pca_cols)], axis=1)

        return X_train, X_val, X_test

    def split_data(self):
        """Prepare train/val/test arrays and targets."""
        if self.processed_df is None:
            self.prepare_features()

        df = self.processed_df.dropna(subset=["target_next_return"]).copy()

        exclude = [
            "target_next_return",
            "target_direction",
            "date",
            "ticker",
            "embedding",
            "weighted_embedding",
        ]
        self.feature_columns = self.feature_engineer.get_feature_columns(df, exclude)
        self.base_feature_columns = self.feature_columns.copy()

        train_df, val_df, test_df = self.train_validation_test_split(df)
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        X_train = train_df[self.feature_columns].copy()
        X_val = val_df[self.feature_columns].copy()
        X_test = test_df[self.feature_columns].copy()

        emb_cols = [c for c in self.feature_columns if c.startswith("weighted_emb_") or c.startswith("latest_emb_")]
        self.embedding_columns = emb_cols
        self.non_embedding_columns = [c for c in self.feature_columns if c not in emb_cols]

        if self.use_pca and emb_cols:
            X_train, X_val, X_test = self._apply_pca(X_train, X_val, X_test, emb_cols)
            self.feature_columns = X_train.columns.tolist()

        y_train_cls = train_df["target_direction"].astype(int)
        y_val_cls = val_df["target_direction"].astype(int)
        y_test_cls = test_df["target_direction"].astype(int)

        y_train_reg = train_df["target_next_return"]
        y_val_reg = val_df["target_next_return"]
        y_test_reg = test_df["target_next_return"]

        return (X_train, X_val, X_test, y_train_cls, y_val_cls, y_test_cls, y_train_reg, y_val_reg, y_test_reg)

    def train_baselines(self, X_train, y_train_cls, y_train_reg, X_val, y_val_cls, y_val_reg, val_df):
        """Train baseline models and store validation metrics."""
        cls_models = self.baseline_models.train_classification_baselines(X_train, y_train_cls)
        reg_models = self.baseline_models.train_regression_baselines(X_train, y_train_reg)

        baseline_cls_results = {
            "NaivePreviousDirection": self.evaluator.evaluate_classification(
                y_val_cls, self.baseline_models.naive_direction(val_df)
            )
        }
        for name, model in cls_models.items():
            preds = model.predict(X_val)
            proba = model.predict_proba(X_val)[:, 1]
            baseline_cls_results[name] = self.evaluator.evaluate_classification(y_val_cls, preds, proba)

        baseline_reg_results = {}
        for name, model in reg_models.items():
            preds = model.predict(X_val)
            baseline_reg_results[name] = self.evaluator.evaluate_regression(y_val_reg, preds)

        self.baseline_metrics = {
            "classification": baseline_cls_results,
            "regression": baseline_reg_results,
        }

    def train_models(self, X_train, y_train_cls, y_train_reg, X_val, y_val_cls, y_val_reg):
        """Train XGBoost classifier and regressor."""
        self.classifier = self.model_trainer.train_classifier(X_train, y_train_cls, X_val, y_val_cls)
        self.regressor = self.model_trainer.train_regressor(X_train, y_train_reg, X_val, y_val_reg)

    def evaluate_models(self, X_test, y_test_cls, y_test_reg):
        """Evaluate models on the test split."""
        cls_pred = self.classifier.predict(X_test)
        cls_proba = self.classifier.predict_proba(X_test)[:, 1]
        reg_pred = self.regressor.predict(X_test)

        self.metrics = {
            "classification": self.evaluator.evaluate_classification(y_test_cls, cls_pred, cls_proba),
            "regression": self.evaluator.evaluate_regression(y_test_reg, reg_pred),
        }

        if self.test_df is not None and "ticker" in self.test_df.columns:
            self.metrics_by_ticker = self.evaluator.evaluate_classification_by_group(
                y_test_cls,
                cls_pred,
                cls_proba,
                self.test_df["ticker"],
            )

    def save_artifacts(self) -> None:
        """Persist trained models and metadata to disk."""
        save_model(self.classifier, MODELS_DIR / "classifier.pkl")
        save_model(self.regressor, MODELS_DIR / "regressor.pkl")
        if self.pca is not None:
            save_model(self.pca, MODELS_DIR / "pca.pkl")
        save_feature_columns(self.feature_columns, MODELS_DIR / "feature_columns.pkl")

    def load_artifacts(self) -> None:
        """Load trained models and metadata from disk."""
        classifier_path = MODELS_DIR / "classifier.pkl"
        regressor_path = MODELS_DIR / "regressor.pkl"
        if not classifier_path.exists() or not regressor_path.exists():
            raise RuntimeError(
                "Models not found. Run the full pipeline to train and save models before forecasting."
            )
        self.classifier = load_model(classifier_path)
        self.regressor = load_model(regressor_path)
        pca_path = MODELS_DIR / "pca.pkl"
        if pca_path.exists():
            self.pca = load_model(pca_path)
        self.feature_columns = load_feature_columns(MODELS_DIR / "feature_columns.pkl")

    def run_full_pipeline(self, force_recompute_news: bool = False) -> None:
        """Execute the full pipeline from data to saved models."""
        self.load_data()
        self.prepare_features(force_recompute_news=force_recompute_news)

        split = self.split_data()
        (
            X_train,
            X_val,
            X_test,
            y_train_cls,
            y_val_cls,
            y_test_cls,
            y_train_reg,
            y_val_reg,
            y_test_reg,
        ) = split

        self.train_models(X_train, y_train_cls, y_train_reg, X_val, y_val_cls, y_val_reg)
        self.evaluate_models(X_test, y_test_cls, y_test_reg)
        self.save_artifacts()

    def run_full_pipeline_on_data(self, price_df: pd.DataFrame, news_df: pd.DataFrame) -> None:
        """Run the full pipeline using provided dataframes."""
        self.price_df = price_df.copy()
        self.news_df = news_df.copy()
        self.prepare_features(force_recompute_news=True)

        split = self.split_data()
        (
            X_train,
            X_val,
            X_test,
            y_train_cls,
            y_val_cls,
            y_test_cls,
            y_train_reg,
            y_val_reg,
            y_test_reg,
        ) = split

        self.train_models(X_train, y_train_cls, y_train_reg, X_val, y_val_cls, y_val_reg)
        self.evaluate_models(X_test, y_test_cls, y_test_reg)
        self.save_artifacts()

    def _prepare_single_row(self, row: pd.DataFrame) -> pd.DataFrame:
        """Prepare a single row for model prediction."""
        if self.pca is None:
            return row[self.feature_columns].copy()

        if not self.embedding_columns:
            self.embedding_columns = [
                c for c in row.columns if c.startswith("weighted_emb_") or c.startswith("latest_emb_")
            ]
            self.non_embedding_columns = [c for c in row.columns if c not in self.embedding_columns]
            if not self.embedding_columns:
                return row[self.feature_columns].copy()

        X_non_emb = row[self.non_embedding_columns].copy().reset_index(drop=True)
        X_emb = row[self.embedding_columns].copy()
        X_emb = self.pca.transform(X_emb)
        pca_cols = [f"pca_{i}" for i in range(self.pca_components)]
        X_pca = pd.DataFrame(X_emb, columns=pca_cols)
        X = pd.concat([X_non_emb, X_pca], axis=1)
        return X

    def predict_latest(self, ticker: str) -> Dict[str, float]:
        """Predict direction and return for the latest row of a ticker."""
        if self.processed_df is None:
            self.prepare_features()
        if self.classifier is None or self.regressor is None:
            self.load_artifacts()

        df = self.processed_df.dropna(subset=["target_next_return"]).copy()
        latest_row = df[df["ticker"] == ticker].sort_values("date").iloc[-1:]
        X = self._prepare_single_row(latest_row)

        proba_up = float(self.classifier.predict_proba(X)[:, 1][0])
        pred_direction = int(self.classifier.predict(X)[0])
        pred_return = float(self.regressor.predict(X)[0])
        latest_close = float(latest_row["close"].iloc[0])
        pred_close = latest_close * (1 + pred_return)

        return {
            "latest_close": latest_close,
            "predicted_direction": pred_direction,
            "probability_up": proba_up,
            "predicted_return": pred_return,
            "predicted_close": pred_close,
        }

    def predict_for_row(self, ticker: str, date: str) -> Dict[str, float]:
        """Predict direction and return for a specific ticker/date."""
        if self.processed_df is None:
            self.prepare_features()
        if self.classifier is None or self.regressor is None:
            self.load_artifacts()

        df = self.processed_df.dropna(subset=["target_next_return"]).copy()
        row = df[(df["ticker"] == ticker) & (df["date"] == pd.to_datetime(date))].iloc[-1:]
        if row.empty:
            raise ValueError("No row found for given ticker/date")

        X = self._prepare_single_row(row)
        proba_up = float(self.classifier.predict_proba(X)[:, 1][0])
        pred_direction = int(self.classifier.predict(X)[0])
        pred_return = float(self.regressor.predict(X)[0])
        latest_close = float(row["close"].iloc[0])
        pred_close = latest_close * (1 + pred_return)

        return {
            "latest_close": latest_close,
            "predicted_direction": pred_direction,
            "probability_up": proba_up,
            "predicted_return": pred_return,
            "predicted_close": pred_close,
        }
