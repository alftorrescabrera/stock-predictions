"""Baseline models for comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass
class BaselineModels:
    """Train simple baselines to compare against XGBoost."""

    random_state: int = 42

    def naive_direction(self, df) -> np.ndarray:
        """Predict direction using prior-day return sign as a naive baseline."""
        return (df["return_lag_1"].fillna(0) > 0).astype(int).to_numpy()

    def train_classification_baselines(self, X_train, y_train) -> Dict[str, object]:
        """Fit simple classification baselines for comparison."""
        models = {}

        lr = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LogisticRegression(max_iter=1000, random_state=self.random_state)),
        ])
        lr.fit(X_train, y_train)
        models["LogisticRegression"] = lr

        rf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=200, random_state=self.random_state)),
        ])
        rf.fit(X_train, y_train)
        models["RandomForestClassifier"] = rf

        return models

    def train_regression_baselines(self, X_train, y_train) -> Dict[str, object]:
        """Fit simple regression baselines for comparison."""
        models = {}

        lr = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression()),
        ])
        lr.fit(X_train, y_train)
        models["LinearRegression"] = lr

        rf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=200, random_state=self.random_state)),
        ])
        rf.fit(X_train, y_train)
        models["RandomForestRegressor"] = rf

        return models
