"""Model training utilities."""

from __future__ import annotations

from dataclasses import dataclass

from xgboost import XGBClassifier, XGBRegressor


@dataclass
class ModelTrainer:
    """Train XGBoost models for classification and regression."""

    random_state: int = 42

    def train_classifier(self, X_train, y_train, X_val, y_val) -> XGBClassifier:
        """Fit an XGBoost classifier using a validation set."""
        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=self.random_state,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model

    def train_regressor(self, X_train, y_train, X_val, y_val) -> XGBRegressor:
        """Fit an XGBoost regressor using a validation set."""
        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model
