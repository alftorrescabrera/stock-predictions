"""Model evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    mean_squared_error,
)


@dataclass
class Evaluator:
    """Evaluate classification and regression models."""

    def evaluate_classification(self, y_true, y_pred, y_proba=None) -> Dict[str, object]:
        """Compute standard classification metrics."""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, zero_division=0, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
        }
        if y_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except ValueError:
                metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = None
        return metrics

    def evaluate_classification_by_group(
        self, y_true, y_pred, y_proba, groups
    ) -> Dict[str, Dict[str, object]]:
        """Compute classification metrics grouped by a label (e.g., ticker)."""
        results: Dict[str, Dict[str, object]] = {}
        group_series = pd.Series(groups).reset_index(drop=True)
        y_true_series = pd.Series(y_true).reset_index(drop=True)
        y_pred_series = pd.Series(y_pred).reset_index(drop=True)
        y_proba_series = None if y_proba is None else pd.Series(y_proba).reset_index(drop=True)

        for group_value in sorted(group_series.dropna().unique()):
            mask = group_series == group_value
            if mask.sum() == 0:
                continue
            group_true = y_true_series[mask]
            group_pred = y_pred_series[mask]
            group_proba = None if y_proba_series is None else y_proba_series[mask]
            metrics = self.evaluate_classification(group_true, group_pred, group_proba)
            metrics["support"] = int(mask.sum())
            metrics["support_pos"] = int((group_true == 1).sum())
            metrics["support_neg"] = int((group_true == 0).sum())
            results[str(group_value)] = metrics

        return results

    def evaluate_regression(self, y_true, y_pred) -> Dict[str, float]:
        """Compute standard regression metrics."""
        try:
            rmse = mean_squared_error(y_true, y_pred, squared=False)
        except TypeError:
            rmse = mean_squared_error(y_true, y_pred) ** 0.5
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(rmse),
            "r2": float(r2_score(y_true, y_pred)),
        }

    @staticmethod
    def comparison_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Format metrics as a comparison table."""
        return pd.DataFrame(results).T
