"""Model persistence utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import joblib


def save_model(model: Any, path: Path) -> None:
    """Serialize a model to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    """Load a serialized model from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Run the full pipeline to train models first."
        )
    return joblib.load(path)


def save_feature_columns(columns: List[str], path: Path) -> None:
    """Persist feature column names to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(columns, path)


def load_feature_columns(path: Path) -> List[str]:
    """Load feature column names from disk."""
    return joblib.load(path)
