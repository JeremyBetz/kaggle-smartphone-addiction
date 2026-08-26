"""One canonical cross-validation scheme and evaluation helpers."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class CVConfig:
    """Canonical validation configuration for all Phase 1 experiments."""

    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42

    def splitter(self) -> StratifiedKFold:
        return StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )


def evaluate_predictions(y_true, probabilities) -> dict[str, float]:
    """Evaluate binary probabilities with ranking, calibration, and threshold metrics."""
    probabilities = np.asarray(probabilities, dtype=float)
    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities),
        "accuracy_0.5": accuracy_score(y_true, probabilities >= 0.5),
    }


def cross_validate_model(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: CVConfig = CVConfig(),
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate OOF probabilities and per-fold metrics using the canonical CV."""
    oof = np.zeros(len(y), dtype=float)
    rows: list[dict[str, float | int]] = []
    for fold, (train_idx, valid_idx) in enumerate(cv.splitter().split(X, y), start=1):
        fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        probabilities = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
        oof[valid_idx] = probabilities
        rows.append({"fold": fold, **evaluate_predictions(y.iloc[valid_idx], probabilities)})
    return pd.DataFrame(rows), oof
