"""Kaggle submission construction and validation."""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import ID_COLUMN, TARGET_COLUMN


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Raise when a submission differs from the required local schema."""
    if list(submission.columns) != list(sample.columns):
        raise ValueError("Submission columns or order do not match sample_submission.csv")
    if len(submission) != len(sample):
        raise ValueError("Submission row count does not match sample_submission.csv")
    if not submission[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("Submission IDs or row order do not match sample_submission.csv")
    predictions = submission[TARGET_COLUMN]
    if predictions.isna().any() or not np.isfinite(predictions).all():
        raise ValueError("Predictions must be finite and non-missing")
    if not predictions.between(0.0, 1.0).all():
        raise ValueError("Binary-class probabilities must be in [0, 1]")


def build_submission(
    sample: pd.DataFrame,
    probabilities,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build, validate, and optionally save a probability submission."""
    submission = sample.copy()
    submission[TARGET_COLUMN] = np.asarray(probabilities, dtype=float)
    validate_submission(submission, sample)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
    return submission
