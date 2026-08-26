"""Data loading and schema helpers."""

from pathlib import Path

import pandas as pd

TARGET_COLUMN = "addicted_label"
ID_COLUMN = "id"


def load_competition_data(data_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and sample submission and validate their core schema."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    required_train = {ID_COLUMN, TARGET_COLUMN}
    if not required_train.issubset(train.columns):
        raise ValueError(f"train.csv must contain {sorted(required_train)}")
    if ID_COLUMN not in test or list(sample.columns) != [ID_COLUMN, TARGET_COLUMN]:
        raise ValueError("Unexpected test or sample-submission schema")
    if train[ID_COLUMN].duplicated().any() or test[ID_COLUMN].duplicated().any():
        raise ValueError("IDs must be unique within train and test")
    if set(train[ID_COLUMN]).intersection(test[ID_COLUMN]):
        raise ValueError("Train and test IDs must not overlap")
    if list(train.drop(columns=TARGET_COLUMN).columns) != list(test.columns):
        raise ValueError("Train and test feature schemas differ")
    if len(sample) != len(test) or not sample[ID_COLUMN].equals(test[ID_COLUMN]):
        raise ValueError("Sample-submission IDs must exactly match test IDs and order")
    return train, test, sample


def split_features_target(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame | None]:
    """Return modeling features (excluding ID), target, and optional test features."""
    feature_columns = [c for c in train.columns if c not in {ID_COLUMN, TARGET_COLUMN}]
    X = train[feature_columns].copy()
    y = train[TARGET_COLUMN].astype("int8").copy()
    X_test = None if test is None else test[feature_columns].copy()
    return X, y, X_test
