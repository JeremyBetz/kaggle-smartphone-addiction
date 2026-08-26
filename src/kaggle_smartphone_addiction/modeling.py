"""Model factories and final fitting."""

from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_baseline_model(random_state: int = 42) -> Pipeline:
    """A transparent linear baseline for mixed numeric/categorical data."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, make_column_selector(dtype_include="number")),
            ("categorical", categorical, make_column_selector(dtype_exclude="number")),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(C=1.0, max_iter=500, random_state=random_state),
            ),
        ]
    )


def fit_final_model(model: BaseEstimator, X, y) -> BaseEstimator:
    """Clone and fit a model on all available labeled rows."""
    fitted = clone(model)
    return fitted.fit(X, y)
