"""Experiment 2: conservative cross-family tabular model comparison."""

from __future__ import annotations

import pickle
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

from .data import load_competition_data, split_features_target
from .modeling import build_baseline_model
from .submission import build_submission
from .validation import CVConfig, evaluate_predictions

SEED = 42


def prepare_native_categories(
    X: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Align pandas categorical dtypes without imputing native numeric missing values."""
    X = X.copy()
    X_test = X_test.copy()
    categorical = list(X.select_dtypes(exclude="number").columns)
    for column in categorical:
        values = pd.concat([X[column], X_test[column]], ignore_index=True).dropna().unique()
        dtype = pd.CategoricalDtype(categories=sorted(values))
        X[column] = X[column].astype(dtype)
        X_test[column] = X_test[column].astype(dtype)
    return X, X_test, categorical


def prepare_catboost(
    X: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Represent categorical missingness explicitly for CatBoost."""
    X = X.copy()
    X_test = X_test.copy()
    categorical = list(X.select_dtypes(exclude="number").columns)
    for column in categorical:
        X[column] = X[column].fillna("__MISSING__").astype(str)
        X_test[column] = X_test[column].fillna("__MISSING__").astype(str)
    return X, X_test, categorical


def build_models(categorical_columns: list[str]) -> dict[str, BaseEstimator]:
    """Return conservative, untuned representatives of five model families."""
    return {
        "LogisticRegression": build_baseline_model(random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            categorical_features="from_dtype",
            random_state=SEED,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=1.0,
            colsample_bytree=1.0,
            tree_method="hist",
            enable_categorical=True,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=SEED,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.1,
            num_leaves=31,
            verbosity=-1,
            n_jobs=-1,
            random_state=SEED,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            learning_rate=0.1,
            depth=6,
            loss_function="Logloss",
            eval_metric="AUC",
            cat_features=categorical_columns,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
            random_seed=SEED,
        ),
    }


def clone_model(model: BaseEstimator) -> BaseEstimator:
    """Clone sklearn estimators, with a safe fallback for CatBoost's list parameters."""
    try:
        return clone(model)
    except RuntimeError:
        return deepcopy(model)


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    X_native: pd.DataFrame,
    X_catboost: pd.DataFrame,
    categorical_columns: list[str],
    cv: CVConfig = CVConfig(),
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate every family on identical precomputed stratified splits."""
    splits = list(cv.splitter().split(X, y))
    models = build_models(categorical_columns)
    rows: list[dict[str, float | int | str]] = []
    predictions: dict[str, np.ndarray] = {}

    for name, model in models.items():
        model_X = X if name == "LogisticRegression" else X_catboost if name == "CatBoost" else X_native
        oof = np.zeros(len(y), dtype=float)
        for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
            fitted = clone_model(model)
            start = time.perf_counter()
            fitted.fit(model_X.iloc[train_idx], y.iloc[train_idx])
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            probabilities = fitted.predict_proba(model_X.iloc[valid_idx])[:, 1]
            predict_seconds = time.perf_counter() - start
            oof[valid_idx] = probabilities
            fold_scores = evaluate_predictions(y.iloc[valid_idx], probabilities)
            rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "fold_auc": fold_scores["roc_auc"],
                    "fold_log_loss": fold_scores["log_loss"],
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "model_size_mb": len(pickle.dumps(fitted, protocol=pickle.HIGHEST_PROTOCOL)) / 1_000_000,
                }
            )
        predictions[name] = oof

    report = pd.DataFrame(rows)
    aggregates: dict[str, dict[str, float]] = {}
    for name, oof in predictions.items():
        scores = evaluate_predictions(y, oof)
        model_rows = report[report["model"] == name]
        aggregates[name] = {
            "oof_auc": scores["roc_auc"],
            "oof_log_loss": scores["log_loss"],
            "oof_accuracy_0.5": scores["accuracy_0.5"],
            "mean_fold_auc": model_rows["fold_auc"].mean(),
            "std_fold_auc": model_rows["fold_auc"].std(ddof=1),
            "total_fit_seconds": model_rows["fit_seconds"].sum(),
            "total_predict_seconds": model_rows["predict_seconds"].sum(),
            "max_model_size_mb": model_rows["model_size_mb"].max(),
        }
    logistic_auc = aggregates["LogisticRegression"]["oof_auc"]
    for name, values in aggregates.items():
        values["delta_vs_logistic"] = values["oof_auc"] - logistic_auc
        for key, value in values.items():
            report.loc[report["model"] == name, key] = value
    return report, predictions


def fit_champion_and_importance(
    champion: str,
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[BaseEstimator, pd.DataFrame, float]:
    """Fit the selected tree family on all rows and return native importance."""
    model = clone_model(build_models(categorical_columns)[champion])
    start = time.perf_counter()
    model.fit(X, y)
    seconds = time.perf_counter() - start
    importance = pd.DataFrame(
        {"feature": X.columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False, ignore_index=True)
    importance["importance_normalized"] = importance["importance"] / importance["importance"].sum()
    return model, importance, seconds


def main() -> None:
    train, test, sample = load_competition_data()
    X, y, X_test = split_features_target(train, test)
    X_native, X_test_native, categorical = prepare_native_categories(X, X_test)
    X_catboost, X_test_catboost, _ = prepare_catboost(X, X_test)

    report, _ = compare_models(X, y, X_native, X_catboost, categorical)
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    report.to_csv(report_dir / "model_comparison.csv", index=False)

    summary = report.drop_duplicates("model").sort_values(
        ["oof_auc", "std_fold_auc", "total_fit_seconds"],
        ascending=[False, True, True],
    )
    champion = str(summary.iloc[0]["model"])
    logistic_auc = float(summary.loc[summary["model"] == "LogisticRegression", "oof_auc"].iloc[0])
    champion_auc = float(summary.iloc[0]["oof_auc"])
    if champion_auc <= logistic_auc:
        raise RuntimeError("No tree model beat logistic regression; no submission created")

    final_X = X_catboost if champion == "CatBoost" else X_native
    final_X_test = X_test_catboost if champion == "CatBoost" else X_test_native
    fitted, importance, final_fit_seconds = fit_champion_and_importance(
        champion, final_X, y, final_X_test, categorical
    )
    importance.to_csv(report_dir / "feature_importance.csv", index=False)
    start = time.perf_counter()
    test_probabilities = fitted.predict_proba(final_X_test)[:, 1]
    final_predict_seconds = time.perf_counter() - start
    build_submission(sample, test_probabilities, "submissions/submission_02_model.csv")
    print(summary[[
        "model", "oof_auc", "oof_log_loss", "mean_fold_auc", "std_fold_auc",
        "delta_vs_logistic", "total_fit_seconds", "total_predict_seconds", "max_model_size_mb",
    ]].to_string(index=False))
    print(f"\nChampion: {champion}; final fit {final_fit_seconds:.2f}s; test prediction {final_predict_seconds:.2f}s")
    print("\nFeature importance:\n", importance.to_string(index=False))


if __name__ == "__main__":
    main()
