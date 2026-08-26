"""Experiment 3: paired OOF diversity, blending, and bootstrap inference."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

from .data import ID_COLUMN, load_competition_data, split_features_target
from .model_comparison import build_models, clone_model, prepare_native_categories
from .submission import build_submission
from .validation import CVConfig, evaluate_predictions

SEED = 42
BOOTSTRAP_RESAMPLES = 500


def generate_paired_oof(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_columns: list[str],
    cv: CVConfig = CVConfig(),
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Generate paired LightGBM/XGBoost OOF probabilities on identical folds."""
    models = build_models(categorical_columns)
    selected = {"lightgbm": models["LightGBM"], "xgboost": models["XGBoost"]}
    splits = list(cv.splitter().split(X, y))
    result = pd.DataFrame(index=X.index)
    result["fold"] = 0
    timings: dict[str, dict[str, float]] = {}

    for fold, (_, valid_idx) in enumerate(splits, start=1):
        result.loc[valid_idx, "fold"] = fold
    for name, model in selected.items():
        oof = np.zeros(len(y), dtype=float)
        fit_seconds = 0.0
        predict_seconds = 0.0
        for train_idx, valid_idx in splits:
            fitted = clone_model(model)
            start = time.perf_counter()
            fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
            fit_seconds += time.perf_counter() - start
            start = time.perf_counter()
            oof[valid_idx] = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
            predict_seconds += time.perf_counter() - start
        result[f"{name}_oof"] = oof
        timings[name] = {
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
        }
    return result, timings


def diversity_statistics(y, lightgbm, xgboost) -> dict[str, object]:
    """Summarize paired probability and threshold-level disagreement."""
    y = np.asarray(y)
    lightgbm = np.asarray(lightgbm)
    xgboost = np.asarray(xgboost)
    absolute_difference = np.abs(lightgbm - xgboost)
    class_disagreement = (lightgbm >= 0.5) != (xgboost >= 0.5)
    quantiles = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    return {
        "pearson_correlation": float(pearsonr(lightgbm, xgboost).statistic),
        "spearman_correlation": float(spearmanr(lightgbm, xgboost).statistic),
        "mean_absolute_probability_difference": float(absolute_difference.mean()),
        "absolute_difference_quantiles": {
            str(q): float(np.quantile(absolute_difference, q)) for q in quantiles
        },
        "class_disagreement_rate": float(class_disagreement.mean()),
        "class_disagreement_rate_y0": float(class_disagreement[y == 0].mean()),
        "class_disagreement_rate_y1": float(class_disagreement[y == 1].mean()),
        "mean_absolute_difference_y0": float(absolute_difference[y == 0].mean()),
        "mean_absolute_difference_y1": float(absolute_difference[y == 1].mean()),
    }


def evaluate_blends(y, lightgbm, xgboost) -> pd.DataFrame:
    """Evaluate LightGBM weights 0.0 through 1.0 in increments of 0.1."""
    lightgbm_scores = evaluate_predictions(y, lightgbm)
    xgboost_scores = evaluate_predictions(y, xgboost)
    rows = []
    for weight in np.linspace(0.0, 1.0, 11):
        probability = weight * lightgbm + (1.0 - weight) * xgboost
        scores = evaluate_predictions(y, probability)
        rows.append(
            {
                "lightgbm_weight": weight,
                "xgboost_weight": 1.0 - weight,
                "oof_auc": scores["roc_auc"],
                "log_loss": scores["log_loss"],
                "delta_auc_vs_lightgbm": scores["roc_auc"] - lightgbm_scores["roc_auc"],
                "delta_auc_vs_xgboost": scores["roc_auc"] - xgboost_scores["roc_auc"],
            }
        )
    return pd.DataFrame(rows)


def paired_stratified_bootstrap(
    y,
    lightgbm,
    xgboost,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Bootstrap paired AUC differences while preserving observed class counts."""
    y = np.asarray(y)
    lightgbm = np.asarray(lightgbm)
    xgboost = np.asarray(xgboost)
    blend = 0.5 * (lightgbm + xgboost)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)
    rows = []
    for resample in range(1, n_resamples + 1):
        sample = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        sample_y = y[sample]
        auc_blend = roc_auc_score(sample_y, blend[sample])
        auc_lightgbm = roc_auc_score(sample_y, lightgbm[sample])
        auc_xgboost = roc_auc_score(sample_y, xgboost[sample])
        rows.append(
            {
                "resample": resample,
                "blend_minus_lightgbm": auc_blend - auc_lightgbm,
                "blend_minus_xgboost": auc_blend - auc_xgboost,
            }
        )
    samples = pd.DataFrame(rows)
    summary = {}
    for column in ["blend_minus_lightgbm", "blend_minus_xgboost"]:
        values = samples[column]
        summary[column] = {
            "mean": float(values.mean()),
            "ci_95_lower": float(values.quantile(0.025)),
            "ci_95_upper": float(values.quantile(0.975)),
            "proportion_positive": float((values > 0).mean()),
            "n_resamples": n_resamples,
            "seed": seed,
        }
    return samples, summary


def main() -> None:
    train, test, sample = load_competition_data()
    X, y, X_test = split_features_target(train, test)
    X_native, X_test_native, categorical = prepare_native_categories(X, X_test)

    oof, timings = generate_paired_oof(X_native, y, categorical, CVConfig())
    oof.insert(0, ID_COLUMN, train[ID_COLUMN].to_numpy())
    oof.insert(1, "target", y.to_numpy())
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    oof.to_csv(report_dir / "oof_predictions_exp03.csv", index=False)

    lightgbm_oof = oof["lightgbm_oof"].to_numpy()
    xgboost_oof = oof["xgboost_oof"].to_numpy()
    blends = evaluate_blends(y, lightgbm_oof, xgboost_oof)
    blends.to_csv(report_dir / "blend_results.csv", index=False)
    diversity = diversity_statistics(y, lightgbm_oof, xgboost_oof)
    bootstrap_samples, bootstrap_summary = paired_stratified_bootstrap(
        y, lightgbm_oof, xgboost_oof
    )
    bootstrap_samples.to_csv(report_dir / "blend_bootstrap_samples.csv", index=False)
    analysis = {
        "timings": timings,
        "diversity": diversity,
        "bootstrap": bootstrap_summary,
    }
    (report_dir / "blend_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")

    fifty = blends.loc[np.isclose(blends["lightgbm_weight"], 0.5)].iloc[0]
    convincing = (
        fifty["delta_auc_vs_lightgbm"] > 0
        and fifty["delta_auc_vs_xgboost"] > 0
        and bootstrap_summary["blend_minus_lightgbm"]["ci_95_lower"] > 0
        and bootstrap_summary["blend_minus_xgboost"]["ci_95_lower"] > 0
    )
    if convincing:
        models = build_models(categorical)
        fitted_lightgbm = clone_model(models["LightGBM"]).fit(X_native, y)
        fitted_xgboost = clone_model(models["XGBoost"]).fit(X_native, y)
        test_probability = 0.5 * (
            fitted_lightgbm.predict_proba(X_test_native)[:, 1]
            + fitted_xgboost.predict_proba(X_test_native)[:, 1]
        )
        build_submission(sample, test_probability, "submissions/submission_03_blend.csv")

    print(blends.to_string(index=False))
    print("\nDiversity:\n", json.dumps(diversity, indent=2))
    print("\nBootstrap:\n", json.dumps(bootstrap_summary, indent=2))
    print("\nSubmission generated:", convincing)


if __name__ == "__main__":
    main()
