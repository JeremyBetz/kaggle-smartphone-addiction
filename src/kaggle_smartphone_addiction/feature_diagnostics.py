"""Experiment 4: feature ablation, missingness, and conservative engineering."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import ID_COLUMN, load_competition_data, split_features_target
from .model_comparison import build_models, clone_model, prepare_native_categories
from .submission import build_submission
from .validation import CVConfig, evaluate_predictions

SEED = 42
MEANINGFUL_MISSINGNESS_DELTA = 0.003
CURRENT_CHAMPION_AUC = 0.9610442845867228
BOOTSTRAP_RESAMPLES = 500

RATIO_FEATURES = {
    "notifications_per_open",
    "opens_per_screen_hour",
    "notifications_per_screen_hour",
    "weekend_vs_daily_screen",
    "entertainment_share_of_screen",
}
TIME_COMPOSITION_FEATURES = {
    "total_entertainment_hours",
    "leisure_screen_proxy",
    "weekend_daily_screen_gap",
    "entertainment_share_of_screen",
}


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while mapping zero denominators and infinities to missing."""
    result = numerator / denominator.mask(denominator == 0)
    return result.replace([np.inf, -np.inf], np.nan)


def engineered_feature_values(X: pd.DataFrame) -> dict[str, pd.Series]:
    """Return eight small, interpretable behavioral candidates."""
    entertainment = X["social_media_hours"] + X["gaming_hours"]
    return {
        "notifications_per_open": safe_ratio(
            X["notifications_per_day"], X["app_opens_per_day"]
        ),
        "opens_per_screen_hour": safe_ratio(
            X["app_opens_per_day"], X["daily_screen_time_hours"]
        ),
        "notifications_per_screen_hour": safe_ratio(
            X["notifications_per_day"], X["daily_screen_time_hours"]
        ),
        "weekend_vs_daily_screen": safe_ratio(
            X["weekend_screen_time"], X["daily_screen_time_hours"]
        ),
        "total_entertainment_hours": entertainment,
        # Proxy assumes work_study_hours is the work/study component of screen time.
        "leisure_screen_proxy": X["daily_screen_time_hours"] - X["work_study_hours"],
        "weekend_daily_screen_gap": X["weekend_screen_time"] - X["daily_screen_time_hours"],
        "entertainment_share_of_screen": safe_ratio(
            entertainment, X["daily_screen_time_hours"]
        ),
    }


def add_engineered_features(X: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Add a selected subset of the declared candidate features."""
    result = X.copy()
    values = engineered_feature_values(X)
    unknown = set(names) - set(values)
    if unknown:
        raise ValueError(f"Unknown engineered features: {sorted(unknown)}")
    for name in names:
        result[name] = values[name]
    return result


def add_missing_indicators(X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Add explicit uint8 missingness indicators for selected original features."""
    result = X.copy()
    for feature in features:
        result[f"{feature}__missing"] = X[feature].isna().astype("uint8")
    return result


def missingness_diagnostics(
    X: pd.DataFrame, X_test: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    """Describe split missingness and target rates when present versus missing."""
    rows = []
    for feature in X.columns:
        missing = X[feature].isna()
        missing_rate = float(y[missing].mean())
        present_rate = float(y[~missing].mean())
        delta = missing_rate - present_rate
        rows.append(
            {
                "feature": feature,
                "train_missing_rate": float(missing.mean()),
                "test_missing_rate": float(X_test[feature].isna().mean()),
                "test_minus_train_missing_rate": float(
                    X_test[feature].isna().mean() - missing.mean()
                ),
                "target_rate_missing": missing_rate,
                "target_rate_present": present_rate,
                "target_rate_missing_minus_present": delta,
                "absolute_target_rate_delta": abs(delta),
                "meaningful_at_0.003": abs(delta) >= MEANINGFUL_MISSINGNESS_DELTA,
                "n_missing_train": int(missing.sum()),
            }
        )
    return pd.DataFrame(rows)


def generate_oof(model, X: pd.DataFrame, y: pd.Series, splits) -> tuple[np.ndarray, list[float], float, float]:
    """Generate OOF probabilities and fold AUCs for fixed precomputed splits."""
    oof = np.zeros(len(y), dtype=float)
    fold_aucs = []
    fit_seconds = 0.0
    predict_seconds = 0.0
    for train_idx, valid_idx in splits:
        fitted = clone_model(model)
        start = time.perf_counter()
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        fit_seconds += time.perf_counter() - start
        start = time.perf_counter()
        probability = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
        predict_seconds += time.perf_counter() - start
        oof[valid_idx] = probability
        fold_aucs.append(float(roc_auc_score(y.iloc[valid_idx], probability)))
    return oof, fold_aucs, fit_seconds, predict_seconds


def result_row(
    experiment: str,
    model: str,
    oof: np.ndarray,
    y: pd.Series,
    fold_aucs: list[float],
    control_auc: float,
    fit_seconds: float,
    predict_seconds: float,
    features: list[str] | None = None,
) -> dict[str, object]:
    """Build a consistent report row for an OOF experiment."""
    scores = evaluate_predictions(y, oof)
    row: dict[str, object] = {
        "experiment": experiment,
        "model": model,
        "features": "|".join(features or []),
        "n_added_features": len(features or []),
        "oof_auc": scores["roc_auc"],
        "log_loss": scores["log_loss"],
        "delta_vs_raw_lightgbm": scores["roc_auc"] - control_auc,
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "std_fold_auc": float(np.std(fold_aucs, ddof=1)),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }
    for index, auc in enumerate(fold_aucs, start=1):
        row[f"fold_{index}_auc"] = auc
    return row


def fold_aucs_from_saved(oof: pd.DataFrame, column: str) -> list[float]:
    return [
        float(roc_auc_score(group["target"], group[column]))
        for _, group in oof.groupby("fold", sort=True)
    ]


def paired_bootstrap_difference(
    y,
    candidate,
    champion,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Paired stratified bootstrap of candidate-minus-champion AUC."""
    y = np.asarray(y)
    candidate = np.asarray(candidate)
    champion = np.asarray(champion)
    negative = np.flatnonzero(y == 0)
    positive = np.flatnonzero(y == 1)
    rng = np.random.default_rng(seed)
    differences = []
    for resample in range(1, n_resamples + 1):
        sample = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        difference = roc_auc_score(y[sample], candidate[sample]) - roc_auc_score(
            y[sample], champion[sample]
        )
        differences.append({"resample": resample, "auc_difference": difference})
    samples = pd.DataFrame(differences)
    summary: dict[str, float | int] = {
        "mean": float(samples["auc_difference"].mean()),
        "ci_95_lower": float(samples["auc_difference"].quantile(0.025)),
        "ci_95_upper": float(samples["auc_difference"].quantile(0.975)),
        "proportion_positive": float((samples["auc_difference"] > 0).mean()),
        "n_resamples": n_resamples,
        "seed": seed,
    }
    return samples, summary


def main() -> None:
    train, test, sample = load_competition_data()
    X, y, X_test = split_features_target(train, test)
    X_native, X_test_native, categorical = prepare_native_categories(X, X_test)
    models = build_models(categorical)
    lightgbm = models["LightGBM"]
    xgboost = models["XGBoost"]
    splits = list(CVConfig().splitter().split(X_native, y))
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    previous_oof = pd.read_csv(report_dir / "oof_predictions_exp03.csv")
    if not previous_oof[ID_COLUMN].equals(train[ID_COLUMN]):
        raise ValueError("Experiment 3 OOF IDs do not match training IDs")
    raw_lgb_oof = previous_oof["lightgbm_oof"].to_numpy()
    raw_xgb_oof = previous_oof["xgboost_oof"].to_numpy()
    raw_blend_oof = 0.5 * (raw_lgb_oof + raw_xgb_oof)
    raw_lgb_auc = float(roc_auc_score(y, raw_lgb_oof))

    # Part 1: full control and leave-one-feature-out ablations.
    ablation_rows = [
        result_row(
            "full_raw_control",
            "LightGBM",
            raw_lgb_oof,
            y,
            fold_aucs_from_saved(previous_oof, "lightgbm_oof"),
            raw_lgb_auc,
            0.0,
            0.0,
        )
    ]
    for feature in X_native.columns:
        print(f"Ablation: {feature}", flush=True)
        ablated = X_native.drop(columns=feature)
        oof, fold_aucs, fit_time, predict_time = generate_oof(
            lightgbm, ablated, y, splits
        )
        ablation_rows.append(
            result_row(
                f"drop_{feature}", "LightGBM", oof, y, fold_aucs,
                raw_lgb_auc, fit_time, predict_time, [feature]
            )
        )
        pd.DataFrame(ablation_rows).to_csv(report_dir / "feature_ablation.csv", index=False)

    # Part 2: missingness diagnostics and explicit indicator tests.
    missingness = missingness_diagnostics(X, X_test, y)
    missingness.to_csv(report_dir / "missingness_diagnostics.csv", index=False)
    meaningful = missingness.loc[missingness["meaningful_at_0.003"], "feature"].tolist()
    experiment_rows = [
        result_row(
            "raw_control_no_explicit_indicators", "LightGBM", raw_lgb_oof, y,
            fold_aucs_from_saved(previous_oof, "lightgbm_oof"), raw_lgb_auc, 0.0, 0.0
        )
    ]
    for experiment, indicator_features in [
        ("missing_indicators_all", list(X.columns)),
        ("missing_indicators_meaningful", meaningful),
    ]:
        print(f"Missingness: {experiment} ({len(indicator_features)} indicators)", flush=True)
        diagnostic_X = add_missing_indicators(X_native, indicator_features)
        oof, fold_aucs, fit_time, predict_time = generate_oof(
            lightgbm, diagnostic_X, y, splits
        )
        experiment_rows.append(
            result_row(
                experiment, "LightGBM", oof, y, fold_aucs, raw_lgb_auc,
                fit_time, predict_time, indicator_features
            )
        )
    pd.DataFrame(experiment_rows).to_csv(
        report_dir / "missingness_experiments.csv", index=False
    )

    # Part 3: each engineered feature alone.
    candidate_names = list(engineered_feature_values(X_native))
    individual_oof: dict[str, np.ndarray] = {}
    for feature in candidate_names:
        print(f"Engineered feature: {feature}", flush=True)
        engineered_X = add_engineered_features(X_native, [feature])
        oof, fold_aucs, fit_time, predict_time = generate_oof(
            lightgbm, engineered_X, y, splits
        )
        individual_oof[feature] = oof
        experiment_rows.append(
            result_row(
                f"individual_{feature}", "LightGBM", oof, y, fold_aucs,
                raw_lgb_auc, fit_time, predict_time, [feature]
            )
        )
        pd.DataFrame(experiment_rows).to_csv(
            report_dir / "feature_engineering.csv", index=False
        )

    individual_report = pd.DataFrame(experiment_rows)
    positive = individual_report.loc[
        individual_report["experiment"].str.startswith("individual_")
        & (individual_report["delta_vs_raw_lightgbm"] > 0)
    ].sort_values("oof_auc", ascending=False)
    positive_names = positive["features"].tolist()
    top_three = positive_names[:3]
    proposed = [
        ("combination_all_positive", positive_names),
        ("combination_top_three", top_three),
        ("combination_positive_ratios", [f for f in positive_names if f in RATIO_FEATURES]),
        ("combination_positive_time_composition", [f for f in positive_names if f in TIME_COMPOSITION_FEATURES]),
    ]
    seen: set[tuple[str, ...]] = set()
    for experiment, features in proposed:
        key = tuple(sorted(features))
        if not features or key in seen:
            continue
        seen.add(key)
        print(f"Combination: {experiment} ({features})", flush=True)
        engineered_X = add_engineered_features(X_native, features)
        oof, fold_aucs, fit_time, predict_time = generate_oof(
            lightgbm, engineered_X, y, splits
        )
        experiment_rows.append(
            result_row(
                experiment, "LightGBM", oof, y, fold_aucs, raw_lgb_auc,
                fit_time, predict_time, features
            )
        )
        pd.DataFrame(experiment_rows).to_csv(
            report_dir / "feature_engineering.csv", index=False
        )

    engineering_report = pd.DataFrame(experiment_rows)
    engineered_only = engineering_report[
        engineering_report["experiment"].str.startswith(("individual_", "combination_"))
    ]
    best = engineered_only.sort_values("oof_auc", ascending=False).iloc[0]
    best_features = str(best["features"]).split("|") if best["features"] else []
    best_lgb_X = add_engineered_features(X_native, best_features)
    # Reuse an individual OOF only when the winner is a single feature; otherwise regenerate
    # once here so the robustness artifact remains self-contained and explicit.
    if len(best_features) == 1 and best_features[0] in individual_oof:
        best_lgb_oof = individual_oof[best_features[0]]
        best_lgb_folds = [
            float(roc_auc_score(y.iloc[v], best_lgb_oof[v])) for _, v in splits
        ]
    else:
        best_lgb_oof, best_lgb_folds, _, _ = generate_oof(
            lightgbm, best_lgb_X, y, splits
        )

    # Part 4: transfer the exact selected feature set to unchanged XGBoost.
    print(f"XGBoost transfer: {best_features}", flush=True)
    best_xgb_oof, best_xgb_folds, xgb_fit, xgb_predict = generate_oof(
        xgboost, best_lgb_X, y, splits
    )
    best_blend_oof = 0.5 * (best_lgb_oof + best_xgb_oof)
    blend_folds = [
        float(roc_auc_score(y.iloc[v], best_blend_oof[v])) for _, v in splits
    ]
    robustness_rows = [
        result_row(
            "selected_feature_set", "LightGBM", best_lgb_oof, y,
            best_lgb_folds, raw_lgb_auc, 0.0, 0.0, best_features
        ),
        result_row(
            "selected_feature_set", "XGBoost", best_xgb_oof, y,
            best_xgb_folds, raw_lgb_auc, xgb_fit, xgb_predict, best_features
        ),
        result_row(
            "selected_feature_set", "Blend50_50", best_blend_oof, y,
            blend_folds, raw_lgb_auc, 0.0, 0.0, best_features
        ),
    ]
    robustness = pd.DataFrame(robustness_rows)
    robustness["delta_vs_current_champion"] = robustness["oof_auc"] - CURRENT_CHAMPION_AUC
    robustness.to_csv(report_dir / "feature_robustness.csv", index=False)
    pd.DataFrame(
        {
            ID_COLUMN: train[ID_COLUMN],
            "target": y,
            "fold": previous_oof["fold"],
            "engineered_lightgbm_oof": best_lgb_oof,
            "engineered_xgboost_oof": best_xgb_oof,
            "engineered_blend_oof": best_blend_oof,
        }
    ).to_csv(report_dir / "oof_predictions_exp04.csv", index=False)

    bootstrap_samples, bootstrap_summary = paired_bootstrap_difference(
        y, best_blend_oof, raw_blend_oof
    )
    bootstrap_samples.to_csv(
        report_dir / "feature_robustness_bootstrap_samples.csv", index=False
    )
    (report_dir / "feature_robustness_analysis.json").write_text(
        json.dumps(
            {
                "selected_features": best_features,
                "missingness_meaningful_features": meaningful,
                "bootstrap_vs_current_champion": bootstrap_summary,
            },
            indent=2,
        )
        + "\n"
    )

    blend_auc = float(roc_auc_score(y, best_blend_oof))
    transfer = (
        roc_auc_score(y, best_lgb_oof) > raw_lgb_auc
        and roc_auc_score(y, best_xgb_oof) > roc_auc_score(y, raw_xgb_oof)
    )
    convincing = (
        blend_auc > CURRENT_CHAMPION_AUC
        and transfer
        and bootstrap_summary["ci_95_lower"] > 0
    )
    if convincing:
        best_test_X = add_engineered_features(X_test_native, best_features)
        fitted_lgb = clone_model(lightgbm).fit(best_lgb_X, y)
        fitted_xgb = clone_model(xgboost).fit(best_lgb_X, y)
        test_probability = 0.5 * (
            fitted_lgb.predict_proba(best_test_X)[:, 1]
            + fitted_xgb.predict_proba(best_test_X)[:, 1]
        )
        build_submission(sample, test_probability, "submissions/submission_04_features.csv")

    print("\nBest features:", best_features)
    print(robustness.to_string(index=False))
    print("\nBootstrap:", json.dumps(bootstrap_summary, indent=2))
    print("Submission generated:", convincing)


if __name__ == "__main__":
    main()
