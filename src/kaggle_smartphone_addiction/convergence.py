"""Experiment 6: boosting-round convergence at lower learning rates."""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from .boosting_tuning import (
    E4_BLEND_AUC,
    FEATURES,
    LIGHTGBM_BASE,
    SELECTED_LIGHTGBM,
    SELECTED_XGBOOST,
    XGBOOST_BASE,
)
from .data import load_competition_data, split_features_target
from .feature_diagnostics import add_engineered_features
from .feature_diagnostics import paired_bootstrap_difference
from .blending import diversity_statistics
from .model_comparison import clone_model, prepare_native_categories
from .submission import build_submission
from .validation import CVConfig, evaluate_predictions

SEED = 42
EARLY_STOPPING_ROUNDS = 150
E5_LIGHTGBM_AUC = 0.9632233232633376
E5_XGBOOST_AUC = 0.9637048560968332
E5_BLEND_AUC = 0.9639433758194731
SELECTED_SCHEDULE = "lr_002_max_5000"
RECOMMENDED_LIGHTGBM_WEIGHT = 0.4

SCHEDULES = {
    "lr_005_max_2500": {"learning_rate": 0.05, "n_estimators": 2500},
    "lr_003_max_3500": {"learning_rate": 0.03, "n_estimators": 3500},
    "lr_002_max_5000": {"learning_rate": 0.02, "n_estimators": 5000},
}


def make_schedule_model(family: str, schedule: dict):
    """Keep E5 tree structure fixed and change only rate/round schedule."""
    if family == "LightGBM":
        params = LIGHTGBM_BASE.copy()
        params.update({"num_leaves": 63, **schedule})
        return LGBMClassifier(**params)
    params = XGBOOST_BASE.copy()
    params.update(
        {
            "max_depth": 7,
            "eval_metric": "auc",
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "verbosity": 0,
            **schedule,
        }
    )
    return XGBClassifier(**params)


def evaluate_schedule(family: str, model, X: pd.DataFrame, y: pd.Series, splits):
    """Evaluate a schedule with fold-local AUC early stopping."""
    oof = np.zeros(len(y), dtype=float)
    fold_aucs = []
    iterations = []
    fit_seconds = 0.0
    predict_seconds = 0.0
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        print(f"  fold {fold}", flush=True)
        fitted = clone_model(model)
        if family == "LightGBM":
            fit_kwargs = {
                "eval_X": X.iloc[valid_idx],
                "eval_y": y.iloc[valid_idx],
                "eval_metric": "auc",
                "callbacks": [
                    lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                    lgb.log_evaluation(0),
                ],
            }
        else:
            fit_kwargs = {
                "eval_set": [(X.iloc[valid_idx], y.iloc[valid_idx])],
                "verbose": False,
            }
        start = time.perf_counter()
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx], **fit_kwargs)
        fit_seconds += time.perf_counter() - start
        start = time.perf_counter()
        probability = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
        predict_seconds += time.perf_counter() - start
        oof[valid_idx] = probability
        fold_aucs.append(float(roc_auc_score(y.iloc[valid_idx], probability)))
        if family == "LightGBM":
            iterations.append(int(fitted.best_iteration_))
        else:
            iterations.append(int(fitted.best_iteration) + 1)
    return oof, fold_aucs, iterations, fit_seconds, predict_seconds


def schedule_row(
    family: str,
    name: str,
    schedule: dict,
    oof: np.ndarray,
    y: pd.Series,
    fold_aucs: list[float],
    iterations: list[int],
    baseline_auc: float,
    fit_seconds: float,
    predict_seconds: float,
) -> dict[str, object]:
    scores = evaluate_predictions(y, oof)
    maximum = int(schedule["n_estimators"])
    row: dict[str, object] = {
        "family": family,
        "schedule": name,
        "learning_rate": schedule["learning_rate"],
        "max_estimators": maximum,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "oof_auc": scores["roc_auc"],
        "log_loss": scores["log_loss"],
        "delta_vs_e5_component": scores["roc_auc"] - baseline_auc,
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "std_fold_auc": float(np.std(fold_aucs, ddof=1)),
        "mean_best_iteration": float(np.mean(iterations)),
        "median_best_iteration": float(np.median(iterations)),
        "min_best_iteration": int(np.min(iterations)),
        "max_best_iteration": int(np.max(iterations)),
        "folds_hit_max": int(sum(i >= maximum for i in iterations)),
        "folds_without_full_stopping_window": int(
            sum((maximum - i) < EARLY_STOPPING_ROUNDS for i in iterations)
        ),
        "all_folds_completed_early_stopping": all(
            (maximum - i) >= EARLY_STOPPING_ROUNDS for i in iterations
        ),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }
    for fold, (auc, iteration) in enumerate(zip(fold_aucs, iterations), start=1):
        row[f"fold_{fold}_auc"] = auc
        row[f"fold_{fold}_best_iteration"] = iteration
    return row


def run_schedule_search(
    family: str,
    X: pd.DataFrame,
    y: pd.Series,
    splits,
    baseline_auc: float,
    report_path: Path,
    oof_path: Path,
) -> None:
    """Run or resume all three schedules for one model family."""
    rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}
    if report_path.exists() and oof_path.exists():
        existing = pd.read_csv(report_path)
        if "folds_without_full_stopping_window" not in existing:
            existing["folds_without_full_stopping_window"] = existing.apply(
                lambda row: sum(
                    (row["max_estimators"] - row[f"fold_{fold}_best_iteration"])
                    < row["early_stopping_rounds"]
                    for fold in range(1, 6)
                ),
                axis=1,
            )
            existing["all_folds_completed_early_stopping"] = (
                existing["folds_without_full_stopping_window"] == 0
            )
            existing.to_csv(report_path, index=False)
        saved = np.load(oof_path)
        for _, row in existing.iterrows():
            rows.append(row.to_dict())
        predictions = {name: saved[name] for name in saved.files}
    complete = {str(row["schedule"]) for row in rows}
    for name, schedule in SCHEDULES.items():
        if name in complete:
            print(f"{family}: reuse {name}", flush=True)
            continue
        print(f"{family}: {name}", flush=True)
        model = make_schedule_model(family, schedule)
        oof, fold_aucs, iterations, fit_time, predict_time = evaluate_schedule(
            family, model, X, y, splits
        )
        predictions[name] = oof
        rows.append(
            schedule_row(
                family, name, schedule, oof, y, fold_aucs, iterations,
                baseline_auc, fit_time, predict_time
            )
        )
        pd.DataFrame(rows).to_csv(report_path, index=False)
        np.savez_compressed(oof_path, **predictions)


def finalize_experiment(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    sample: pd.DataFrame,
    report_dir: Path,
) -> None:
    """Compare selected schedules, quantify diversity, and conditionally submit."""
    e5 = pd.read_csv(report_dir / "oof_predictions_exp05.csv")
    lgb_oof = np.load(report_dir / "lightgbm_convergence_oof.npz")[SELECTED_SCHEDULE]
    xgb_oof = np.load(report_dir / "xgboost_convergence_oof.npz")[SELECTED_SCHEDULE]
    e5_lgb = e5["tuned_lightgbm_oof"].to_numpy()
    e5_xgb = e5["tuned_xgboost_oof"].to_numpy()
    e5_blend = 0.5 * (e5_lgb + e5_xgb)

    blend_rows = []
    for weight in np.linspace(0.0, 1.0, 11):
        probability = weight * lgb_oof + (1.0 - weight) * xgb_oof
        scores = evaluate_predictions(y, probability)
        blend_rows.append(
            {
                "lightgbm_weight": weight,
                "xgboost_weight": 1.0 - weight,
                "oof_auc": scores["roc_auc"],
                "log_loss": scores["log_loss"],
                "delta_vs_e5_blend": scores["roc_auc"] - E5_BLEND_AUC,
            }
        )
    blend_report = pd.DataFrame(blend_rows)
    blend_report.to_csv(report_dir / "convergence_blend_results.csv", index=False)
    proposed = (
        RECOMMENDED_LIGHTGBM_WEIGHT * lgb_oof
        + (1.0 - RECOMMENDED_LIGHTGBM_WEIGHT) * xgb_oof
    )
    bootstrap_samples, bootstrap_summary = paired_bootstrap_difference(
        y, proposed, e5_blend
    )
    bootstrap_samples.to_csv(
        report_dir / "convergence_bootstrap_samples.csv", index=False
    )

    lgb_report = pd.read_csv(report_dir / "lightgbm_convergence.csv")
    xgb_report = pd.read_csv(report_dir / "xgboost_convergence.csv")
    selected_lgb = lgb_report[lgb_report["schedule"] == SELECTED_SCHEDULE].iloc[0]
    selected_xgb = xgb_report[xgb_report["schedule"] == SELECTED_SCHEDULE].iloc[0]
    lgb_rounds = int(selected_lgb["median_best_iteration"])
    xgb_rounds = int(selected_xgb["median_best_iteration"])
    analysis = {
        "selected_schedule": SELECTED_SCHEDULE,
        "full_fit_lightgbm_rounds": lgb_rounds,
        "full_fit_xgboost_rounds": xgb_rounds,
        "recommended_blend": {
            "lightgbm_weight": RECOMMENDED_LIGHTGBM_WEIGHT,
            "xgboost_weight": 1.0 - RECOMMENDED_LIGHTGBM_WEIGHT,
            "oof_auc": float(roc_auc_score(y, proposed)),
            "delta_vs_e5_blend": float(roc_auc_score(y, proposed) - E5_BLEND_AUC),
        },
        "bootstrap_vs_e5_blend": bootstrap_summary,
        "prediction_diversity": {
            "lightgbm_e5_vs_e6": diversity_statistics(y, e5_lgb, lgb_oof),
            "xgboost_e5_vs_e6": diversity_statistics(y, e5_xgb, xgb_oof),
            "blend_e5_vs_e6": diversity_statistics(y, e5_blend, proposed),
        },
    }
    (report_dir / "convergence_analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n"
    )
    pd.DataFrame(
        {
            "target": y,
            "e5_lightgbm_oof": e5_lgb,
            "e6_lightgbm_oof": lgb_oof,
            "e5_xgboost_oof": e5_xgb,
            "e6_xgboost_oof": xgb_oof,
            "e5_blend_oof": e5_blend,
            "e6_blend_oof": proposed,
        }
    ).to_csv(report_dir / "oof_predictions_exp06.csv", index=False)

    robust = bootstrap_summary["ci_95_lower"] > 0
    if robust:
        lgb_params = LIGHTGBM_BASE.copy()
        lgb_params.update(
            {"num_leaves": 63, "learning_rate": 0.02, "n_estimators": lgb_rounds}
        )
        xgb_params = XGBOOST_BASE.copy()
        xgb_params.update(
            {"max_depth": 7, "learning_rate": 0.02, "n_estimators": xgb_rounds}
        )
        fitted_lgb = LGBMClassifier(**lgb_params).fit(X, y)
        fitted_xgb = XGBClassifier(**xgb_params).fit(X, y)
        test_probability = (
            RECOMMENDED_LIGHTGBM_WEIGHT * fitted_lgb.predict_proba(X_test)[:, 1]
            + (1.0 - RECOMMENDED_LIGHTGBM_WEIGHT)
            * fitted_xgb.predict_proba(X_test)[:, 1]
        )
        test_probability = np.clip(test_probability, 0.0, 1.0)
        build_submission(
            sample, test_probability, "submissions/submission_06_convergence.csv"
        )
    print(json.dumps(analysis, indent=2))
    print("Submission generated:", robust)


def main() -> None:
    train, test, sample = load_competition_data()
    X, y, X_test = split_features_target(train, test)
    X_native, X_test_native, _ = prepare_native_categories(X, X_test)
    X_engineered = add_engineered_features(X_native, FEATURES)
    X_test_engineered = add_engineered_features(X_test_native, FEATURES)
    splits = list(CVConfig().splitter().split(X_engineered, y))
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    run_schedule_search(
        "LightGBM", X_engineered, y, splits, E5_LIGHTGBM_AUC,
        report_dir / "lightgbm_convergence.csv",
        report_dir / "lightgbm_convergence_oof.npz",
    )
    run_schedule_search(
        "XGBoost", X_engineered, y, splits, E5_XGBOOST_AUC,
        report_dir / "xgboost_convergence.csv",
        report_dir / "xgboost_convergence_oof.npz",
    )
    finalize_experiment(
        X_engineered, X_test_engineered, y, sample, report_dir
    )


if __name__ == "__main__":
    main()
