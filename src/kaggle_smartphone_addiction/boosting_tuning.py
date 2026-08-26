"""Experiment 5: conservative LightGBM/XGBoost tuning and simple blending."""

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

from .data import ID_COLUMN, load_competition_data, split_features_target
from .feature_diagnostics import add_engineered_features, paired_bootstrap_difference
from .model_comparison import clone_model, prepare_native_categories
from .submission import build_submission
from .validation import CVConfig, evaluate_predictions

SEED = 42
FEATURES = ["leisure_screen_proxy"]
E4_LIGHTGBM_AUC = 0.9607973491608738
E4_XGBOOST_AUC = 0.9609049414923644
E4_BLEND_AUC = 0.9613259226414592
SELECTED_LIGHTGBM = "wide_leaves_63"
SELECTED_XGBOOST = "depth_7"

LIGHTGBM_BASE = {
    "n_estimators": 300,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": SEED,
}

XGBOOST_BASE = {
    "n_estimators": 300,
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "tree_method": "hist",
    "enable_categorical": True,
    "eval_metric": "logloss",
    "n_jobs": -1,
    "random_state": SEED,
}

# Each candidate represents a documented pattern, not a Cartesian product.
LIGHTGBM_CANDIDATES = {
    "e4_baseline": {},
    "slow_lr_005": {"learning_rate": 0.05, "n_estimators": 1200},
    "slow_lr_003": {"learning_rate": 0.03, "n_estimators": 1800},
    "compact_leaves_15": {
        "learning_rate": 0.05, "n_estimators": 1200, "num_leaves": 15
    },
    "wide_leaves_63": {
        "learning_rate": 0.05, "n_estimators": 1200, "num_leaves": 63
    },
    "regularized_children": {
        "learning_rate": 0.05, "n_estimators": 1200,
        "min_child_samples": 50, "reg_alpha": 0.1, "reg_lambda": 1.0
    },
    "row_col_sampling": {
        "learning_rate": 0.05, "n_estimators": 1200,
        "subsample": 0.9, "subsample_freq": 1, "colsample_bytree": 0.9
    },
    "shallow_regularized": {
        "learning_rate": 0.05, "n_estimators": 1200,
        "max_depth": 6, "num_leaves": 31, "min_child_samples": 40,
        "reg_lambda": 1.0
    },
}

XGBOOST_CANDIDATES = {
    "e4_baseline": {},
    "slow_lr_005": {"learning_rate": 0.05, "n_estimators": 1200},
    "slow_lr_003": {"learning_rate": 0.03, "n_estimators": 1800},
    "depth_4": {
        "learning_rate": 0.05, "n_estimators": 1200, "max_depth": 4
    },
    "depth_5": {
        "learning_rate": 0.05, "n_estimators": 1200, "max_depth": 5
    },
    "depth_7": {
        "learning_rate": 0.05, "n_estimators": 1200, "max_depth": 7
    },
    "regularized": {
        "learning_rate": 0.05, "n_estimators": 1200,
        "min_child_weight": 5.0, "reg_alpha": 0.1, "reg_lambda": 2.0
    },
    "row_col_sampling": {
        "learning_rate": 0.05, "n_estimators": 1200,
        "subsample": 0.9, "colsample_bytree": 0.9
    },
    "gamma_01": {
        "learning_rate": 0.05, "n_estimators": 1200, "gamma": 0.1
    },
}


def candidate_params(base: dict, overrides: dict) -> dict:
    params = base.copy()
    params.update(overrides)
    return params


def make_candidate(family: str, overrides: dict, early_stopping: bool):
    """Build one candidate using fixed family defaults plus explicit overrides."""
    if family == "LightGBM":
        return LGBMClassifier(**candidate_params(LIGHTGBM_BASE, overrides))
    params = candidate_params(XGBOOST_BASE, overrides)
    if early_stopping:
        params["early_stopping_rounds"] = 75
        params["eval_metric"] = "auc"
    params["verbosity"] = 0
    return XGBClassifier(**params)


def tune_candidate(
    family: str,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    splits,
    early_stopping: bool,
) -> tuple[np.ndarray, list[float], list[int], float, float]:
    """Evaluate one candidate with fold-local early stopping."""
    oof = np.zeros(len(y), dtype=float)
    fold_aucs: list[float] = []
    best_iterations: list[int] = []
    fit_seconds = 0.0
    predict_seconds = 0.0
    for train_idx, valid_idx in splits:
        fitted = clone_model(model)
        fit_kwargs = {}
        if early_stopping:
            if family == "LightGBM":
                fit_kwargs["eval_X"] = X.iloc[valid_idx]
                fit_kwargs["eval_y"] = y.iloc[valid_idx]
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(75, verbose=False),
                    lgb.log_evaluation(0),
                ]
            else:
                fit_kwargs["eval_set"] = [(X.iloc[valid_idx], y.iloc[valid_idx])]
                fit_kwargs["verbose"] = False
        start = time.perf_counter()
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx], **fit_kwargs)
        fit_seconds += time.perf_counter() - start
        start = time.perf_counter()
        probability = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
        predict_seconds += time.perf_counter() - start
        oof[valid_idx] = probability
        fold_aucs.append(float(roc_auc_score(y.iloc[valid_idx], probability)))
        if not early_stopping:
            best_iterations.append(int(fitted.get_params()["n_estimators"]))
        elif family == "LightGBM":
            best_iterations.append(int(fitted.best_iteration_))
        else:
            best_iterations.append(int(fitted.best_iteration) + 1)
    return oof, fold_aucs, best_iterations, fit_seconds, predict_seconds


def tuning_row(
    family: str,
    candidate: str,
    params: dict,
    oof: np.ndarray,
    y: pd.Series,
    fold_aucs: list[float],
    best_iterations: list[int],
    baseline_auc: float,
    fit_seconds: float,
    predict_seconds: float,
) -> dict[str, object]:
    scores = evaluate_predictions(y, oof)
    row: dict[str, object] = {
        "family": family,
        "candidate": candidate,
        "parameters": json.dumps(params, sort_keys=True),
        "oof_auc": scores["roc_auc"],
        "log_loss": scores["log_loss"],
        "delta_vs_e4_component": scores["roc_auc"] - baseline_auc,
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "std_fold_auc": float(np.std(fold_aucs, ddof=1)),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "mean_best_iteration": float(np.mean(best_iterations)),
        "min_best_iteration": int(np.min(best_iterations)),
        "max_best_iteration": int(np.max(best_iterations)),
    }
    for fold, (auc, iteration) in enumerate(zip(fold_aucs, best_iterations), start=1):
        row[f"fold_{fold}_auc"] = auc
        row[f"fold_{fold}_best_iteration"] = iteration
    return row


def run_family_search(
    family: str,
    candidates: dict[str, dict],
    X: pd.DataFrame,
    y: pd.Series,
    splits,
    baseline_auc: float,
    output_csv: Path,
    output_npz: Path,
) -> None:
    if output_csv.exists() and output_npz.exists():
        completed = pd.read_csv(output_csv)
        saved = np.load(output_npz)
        if len(completed) == len(candidates) and set(saved.files) == set(candidates):
            if family == "LightGBM" and "iteration_indexing_normalized" not in completed:
                iteration_columns = [f"fold_{fold}_best_iteration" for fold in range(1, 6)]
                tuned = completed["candidate"] != "e4_baseline"
                completed.loc[tuned, iteration_columns] = (
                    completed.loc[tuned, iteration_columns] - 1
                )
                completed.loc[~tuned, iteration_columns] = 300
                completed["mean_best_iteration"] = completed[iteration_columns].mean(axis=1)
                completed["min_best_iteration"] = completed[iteration_columns].min(axis=1)
                completed["max_best_iteration"] = completed[iteration_columns].max(axis=1)
                completed["iteration_indexing_normalized"] = True
                completed.to_csv(output_csv, index=False)
            print(f"{family}: reusing {len(completed)} completed candidates", flush=True)
            return
    rows = []
    predictions = {}
    for name, overrides in candidates.items():
        early_stopping = name != "e4_baseline"
        print(f"{family}: {name}", flush=True)
        model = make_candidate(family, overrides, early_stopping)
        oof, fold_aucs, iterations, fit_time, predict_time = tune_candidate(
            family, model, X, y, splits, early_stopping
        )
        predictions[name] = oof
        rows.append(
            tuning_row(
                family, name, model.get_params(),
                oof, y, fold_aucs, iterations, baseline_auc, fit_time, predict_time
            )
        )
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        np.savez_compressed(output_npz, **predictions)


def blend_row(
    blend_type: str,
    left_model: str,
    right_model: str,
    left_weight: float,
    left: np.ndarray,
    right: np.ndarray,
    y: pd.Series,
) -> dict[str, object]:
    probability = left_weight * left + (1.0 - left_weight) * right
    scores = evaluate_predictions(y, probability)
    return {
        "blend_type": blend_type,
        "left_model": left_model,
        "right_model": right_model,
        "left_weight": left_weight,
        "right_weight": 1.0 - left_weight,
        "oof_auc": scores["roc_auc"],
        "log_loss": scores["log_loss"],
        "delta_vs_e4_blend": scores["roc_auc"] - E4_BLEND_AUC,
    }


def finalize_blend_analysis(y: pd.Series, report_dir: Path) -> None:
    """Evaluate simple tuned/untuned pairings from persisted complete OOF vectors."""
    lightgbm = np.load(report_dir / "lightgbm_tuning_oof.npz")
    xgboost = np.load(report_dir / "xgboost_tuning_oof.npz")
    e4_lgb = lightgbm["e4_baseline"]
    tuned_lgb = lightgbm[SELECTED_LIGHTGBM]
    e4_xgb = xgboost["e4_baseline"]
    tuned_xgb = xgboost[SELECTED_XGBOOST]
    pairs = [
        ("tuned_pair", "tuned_lightgbm", "tuned_xgboost", tuned_lgb, tuned_xgb),
        ("tuned_lgb_e4_xgb", "tuned_lightgbm", "e4_xgboost", tuned_lgb, e4_xgb),
        ("e4_lgb_tuned_xgb", "e4_lightgbm", "tuned_xgboost", e4_lgb, tuned_xgb),
    ]
    rows = []
    for blend_type, left_name, right_name, left, right in pairs:
        for weight in np.linspace(0.0, 1.0, 11):
            rows.append(
                blend_row(
                    blend_type, left_name, right_name, float(weight), left, right, y
                )
            )
    four_model = 0.25 * (e4_lgb + tuned_lgb + e4_xgb + tuned_xgb)
    four_scores = evaluate_predictions(y, four_model)
    rows.append(
        {
            "blend_type": "four_model_equal",
            "left_model": "all_four",
            "right_model": "all_four",
            "left_weight": 0.5,
            "right_weight": 0.5,
            "oof_auc": four_scores["roc_auc"],
            "log_loss": four_scores["log_loss"],
            "delta_vs_e4_blend": four_scores["roc_auc"] - E4_BLEND_AUC,
        }
    )
    pd.DataFrame(rows).to_csv(report_dir / "tuned_blend_results.csv", index=False)
    pd.DataFrame(
        {
            "target": y,
            "e4_lightgbm_oof": e4_lgb,
            "tuned_lightgbm_oof": tuned_lgb,
            "e4_xgboost_oof": e4_xgb,
            "tuned_xgboost_oof": tuned_xgb,
        }
    ).to_csv(report_dir / "oof_predictions_exp05.csv", index=False)


def finalize_experiment(
    X: pd.DataFrame,
    X_test: pd.DataFrame,
    y: pd.Series,
    sample: pd.DataFrame,
    report_dir: Path,
) -> None:
    """Bootstrap the robust 50/50 tuned blend and conditionally create submission 05."""
    lightgbm_oof = np.load(report_dir / "lightgbm_tuning_oof.npz")
    xgboost_oof = np.load(report_dir / "xgboost_tuning_oof.npz")
    tuned_blend = 0.5 * (
        lightgbm_oof[SELECTED_LIGHTGBM] + xgboost_oof[SELECTED_XGBOOST]
    )
    e4_blend = 0.5 * (
        lightgbm_oof["e4_baseline"] + xgboost_oof["e4_baseline"]
    )
    samples, summary = paired_bootstrap_difference(y, tuned_blend, e4_blend)
    samples.to_csv(report_dir / "tuning_bootstrap_samples.csv", index=False)
    selected_lgb_params = candidate_params(
        LIGHTGBM_BASE, LIGHTGBM_CANDIDATES[SELECTED_LIGHTGBM]
    )
    selected_xgb_params = candidate_params(
        XGBOOST_BASE, XGBOOST_CANDIDATES[SELECTED_XGBOOST]
    )
    analysis = {
        "selected_lightgbm": SELECTED_LIGHTGBM,
        "selected_lightgbm_parameters": selected_lgb_params,
        "selected_xgboost": SELECTED_XGBOOST,
        "selected_xgboost_parameters": selected_xgb_params,
        "recommended_blend": {
            "lightgbm_weight": 0.5,
            "xgboost_weight": 0.5,
            "oof_auc": float(roc_auc_score(y, tuned_blend)),
            "delta_vs_e4_blend": float(roc_auc_score(y, tuned_blend) - E4_BLEND_AUC),
        },
        "bootstrap_vs_e4_blend": summary,
    }
    (report_dir / "tuning_analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n"
    )
    robust = summary["ci_95_lower"] > 0
    if robust:
        fitted_lgb = LGBMClassifier(**selected_lgb_params).fit(X, y)
        fitted_xgb = XGBClassifier(**selected_xgb_params).fit(X, y)
        test_probability = 0.5 * (
            fitted_lgb.predict_proba(X_test)[:, 1]
            + fitted_xgb.predict_proba(X_test)[:, 1]
        )
        build_submission(sample, test_probability, "submissions/submission_05_tuned.csv")
    print(json.dumps(analysis, indent=2))
    print("Submission generated:", robust)


def main() -> None:
    train, test, sample = load_competition_data()
    X, y, X_test = split_features_target(train, test)
    X_native, X_test_native, _ = prepare_native_categories(X, X_test)
    X_engineered = add_engineered_features(X_native, FEATURES)
    splits = list(CVConfig().splitter().split(X_engineered, y))
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    run_family_search(
        "LightGBM", LIGHTGBM_CANDIDATES, X_engineered, y, splits,
        E4_LIGHTGBM_AUC, report_dir / "lightgbm_tuning.csv",
        report_dir / "lightgbm_tuning_oof.npz",
    )
    run_family_search(
        "XGBoost", XGBOOST_CANDIDATES, X_engineered, y, splits,
        E4_XGBOOST_AUC, report_dir / "xgboost_tuning.csv",
        report_dir / "xgboost_tuning_oof.npz",
    )
    finalize_blend_analysis(y, report_dir)
    finalize_experiment(X_engineered, add_engineered_features(X_test_native, FEATURES), y, sample, report_dir)


if __name__ == "__main__":
    main()
