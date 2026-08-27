"""Experiment 7: complementary model families around the E6 ensemble."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.stats import pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

from .boosting_tuning import FEATURES, LIGHTGBM_BASE, XGBOOST_BASE
from .data import load_competition_data, split_features_target
from .feature_diagnostics import add_engineered_features, paired_bootstrap_difference
from .model_comparison import clone_model, prepare_catboost, prepare_native_categories
from .submission import build_submission
from .validation import CVConfig, evaluate_predictions

SEED = 42
E6_AUC = 0.9642892514216848
E6_LIGHTGBM_WEIGHT = 0.4
E6_XGBOOST_WEIGHT = 0.6
CATBOOST_EARLY_STOPPING = 150
BOOTSTRAP_RESAMPLES = 500
CANDIDATE_WEIGHTS = np.arange(0.0, 0.31, 0.05)

CATBOOST_CANDIDATES = {
    "catboost_depth6_lr003": {
        "iterations": 2500,
        "learning_rate": 0.03,
        "depth": 6,
    },
    "catboost_depth7_lr003": {
        "iterations": 2500,
        "learning_rate": 0.03,
        "depth": 7,
    },
}

EXTRATREES_CANDIDATES = {
    "extra_trees_sqrt_leaf2": {
        "n_estimators": 300,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "max_depth": None,
        "n_jobs": -1,
        "random_state": SEED,
    }
}


def load_e6_predictions(path: Path, y: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and verify ignored E6 OOF predictions rather than retraining them."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required locally. Regenerate Experiment 6 before Experiment 7."
        )
    saved = pd.read_csv(path)
    required = {"target", "e6_lightgbm_oof", "e6_xgboost_oof", "e6_blend_oof"}
    if not required.issubset(saved):
        raise ValueError(f"E6 OOF file lacks columns: {sorted(required - set(saved))}")
    if len(saved) != len(y) or not np.array_equal(saved["target"].to_numpy(), y.to_numpy()):
        raise ValueError("E6 OOF targets do not align with the current training rows")
    lightgbm = saved["e6_lightgbm_oof"].to_numpy(float)
    xgboost = saved["e6_xgboost_oof"].to_numpy(float)
    blend = saved["e6_blend_oof"].to_numpy(float)
    reconstructed = E6_LIGHTGBM_WEIGHT * lightgbm + E6_XGBOOST_WEIGHT * xgboost
    if not np.allclose(blend, reconstructed, rtol=0.0, atol=1e-12):
        raise ValueError("Saved E6 blend does not equal the documented 40/60 blend")
    observed = {
        "lightgbm": roc_auc_score(y, lightgbm),
        "xgboost": roc_auc_score(y, xgboost),
        "blend": roc_auc_score(y, blend),
    }
    expected = {"lightgbm": 0.963424318502328, "xgboost": 0.9639908740203345, "blend": E6_AUC}
    for name in expected:
        if not np.isclose(observed[name], expected[name], atol=1e-12):
            raise ValueError(f"Unexpected E6 {name} AUC: {observed[name]:.12f}")
    return lightgbm, xgboost, blend


def make_catboost(params: dict, categorical: list[str]) -> CatBoostClassifier:
    """Build a conservative native-categorical CatBoost candidate."""
    return CatBoostClassifier(
        **params,
        loss_function="Logloss",
        eval_metric="AUC",
        cat_features=categorical,
        od_type="Iter",
        od_wait=CATBOOST_EARLY_STOPPING,
        use_best_model=True,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
        random_seed=SEED,
    )


def make_extra_trees(params: dict, X: pd.DataFrame):
    """Build fold-local imputation/encoding plus a randomized-tree estimator."""
    categorical = list(X.select_dtypes(exclude="number").columns)
    numeric = [column for column in X.columns if column not in categorical]
    preprocessing = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return make_pipeline(preprocessing, ExtraTreesClassifier(**params))


def evaluate_candidate(
    family: str,
    candidate: str,
    params: dict,
    model,
    X: pd.DataFrame,
    y: pd.Series,
    splits,
) -> tuple[dict[str, object], np.ndarray]:
    """Generate complete OOF probabilities on the canonical folds."""
    oof = np.zeros(len(y), dtype=float)
    fold_aucs: list[float] = []
    fold_log_losses: list[float] = []
    best_iterations: list[int | None] = []
    fit_seconds = 0.0
    predict_seconds = 0.0
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        print(f"{candidate}: fold {fold}", flush=True)
        fitted = clone_model(model)
        fit_kwargs: dict[str, object] = {}
        if family == "CatBoost":
            fit_kwargs = {
                "eval_set": (X.iloc[valid_idx], y.iloc[valid_idx]),
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
        fold_log_losses.append(float(log_loss(y.iloc[valid_idx], probability)))
        best_iterations.append(
            int(fitted.get_best_iteration()) + 1 if family == "CatBoost" else None
        )
    scores = evaluate_predictions(y, oof)
    row: dict[str, object] = {
        "family": family,
        "candidate": candidate,
        "parameters": json.dumps(params, sort_keys=True),
        "oof_auc": scores["roc_auc"],
        "log_loss": scores["log_loss"],
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "std_fold_auc": float(np.std(fold_aucs, ddof=1)),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "mean_best_iteration": (
            float(np.mean(best_iterations)) if family == "CatBoost" else np.nan
        ),
        "median_best_iteration": (
            float(np.median(best_iterations)) if family == "CatBoost" else np.nan
        ),
    }
    for fold, (auc, loss, iteration) in enumerate(
        zip(fold_aucs, fold_log_losses, best_iterations), start=1
    ):
        row[f"fold_{fold}_auc"] = auc
        row[f"fold_{fold}_log_loss"] = loss
        row[f"fold_{fold}_best_iteration"] = iteration
    return row, oof


def diversity_row(
    candidate: str, family: str, y: pd.Series, prediction: np.ndarray, e6: np.ndarray
) -> dict[str, object]:
    """Summarize probability, ranking, threshold, and conditional disagreement."""
    y_array = y.to_numpy()
    difference = np.abs(prediction - e6)
    disagreement = (prediction >= 0.5) != (e6 >= 0.5)
    substantial_cutoff = float(np.quantile(difference, 0.9))
    substantial = difference >= substantial_cutoff
    candidate_class = prediction >= 0.5
    e6_class = e6 >= 0.5
    candidate_correct = candidate_class == y_array
    e6_correct = e6_class == y_array
    row: dict[str, object] = {
        "family": family,
        "candidate": candidate,
        "pearson_vs_e6": float(pearsonr(prediction, e6).statistic),
        "spearman_vs_e6": float(spearmanr(prediction, e6).statistic),
        "mean_absolute_probability_difference": float(difference.mean()),
        "absolute_difference_q50": float(np.quantile(difference, 0.50)),
        "absolute_difference_q75": float(np.quantile(difference, 0.75)),
        "absolute_difference_q90": substantial_cutoff,
        "absolute_difference_q95": float(np.quantile(difference, 0.95)),
        "absolute_difference_q99": float(np.quantile(difference, 0.99)),
        "threshold_disagreement": float(disagreement.mean()),
        "threshold_disagreement_y0": float(disagreement[y_array == 0].mean()),
        "threshold_disagreement_y1": float(disagreement[y_array == 1].mean()),
        "substantial_difference_threshold_q90": substantial_cutoff,
        "substantial_subset_rows": int(substantial.sum()),
        "substantial_subset_e6_auc": float(roc_auc_score(y_array[substantial], e6[substantial])),
        "substantial_subset_candidate_auc": float(
            roc_auc_score(y_array[substantial], prediction[substantial])
        ),
        "substantial_subset_e6_log_loss": float(log_loss(y_array[substantial], e6[substantial])),
        "substantial_subset_candidate_log_loss": float(
            log_loss(y_array[substantial], prediction[substantial])
        ),
        "candidate_correct_e6_wrong_rate": float((candidate_correct & ~e6_correct).mean()),
        "e6_correct_candidate_wrong_rate": float((e6_correct & ~candidate_correct).mean()),
        "candidate_net_threshold_corrections": int(
            (candidate_correct & ~e6_correct).sum() - (e6_correct & ~candidate_correct).sum()
        ),
    }
    return row


def candidate_blend_curve(
    candidate: str, family: str, y: pd.Series, prediction: np.ndarray, e6: np.ndarray
) -> pd.DataFrame:
    """Test coarse candidate weights while preserving E6's internal 40/60 ratio."""
    rows = []
    for weight in CANDIDATE_WEIGHTS:
        blended = (1.0 - weight) * e6 + weight * prediction
        scores = evaluate_predictions(y, blended)
        rows.append(
            {
                "blend_type": "single_candidate",
                "candidate": candidate,
                "family": family,
                "candidate_weight": float(weight),
                "e6_weight": float(1.0 - weight),
                "effective_lightgbm_weight": float((1.0 - weight) * E6_LIGHTGBM_WEIGHT),
                "effective_xgboost_weight": float((1.0 - weight) * E6_XGBOOST_WEIGHT),
                "oof_auc": scores["roc_auc"],
                "log_loss": scores["log_loss"],
                "delta_vs_e6": scores["roc_auc"] - E6_AUC,
            }
        )
    return pd.DataFrame(rows)


def has_broad_improvement(curve: pd.DataFrame) -> bool:
    """Require at least two adjacent positive non-zero candidate weights."""
    positive = curve.loc[curve["candidate_weight"] > 0, "delta_vs_e6"].to_numpy() > 0
    return bool(np.any(positive[:-1] & positive[1:]))


def combined_curves(
    useful: list[str], predictions: dict[str, np.ndarray], y: pd.Series, e6: np.ndarray
) -> pd.DataFrame:
    """Evaluate a few small two-candidate allocations without fine optimization."""
    if len(useful) < 2:
        return pd.DataFrame()
    first, second = useful[:2]
    allocations = [(0.05, 0.05), (0.10, 0.05), (0.05, 0.10), (0.10, 0.10)]
    rows = []
    for first_weight, second_weight in allocations:
        e6_weight = 1.0 - first_weight - second_weight
        probability = (
            e6_weight * e6
            + first_weight * predictions[first]
            + second_weight * predictions[second]
        )
        scores = evaluate_predictions(y, probability)
        rows.append(
            {
                "blend_type": "combined_candidates",
                "candidate": f"{first}+{second}",
                "family": "Combined",
                "candidate_weight": first_weight + second_weight,
                "candidate_1": first,
                "candidate_1_weight": first_weight,
                "candidate_2": second,
                "candidate_2_weight": second_weight,
                "e6_weight": e6_weight,
                "effective_lightgbm_weight": e6_weight * E6_LIGHTGBM_WEIGHT,
                "effective_xgboost_weight": e6_weight * E6_XGBOOST_WEIGHT,
                "oof_auc": scores["roc_auc"],
                "log_loss": scores["log_loss"],
                "delta_vs_e6": scores["roc_auc"] - E6_AUC,
            }
        )
    return pd.DataFrame(rows)


def proposed_probability(
    row: pd.Series, predictions: dict[str, np.ndarray], e6: np.ndarray
) -> np.ndarray:
    if row["blend_type"] == "single_candidate":
        return row["e6_weight"] * e6 + row["candidate_weight"] * predictions[row["candidate"]]
    return (
        row["e6_weight"] * e6
        + row["candidate_1_weight"] * predictions[row["candidate_1"]]
        + row["candidate_2_weight"] * predictions[row["candidate_2"]]
    )


def run_experiment(report_dir: Path = Path("reports")) -> dict[str, object]:
    """Run/resume candidates, diversity analysis, coarse blends, and uncertainty."""
    train, test, sample = load_competition_data()
    X, y, X_test = split_features_target(train, test)
    X_catboost_raw, X_test_catboost_raw, categorical = prepare_catboost(X, X_test)
    X_native, X_test_native, _ = prepare_native_categories(X, X_test)
    X_engineered = add_engineered_features(X_native, FEATURES)
    X_test_engineered = add_engineered_features(X_test_native, FEATURES)
    X_catboost = add_engineered_features(X_catboost_raw, FEATURES)
    X_test_catboost = add_engineered_features(X_test_catboost_raw, FEATURES)
    splits = list(CVConfig().splitter().split(X_engineered, y))
    e6_lgb, e6_xgb, e6 = load_e6_predictions(
        report_dir / "oof_predictions_exp06.csv", y
    )

    oof_path = report_dir / "diversity_oof.npz"
    report_path = report_dir / "diversity_model_results.csv"
    predictions: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    if oof_path.exists() and report_path.exists():
        saved = np.load(oof_path)
        predictions = {name: saved[name] for name in saved.files}
        rows = pd.read_csv(report_path).to_dict("records")
    completed = set(predictions)

    candidate_specs: list[tuple[str, str, dict, object, pd.DataFrame]] = []
    for name, params in CATBOOST_CANDIDATES.items():
        candidate_specs.append(
            ("CatBoost", name, params, make_catboost(params, categorical), X_catboost)
        )
    for name, params in EXTRATREES_CANDIDATES.items():
        candidate_specs.append(
            ("ExtraTrees", name, params, make_extra_trees(params, X_engineered), X_engineered)
        )

    for family, name, params, model, model_X in candidate_specs:
        if name in completed:
            print(f"reuse {name}", flush=True)
            continue
        row, oof = evaluate_candidate(family, name, params, model, model_X, y, splits)
        predictions[name] = oof
        rows.append(row)
        pd.DataFrame(rows).to_csv(report_path, index=False)
        np.savez_compressed(oof_path, **predictions)

    model_report = pd.DataFrame(rows)
    model_report.to_csv(report_path, index=False)
    family_by_candidate = model_report.set_index("candidate")["family"].to_dict()
    diversity = pd.DataFrame(
        [
            diversity_row(name, family_by_candidate[name], y, prediction, e6)
            for name, prediction in predictions.items()
        ]
    ).sort_values("candidate")
    diversity.to_csv(report_dir / "diversity_correlations.csv", index=False)

    curves = pd.concat(
        [
            candidate_blend_curve(name, family_by_candidate[name], y, prediction, e6)
            for name, prediction in predictions.items()
        ],
        ignore_index=True,
    )
    broad = {
        name: has_broad_improvement(curves[curves["candidate"] == name])
        for name in predictions
    }
    selected_catboost = (
        curves[curves["family"] == "CatBoost"]
        .sort_values("oof_auc", ascending=False)
        .iloc[0]["candidate"]
    )
    useful = [
        name
        for name in [selected_catboost, *EXTRATREES_CANDIDATES]
        if broad.get(name, False)
    ]
    combined = combined_curves(useful, predictions, y, e6)
    if not combined.empty:
        curves = pd.concat([curves, combined], ignore_index=True, sort=False)
    curves.to_csv(report_dir / "diversity_blend_results.csv", index=False)

    best = curves.sort_values("oof_auc", ascending=False).iloc[0]
    bootstrap_summary = None
    bootstrap_observed = float(best["delta_vs_e6"])
    if bootstrap_observed > 0:
        proposed = proposed_probability(best, predictions, e6)
        bootstrap_samples, bootstrap_summary = paired_bootstrap_difference(
            y, proposed, e6, n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED
        )
        bootstrap_samples.to_csv(
            report_dir / "diversity_bootstrap_samples.csv", index=False
        )

    robust = bool(
        bootstrap_summary is not None
        and bootstrap_summary["ci_95_lower"] > 0
        and bootstrap_summary["proportion_positive"] >= 0.95
    )
    analysis: dict[str, object] = {
        "experiment": 7,
        "e6_reconstruction": {
            "lightgbm_oof_auc": float(roc_auc_score(y, e6_lgb)),
            "xgboost_oof_auc": float(roc_auc_score(y, e6_xgb)),
            "blend_oof_auc": float(roc_auc_score(y, e6)),
            "lightgbm_weight": E6_LIGHTGBM_WEIGHT,
            "xgboost_weight": E6_XGBOOST_WEIGHT,
        },
        "candidate_count": len(predictions),
        "selected_catboost_for_ensemble_analysis": selected_catboost,
        "broad_individual_improvement": broad,
        "combined_ensemble_tested": not combined.empty,
        "best_proposed_ensemble": {
            key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
            for key, value in best.to_dict().items()
        },
        "observed_delta_vs_e6": bootstrap_observed,
        "paired_bootstrap_vs_e6": bootstrap_summary,
        "robust_positive_interval": robust,
        "selection_bias_note": (
            "Candidate configuration and coarse weight were selected on these same OOF "
            "predictions; the paired interval quantifies sampling uncertainty, not selection bias."
        ),
        "submission_recommended_by_statistical_rules": robust,
        "ignored_local_artifacts": [
            "reports/diversity_oof.npz",
            "reports/diversity_bootstrap_samples.csv",
        ],
    }
    (report_dir / "diversity_analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n"
    )
    print(json.dumps(analysis, indent=2), flush=True)
    return analysis


def fit_submission_from_analysis(report_dir: Path = Path("reports")) -> None:
    """Fit the approved E7 ensemble on all rows and validate its submission."""
    analysis = json.loads((report_dir / "diversity_analysis.json").read_text())
    if not analysis["submission_recommended_by_statistical_rules"]:
        raise RuntimeError("Experiment 7 evidence does not justify a submission")
    best = analysis["best_proposed_ensemble"]
    train, test, sample = load_competition_data()
    X, y, X_test = split_features_target(train, test)
    X_catboost_raw, X_test_catboost_raw, categorical = prepare_catboost(X, X_test)
    X_native, X_test_native, _ = prepare_native_categories(X, X_test)
    X_engineered = add_engineered_features(X_native, FEATURES)
    X_test_engineered = add_engineered_features(X_test_native, FEATURES)
    X_catboost = add_engineered_features(X_catboost_raw, FEATURES)
    X_test_catboost = add_engineered_features(X_test_catboost_raw, FEATURES)

    lgb_params = LIGHTGBM_BASE.copy()
    lgb_params.update({"num_leaves": 63, "learning_rate": 0.02, "n_estimators": 3808})
    xgb_params = XGBOOST_BASE.copy()
    xgb_params.update({"max_depth": 7, "learning_rate": 0.02, "n_estimators": 4720})
    lgb = LGBMClassifier(**lgb_params).fit(X_engineered, y)
    xgb = XGBClassifier(**xgb_params).fit(X_engineered, y)
    e6_test = (
        E6_LIGHTGBM_WEIGHT * lgb.predict_proba(X_test_engineered)[:, 1]
        + E6_XGBOOST_WEIGHT * xgb.predict_proba(X_test_engineered)[:, 1]
    )

    candidate_names = (
        [best["candidate"]]
        if best["blend_type"] == "single_candidate"
        else [best["candidate_1"], best["candidate_2"]]
    )
    candidate_predictions: dict[str, np.ndarray] = {}
    model_report = pd.read_csv(report_dir / "diversity_model_results.csv").set_index("candidate")
    for name in candidate_names:
        family = model_report.loc[name, "family"]
        params = json.loads(model_report.loc[name, "parameters"])
        if family == "CatBoost":
            params["iterations"] = int(model_report.loc[name, "median_best_iteration"])
            fitted = make_catboost(params, categorical)
            fitted.set_params(use_best_model=False)
            fitted.fit(X_catboost, y, verbose=False)
            candidate_predictions[name] = fitted.predict_proba(X_test_catboost)[:, 1]
        else:
            fitted = make_extra_trees(params, X_engineered).fit(X_engineered, y)
            candidate_predictions[name] = fitted.predict_proba(X_test_engineered)[:, 1]

    if best["blend_type"] == "single_candidate":
        probability = (
            best["e6_weight"] * e6_test
            + best["candidate_weight"] * candidate_predictions[best["candidate"]]
        )
    else:
        probability = (
            best["e6_weight"] * e6_test
            + best["candidate_1_weight"] * candidate_predictions[best["candidate_1"]]
            + best["candidate_2_weight"] * candidate_predictions[best["candidate_2"]]
        )
    build_submission(
        sample,
        np.clip(probability, 0.0, 1.0),
        "submissions/submission_07_diversity.csv",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fit-submission",
        action="store_true",
        help="Fit full-data models only after the saved evidence recommends submission.",
    )
    args = parser.parse_args()
    if args.fit_submission:
        fit_submission_from_analysis()
    else:
        run_experiment()


if __name__ == "__main__":
    main()
