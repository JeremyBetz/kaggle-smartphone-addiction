"""Experiment 9: bounded behavioral feature discovery and confirmation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from .boosting_tuning import FEATURES, LIGHTGBM_BASE, XGBOOST_BASE
from .data import ID_COLUMN, load_competition_data, split_features_target
from .feature_diagnostics import add_engineered_features, safe_ratio
from .model_comparison import clone_model, prepare_native_categories
from .repeated_validation import FOLD_SEEDS, paired_summary
from .submission import build_submission
from .validation import CVConfig

MODEL_SEED = 42
DEVELOPMENT_SEED = 42
N_SPLITS = 5
LGB_WEIGHT = 0.4
SCREEN_MINIMUM_GAIN = 0.00005
MIN_POSITIVE_FOLDS = 3
SEED42_CONFIRMATION_GAIN = 0.00005

# Frozen before any E9 model was evaluated. E4 already tested and rejected the
# unchanged notifications/open, opens/screen-hour, notifications/screen-hour,
# weekend/daily ratio and gap, entertainment total, and entertainment share.
CANDIDATES = {
    "social_media_screen_share": {
        "family": "composition",
        "hypothesis": "Social-media share may be more informative than absolute hours.",
    },
    "gaming_screen_share": {
        "family": "composition",
        "hypothesis": "Gaming share may distinguish screen-time composition.",
    },
    "work_study_screen_share": {
        "family": "composition",
        "hypothesis": "Work/study share may complement the existing difference feature.",
    },
    "unallocated_screen_contrast": {
        "family": "composition",
        "hypothesis": "Total screen time not represented by three named activities may add signal.",
    },
    "leisure_productive_balance": {
        "family": "composition",
        "hypothesis": "A bounded relative screen/work-study contrast may add scale invariance.",
    },
    "weekend_relative_deviation": {
        "family": "weekday_weekend",
        "hypothesis": "A bounded weekend deviation may be stabler than E4's ratio or raw gap.",
    },
    "joint_engagement_intensity": {
        "family": "engagement",
        "hypothesis": "The geometric mean of notifications and opens may capture joint intensity.",
    },
    "social_engagement_load": {
        "family": "interaction",
        "hypothesis": "Social-media hours and app opens may jointly represent engagement load.",
    },
    "cross_context_screen_load": {
        "family": "weekday_weekend",
        "hypothesis": "Weekday-like and weekend screen levels may interact across contexts.",
    },
    "screen_sleep_balance": {
        "family": "balance",
        "hypothesis": "The difference between screen and sleep hours may expose a behavioral balance.",
    },
}


def candidate_values(X: pd.DataFrame) -> dict[str, pd.Series]:
    """Calculate the frozen E9 candidates without fitted global preprocessing."""
    screen = X["daily_screen_time_hours"]
    work = X["work_study_hours"]
    notifications = X["notifications_per_day"]
    opens = X["app_opens_per_day"]
    weekend = X["weekend_screen_time"]
    leisure = screen - work
    return {
        "social_media_screen_share": safe_ratio(X["social_media_hours"], screen),
        "gaming_screen_share": safe_ratio(X["gaming_hours"], screen),
        "work_study_screen_share": safe_ratio(work, screen),
        "unallocated_screen_contrast": (
            screen - X["social_media_hours"] - X["gaming_hours"] - work
        ),
        "leisure_productive_balance": safe_ratio(leisure, screen + work),
        "weekend_relative_deviation": safe_ratio(weekend - screen, weekend + screen),
        "joint_engagement_intensity": np.sqrt(notifications * opens),
        "social_engagement_load": X["social_media_hours"] * opens,
        "cross_context_screen_load": screen * weekend,
        "screen_sleep_balance": screen - X["sleep_hours"],
    }


def add_discovery_features(X: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Add a declared feature set consistently to train or test data."""
    unknown = set(names) - set(CANDIDATES)
    if unknown:
        raise ValueError(f"Unknown E9 features: {sorted(unknown)}")
    result = X.copy()
    values = candidate_values(X)
    for name in names:
        value = values[name].replace([np.inf, -np.inf], np.nan)
        result[name] = value
    return result


def make_model(family: str, stage: str):
    """Build either the inexpensive E4 screen or fixed deployed E6 model."""
    if stage not in {"screen", "e6"}:
        raise ValueError(f"Unknown stage: {stage}")
    if family == "LightGBM":
        params = LIGHTGBM_BASE.copy()
        if stage == "e6":
            params.update({"learning_rate": 0.02, "n_estimators": 3808, "num_leaves": 63})
        params["random_state"] = MODEL_SEED
        return LGBMClassifier(**params)
    params = XGBOOST_BASE.copy()
    if stage == "e6":
        params.update({"learning_rate": 0.02, "n_estimators": 4720, "max_depth": 7})
    params.update({"random_state": MODEL_SEED, "verbosity": 0})
    return XGBClassifier(**params)


def generate_oof(model, X: pd.DataFrame, y: pd.Series, splits, label: str):
    """Generate complete OOF probabilities with timing and fold AUCs."""
    oof = np.zeros(len(y), dtype=float)
    fold_aucs = []
    fit_seconds = 0.0
    predict_seconds = 0.0
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        print(f"{label}: fold {fold}", flush=True)
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


def metric_row(
    stage: str,
    name: str,
    family: str,
    features: list[str],
    y: pd.Series,
    probability: np.ndarray,
    splits,
    control_probability: np.ndarray,
    fit_seconds: float,
    predict_seconds: float,
    fold_seed: int = DEVELOPMENT_SEED,
) -> dict[str, object]:
    """Build one compact paired result row."""
    fold_aucs = []
    control_fold_aucs = []
    for _, valid_idx in splits:
        fold_aucs.append(float(roc_auc_score(y.iloc[valid_idx], probability[valid_idx])))
        control_fold_aucs.append(
            float(roc_auc_score(y.iloc[valid_idx], control_probability[valid_idx]))
        )
    row: dict[str, object] = {
        "stage": stage,
        "name": name,
        "model": family,
        "features": "|".join(features),
        "n_added_features": len(features),
        "fold_seed": fold_seed,
        "oof_auc": float(roc_auc_score(y, probability)),
        "control_auc": float(roc_auc_score(y, control_probability)),
        "delta_vs_control": float(
            roc_auc_score(y, probability) - roc_auc_score(y, control_probability)
        ),
        "log_loss": float(log_loss(y, probability)),
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "std_fold_auc": float(np.std(fold_aucs, ddof=1)),
        "positive_fold_deltas": int(
            (np.asarray(fold_aucs) > np.asarray(control_fold_aucs)).sum()
        ),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }
    for fold, auc in enumerate(fold_aucs, start=1):
        row[f"fold_{fold}_auc"] = auc
    return row


def save_checkpoint(path: Path, probabilities: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **probabilities)


def load_checkpoint(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    archive = np.load(path)
    return {name: archive[name] for name in archive.files}


def screen_candidates(X: pd.DataFrame, y: pd.Series, report_dir: Path):
    """Evaluate all frozen candidates with the cheap E4 configurations."""
    prior = pd.read_csv(report_dir / "oof_predictions_exp04.csv")
    if not prior[ID_COLUMN].is_unique or len(prior) != len(y):
        raise ValueError("Experiment 4 OOF artifact is not aligned")
    control = {
        "LightGBM": prior["engineered_lightgbm_oof"].to_numpy(),
        "XGBoost": prior["engineered_xgboost_oof"].to_numpy(),
    }
    control["Blend"] = 0.5 * control["LightGBM"] + 0.5 * control["XGBoost"]
    splits = list(CVConfig().splitter().split(X, y))
    cache_path = report_dir / "feature_discovery_screen_oof.npz"
    cache = load_checkpoint(cache_path)
    rows = []
    for name in CANDIDATES:
        candidate_X = add_discovery_features(X, [name])
        component = {}
        for family in ["LightGBM", "XGBoost"]:
            key = f"{name}__{family.lower()}"
            if key not in cache:
                cache[key], _, fit_time, predict_time = generate_oof(
                    make_model(family, "screen"), candidate_X, y, splits,
                    f"screen {name} {family}",
                )
                save_checkpoint(cache_path, cache)
            else:
                print(f"reuse {key}", flush=True)
                fit_time = predict_time = 0.0
            component[family] = cache[key]
            rows.append(metric_row(
                "individual", name, family, [name], y, cache[key], splits,
                control[family], fit_time, predict_time,
            ))
        blend = 0.5 * component["LightGBM"] + 0.5 * component["XGBoost"]
        rows.append(metric_row(
            "individual", name, "Blend50_50", [name], y, blend, splits,
            control["Blend"], 0.0, 0.0,
        ))
        pd.DataFrame(rows).to_csv(report_dir / "feature_discovery_screen.csv", index=False)
    return pd.DataFrame(rows), cache, control, splits


def credible_individuals(screen: pd.DataFrame) -> list[str]:
    pivot = screen.pivot(index="name", columns="model", values="delta_vs_control")
    blend_rows = screen[screen["model"] == "Blend50_50"].set_index("name")
    credible = pivot[
        (pivot["LightGBM"] > 0)
        & (pivot["XGBoost"] > 0)
        & (pivot["Blend50_50"] >= SCREEN_MINIMUM_GAIN)
        & (blend_rows["positive_fold_deltas"] >= MIN_POSITIVE_FOLDS)
    ].index.tolist()
    return sorted(
        credible,
        key=lambda name: float(pivot.loc[name, "Blend50_50"]),
        reverse=True,
    )


def declared_packages(credible: list[str]) -> list[tuple[str, list[str]]]:
    """Create at most two packages using a frozen, non-combinatorial rule."""
    if not credible:
        return []
    packages = [("best_individual", [credible[0]])]
    first_family = CANDIDATES[credible[0]]["family"]
    distinct = next(
        (name for name in credible[1:] if CANDIDATES[name]["family"] != first_family),
        None,
    )
    if distinct is not None:
        packages.append(("top_distinct_families", [credible[0], distinct]))
    return packages


def screen_packages(
    packages, X, y, report_dir, cache, control, splits
) -> pd.DataFrame:
    rows = []
    cache_path = report_dir / "feature_discovery_screen_oof.npz"
    for package_name, features in packages:
        package_X = add_discovery_features(X, features)
        component = {}
        for family in ["LightGBM", "XGBoost"]:
            key = f"package_{package_name}__{family.lower()}"
            if key not in cache:
                cache[key], _, fit_time, predict_time = generate_oof(
                    make_model(family, "screen"), package_X, y, splits,
                    f"package {package_name} {family}",
                )
                save_checkpoint(cache_path, cache)
            else:
                fit_time = predict_time = 0.0
            component[family] = cache[key]
            rows.append(metric_row(
                "package", package_name, family, features, y, cache[key], splits,
                control[family], fit_time, predict_time,
            ))
        blend = 0.5 * component["LightGBM"] + 0.5 * component["XGBoost"]
        rows.append(metric_row(
            "package", package_name, "Blend50_50", features, y, blend, splits,
            control["Blend"], 0.0, 0.0,
        ))
    result = pd.DataFrame(rows)
    result.to_csv(report_dir / "feature_discovery_packages.csv", index=False)
    return result


def select_for_confirmation(screen: pd.DataFrame, packages: pd.DataFrame) -> list[str]:
    all_rows = pd.concat([screen, packages], ignore_index=True)
    eligible = all_rows[all_rows["model"] == "Blend50_50"].copy()
    component = all_rows.pivot_table(
        index=["stage", "name", "features"], columns="model", values="delta_vs_control"
    ).reset_index()
    eligible = eligible.merge(component, on=["stage", "name", "features"], suffixes=("", "_component"))
    eligible = eligible[
        (eligible["delta_vs_control"] >= SCREEN_MINIMUM_GAIN)
        & (eligible["positive_fold_deltas"] >= MIN_POSITIVE_FOLDS)
        & (eligible["LightGBM"] > 0)
        & (eligible["XGBoost"] > 0)
    ]
    if eligible.empty:
        return []
    best = eligible.sort_values(
        ["delta_vs_control", "n_added_features"], ascending=[False, True]
    ).iloc[0]
    return str(best["features"]).split("|")


def evaluate_e6_seed(
    fold_seed: int,
    features: list[str],
    X: pd.DataFrame,
    y: pd.Series,
    report_dir: Path,
    control_cache: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Evaluate one frozen E9 package with fixed deployed E6 models."""
    splits = list(StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=fold_seed
    ).split(X, y))
    candidate_X = add_discovery_features(X, features)
    path = report_dir / f"feature_discovery_e6_seed_{fold_seed}_oof.npz"
    cache = load_checkpoint(path)
    control = {} if control_cache is None else control_cache.copy()
    rows = []
    candidate = {}
    for family in ["LightGBM", "XGBoost"]:
        lower = family.lower()
        control_key = f"control_{lower}"
        candidate_key = f"candidate_{lower}"
        if control_key not in control:
            if control_key not in cache:
                cache[control_key], _, _, _ = generate_oof(
                    make_model(family, "e6"), X, y, splits,
                    f"E6 control seed {fold_seed} {family}",
                )
                save_checkpoint(path, cache)
            control[control_key] = cache[control_key]
        if candidate_key not in cache:
            cache[candidate_key], _, fit_time, predict_time = generate_oof(
                make_model(family, "e6"), candidate_X, y, splits,
                f"E9 candidate seed {fold_seed} {family}",
            )
            save_checkpoint(path, cache)
        else:
            fit_time = predict_time = 0.0
        candidate[candidate_key] = cache[candidate_key]
        rows.append(metric_row(
            "e6_confirmation", "frozen_candidate", family, features, y,
            cache[candidate_key], splits, control[control_key], fit_time,
            predict_time, fold_seed,
        ))
    candidate_blend = (
        LGB_WEIGHT * candidate["candidate_lightgbm"]
        + (1.0 - LGB_WEIGHT) * candidate["candidate_xgboost"]
    )
    control_blend = (
        LGB_WEIGHT * control["control_lightgbm"]
        + (1.0 - LGB_WEIGHT) * control["control_xgboost"]
    )
    rows.append(metric_row(
        "e6_confirmation", "frozen_candidate", "Blend40_60", features, y,
        candidate_blend, splits, control_blend, 0.0, 0.0, fold_seed,
    ))
    return rows, candidate_blend


def analyze_confirmation(rows: pd.DataFrame, report_dir: Path) -> dict[str, object]:
    blend = rows[rows["model"] == "Blend40_60"].sort_values("fold_seed")
    repeated = blend[blend["fold_seed"].isin(FOLD_SEEDS)]
    analysis: dict[str, object] = {}
    if len(repeated) == len(FOLD_SEEDS):
        rng = np.random.default_rng(42)
        paired, samples = paired_summary(repeated["delta_vs_control"].to_numpy(), rng)
        pd.DataFrame({"auc_difference": samples}).to_csv(
            report_dir / "feature_discovery_bootstrap_samples.csv", index=False
        )
        analysis["repeated_seed_comparison"] = paired
        seed42 = float(blend.loc[blend["fold_seed"] == 42, "delta_vs_control"].iloc[0])
        analysis["seed42_context"] = {
            "difference": seed42,
            "additional_seed_mean": float(repeated["delta_vs_control"].mean()),
            "difference_minus_additional_mean": float(
                seed42 - repeated["delta_vs_control"].mean()
            ),
            "percentile_among_additional_seeds": float(
                (repeated["delta_vs_control"] <= seed42).mean()
            ),
        }
    return analysis


def candidate_manifest(X: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    train_values = candidate_values(X)
    test_values = candidate_values(X_test)
    for name, details in CANDIDATES.items():
        train_value = train_values[name]
        test_value = test_values[name]
        rows.append({
            "feature": name,
            **details,
            "train_missing_rate": float(train_value.isna().mean()),
            "test_missing_rate": float(test_value.isna().mean()),
            "train_infinite_count": int(np.isinf(train_value.to_numpy()).sum()),
            "test_infinite_count": int(np.isinf(test_value.to_numpy()).sum()),
            "train_min": float(train_value.min()),
            "train_median": float(train_value.median()),
            "train_max": float(train_value.max()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    train, test, sample = load_competition_data()
    X_raw, y, X_test_raw = split_features_target(train, test)
    X_native, X_test_native, _ = prepare_native_categories(X_raw, X_test_raw)
    X = add_engineered_features(X_native, FEATURES)
    X_test = add_engineered_features(X_test_native, FEATURES)
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)

    manifest = candidate_manifest(X, X_test)
    manifest.to_csv(report_dir / "feature_discovery_candidates.csv", index=False)
    screen, cache, controls, splits = screen_candidates(X, y, report_dir)
    credible = credible_individuals(screen)
    packages = declared_packages(credible)
    package_results = screen_packages(
        packages, X, y, report_dir, cache, controls, splits
    )
    selected = select_for_confirmation(screen, package_results)

    analysis: dict[str, object] = {
        "experiment": 9,
        "purpose": "targeted_feature_discovery",
        "predeclared_candidates": CANDIDATES,
        "development_configuration": "fixed E4 LightGBM and XGBoost; seed 42",
        "development_gate": {
            "minimum_blend_auc_gain": SCREEN_MINIMUM_GAIN,
            "both_components_must_improve": True,
            "minimum_positive_fold_deltas": MIN_POSITIVE_FOLDS,
        },
        "credible_individuals": credible,
        "tested_packages": [
            {"name": name, "features": features} for name, features in packages
        ],
        "selected_for_e6_confirmation": selected,
        "selected_feature_definition": (
            "daily_screen_time_hours - social_media_hours - gaming_hours "
            "- work_study_hours" if selected else None
        ),
        "development_screen_results": screen[
            ["name", "model", "oof_auc", "delta_vs_control", "positive_fold_deltas"]
        ].to_dict("records"),
        "confirmation_reached": bool(selected),
        "submission_created": False,
    }

    confirmation_rows: list[dict[str, object]] = []
    robust = False
    if selected:
        seed42_rows, _ = evaluate_e6_seed(42, selected, X, y, report_dir)
        confirmation_rows.extend(seed42_rows)
        seed42 = pd.DataFrame(seed42_rows).set_index("model")
        pass_seed42 = bool(
            seed42.loc["Blend40_60", "delta_vs_control"] >= SEED42_CONFIRMATION_GAIN
            and seed42.loc["LightGBM", "delta_vs_control"] > 0
            and seed42.loc["XGBoost", "delta_vs_control"] > 0
        )
        analysis["seed42_confirmation_gate_passed"] = pass_seed42
        analysis["seed42_confirmation"] = seed42[
            ["oof_auc", "control_auc", "delta_vs_control", "positive_fold_deltas"]
        ].reset_index().to_dict("records")
        if pass_seed42:
            for seed in FOLD_SEEDS:
                archive = np.load(report_dir / f"repeated_cv_seed_{seed}_oof.npz")
                control_cache = {
                    "control_lightgbm": archive["e6_lightgbm"],
                    "control_xgboost": archive["e6_xgboost"],
                }
                seed_rows, _ = evaluate_e6_seed(
                    seed, selected, X, y, report_dir, control_cache
                )
                confirmation_rows.extend(seed_rows)
        confirmation = pd.DataFrame(confirmation_rows)
        confirmation.to_csv(
            report_dir / "feature_discovery_confirmation.csv", index=False
        )
        analysis.update(analyze_confirmation(confirmation, report_dir))
        repeated_components = confirmation[
            confirmation["fold_seed"].isin(FOLD_SEEDS)
        ].groupby("model")["delta_vs_control"].agg(["mean", "median", "std", "min", "max"])
        analysis["repeated_component_differences"] = (
            repeated_components.reset_index().to_dict("records")
        )
        paired = analysis.get("repeated_seed_comparison", {})
        robust = bool(
            pass_seed42
            and paired
            and paired["seed_bootstrap_ci_95_lower"] > 0
            and paired["wins"] >= 4
            and paired["mean_difference"] > 0
        )

    analysis["promotion_passed"] = robust
    analysis["champion"] = "E9" if robust else "E6"
    analysis["limitations"] = [
        "Candidates and limited packages share the development folds, so selection bias remains.",
        "Five confirmation seeds are a small independent paired sample; their 25 folds are not independent.",
        "Engineered quantities are predictive representations, not causal or literal behavioral measurements.",
    ]
    if robust:
        fitted_lgb = make_model("LightGBM", "e6").fit(
            add_discovery_features(X, selected), y
        )
        fitted_xgb = make_model("XGBoost", "e6").fit(
            add_discovery_features(X, selected), y
        )
        test_candidate = add_discovery_features(X_test, selected)
        probability = (
            LGB_WEIGHT * fitted_lgb.predict_proba(test_candidate)[:, 1]
            + (1.0 - LGB_WEIGHT) * fitted_xgb.predict_proba(test_candidate)[:, 1]
        )
        probability = np.clip(probability, 0.0, 1.0)
        build_submission(sample, probability, "submissions/submission_09_features.csv")
        analysis["submission_created"] = True

    (report_dir / "feature_discovery_analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n"
    )
    print(json.dumps(analysis, indent=2), flush=True)


if __name__ == "__main__":
    main()
