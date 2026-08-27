"""Experiment 8: repeated-split robustness of the E4/E5/E6 decisions."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import t
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from .boosting_tuning import FEATURES, LIGHTGBM_BASE, XGBOOST_BASE
from .data import load_competition_data, split_features_target
from .feature_diagnostics import add_engineered_features
from .model_comparison import clone_model, prepare_native_categories

FOLD_SEEDS = [7, 21, 84, 123, 2026]
N_SPLITS = 5
MODEL_SEED = 42
BLEND_WEIGHTS = [0.3, 0.4, 0.5]
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42

CONFIGURATIONS = {
    "e4": {
        "LightGBM": {"learning_rate": 0.1, "n_estimators": 300, "num_leaves": 31},
        "XGBoost": {"learning_rate": 0.1, "n_estimators": 300, "max_depth": 6},
        "lightgbm_weight": 0.5,
    },
    "e5": {
        "LightGBM": {"learning_rate": 0.05, "n_estimators": 1200, "num_leaves": 63},
        "XGBoost": {"learning_rate": 0.05, "n_estimators": 1200, "max_depth": 7},
        "lightgbm_weight": 0.5,
    },
    "e6": {
        "LightGBM": {"learning_rate": 0.02, "n_estimators": 3808, "num_leaves": 63},
        "XGBoost": {"learning_rate": 0.02, "n_estimators": 4720, "max_depth": 7},
        "lightgbm_weight": 0.4,
    },
}

ORIGINAL_SEED42 = {
    "e4_blend": 0.9613259226414593,
    "e5_blend": 0.9639433758194731,
    "e6_blend": 0.9642892514216848,
}


def make_model(configuration: str, family: str):
    """Build the fixed deployed component without adapting it to a new seed."""
    overrides = CONFIGURATIONS[configuration][family]
    if family == "LightGBM":
        params = LIGHTGBM_BASE.copy()
        params.update(overrides)
        params["random_state"] = MODEL_SEED
        return LGBMClassifier(**params)
    params = XGBOOST_BASE.copy()
    params.update(overrides)
    params.update({"random_state": MODEL_SEED, "verbosity": 0})
    return XGBClassifier(**params)


def evaluate_component(
    configuration: str,
    family: str,
    X: pd.DataFrame,
    y: pd.Series,
    splits,
) -> tuple[np.ndarray, dict[str, object]]:
    """Generate one complete five-fold OOF vector for a fixed configuration."""
    oof = np.zeros(len(y), dtype=float)
    fold_aucs: list[float] = []
    fit_seconds = 0.0
    predict_seconds = 0.0
    model = make_model(configuration, family)
    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        print(f"{configuration} {family}: fold {fold}", flush=True)
        fitted = clone_model(model)
        start = time.perf_counter()
        fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
        fit_seconds += time.perf_counter() - start
        start = time.perf_counter()
        probability = fitted.predict_proba(X.iloc[valid_idx])[:, 1]
        predict_seconds += time.perf_counter() - start
        oof[valid_idx] = probability
        fold_aucs.append(float(roc_auc_score(y.iloc[valid_idx], probability)))
    row: dict[str, object] = {
        "configuration": configuration,
        "result_type": "component",
        "model": family,
        "lightgbm_weight": np.nan,
        "xgboost_weight": np.nan,
        "oof_auc": float(roc_auc_score(y, oof)),
        "log_loss": float(log_loss(y, oof)),
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "std_fold_auc": float(np.std(fold_aucs, ddof=1)),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
    }
    for fold, auc in enumerate(fold_aucs, start=1):
        row[f"fold_{fold}_auc"] = auc
    return oof, row


def blend_row(
    configuration: str,
    model: str,
    y: pd.Series,
    probability: np.ndarray,
    splits,
    lightgbm_weight: float,
) -> dict[str, object]:
    fold_aucs = [
        float(roc_auc_score(y.iloc[valid_idx], probability[valid_idx]))
        for _, valid_idx in splits
    ]
    row: dict[str, object] = {
        "configuration": configuration,
        "result_type": "blend",
        "model": model,
        "lightgbm_weight": lightgbm_weight,
        "xgboost_weight": 1.0 - lightgbm_weight,
        "oof_auc": float(roc_auc_score(y, probability)),
        "log_loss": float(log_loss(y, probability)),
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "std_fold_auc": float(np.std(fold_aucs, ddof=1)),
        "fit_seconds": 0.0,
        "predict_seconds": 0.0,
    }
    for fold, auc in enumerate(fold_aucs, start=1):
        row[f"fold_{fold}_auc"] = auc
    return row


def run_seed(
    fold_seed: int,
    X: pd.DataFrame,
    y: pd.Series,
    report_dir: Path,
) -> list[dict[str, object]]:
    """Run or resume all six fixed components for one fold seed."""
    checkpoint = report_dir / f"repeated_cv_seed_{fold_seed}_oof.npz"
    metric_checkpoint = report_dir / f"repeated_cv_seed_{fold_seed}_checkpoint.csv"
    saved: dict[str, np.ndarray] = {}
    if checkpoint.exists():
        archive = np.load(checkpoint)
        saved = {name: archive[name] for name in archive.files}
    saved_rows: dict[str, dict[str, object]] = {}
    if metric_checkpoint.exists():
        checkpoint_rows = pd.read_csv(metric_checkpoint)
        saved_rows = {
            f"{row['configuration']}_{str(row['model']).lower()}": row
            for row in checkpoint_rows.to_dict("records")
        }
    splitter = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=fold_seed
    )
    splits = list(splitter.split(X, y))
    rows: list[dict[str, object]] = []
    for configuration in ["e4", "e5", "e6"]:
        for family in ["LightGBM", "XGBoost"]:
            key = f"{configuration}_{family.lower()}"
            if key in saved and key in saved_rows:
                print(f"seed {fold_seed}: reuse {key}", flush=True)
                probability = saved[key]
                row = saved_rows[key]
            else:
                probability, row = evaluate_component(
                    configuration, family, X, y, splits
                )
                saved[key] = probability
                np.savez_compressed(checkpoint, **saved)
                saved_rows[key] = row
                pd.DataFrame(saved_rows.values()).to_csv(
                    metric_checkpoint, index=False
                )
            row["fold_seed"] = fold_seed
            rows.append(row)

        lightgbm = saved[f"{configuration}_lightgbm"]
        xgboost = saved[f"{configuration}_xgboost"]
        selected_weight = float(CONFIGURATIONS[configuration]["lightgbm_weight"])
        probability = selected_weight * lightgbm + (1.0 - selected_weight) * xgboost
        row = blend_row(
            configuration,
            f"{configuration.upper()} selected blend",
            y,
            probability,
            splits,
            selected_weight,
        )
        row["fold_seed"] = fold_seed
        rows.append(row)

        if configuration == "e6":
            for weight in BLEND_WEIGHTS:
                grid_probability = weight * lightgbm + (1.0 - weight) * xgboost
                grid_row = blend_row(
                    "e6",
                    f"E6 grid LGB {weight:.1f}",
                    y,
                    grid_probability,
                    splits,
                    weight,
                )
                grid_row["fold_seed"] = fold_seed
                rows.append(grid_row)
    return rows


def paired_summary(
    differences: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict[str, object], np.ndarray]:
    """Summarize five paired seed differences without treating folds as independent."""
    differences = np.asarray(differences, dtype=float)
    count = len(differences)
    bootstrap = differences[
        rng.integers(0, count, size=(BOOTSTRAP_RESAMPLES, count))
    ].mean(axis=1)
    mean = float(differences.mean())
    standard_error = float(differences.std(ddof=1) / np.sqrt(count))
    t_critical = float(t.ppf(0.975, df=count - 1))
    summary = {
        "mean_difference": mean,
        "median_difference": float(np.median(differences)),
        "std_difference": float(differences.std(ddof=1)),
        "min_difference": float(differences.min()),
        "max_difference": float(differences.max()),
        "wins": int((differences > 0).sum()),
        "ties": int((differences == 0).sum()),
        "losses": int((differences < 0).sum()),
        "paired_t_ci_95_lower": mean - t_critical * standard_error,
        "paired_t_ci_95_upper": mean + t_critical * standard_error,
        "seed_bootstrap_ci_95_lower": float(np.quantile(bootstrap, 0.025)),
        "seed_bootstrap_ci_95_upper": float(np.quantile(bootstrap, 0.975)),
        "seed_bootstrap_positive": float((bootstrap > 0).mean()),
        "n_independent_seeds": count,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    return summary, bootstrap


def analyze_results(results: pd.DataFrame, report_dir: Path) -> dict[str, object]:
    selected = results[
        (results["result_type"] == "blend")
        & results["model"].str.contains("selected blend")
    ].pivot(index="fold_seed", columns="configuration", values="oof_auc")
    comparisons = pd.DataFrame(index=selected.index)
    comparisons["e5_minus_e4"] = selected["e5"] - selected["e4"]
    comparisons["e6_minus_e5"] = selected["e6"] - selected["e5"]
    comparisons["e6_minus_e4"] = selected["e6"] - selected["e4"]
    comparisons = comparisons.reset_index()
    comparisons.to_csv(report_dir / "repeated_cv_comparisons.csv", index=False)

    grid = results[
        (results["configuration"] == "e6")
        & results["model"].str.contains("grid")
    ][
        ["fold_seed", "lightgbm_weight", "xgboost_weight", "oof_auc", "log_loss"]
    ].copy()
    grid.to_csv(report_dir / "repeated_cv_blend_grid.csv", index=False)
    grid_pivot = grid.pivot(index="fold_seed", columns="lightgbm_weight", values="oof_auc")
    grid_differences = pd.DataFrame(
        {
            "fold_seed": grid_pivot.index,
            "weight_04_minus_03": grid_pivot[0.4] - grid_pivot[0.3],
            "weight_04_minus_05": grid_pivot[0.4] - grid_pivot[0.5],
        }
    ).reset_index(drop=True)
    grid_differences.to_csv(
        report_dir / "repeated_cv_blend_comparisons.csv", index=False
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparison_summaries: dict[str, object] = {}
    bootstrap_columns: dict[str, np.ndarray] = {}
    for column in ["e5_minus_e4", "e6_minus_e5", "e6_minus_e4"]:
        summary, bootstrap = paired_summary(comparisons[column].to_numpy(), rng)
        comparison_summaries[column] = summary
        bootstrap_columns[column] = bootstrap
    blend_summaries: dict[str, object] = {}
    for column in ["weight_04_minus_03", "weight_04_minus_05"]:
        summary, bootstrap = paired_summary(grid_differences[column].to_numpy(), rng)
        blend_summaries[column] = summary
        bootstrap_columns[column] = bootstrap
    pd.DataFrame(bootstrap_columns).to_csv(
        report_dir / "repeated_cv_bootstrap_samples.csv", index=False
    )

    aggregate_rows = []
    for (configuration, result_type, model), group in results.groupby(
        ["configuration", "result_type", "model"], sort=False
    ):
        aggregate_rows.append(
            {
                "configuration": configuration,
                "result_type": result_type,
                "model": model,
                "lightgbm_weight": group["lightgbm_weight"].iloc[0],
                "mean_seed_auc": group["oof_auc"].mean(),
                "median_seed_auc": group["oof_auc"].median(),
                "std_seed_auc": group["oof_auc"].std(ddof=1),
                "min_seed_auc": group["oof_auc"].min(),
                "max_seed_auc": group["oof_auc"].max(),
                "total_fit_seconds": group["fit_seconds"].sum(min_count=1),
                "total_predict_seconds": group["predict_seconds"].sum(min_count=1),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(report_dir / "repeated_cv_summary.csv", index=False)

    original_differences = {
        "e5_minus_e4": ORIGINAL_SEED42["e5_blend"] - ORIGINAL_SEED42["e4_blend"],
        "e6_minus_e5": ORIGINAL_SEED42["e6_blend"] - ORIGINAL_SEED42["e5_blend"],
        "e6_minus_e4": ORIGINAL_SEED42["e6_blend"] - ORIGINAL_SEED42["e4_blend"],
    }
    original_context = {}
    for name, value in original_differences.items():
        new = comparisons[name].to_numpy()
        original_context[name] = {
            "original_seed42_difference": float(value),
            "additional_seed_mean": float(new.mean()),
            "additional_seed_median": float(np.median(new)),
            "original_minus_additional_mean": float(value - new.mean()),
            "original_percentile_among_additional_seeds": float((new <= value).mean()),
        }

    analysis = {
        "experiment": 8,
        "purpose": "confirmation_only_no_tuning",
        "fold_seeds": FOLD_SEEDS,
        "n_splits_per_seed": N_SPLITS,
        "independent_units_for_inference": "five repeated split seeds, not 25 folds",
        "configurations": CONFIGURATIONS,
        "selected_blend_comparisons": comparison_summaries,
        "e6_blend_weight_comparisons": blend_summaries,
        "original_seed42_context": original_context,
        "limitations": [
            "Five seeds provide a small paired sample. Bootstrap intervals are discrete, "
            "and t intervals rely on an approximate normality assumption.",
            "The original seed-42 E6 score used fold-local early stopping to choose the "
            "reported full-data round counts. E8 applies those fixed deployed round counts "
            "to every new fold, so the seed-42 context is informative but not a perfectly "
            "identical procedural replicate.",
        ],
        "decision_rule": "E6 remains champion unless repeated validation gives meaningful evidence against it.",
        "ignored_local_artifacts": [
            f"reports/repeated_cv_seed_{seed}_oof.npz" for seed in FOLD_SEEDS
        ] + [
            f"reports/repeated_cv_seed_{seed}_checkpoint.csv" for seed in FOLD_SEEDS
        ] + ["reports/repeated_cv_bootstrap_samples.csv"],
    }
    (report_dir / "repeated_cv_analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n"
    )
    return analysis


def main() -> None:
    train, test, _ = load_competition_data()
    X, y, X_test = split_features_target(train, test)
    X_native, _, _ = prepare_native_categories(X, X_test)
    X_engineered = add_engineered_features(X_native, FEATURES)
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    output_path = report_dir / "repeated_cv_seed_results.csv"

    all_rows: list[dict[str, object]] = []
    completed_seeds: set[int] = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        for seed, group in existing.groupby("fold_seed"):
            if len(group) == 12:
                all_rows.extend(group.to_dict("records"))
                completed_seeds.add(int(seed))
    for seed in FOLD_SEEDS:
        if seed in completed_seeds:
            print(f"reuse complete seed {seed}", flush=True)
            continue
        print(f"start fold seed {seed}", flush=True)
        seed_rows = run_seed(seed, X_engineered, y, report_dir)
        all_rows.extend(seed_rows)
        pd.DataFrame(all_rows).to_csv(output_path, index=False)

    results = pd.DataFrame(all_rows).sort_values(
        ["fold_seed", "configuration", "result_type", "model"]
    )
    results.to_csv(output_path, index=False)
    analysis = analyze_results(results, report_dir)
    print(json.dumps(analysis, indent=2), flush=True)


if __name__ == "__main__":
    main()
