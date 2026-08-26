"""Run reproducible Phase 1 baselines and optionally create the first submission."""

import json
from pathlib import Path

import numpy as np

from .data import load_competition_data, split_features_target
from .modeling import build_baseline_model, fit_final_model
from .submission import build_submission
from .validation import CVConfig, cross_validate_model, evaluate_predictions


def main() -> None:
    train, test, sample = load_competition_data()
    X, y, X_test = split_features_target(train, test)
    prevalence = float(y.mean())
    trivial_oof = np.full(len(y), prevalence)
    trivial = evaluate_predictions(y, trivial_oof)

    model = build_baseline_model(random_state=42)
    folds, ml_oof = cross_validate_model(model, X, y, CVConfig())
    ml = evaluate_predictions(y, ml_oof)
    results = {
        "validation": "5-fold StratifiedKFold(shuffle=True, random_state=42)",
        "trivial_constant_prevalence": trivial,
        "logistic_regression_oof": ml,
        "folds": folds.to_dict(orient="records"),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/baseline_metrics.json").write_text(json.dumps(results, indent=2) + "\n")

    if ml["roc_auc"] <= trivial["roc_auc"]:
        raise RuntimeError("ML baseline did not beat the trivial baseline; no submission created")
    fitted = fit_final_model(model, X, y)
    probabilities = fitted.predict_proba(X_test)[:, 1]
    build_submission(sample, probabilities, "submissions/submission_01_baseline.csv")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
