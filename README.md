# Kaggle Playground Series S6E8 — Smartphone Addiction

Phase 1 established local-only reconnaissance and a reusable baseline system. Phase 2 compares five untuned model families. No public notebooks, discussions, external datasets, or leaderboard strategies were inspected.

## Problem and data

- Problem: binary classification; predict the probability that `addicted_label == 1`.
- Target: `addicted_label`.
- ID: `id` (unique, non-overlapping, sequential train/test ranges; excluded from modeling).
- Train: 691,369 rows × 14 columns (ID + 12 features + target).
- Test: 296,302 rows × 13 columns (ID + the same 12 features).
- Features: 9 numeric and 3 categorical; every modeling feature has missing values.
- Target distribution: 490,474 positive (70.9424%) and 200,895 negative (29.0576%).
- Submission: 296,302 rows with columns `id, addicted_label`; predictions are probabilities in `[0, 1]` and IDs must match test order exactly.

## Metric and validation

The working official competition metric is **ROC AUC**. Log loss and accuracy at 0.5 are also recorded as diagnostics.

All model comparisons use one reproducible scheme: 5-fold `StratifiedKFold(shuffle=True, random_state=42)`. Every family receives the identical precomputed row splits. Reported model scores are computed once from complete out-of-fold probabilities. Final test predictions are made only after refitting the selected pipeline on all training rows.

## Baseline results

| Experiment | Model | OOF ROC AUC ↑ | OOF log loss ↓ | OOF accuracy @ 0.5 ↑ |
|---|---|---:|---:|---:|
| EXP-000 | Constant training prevalence (0.709424) | 0.500000 | 0.602666 | 0.709424 |
| EXP-001 | Median/mode imputation + missing indicators + scaling/one-hot + logistic regression | 0.913785 | 0.338347 | 0.842285 |

The ML fold ROC AUC range is 0.912733–0.914733. Because EXP-001 clearly beats the trivial baseline, `submissions/submission_01_baseline.csv` was generated and schema-validated locally. It has not been submitted.

## Model-family comparison

Conservative near-default settings were used without feature engineering or broad tuning. Logistic regression retains its imputation/one-hot pipeline. The tree models preserve numeric missing values; HistGradientBoosting, XGBoost, and LightGBM use aligned pandas categorical dtypes, while CatBoost receives the categorical columns natively with an explicit missing category.

| Model | OOF AUC ↑ | Mean fold AUC | Fold AUC SD ↓ | OOF log loss ↓ | Δ vs logistic | CV fit time | CV predict time | Max serialized model |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | **0.960604** | 0.960605 | 0.000769 | 0.232012 | +0.046819 | 12.61 s | 1.44 s | 1.06 MB |
| XGBoost | 0.960596 | 0.960598 | **0.000651** | **0.231060** | +0.046811 | 11.91 s | 0.38 s | 1.39 MB |
| HistGradientBoosting | 0.957157 | 0.957160 | 0.000748 | 0.242273 | +0.043372 | 12.55 s | 0.99 s | 0.64 MB |
| CatBoost | 0.953305 | 0.953308 | 0.000689 | 0.251253 | +0.039520 | 103.04 s | **0.21 s** | 0.34 MB |
| LogisticRegression | 0.913785 | 0.913787 | 0.000861 | 0.338347 | — | **4.91 s** | 0.38 s | **0.004 MB** |

LightGBM is the current champion by OOF ROC AUC. Its 0.0000074 advantage over XGBoost is negligible, so XGBoost remains effectively tied and has slightly better consistency, log loss, and prediction runtime. LightGBM was selected deterministically by OOF AUC, then fold consistency and runtime. It was refit on all training rows, and `submissions/submission_02_model.csv` was schema-validated locally. It has not been submitted.

Native split-count importance for the fitted LightGBM is reconnaissance only: notifications per day (18.28%), app opens per day (18.26%), daily screen time (13.73%), weekend screen time (12.67%), and social media hours (11.52%) rank highest. Split-count importance is biased toward features with many candidate thresholds and is not causal.

## Reconnaissance notes

- No full duplicate rows, duplicate IDs, or duplicate feature rows occur within train or test.
- Train/test IDs do not overlap. Two exact feature vectors appear across train and test; this is too small to support a lookup approach and should not be used as validation evidence.
- Numeric train/test distributions are very close (maximum two-sample KS statistic ≈ 0.0027), while feature missingness shifts by up to about 3.4 percentage points.
- Categorical level proportions are close; the largest total-variation distance is about 0.0228 for `academic_work_impact`, driven mainly by missingness.
- No high-cardinality categoricals exist: cardinalities are 2–3 observed levels per categorical (plus missing).
- `daily_screen_time_hours`, `weekend_screen_time`, and `social_media_hours` are strongly associated with the target. This is plausible signal but deserves leakage scrutiny because the target definition is unknown.
- `id` is a row identifier and possible generation-order proxy; it is excluded. Its point-biserial/Pearson correlation with the target is only about 0.0011.

## Experiment log

| ID | Date | Validation | Change | Primary score | Artifact | Status |
|---|---|---|---|---:|---|---|
| EXP-000 | 2026-08-26 | 5-fold stratified CV, seed 42 | Constant prevalence | ROC AUC 0.500000 | — | Complete |
| EXP-001 | 2026-08-26 | 5-fold stratified CV, seed 42 | Untuned logistic pipeline | ROC AUC 0.913785 | `submission_01_baseline.csv` | Complete |
| EXP-002 | 2026-08-26 | Same 5 folds as EXP-001 | Five untuned model families; LightGBM champion | ROC AUC 0.960604 | `submission_02_model.csv` | Complete |

## Reproducing

```bash
uv run python -m kaggle_smartphone_addiction.run_baseline
uv run compare-models
uv run jupyter nbconvert --to notebook --execute notebooks/01_reconnaissance.ipynb --output 01_reconnaissance.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/02_model_comparison.ipynb --output 02_model_comparison.ipynb
```

Detailed results are written to `reports/baseline_metrics.json`, `reports/model_comparison.csv`, and `reports/feature_importance.csv`. Reusable loaders, schema definitions, CV, evaluation, model comparison, final fitting, and submission validation live under `src/kaggle_smartphone_addiction/`.
