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

Both Experiment 2 models achieved a public Kaggle ROC AUC of **0.96197** (`submission_02_model.csv` for LightGBM and `submission_02_xgboost.csv` for XGBoost). These public scores are recorded for context only and were not used to tune or select the Experiment 3 blend.

Native split-count importance for the fitted LightGBM is reconnaissance only: notifications per day (18.28%), app opens per day (18.26%), daily screen time (13.73%), weekend screen time (12.67%), and social media hours (11.52%) rank highest. Split-count importance is biased toward features with many candidate thresholds and is not causal.

## Probability blending

Experiment 3 regenerated complete paired OOF predictions from the unchanged LightGBM and XGBoost configurations on the exact Experiment 2 folds. Their probabilities are highly correlated (Pearson 0.994432; Spearman 0.991288), but differ by 0.018883 on average and produce different 0.5-threshold classes for 1.9905% of rows. Disagreement is higher for true negatives (3.4078%) than true positives (1.4101%).

Every interior LightGBM weight from 0.1 through 0.9 beats both individual models. The numerical optimum is the simple 50/50 blend:

| LightGBM weight | XGBoost weight | OOF AUC ↑ | Log loss ↓ | Δ vs LightGBM | Δ vs XGBoost |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 1.0 | 0.960596 | 0.231060 | -0.000007 | 0.000000 |
| 0.1 | 0.9 | 0.960755 | 0.230670 | +0.000151 | +0.000158 |
| 0.2 | 0.8 | 0.960878 | 0.230353 | +0.000274 | +0.000282 |
| 0.3 | 0.7 | 0.960967 | 0.230104 | +0.000364 | +0.000371 |
| 0.4 | 0.6 | 0.961023 | 0.229922 | +0.000419 | +0.000426 |
| **0.5** | **0.5** | **0.961044** | 0.229809 | **+0.000441** | **+0.000448** |
| 0.6 | 0.4 | 0.961032 | **0.229765** | +0.000429 | +0.000436 |
| 0.7 | 0.3 | 0.960987 | 0.229796 | +0.000383 | +0.000390 |
| 0.8 | 0.2 | 0.960906 | 0.229910 | +0.000302 | +0.000309 |
| 0.9 | 0.1 | 0.960785 | 0.230128 | +0.000182 | +0.000189 |
| 1.0 | 0.0 | 0.960604 | 0.232012 | 0.000000 | +0.000007 |

A 500-resample paired stratified bootstrap with seed 42 estimates the 50/50 AUC gain over LightGBM at 0.000440 (95% CI 0.000390–0.000487) and over XGBoost at 0.000449 (95% CI 0.000409–0.000488). The blend beat each model in 100% of bootstrap samples. Because the improvement spans a broad range and the optimum is flat, the robust 50/50 blend—not a finely selected weight—was fitted and written to `submissions/submission_03_blend.csv`. It has not been submitted.

The Experiment 3 blend achieved a public Kaggle ROC AUC of **0.96236**. This score is recorded for context only and was not used for Experiment 4 feature selection.

## Feature diagnostics and conservative engineering

Experiment 4 used unchanged LightGBM and XGBoost configurations and the exact existing folds. Leave-one-feature-out LightGBM ablations identify daily screen time (-0.011386 AUC), weekend screen time (-0.010317), notifications per day (-0.008683), app opens per day (-0.007472), and social media hours (-0.007145) as the largest marginal contributors. Dropping gender (+0.000026) or stress level (+0.000055) produced tiny apparent gains that are too small to justify removing them without cross-model confirmation.

Missingness target-rate differences are modest. Using a predeclared absolute difference threshold of 0.003 selected age, sleep hours, and app opens per day. Explicit indicators for these features reproduce the native-missing control exactly; indicators for all 12 features slightly reduce AUC by 0.000009. This suggests native tree missing-value handling is sufficient, despite train/test missing-rate shifts as large as 3.38 percentage points.

Eight interpretable features were evaluated individually. The best is `leisure_screen_proxy = daily_screen_time_hours - work_study_hours`, based on the explicit assumption that work/study hours are a component of total daily screen time. It improves LightGBM from 0.960604 to 0.960797. The all-positive three-feature combination reaches only 0.960640, so extra derived columns dilute the strongest feature rather than adding complementary signal.

The single selected feature transfers to unchanged XGBoost and the blend:

| Model | Raw OOF AUC | Engineered OOF AUC | Component gain | Δ vs E3 blend |
|---|---:|---:|---:|---:|
| LightGBM | 0.960604 | 0.960797 | +0.000194 | -0.000247 |
| XGBoost | 0.960596 | 0.960905 | +0.000309 | -0.000139 |
| 50/50 blend | 0.961044 | **0.961326** | — | **+0.000282** |

A 500-resample paired stratified bootstrap with seed 42 estimates the engineered-blend gain over E3 at 0.000282 (95% CI 0.000238–0.000326), positive in every resample. Because the feature transfers across both models and improves their blend with a positive paired interval, `submissions/submission_04_features.csv` was generated and validated locally. It has not been submitted.

## Conservative boosting tuning

Experiment 5 keeps the exact Experiment 4 feature set and folds. Eight LightGBM and nine XGBoost candidates—including each E4 control—cover slower learning, neighboring capacity, regularization, and row/column sampling patterns. Fold-local early stopping uses the validation fold for candidate evaluation; no leaderboard results enter selection.

Improvement is broad. LightGBM's 0.05/1,200-tree variants range from 0.961325 to 0.963223, with regularized neighbors at 0.963026 and 0.962954. XGBoost depth 7 reaches 0.963705, closely supported by row/column sampling at 0.963672, regularization at 0.963414, and the plain 0.05 learning-rate candidate at 0.963392.

Selected configurations:

- LightGBM: 1,200 estimators, learning rate 0.05, 63 leaves; other parameters unchanged.
- XGBoost: 1,200 estimators, learning rate 0.05, depth 7; subsample and column sample 1.0, other parameters unchanged.

The tuned-pair weight curve is flat near its optimum: LightGBM weights 0.3, 0.4, and 0.5 score 0.963963, 0.963972, and 0.963943. Although 40/60 is numerically best, its advantage over 50/50 is only 0.000029, so the symmetric 50/50 blend is retained to reduce same-OOF weight-selection sensitivity.

| Model | E4 OOF AUC | Tuned OOF AUC | Gain |
|---|---:|---:|---:|
| LightGBM | 0.960797 | 0.963223 | +0.002426 |
| XGBoost | 0.960905 | 0.963705 | +0.002800 |
| 50/50 blend | 0.961326 | **0.963943** | **+0.002617** |

A 500-resample paired stratified bootstrap with seed 42 estimates the tuned 50/50 gain over E4 at 0.002618 (95% CI 0.002547–0.002704), positive in every resample. Both selected candidates are supported by neighboring improvements and the final blend is simple, so `submissions/submission_05_tuned.csv` was generated and validated locally. It has not been submitted.

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
| EXP-003 | 2026-08-26 | Same 5 folds as EXP-002 | Paired LightGBM/XGBoost weight grid; 50/50 blend | ROC AUC 0.961044 | `submission_03_blend.csv` | Complete |
| EXP-004 | 2026-08-26 | Same 5 folds as EXP-003 | Ablations, missingness, 8 conservative candidates; leisure-screen proxy selected | ROC AUC 0.961326 | `submission_04_features.csv` | Complete |
| EXP-005 | 2026-08-26 | Same 5 folds as EXP-004 | 8 LightGBM and 9 XGBoost conservative candidates; tuned 50/50 blend | ROC AUC 0.963943 | `submission_05_tuned.csv` | Complete |

## Reproducing

```bash
uv run python -m kaggle_smartphone_addiction.run_baseline
uv run compare-models
uv run evaluate-blend
uv run diagnose-features
uv run tune-boosters
uv run jupyter nbconvert --to notebook --execute notebooks/01_reconnaissance.ipynb --output 01_reconnaissance.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/02_model_comparison.ipynb --output 02_model_comparison.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/03_blending.ipynb --output 03_blending.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/04_feature_diagnostics.ipynb --output 04_feature_diagnostics.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/05_boosting_tuning.ipynb --output 05_boosting_tuning.ipynb
```

Detailed results are written under `reports/`, including complete Experiment 3 OOF predictions, blend scores, diversity statistics, and bootstrap samples. Reusable loaders, schema definitions, CV, evaluation, model comparison, blending, final fitting, and submission validation live under `src/kaggle_smartphone_addiction/`.
