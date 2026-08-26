# Kaggle Playground Series S6E8 — Smartphone Addiction

Phase 1 establishes local-only reconnaissance and a reusable baseline system. No public notebooks, discussions, external datasets, or leaderboard strategies were inspected.

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

The sample submission contains probabilities, but the official competition metric cannot be confirmed reliably from the supplied CSVs alone. **Working assumption: ROC AUC**. Log loss and accuracy at 0.5 are also recorded so experiments remain interpretable if the official metric differs.

All Phase 1 model comparisons use one reproducible scheme: 5-fold `StratifiedKFold(shuffle=True, random_state=42)`. Reported model scores are computed once from complete out-of-fold probabilities. Final test predictions are made only after refitting the selected pipeline on all training rows.

## Baseline results

| Experiment | Model | OOF ROC AUC ↑ | OOF log loss ↓ | OOF accuracy @ 0.5 ↑ |
|---|---|---:|---:|---:|
| EXP-000 | Constant training prevalence (0.709424) | 0.500000 | 0.602666 | 0.709424 |
| EXP-001 | Median/mode imputation + missing indicators + scaling/one-hot + logistic regression | 0.913785 | 0.338347 | 0.842285 |

The ML fold ROC AUC range is 0.912733–0.914733. Because EXP-001 clearly beats the trivial baseline, `submissions/submission_01_baseline.csv` was generated and schema-validated locally. It has not been submitted.

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

## Reproducing

```bash
uv run python -m kaggle_smartphone_addiction.run_baseline
uv run jupyter nbconvert --to notebook --execute notebooks/01_reconnaissance.ipynb --output 01_reconnaissance.ipynb
```

Detailed metrics are written to `reports/baseline_metrics.json`. Reusable loaders, schema definitions, CV, evaluation, final fitting, and submission validation live under `src/kaggle_smartphone_addiction/`.
