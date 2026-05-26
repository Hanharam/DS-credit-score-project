# Data Directory

This folder stores the data files used for the project.

## Dataset

- Dataset: Credit Score Classification
- Source: [Kaggle](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)
- Task: Multi-class Classification
- Target: `Credit_Score`
- Classes: `Good`, `Standard`, `Poor`

## File Structure

```text
data/
├── raw/
│   └── train.csv          # 100,000 rows x 28 columns (with Credit_Score label)
├── processed/             # (optional) cached intermediate frames
└── README.md
```

## Why `test.csv` Was Removed

The Kaggle download originally shipped two files:

- `train.csv` — 100,000 rows, **includes** the `Credit_Score` target column
- `test.csv` — 50,000 rows, **does not include** `Credit_Score`

`test.csv` exists only so that Kaggle competition submissions can be scored on
their server. Locally we have no way to compute accuracy / macro-F1 / per-class
recall against it because the ground-truth labels are not in the file.

Since this project evaluates models locally (no Kaggle submission), `test.csv`
contributes nothing to the workflow and just inflates the repository size.
It was deleted on 2026-05-26.

### How evaluation is done instead

All validation happens inside `train.csv`:

1. Stratified k-fold cross-validation (`evaluate_cv`, default k = 5) over the
   full 100,000 rows — used for model comparison, GridSearchCV tuning, and the
   outlier / class-imbalance / scaler-encoder ablations.
2. A held-out 80 / 20 stratified split (`train_test_split(... stratify=y)`)
   produces the final hold-out metrics and confusion matrix saved under
   `reports/`.

### If you need `test.csv` later

Re-download from the Kaggle dataset page and drop it back into `data/raw/`.
Nothing in `src/` references it, so no code change is required to use it for
generating Kaggle-style submissions.
