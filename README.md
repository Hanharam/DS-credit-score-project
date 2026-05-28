# Credit Score Classification

Data Science Term Project - team 6

End-to-end pipeline that predicts a customer's credit score class (`Good` / `Standard` / `Poor`) from 28 personal and financial-behaviour features, with explicit ablations on **extreme outlier handling** and **class imbalance handling**.

### Headline results (full 100k, Stratified 5-fold + GridSearchCV)

| Metric | Proposal target | Achieved (RandomForest, tuned) |
|---|---|---|
| Macro-F1 (CV) | ≥ 0.75 | **0.7697 ± 0.0041** |
| Macro-F1 (hold-out 20k) | — | **0.8114** |
| Macro-Recall (hold-out) | — | **0.8122** |
| Per-class Recall | ≥ 0.70 | Good 0.777 / Poor 0.837 / Standard 0.823 |
| Accuracy (hold-out) | — | 0.8187 |

Best params: `n_estimators=400, max_depth=None, min_samples_leaf=1`. Full result tables, ablation findings, and auxiliary-task scores are in [section 7](#7-results-full-run-100000-rows-stratified-5-fold-gridsearchcv).

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Project Structure](#2-project-structure)
- [3. Environment Setup](#3-environment-setup)
- [4. Quickstart](#4-quickstart)
- [5. What the Pipeline Does](#5-what-the-pipeline-does)
  - [5.1 Cleaning](#51-cleaning-srcpreprocessingpy)
  - [5.2 Imputation](#52-imputation)
  - [5.3 Feature Engineering](#53-feature-engineering)
  - [5.4 Preprocessing Matrix](#54-preprocessing-matrix-build_preprocessor)
  - [5.5 Models](#55-models-srcmodelpy)
  - [5.6 Evaluation](#56-evaluation-srcevaluatepy)
  - [5.7 Ablations](#57-ablations)
  - [5.8 Auxiliary Tasks](#58-auxiliary-tasks-srcauxiliarypy)
- [6. Output Artifacts](#6-output-artifacts-reports)
- [7. Results](#7-results-full-run-100000-rows-stratified-5-fold-gridsearchcv)
  - [7.1 Baseline Model CV](#71-stratified-5-fold-cv--baseline-models)
  - [7.2 Tuned Best Model](#72-gridsearchcv-tuned-best-model--final-hold-out)
  - [7.3 Outlier Handling Ablation](#73-outlier-handling-ablation-randomforest-5-fold)
  - [7.4 Class Imbalance Ablation](#74-class-imbalance-handling-ablation-randomforest-5-fold)
  - [7.5 Scaler × Encoder Sweep](#75-scaler--encoder-sweep-decisiontree-5-fold)
  - [7.6 Auxiliary Tasks](#76-auxiliary-tasks)
  - [7.7 Summary Against Proposal](#77-summary-against-the-proposal)
- [8. Notebooks vs `src/` Modules](#8-notebooks-vs-src-modules)
- [9. Known Issues / Notes](#9-known-issues--notes)

---

## 1. Project Overview

- Dataset: [Credit Score Classification (Kaggle)](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)
- Task: Multi-class classification (3 classes, imbalanced: Standard 53% / Poor 29% / Good 18%)
- Target: `Credit_Score`
- Models compared: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- Primary metric: Macro-F1 (chosen because of class imbalance)
- Secondary metrics: per-class Recall, Accuracy, Confusion Matrix
- Auxiliary tasks: regression on `Monthly_Balance`, KMeans clustering on payment-behaviour signals

The pipeline covers every stage of an industry-style ML workflow:

1. EDA (Jupyter notebooks)
2. Data cleaning — placeholder removal, dtype coercion, domain-based outlier cutoffs, optional percentile trimming
3. Group-aware missing-value imputation across the 8 monthly records per `Customer_ID`
4. Feature engineering — `Debt_to_Income`, `EMI_to_Salary`, `Savings_Rate`, `Delay_per_Loan`
5. Multi-label encoding of `Type_of_Loan` via `MultiLabelBinarizer`
6. Configurable preprocessing pipeline: 3 scalers × 3 encoders
7. Class imbalance handling: none / `class_weight="balanced"` / SMOTE
8. Stratified k-fold cross-validation and `GridSearchCV` tuning
9. Hold-out evaluation with confusion matrix
10. Persisted artifacts (CSV reports, PNG figures, fitted pipeline)

---

## 2. Project Structure

```text
DataScience_termProject/
├── README.md
├── environment.yml             # conda environment definition
├── requirements.txt            # legacy pip mirror
├── .gitignore
│
├── TermProject_Proposal_team6.docx
│
├── data/
│   ├── raw/
│   │   └── train.csv           # 100,000 x 28 (test.csv removed - see data/README.md)
│   └── README.md
│
├── notebooks/
│   ├── 01_eda.ipynb            # target dist, missing, dirty-value scan, boxplots, categorical dist
│   ├── 02_preprocessing.ipynb  # initial cleaning experiments
│   └── 03_model_test.ipynb     # quick baseline comparison
│
├── src/                        # production-style modular code
│   ├── __init__.py
│   ├── preprocessing.py        # clean_data, group_impute, encode_type_of_loan,
│   │                           # add_engineered_features, build_preprocessor
│   ├── eda.py                  # programmatic EDA mirroring 01_eda.ipynb
│   ├── visualize.py            # all plot generation (EDA + ablations + CV)
│   ├── model.py                # get_models, get_param_grids, build_pipeline (SMOTE-aware)
│   ├── evaluate.py             # evaluate_cv, evaluate_holdout, tune_with_gridsearch,
│   │                           # ablation_outlier, ablation_imbalance, ablation_preprocessing
│   ├── auxiliary.py            # regress_monthly_balance, cluster_payment_behaviour, kmeans_elbow
│   └── train.py                # end-to-end CLI
│
└── reports/                    # generated by src/train.py
    ├── eda/                    # EDA tables (target dist, missing, dirty values, describe, ...)
    │   ├── target_distribution.csv
    │   ├── missing_report.csv
    │   ├── dirty_values.csv
    │   ├── numeric_describe_raw.csv
    │   ├── numeric_describe_clean.csv
    │   ├── outlier_summary.csv
    │   ├── correlation_matrix.csv
    │   └── categorical_distributions.csv
    ├── figures/
    │   ├── eda/                # fig01_target_distribution.png ... fig06_boxplot_by_target.png
    │   ├── ablation_outlier.png
    │   ├── ablation_imbalance.png
    │   ├── ablation_preprocessing.png
    │   ├── cv_baseline.png
    │   ├── confusion_matrix.png
    │   ├── per_class_metrics.png
    │   └── aux_kmeans_elbow.png
    ├── ablation_outlier.csv
    ├── ablation_imbalance.csv
    ├── ablation_preprocessing.csv
    ├── cv_baseline.csv
    ├── final_summary.json
    ├── aux_regression.json
    ├── aux_kmeans_summary.json
    ├── aux_kmeans_centroids.csv
    ├── aux_kmeans_elbow.csv
    └── final_pipeline.joblib   # opt-in via --save-pipeline (large; gitignored)
```

---

## 3. Environment Setup

### Recommended: conda

```bash
conda env create -f environment.yml
conda activate ds_credit
```

This installs Python 3.11, scikit-learn 1.5.x, pandas, numpy, matplotlib, seaborn, jupyter, ipykernel, joblib, plus `llvm-openmp` / `intel-openmp` (required on Windows so SMOTE's nearest-neighbour search does not crash with a delay-load DLL error), and via pip: `imbalanced-learn`, `mlflow`.

### Fallback: pip / venv

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install imbalanced-learn
```

> On Windows, the bare `pip install scikit-learn` path may hit an OpenMP DLL load failure inside SMOTE. The `environment.yml` route is the supported one.

---

## 4. Quickstart

### 4.1 Full run (full dataset, k = 5, GridSearchCV, all ablations, auxiliary tasks)

```bash
python -m src.train --all
```

### 4.2 Fast smoke-test (subsampled, k = 3, no GridSearch)

```bash
python -m src.train --sample-rows 10000 --cv-folds 3 --skip-grid --all
```

### 4.3 EDA tables + figures only (no model training)

```bash
python -m src.train --run-eda --skip-grid
```

### 4.4 Only specific ablations

```bash
python -m src.train --run-outlier-ablation
python -m src.train --run-imbalance-ablation
python -m src.train --run-preproc-ablation --preproc-model DecisionTree
python -m src.train --run-auxiliary
```

### 4.5 Persist the fitted final pipeline

```bash
python -m src.train --all --save-pipeline   # writes reports/final_pipeline.joblib (can be >1 GB)
```

### 4.6 All CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--train-csv PATH` | `data/raw/train.csv` | Path to raw training CSV |
| `--out-dir PATH` | `reports/` | Where to write CSV/JSON/PNG/joblib artifacts |
| `--cv-folds N` | `5` | Stratified k-fold splits |
| `--sample-rows N` | `0` (= use all) | Subsample for fast experiments |
| `--skip-grid` | off | Skip `GridSearchCV` tuning |
| `--run-eda` | off | Write EDA tables under `reports/eda/` and figures under `reports/figures/eda/` |
| `--run-outlier-ablation` | off | Compare `none` / `domain` / `domain_percentile` |
| `--run-imbalance-ablation` | off | Compare `none` / `class_weight` / `SMOTE` |
| `--run-preproc-ablation` | off | Sweep 3 scalers × 3 encoders |
| `--run-auxiliary` | off | Run regression + KMeans auxiliary tasks |
| `--save-pipeline` | off | Persist fitted pipeline (large file, opt-in) |
| `--ablation-models M1 M2 ...` | `DecisionTree RandomForest` | Models used inside outlier / imbalance ablations |
| `--preproc-model NAME` | `DecisionTree` | Model used inside the scaler × encoder sweep |
| `--n-clusters N` | `4` | KMeans cluster count |
| `--all` | off | Enable EDA + all ablations + auxiliary tasks |

Every applicable stage automatically saves a matching PNG into `reports/figures/`. Ablation CSVs and figures stay in sync — no separate `--save-plots` flag is needed.

---

## 5. What the Pipeline Does

### 5.1 Cleaning (`src/preprocessing.py`)

- Replace placeholders `_`, `_______`, `!@9#%8` with `NaN`
- Coerce object-typed numeric columns (`Age`, `Annual_Income`, `Outstanding_Debt`, ...) to numeric, stripping trailing underscores
- Convert `Credit_History_Age` (e.g. `"22 Years and 1 Months"`) to integer months
- Drop `ID`, `Name`, `SSN`; keep `Customer_ID` until group-imputation is done, then drop
- Domain outlier cutoffs (e.g. `Age` ∈ [18, 100], `Interest_Rate` ∈ [0, 100], `Num_Bank_Accounts` ∈ [0, 20])
- Optional 0.5%–99.5% percentile trim on numeric columns (`outlier_mode="domain_percentile"`)

### 5.2 Imputation

- Per-`Customer_ID` forward-fill then backward-fill across the 8 monthly records, preserving temporal locality
- Remaining gaps imputed inside the `ColumnTransformer` with median (numeric) / most-frequent (categorical)

### 5.3 Feature Engineering

- `Debt_to_Income = Outstanding_Debt / (Annual_Income + 1)`
- `EMI_to_Salary  = Total_EMI_per_month / (Monthly_Inhand_Salary + 1)`
- `Savings_Rate   = Amount_invested_monthly / (Monthly_Inhand_Salary + 1)`
- `Delay_per_Loan = Num_of_Delayed_Payment / (Num_of_Loan + 1)`
- `Type_of_Loan` exploded by comma, then `MultiLabelBinarizer` → one binary column per loan kind

### 5.4 Preprocessing matrix (`build_preprocessor`)

| Scaler | Encoder |
|---|---|
| `StandardScaler` | `OneHotEncoder` |
| `MinMaxScaler` | `OrdinalEncoder` |
| `RobustScaler` | `TargetEncoder` (sklearn ≥ 1.3 built-in) |

### 5.5 Models (`src/model.py`)

Baseline → ensembles, all with `class_weight="balanced"` where supported:

- `LogisticRegression(max_iter=2000)`
- `DecisionTreeClassifier(max_depth=15)`
- `RandomForestClassifier(n_estimators=200, max_depth=20)`
- `GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3)`

`GridSearchCV` grids are defined per-model in `get_param_grids()`.

### 5.6 Evaluation (`src/evaluate.py`)

- `evaluate_cv`: Stratified k-fold returning mean ± std for accuracy / macro-F1 / macro-recall
- `evaluate_holdout`: single 80/20 stratified split with `classification_report`
- `tune_with_gridsearch`: `GridSearchCV` with `f1_macro` scoring
- Confusion matrix and per-class metric plots are saved under `reports/figures/` by `src/train.py`

### 5.7 Ablations

#### Outlier handling
| `outlier_mode` | What it does |
|---|---|
| `none` | Placeholders cleaned, no numeric outlier handling |
| `domain` | Apply domain-bound cutoffs (e.g. `Age` ∈ [18, 100]) → `NaN` |
| `domain_percentile` | Domain cutoffs + 0.5%–99.5% percentile trim |

#### Class imbalance handling
| Strategy | Mechanism |
|---|---|
| `none` | Untouched |
| `class_weight` | `class_weight="balanced"` inside each estimator |
| `smote` | `SMOTE(random_state=42)` inserted via `imblearn.pipeline.Pipeline` |

### 5.8 Auxiliary Tasks (`src/auxiliary.py`)

- `regress_monthly_balance` — Ridge + GradientBoostingRegressor predicting `Monthly_Balance`, reports R² and MAE
- `cluster_payment_behaviour` — KMeans on 7 spending/saving features, reports silhouette and centroids
- `kmeans_elbow` — sweep `k = 2..8`, returns inertia and silhouette for elbow plotting

---

## 6. Output Artifacts (`reports/`)

### 6.1 Tables (CSV / JSON)

| File | Content |
|---|---|
| `eda/target_distribution.csv` | Class counts and ratios for `Credit_Score` |
| `eda/missing_report.csv` | Per-column missing counts and ratios |
| `eda/dirty_values.csv` | Where each placeholder (`_`, `_______`, `!@9#%8`) appears |
| `eda/numeric_describe_raw.csv` | `describe()` of object→numeric coerced columns (shows the impossible min/max) |
| `eda/numeric_describe_clean.csv` | Same after domain cleaning |
| `eda/outlier_summary.csv` | Side-by-side raw vs cleaned min/median/max + `% rows dropped` |
| `eda/correlation_matrix.csv` | Numeric Pearson correlation matrix |
| `eda/categorical_distributions.csv` | Stacked count + ratio for key categoricals |
| `ablation_outlier.csv` | Macro-F1 / accuracy per outlier strategy × model |
| `ablation_imbalance.csv` | Macro-F1 / accuracy per imbalance strategy × model |
| `ablation_preprocessing.csv` | Macro-F1 / accuracy per scaler × encoder |
| `cv_baseline.csv` | Stratified k-fold means and stds across all base models |
| `final_summary.json` | Best model, best `GridSearchCV` params, hold-out metrics, per-class report |
| `aux_regression.json` | Ridge and GBR results on `Monthly_Balance` (R², MAE) |
| `aux_kmeans_summary.json` | KMeans silhouette and cluster sizes |
| `aux_kmeans_centroids.csv` | Cluster centroids in original feature scale |
| `aux_kmeans_elbow.csv` | Inertia and silhouette for `k = 2..8` |
| `final_pipeline.joblib` | Fitted preprocessing + model pipeline (only with `--save-pipeline`) |

### 6.2 Figures (PNG) — report-ready

EDA figures matching the proposal's Figure 1 – Figure 7:

| File | Maps to |
|---|---|
| `figures/eda/fig01_target_distribution.png` | Proposal Figure 1 — class distribution |
| `figures/eda/fig02_numeric_histograms.png` | Proposal Figure 2 — numeric histograms |
| `figures/eda/fig03_boxplot_before_after.png` | Proposal Figures 3 & 4 — boxplots before vs after outlier cleaning |
| `figures/eda/fig04_correlation_matrix.png` | Proposal Figure 5 — Pearson correlation heatmap |
| `figures/eda/fig05_categorical_distributions.png` | Proposal Figure 6 — categorical countplots |
| `figures/eda/fig06_boxplot_by_target.png` | Proposal Figure 7 — key features by `Credit_Score` |

Modeling + ablation figures:

| File | Content |
|---|---|
| `figures/ablation_outlier.png` | Macro-F1 bar chart across `none` / `domain` / `domain_percentile` per model |
| `figures/ablation_imbalance.png` | Macro-F1 + Macro-Recall bar charts across `none` / `class_weight` / `SMOTE` per model |
| `figures/ablation_preprocessing.png` | 3 × 3 scaler × encoder Macro-F1 heatmap |
| `figures/cv_baseline.png` | Per-model k-fold Macro-F1 bar chart with std error bars + 0.75 target line |
| `figures/confusion_matrix.png` | Hold-out confusion matrix of the tuned best model |
| `figures/per_class_metrics.png` | Per-class precision / recall / F1 bar chart + 0.70 recall target line |
| `figures/aux_kmeans_elbow.png` | KMeans inertia and silhouette over `k = 2..8` |

---

## 7. Results (full run, 100,000 rows, Stratified 5-fold, GridSearchCV)

Proposal targets:
- **Macro-F1 ≥ 0.75** under Stratified 5-fold
- per-class **Recall ≥ 0.70**

To reproduce:

```bash
python -m src.train --all
```

### 7.1 Stratified 5-fold CV — baseline models

From [`reports/cv_baseline.csv`](reports/cv_baseline.csv):

| Model | Macro-F1 (mean ± std) | Macro-Recall | Accuracy |
|---|---|---|---|
| **RandomForest** | **0.7697 ± 0.0041** | 0.8001 | 0.7794 |
| DecisionTree | 0.7179 ± 0.0025 | 0.7584 | 0.7239 |
| GradientBoosting | 0.6948 ± 0.0031 | 0.7038 | 0.7142 |
| LogisticRegression | 0.6567 ± 0.0034 | 0.7032 | 0.6647 |

RandomForest clears the **0.75 Macro-F1** bar at the CV stage already.

### 7.2 GridSearchCV-tuned best model — final hold-out

From [`reports/final_summary.json`](reports/final_summary.json):

- Best model: **RandomForest**
- Best params: `n_estimators=400, max_depth=None, min_samples_leaf=1`
- Hold-out (20,000 rows):

| Metric | Value |
|---|---|
| Accuracy | **0.8187** |
| Macro-F1 | **0.8114** |
| Macro-Recall | **0.8122** |

Per-class on the hold-out:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Good | 0.795 | **0.777** | 0.786 | 3,566 |
| Poor | 0.801 | **0.837** | 0.818 | 5,799 |
| Standard | 0.837 | **0.823** | 0.830 | 10,635 |

All three per-class recalls clear the **0.70 target**. The minority class (Good, 17.8% of the data) is the hardest, as expected.

### 7.3 Outlier handling ablation (RandomForest, 5-fold)

From [`reports/ablation_outlier.csv`](reports/ablation_outlier.csv):

| Outlier mode | Macro-F1 | Macro-Recall | Accuracy |
|---|---|---|---|
| `none` | 0.7626 | 0.7947 | 0.7726 |
| `domain` | 0.7697 | 0.8001 | 0.7794 |
| `domain_percentile` | **0.7718** | **0.8023** | **0.7815** |

**Finding:** outlier cleaning is worth roughly +0.009 Macro-F1 over no handling, with `domain` + `percentile` slightly better than `domain` alone. The extreme garbage values (Age = 8,698; Interest_Rate = 5,797%; etc.) genuinely hurt downstream performance.

### 7.4 Class imbalance handling ablation (RandomForest, 5-fold)

From [`reports/ablation_imbalance.csv`](reports/ablation_imbalance.csv):

| Strategy | Macro-F1 | Macro-Recall | Accuracy |
|---|---|---|---|
| `none` | **0.7920** | 0.7994 | **0.8017** |
| `class_weight` | 0.7697 | **0.8001** | 0.7794 |
| `smote` | 0.7417 | 0.7765 | 0.7522 |

**Finding (counter-intuitive):** for RandomForest on this dataset, applying no special imbalance handling actually gives the highest Macro-F1 — the tree ensemble already handles the 53/29/18 split well, and `class_weight="balanced"` over-corrects, costing accuracy without proportional recall gains. SMOTE underperforms in every metric, likely because synthetic samples in this high-dimensional one-hot-encoded space introduce noise. Conclusion: **the class imbalance is real but does not require special handling for tree ensembles**, only for the linear baseline.

### 7.5 Scaler × Encoder sweep (DecisionTree, 5-fold)

From [`reports/ablation_preprocessing.csv`](reports/ablation_preprocessing.csv):

| | OneHot | Ordinal | Target |
|---|---|---|---|
| Standard | 0.7180 | 0.7041 | 0.7106 |
| MinMax | **0.7181** | 0.7045 | 0.7105 |
| Robust | 0.7179 | 0.7040 | 0.7107 |

**Finding:** scaler choice is essentially irrelevant for tree models (all three within 0.0001). Encoder choice matters more — `OneHot` beats `Ordinal` and `Target` by ~0.01 Macro-F1.

### 7.6 Auxiliary tasks

#### Regression on `Monthly_Balance` — [`reports/aux_regression.json`](reports/aux_regression.json)

| Model | R² | MAE |
|---|---|---|
| Ridge | 0.702 | 83.18 |
| GradientBoosting | **0.917** | **33.14** |

#### KMeans clustering on payment behaviour (`k = 4`) — [`reports/aux_kmeans_summary.json`](reports/aux_kmeans_summary.json)

- Silhouette: 0.348
- Cluster sizes: {0: 22,241 / 1: 4,488 / 2: 72,579 / 3: 692}
- Dominant cluster (2) holds ~73% of customers; outlier-like clusters (1, 3) are small and concentrated in extreme spending/saving behaviour. Centroids are in [`reports/aux_kmeans_centroids.csv`](reports/aux_kmeans_centroids.csv); elbow + silhouette over `k = 2..8` are in [`reports/aux_kmeans_elbow.csv`](reports/aux_kmeans_elbow.csv).

### 7.7 Summary against the proposal

| Goal | Target | Achieved | Status |
|---|---|---|---|
| Macro-F1 (CV) | ≥ 0.75 | 0.7697 (RF) | ✓ |
| Macro-F1 (hold-out, tuned) | n/a | 0.8114 | ✓ |
| Per-class Recall | ≥ 0.70 | 0.777 / 0.837 / 0.823 | ✓ |
| Decision Tree → ensemble comparison | required | done across 4 models | ✓ |
| 3 scaler × 3 encoder sweep | required | done | ✓ |
| Stratified k-fold | required | k = 5 | ✓ |
| Group-aware imputation | required | per-`Customer_ID` ffill / bfill | ✓ |
| `Type_of_Loan` MultiLabel encoding | required | `MultiLabelBinarizer` | ✓ |
| Outlier impact analysis | proposed | ablation table 7.3 | ✓ |
| Class imbalance impact analysis | proposed | ablation table 7.4 | ✓ |
| Auxiliary task | choose one | both regression and clustering done | ✓ |

---

## 8. Notebooks vs `src/` Modules

The original notebooks under `notebooks/` are kept for reference, but every analysis they perform has been ported to importable Python modules so the same numbers and figures can be reproduced from one CLI command:

| Notebook cell | Module |
|---|---|
| 01_eda — target distribution, missing, dirty values, raw `describe`, histograms, boxplots, correlation, categorical countplots | [`src/eda.py`](src/eda.py) (tables) + [`src/visualize.py`](src/visualize.py) (figures) |
| 02_preprocessing — placeholder removal, dtype coercion, domain cutoffs, drop columns | [`src/preprocessing.py`](src/preprocessing.py) `clean_data` |
| 03_model_test — train/valid split, baseline model loop, classification report, confusion matrix | [`src/model.py`](src/model.py), [`src/evaluate.py`](src/evaluate.py), [`src/train.py`](src/train.py) |

New experiments should go through `python -m src.train` so that ablations, CV, tables, and figures stay in sync.

---

## 9. Known Issues / Notes

- On Windows, `pip install scikit-learn` (without the conda OpenMP runtimes) may hit `OSError: [WinError -1066598273]` deep inside SMOTE. Use `environment.yml`.
- `category_encoders` is intentionally not used — sklearn ≥ 1.3 ships its own `TargetEncoder`.
- Only `data/raw/train.csv` is kept. The Kaggle `test.csv` was removed because it ships **without** the `Credit_Score` label, so it cannot be used for local accuracy / F1 / recall scoring. See [data/README.md](data/README.md) for the full reasoning. All evaluation is done inside `train.csv` via Stratified k-fold + an 80 / 20 hold-out split.
- `data/raw/train.csv` is ~30 MB. Consider keeping it out of git via `.gitignore` and re-downloading from Kaggle when needed.
- GradientBoosting in scikit-learn does not accept `class_weight`; the imbalance ablation silently ignores that argument for it.
- `final_pipeline.joblib` for the tuned RandomForest can be >1 GB. It is opt-in via `--save-pipeline` and ignored by git through the `*.joblib` rule in `.gitignore`.
- A full `--all` run on the 100,000-row train set with `--cv-folds 5` and GridSearchCV takes roughly 30 – 120 minutes on a laptop, dominated by the RandomForest grid (24 combinations × 5 folds). Use `--sample-rows N --cv-folds 3 --skip-grid` for fast iteration; reach for `--all` only when you want the canonical numbers in section 7.
