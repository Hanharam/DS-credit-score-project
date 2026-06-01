"""Evaluation utilities for the term-project pipeline.

Contains:

* :func:`evaluate_cv` and :func:`evaluate_holdout` - reusable wrappers
  around :func:`sklearn.model_selection.cross_validate` and a single
  fit/predict on a stratified hold-out, returning consistent metric
  dictionaries.
* Three ablation drivers:
    - :func:`ablation_outlier`        (Section 5.4 of the report)
    - :func:`ablation_imbalance`      (Section 5.5)
    - :func:`ablation_preprocessing`  (Section 5.6)
* :func:`tune_with_gridsearch` - the GridSearchCV setup used in
  Section 5.2 of the report.

All functions reuse a single :data:`SCORING` dictionary so every CV-style
output reports the same three metrics (accuracy, Macro-F1, Macro-Recall).

Non-Lab sklearn pieces used here (Appendix B of the report):

* :class:`sklearn.model_selection.StratifiedKFold`
* :func:`sklearn.model_selection.cross_validate`
* :class:`sklearn.model_selection.GridSearchCV`
* :func:`sklearn.metrics.classification_report`
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from .model import build_pipeline, get_models, get_param_grids
from .preprocessing import (
    add_engineered_features,
    build_preprocessor,
    clean_data,
    encode_type_of_loan,
    group_impute,
    split_features,
)


# Maps the project metric label -> sklearn scoring identifier. Used by
# both ``evaluate_cv`` and ``cross_validate`` so every CV reports the
# same three numbers.
SCORING = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "macro_recall": "recall_macro",
}


def evaluate_holdout(pipeline, X_train, X_valid, y_train, y_valid) -> dict:
    """Fit a pipeline on ``(X_train, y_train)`` and score it on the hold-out.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The full preprocessor + model pipeline returned by
        :func:`src.model.build_pipeline`.
    X_train, y_train : array-like
        Training features and target.
    X_valid, y_valid : array-like
        Hold-out features and target.

    Returns
    -------
    dict
        Keys ``accuracy``, ``macro_f1``, ``macro_recall``, ``report``
        (dict from ``classification_report``), and ``y_pred`` for plots.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_valid)
    return {
        "accuracy": accuracy_score(y_valid, y_pred),
        "macro_f1": f1_score(y_valid, y_pred, average="macro"),
        "macro_recall": recall_score(y_valid, y_pred, average="macro"),
        # ``output_dict=True`` returns a nested dict that
        # ``src.visualize.plot_per_class_metrics`` can consume directly.
        "report": classification_report(y_valid, y_pred, output_dict=True),
        "y_pred": y_pred,
    }


def evaluate_cv(pipeline, X, y, cv: int = 5, n_jobs: int = 1) -> dict:
    """Stratified k-fold cross-validation with three metrics.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
    X, y : array-like
    cv : int, default=5
        Number of folds.
    n_jobs : int, default=1
        Forwarded to :func:`cross_validate`. The default is 1 so the
        outer parallelism (e.g. GridSearchCV's ``n_jobs=-1``) does not
        cause nested-parallel oversubscription.

    Returns
    -------
    dict
        ``{metric: {'mean': float, 'std': float}}`` for each metric in
        :data:`SCORING`.
    """
    # random_state pinned so the fold splits are reproducible across
    # ablation runs and the main CV.
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=skf,
        scoring=SCORING,
        n_jobs=n_jobs,
        return_train_score=False,
    )
    # Collapse the per-fold arrays into mean +/- std summaries.
    return {
        metric: {
            "mean": float(scores[f"test_{metric}"].mean()),
            "std": float(scores[f"test_{metric}"].std()),
        }
        for metric in SCORING
    }


def _build_dataset(raw_df: pd.DataFrame, outlier_mode: str) -> pd.DataFrame:
    """Run the cleaning -> imputation -> MLB -> feature-engineering chain.

    Used by :func:`ablation_outlier` so each outlier mode is paired with
    a freshly imputed dataset (since the cleaning step changes which
    cells are NaN, the imputation result also changes).
    """
    df = clean_data(raw_df, outlier_mode=outlier_mode, keep_customer_id=True)
    df = group_impute(df, group_col="Customer_ID")
    df = df.drop(columns=["Customer_ID"])
    df, _ = encode_type_of_loan(df, col="Type_of_Loan")
    df = add_engineered_features(df)
    return df


def ablation_outlier(
    raw_df: pd.DataFrame,
    models_subset: list[str] | None = None,
    cv: int = 3,
) -> pd.DataFrame:
    """Ablation 1: compare ``none`` / ``domain`` / ``domain+percentile``.

    Each outlier mode triggers a fresh cleaning + imputation chain, so
    the comparison is fair end-to-end.

    Parameters
    ----------
    raw_df : pandas.DataFrame
        Raw frame loaded from ``data/raw/train.csv``.
    models_subset : list of str, optional
        If given, only these model names from :func:`get_models` are
        evaluated. Defaults to all four. Use ``['DecisionTree',
        'RandomForest']`` to bound runtime.
    cv : int, default=3
        Cross-validation folds.

    Returns
    -------
    pandas.DataFrame
        Columns: ``outlier_mode, model, macro_f1, macro_f1_std,
        macro_recall, accuracy``. One row per (mode, model) pair.
    """
    rows = []
    for mode in ["none", "domain", "domain_percentile"]:
        # Rebuild the dataset from scratch under each outlier mode.
        df = _build_dataset(raw_df, outlier_mode=mode)
        X, y, num, cat = split_features(df)
        # The scaler/encoder pair is held fixed so the only varying
        # factor in this ablation is the outlier strategy.
        preprocessor = build_preprocessor(num, cat, scaler="robust", encoder="onehot")
        # class_weight='balanced' is the baseline used everywhere else,
        # keeping this ablation comparable to Section 5.1's CV table.
        models = get_models(class_weight="balanced")
        for name, model in models.items():
            if models_subset and name not in models_subset:
                continue
            pipeline = build_pipeline(preprocessor, model)
            scores = evaluate_cv(pipeline, X, y, cv=cv)
            rows.append(
                {
                    "outlier_mode": mode,
                    "model": name,
                    "macro_f1": scores["macro_f1"]["mean"],
                    "macro_f1_std": scores["macro_f1"]["std"],
                    "macro_recall": scores["macro_recall"]["mean"],
                    "accuracy": scores["accuracy"]["mean"],
                }
            )
    return pd.DataFrame(rows)


def ablation_imbalance(
    df: pd.DataFrame,
    models_subset: list[str] | None = None,
    cv: int = 3,
) -> pd.DataFrame:
    """Ablation 2: compare ``none`` / ``class_weight`` / ``SMOTE``.

    Unlike the outlier ablation, here we accept an already-cleaned ``df``
    because the cleaning chain does not change across the three
    imbalance strategies.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`_build_dataset` (cleaned + imputed + engineered).
    models_subset : list of str, optional
        Same semantics as in :func:`ablation_outlier`.
    cv : int, default=3
    """
    X, y, num, cat = split_features(df)
    preprocessor = build_preprocessor(num, cat, scaler="robust", encoder="onehot")

    # (label, class_weight value, whether to wrap with SMOTE pipeline)
    strategies = [
        ("none", None, False),
        ("class_weight", "balanced", False),
        ("smote", None, True),
    ]
    rows = []
    for label, cw, use_smote in strategies:
        # Build a fresh estimator for each strategy so that class_weight
        # is set correctly at construction time (it is a constructor arg).
        models = get_models(class_weight=cw)
        for name, model in models.items():
            if models_subset and name not in models_subset:
                continue
            pipeline = build_pipeline(preprocessor, model, use_smote=use_smote)
            scores = evaluate_cv(pipeline, X, y, cv=cv)
            rows.append(
                {
                    "strategy": label,
                    "model": name,
                    "macro_f1": scores["macro_f1"]["mean"],
                    "macro_f1_std": scores["macro_f1"]["std"],
                    "macro_recall": scores["macro_recall"]["mean"],
                    "accuracy": scores["accuracy"]["mean"],
                }
            )
    return pd.DataFrame(rows)


def ablation_preprocessing(
    df: pd.DataFrame,
    model_name: str = "RandomForest",
    cv: int = 3,
) -> pd.DataFrame:
    """Ablation 3: 3 scalers x 3 encoders = 9 combinations on a fixed model.

    Used to study whether scaler or encoder dominates performance on a
    tree-based model (Section 5.6 of the report).

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned + imputed + engineered frame.
    model_name : str, default='RandomForest'
        Which model from :func:`get_models` to evaluate. The report uses
        ``DecisionTree`` for this ablation; the default kept here is the
        project-wide best model.
    cv : int, default=3

    Returns
    -------
    pandas.DataFrame
        Columns: ``scaler, encoder, macro_f1, macro_f1_std, accuracy``.
        If a particular encoder fails (e.g. TargetEncoder choking on a
        column with too few classes), the row reports NaN and the error
        message.
    """
    X, y, num, cat = split_features(df)
    models = get_models(class_weight="balanced")
    model = models[model_name]

    rows = []
    for scaler in ["standard", "minmax", "robust"]:
        for encoder in ["onehot", "ordinal", "target"]:
            try:
                preprocessor = build_preprocessor(num, cat, scaler=scaler, encoder=encoder)
                pipeline = build_pipeline(preprocessor, model)
                scores = evaluate_cv(pipeline, X, y, cv=cv)
                rows.append(
                    {
                        "scaler": scaler,
                        "encoder": encoder,
                        "macro_f1": scores["macro_f1"]["mean"],
                        "macro_f1_std": scores["macro_f1"]["std"],
                        "accuracy": scores["accuracy"]["mean"],
                    }
                )
            except Exception as exc:
                # Recording the error keeps the leaderboard square (9 rows)
                # and surfaces the failure to the caller instead of crashing
                # the whole ablation.
                rows.append(
                    {
                        "scaler": scaler,
                        "encoder": encoder,
                        "macro_f1": np.nan,
                        "macro_f1_std": np.nan,
                        "accuracy": np.nan,
                        "error": str(exc),
                    }
                )
    return pd.DataFrame(rows)


def tune_with_gridsearch(
    pipeline,
    param_grid: dict,
    X,
    y,
    cv: int = 5,
    scoring: str = "f1_macro",
    n_jobs: int = -1,
) -> GridSearchCV:
    """Wrap :class:`GridSearchCV` with the project's standard knobs.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
    param_grid : dict
        Usually fetched from :func:`get_param_grid_for`.
    X, y : array-like
    cv : int, default=5
        Stratified k-fold splits.
    scoring : str, default='f1_macro'
        Forwarded to GridSearchCV.
    n_jobs : int, default=-1
        Parallelism; -1 uses every core.

    Returns
    -------
    sklearn.model_selection.GridSearchCV
        The fitted GridSearchCV. ``.best_estimator_`` is already refit on
        the full data thanks to ``refit=True``.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=skf,
        n_jobs=n_jobs,
        # refit=True so the returned object has a fitted best_estimator_.
        refit=True,
        verbose=1,
    )
    grid.fit(X, y)
    return grid


def get_param_grid_for(model_name: str) -> dict[str, list]:
    """Convenience wrapper exposing :func:`get_param_grids` by model name."""
    return get_param_grids()[model_name]


def plot_confusion_matrix(*args, **kwargs):
    """Back-compat shim. The canonical implementation lives in src.visualize."""
    from .visualize import plot_confusion_matrix as _impl
    return _impl(*args, **kwargs)
