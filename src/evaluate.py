"""Evaluation utilities: CV, ablations over outlier handling, class imbalance,
and scaler/encoder choice; GridSearchCV tuning; confusion matrix plotting."""

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


SCORING = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "macro_recall": "recall_macro",
}


def evaluate_holdout(pipeline, X_train, X_valid, y_train, y_valid) -> dict:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_valid)
    return {
        "accuracy": accuracy_score(y_valid, y_pred),
        "macro_f1": f1_score(y_valid, y_pred, average="macro"),
        "macro_recall": recall_score(y_valid, y_pred, average="macro"),
        "report": classification_report(y_valid, y_pred, output_dict=True),
        "y_pred": y_pred,
    }


def evaluate_cv(pipeline, X, y, cv: int = 5, n_jobs: int = 1) -> dict:
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
    return {
        metric: {
            "mean": float(scores[f"test_{metric}"].mean()),
            "std": float(scores[f"test_{metric}"].std()),
        }
        for metric in SCORING
    }


def _build_dataset(raw_df: pd.DataFrame, outlier_mode: str) -> pd.DataFrame:
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
    """Compare none / domain / domain+percentile outlier strategies."""
    rows = []
    for mode in ["none", "domain", "domain_percentile"]:
        df = _build_dataset(raw_df, outlier_mode=mode)
        X, y, num, cat = split_features(df)
        preprocessor = build_preprocessor(num, cat, scaler="robust", encoder="onehot")
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
    """Compare none / class_weight / SMOTE imbalance handling strategies."""
    X, y, num, cat = split_features(df)
    preprocessor = build_preprocessor(num, cat, scaler="robust", encoder="onehot")

    strategies = [
        ("none", None, False),
        ("class_weight", "balanced", False),
        ("smote", None, True),
    ]
    rows = []
    for label, cw, use_smote in strategies:
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
    """3 scalers x 3 encoders on a fixed model."""
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
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=skf,
        n_jobs=n_jobs,
        refit=True,
        verbose=1,
    )
    grid.fit(X, y)
    return grid


def get_param_grid_for(model_name: str) -> dict[str, list]:
    return get_param_grids()[model_name]


def plot_confusion_matrix(*args, **kwargs):
    """Back-compat shim. The canonical implementation lives in src.visualize."""
    from .visualize import plot_confusion_matrix as _impl
    return _impl(*args, **kwargs)
