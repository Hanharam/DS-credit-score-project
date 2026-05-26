"""Programmatic EDA — mirrors the analysis cells of notebooks/01_eda.ipynb so
the same numbers can be reproduced from a script and dumped to CSV.

Each function returns a DataFrame; `write_eda_reports` saves the lot under a
directory of your choice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .preprocessing import (
    NUMERIC_LIKE_OBJECT_COLS,
    PLACEHOLDERS,
    _coerce_numeric_like,
    clean_data,
)


DEFAULT_NUMERIC_EDA_COLS = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
]

DEFAULT_CATEGORICAL_EDA_COLS = [
    "Month",
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]

KEY_FEATURES_BY_TARGET = [
    "Outstanding_Debt",
    "Interest_Rate",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Credit_Utilization_Ratio",
    "Monthly_Balance",
]


def target_distribution(df: pd.DataFrame, target: str = "Credit_Score") -> pd.DataFrame:
    counts = df[target].value_counts()
    ratio = df[target].value_counts(normalize=True) * 100
    return pd.DataFrame({"count": counts, "ratio_pct": ratio})


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "missing_count": df.isnull().sum(),
            "missing_ratio_pct": df.isnull().mean() * 100,
        }
    )
    return out[out["missing_count"] > 0].sort_values(
        "missing_count", ascending=False
    )


def dirty_value_scan(
    df: pd.DataFrame, placeholders: Iterable[str] | None = None
) -> pd.DataFrame:
    placeholders = list(placeholders) if placeholders else PLACEHOLDERS
    rows = []
    for value in placeholders:
        for col in df.columns:
            count = int((df[col] == value).sum())
            if count > 0:
                rows.append(
                    {
                        "dirty_value": value,
                        "column": col,
                        "count": count,
                        "ratio_pct": count / len(df) * 100,
                    }
                )
    return pd.DataFrame(rows)


def numeric_describe_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce object-typed numeric columns then describe — matches notebook cell 7."""
    work = df.copy()
    work = _coerce_numeric_like(work)
    cols = [c for c in NUMERIC_LIKE_OBJECT_COLS if c in work.columns]
    return work[cols].describe().T


def numeric_describe_after_cleaning(
    df: pd.DataFrame, outlier_mode: str = "domain"
) -> pd.DataFrame:
    cleaned = clean_data(df, outlier_mode=outlier_mode, keep_customer_id=False)
    cols = [c for c in DEFAULT_NUMERIC_EDA_COLS if c in cleaned.columns]
    return cleaned[cols].describe().T


def categorical_distribution(
    df: pd.DataFrame, col: str, top_n: int | None = None
) -> pd.DataFrame:
    counts = df[col].value_counts(dropna=False)
    if top_n:
        counts = counts.head(top_n)
    ratio = counts / len(df) * 100
    return pd.DataFrame({"count": counts, "ratio_pct": ratio})


def correlation_matrix(
    df: pd.DataFrame, method: str = "pearson"
) -> pd.DataFrame:
    """Numeric-only correlation after domain cleaning so impossible values
    don't dominate the picture."""
    cleaned = clean_data(df, outlier_mode="domain", keep_customer_id=False)
    numeric = cleaned.select_dtypes(include=[np.number])
    return numeric.corr(method=method)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side min/median/max before and after domain cutoffs for the
    columns the proposal flagged as having extreme garbage values."""
    work = df.copy()
    work = _coerce_numeric_like(work)
    cleaned = clean_data(df, outlier_mode="domain", keep_customer_id=False)

    rows = []
    cols = [c for c in DEFAULT_NUMERIC_EDA_COLS if c in work.columns and c in cleaned.columns]
    for col in cols:
        raw = work[col]
        clean = cleaned[col]
        rows.append(
            {
                "column": col,
                "raw_min": raw.min(),
                "raw_median": raw.median(),
                "raw_max": raw.max(),
                "clean_min": clean.min(),
                "clean_median": clean.median(),
                "clean_max": clean.max(),
                "dropped_pct": float(clean.isna().mean() - raw.isna().mean()) * 100,
            }
        )
    return pd.DataFrame(rows)


def write_eda_reports(df: pd.DataFrame, out_dir: str | Path) -> dict:
    """Persist every EDA table under `out_dir` as CSV. Returns the dict of
    written paths for logging."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    writes = {
        "target_distribution.csv": target_distribution(df),
        "missing_report.csv": missing_report(df),
        "dirty_values.csv": dirty_value_scan(df),
        "numeric_describe_raw.csv": numeric_describe_raw(df),
        "numeric_describe_clean.csv": numeric_describe_after_cleaning(df),
        "outlier_summary.csv": outlier_summary(df),
        "correlation_matrix.csv": correlation_matrix(df),
    }
    for fname, frame in writes.items():
        frame.to_csv(out_path / fname)

    # Categorical tables in one file
    cat_frames = []
    for col in DEFAULT_CATEGORICAL_EDA_COLS:
        if col in df.columns:
            sub = categorical_distribution(df, col)
            sub.insert(0, "column", col)
            sub.insert(1, "value", sub.index)
            cat_frames.append(sub.reset_index(drop=True))
    if cat_frames:
        pd.concat(cat_frames, ignore_index=True).to_csv(
            out_path / "categorical_distributions.csv", index=False
        )

    return {fname: str(out_path / fname) for fname in writes}
