"""Programmatic EDA - reproduces every analysis table in Section 2 of the report.

Each function returns a tidy pandas DataFrame so that the table can be
embedded in the report verbatim. :func:`write_eda_reports` writes the
entire bundle to disk as a set of CSVs under a chosen directory; the
matching figures are produced by ``src.visualize`` using the same
DataFrames.

Why a dedicated module: keeping EDA logic in importable functions (as
opposed to one-shot notebook cells) means the numbers in the report stay
in sync with the dataset whenever the raw CSV changes, and the same
helpers can be reused by ``src/visualize.py`` so tables and figures never
drift apart.
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


# ---------------------------------------------------------------------------
# Column lists used across the EDA helpers and figures.
# ---------------------------------------------------------------------------

# Numeric columns we describe / histogram / boxplot in Figures 2, 3, 6.
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

# Categorical columns we count-plot in Figure 5.
DEFAULT_CATEGORICAL_EDA_COLS = [
    "Month",
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]

# Features used in the per-target boxplots (Figure 6).
KEY_FEATURES_BY_TARGET = [
    "Outstanding_Debt",
    "Interest_Rate",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Credit_Utilization_Ratio",
    "Monthly_Balance",
]


def target_distribution(df: pd.DataFrame, target: str = "Credit_Score") -> pd.DataFrame:
    """Per-class count and ratio for the target column.

    Returns
    -------
    pandas.DataFrame
        Two columns: ``count`` and ``ratio_pct``. Used to produce
        ``reports/eda/target_distribution.csv`` and Figure 1.
    """
    counts = df[target].value_counts()
    ratio = df[target].value_counts(normalize=True) * 100
    return pd.DataFrame({"count": counts, "ratio_pct": ratio})


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing-value summary for every column that has any NaN.

    Returns
    -------
    pandas.DataFrame
        Columns ``missing_count`` and ``missing_ratio_pct``, sorted by
        count descending. Columns with zero NaN are dropped.
    """
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
    """Count how often each placeholder token appears in each column.

    Used to document Section 2.3 / Table 3 of the report (where the
    three placeholders '_' / '_______' / '!@9#%8' show up).
    """
    placeholders = list(placeholders) if placeholders else PLACEHOLDERS
    rows = []
    # Nested loop: (placeholder x column). Only emit a row when the
    # placeholder actually appears, to keep the output compact.
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
    """``describe()`` of the numeric-like-object columns *before* cleaning.

    Coerces the dtypes first (using :func:`_coerce_numeric_like`) so
    pandas can compute statistics, but does NOT apply domain cutoffs or
    placeholder substitution. The intent is to show the raw garbage
    values (Age = 8,698, Interest_Rate = 5,797, etc.) - the same numbers
    that motivated the domain cutoffs.
    """
    work = df.copy()
    work = _coerce_numeric_like(work)
    cols = [c for c in NUMERIC_LIKE_OBJECT_COLS if c in work.columns]
    return work[cols].describe().T


def numeric_describe_after_cleaning(
    df: pd.DataFrame, outlier_mode: str = "domain"
) -> pd.DataFrame:
    """Same as :func:`numeric_describe_raw` but after a full clean pass.

    Used in the report to show that domain cutoffs collapse the wild
    min/max values back into realistic ranges.
    """
    cleaned = clean_data(df, outlier_mode=outlier_mode, keep_customer_id=False)
    cols = [c for c in DEFAULT_NUMERIC_EDA_COLS if c in cleaned.columns]
    return cleaned[cols].describe().T


def categorical_distribution(
    df: pd.DataFrame, col: str, top_n: int | None = None
) -> pd.DataFrame:
    """Value counts (with NaN preserved) for a single categorical column.

    Parameters
    ----------
    df : pandas.DataFrame
    col : str
        Column to summarise.
    top_n : int, optional
        Keep only the top-N most frequent values; useful for
        long-tailed columns like ``Occupation``.
    """
    # dropna=False keeps NaN as a category, which is informative when the
    # column has a high missing rate (e.g. Type_of_Loan).
    counts = df[col].value_counts(dropna=False)
    if top_n:
        counts = counts.head(top_n)
    ratio = counts / len(df) * 100
    return pd.DataFrame({"count": counts, "ratio_pct": ratio})


def correlation_matrix(
    df: pd.DataFrame, method: str = "pearson"
) -> pd.DataFrame:
    """Numeric Pearson correlation after a domain-clean pass.

    Without the clean step the extreme outliers would dominate the
    correlation values (e.g. Age = 8,698 would pull every coefficient).
    """
    cleaned = clean_data(df, outlier_mode="domain", keep_customer_id=False)
    numeric = cleaned.select_dtypes(include=[np.number])
    return numeric.corr(method=method)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side raw vs. cleaned min/median/max + drop fraction.

    For every numeric column we tabulate:
      * ``raw_min, raw_median, raw_max`` - after dtype coercion but BEFORE
        outlier handling (shows the impossible values).
      * ``clean_min, clean_median, clean_max`` - AFTER domain cutoffs.
      * ``dropped_pct`` - extra fraction of NaN introduced by the
        cleaning step.
    """
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
    """Materialise every EDA table to disk as CSV.

    Used by ``python -m src.train --run-eda`` to populate
    ``reports/eda/`` before any modelling. Categorical distributions are
    concatenated into a single file (``categorical_distributions.csv``)
    so the report can cite a single artifact.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw frame (before any cleaning). EDA needs the dirty values for
        the dirty-value scan.
    out_dir : str or Path
        Destination directory; created if missing.

    Returns
    -------
    dict
        ``{filename: full_path_string}`` for logging.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Standard one-file-per-table outputs.
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

    # Categorical distributions are emitted as one long file with a
    # 'column' marker so the report can cite a single CSV.
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
