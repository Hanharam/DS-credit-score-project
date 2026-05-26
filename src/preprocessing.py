"""Data cleaning, feature engineering, and preprocessing pipeline builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Opt in to pandas 3.0 behaviour: no silent dtype downcasting after ffill/bfill.
# Without this, groupby().transform(ffill/bfill) emits a FutureWarning on every call.
pd.set_option("future.no_silent_downcasting", True)

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    MultiLabelBinarizer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


PLACEHOLDERS = ["_", "_______", "!@9#%8"]

NUMERIC_LIKE_OBJECT_COLS = [
    "Age",
    "Annual_Income",
    "Num_of_Loan",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Outstanding_Debt",
    "Amount_invested_monthly",
    "Monthly_Balance",
]

DOMAIN_BOUNDS = {
    "Age": (18, 100),
    "Num_Bank_Accounts": (0, 20),
    "Num_Credit_Card": (0, 20),
    "Interest_Rate": (0, 100),
    "Num_of_Loan": (0, 20),
    "Num_of_Delayed_Payment": (0, 100),
    "Num_Credit_Inquiries": (0, 100),
}

DOMAIN_LOWER_ONLY = {
    "Delay_from_due_date": 0,
    "Monthly_Balance": 0,
}


def convert_credit_history_age(value):
    if pd.isna(value):
        return np.nan
    text = str(value)
    years = 0
    months = 0
    if "Years" in text:
        years = int(text.split(" Years")[0])
    if "Months" in text:
        months_part = text.split("and ")[-1].split(" Months")[0]
        months = int(months_part)
    return years * 12 + months


def _coerce_numeric_like(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_LIKE_OBJECT_COLS:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.replace("_", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _apply_domain_cutoffs(df: pd.DataFrame) -> pd.DataFrame:
    for col, (lo, hi) in DOMAIN_BOUNDS.items():
        if col in df.columns:
            df.loc[~df[col].between(lo, hi), col] = np.nan
    for col, lo in DOMAIN_LOWER_ONLY.items():
        if col in df.columns:
            df.loc[df[col] < lo, col] = np.nan
    return df


def _apply_percentile_trim(
    df: pd.DataFrame, lower: float = 0.005, upper: float = 0.995
) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    return df


def clean_data(
    df: pd.DataFrame,
    outlier_mode: str = "domain",
    keep_customer_id: bool = False,
) -> pd.DataFrame:
    """Clean raw credit-score data.

    outlier_mode: 'none' | 'domain' | 'domain_percentile'
    keep_customer_id: keep Customer_ID for group-aware imputation downstream.
    """
    if outlier_mode not in {"none", "domain", "domain_percentile"}:
        raise ValueError(f"unknown outlier_mode: {outlier_mode}")

    df = df.copy()
    df = df.replace(PLACEHOLDERS, np.nan)
    df = _coerce_numeric_like(df)

    if "Credit_History_Age" in df.columns:
        df["Credit_History_Age_Months"] = df["Credit_History_Age"].apply(
            convert_credit_history_age
        )
        df = df.drop(columns=["Credit_History_Age"])

    drop_cols = ["ID", "Name", "SSN"]
    if not keep_customer_id:
        drop_cols.append("Customer_ID")
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if outlier_mode in {"domain", "domain_percentile"}:
        df = _apply_domain_cutoffs(df)
    if outlier_mode == "domain_percentile":
        df = _apply_percentile_trim(df)

    return df


def group_impute(
    df: pd.DataFrame, group_col: str = "Customer_ID"
) -> pd.DataFrame:
    """Per-customer forward/backward fill across the 8 monthly records."""
    if group_col not in df.columns:
        return df
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    fill_cols = [c for c in numeric_cols + object_cols if c != group_col]

    grouped = df.groupby(group_col, sort=False)
    filled = grouped[fill_cols].transform(lambda g: g.ffill().bfill())
    df[fill_cols] = filled.infer_objects(copy=False)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"Outstanding_Debt", "Annual_Income"}.issubset(df.columns):
        df["Debt_to_Income"] = df["Outstanding_Debt"] / (df["Annual_Income"] + 1)
    if {"Total_EMI_per_month", "Monthly_Inhand_Salary"}.issubset(df.columns):
        df["EMI_to_Salary"] = df["Total_EMI_per_month"] / (
            df["Monthly_Inhand_Salary"] + 1
        )
    if {"Amount_invested_monthly", "Monthly_Inhand_Salary"}.issubset(df.columns):
        df["Savings_Rate"] = df["Amount_invested_monthly"] / (
            df["Monthly_Inhand_Salary"] + 1
        )
    if {"Num_of_Delayed_Payment", "Num_of_Loan"}.issubset(df.columns):
        df["Delay_per_Loan"] = df["Num_of_Delayed_Payment"] / (
            df["Num_of_Loan"] + 1
        )
    return df


def encode_type_of_loan(
    df: pd.DataFrame, col: str = "Type_of_Loan"
) -> tuple[pd.DataFrame, MultiLabelBinarizer | None]:
    """Split comma-separated loan types and MultiLabelBinarize them."""
    if col not in df.columns:
        return df, None

    def split_loans(v):
        if pd.isna(v):
            return []
        parts = [p.strip().replace("and ", "") for p in str(v).split(",")]
        return [p for p in parts if p]

    series = df[col].apply(split_loans)
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(series)
    encoded_df = pd.DataFrame(
        encoded,
        columns=[f"Loan_{c.replace(' ', '_')}" for c in mlb.classes_],
        index=df.index,
    )
    df = df.drop(columns=[col]).join(encoded_df)
    return df, mlb


SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def _build_encoder(kind: str):
    kind = kind.lower()
    if kind == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if kind == "ordinal":
        return OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    if kind == "target":
        from sklearn.preprocessing import TargetEncoder
        return TargetEncoder(random_state=42)
    raise ValueError(f"unknown encoder: {kind}")


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scaler: str = "robust",
    encoder: str = "onehot",
) -> ColumnTransformer:
    scaler_cls = SCALERS[scaler.lower()]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler_cls()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _build_encoder(encoder)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def split_features(df: pd.DataFrame, target: str = "Credit_Score"):
    X = df.drop(columns=[target])
    y = df[target]
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return X, y, numeric, categorical
