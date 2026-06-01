"""Data cleaning, feature engineering, and preprocessing pipeline builders.

This module implements every step that transforms the raw Kaggle
``train.csv`` (100,000 rows x 28 columns) into a model-ready feature
matrix. It is the single source of truth for the project's data
preprocessing logic and is invoked from ``src/train.py``,
``src/evaluate.py``, and ``src/sweep.py``.

High-level pipeline (matches Section 3 of the report):

    1. Replace 3 placeholder strings ('_', '_______', '!@9#%8') with NaN.
    2. Coerce 8 numeric-like object columns (Age, Annual_Income, etc.) to
       proper numeric dtype, stripping trailing underscores first.
    3. Parse ``Credit_History_Age`` from 'X Years and Y Months' to total
       months.
    4. Drop identifier columns (ID, Name, SSN); keep ``Customer_ID`` until
       group-wise imputation finishes, then drop.
    5. Outlier handling - ``outlier_mode`` switches between
       ``none``, ``domain``, and ``domain_percentile``.
    6. Per-Customer_ID forward/backward fill across the 8 monthly rows
       (preserves the panel structure of the dataset).
    7. Engineer four financial ratios (Debt_to_Income, EMI_to_Salary,
       Savings_Rate, Delay_per_Loan).
    8. Multi-label-binarize the ``Type_of_Loan`` column (one binary
       column per loan kind).
    9. Build a :class:`sklearn.compose.ColumnTransformer` that runs
       ``SimpleImputer + scaler`` on numerics and
       ``SimpleImputer + encoder`` on categoricals.

Non-lab sklearn pieces used here (documented in the report Appendix B):

* :class:`sklearn.preprocessing.MultiLabelBinarizer` - for the multi-label
  Type_of_Loan column.
* :class:`sklearn.preprocessing.TargetEncoder` - one of the three encoders
  in the Ablation 3 sweep.
* :class:`sklearn.compose.ColumnTransformer` /
  :class:`sklearn.pipeline.Pipeline` - to wire imputers/scalers/encoders
  into a single estimator that can be passed to GridSearchCV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pandas 3.0 forward-compatibility flag.
#
# Without this option, ``groupby(...).transform(lambda g: g.ffill().bfill())``
# would emit a noisy ``FutureWarning`` on every call because pandas 2.x
# silently downcasts object dtype after the fill, while pandas 3.x will not.
# Opting in here makes the warning go away and forces the new behaviour now.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Domain constants (Section 3 of the report)
# ---------------------------------------------------------------------------

# Three placeholder tokens the original dataset uses to mean "missing".
# They appear in different columns: '_' (Credit_Mix), '_______' (Occupation),
# '!@9#%8' (Payment_Behaviour). We replace all three with NaN up front.
PLACEHOLDERS = ["_", "_______", "!@9#%8"]

# Columns that look numeric but are stored as object/string. They typically
# carry a trailing underscore (e.g. "28_") which has to be stripped before
# ``pd.to_numeric`` can convert them. Identified by manual inspection during
# EDA (Section 2.3 of the report).
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

# Domain-realistic ranges per variable (both lower and upper bound apply).
# Anything outside is treated as an outlier and replaced with NaN by
# ``_apply_domain_cutoffs``. Bounds come from financial common sense
# (e.g. Age >= 18 because the dataset is about credit history).
DOMAIN_BOUNDS = {
    "Age": (18, 100),
    "Num_Bank_Accounts": (0, 20),
    "Num_Credit_Card": (0, 20),
    "Interest_Rate": (0, 100),
    "Num_of_Loan": (0, 20),
    "Num_of_Delayed_Payment": (0, 100),
    "Num_Credit_Inquiries": (0, 100),
}

# Variables where only a lower bound is meaningful (negatives are obviously
# wrong but the upper tail can be legitimately large).
DOMAIN_LOWER_ONLY = {
    "Delay_from_due_date": 0,
    "Monthly_Balance": 0,
}


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def convert_credit_history_age(value):
    """Parse ``Credit_History_Age`` strings like '22 Years and 1 Months'
    into a total number of months (int).

    Parameters
    ----------
    value : str or NaN
        Raw cell value from the Credit_History_Age column. Format is
        ``"<X> Years and <Y> Months"``.

    Returns
    -------
    int or np.nan
        Total months (``X * 12 + Y``). Returns NaN if the input is NaN.

    Examples
    --------
    >>> convert_credit_history_age("22 Years and 1 Months")
    265
    """
    if pd.isna(value):
        return np.nan
    text = str(value)
    years = 0
    months = 0
    # The string always has a "Years" segment but we still guard it.
    if "Years" in text:
        years = int(text.split(" Years")[0])
    # "Months" piece sits after the literal "and ".
    if "Months" in text:
        months_part = text.split("and ")[-1].split(" Months")[0]
        months = int(months_part)
    return years * 12 + months


def _coerce_numeric_like(df: pd.DataFrame) -> pd.DataFrame:
    """Cast every column in :data:`NUMERIC_LIKE_OBJECT_COLS` to numeric.

    Two-step procedure:
      1. Remove any underscore character from the string representation
         (handles the corruption pattern "28_" -> "28").
      2. ``pd.to_numeric(errors='coerce')`` so anything still unparseable
         becomes NaN instead of raising.
    """
    for col in NUMERIC_LIKE_OBJECT_COLS:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.replace("_", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _apply_domain_cutoffs(df: pd.DataFrame) -> pd.DataFrame:
    """Set out-of-range values to NaN according to :data:`DOMAIN_BOUNDS`
    (two-sided) and :data:`DOMAIN_LOWER_ONLY` (one-sided)."""
    # Two-sided bounds.
    for col, (lo, hi) in DOMAIN_BOUNDS.items():
        if col in df.columns:
            # ``between`` is inclusive by default; the negated mask flips
            # the in-range rows back to True so we overwrite the out-of-range.
            df.loc[~df[col].between(lo, hi), col] = np.nan
    # One-sided (lower) bounds.
    for col, lo in DOMAIN_LOWER_ONLY.items():
        if col in df.columns:
            df.loc[df[col] < lo, col] = np.nan
    return df


def _apply_percentile_trim(
    df: pd.DataFrame, lower: float = 0.005, upper: float = 0.995
) -> pd.DataFrame:
    """Replace values outside ``[lower, upper]`` quantiles with NaN.

    Run *after* the domain cutoff so it only fights statistical extremes
    that survived the hand-picked bounds (Ablation 1 in the report shows
    this earns an extra ~+0.002 Macro-F1 on Random Forest).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Public cleaning entry point
# ---------------------------------------------------------------------------

def clean_data(
    df: pd.DataFrame,
    outlier_mode: str = "domain",
    keep_customer_id: bool = False,
) -> pd.DataFrame:
    """Clean the raw credit-score dataframe.

    Pipeline (see Section 3.1 of the report):
      1. Placeholder strings -> NaN.
      2. Numeric-like object columns -> numeric (underscore stripped).
      3. Credit_History_Age string -> total months (new column).
      4. Drop ID / Name / SSN (optionally Customer_ID).
      5. Outlier handling depending on ``outlier_mode``.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw frame loaded directly from ``data/raw/train.csv``.
    outlier_mode : {'none', 'domain', 'domain_percentile'}, default='domain'
        - ``'none'``: keep all values after placeholder cleaning.
        - ``'domain'``: apply :data:`DOMAIN_BOUNDS` and
          :data:`DOMAIN_LOWER_ONLY` cutoffs.
        - ``'domain_percentile'``: domain cutoffs *and* a 0.5%-99.5%
          percentile trim on every numeric column.
    keep_customer_id : bool, default=False
        Keep the ``Customer_ID`` column for downstream group-wise
        imputation. The caller is responsible for dropping it later.

    Returns
    -------
    pandas.DataFrame
        Cleaned frame. The new ``Credit_History_Age_Months`` column
        replaces ``Credit_History_Age``.

    Raises
    ------
    ValueError
        If ``outlier_mode`` is not one of the three accepted strings.

    Examples
    --------
    >>> raw = pd.read_csv('data/raw/train.csv', low_memory=False)
    >>> clean = clean_data(raw, outlier_mode='domain_percentile', keep_customer_id=True)
    """
    if outlier_mode not in {"none", "domain", "domain_percentile"}:
        raise ValueError(f"unknown outlier_mode: {outlier_mode}")

    # Work on a copy so the caller's frame is untouched.
    df = df.copy()
    # Step 1: placeholder tokens -> NaN.
    df = df.replace(PLACEHOLDERS, np.nan)
    # Step 2: numeric-like-object columns -> proper numeric dtype.
    df = _coerce_numeric_like(df)

    # Step 3: Credit_History_Age (string) -> Credit_History_Age_Months (int).
    if "Credit_History_Age" in df.columns:
        df["Credit_History_Age_Months"] = df["Credit_History_Age"].apply(
            convert_credit_history_age
        )
        df = df.drop(columns=["Credit_History_Age"])

    # Step 4: drop high-cardinality identifier columns. We keep Customer_ID
    # only when the caller plans to do group-wise imputation downstream.
    drop_cols = ["ID", "Name", "SSN"]
    if not keep_customer_id:
        drop_cols.append("Customer_ID")
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Step 5: outlier handling - 'none' falls through with no extra work.
    if outlier_mode in {"domain", "domain_percentile"}:
        df = _apply_domain_cutoffs(df)
    if outlier_mode == "domain_percentile":
        df = _apply_percentile_trim(df)

    return df


def group_impute(
    df: pd.DataFrame, group_col: str = "Customer_ID"
) -> pd.DataFrame:
    """Forward/backward fill missing values inside each customer group.

    Every customer contributes 8 consecutive monthly rows. Within that
    group most attributes (Occupation, Monthly_Inhand_Salary, etc.) are
    effectively constant or smoothly varying, so a per-group ffill->bfill
    is a much more faithful imputation than the global median/mode that
    runs later inside the ColumnTransformer.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame that still contains ``group_col``.
    group_col : str, default='Customer_ID'
        Column to group on. If absent, the dataframe is returned as-is.

    Returns
    -------
    pandas.DataFrame
        Same shape as ``df`` with intra-group missing values filled
        wherever possible.

    Notes
    -----
    Any column that is entirely missing for a customer stays missing
    after this step; those cells get a global median/mode later in the
    ColumnTransformer.
    """
    if group_col not in df.columns:
        return df
    df = df.copy()

    # Apply the fill to numeric + object/categorical columns. (Booleans and
    # datetimes do not appear in this dataset.)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    fill_cols = [c for c in numeric_cols + object_cols if c != group_col]

    # ``sort=False`` keeps the original row order so the eight monthly
    # rows stay chronologically aligned for ffill/bfill.
    grouped = df.groupby(group_col, sort=False)
    filled = grouped[fill_cols].transform(lambda g: g.ffill().bfill())

    # ``infer_objects`` collapses any object columns whose underlying values
    # are now homogeneous numeric back to their proper dtype.
    df[fill_cols] = filled.infer_objects(copy=False)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append four domain-motivated ratio features.

    All four divisions add ``+ 1`` to the denominator to avoid
    division-by-zero for customers whose salary or loan count is zero.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned frame that already holds the base columns.

    Returns
    -------
    pandas.DataFrame
        Same frame with up to four new columns:
          * ``Debt_to_Income``  = Outstanding_Debt / (Annual_Income + 1)
          * ``EMI_to_Salary``   = Total_EMI_per_month / (Monthly_Inhand_Salary + 1)
          * ``Savings_Rate``    = Amount_invested_monthly / (Monthly_Inhand_Salary + 1)
          * ``Delay_per_Loan``  = Num_of_Delayed_Payment / (Num_of_Loan + 1)
    """
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
    """Split the multi-label ``Type_of_Loan`` column and binarize it.

    Example input cell:
        ``"Auto Loan, Credit-Builder Loan, Personal Loan, and Home Equity Loan"``

    After this function the four loan kinds become four binary columns
    named ``Loan_Auto_Loan``, ``Loan_Credit-Builder_Loan``, etc.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame containing ``col``.
    col : str, default='Type_of_Loan'
        Column to split.

    Returns
    -------
    df : pandas.DataFrame
        Frame with ``col`` removed and one binary column per loan kind
        appended.
    mlb : sklearn.preprocessing.MultiLabelBinarizer or None
        The fitted binarizer (``None`` if the column was absent), kept
        for inspecting ``mlb.classes_`` downstream.
    """
    if col not in df.columns:
        return df, None

    def split_loans(v):
        """Tokenise one cell into a list of clean loan strings."""
        if pd.isna(v):
            return []
        # The raw text often contains "..., and X" which we want to treat
        # as just "X". Strip whitespace and drop empty tokens.
        parts = [p.strip().replace("and ", "") for p in str(v).split(",")]
        return [p for p in parts if p]

    # Convert the column into a Series of token lists, which is what
    # MultiLabelBinarizer expects as input.
    series = df[col].apply(split_loans)
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(series)

    # Build a DataFrame with helpful column names (spaces -> underscores).
    encoded_df = pd.DataFrame(
        encoded,
        columns=[f"Loan_{c.replace(' ', '_')}" for c in mlb.classes_],
        index=df.index,
    )
    df = df.drop(columns=[col]).join(encoded_df)
    return df, mlb


# ---------------------------------------------------------------------------
# Configurable preprocessing pipeline (Ablation 3)
# ---------------------------------------------------------------------------

# Lookup so ``build_preprocessor`` and the sweep can pick a scaler by name.
SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def _build_encoder(kind: str):
    """Instantiate one of the three categorical encoders.

    Returns
    -------
    sklearn encoder
        - ``'onehot'``  -> :class:`OneHotEncoder` (dense, unknown categories
          handled silently with ``handle_unknown='ignore'``).
        - ``'ordinal'`` -> :class:`OrdinalEncoder` (unknowns mapped to -1).
        - ``'target'``  -> :class:`TargetEncoder` (sklearn >= 1.3). Imported
          lazily so older environments still pick up the rest of the module.
    """
    kind = kind.lower()
    if kind == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if kind == "ordinal":
        return OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    if kind == "target":
        # TargetEncoder shipped in sklearn 1.3+. The lazy import avoids
        # ImportError on older installs.
        from sklearn.preprocessing import TargetEncoder
        return TargetEncoder(random_state=42)
    raise ValueError(f"unknown encoder: {kind}")


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scaler: str = "robust",
    encoder: str = "onehot",
) -> ColumnTransformer:
    """Construct the ColumnTransformer used by every model.

    Numeric branch: ``SimpleImputer(strategy='median')`` -> chosen scaler.
    Categorical branch: ``SimpleImputer(strategy='most_frequent')`` ->
    chosen encoder.

    Both branches are wrapped in their own sklearn :class:`Pipeline` so
    GridSearchCV can address sub-steps by name if needed (e.g.
    ``preprocessor__num__scaler``).

    Parameters
    ----------
    numeric_features, categorical_features : list of str
        Column names per branch (usually obtained from :func:`split_features`).
    scaler : {'standard', 'minmax', 'robust'}, default='robust'
        Numeric scaler. RobustScaler is the project default because it is
        based on quantiles and therefore tolerant of the heavy tails that
        survive percentile trimming.
    encoder : {'onehot', 'ordinal', 'target'}, default='onehot'
        Categorical encoder. OneHotEncoder is the project default; the
        other two are compared in Ablation 3 (Section 5.6 of the report).

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Ready to be chained with a model in a Pipeline.
    """
    scaler_cls = SCALERS[scaler.lower()]
    numeric_pipeline = Pipeline(
        [
            # Median is robust to the long-tailed financial columns.
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler_cls()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            # Most-frequent is the standard choice for categorical NaN.
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
    """Split ``df`` into features ``X``, target ``y``, and feature-name lists.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned frame including the target column.
    target : str, default='Credit_Score'
        Name of the target column.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix (target column dropped).
    y : pandas.Series
        Target vector.
    numeric : list of str
        Names of numeric columns in ``X``.
    categorical : list of str
        Names of object/categorical columns in ``X``.
    """
    X = df.drop(columns=[target])
    y = df[target]
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return X, y, numeric, categorical
