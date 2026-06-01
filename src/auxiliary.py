"""Auxiliary tasks defined in the proposal (Section 6 of the report).

Two analyses live here:

1. :func:`regress_monthly_balance` - regression task that predicts
   ``Monthly_Balance`` from the remaining cleaned features, comparing
   :class:`sklearn.linear_model.Ridge` to
   :class:`sklearn.ensemble.GradientBoostingRegressor` (Section 6.1).

2. :func:`cluster_payment_behaviour` plus :func:`kmeans_elbow` - KMeans
   clustering on seven spending / saving features, with silhouette score
   and a separate elbow-curve sweep (Section 6.2).

Both functions assume the dataframe has already been cleaned by
:func:`src.preprocessing.clean_data` and friends. The auxiliary tasks
deliberately reuse the project's main ColumnTransformer so they enjoy
the same imputation and encoding choices as the classification path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .preprocessing import build_preprocessor, split_features


def regress_monthly_balance(
    df: pd.DataFrame,
    target: str = "Monthly_Balance",
    test_size: float = 0.2,
) -> dict:
    """Compare Ridge and GradientBoostingRegressor on ``Monthly_Balance``.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned / imputed / engineered frame (output of
        :func:`src.evaluate._build_dataset`).
    target : str, default='Monthly_Balance'
        Column to predict.
    test_size : float, default=0.2
        Fraction held out for evaluation. ``random_state=42`` is pinned
        for reproducibility.

    Returns
    -------
    dict
        ``{model_name: {'r2': float, 'mae': float}}``.

    Raises
    ------
    KeyError
        If ``target`` is not in ``df``.

    Notes
    -----
    ``Credit_Score`` is dropped from the feature set so the regression
    task does not "see" the classification label.
    """
    if target not in df.columns:
        raise KeyError(f"{target} not in DataFrame")

    # Rows with missing target cannot contribute to a regression metric.
    work = df.dropna(subset=[target]).copy()
    y = work[target].astype(float)
    # Drop the classification label so the regression cannot peek at it.
    work = work.drop(columns=["Credit_Score"], errors="ignore")
    X = work.drop(columns=[target])

    # Reuse the same preprocessing strategy as the classification path
    # (median impute + RobustScaler + OneHotEncoder).
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    preprocessor = build_preprocessor(numeric, categorical, scaler="robust", encoder="onehot")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Local import keeps the module's top-level import block tidy.
    from sklearn.pipeline import Pipeline

    results = {}
    # Compare a linear baseline (Ridge) and a non-linear ensemble (GBR).
    for name, model in [
        ("Ridge", Ridge(alpha=1.0)),
        ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
    ]:
        pipe = Pipeline([("pre", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_valid)
        results[name] = {
            "r2": float(r2_score(y_valid, pred)),
            "mae": float(mean_absolute_error(y_valid, pred)),
        }
    return results


# Seven spending / saving variables that together capture a customer's
# financial behaviour pattern. Used by both clustering helpers below so
# the cluster space is identical between them.
PAYMENT_BEHAVIOUR_FEATURES = [
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
]


def cluster_payment_behaviour(
    df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42
) -> dict:
    """KMeans clustering on payment-behaviour signals.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame containing (at least some of) the columns in
        :data:`PAYMENT_BEHAVIOUR_FEATURES`. Columns missing from ``df``
        are silently skipped.
    n_clusters : int, default=4
        ``k`` for KMeans. Report uses ``k=4`` (Section 6.2).
    random_state : int, default=42

    Returns
    -------
    dict
        Keys:
          * ``n_clusters``      - same as the argument.
          * ``silhouette``      - silhouette score on (a subsample of) the
            scaled feature space.
          * ``cluster_sizes``   - {label: count}.
          * ``centroids``       - DataFrame in original feature scale
            (inverse-transformed).
          * ``labels``          - per-row cluster assignments.
          * ``features_used``   - columns actually used.

    Raises
    ------
    ValueError
        If after dropping rows with NaN there are fewer than
        ``n_clusters * 10`` rows left.
    """
    # Only keep the requested feature columns that actually exist.
    feats = [c for c in PAYMENT_BEHAVIOUR_FEATURES if c in df.columns]
    work = df[feats].dropna().copy()
    if len(work) < n_clusters * 10:
        raise ValueError("not enough complete rows for clustering")

    # Standardise so the KMeans Euclidean distance is balanced across
    # variables that are otherwise on very different scales
    # (Annual_Income vs Credit_Utilization_Ratio etc.).
    scaler = StandardScaler()
    X = scaler.fit_transform(work)

    # n_init=10 runs the KMeans initialisation 10 times and keeps the
    # best to avoid local minima.
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)

    # silhouette_score is O(n^2) so a 20k subsample keeps it tractable
    # without changing the score noticeably.
    sample = min(20000, len(X))
    sil = float(
        silhouette_score(X[:sample], labels[:sample], random_state=random_state)
    )

    # Centroids are reported in the original feature scale (inverse the
    # StandardScaler) so the values are human-readable.
    centroids = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_), columns=feats
    )
    cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()

    return {
        "n_clusters": n_clusters,
        "silhouette": sil,
        "cluster_sizes": cluster_sizes,
        "centroids": centroids,
        "labels": labels,
        "features_used": feats,
    }


def kmeans_elbow(
    df: pd.DataFrame, k_range=range(2, 9), random_state: int = 42
) -> pd.DataFrame:
    """Sweep ``k`` from 2 to 8 and return inertia + silhouette per ``k``.

    Used by ``src/train.py`` to generate the elbow + silhouette curve
    figure (``reports/figures/aux_kmeans_elbow.png``) that motivates
    the ``k=4`` choice.

    Parameters
    ----------
    df : pandas.DataFrame
    k_range : iterable of int, default=range(2, 9)
        Cluster counts to evaluate.
    random_state : int, default=42

    Returns
    -------
    pandas.DataFrame
        Columns ``k``, ``inertia``, ``silhouette``.
    """
    feats = [c for c in PAYMENT_BEHAVIOUR_FEATURES if c in df.columns]
    work = df[feats].dropna().copy()
    X = StandardScaler().fit_transform(work)

    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        # Same 20k subsample trick as in cluster_payment_behaviour.
        sample = min(20000, len(X))
        sil = silhouette_score(X[:sample], labels[:sample], random_state=random_state)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": float(sil)})
    return pd.DataFrame(rows)
