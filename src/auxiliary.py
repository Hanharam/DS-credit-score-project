"""Auxiliary tasks from the proposal:
- regression on Monthly_Balance
- KMeans clustering on Payment_Behaviour
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
    """Predict Monthly_Balance from the remaining features."""
    if target not in df.columns:
        raise KeyError(f"{target} not in DataFrame")

    work = df.dropna(subset=[target]).copy()
    y = work[target].astype(float)
    work = work.drop(columns=["Credit_Score"], errors="ignore")
    X = work.drop(columns=[target])

    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    preprocessor = build_preprocessor(numeric, categorical, scaler="robust", encoder="onehot")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    from sklearn.pipeline import Pipeline

    results = {}
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
    """KMeans on spending/saving behaviour signals; report silhouette and centroids."""
    feats = [c for c in PAYMENT_BEHAVIOUR_FEATURES if c in df.columns]
    work = df[feats].dropna().copy()
    if len(work) < n_clusters * 10:
        raise ValueError("not enough complete rows for clustering")

    scaler = StandardScaler()
    X = scaler.fit_transform(work)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)

    sample = min(20000, len(X))
    sil = float(
        silhouette_score(X[:sample], labels[:sample], random_state=random_state)
    )

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
    feats = [c for c in PAYMENT_BEHAVIOUR_FEATURES if c in df.columns]
    work = df[feats].dropna().copy()
    X = StandardScaler().fit_transform(work)

    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        sample = min(20000, len(X))
        sil = silhouette_score(X[:sample], labels[:sample], random_state=random_state)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": float(sil)})
    return pd.DataFrame(rows)
