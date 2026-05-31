"""All plot generation. Functions accept a save_path; if given, the figure is
written to disk and closed. Used by src/train.py via --save-plots."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")  # headless-safe default
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .eda import (
    DEFAULT_CATEGORICAL_EDA_COLS,
    DEFAULT_NUMERIC_EDA_COLS,
    KEY_FEATURES_BY_TARGET,
)
from .preprocessing import (
    NUMERIC_LIKE_OBJECT_COLS,
    _coerce_numeric_like,
    clean_data,
)


sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, save_path: str | Path | None) -> plt.Figure:
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------- EDA plots (mirror proposal Figures 1-7) ----------------------

def plot_target_distribution(
    df: pd.DataFrame,
    target: str = "Credit_Score",
    save_path: str | None = None,
) -> plt.Figure:
    """Figure 1 — class counts of the target."""
    counts = df[target].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Blues_d")
    total = counts.sum()
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v / total * 100:.1f}%)", ha="center", va="bottom")
    ax.set_title(f"Distribution of {target}")
    ax.set_xlabel(target)
    ax.set_ylabel("count")
    return _save(fig, save_path)


def plot_numeric_histograms(
    df: pd.DataFrame,
    cols: Iterable[str] | None = None,
    bins: int = 50,
    save_path: str | None = None,
) -> plt.Figure:
    """Figure 2 — grid of histograms after light cleaning so impossible
    values do not crush the x-axis."""
    cleaned = clean_data(df, outlier_mode="domain", keep_customer_id=False)
    cols = list(cols) if cols else [c for c in DEFAULT_NUMERIC_EDA_COLS if c in cleaned.columns]

    n = len(cols)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, col in zip(axes, cols):
        sns.histplot(cleaned[col].dropna(), bins=bins, ax=ax, kde=False)
        ax.set_title(col)
        ax.set_xlabel("")
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Numeric feature distributions (post domain-cleaning)", y=1.02)
    return _save(fig, save_path)


def plot_boxplots_before_after(
    df: pd.DataFrame,
    cols: Iterable[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Figures 3 & 4 — side-by-side boxplots before vs after outlier cleaning.

    Uses raw numeric-coerced values for `before` and domain-cut values for
    `after` so reviewers can see the IQR boxes that used to be invisible.
    """
    raw = df.copy()
    raw = _coerce_numeric_like(raw)
    cleaned = clean_data(df, outlier_mode="domain", keep_customer_id=False)

    cols = list(cols) if cols else [c for c in DEFAULT_NUMERIC_EDA_COLS if c in cleaned.columns]
    n = len(cols)
    nrows = n
    fig, axes = plt.subplots(nrows, 2, figsize=(10, 2.2 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    for i, col in enumerate(cols):
        sns.boxplot(x=raw[col], ax=axes[i, 0], color="#f4a261")
        axes[i, 0].set_title(f"{col} — before")
        sns.boxplot(x=cleaned[col], ax=axes[i, 1], color="#2a9d8f")
        axes[i, 1].set_title(f"{col} — after")
    return _save(fig, save_path)


def plot_correlation_matrix(
    df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Figure 5 — Pearson correlation heatmap on cleaned numeric columns."""
    cleaned = clean_data(df, outlier_mode="domain", keep_customer_id=False)
    numeric = cleaned.select_dtypes(include=[np.number])
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(corr)), max(7, 0.45 * len(corr))))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"shrink": 0.75},
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Pearson correlation (post domain-cleaning)")
    return _save(fig, save_path)


def plot_categorical_distributions(
    df: pd.DataFrame,
    cols: Iterable[str] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Figure 6 — count plots of key categorical variables."""
    cols = list(cols) if cols else [c for c in DEFAULT_CATEGORICAL_EDA_COLS if c in df.columns]
    n = len(cols)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, col in zip(axes, cols):
        order = df[col].value_counts().index
        sns.countplot(data=df, x=col, order=order, ax=ax, palette="Set2")
        ax.set_title(col)
        ax.set_xlabel("")
        for label in ax.get_xticklabels():
            label.set_rotation(40)
            label.set_ha("right")
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    return _save(fig, save_path)


def plot_boxplot_by_target(
    df: pd.DataFrame,
    cols: Iterable[str] | None = None,
    target: str = "Credit_Score",
    save_path: str | None = None,
) -> plt.Figure:
    """Figure 7 — boxplots of key signal features grouped by target class."""
    cleaned = clean_data(df, outlier_mode="domain", keep_customer_id=False)
    cols = list(cols) if cols else [c for c in KEY_FEATURES_BY_TARGET if c in cleaned.columns]
    n = len(cols)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    order = ["Poor", "Standard", "Good"]
    for ax, col in zip(axes, cols):
        sns.boxplot(data=cleaned, x=target, y=col, order=order, ax=ax, palette="Set3")
        ax.set_title(f"{col} by {target}")
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    return _save(fig, save_path)


# ---------- Ablation / CV result plots ----------------------------------

def plot_ablation_outlier(
    ablation_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=ablation_df,
        x="outlier_mode",
        y="macro_f1",
        hue="model",
        ax=ax,
        palette="Set2",
        order=["none", "domain", "domain_percentile"],
    )
    ax.set_title("Outlier handling vs Macro-F1 (Stratified k-fold mean)")
    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("Outlier strategy")
    ax.legend(title="model", loc="lower right")
    return _save(fig, save_path)


def plot_ablation_imbalance(
    ablation_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in zip(
        axes,
        ["macro_f1", "macro_recall"],
        ["Macro-F1", "Macro-Recall"],
    ):
        sns.barplot(
            data=ablation_df,
            x="strategy",
            y=metric,
            hue="model",
            ax=ax,
            palette="Set1",
            order=["none", "class_weight", "smote"],
        )
        ax.set_title(f"Class imbalance handling vs {title}")
        ax.set_ylabel(title)
        ax.set_xlabel("Imbalance strategy")
    return _save(fig, save_path)


def plot_ablation_preprocessing(
    ablation_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Heatmap of macro-F1 across 3 scalers x 3 encoders."""
    pivot = ablation_df.pivot(index="scaler", columns="encoder", values="macro_f1")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".4f",
        cmap="YlGnBu",
        cbar_kws={"label": "Macro-F1"},
        ax=ax,
    )
    ax.set_title("Scaler x Encoder sweep (Macro-F1)")
    return _save(fig, save_path)


def plot_cv_baseline(
    cv_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart with error bars across baseline models."""
    df = cv_df.copy().sort_values("macro_f1_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        df["model"],
        df["macro_f1_mean"],
        yerr=df["macro_f1_std"],
        capsize=6,
        color=sns.color_palette("crest", len(df)),
        edgecolor="black",
    )
    ax.axhline(0.75, ls="--", color="red", alpha=0.6, label="proposal target 0.75")
    ax.set_ylabel("Macro-F1 (mean +/- std)")
    ax.set_title("Stratified k-fold CV — baseline models")
    ax.legend()
    for i, (m, s) in enumerate(zip(df["macro_f1_mean"], df["macro_f1_std"])):
        ax.text(i, m + s + 0.005, f"{m:.3f}", ha="center", va="bottom")
    return _save(fig, save_path)


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
) -> plt.Figure:
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return _save(fig, save_path)


def plot_per_class_metrics(
    report_dict: dict,
    save_path: str | None = None,
) -> plt.Figure:
    """Bar chart of precision/recall/f1 per target class. Pass the dict from
    sklearn.metrics.classification_report(..., output_dict=True)."""
    class_rows = {
        k: v
        for k, v in report_dict.items()
        if k not in {"accuracy", "macro avg", "weighted avg"}
    }
    classes = list(class_rows.keys())
    metrics = ["precision", "recall", "f1-score"]
    data = pd.DataFrame(
        [
            {
                "class": cls,
                "metric": m,
                "value": class_rows[cls][m],
            }
            for cls in classes
            for m in metrics
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=data, x="class", y="value", hue="metric", ax=ax, palette="muted")
    ax.axhline(0.70, ls="--", color="red", alpha=0.5, label="recall target 0.70")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class precision / recall / F1 (hold-out)")
    ax.legend(loc="lower right")
    return _save(fig, save_path)


def plot_kmeans_elbow(
    elbow_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(elbow_df["k"], elbow_df["inertia"], "o-", color="tab:blue", label="inertia")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia (lower is tighter)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(elbow_df["k"], elbow_df["silhouette"], "s--", color="tab:orange", label="silhouette")
    ax2.set_ylabel("Silhouette (higher is better)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_title("KMeans elbow + silhouette over k")
    return _save(fig, save_path)


def plot_full_sweep_top5(
    top_k_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart of the top-K leaderboard from run_full_sweep.

    Expects columns: rank, scaler, encoder, model, params, macro_f1, macro_f1_std.
    """
    df = top_k_df.copy()
    df["label"] = df.apply(
        lambda r: f"#{int(r['rank'])} {r['model']} | "
                  f"{r['scaler']}+{r['encoder']}\n{r['params']}",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.9 * len(df))))
    bars = ax.barh(
        df["label"][::-1],
        df["macro_f1"][::-1],
        xerr=df["macro_f1_std"][::-1],
        color=sns.color_palette("crest", n_colors=len(df))[::-1],
    )
    ax.set_xlabel("Macro-F1 (CV mean)")
    ax.set_title("Top 5 sweep configurations (scaler × encoder × model × params)")
    ax.set_xlim(min(df["macro_f1"].min() - 0.02, 0.6), 1.0)
    for bar, value in zip(bars, df["macro_f1"][::-1]):
        ax.text(value + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}", va="center")
    return _save(fig, save_path)
