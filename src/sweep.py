"""Single top-level full sweep over preprocessing × model × hyperparameter space.

This module fulfils the *Open Source SW Contribution* requirement of the
Data Science term project specification (TermProject_2026.pdf, slide 6):
all preprocessing combinations, all models, all hyperparameter combinations,
and k-fold evaluation are wrapped inside one entry point
:func:`run_full_sweep`. The function returns a leaderboard ranked by
Macro-F1 plus the top-K and the single best configuration.

Examples
--------
>>> from src.preprocessing import (
...     clean_data, group_impute, add_engineered_features, encode_type_of_loan,
... )
>>> import pandas as pd
>>> raw = pd.read_csv('data/raw/train.csv', low_memory=False)
>>> df = clean_data(raw, outlier_mode='domain', keep_customer_id=True)
>>> df = group_impute(df, group_col='Customer_ID').drop(columns=['Customer_ID'])
>>> df, _ = encode_type_of_loan(df)
>>> df = add_engineered_features(df)
>>> board, top5, best = run_full_sweep(df, top_k=5, cv=3, sample_size=20000)
>>> top5.to_csv('reports/full_sweep_top5.csv', index=False)
>>> print(best['model'], best['scaler'], best['encoder'], best['macro_f1'])
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from .model import build_pipeline, get_models
from .preprocessing import build_preprocessor, split_features


# ---------- Defaults ---------------------------------------------------------

DEFAULT_SCALERS: tuple[str, ...] = ("standard", "minmax", "robust")
DEFAULT_ENCODERS: tuple[str, ...] = ("onehot", "ordinal", "target")

# Compact per-model hyperparameter grids. Designed to keep the full sweep
# (scaler × encoder × model × params × k-fold) tractable on a laptop while
# still showing meaningful parameter variation.
DEFAULT_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "LogisticRegression": {
        "model__C": [0.1, 1.0, 5.0],
    },
    "DecisionTree": {
        "model__max_depth": [10, 25],
        "model__min_samples_leaf": [1, 20],
    },
    "RandomForest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [15, None],
    },
    "GradientBoosting": {
        "model__n_estimators": [150],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [3, 5],
    },
}

DEFAULT_SCORING: dict[str, str] = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "macro_recall": "recall_macro",
}


# ---------- Helpers ----------------------------------------------------------

def _iter_param_grid(grid: dict[str, list]) -> Iterable[dict]:
    """Yield every concrete parameter combination from a sklearn-style grid."""
    if not grid:
        yield {}
        return
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))


def _format_params(params: dict) -> str:
    if not params:
        return "{}"
    parts = []
    for k, v in params.items():
        short = k.replace("model__", "")
        parts.append(f"{short}={v}")
    return ", ".join(parts)


@dataclass
class SweepResult:
    """Container for one (scaler, encoder, model, params) configuration."""
    scaler: str
    encoder: str
    model: str
    params: dict
    macro_f1: float
    macro_f1_std: float
    accuracy: float
    macro_recall: float
    fit_seconds: float
    extras: dict = field(default_factory=dict)


# ---------- Top-level function ----------------------------------------------

def run_full_sweep(
    df: pd.DataFrame,
    target: str = "Credit_Score",
    scalers: Iterable[str] = DEFAULT_SCALERS,
    encoders: Iterable[str] = DEFAULT_ENCODERS,
    models: Iterable[str] | None = None,
    param_grids: dict[str, dict[str, list]] | None = None,
    cv: int = 3,
    scoring: dict[str, str] | None = None,
    primary_metric: str = "macro_f1",
    top_k: int = 5,
    class_weight: str | None = "balanced",
    use_smote: bool = False,
    sample_size: int | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Exhaustive sweep over scaler × encoder × model × hyperparameters.

    Implements the Open Source SW Contribution requirement of the course
    spec: all preprocessing and model variants are evaluated inside a
    single top-level function and ranked, instead of being copy-pasted
    across notebooks.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned feature frame that already includes the target column.
        The caller is responsible for upstream cleaning (placeholder
        replacement, outlier handling, group-aware imputation,
        ``Type_of_Loan`` multi-label expansion, engineered features).
    target : str, default='Credit_Score'
        Name of the supervised target column in ``df``.
    scalers : iterable of {'standard', 'minmax', 'robust'}
        Numeric scalers to sweep. See ``src.preprocessing.SCALERS``.
    encoders : iterable of {'onehot', 'ordinal', 'target'}
        Categorical encoders to sweep. See ``src.preprocessing._build_encoder``.
    models : iterable of str, optional
        Subset of model names defined in ``src.model.get_models``.
        Defaults to all four: LogisticRegression, DecisionTree,
        RandomForest, GradientBoosting.
    param_grids : dict, optional
        Mapping ``{model_name: {param: [values, ...]}}`` in scikit-learn
        ``GridSearchCV`` format. Defaults to ``DEFAULT_PARAM_GRIDS``.
    cv : int, default=3
        Number of folds for stratified k-fold cross-validation.
    scoring : dict, optional
        Mapping ``{label: sklearn_scoring_id}``. Must include the
        ``primary_metric``. Defaults to accuracy / macro-F1 / macro-recall.
    primary_metric : str, default='macro_f1'
        Key in ``scoring`` used to rank the leaderboard.
    top_k : int, default=5
        Number of rows returned in the ``top_k`` table. The spec asks
        for "top five and best combination".
    class_weight : {'balanced', None} or dict, default='balanced'
        Forwarded to every classifier that supports the kwarg.
    use_smote : bool, default=False
        If True, insert ``imblearn.SMOTE`` between preprocessor and model.
        Mutually exclusive with ``class_weight`` for a fair comparison;
        callers wanting both should re-invoke the sweep.
    sample_size : int, optional
        If set, randomly subsample this many rows from ``df`` *before*
        the sweep. Useful for keeping runtime bounded on a laptop;
        ``None`` runs the full dataset.
    random_state : int, default=42
        Random seed for fold splits and subsampling.
    verbose : bool, default=True
        If True, print progress for every configuration as it completes.

    Returns
    -------
    leaderboard : pandas.DataFrame
        All evaluated configurations sorted by ``primary_metric``
        descending. Columns: ``scaler, encoder, model, params,
        macro_f1, macro_f1_std, macro_recall, accuracy, fit_seconds``.
    top_k_df : pandas.DataFrame
        First ``top_k`` rows of ``leaderboard``.
    best : dict
        First row of ``leaderboard`` as a plain dict.

    Notes
    -----
    The number of fits performed is::

        n_fits = len(scalers) * len(encoders)
                 * sum(prod(len(v) for v in grid.values()) for grid in param_grids)
                 * cv

    With the defaults (3 × 3 × (3 + 4 + 4 + 4) × 3) = 405 fits.

    Examples
    --------
    >>> board, top5, best = run_full_sweep(df, top_k=5, sample_size=20000)
    >>> top5[['scaler', 'encoder', 'model', 'macro_f1']]
    """
    # Defaults
    models = list(models) if models else list(DEFAULT_PARAM_GRIDS.keys())
    param_grids = param_grids if param_grids is not None else DEFAULT_PARAM_GRIDS
    scoring = scoring if scoring is not None else DEFAULT_SCORING
    if primary_metric not in scoring:
        raise ValueError(
            f"primary_metric={primary_metric!r} not in scoring={list(scoring)}"
        )

    work_df = df
    if sample_size is not None and sample_size < len(df):
        work_df = df.sample(n=sample_size, random_state=random_state).reset_index(
            drop=True
        )
    X, y, numeric, categorical = split_features(work_df, target=target)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    # Pre-build estimators once per (model, params) tuple, then reuse them
    # against every scaler×encoder pipeline.
    results: list[SweepResult] = []
    total = (
        len(list(scalers))
        * len(list(encoders))
        * sum(
            max(1, np.prod([len(v) for v in param_grids[m].values()]))
            for m in models
        )
    )
    if verbose:
        print(
            f"[sweep] dataset shape={work_df.shape}  "
            f"configs={total}  folds={cv}  total_fits={total * cv}",
            flush=True,
        )

    config_idx = 0
    for scaler_name, encoder_name in itertools.product(scalers, encoders):
        preprocessor = build_preprocessor(
            numeric, categorical, scaler=scaler_name, encoder=encoder_name
        )
        for model_name in models:
            grid = param_grids.get(model_name, {})
            for params in _iter_param_grid(grid):
                config_idx += 1
                base = get_models(class_weight=class_weight)[model_name]
                try:
                    base.set_params(**{k.replace("model__", ""): v
                                       for k, v in params.items()})
                except ValueError as exc:
                    if verbose:
                        print(f"  [skip] {model_name} {params}: {exc}")
                    continue

                pipeline = build_pipeline(preprocessor, base, use_smote=use_smote)

                t0 = time.time()
                try:
                    scores = cross_validate(
                        pipeline,
                        X,
                        y,
                        cv=skf,
                        scoring=scoring,
                        n_jobs=1,
                        return_train_score=False,
                    )
                except Exception as exc:
                    if verbose:
                        print(
                            f"  [{config_idx}/{total}] FAILED "
                            f"{scaler_name}/{encoder_name}/{model_name} "
                            f"({_format_params(params)}): {exc}"
                        )
                    continue
                elapsed = time.time() - t0

                row = SweepResult(
                    scaler=scaler_name,
                    encoder=encoder_name,
                    model=model_name,
                    params=params,
                    macro_f1=float(scores["test_macro_f1"].mean()),
                    macro_f1_std=float(scores["test_macro_f1"].std()),
                    accuracy=float(scores["test_accuracy"].mean()),
                    macro_recall=float(scores["test_macro_recall"].mean()),
                    fit_seconds=elapsed,
                )
                results.append(row)
                if verbose:
                    print(
                        f"  [{config_idx}/{total}] "
                        f"{scaler_name:8s}/{encoder_name:7s}/{model_name:18s} "
                        f"{_format_params(params):40s}  "
                        f"f1={row.macro_f1:.4f}±{row.macro_f1_std:.4f}  "
                        f"({elapsed:.1f}s)",
                        flush=True,
                    )

    # Assemble leaderboard
    rows = [
        {
            "scaler": r.scaler,
            "encoder": r.encoder,
            "model": r.model,
            "params": _format_params(r.params),
            "macro_f1": r.macro_f1,
            "macro_f1_std": r.macro_f1_std,
            "macro_recall": r.macro_recall,
            "accuracy": r.accuracy,
            "fit_seconds": r.fit_seconds,
        }
        for r in results
    ]
    leaderboard = pd.DataFrame(rows).sort_values(
        primary_metric, ascending=False
    ).reset_index(drop=True)
    leaderboard.insert(0, "rank", leaderboard.index + 1)

    top_k_df = leaderboard.head(top_k).reset_index(drop=True)
    best = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}

    return leaderboard, top_k_df, best


def save_sweep_artifacts(
    leaderboard: pd.DataFrame,
    top_k_df: pd.DataFrame,
    best: dict,
    out_dir: str | Path,
) -> dict[str, Path]:
    """Persist sweep artifacts to ``out_dir``.

    Writes ``full_sweep.csv``, ``full_sweep_top5.csv``, ``full_sweep_best.json``.

    Returns
    -------
    paths : dict
        Mapping ``{name: Path}`` of the files written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "leaderboard": out_dir / "full_sweep.csv",
        "top_k": out_dir / "full_sweep_top5.csv",
        "best": out_dir / "full_sweep_best.json",
    }
    leaderboard.to_csv(paths["leaderboard"], index=False)
    top_k_df.to_csv(paths["top_k"], index=False)
    with open(paths["best"], "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    return paths
