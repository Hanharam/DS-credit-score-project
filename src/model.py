"""Model factories, parameter grids, and pipeline builders."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def get_models(class_weight: str | dict | None = None) -> dict[str, Any]:
    """GradientBoosting silently ignores class_weight (sklearn limitation)."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            n_jobs=-1,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=15,
            class_weight=class_weight,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }


def get_param_grids() -> dict[str, dict[str, list]]:
    return {
        "LogisticRegression": {
            "model__C": [0.1, 1.0, 5.0],
        },
        "DecisionTree": {
            "model__max_depth": [10, 15, 25],
            "model__min_samples_leaf": [1, 5, 20],
        },
        "RandomForest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [15, 25, None],
            "model__min_samples_leaf": [1, 5],
        },
        "GradientBoosting": {
            "model__n_estimators": [150, 300],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5],
        },
    }


def build_pipeline(preprocessor, model, use_smote: bool = False):
    """Wrap preprocessor + model. If use_smote, use imblearn.Pipeline."""
    if use_smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline(
            [
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", model),
            ]
        )

    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
