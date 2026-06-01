"""Model factories, hyperparameter grids, and pipeline builders.

Centralises every classifier used in the project so the rest of the
codebase (training, evaluation, ablations, sweep) refers to models by
name only. This makes it trivial to add or remove a model in one place.

Models compared (Section 4.1 / 5.1 of the report):

* :class:`sklearn.linear_model.LogisticRegression` - linear baseline.
* :class:`sklearn.tree.DecisionTreeClassifier`     - single-tree baseline
  (the model taught in class).
* :class:`sklearn.ensemble.RandomForestClassifier` - bagging ensemble.
* :class:`sklearn.ensemble.GradientBoostingClassifier` - boosting ensemble.

GradientBoosting in scikit-learn does NOT accept a ``class_weight``
argument - this is documented in the report Section 5.5 and Appendix B.
``get_models`` therefore passes it to the other three only.

When SMOTE is requested, :func:`build_pipeline` swaps the standard
sklearn Pipeline for :class:`imblearn.pipeline.Pipeline`, which is
mandatory because SMOTE has a ``fit_resample`` method that the base
sklearn Pipeline does not know how to call.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def get_models(class_weight: str | dict | None = None) -> dict[str, Any]:
    """Return a freshly instantiated ``{name: estimator}`` dictionary.

    A new dictionary is built on every call so callers can safely
    re-fit the same factory inside a CV loop without sharing state.

    Parameters
    ----------
    class_weight : {'balanced', None} or dict, default=None
        Forwarded to LogisticRegression, DecisionTree, and RandomForest.
        Silently ignored by GradientBoostingClassifier because
        scikit-learn does not support it there.

    Returns
    -------
    dict[str, sklearn.base.BaseEstimator]
        Mapping with keys ``LogisticRegression``, ``DecisionTree``,
        ``RandomForest``, ``GradientBoosting``. Initial hyperparameters
        match Table 7 of the report.
    """
    return {
        # Linear baseline. max_iter=2000 because the default 100 is too
        # small for a 100k-row dataset with one-hot expansion. n_jobs=-1
        # uses all available cores during fit.
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            n_jobs=-1,
        ),
        # Single tree, depth-capped at 15 to keep fit time predictable and
        # discourage extreme overfitting on the 100k-row training set.
        "DecisionTree": DecisionTreeClassifier(
            max_depth=15,
            class_weight=class_weight,
            random_state=42,
        ),
        # Bagging ensemble - the project default and the winning model.
        # 200 trees / depth 20 is the report's baseline; GridSearchCV
        # later tunes these.
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        ),
        # Boosting ensemble. learning_rate=0.05 with 150 stages is the
        # report's baseline configuration. class_weight is intentionally
        # not passed - sklearn's GradientBoosting does not accept it.
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }


def get_param_grids() -> dict[str, dict[str, list]]:
    """Hyperparameter grids consumed by GridSearchCV.

    Keys are nested with the ``model__`` prefix so they line up with the
    sklearn Pipeline step name used in :func:`build_pipeline`
    (``Pipeline(..., ('model', estimator), ...)``).

    Sizes per model (used to compute total fit count in the report):
      * LogisticRegression - 3 combos
      * DecisionTree       - 3 x 3 = 9 combos
      * RandomForest       - 2 x 3 x 2 = 12 combos (search expanded in
        Section 4.2(2) of the report)
      * GradientBoosting   - 2 x 2 x 2 = 8 combos
    """
    return {
        "LogisticRegression": {
            # Regularization strength: smaller C = stronger regularization.
            "model__C": [0.1, 1.0, 5.0],
        },
        "DecisionTree": {
            # Cap tree depth to control variance.
            "model__max_depth": [10, 15, 25],
            # Larger leaves -> smoother decisions, less overfit.
            "model__min_samples_leaf": [1, 5, 20],
        },
        "RandomForest": {
            "model__n_estimators": [200, 400],
            # ``None`` means unbounded (let the leaf-size rule decide).
            "model__max_depth": [15, 25, None],
            "model__min_samples_leaf": [1, 5],
        },
        "GradientBoosting": {
            "model__n_estimators": [150, 300],
            # Lower learning rate often needs more trees and vice versa.
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5],
        },
    }


def build_pipeline(preprocessor, model, use_smote: bool = False):
    """Compose ``preprocessor + (SMOTE) + model`` into a single estimator.

    Parameters
    ----------
    preprocessor : sklearn.compose.ColumnTransformer
        Usually the one returned by :func:`src.preprocessing.build_preprocessor`.
    model : sklearn.base.BaseEstimator
        Any of the classifiers from :func:`get_models`.
    use_smote : bool, default=False
        If True, insert :class:`imblearn.over_sampling.SMOTE` between the
        preprocessor and the model. Note that this requires the
        imblearn-flavoured Pipeline because sklearn's own Pipeline does
        not understand ``fit_resample``. SMOTE is applied only to the
        training fold (no test-time data leakage).

    Returns
    -------
    sklearn.pipeline.Pipeline or imblearn.pipeline.Pipeline
        A Pipeline ready to be passed to ``cross_validate`` / GridSearchCV.
    """
    if use_smote:
        # imblearn is only imported here to keep the project usable when
        # imbalanced-learn is not installed (relevant for environments
        # that skip the imbalance ablation).
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline(
            [
                ("preprocessor", preprocessor),
                # random_state pinned so the synthetic samples are
                # reproducible across runs.
                ("smote", SMOTE(random_state=42)),
                ("model", model),
            ]
        )

    # Standard sklearn pipeline path - no resampling.
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
