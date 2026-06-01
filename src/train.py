"""End-to-end training driver.

This module is the project's single CLI entry point. Running

    python -m src.train --all

reproduces every table, figure, and metric quoted in the report from
``data/raw/train.csv``.

Stages (each is independently togglable via a flag):

    1. Load raw train.csv.
    2. EDA tables + figures (``--run-eda``).
    3. Clean + group-aware imputation + ``Type_of_Loan`` MLB +
       feature engineering. Always runs because the downstream stages
       need the cleaned dataframe.
    4. Ablations (each gated by its own flag):
         * outlier handling          (``--run-outlier-ablation``)
         * class imbalance handling  (``--run-imbalance-ablation``)
         * scaler x encoder sweep    (``--run-preproc-ablation``)
       And the Open Source SW Contribution full sweep
       (``--run-sweep``) - the single top-level function
       ``run_full_sweep()`` defined in :mod:`src.sweep`.
    5. Stratified k-fold evaluation across all base models.
    6. GridSearchCV tuning on the best base model
       (skipped with ``--skip-grid`` for fast smoke tests).
    7. Final hold-out evaluation + confusion matrix + per-class metrics.
    8. Auxiliary tasks: regression on ``Monthly_Balance`` and KMeans on
       payment behaviour (``--run-auxiliary``).
    9. Persist every artifact under ``reports/`` and ``reports/figures/``
       (optionally including the fitted pipeline via ``--save-pipeline``).

CLI flags
---------
The ``--all`` super-flag turns on every stage. See
``parse_args()`` for the full list of switches.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auxiliary import (
    cluster_payment_behaviour,
    kmeans_elbow,
    regress_monthly_balance,
)
from src.eda import write_eda_reports
from src.evaluate import (
    ablation_imbalance,
    ablation_outlier,
    ablation_preprocessing,
    evaluate_cv,
    evaluate_holdout,
    tune_with_gridsearch,
)
from src.model import build_pipeline, get_models, get_param_grids
from src.preprocessing import (
    add_engineered_features,
    build_preprocessor,
    clean_data,
    encode_type_of_loan,
    group_impute,
    split_features,
)
from src.sweep import run_full_sweep, save_sweep_artifacts
from src.visualize import (
    plot_ablation_imbalance,
    plot_ablation_outlier,
    plot_ablation_preprocessing,
    plot_boxplot_by_target,
    plot_boxplots_before_after,
    plot_categorical_distributions,
    plot_confusion_matrix,
    plot_correlation_matrix,
    plot_cv_baseline,
    plot_full_sweep_top5,
    plot_kmeans_elbow,
    plot_numeric_histograms,
    plot_per_class_metrics,
    plot_target_distribution,
)


DEFAULT_OUT = ROOT / "reports"


def _log(msg: str) -> None:
    """Timestamped print used everywhere in this module.

    ``flush=True`` keeps the output live even when stdout is being
    piped to a file (``tee``) or captured by a CI runner.
    """
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare_dataset(raw_csv: Path, outlier_mode: str = "domain") -> pd.DataFrame:
    """Run the full preprocessing chain and return the model-ready frame.

    Sequence:
        ``clean_data`` (keeping Customer_ID)
        -> ``group_impute`` (per-Customer_ID ffill/bfill)
        -> drop Customer_ID
        -> ``encode_type_of_loan`` (MultiLabelBinarizer)
        -> ``add_engineered_features`` (4 ratio features).

    Parameters
    ----------
    raw_csv : pathlib.Path
        Path to ``data/raw/train.csv``.
    outlier_mode : {'none', 'domain', 'domain_percentile'}
        Forwarded to :func:`src.preprocessing.clean_data`.
    """
    _log(f"loading {raw_csv}")
    raw = pd.read_csv(raw_csv, low_memory=False)
    _log(f"raw shape: {raw.shape}")

    # 1) Cleaning + outlier handling. We keep Customer_ID for the next step.
    df = clean_data(raw, outlier_mode=outlier_mode, keep_customer_id=True)
    # 2) Per-customer forward/backward fill (panel-structure aware).
    df = group_impute(df, group_col="Customer_ID")
    df = df.drop(columns=["Customer_ID"])
    # 3) Multi-label binarize the Type_of_Loan column.
    df, _ = encode_type_of_loan(df, col="Type_of_Loan")
    # 4) Append the four engineered ratio features.
    df = add_engineered_features(df)
    _log(f"prepared shape: {df.shape}")
    return df


def _run_eda(raw_df: pd.DataFrame, out_dir: Path, fig_dir: Path) -> None:
    """Generate Section 2 of the report: 8 EDA tables + 6 figures.

    Tables are written to ``out_dir/eda/`` and figures to
    ``fig_dir/eda/``. Cleaning is applied inside the plotting functions
    where appropriate so that figures show realistic ranges.
    """
    _log("EDA: writing tables")
    eda_dir = out_dir / "eda"
    written = write_eda_reports(raw_df, eda_dir)
    for fname in written:
        _log(f"  wrote {Path(fname).name}")

    _log("EDA: writing figures")
    eda_fig_dir = fig_dir / "eda"
    # Figure 1..6 of the report. Naming matches the citations in the
    # docx so the reviewer can map images to text trivially.
    plot_target_distribution(raw_df, save_path=eda_fig_dir / "fig01_target_distribution.png")
    plot_numeric_histograms(raw_df, save_path=eda_fig_dir / "fig02_numeric_histograms.png")
    plot_boxplots_before_after(raw_df, save_path=eda_fig_dir / "fig03_boxplot_before_after.png")
    plot_correlation_matrix(raw_df, save_path=eda_fig_dir / "fig04_correlation_matrix.png")
    plot_categorical_distributions(raw_df, save_path=eda_fig_dir / "fig05_categorical_distributions.png")
    plot_boxplot_by_target(raw_df, save_path=eda_fig_dir / "fig06_boxplot_by_target.png")
    _log(f"  EDA figures saved under {eda_fig_dir}")


def main(args: argparse.Namespace) -> None:
    """Run all enabled stages in order. Called from ``__main__``."""
    raw_csv = Path(args.train_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Figures live in a separate subdir to keep the table CSVs uncluttered.
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --sample-rows 0 means "use the full dataset"; anything positive
    # is treated as a row count to subsample for fast iteration.
    sample_n = args.sample_rows if args.sample_rows > 0 else None
    if sample_n:
        _log(f"sampling {sample_n} rows for fast mode")

    # Load raw CSV once; downstream stages reuse this frame.
    raw_df = pd.read_csv(raw_csv, low_memory=False)
    if sample_n:
        # ``random_state=42`` pinned so the subsample is reproducible.
        raw_df = raw_df.sample(n=sample_n, random_state=42).reset_index(drop=True)

    # ---- Stage 1: EDA tables + figures ------------------------------
    if args.run_eda:
        _run_eda(raw_df, out_dir, fig_dir)

    # ---- Stage 2: outlier-handling ablation -------------------------
    # Note: this ablation needs the raw frame (each mode triggers a
    # fresh cleaning chain inside :func:`ablation_outlier`).
    if args.run_outlier_ablation:
        _log("ablation: outlier handling")
        outlier_df = ablation_outlier(
            raw_df,
            models_subset=args.ablation_models,
            cv=args.cv_folds,
        )
        outlier_df.to_csv(out_dir / "ablation_outlier.csv", index=False)
        _log("\n" + outlier_df.to_string(index=False))
        plot_ablation_outlier(outlier_df, save_path=fig_dir / "ablation_outlier.png")

    df = prepare_dataset(raw_csv, outlier_mode="domain")
    if sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

    # 3. ablation - class imbalance handling
    if args.run_imbalance_ablation:
        _log("ablation: class imbalance handling")
        imb_df = ablation_imbalance(
            df,
            models_subset=args.ablation_models,
            cv=args.cv_folds,
        )
        imb_df.to_csv(out_dir / "ablation_imbalance.csv", index=False)
        _log("\n" + imb_df.to_string(index=False))
        plot_ablation_imbalance(imb_df, save_path=fig_dir / "ablation_imbalance.png")

    # 3b. Open-source SW contribution: single top-level full sweep
    #     (scaler × encoder × model × hyperparameter) ranked top-5 + best.
    if args.run_sweep:
        _log("full sweep: scaler x encoder x model x params")
        board, top5, best = run_full_sweep(
            df,
            cv=args.sweep_cv,
            sample_size=args.sweep_sample,
            top_k=5,
            verbose=True,
        )
        paths = save_sweep_artifacts(board, top5, best, out_dir)
        for name, p in paths.items():
            _log(f"  wrote {p.name}")
        _log("\nTop 5 configurations:\n" + top5.to_string(index=False))
        _log(
            f"BEST: {best.get('model')} | {best.get('scaler')}+{best.get('encoder')} "
            f"| {best.get('params')} | macro_f1={best.get('macro_f1'):.4f}"
        )
        plot_full_sweep_top5(top5, save_path=fig_dir / "full_sweep_top5.png")

    # ---- Stage 4: scaler x encoder ablation -------------------------
    if args.run_preproc_ablation:
        _log("ablation: scaler x encoder")
        prep_df = ablation_preprocessing(
            df, model_name=args.preproc_model, cv=args.cv_folds
        )
        prep_df.to_csv(out_dir / "ablation_preprocessing.csv", index=False)
        _log("\n" + prep_df.to_string(index=False))
        plot_ablation_preprocessing(prep_df, save_path=fig_dir / "ablation_preprocessing.png")

    # ---- Stage 5: Stratified k-fold baseline comparison --------------
    # Section 5.1 of the report. Same RobustScaler + OneHotEncoder
    # preprocessor as the ablations so the table is directly comparable.
    _log("stratified k-fold across base models")
    X, y, num_feat, cat_feat = split_features(df)
    preprocessor = build_preprocessor(num_feat, cat_feat, scaler="robust", encoder="onehot")
    # class_weight='balanced' matches the ablation 2 'class_weight' cell.
    base_models = get_models(class_weight="balanced")

    cv_rows = []
    for name, model in base_models.items():
        pipeline = build_pipeline(preprocessor, model)
        scores = evaluate_cv(pipeline, X, y, cv=args.cv_folds)
        cv_rows.append(
            {
                "model": name,
                "macro_f1_mean": scores["macro_f1"]["mean"],
                "macro_f1_std": scores["macro_f1"]["std"],
                "macro_recall_mean": scores["macro_recall"]["mean"],
                "accuracy_mean": scores["accuracy"]["mean"],
            }
        )
        _log(
            f"  {name}: f1={scores['macro_f1']['mean']:.4f}"
            f" (+/- {scores['macro_f1']['std']:.4f})"
        )
    cv_df = pd.DataFrame(cv_rows).sort_values("macro_f1_mean", ascending=False)
    cv_df.to_csv(out_dir / "cv_baseline.csv", index=False)
    _log("\n" + cv_df.to_string(index=False))
    plot_cv_baseline(cv_df, save_path=fig_dir / "cv_baseline.png")

    # ---- Stage 6: GridSearchCV on the CV winner ---------------------
    # Best model from Stage 5 is the row 0 of the sorted CV table.
    best_name = cv_df.iloc[0]["model"]
    _log(f"GridSearchCV on best model: {best_name}")
    best_model = base_models[best_name]
    best_pipeline = build_pipeline(preprocessor, best_model)
    grid = get_param_grids()[best_name]

    if args.skip_grid:
        # Smoke-test mode: skip the expensive grid search.
        _log("--skip-grid: using untuned pipeline")
        tuned_pipeline = best_pipeline
        best_params = None
    else:
        grid_obj = tune_with_gridsearch(
            best_pipeline, grid, X, y, cv=args.cv_folds, scoring="f1_macro"
        )
        # refit=True inside tune_with_gridsearch ensures the best
        # estimator is already refitted on the full ``(X, y)`` here.
        tuned_pipeline = grid_obj.best_estimator_
        best_params = grid_obj.best_params_
        _log(f"best params: {best_params}")
        _log(f"best CV macro_f1: {grid_obj.best_score_:.4f}")

    # ---- Stage 7: Final 80/20 hold-out evaluation -------------------
    # ``stratify=y`` preserves the 53/29/18 class ratio in both splits;
    # ``random_state=42`` makes the split reproducible.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    holdout = evaluate_holdout(tuned_pipeline, X_train, X_valid, y_train, y_valid)
    _log(
        "final hold-out: "
        f"acc={holdout['accuracy']:.4f}  "
        f"macro_f1={holdout['macro_f1']:.4f}  "
        f"macro_recall={holdout['macro_recall']:.4f}"
    )

    plot_confusion_matrix(
        y_valid,
        holdout["y_pred"],
        labels=sorted(y.unique()),
        title=f"Confusion Matrix - {best_name}",
        save_path=fig_dir / "confusion_matrix.png",
    )
    plot_per_class_metrics(
        holdout["report"], save_path=fig_dir / "per_class_metrics.png"
    )

    final_summary = {
        "best_model": best_name,
        "best_params": best_params,
        "holdout": {
            "accuracy": holdout["accuracy"],
            "macro_f1": holdout["macro_f1"],
            "macro_recall": holdout["macro_recall"],
            "report": holdout["report"],
        },
    }
    with open(out_dir / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)

    # ---- Stage 8: auxiliary tasks (Section 6 of the report) --------
    # Both are wrapped in try/except so a failure here does not undo the
    # successful classification artifacts already on disk.
    if args.run_auxiliary:
        _log("auxiliary: regression on Monthly_Balance")
        try:
            reg = regress_monthly_balance(df)
            _log(json.dumps(reg, indent=2))
            with open(out_dir / "aux_regression.json", "w", encoding="utf-8") as f:
                json.dump(reg, f, indent=2)
        except Exception as exc:
            _log(f"regression failed: {exc}")

        _log("auxiliary: KMeans on payment behaviour")
        try:
            clust = cluster_payment_behaviour(df, n_clusters=args.n_clusters)
            _log(
                f"silhouette={clust['silhouette']:.4f}  "
                f"sizes={clust['cluster_sizes']}"
            )
            clust["centroids"].to_csv(
                out_dir / "aux_kmeans_centroids.csv", index=False
            )
            with open(out_dir / "aux_kmeans_summary.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "n_clusters": clust["n_clusters"],
                        "silhouette": clust["silhouette"],
                        "cluster_sizes": clust["cluster_sizes"],
                        "features_used": clust["features_used"],
                    },
                    f,
                    indent=2,
                )

            elbow_df = kmeans_elbow(df)
            elbow_df.to_csv(out_dir / "aux_kmeans_elbow.csv", index=False)
            plot_kmeans_elbow(elbow_df, save_path=fig_dir / "aux_kmeans_elbow.png")
        except Exception as exc:
            _log(f"clustering failed: {exc}")

    # ---- Stage 9: optionally persist the fitted final pipeline -----
    # joblib dump of a tuned 400-tree RandomForest can exceed 1 GB,
    # so it is opt-in to keep the repo light by default.
    if args.save_pipeline:
        joblib.dump(tuned_pipeline, out_dir / "final_pipeline.joblib")
        _log(f"saved final_pipeline.joblib (size may be large)")
    else:
        _log("skipping final_pipeline.joblib (pass --save-pipeline to keep it)")
    _log(f"all artifacts under {out_dir}")


def parse_args() -> argparse.Namespace:
    """Define and parse the command-line interface.

    Returns
    -------
    argparse.Namespace
        ``args`` object consumed by :func:`main`. The ``--all`` flag
        flips every stage switch on at once.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", default=str(ROOT / "data" / "raw" / "train.csv"))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument(
        "--sample-rows",
        type=int,
        default=0,
        help="if > 0, randomly subsample this many rows for fast experiments",
    )
    p.add_argument("--skip-grid", action="store_true")
    p.add_argument("--run-eda", action="store_true",
                   help="produce EDA tables and figures from the raw frame")
    p.add_argument("--run-outlier-ablation", action="store_true")
    p.add_argument("--run-imbalance-ablation", action="store_true")
    p.add_argument("--run-preproc-ablation", action="store_true")
    p.add_argument("--run-auxiliary", action="store_true")
    p.add_argument(
        "--run-sweep",
        action="store_true",
        help="run the single-top-level full sweep (scaler x encoder x model x params)",
    )
    p.add_argument(
        "--sweep-cv",
        type=int,
        default=3,
        help="cv folds for the full sweep (default 3 for tractability)",
    )
    p.add_argument(
        "--sweep-sample",
        type=int,
        default=20000,
        help="subsample rows for the full sweep; 0 = full dataset",
    )
    p.add_argument(
        "--save-pipeline",
        action="store_true",
        help="persist the fitted final pipeline (large file; opt-in)",
    )
    p.add_argument(
        "--ablation-models",
        nargs="*",
        default=["DecisionTree", "RandomForest"],
        help="subset of model names to evaluate in ablations to keep runtime tractable",
    )
    p.add_argument(
        "--preproc-model",
        default="DecisionTree",
        help="model to use for scaler x encoder sweep",
    )
    p.add_argument("--n-clusters", type=int, default=4)
    p.add_argument(
        "--all",
        action="store_true",
        help="enable EDA, all ablations, and auxiliary tasks",
    )
    args = p.parse_args()
    if args.all:
        args.run_eda = True
        args.run_outlier_ablation = True
        args.run_imbalance_ablation = True
        args.run_preproc_ablation = True
        args.run_auxiliary = True
        args.run_sweep = True
    # convert 0 → None for sweep_sample (means full dataset)
    if args.sweep_sample == 0:
        args.sweep_sample = None
    return args


if __name__ == "__main__":
    main(parse_args())
