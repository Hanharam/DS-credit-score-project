"""End-to-end training script.

Stages (each is independently togglable):
1. Load raw train.csv
2. EDA tables + figures (--run-eda)
3. Clean + group-aware imputation + Type_of_Loan MLB + feature engineering
4. Ablations: outlier handling, class imbalance, scaler x encoder
5. Stratified k-fold evaluation across all base models
6. GridSearchCV tuning on the best base model
7. Final hold-out evaluation + confusion matrix + per-class metrics
8. Auxiliary tasks: regression on Monthly_Balance + KMeans on payment behaviour
9. Save artifacts (CSV reports, PNG figures, optionally fitted model)
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
    plot_kmeans_elbow,
    plot_numeric_histograms,
    plot_per_class_metrics,
    plot_target_distribution,
)


DEFAULT_OUT = ROOT / "reports"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare_dataset(raw_csv: Path, outlier_mode: str = "domain") -> pd.DataFrame:
    _log(f"loading {raw_csv}")
    raw = pd.read_csv(raw_csv, low_memory=False)
    _log(f"raw shape: {raw.shape}")

    df = clean_data(raw, outlier_mode=outlier_mode, keep_customer_id=True)
    df = group_impute(df, group_col="Customer_ID")
    df = df.drop(columns=["Customer_ID"])
    df, _ = encode_type_of_loan(df, col="Type_of_Loan")
    df = add_engineered_features(df)
    _log(f"prepared shape: {df.shape}")
    return df


def _run_eda(raw_df: pd.DataFrame, out_dir: Path, fig_dir: Path) -> None:
    _log("EDA: writing tables")
    eda_dir = out_dir / "eda"
    written = write_eda_reports(raw_df, eda_dir)
    for fname in written:
        _log(f"  wrote {Path(fname).name}")

    _log("EDA: writing figures")
    eda_fig_dir = fig_dir / "eda"
    plot_target_distribution(raw_df, save_path=eda_fig_dir / "fig01_target_distribution.png")
    plot_numeric_histograms(raw_df, save_path=eda_fig_dir / "fig02_numeric_histograms.png")
    plot_boxplots_before_after(raw_df, save_path=eda_fig_dir / "fig03_boxplot_before_after.png")
    plot_correlation_matrix(raw_df, save_path=eda_fig_dir / "fig04_correlation_matrix.png")
    plot_categorical_distributions(raw_df, save_path=eda_fig_dir / "fig05_categorical_distributions.png")
    plot_boxplot_by_target(raw_df, save_path=eda_fig_dir / "fig06_boxplot_by_target.png")
    _log(f"  EDA figures saved under {eda_fig_dir}")


def main(args: argparse.Namespace) -> None:
    raw_csv = Path(args.train_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sample_n = args.sample_rows if args.sample_rows > 0 else None
    if sample_n:
        _log(f"sampling {sample_n} rows for fast mode")

    raw_df = pd.read_csv(raw_csv, low_memory=False)
    if sample_n:
        raw_df = raw_df.sample(n=sample_n, random_state=42).reset_index(drop=True)

    # 1. EDA
    if args.run_eda:
        _run_eda(raw_df, out_dir, fig_dir)

    # 2. ablation - outlier handling (operates on raw)
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

    # 4. ablation - scaler x encoder
    if args.run_preproc_ablation:
        _log("ablation: scaler x encoder")
        prep_df = ablation_preprocessing(
            df, model_name=args.preproc_model, cv=args.cv_folds
        )
        prep_df.to_csv(out_dir / "ablation_preprocessing.csv", index=False)
        _log("\n" + prep_df.to_string(index=False))
        plot_ablation_preprocessing(prep_df, save_path=fig_dir / "ablation_preprocessing.png")

    # 5. Stratified k-fold baseline comparison
    _log("stratified k-fold across base models")
    X, y, num_feat, cat_feat = split_features(df)
    preprocessor = build_preprocessor(num_feat, cat_feat, scaler="robust", encoder="onehot")
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

    # 6. tune best model with GridSearchCV
    best_name = cv_df.iloc[0]["model"]
    _log(f"GridSearchCV on best model: {best_name}")
    best_model = base_models[best_name]
    best_pipeline = build_pipeline(preprocessor, best_model)
    grid = get_param_grids()[best_name]

    if args.skip_grid:
        _log("--skip-grid: using untuned pipeline")
        tuned_pipeline = best_pipeline
        best_params = None
    else:
        grid_obj = tune_with_gridsearch(
            best_pipeline, grid, X, y, cv=args.cv_folds, scoring="f1_macro"
        )
        tuned_pipeline = grid_obj.best_estimator_
        best_params = grid_obj.best_params_
        _log(f"best params: {best_params}")
        _log(f"best CV macro_f1: {grid_obj.best_score_:.4f}")

    # 7. Final hold-out evaluation
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

    # 8. auxiliary tasks
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

    # 9. persist final pipeline (opt-in; the artifact can be >1 GB for RF)
    if args.save_pipeline:
        joblib.dump(tuned_pipeline, out_dir / "final_pipeline.joblib")
        _log(f"saved final_pipeline.joblib (size may be large)")
    else:
        _log("skipping final_pipeline.joblib (pass --save-pipeline to keep it)")
    _log(f"all artifacts under {out_dir}")


def parse_args() -> argparse.Namespace:
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
    return args


if __name__ == "__main__":
    main(parse_args())
