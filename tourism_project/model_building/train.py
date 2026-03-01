"""
train.py — Model Training & Registration Script
================================================
Responsibilities:
  1. Load train/test splits directly from HF Dataset repo.
  2. Build a sklearn Pipeline (imputer → scaler/encoder → XGBoost).
  3. Hyperparameter-tune with GridSearchCV; log EVERY param set as a
     nested MLflow child run.
  4. Log best params + full evaluation metrics (accuracy, precision,
     recall, F1, ROC-AUC) for both train and test sets.
  5. Save model artifact + model_summary.json (includes threshold).
  6. Upload best model and summary to HF Model Hub.
"""
import os
import json
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from huggingface_hub import HfApi, create_repo
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
)
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    hf_dataset_repo     = os.environ.get("HF_DATASET_REPO")
    hf_model_repo       = os.environ.get("HF_MODEL_REPO")
    target_col          = os.environ.get("TARGET_COL",              "ProdTaken")
    train_path_in_repo  = os.environ.get("TRAIN_PATH_IN_REPO",      "processed/train.csv")
    test_path_in_repo   = os.environ.get("TEST_PATH_IN_REPO",       "processed/test.csv")
    mlflow_uri          = os.environ.get("MLFLOW_TRACKING_URI",     "file:./mlruns")
    mlflow_experiment   = os.environ.get("MLFLOW_EXPERIMENT",       "MLOps_experiment")
    threshold           = float(os.environ.get("CLASSIFICATION_THRESHOLD", "0.45"))

    if not hf_dataset_repo:
        raise ValueError("Environment variable HF_DATASET_REPO is not set.")
    if not hf_model_repo:
        raise ValueError("Environment variable HF_MODEL_REPO is not set.")

    # ── Configure MLflow ─────────────────────────────────────────────────────
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_experiment)

    # ── Load data from HF ────────────────────────────────────────────────────
    print("[Step 1] Loading train/test splits from HF ...")
    train_df = pd.read_csv(f"hf://datasets/{hf_dataset_repo}/{train_path_in_repo}")
    test_df  = pd.read_csv(f"hf://datasets/{hf_dataset_repo}/{test_path_in_repo}")
    print(f"  Train: {train_df.shape} | Test: {test_df.shape}")

    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data.")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test  = test_df.drop(columns=[target_col])
    y_test  = test_df[target_col]

    # ── Build preprocessing pipeline ─────────────────────────────────────────
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    numeric_pipe = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    categorical_pipe = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    )
    preprocessor = make_column_transformer(
        (numeric_pipe,    num_cols),
        (categorical_pipe, cat_cols),
        remainder="drop",
    )

    model    = xgb.XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)
    pipeline = make_pipeline(preprocessor, model)

    # ── Hyperparameter grid ───────────────────────────────────────────────────
    param_grid = {
        "xgbclassifier__n_estimators":    [80, 120],
        "xgbclassifier__max_depth":        [3, 4],
        "xgbclassifier__learning_rate":    [0.05, 0.1],
        "xgbclassifier__subsample":        [0.8, 1.0],
        "xgbclassifier__colsample_bytree": [0.7, 1.0],
        "xgbclassifier__reg_lambda":       [0.5, 1.0],
    }

    # ── GridSearchCV + MLflow logging ─────────────────────────────────────────
    print(f"\n[Step 2] Running GridSearchCV ({2**6} combinations × 5-fold CV) ...")
    with mlflow.start_run(run_name="tourism_prod_gridsearch") as parent_run:

        grid = GridSearchCV(
            pipeline, param_grid=param_grid,
            cv=5, n_jobs=-1, scoring="f1", refit=True,
        )
        grid.fit(X_train, y_train)

        # Log every parameter combination as a nested child run
        results = grid.cv_results_
        for i in range(len(results["params"])):
            with mlflow.start_run(nested=True, run_name=f"param_set_{i+1}"):
                mlflow.log_params(results["params"][i])
                mlflow.log_metric("mean_cv_f1",  float(results["mean_test_score"][i]))
                mlflow.log_metric("std_cv_f1",   float(results["std_test_score"][i]))

        print(f"  Best CV F1 : {grid.best_score_:.4f}")
        print(f"  Best params: {grid.best_params_}")

        # Log best params to parent run
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("classification_threshold", threshold)

        best_model = grid.best_estimator_

        # ── Evaluate with custom threshold ───────────────────────────────────
        def evaluate(X, y, label):
            proba = best_model.predict_proba(X)[:, 1]
            pred  = (proba >= threshold).astype(int)
            report = classification_report(y, pred, output_dict=True, zero_division=0)
            auc    = roc_auc_score(y, proba)
            cm     = confusion_matrix(y, pred).tolist()
            print(f"\n  [{label}] accuracy={report['accuracy']:.4f}  "
                  f"f1={report['1']['f1-score']:.4f}  auc={auc:.4f}")
            print(f"  Confusion matrix: {cm}")
            return report, auc, cm

        print("\n[Step 3] Evaluating best model ...")
        train_report, train_auc, train_cm = evaluate(X_train, y_train, "Train")
        test_report,  test_auc,  test_cm  = evaluate(X_test,  y_test,  "Test")

        # Log all evaluation metrics
        mlflow.log_metrics({
            "train_accuracy":    float(train_report["accuracy"]),
            "train_precision_1": float(train_report["1"]["precision"]),
            "train_recall_1":    float(train_report["1"]["recall"]),
            "train_f1_1":        float(train_report["1"]["f1-score"]),
            "train_roc_auc":     float(train_auc),
            "test_accuracy":     float(test_report["accuracy"]),
            "test_precision_1":  float(test_report["1"]["precision"]),
            "test_recall_1":     float(test_report["1"]["recall"]),
            "test_f1_1":         float(test_report["1"]["f1-score"]),
            "test_roc_auc":      float(test_auc),
        })

        # ── Persist model + summary ───────────────────────────────────────────
        model_filename   = "best_tourism_model_v1.joblib"
        summary_filename = "model_summary.json"

        joblib.dump(best_model, model_filename)

        summary = {
            "best_params":  grid.best_params_,
            "threshold":    threshold,          # ← single source of truth
            "best_cv_f1":   float(grid.best_score_),
            "metrics": {
                "train": {
                    "accuracy":    float(train_report["accuracy"]),
                    "precision_1": float(train_report["1"]["precision"]),
                    "recall_1":    float(train_report["1"]["recall"]),
                    "f1_1":        float(train_report["1"]["f1-score"]),
                    "roc_auc":     float(train_auc),
                    "confusion_matrix": train_cm,
                },
                "test": {
                    "accuracy":    float(test_report["accuracy"]),
                    "precision_1": float(test_report["1"]["precision"]),
                    "recall_1":    float(test_report["1"]["recall"]),
                    "f1_1":        float(test_report["1"]["f1-score"]),
                    "roc_auc":     float(test_auc),
                    "confusion_matrix": test_cm,
                },
            },
        }
        with open(summary_filename, "w") as fh:
            json.dump(summary, fh, indent=2)

        mlflow.log_artifact(model_filename,   artifact_path="model")
        mlflow.log_artifact(summary_filename, artifact_path="model")

        print(f"\n  MLflow run id: {parent_run.info.run_id}")

    # ── Upload to HF Model Hub ────────────────────────────────────────────────
    print("\n[Step 4] Uploading model artifacts to HF Model Hub ...")
    create_repo(repo_id=hf_model_repo, repo_type="model", exist_ok=True)
    hf_api = HfApi()
    for fname in [model_filename, summary_filename]:
        hf_api.upload_file(
            path_or_fileobj=fname,
            path_in_repo=fname,
            repo_id=hf_model_repo,
            repo_type="model",
        )
        print(f"  Uploaded {fname} → hf://{hf_model_repo}/{fname}")

    print("\n[train.py] DONE.")


if __name__ == "__main__":
    main()
