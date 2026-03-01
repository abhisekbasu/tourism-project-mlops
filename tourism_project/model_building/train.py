import os
import json
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from huggingface_hub import HfApi, create_repo
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb


def main():
    # Reading all configuration from environment variables
    hf_dataset_repo   = os.environ.get("HF_DATASET_REPO")
    hf_model_repo     = os.environ.get("HF_MODEL_REPO")
    target_col        = os.environ.get("TARGET_COL",             "ProdTaken")
    train_path        = os.environ.get("TRAIN_PATH_IN_REPO",     "processed/train.csv")
    test_path         = os.environ.get("TEST_PATH_IN_REPO",      "processed/test.csv")
    mlflow_uri        = os.environ.get("MLFLOW_TRACKING_URI",    "file:./mlruns")
    mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT",      "MLOps_experiment")
    threshold         = float(os.environ.get("CLASSIFICATION_THRESHOLD", "0.45"))

    if not hf_dataset_repo:
        raise ValueError("HF_DATASET_REPO environment variable is not set.")
    if not hf_model_repo:
        raise ValueError("HF_MODEL_REPO environment variable is not set.")

    # Configuring MLflow to track this experiment
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_experiment)

    # Step 1: Loading train and test data directly from Hugging Face
    print("[Step 1] Loading train and test splits from Hugging Face ...")
    train_df = pd.read_csv(f"hf://datasets/{hf_dataset_repo}/{train_path}")
    test_df  = pd.read_csv(f"hf://datasets/{hf_dataset_repo}/{test_path}")
    print(f"  Train: {train_df.shape} | Test: {test_df.shape}")

    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data.")

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test  = test_df.drop(columns=[target_col])
    y_test  = test_df[target_col]

    # Step 2: Building the preprocessing pipeline
    # Numeric columns: fill missing with median, then scale
    # Categorical columns: fill missing with most frequent value, then one-hot encode
    # This is done inside the pipeline to prevent data leakage
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
        (numeric_pipe,     num_cols),
        (categorical_pipe, cat_cols),
        remainder="drop",
    )

    # Defining the XGBoost classifier and combining it with the preprocessor
    model    = xgb.XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)
    pipeline = make_pipeline(preprocessor, model)

    # Step 3: Defining the hyperparameter grid for tuning
    # 2 values per parameter x 6 parameters = 64 combinations x 5-fold CV = 320 fits
    param_grid = {
        "xgbclassifier__n_estimators":    [80, 120],
        "xgbclassifier__max_depth":        [3, 4],
        "xgbclassifier__learning_rate":    [0.05, 0.1],
        "xgbclassifier__subsample":        [0.8, 1.0],
        "xgbclassifier__colsample_bytree": [0.7, 1.0],
        "xgbclassifier__reg_lambda":       [0.5, 1.0],
    }

    # Step 4: Running GridSearchCV and logging every combination to MLflow
    print(f"\n[Step 2] Running GridSearchCV (64 combinations x 5-fold CV) ...")
    with mlflow.start_run(run_name="tourism_gridsearch") as parent_run:

        grid = GridSearchCV(
            pipeline, param_grid=param_grid,
            cv=5, n_jobs=-1, scoring="f1", refit=True,
        )
        grid.fit(X_train, y_train)

        # Logging every parameter combination as a nested child run
        results = grid.cv_results_
        for i in range(len(results["params"])):
            with mlflow.start_run(nested=True, run_name=f"param_set_{i+1}"):
                mlflow.log_params(results["params"][i])
                mlflow.log_metric("mean_cv_f1", float(results["mean_test_score"][i]))
                mlflow.log_metric("std_cv_f1",  float(results["std_test_score"][i]))

        print(f"  Best CV F1 : {grid.best_score_:.4f}")
        print(f"  Best params: {grid.best_params_}")

        # Logging best parameters and threshold to the parent run
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("classification_threshold", threshold)

        best_model = grid.best_estimator_

        # Step 5: Evaluating the best model using the custom threshold
        def evaluate(X, y, label):
            proba  = best_model.predict_proba(X)[:, 1]
            pred   = (proba >= threshold).astype(int)
            report = classification_report(y, pred, output_dict=True, zero_division=0)
            auc    = roc_auc_score(y, proba)
            cm     = confusion_matrix(y, pred).tolist()
            print(f"\n  [{label}] accuracy={report['accuracy']:.4f}  "
                  f"f1={report['1']['f1-score']:.4f}  roc_auc={auc:.4f}")
            print(f"  Confusion matrix: {cm}")
            return report, auc, cm

        print("\n[Step 3] Evaluating best model on train and test sets ...")
        train_report, train_auc, train_cm = evaluate(X_train, y_train, "Train")
        test_report,  test_auc,  test_cm  = evaluate(X_test,  y_test,  "Test")

        # Logging all evaluation metrics to MLflow
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

        # Saving the model and summary to disk
        model_filename   = "best_tourism_model_v1.joblib"
        summary_filename = "model_summary.json"
        joblib.dump(best_model, model_filename)

        # The threshold is stored here so the deployment app reads it from one place
        summary = {
            "best_params": grid.best_params_,
            "threshold":   threshold,
            "best_cv_f1":  float(grid.best_score_),
            "metrics": {
                "train": {
                    "accuracy":         float(train_report["accuracy"]),
                    "precision_1":      float(train_report["1"]["precision"]),
                    "recall_1":         float(train_report["1"]["recall"]),
                    "f1_1":             float(train_report["1"]["f1-score"]),
                    "roc_auc":          float(train_auc),
                    "confusion_matrix": train_cm,
                },
                "test": {
                    "accuracy":         float(test_report["accuracy"]),
                    "precision_1":      float(test_report["1"]["precision"]),
                    "recall_1":         float(test_report["1"]["recall"]),
                    "f1_1":             float(test_report["1"]["f1-score"]),
                    "roc_auc":          float(test_auc),
                    "confusion_matrix": test_cm,
                },
            },
        }
        with open(summary_filename, "w") as fh:
            json.dump(summary, fh, indent=2)

        # Logging model artifacts to MLflow
        mlflow.log_artifact(model_filename,   artifact_path="model")
        mlflow.log_artifact(summary_filename, artifact_path="model")
        print(f"\n  MLflow run id: {parent_run.info.run_id}")

    # Step 6: Uploading the best model and summary to Hugging Face Model Hub
    print("\n[Step 4] Uploading model artifacts to Hugging Face Model Hub ...")
    create_repo(repo_id=hf_model_repo, repo_type="model", exist_ok=True)
    hf_api = HfApi()
    for fname in [model_filename, summary_filename]:
        hf_api.upload_file(
            path_or_fileobj=fname,
            path_in_repo=fname,
            repo_id=hf_model_repo,
            repo_type="model",
        )
        print(f"  Uploaded {fname}")

    print("\n[train.py] All steps completed successfully.")


if __name__ == "__main__":
    main()
