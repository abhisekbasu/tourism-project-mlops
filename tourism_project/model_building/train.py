# for data manipulation
import os
import json
import pandas as pd

# for data preprocessing and pipeline creation
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# for model serialization
import joblib

# for hugging face model hub upload
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# for experimentation tracking
import mlflow


def main():
    # -------------------------
    # ENV VARS (set in Colab now; later in GitHub Actions)
    # -------------------------
    HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO")     # abhisekbasu/tourism-project-data
    HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO")         # abhisekbasu/tourism-project-model
    TARGET_COL = os.environ.get("TARGET_COL", "ProdTaken")

    TRAIN_PATH_IN_REPO = os.environ.get("TRAIN_PATH_IN_REPO", "processed/train.csv")
    TEST_PATH_IN_REPO  = os.environ.get("TEST_PATH_IN_REPO",  "processed/test.csv")

    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "MLOps_experiment")

    CLASSIFICATION_THRESHOLD = float(os.environ.get("CLASSIFICATION_THRESHOLD", "0.45"))

    if not HF_DATASET_REPO:
        raise ValueError("Missing HF_DATASET_REPO (example: 'abhisekbasu/tourism-project-data')")
    if not HF_MODEL_REPO:
        raise ValueError("Missing HF_MODEL_REPO (example: 'abhisekbasu/tourism-project-model')")

    # -------------------------
    # MLflow config (env-driven)
    # -------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # -------------------------
    # Load train/test from HF Dataset repo (hf:// style like sample)
    # -------------------------
    Xtrain_path = f"hf://datasets/{HF_DATASET_REPO}/{TRAIN_PATH_IN_REPO}"
    Xtest_path  = f"hf://datasets/{HF_DATASET_REPO}/{TEST_PATH_IN_REPO}"

    train_df = pd.read_csv(Xtrain_path)
    test_df  = pd.read_csv(Xtest_path)

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"Target '{TARGET_COL}' not found. Columns: {list(train_df.columns)}")

    Xtrain = train_df.drop(columns=[TARGET_COL])
    ytrain = train_df[TARGET_COL].astype(int)

    Xtest = test_df.drop(columns=[TARGET_COL])
    ytest = test_df[TARGET_COL].astype(int)

    # -------------------------
    # One-hot encode categorical + scale numeric (explicit lists like sample)
    # -------------------------
    numeric_features = [
        "Age",
        "CityTier",
        "DurationOfPitch",
        "NumberOfPersonVisiting",
        "NumberOfFollowups",
        "PreferredPropertyStar",
        "NumberOfTrips",
        "Passport",
        "PitchSatisfactionScore",
        "OwnCar",
        "NumberOfChildrenVisiting",
        "MonthlyIncome",
    ]
    categorical_features = [
        "TypeofContact",
        "Occupation",
        "Gender",
        "ProductPitched",
        "MaritalStatus",
        "Designation",
    ]

    # Optional safety checks (helps avoid silent failures in CI/CD)
    missing_num = [c for c in numeric_features if c not in Xtrain.columns]
    missing_cat = [c for c in categorical_features if c not in Xtrain.columns]
    if missing_num or missing_cat:
        raise ValueError(f"Feature mismatch. Missing numeric={missing_num}, missing categorical={missing_cat}")

    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features)
    )

    # -------------------------
    # Handle class imbalance (same idea as sample)
    # -------------------------
    class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

    # -------------------------
    # Define base XGBoost model + hyperparameter grid (sample style)
    # -------------------------
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=class_weight,
        random_state=42,
        eval_metric="logloss"
    )

    param_grid = {
        "xgbclassifier__n_estimators": [80, 120],
        "xgbclassifier__max_depth": [3, 4],
        "xgbclassifier__colsample_bytree": [0.5, 0.7],
        "xgbclassifier__learning_rate": [0.05, 0.1],
        "xgbclassifier__reg_lambda": [0.5, 1.0],
    }

    model_pipeline = make_pipeline(preprocessor, xgb_model)

    # -------------------------
    # Train + log ALL tuned parameters (nested runs like sample)
    # -------------------------
    with mlflow.start_run(run_name="tourism_prod_gridsearch"):
        grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(Xtrain, ytrain)

        results = grid_search.cv_results_
        for i in range(len(results["params"])):
            param_set = results["params"][i]
            mean_score = results["mean_test_score"][i]
            std_score = results["std_test_score"][i]

            with mlflow.start_run(nested=True):
                mlflow.log_params(param_set)
                mlflow.log_metric("mean_test_score", float(mean_score))
                mlflow.log_metric("std_test_score", float(std_score))

        # Best params in parent run
        mlflow.log_params(grid_search.best_params_)

        best_model = grid_search.best_estimator_

        # Evaluate best model with fixed threshold
        y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
        y_pred_train = (y_pred_train_proba >= CLASSIFICATION_THRESHOLD).astype(int)

        y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
        y_pred_test = (y_pred_test_proba >= CLASSIFICATION_THRESHOLD).astype(int)

        train_report = classification_report(ytrain, y_pred_train, output_dict=True)
        test_report = classification_report(ytest, y_pred_test, output_dict=True)

        mlflow.log_metrics({
            "train_accuracy": train_report["accuracy"],
            "train_precision": train_report["1"]["precision"],
            "train_recall": train_report["1"]["recall"],
            "train_f1": train_report["1"]["f1-score"],
            "test_accuracy": test_report["accuracy"],
            "test_precision": test_report["1"]["precision"],
            "test_recall": test_report["1"]["recall"],
            "test_f1": test_report["1"]["f1-score"],
        })

        # -------------------------
        # Save model locally + log artifacts
        # -------------------------
        os.makedirs("artifacts", exist_ok=True)

        model_path = "artifacts/best_tourism_model_v1.joblib"
        joblib.dump(best_model, model_path)

        summary = {
            "hf_dataset_repo": HF_DATASET_REPO,
            "hf_model_repo": HF_MODEL_REPO,
            "target_col": TARGET_COL,
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "best_params": grid_search.best_params_,
            "metrics": {
                "train_accuracy": train_report["accuracy"],
                "train_precision": train_report["1"]["precision"],
                "train_recall": train_report["1"]["recall"],
                "train_f1": train_report["1"]["f1-score"],
                "test_accuracy": test_report["accuracy"],
                "test_precision": test_report["1"]["precision"],
                "test_recall": test_report["1"]["recall"],
                "test_f1": test_report["1"]["f1-score"],
            }
        }
        summary_path = "artifacts/model_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact(summary_path, artifact_path="model")

        # -------------------------
        # Upload best model to HF Model Hub
        # -------------------------
        api = HfApi()
        try:
            api.repo_info(repo_id=HF_MODEL_REPO, repo_type="model")
        except RepositoryNotFoundError:
            create_repo(repo_id=HF_MODEL_REPO, repo_type="model", private=False)

        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="best_tourism_model_v1.joblib",
            repo_id=HF_MODEL_REPO,
            repo_type="model",
        )
        api.upload_file(
            path_or_fileobj=summary_path,
            path_in_repo="model_summary.json",
            repo_id=HF_MODEL_REPO,
            repo_type="model",
        )

        print("✅ Best model uploaded to HF Model Hub:", HF_MODEL_REPO)


if __name__ == "__main__":
    main()
