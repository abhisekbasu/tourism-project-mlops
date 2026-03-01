import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, create_repo

def upload_to_hf_dataset(repo_id: str, local_path: str, path_in_repo: str) -> None:
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset"
    )

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Drop obvious non-predictive / index-like columns if present
    drop_cols = [c for c in ["CustomerID", "Unnamed: 0"] if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)

    # Basic string cleanup for object columns (safe, non-destructive)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    return df

def main():
    # Required env vars (friendly for CI/CD)
    hf_repo_id = os.environ.get("HF_DATASET_REPO")
    local_csv_path = os.environ.get("LOCAL_CSV_PATH")  # repo path in CI/CD will be like: "data/tourism.csv"

    if not hf_repo_id:
        raise ValueError("Missing HF_DATASET_REPO. Example: 'abhisekbasu/tourism-package-data'")
    if not local_csv_path:
        raise ValueError("Missing LOCAL_CSV_PATH. Example: 'data/tourism.csv' or '/content/.../data/tourism.csv'")
    if not os.path.exists(local_csv_path):
        raise FileNotFoundError(f"Raw CSV not found at: {local_csv_path}")

    target_col = os.environ.get("TARGET_COL", "ProdTaken")
    test_size = float(os.environ.get("TEST_SIZE", "0.2"))
    random_state = int(os.environ.get("RANDOM_STATE", "42"))

    # Output paths (local)
    output_dir = os.environ.get("OUTPUT_DIR", "data/processed")
    os.makedirs(output_dir, exist_ok=True)

    # 1) Upload raw data to HF (Option 2 requirement)
    upload_to_hf_dataset(hf_repo_id, local_csv_path, "raw/tourism.csv")
    print("Uploaded raw data to HF:", f"{hf_repo_id}/raw/tourism.csv")

    # 2) Load + clean
    df = pd.read_csv(local_csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    df_clean = clean_data(df)

    # 3) Split (stratified to preserve target distribution)
    y = df_clean[target_col]
    X = df_clean.drop(columns=[target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Saved locally:", train_path, test_path)

    # 4) Upload train/test back to HF dataset repo
    upload_to_hf_dataset(hf_repo_id, train_path, "processed/train.csv")
    upload_to_hf_dataset(hf_repo_id, test_path, "processed/test.csv")

    print("Uploaded processed splits to HF:",
          f"{hf_repo_id}/processed/train.csv and {hf_repo_id}/processed/test.csv")

if __name__ == "__main__":
    main()
