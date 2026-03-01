"""
prep.py — Data Preparation Script
==================================
Responsibilities:
  1. Upload raw CSV to HF Dataset repo (raw/ subfolder).
  2. Load the raw CSV directly from HF.
  3. Clean the data (drop non-predictive columns, strip strings).
  4. Validate schema and log basic dataset statistics.
  5. Stratified train/test split (80/20, random_state=42).
  6. Save splits locally and upload to HF Dataset repo (processed/ subfolder).

All configuration is injected via environment variables so the script works
identically in Colab and in GitHub Actions CI/CD.
"""
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, create_repo

# ── Expected schema ───────────────────────────────────────────────────────────
EXPECTED_COLS = {
    "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def upload_to_hf_dataset(repo_id: str, local_path: str, path_in_repo: str) -> None:
    """Upload a local file to the HF Dataset repository."""
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  Uploaded → hf://datasets/{repo_id}/{path_in_repo}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw dataframe:
      - Drop non-predictive / index columns (CustomerID, Unnamed: 0).
      - Strip leading/trailing whitespace from string columns.
    Note: imputation and encoding are handled inside the sklearn pipeline
    in train.py so that train/test leakage is impossible.
    """
    df = df.copy()
    drop_cols = [c for c in ["CustomerID", "Unnamed: 0"] if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        print(f"  Dropped columns: {drop_cols}")
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def validate_schema(df: pd.DataFrame, target_col: str) -> None:
    """Raise if expected columns are missing from the dataframe."""
    required = EXPECTED_COLS | {target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Schema validation failed — missing columns: {missing}")
    print(f"  Schema validation passed ({len(df.columns)} columns, {len(df)} rows)")


def log_dataset_stats(df: pd.DataFrame, target_col: str) -> None:
    """Print basic dataset statistics for audit trail."""
    counts = df[target_col].value_counts()
    pos_rate = counts.get(1, 0) / len(df) * 100
    print(f"  Rows: {len(df)} | Positive rate: {pos_rate:.1f}%")
    print(f"  Missing values: {df.isnull().sum().sum()} total")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    hf_repo_id    = os.environ.get("HF_DATASET_REPO")
    local_csv_path = os.environ.get("LOCAL_CSV_PATH")

    if not hf_repo_id:
        raise ValueError("Environment variable HF_DATASET_REPO is not set.")
    if not local_csv_path:
        raise ValueError("Environment variable LOCAL_CSV_PATH is not set.")
    if not os.path.exists(local_csv_path):
        raise FileNotFoundError(f"Raw CSV not found at: {local_csv_path}")

    target_col   = os.environ.get("TARGET_COL",    "ProdTaken")
    test_size    = float(os.environ.get("TEST_SIZE",      "0.2"))
    random_state = int(os.environ.get("RANDOM_STATE",     "42"))
    output_dir   = os.environ.get("OUTPUT_DIR",    "data/processed")
    os.makedirs(output_dir, exist_ok=True)

    # 1) Upload raw CSV to HF
    print("\n[Step 1] Uploading raw CSV to HF Dataset repo ...")
    upload_to_hf_dataset(hf_repo_id, local_csv_path, "raw/tourism.csv")

    # 2) Load directly from HF (proves HF read path works)
    print("\n[Step 2] Loading raw CSV from HF ...")
    hf_raw_path = f"hf://datasets/{hf_repo_id}/raw/tourism.csv"
    df = pd.read_csv(hf_raw_path)
    print(f"  Loaded {df.shape[0]} rows × {df.shape[1]} columns")

    # 3) Validate schema
    print("\n[Step 3] Validating schema ...")
    validate_schema(df, target_col)

    # 4) Clean
    print("\n[Step 4] Cleaning data ...")
    df_clean = clean_data(df)
    log_dataset_stats(df_clean, target_col)

    # 5) Stratified split
    print("\n[Step 5] Splitting data (train/test = 80/20, stratified) ...")
    y = df_clean[target_col]
    X = df_clean.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df  = pd.concat([X_test,  y_test],  axis=1)
    print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    # 6) Save locally
    train_path = os.path.join(output_dir, "train.csv")
    test_path  = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)
    print(f"  Saved locally → {train_path}, {test_path}")

    # 7) Upload processed splits to HF
    print("\n[Step 6] Uploading processed splits to HF Dataset repo ...")
    upload_to_hf_dataset(hf_repo_id, train_path, "processed/train.csv")
    upload_to_hf_dataset(hf_repo_id, test_path,  "processed/test.csv")

    print("\n[prep.py] DONE.")


if __name__ == "__main__":
    main()
