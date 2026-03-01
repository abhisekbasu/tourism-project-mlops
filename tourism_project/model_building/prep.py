import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, create_repo

# Defining the set of columns we expect in the dataset for schema validation
EXPECTED_COLS = {
    "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
}

def upload_to_hf_dataset(repo_id, local_path, path_in_repo):
    # Uploading a local file to the Hugging Face Dataset repository
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  Uploaded to hf://datasets/{repo_id}/{path_in_repo}")

def clean_data(df):
    # Dropping non-predictive columns like CustomerID and index columns
    # Imputation and encoding are handled inside the sklearn pipeline in train.py
    # to prevent any data leakage between train and test sets
    df = df.copy()
    drop_cols = [c for c in ["CustomerID", "Unnamed: 0"] if c in df.columns]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        print(f"  Dropped columns: {drop_cols}")
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df

def validate_schema(df, target_col):
    # Checking all expected columns are present before proceeding
    missing = (EXPECTED_COLS | {target_col}) - set(df.columns)
    if missing:
        raise ValueError(f"Schema check failed — missing columns: {missing}")
    print(f"  Schema check passed ({len(df.columns)} columns, {len(df)} rows)")

def log_dataset_stats(df, target_col):
    # Printing basic statistics to confirm the data looks correct
    pos_rate = df[target_col].value_counts().get(1, 0) / len(df) * 100
    print(f"  Rows: {len(df)} | Positive rate (ProdTaken=1): {pos_rate:.1f}%")
    print(f"  Missing values total: {df.isnull().sum().sum()}")

def main():
    # Reading all configuration from environment variables
    # This makes the script work the same way in Colab and GitHub Actions
    hf_repo_id     = os.environ.get("HF_DATASET_REPO")
    local_csv_path = os.environ.get("LOCAL_CSV_PATH")
    target_col     = os.environ.get("TARGET_COL",    "ProdTaken")
    test_size      = float(os.environ.get("TEST_SIZE",   "0.2"))
    random_state   = int(os.environ.get("RANDOM_STATE",  "42"))
    output_dir     = os.environ.get("OUTPUT_DIR",    "data/processed")

    if not hf_repo_id:
        raise ValueError("HF_DATASET_REPO environment variable is not set.")
    if not local_csv_path or not os.path.exists(local_csv_path):
        raise FileNotFoundError(f"Raw CSV not found at: {local_csv_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Uploading raw CSV to HF Dataset repo
    print("\n[Step 1] Uploading raw CSV to Hugging Face ...")
    upload_to_hf_dataset(hf_repo_id, local_csv_path, "raw/tourism.csv")

    # Step 2: Loading the dataset directly from Hugging Face
    print("\n[Step 2] Loading raw CSV from Hugging Face ...")
    df = pd.read_csv(f"hf://datasets/{hf_repo_id}/raw/tourism.csv")
    print(f"  Loaded {df.shape[0]} rows and {df.shape[1]} columns")

    # Step 3: Validating schema
    print("\n[Step 3] Validating schema ...")
    validate_schema(df, target_col)

    # Step 4: Cleaning the data
    print("\n[Step 4] Cleaning data ...")
    df_clean = clean_data(df)
    log_dataset_stats(df_clean, target_col)

    # Step 5: Splitting into train and test sets with stratification
    print("\n[Step 5] Splitting into train and test sets (80/20, stratified) ...")
    y = df_clean[target_col]
    X = df_clean.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df  = pd.concat([X_test,  y_test],  axis=1)
    print(f"  Train set: {len(train_df)} rows | Test set: {len(test_df)} rows")

    # Step 6: Saving splits locally
    train_path = os.path.join(output_dir, "train.csv")
    test_path  = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)
    print(f"  Saved locally to {output_dir}/")

    # Step 7: Uploading processed splits back to HF
    print("\n[Step 6] Uploading processed train and test splits to Hugging Face ...")
    upload_to_hf_dataset(hf_repo_id, train_path, "processed/train.csv")
    upload_to_hf_dataset(hf_repo_id, test_path,  "processed/test.csv")

    print("\n[prep.py] All steps completed successfully.")

if __name__ == "__main__":
    main()
