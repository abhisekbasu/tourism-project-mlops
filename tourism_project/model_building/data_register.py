import os
from huggingface_hub import HfApi, create_repo

def register_data():
    """
    Registers raw data on Hugging Face Dataset Hub.
    Expects:
      HF_DATASET_REPO : e.g., "username/tourism-package-data"
      LOCAL_CSV_PATH  : e.g., "/content/drive/MyDrive/Colab_Notebooks/tourism_project/data/tourism.csv"
      HF_RAW_FILENAME : optional, default "tourism.csv"
    """
    hf_repo_id = os.environ.get("HF_DATASET_REPO")
    local_csv_path = os.environ.get("LOCAL_CSV_PATH")
    hf_filename = os.environ.get("HF_RAW_FILENAME", "tourism.csv")

    if not hf_repo_id:
        raise ValueError("HF_DATASET_REPO is missing. Example: 'username/tourism-package-data'")
    if not local_csv_path:
        raise ValueError("LOCAL_CSV_PATH is missing. Example: '/content/drive/MyDrive/Colab_Notebooks/tourism_project/data/tourism.csv'")
    if not os.path.exists(local_csv_path):
        raise FileNotFoundError(f"CSV file not found at: {local_csv_path}")

    # Create dataset repo (if it already exists, exist_ok=True prevents failure)
    create_repo(repo_id=hf_repo_id, repo_type="dataset", exist_ok=True)

    # Upload raw CSV to HF dataset repo
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_csv_path,
        path_in_repo=hf_filename,
        repo_id=hf_repo_id,
        repo_type="dataset"
    )

    print(f" Data registered successfully to HF dataset repo: {hf_repo_id} (file: {hf_filename})")

if __name__ == "__main__":
    register_data()
