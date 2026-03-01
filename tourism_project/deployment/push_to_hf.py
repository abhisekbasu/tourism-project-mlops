import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

def main():
    hf_space_repo = os.environ.get("HF_SPACE_REPO")
    if not hf_space_repo:
        raise ValueError("Missing HF_SPACE_REPO (example: 'abhisekbasu/tourism-project-app')")

    api = HfApi()

    # Create Docker Space if missing
    try:
        api.repo_info(repo_id=hf_space_repo, repo_type="space")
        print(f"✅ Space exists: {hf_space_repo}")
    except RepositoryNotFoundError:
        create_repo(
            repo_id=hf_space_repo,
            repo_type="space",
            space_sdk="docker",
            private=False
        )
        print(f"✅ Created Docker Space: {hf_space_repo}")

    # Upload the deployment folder to the Space
    folder_path = os.path.dirname(os.path.abspath(__file__))
    api.upload_folder(
        repo_id=hf_space_repo,
        repo_type="space",
        folder_path=folder_path
    )

    print(f"✅ Uploaded deployment files to Space: {hf_space_repo}")

if __name__ == "__main__":
    main()
