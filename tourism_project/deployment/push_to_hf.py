"""
push_to_hf.py — Deploy deployment files to Hugging Face Docker Space
=====================================================================
Uploads Dockerfile, requirements.txt, and app.py to the HF Space repo.
All configuration is injected via environment variables.
"""
import os
from huggingface_hub import HfApi, create_repo


def main():
    hf_space_repo = os.environ.get("HF_SPACE_REPO")
    if not hf_space_repo:
        raise ValueError("Environment variable HF_SPACE_REPO is not set.")

    create_repo(
        repo_id=hf_space_repo,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )
    api  = HfApi()
    base = os.path.dirname(os.path.abspath(__file__))

    for filename in ["Dockerfile", "requirements.txt", "app.py"]:
        local_path = os.path.join(base, filename)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=hf_space_repo,
            repo_type="space",
        )
        print(f"  Uploaded {filename} → hf://spaces/{hf_space_repo}/{filename}")

    print("\n[push_to_hf.py] Deployment complete.")
    print(f"  Space URL: https://huggingface.co/spaces/{hf_space_repo}")


if __name__ == "__main__":
    main()
