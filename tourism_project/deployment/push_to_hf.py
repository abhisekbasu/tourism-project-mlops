import os
from huggingface_hub import HfApi, create_repo

def main():
    # Reading the target HF Space repo from environment variable
    hf_space_repo = os.environ.get("HF_SPACE_REPO")
    if not hf_space_repo:
        raise ValueError("HF_SPACE_REPO environment variable is not set.")

    # Creating the Docker Space if it does not already exist
    create_repo(repo_id=hf_space_repo, repo_type="space", space_sdk="docker", exist_ok=True)

    api  = HfApi()
    base = os.path.dirname(os.path.abspath(__file__))

    # Uploading all three deployment files to the Hugging Face Space
    for filename in ["Dockerfile", "requirements.txt", "app.py"]:
        local_path = os.path.join(base, filename)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=hf_space_repo,
            repo_type="space",
        )
        print(f"  Uploaded {filename} to hf://spaces/{hf_space_repo}/")

    print("\nDeployment complete.")
    print(f"Space URL: https://huggingface.co/spaces/{hf_space_repo}")

if __name__ == "__main__":
    main()
