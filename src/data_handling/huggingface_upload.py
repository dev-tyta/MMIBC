
# --- Hugging Face Upload Function ---
def upload_to_huggingface(local_folder_path, repo_id, token):
    """
    Uploads a local folder to a Hugging Face dataset repository.
    """
    print(f"\nAttempting to upload folder '{local_folder_path}' to Hugging Face repo: {repo_id}")

    try:
        api = HfApi()

        # Optional: Create the repository if it doesn't exist
        # create_repo(repo_id=repo_id, repo_type="dataset", token=token, exist_ok=True)
        # print(f"Repository '{repo_id}' ensured to exist.")

        api.upload_folder(
            folder_path=local_folder_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Upload initial organized multimodal breast cancer dataset"
        )

        print(f"\nSuccessfully uploaded dataset to https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        print(f"\nError during Hugging Face upload: {e}")
        print("Please ensure:")
        print(f"- Your Hugging Face token is correct and has write access.")
        print(f"- The repository '{repo_id}' exists or you uncommented the create_repo line.")
        print(f"- You have internet connectivity.")


if __name__ == "__main__":
    # 3. Prepare Dataset Card (README.md)
    # Ensure your README.md (Dataset Card) is in the base directory you will upload from.
    # You can copy the content from the immersive artifact into a file named README.md

    # 4. Upload to Hugging Face
    upload_to_huggingface(
        ORGANIZED_DATASET_BASE_PATH,
        HF_REPO_NAME,
        HF_TOKEN
    )
