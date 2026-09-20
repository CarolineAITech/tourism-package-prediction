"""
hosting.py
-----------
Stage 4 of the MLOps pipeline: hosting.

Pushes every file in the deployment folder (app.py, Dockerfile,
requirements.txt) into a Docker-backed Hugging Face Space, which rebuilds
and redeploys the Streamlit app automatically on each upload.
"""

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
HF_USERNAME = "YOUR_HF_USERNAME"
SPACE_REPO = f"{HF_USERNAME}/tourism-package-app"
DEPLOYMENT_FOLDER = "tourism_project/deployment"
HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

# ----------------------------------------------------------------------
# Create the Space if it does not exist. sdk="docker" makes the Space build
# from our Dockerfile rather than using the managed Streamlit runtime.
# ----------------------------------------------------------------------
try:
    api.repo_info(repo_id=SPACE_REPO, repo_type="space")
    print(f"Space '{SPACE_REPO}' already exists. Updating it.")
except RepositoryNotFoundError:
    print(f"Space '{SPACE_REPO}' not found. Creating it...")
    create_repo(repo_id=SPACE_REPO, repo_type="space",
                space_sdk="docker", private=False, token=HF_TOKEN)
    print("Created.")

# ----------------------------------------------------------------------
# Upload the whole deployment folder in one commit
# ----------------------------------------------------------------------
api.upload_folder(
    folder_path=DEPLOYMENT_FOLDER,
    repo_id=SPACE_REPO,
    repo_type="space",
    commit_message="Deploy Wellness Tourism Package predictor",
)

print(f"Deployed to https://huggingface.co/spaces/{SPACE_REPO}")
