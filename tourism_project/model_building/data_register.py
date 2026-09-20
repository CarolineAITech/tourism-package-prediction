"""
data_register.py
-----------------
Registers the raw tourism.csv dataset on the Hugging Face Hub as a dataset repo.
This is the first stage of the MLOps pipeline: it makes the raw data a
versioned, remotely addressable artifact that every later stage pulls from.
"""

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ----------------------------------------------------------------------
# Configuration -- change HF_USERNAME to your own Hugging Face username
# ----------------------------------------------------------------------
HF_USERNAME = "CarolineBuildsAI"
DATASET_REPO = f"{HF_USERNAME}/tourism-package-data"
LOCAL_DATA_FILE = "tourism_project/data/tourism.csv"

# The token is read from the environment so the same script works locally,
# in Colab, and inside a GitHub Actions runner without code changes.
HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

# ----------------------------------------------------------------------
# Create the dataset repo if it does not already exist (idempotent)
# ----------------------------------------------------------------------
try:
    api.repo_info(repo_id=DATASET_REPO, repo_type="dataset")
    print(f"Dataset repo '{DATASET_REPO}' already exists. Reusing it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{DATASET_REPO}' not found. Creating it...")
    create_repo(repo_id=DATASET_REPO, repo_type="dataset",
                private=False, token=HF_TOKEN)
    print("Created.")

# ----------------------------------------------------------------------
# Upload the raw CSV into the dataset space
# ----------------------------------------------------------------------
api.upload_file(
    path_or_fileobj=LOCAL_DATA_FILE,
    path_in_repo="tourism.csv",          # name it will carry on the Hub
    repo_id=DATASET_REPO,
    repo_type="dataset",
)

print(f"Uploaded tourism.csv to https://huggingface.co/datasets/{DATASET_REPO}")
