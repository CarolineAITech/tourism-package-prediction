"""
prep.py
--------
Stage 2 of the MLOps pipeline: data preparation.

  1. Loads the raw dataset straight from the Hugging Face dataset space.
  2. Cleans it (drops index/ID columns, fixes inconsistent category labels).
  3. Splits into stratified train/test sets and saves them locally.
  4. Pushes Xtrain/Xtest/ytrain/ytest back to the Hugging Face dataset space.
"""

import os
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
HF_USERNAME = "YOUR_HF_USERNAME"
DATASET_REPO = f"{HF_USERNAME}/tourism-package-data"
HF_TOKEN = os.getenv("HF_TOKEN")

TARGET = "ProdTaken"
OUTPUT_DIR = "tourism_project/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

api = HfApi(token=HF_TOKEN)

# ----------------------------------------------------------------------
# 1. Load the dataset directly from the Hugging Face data space
#    (hf:// paths are resolved natively by pandas via huggingface_hub)
# ----------------------------------------------------------------------
DATASET_PATH = f"hf://datasets/{DATASET_REPO}/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print(f"Loaded raw dataset from the Hub with shape {df.shape}")

# ----------------------------------------------------------------------
# 2. Data cleaning
# ----------------------------------------------------------------------

# 'Unnamed: 0' is a leftover pandas index and 'CustomerID' is a unique
# identifier. Neither carries predictive signal, and leaving the ID in
# would let tree models memorise individual customers, so both are dropped.
drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
df = df.drop(columns=drop_cols)
print(f"Dropped non-predictive columns: {drop_cols}")

# The Gender column contains a data-entry artefact: 'Fe Male' is the same
# category as 'Female'. Merging them prevents a spurious third category.
df["Gender"] = df["Gender"].replace("Fe Male", "Female")

# 'Unmarried' and 'Single' describe the same marital state. Collapsing them
# reduces sparsity in the one-hot encoding without losing information.
df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried", "Single")

# Defensive handling of missing values. The supplied file is complete, but
# the pipeline re-runs on refreshed data, so we impute rather than assume.
for col in df.columns:
    if df[col].isna().any():
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

# Duplicate customer records would leak between train and test, so remove them.
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate rows. Final shape: {df.shape}")

# ----------------------------------------------------------------------
# 3. Train/test split (stratified -- the target is imbalanced at ~19% positive)
# ----------------------------------------------------------------------
X = df.drop(columns=[TARGET])
y = df[TARGET]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {Xtrain.shape}, Test: {Xtest.shape}")
print(f"Positive rate -- train: {ytrain.mean():.3f}, test: {ytest.mean():.3f}")

# Save locally
Xtrain.to_csv(f"{OUTPUT_DIR}/Xtrain.csv", index=False)
Xtest.to_csv(f"{OUTPUT_DIR}/Xtest.csv", index=False)
ytrain.to_csv(f"{OUTPUT_DIR}/ytrain.csv", index=False)
ytest.to_csv(f"{OUTPUT_DIR}/ytest.csv", index=False)

# ----------------------------------------------------------------------
# 4. Upload the prepared splits back to the Hugging Face dataset space
# ----------------------------------------------------------------------
for fname in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=f"{OUTPUT_DIR}/{fname}",
        path_in_repo=fname,
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"Uploaded {fname} to the Hub.")

print("Data preparation complete.")
