"""
train.py
---------
Stage 3 of the MLOps pipeline: model building with experiment tracking.

  1. Loads the prepared train/test splits from the Hugging Face dataset space.
  2. Defines an XGBoost classifier inside a preprocessing pipeline.
  3. Tunes it with GridSearchCV over a defined parameter grid.
  4. Logs every tuned parameter combination and all metrics to MLflow.
  5. Evaluates the best model on the held-out test set.
  6. Registers the best model in the Hugging Face model hub.
"""

import os
import json
import warnings
import joblib
import pandas as pd
import mlflow

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, classification_report,
                             confusion_matrix)
from xgboost import XGBClassifier

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
HF_USERNAME = "YOUR_HF_USERNAME"
DATASET_REPO = f"{HF_USERNAME}/tourism-package-data"
MODEL_REPO = f"{HF_USERNAME}/tourism-package-model"
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_DIR = "tourism_project/model_building"
MODEL_PATH = f"{MODEL_DIR}/best_tourism_model.joblib"

# Point MLflow at the local tracking server started by the workflow.
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tourism-package-prediction")

# ----------------------------------------------------------------------
# 1. Load the prepared train/test data from the Hugging Face data space
# ----------------------------------------------------------------------
base = f"hf://datasets/{DATASET_REPO}"
Xtrain = pd.read_csv(f"{base}/Xtrain.csv")
Xtest = pd.read_csv(f"{base}/Xtest.csv")
ytrain = pd.read_csv(f"{base}/ytrain.csv").squeeze()
ytest = pd.read_csv(f"{base}/ytest.csv").squeeze()
print(f"Train {Xtrain.shape} | Test {Xtest.shape}")

# ----------------------------------------------------------------------
# 2. Define the preprocessing + model pipeline
# ----------------------------------------------------------------------
categorical = Xtrain.select_dtypes(include="object").columns.tolist()
numeric = [c for c in Xtrain.columns if c not in categorical]
print(f"Categorical features: {categorical}")
print(f"Numeric features: {numeric}")

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
])

# Only ~19% of customers convert. scale_pos_weight rebalances the loss so the
# model does not simply predict "no purchase" for everyone -- recall on the
# buyers is what actually matters commercially.
scale_pos_weight = (ytrain == 0).sum() / (ytrain == 1).sum()
print(f"scale_pos_weight = {scale_pos_weight:.3f}")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )),
])

# ----------------------------------------------------------------------
# 3. Parameter grid for tuning
# ----------------------------------------------------------------------
param_grid = {
    "classifier__n_estimators": [200, 400],
    "classifier__max_depth": [4, 6, 8],
    "classifier__learning_rate": [0.05, 0.1],
    "classifier__subsample": [0.8, 1.0],
    "classifier__colsample_bytree": [0.8, 1.0],
}

# F1 is the scoring metric because the classes are imbalanced: it balances
# catching real buyers (recall) against wasting sales calls (precision).
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1,
)

# ----------------------------------------------------------------------
# 4 & 5. Run the search inside an MLflow run, logging every combination
# ----------------------------------------------------------------------
with mlflow.start_run(run_name="xgboost_gridsearch"):

    grid_search.fit(Xtrain, ytrain)

    # Log each candidate parameter set as a nested child run so the whole
    # search space is inspectable in the MLflow UI, not just the winner.
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])
            mlflow.log_metric("std_cv_f1", results["std_test_score"][i])

    best_model = grid_search.best_estimator_
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV F1: {grid_search.best_score_:.4f}")

    # Log the winning configuration on the parent run
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)

    # -- Evaluate on both splits ---------------------------------------
    def evaluate(X, y, split):
        pred = best_model.predict(X)
        proba = best_model.predict_proba(X)[:, 1]
        m = {
            f"{split}_accuracy": accuracy_score(y, pred),
            f"{split}_precision": precision_score(y, pred),
            f"{split}_recall": recall_score(y, pred),
            f"{split}_f1": f1_score(y, pred),
            f"{split}_roc_auc": roc_auc_score(y, proba),
        }
        mlflow.log_metrics(m)
        return m, pred

    train_metrics, _ = evaluate(Xtrain, ytrain, "train")
    test_metrics, test_pred = evaluate(Xtest, ytest, "test")

    print("\nTrain metrics:", json.dumps(train_metrics, indent=2))
    print("Test metrics:", json.dumps(test_metrics, indent=2))
    print("\nClassification report (test):")
    print(classification_report(ytest, test_pred, digits=4))
    print("Confusion matrix (test):")
    print(confusion_matrix(ytest, test_pred))

    # Persist the fitted pipeline (preprocessing + model in one object)
    joblib.dump(best_model, MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

# ----------------------------------------------------------------------
# 6. Register the best model in the Hugging Face model hub
# ----------------------------------------------------------------------
api = HfApi(token=HF_TOKEN)

try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
    print(f"Model repo '{MODEL_REPO}' already exists. Reusing it.")
except RepositoryNotFoundError:
    create_repo(repo_id=MODEL_REPO, repo_type="model",
                private=False, token=HF_TOKEN)
    print(f"Created model repo '{MODEL_REPO}'.")

api.upload_file(
    path_or_fileobj=MODEL_PATH,
    path_in_repo="best_tourism_model.joblib",
    repo_id=MODEL_REPO,
    repo_type="model",
)
print(f"Registered model at https://huggingface.co/{MODEL_REPO}")
