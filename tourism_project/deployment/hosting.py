"""
hosting.py
-----------
Stage 4 of the MLOps pipeline: hosting.

We deploy the Streamlit app via Streamlit Community Cloud
(https://share.streamlit.io) rather than a Hugging Face Space, because HF now
requires a verified payment method / PRO subscription to run any interactive
Space (Gradio or Docker) on the free cpu-basic tier -- only fully static
Spaces are exempt, and those cannot run a Python backend.

Streamlit Community Cloud deploys straight from a connected GitHub repo and
redeploys automatically on every push to the tracked branch, so there is no
API call needed here to push files anywhere. This script's job is just to
verify the deployment files are present before the pipeline reports success.
"""

import os

DEPLOYMENT_FOLDER = "tourism_project/deployment"
REQUIRED_FILES = ["app.py", "requirements.txt"]

missing = [f for f in REQUIRED_FILES
           if not os.path.exists(f"{DEPLOYMENT_FOLDER}/{f}")]
if missing:
    raise FileNotFoundError(f"Missing deployment files: {missing}")

print(f"Deployment files present in {DEPLOYMENT_FOLDER}: {REQUIRED_FILES}")
print("Streamlit Community Cloud auto-redeploys from GitHub on every push "
      "to the tracked branch once the app is connected at "
      "https://share.streamlit.io (main file path: "
      "tourism_project/deployment/app.py).")
