"""
EcoSort – Render Build-Time Model Downloader
=============================================
This script is called during `render.yaml` buildCommand.

On Render, committed model weights (.pth) are typically too large for Git
(100 MB+ for MobileNetV2 fine-tuned weights). Instead:

Option A (RECOMMENDED for free Render tier):
  - Upload your trained model to Google Drive (with public share link).
  - Set the GDRIVE_MODEL_FILE_ID environment variable on Render.
  - This script downloads it automatically during build.

Option B:
  - Use Hugging Face Hub (free model hosting, no Git LFS needed).
  - Set MODEL_HF_REPO and MODEL_HF_FILENAME env vars.

Option C:
  - Commit the .pth file directly to your repo (if < 100 MB, or use Git LFS).
  - Skip this script entirely in render.yaml.

If neither option is configured, the server starts without a model
(health check returns model_ready: false).
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s – %(message)s")
log = logging.getLogger("download_model")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "ecosort_mobilenetv2.pth"


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    # If model already exists (e.g. committed to repo), skip download
    if MODEL_PATH.exists():
        log.info("Model already present at '%s' – skipping download.", MODEL_PATH)
        return

    gdrive_id = os.environ.get("GDRIVE_MODEL_FILE_ID", "").strip()
    hf_repo = os.environ.get("MODEL_HF_REPO", "").strip()
    hf_filename = os.environ.get("MODEL_HF_FILENAME", "ecosort_mobilenetv2.pth").strip()

    # -----------------------------------------------------------------------
    # Option A: Google Drive
    # -----------------------------------------------------------------------
    if gdrive_id:
        log.info("Downloading model from Google Drive (id=%s) …", gdrive_id)
        try:
            import gdown
            gdown.download(id=gdrive_id, output=str(MODEL_PATH), quiet=False)
            log.info("Model downloaded successfully.")
            return
        except Exception as exc:
            log.error("Google Drive download failed: %s", exc)

    # -----------------------------------------------------------------------
    # Option B: Hugging Face Hub
    # -----------------------------------------------------------------------
    if hf_repo:
        log.info("Downloading model from HuggingFace Hub (%s / %s) …", hf_repo, hf_filename)
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=hf_repo, filename=hf_filename)
            import shutil
            shutil.copy(path, MODEL_PATH)
            log.info("Model downloaded from HuggingFace Hub.")
            return
        except ImportError:
            log.warning("huggingface_hub not installed; skipping HF download.")
        except Exception as exc:
            log.error("HuggingFace Hub download failed: %s", exc)

    # -----------------------------------------------------------------------
    # No model source configured
    # -----------------------------------------------------------------------
    log.warning(
        "No model source configured. The application will start without a model.\n"
        "To enable inference:\n"
        "  1. Train locally:        python train.py --download --epochs 15\n"
        "  2. Upload the model to Google Drive and set GDRIVE_MODEL_FILE_ID on Render.\n"
        "     OR commit models/ecosort_mobilenetv2.pth to your Git repository.\n"
        "  3. Re-deploy."
    )


if __name__ == "__main__":
    main()

