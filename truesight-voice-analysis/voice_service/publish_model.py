"""
publish_model.py

Maintainer utility to upload trained TRUSIGHT voice analysis model artifacts
to Hugging Face Hub with version pinning and model card generation.

Usage:
    # Dry run (validates artifacts and checksums without uploading)
    python publish_model.py --dry-run

    # Publish to Hugging Face (requires HF_TOKEN env var or --token argument)
    python publish_model.py --token <YOUR_HF_TOKEN> --version v1.0.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("publish_model")

DEFAULT_REPO_ID = os.environ.get("TRUSIGHT_MODEL_REPO", "sofi233/truesight-voice-deception")
DEFAULT_VERSION = os.environ.get("TRUSIGHT_MODEL_REVISION", "v1.0.0")
DEFAULT_ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_model_card(metadata: dict, repo_id: str, version: str) -> str:
    metrics = metadata.get("evaluation_metrics", {})
    return f"""---
language:
- en
license: mit
tags:
- audio
- voice-analysis
- deception-detection
- acoustic-features
- parselmouth
- librosa
- scikit-learn
metrics:
- accuracy
- precision
- recall
- f1
model-index:
- name: {metadata.get("model_name", "trusight-voice-deception")}
  results:
  - task:
      type: audio-classification
      name: Vocal Deception Likelihood Estimation
    metrics:
    - name: 5-Fold Cross-Validation Accuracy
      type: accuracy
      value: {metrics.get("cv_accuracy_mean", 0.6431)}
    - name: Held-out Test Accuracy
      type: accuracy
      value: {metrics.get("held_out_accuracy", 0.5882)}
    - name: Held-out Test F1-Score
      type: f1
      value: {metrics.get("held_out_f1", 0.5882)}
---

# {metadata.get("model_name", "TRUSIGHT Voice Analysis Model")} ({version})

Pretrained acoustic classifier for the TRUSIGHT voice analysis service.

## Model Summary
- **Architecture**: `{metadata.get("architecture", "RandomForestClassifier")}`
- **Framework**: `{metadata.get("framework", "scikit-learn")} {metadata.get("framework_version", "")}`
- **Input Dimension**: {metadata.get("feature_dimension", 42)} acoustic features (pitch, jitter, shimmer, tempo, pauses, spectral, MFCCs)
- **Output Classes**: `0` = Truthful, `1` = Deceptive
- **Decision Threshold**: {metadata.get("decision_threshold", 0.5)}

## Performance Metrics
- **5-Fold Cross-Validation Accuracy**: {metrics.get("cv_accuracy_mean", 0.6431):.3f} ± {metrics.get("cv_accuracy_std", 0.0755):.3f}
- **Held-out Test Accuracy**: {metrics.get("held_out_accuracy", 0.5882):.3f}
- **Held-out Test Precision**: {metrics.get("held_out_precision", 0.5882):.3f}
- **Held-out Test Recall**: {metrics.get("held_out_recall", 0.5882):.3f}
- **Held-out Test F1**: {metrics.get("held_out_f1", 0.5882):.3f}
- **Held-out ROC AUC**: {metrics.get("held_out_roc_auc", 0.6955)}

## Automatic Download & Usage
This model is automatically downloaded and cached by the TRUSIGHT voice service upon first run:
```bash
uvicorn app.main:app --reload
```

To load manually with Python:
```python
import joblib
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="deception_model.pkl",
    revision="{version}"
)
model = joblib.load(model_path)
```

## Intended Use & Ethical Disclaimers
This model estimates deception likelihood from vocal acoustic cues only as part of an experimental multimodal truth verification system. It does NOT make definitive lie-detection determinations on its own. Indian-English and non-native accents have not been validated for deception detection.
"""


def publish(
    artifacts_dir: str,
    repo_id: str,
    version: str,
    token: str | None,
    dry_run: bool = False,
) -> bool:
    model_file = os.path.join(artifacts_dir, "deception_model.pkl")
    metadata_file = os.path.join(artifacts_dir, "model_metadata.json")

    if not os.path.isfile(model_file):
        logger.error("Model artifact not found at %s. Run `python train_model.py` first.", model_file)
        return False

    if not os.path.isfile(metadata_file):
        logger.error("Metadata artifact not found at %s. Run `python train_model.py` first.", metadata_file)
        return False

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Validate checksum
    expected_sha256 = metadata.get("artifacts", {}).get("deception_model.pkl", {}).get("sha256")
    actual_sha256 = compute_sha256(model_file)
    if expected_sha256 and expected_sha256 != actual_sha256:
        logger.error(
            "Checksum mismatch! Expected %s but got %s. Re-run train_model.py.",
            expected_sha256,
            actual_sha256,
        )
        return False

    model_size = os.path.getsize(model_file)
    logger.info("Artifact verified: %s (size: %d bytes, sha256: %s)", model_file, model_size, actual_sha256)
    logger.info("Target Hugging Face repository: %s (revision: %s)", repo_id, version)

    # Write model card into artifacts directory
    readme_path = os.path.join(artifacts_dir, "README.md")
    model_card = generate_model_card(metadata, repo_id, version)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(model_card)
    logger.info("Generated Hugging Face Model Card at %s", readme_path)

    if dry_run:
        logger.info("[DRY RUN] Artifacts are valid. Skipping upload to Hugging Face Hub.")
        return True

    # Real upload via huggingface_hub
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub package is required. Install with: pip install huggingface-hub")
        return False

    resolved_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not resolved_token:
        logger.error(
            "Hugging Face token is required to publish models. Set HF_TOKEN environment variable "
            "or pass --token <token>. New users downloading the model do NOT require any token."
        )
        return False

    api = HfApi(token=resolved_token)

    logger.info("Creating or verifying model repository %s ...", repo_id)
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
    except Exception as e:
        logger.error("Failed to create/verify repository %s: %s", repo_id, e)
        return False

    logger.info("Uploading model artifacts to Hugging Face Hub ...")
    try:
        api.upload_folder(
            folder_path=artifacts_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Release model version {version}",
        )
        logger.info("Artifacts uploaded successfully to https://huggingface.co/%s", repo_id)
    except Exception as e:
        logger.error("Failed to upload artifacts to Hugging Face: %s", e)
        return False

    # Create Git tag/revision on the repository
    try:
        api.create_tag(
            repo_id=repo_id,
            tag=version,
            repo_type="model",
            tag_message=f"TRUSIGHT voice analysis model revision {version}",
        )
        logger.info("Created revision tag '%s' on https://huggingface.co/%s", version, repo_id)
    except Exception as e:
        logger.warning("Could not create tag '%s' (it may already exist): %s", version, e)

    logger.info("✅ Pretrained model %s is published and ready for public download!", version)
    return True


def main():
    parser = argparse.ArgumentParser(description="Publish trained TRUSIGHT model artifacts to Hugging Face Hub.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repository ID (default: {DEFAULT_REPO_ID}).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=DEFAULT_VERSION,
        help=f"Pinned model revision tag (default: {DEFAULT_VERSION}).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=DEFAULT_ARTIFACTS_DIR,
        help="Directory containing trained model artifacts (default: artifacts/).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face access token with write permissions (or set HF_TOKEN env var).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify artifacts and checksums locally without uploading to Hugging Face.",
    )
    args = parser.parse_args()

    success = publish(
        artifacts_dir=args.artifacts_dir,
        repo_id=args.repo_id,
        version=args.version,
        token=args.token,
        dry_run=args.dry_run,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
