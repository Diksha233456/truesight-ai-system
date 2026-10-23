"""
train_model.py

Trains the Random Forest classifier used by app/predictor.py.

USAGE
-----
Two modes:

1. Train on real labeled audio (recommended before any real use):

    python train_model.py --data-dir path/to/dataset

   The data directory must contain two subfolders:
       path/to/dataset/truthful/*.wav
       path/to/dataset/deceptive/*.wav

   Public datasets with deception-labeled speech you can adapt include the
   Columbia-SRI-Colorado (CSC) deceptive speech corpus and the
   Real-Life Trial dataset (Perez-Rosas et al., 2015). Both require
   separate access requests from their maintainers — see README.md.

2. No dataset available (prototype / smoke-test mode):

    python train_model.py

   This generates a small synthetic feature dataset (NOT real speech) with
   two separable Gaussian-ish clusters standing in for "truthful" and
   "deceptive" acoustic profiles, purely so the API has a loadable model to
   demonstrate the full request/response pipeline end-to-end. A model
   trained this way MUST NOT be used to draw any real conclusions — it only
   proves the plumbing works. Replace it with a model trained on real,
   labeled data before any actual evaluation.

Either way, the trained model is written to app/models/deception_model.pkl.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.feature_extractor import FeatureExtractor, FEATURE_ORDER  # noqa: E402
from app.utils import load_audio_from_bytes  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_model")

MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "app", "models", "deception_model.pkl")
N_FEATURES = len(FEATURE_ORDER)


def extract_features_from_dir(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Walk truthful/ and deceptive/ subfolders and extract features from each wav."""
    extractor = FeatureExtractor()
    X, y = [], []

    for label_name, label_value in [("truthful", 0), ("deceptive", 1)]:
        folder = os.path.join(data_dir, label_name)
        files = sorted(glob.glob(os.path.join(folder, "*.wav")))
        if not files:
            logger.warning("No .wav files found in %s", folder)
        for path in files:
            try:
                with open(path, "rb") as f:
                    raw = f.read()
                audio = load_audio_from_bytes(raw, os.path.basename(path))
                _, vector = extractor.extract(audio)
                X.append(vector)
                y.append(label_value)
            except Exception as e:
                logger.warning("Skipping %s: %s", path, e)

    if not X:
        raise RuntimeError(
            f"No usable audio found under {data_dir}. Expected {data_dir}/truthful/*.wav "
            f"and {data_dir}/deceptive/*.wav"
        )

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def synthetic_dataset(n_per_class: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic, feature-space-only dataset for smoke-testing the
    pipeline when no real labeled audio is available. This is NOT derived
    from real speech and encodes only loosely plausible separations
    (e.g. deceptive class centered on slightly higher pitch variability,
    jitter, shimmer, and pause ratio) based on commonly cited — but far
    from settled — findings in the vocal-deception literature.
    """
    rng = np.random.default_rng(seed)

    def make_class(mean_overrides: dict) -> np.ndarray:
        base_means = {
            "pitch_mean": 170.0, "pitch_std": 18.0, "pitch_min": 90.0, "pitch_max": 260.0,
            "jitter_local": 0.010, "shimmer_local": 0.035,
            "speech_tempo": 2.2, "pause_ratio": 0.12, "pause_count": 3.0, "silence_duration": 1.0,
            "rms_energy_mean": 0.08, "rms_energy_std": 0.02,
            "zero_crossing_rate": 0.07, "spectral_centroid": 1800.0,
            "spectral_bandwidth": 1500.0, "spectral_rolloff": 3300.0,
        }
        base_means.update(mean_overrides)
        mfcc_means = {f"mfcc_mean_{i}": 0.0 for i in range(13)}
        mfcc_stds = {f"mfcc_std_{i}": 5.0 for i in range(13)}
        base_means.update(mfcc_means)
        base_means.update(mfcc_stds)

        rows = []
        for _ in range(n_per_class):
            row = [
                base_means[name] * (1 + rng.normal(0, 0.08))
                if base_means[name] != 0
                else rng.normal(0, 1.0)
                for name in FEATURE_ORDER
            ]
            rows.append(row)
        return np.array(rows, dtype=np.float32)

    truthful = make_class({})
    deceptive = make_class({
        "pitch_std": 30.0,
        "jitter_local": 0.020,
        "shimmer_local": 0.070,
        "pause_ratio": 0.28,
        "pause_count": 6.0,
        "speech_tempo": 1.4,
        "rms_energy_std": 0.05,
    })

    X = np.vstack([truthful, deceptive])
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.int64)
    return X, y


import datetime
import hashlib
import json
import sklearn
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

DEFAULT_VERSION = "v1.0.0"
DEFAULT_EXPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def compute_sha256(filepath: str) -> str:
    """Compute hex SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def train(X: np.ndarray, y: np.ndarray) -> tuple[RandomForestClassifier, dict]:
    """
    Train classifier and compute validation metrics.
    Returns:
        final_clf: fitted RandomForestClassifier on full dataset
        metrics: dictionary of evaluation metrics
    """
    # 1. 5-Fold Stratified Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    cv_scores = cross_val_score(cv_clf, X, y, cv=cv, scoring="accuracy")
    logger.info("5-Fold Cross-Validation Accuracy: %.3f ± %.3f", float(np.mean(cv_scores)), float(np.std(cv_scores)))

    # 2. Report accuracy using a 20% validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf_eval = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf_eval.fit(X_train, y_train)

    y_pred = clf_eval.predict(X_test)
    y_proba = clf_eval.predict_proba(X_test)[:, 1]
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        auc = None

    logger.info("Random Split Validation (held-out) accuracy: %.3f", acc)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["Truthful", "Deceptive"]))

    # 3. Train final production classifier on 100% of the dataset
    logger.info("Training final production classifier on 100% of the dataset...")
    final_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    final_clf.fit(X, y)
    
    train_acc = float(accuracy_score(y, final_clf.predict(X)))
    logger.info("Training accuracy on full dataset: %.3f", train_acc)

    metrics = {
        "held_out_accuracy": round(acc, 4),
        "held_out_precision": round(prec, 4),
        "held_out_recall": round(rec, 4),
        "held_out_f1": round(f1, 4),
        "held_out_roc_auc": round(auc, 4) if auc is not None else None,
        "cv_accuracy_mean": round(float(np.mean(cv_scores)), 4),
        "cv_accuracy_std": round(float(np.std(cv_scores)), 4),
        "full_dataset_accuracy": round(train_acc, 4),
    }

    return final_clf, metrics


def export_artifacts(
    clf: RandomForestClassifier,
    metrics: dict,
    export_dir: str,
    version: str,
    dataset_name: str,
    n_samples: int,
    n_truthful: int,
    n_deceptive: int,
) -> tuple[str, str]:
    """Export model and metadata JSON with checksums to export_dir."""
    os.makedirs(export_dir, exist_ok=True)
    model_filename = "deception_model.pkl"
    metadata_filename = "model_metadata.json"

    model_path = os.path.join(export_dir, model_filename)
    joblib.dump(clf, model_path)
    file_size = os.path.getsize(model_path)
    sha256 = compute_sha256(model_path)

    metadata = {
        "model_name": "trusight-voice-deception",
        "model_version": version,
        "framework": "scikit-learn",
        "framework_version": sklearn.__version__,
        "architecture": "RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)",
        "feature_dimension": N_FEATURES,
        "feature_names": FEATURE_ORDER,
        "classes": [0, 1],
        "class_names": ["Truthful", "Deceptive"],
        "decision_threshold": 0.5,
        "dataset_info": {
            "name": dataset_name,
            "total_samples": n_samples,
            "truthful_samples": n_truthful,
            "deceptive_samples": n_deceptive,
        },
        "evaluation_metrics": metrics,
        "artifacts": {
            model_filename: {
                "sha256": sha256,
                "size_bytes": file_size,
            }
        },
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    metadata_path = os.path.join(export_dir, metadata_filename)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved model artifact to %s (size=%d bytes, sha256=%s)", model_path, file_size, sha256)
    logger.info("Saved metadata artifact to %s", metadata_path)

    # Also sync into app/models for immediate local execution
    app_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "models")
    os.makedirs(app_models_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(app_models_dir, model_filename))
    with open(os.path.join(app_models_dir, metadata_filename), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Synchronized local copy in %s", app_models_dir)

    return model_path, metadata_path


def main():
    parser = argparse.ArgumentParser(description="Train and export the TRUSIGHT voice deception classifier.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing truthful/ and deceptive/ subfolders of .wav files. "
        "Defaults to 'dolos_dataset' if present, otherwise uses synthetic prototype dataset.",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=DEFAULT_EXPORT_DIR,
        help="Directory to save the trained model artifact and model_metadata.json (default: artifacts/).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=DEFAULT_VERSION,
        help="Model version string (e.g., v1.0.0).",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    # Auto-detect dolos_dataset if not specified
    if not data_dir:
        default_dolos = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dolos_dataset")
        if os.path.isdir(default_dolos) and os.path.isdir(os.path.join(default_dolos, "truthful")):
            data_dir = default_dolos
            logger.info("Auto-detected local dataset at %s", data_dir)

    if data_dir and os.path.isdir(data_dir):
        logger.info("Extracting features from real audio in %s ...", data_dir)
        X, y = extract_features_from_dir(data_dir)
        dataset_name = os.path.basename(os.path.abspath(data_dir))
    else:
        logger.warning(
            "No --data-dir provided and no local dolos_dataset found. Training on a SYNTHETIC "
            "placeholder dataset. For production releases, provide a real labeled dataset."
        )
        X, y = synthetic_dataset()
        dataset_name = "synthetic_prototype"

    n_samples = len(y)
    n_truthful = int(np.sum(y == 0))
    n_deceptive = int(np.sum(y == 1))
    logger.info("Dataset shape: X=%s, y=%s (truthful=%d, deceptive=%d)", X.shape, y.shape, n_truthful, n_deceptive)

    clf, metrics = train(X, y)

    model_path, metadata_path = export_artifacts(
        clf=clf,
        metrics=metrics,
        export_dir=args.export_dir,
        version=args.version,
        dataset_name=dataset_name,
        n_samples=n_samples,
        n_truthful=n_truthful,
        n_deceptive=n_deceptive,
    )
    logger.info("Training and artifact export complete: version=%s, export_dir=%s", args.version, args.export_dir)


if __name__ == "__main__":
    main()
