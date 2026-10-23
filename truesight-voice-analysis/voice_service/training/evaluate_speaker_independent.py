"""
training/evaluate_speaker_independent.py

Runs speaker-independent GroupKFold cross-validation on the TRUSIGHT voice dataset.
Compares standard random k-fold vs speaker-grouped k-fold to measure performance drop under speaker domain shifts.
Saves json metrics to reports/speaker_independent_metrics.json.
"""

import argparse
import glob
import json
import logging
import os
import re
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.feature_extractor import FeatureExtractor, FEATURE_ORDER
from app.utils import load_audio_from_bytes

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("evaluate_speaker_independent")

REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))

def infer_speaker_id(filename: str) -> str:
    """
    Derives speaker identity from filename patterns.
    Examples:
        reallife_deceptive_trial_lie_001.wav -> trial_lie_001
        torgo_spk01_chunk2.wav -> torgo_spk01
        spk_005_12.wav -> spk_005
    """
    base = os.path.basename(filename)
    match = re.search(r'(trial_[a-z]+_\d+|spk_\d+|torgo_[a-z0-9]+|person_\d+|clip_\d+)', base, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    # Fallback to chunk prefix before numeric suffix
    parts = base.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return base

def load_or_generate_dataset(data_dir: str = None):
    if data_dir and os.path.exists(data_dir):
        logger.info("Extracting acoustic features from %s...", data_dir)
        extractor = FeatureExtractor()
        X, y, groups = [], [], []

        for label_name, label_val in [("truthful", 0), ("deceptive", 1)]:
            folder = os.path.join(data_dir, label_name)
            files = sorted(glob.glob(os.path.join(folder, "*.wav")))
            for path in files:
                try:
                    with open(path, "rb") as f:
                        raw = f.read()
                    audio = load_audio_from_bytes(raw, os.path.basename(path))
                    _, vec = extractor.extract(audio)
                    spk = infer_speaker_id(path)
                    X.append(vec)
                    y.append(label_val)
                    groups.append(spk)
                except Exception as e:
                    logger.warning("Skipping %s: %s", path, e)
        if len(X) > 0:
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(groups)

    logger.warning("No real data found or specified. Generating speaker-grouped synthetic dataset...")
    rng = np.random.default_rng(42)
    n_speakers = 20
    clips_per_speaker = 10
    X, y, groups = [], [], []
    for spk_idx in range(n_speakers):
        spk_id = f"speaker_{spk_idx:02d}"
        # Random label assigned to speaker
        spk_label = 0 if spk_idx % 2 == 0 else 1
        spk_base_pitch = 120.0 + rng.normal(0, 30.0)
        for c in range(clips_per_speaker):
            row = []
            for name in FEATURE_ORDER:
                val = rng.normal(0, 1.0)
                if name == "pitch_mean":
                    val = spk_base_pitch + rng.normal(0, 5.0)
                elif name == "jitter_local" and spk_label == 1:
                    val = 0.02 + rng.normal(0, 0.005)
                row.append(val)
            X.append(row)
            y.append(spk_label)
            groups.append(spk_id)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(groups)

def evaluate(X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Standard Stratified K-Fold (Random Split)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_accs, random_f1s = [], []
    for train_idx, test_idx in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        random_accs.append(accuracy_score(y[test_idx], preds))
        p, r, f1, _ = precision_recall_fscore_support(y[test_idx], preds, average="macro", zero_division=0)
        random_f1s.append(f1)

    # 2. Speaker-Independent GroupKFold
    gkf = GroupKFold(n_splits=5)
    speaker_accs, speaker_f1s = [], []
    conf_matrices = []

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])
        speaker_accs.append(accuracy_score(y[test_idx], preds))
        p, r, f1, _ = precision_recall_fscore_support(y[test_idx], preds, average="macro", zero_division=0)
        speaker_f1s.append(f1)
        conf_matrices.append(confusion_matrix(y[test_idx], preds, labels=[0, 1]).tolist())

    results = {
        "n_samples": int(len(y)),
        "n_unique_speakers": int(len(set(groups))),
        "random_split_eval": {
            "mean_accuracy": float(np.mean(random_accs)),
            "std_accuracy": float(np.std(random_accs)),
            "mean_f1_macro": float(np.mean(random_f1s))
        },
        "speaker_independent_eval": {
            "mean_accuracy": float(np.mean(speaker_accs)),
            "std_accuracy": float(np.std(speaker_accs)),
            "mean_f1_macro": float(np.mean(speaker_f1s)),
            "fold_accuracies": [float(a) for a in speaker_accs]
        }
    }

    out_file = os.path.join(REPORTS_DIR, "speaker_independent_metrics.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Evaluation Complete. Results saved to %s", out_file)
    logger.info("Random Split Accuracy: %.3f ± %.3f", results["random_split_eval"]["mean_accuracy"], results["random_split_eval"]["std_accuracy"])
    logger.info("Speaker Independent Accuracy: %.3f ± %.3f", results["speaker_independent_eval"]["mean_accuracy"], results["speaker_independent_eval"]["std_accuracy"])

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="dolos_dataset")
    args = parser.parse_args()

    X, y, groups = load_or_generate_dataset(args.data_dir if os.path.exists(args.data_dir) else None)
    evaluate(X, y, groups)

if __name__ == "__main__":
    main()
