"""
test_deep_learning_pipeline.py

Tests Wav2Vec2 feature extraction + XGBoost / GradientBoosting classifier on the dataset.
"""

from __future__ import annotations

import os
import glob
import logging
import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("test_dl")

# Load Wav2Vec2 Model
MODEL_NAME = "facebook/wav2vec2-base-960h"
logger.info("Loading Wav2Vec2 model (%s)...", MODEL_NAME)
processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
model.eval()

from app.feature_extractor import FeatureExtractor
from app.utils import load_audio_from_bytes

base_extractor = FeatureExtractor()

def extract_deep_features(audio_path: str) -> np.ndarray:
    with open(audio_path, "rb") as f:
        raw = f.read()
    audio = load_audio_from_bytes(raw, os.path.basename(audio_path))
    
    # 1. Base acoustic features (42-dim)
    _, acoustic_vec = base_extractor.extract(audio)
    
    # 2. Wav2Vec2 Deep Embeddings (768-dim)
    inputs = processor(audio.samples, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        # Pool across time frames to get a single 768-dim vector
        deep_vec = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        
    # Combine acoustic + deep features
    return np.hstack([acoustic_vec, deep_vec])

def main():
    data_dir = "dolos_dataset"
    X, y = [], []
    
    for label_name, label_val in [("truthful", 0), ("deceptive", 1)]:
        folder = os.path.join(data_dir, label_name)
        files = sorted(glob.glob(os.path.join(folder, "*.wav")))
        logger.info("Found %d files in %s", len(files), label_name)
        for f in files:
            try:
                vec = extract_deep_features(f)
                X.append(vec)
                y.append(label_val)
            except Exception as e:
                logger.warning("Error processing %s: %s", f, e)
                
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    logger.info("Extracted dataset shape: X=%s, y=%s", X.shape, y.shape)
    
    # Evaluate classifiers with 5-fold Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        "XGBoost": make_pipeline(StandardScaler(), XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")),
        "GradientBoosting": make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
        "RandomForest": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
    }
    
    for name, clf in models.items():
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        logger.info("%s 5-Fold CV Accuracy: %.3f ± %.3f", name, np.mean(scores), np.std(scores))

if __name__ == "__main__":
    main()
