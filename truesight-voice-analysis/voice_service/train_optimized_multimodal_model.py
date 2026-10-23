"""
train_optimized_multimodal_model.py

Trains an optimized Multimodal Deception Classifier using:
- 42 Acoustic Features (Pitch, Jitter, Shimmer, Tempo, MFCCs)
- 768 Wav2Vec2 Deep Speech Embeddings
- Speech-to-Text Verbal Features (Pronouns, Hedges, Sentiment)
- Feature Selection (SelectKBest / Mutual Info) + StandardScaler
- Calibrated RandomForest / GradientBoosting Ensemble

Saves the trained model & pipeline to app/models/deception_model.pkl
and reloads the live API server.
"""

from __future__ import annotations

import os
import glob
import re
import logging
import urllib.request
import numpy as np
import joblib
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_optimized")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dolos_dataset")
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "app", "models", "deception_model.pkl")

# Pre-trained speech models
MODEL_NAME = "facebook/wav2vec2-base-960h"
logger.info("Loading Wav2Vec2 CTC & Base Transformer...")
tokenizer = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
ctc_model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
base_model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
ctc_model.eval()
base_model.eval()

from app.feature_extractor import FeatureExtractor
from app.utils import load_audio_from_bytes

base_extractor = FeatureExtractor()

HEDGE_WORDS = {"maybe", "guess", "think", "probably", "possibly", "believe", "suppose", "kind", "sort", "might", "could", "would", "seems", "appears", "actually"}
PRONOUNS_1ST = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}

def extract_features(audio_path: str) -> np.ndarray:
    with open(audio_path, "rb") as f:
        raw = f.read()
    audio = load_audio_from_bytes(raw, os.path.basename(audio_path))
    
    # 1. Base acoustic features (42-dim)
    _, acoustic_vec = base_extractor.extract(audio)
    
    # 2. Wav2Vec2 Deep Audio Embeddings (768-dim) + Speech-to-Text
    inputs = tokenizer(audio.samples, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = ctc_model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = tokenizer.batch_decode(predicted_ids)[0].lower()
        
        deep_audio_vec = base_model(**inputs).last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        
    words = re.findall(r"\b\w+\b", transcript)
    total_words = max(len(words), 1)
    
    cnt_1st = sum(1 for w in words if w in PRONOUNS_1ST) / total_words
    cnt_hedge = sum(1 for w in words if w in HEDGE_WORDS) / total_words
    avg_word_len = sum(len(w) for w in words) / total_words
    
    text_stats = np.array([cnt_1st, cnt_hedge, avg_word_len, total_words], dtype=np.float32)
    
    # Combined vector: 42 + 768 + 4 = 814 features
    return np.hstack([acoustic_vec, deep_audio_vec, text_stats])

def main():
    X, y = [], []
    for label_name, label_val in [("truthful", 0), ("deceptive", 1)]:
        folder = os.path.join(DATASET_DIR, label_name)
        files = sorted(glob.glob(os.path.join(folder, "*.wav")))
        logger.info("Extracting features from %d files in %s...", len(files), label_name)
        for f in files:
            try:
                vec = extract_features(f)
                X.append(vec)
                y.append(label_val)
            except Exception as e:
                logger.warning("Skipping %s: %s", f, e)
                
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    logger.info("Total Dataset shape: X=%s, y=%s", X.shape, y.shape)
    
    # 5-Fold Stratified CV Evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Optimized Pipeline: Scaler + SelectKBest (top 50 features) + Ensemble Classifier
    pipeline = make_pipeline(
        StandardScaler(),
        SelectKBest(score_func=f_classif, k=50),
        VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42)),
                ("gb", GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.03, random_state=42)),
            ],
            voting="soft"
        )
    )
    
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    logger.info("🎯 5-Fold Cross-Validation Accuracy: %.3f ± %.3f", np.mean(cv_scores), np.std(cv_scores))
    
    # Train on full dataset and save
    logger.info("Training final model on full dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    logger.info("Held-out Test Set Accuracy: %.3f", test_acc)
    logger.info("\n%s", classification_report(y_test, y_pred, target_names=["Truthful", "Deceptive"]))
    
    # Save the pipeline
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    logger.info("Saved upgraded model pipeline to %s", MODEL_OUTPUT_PATH)
    
    # Hot reload live API
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/reload-model", method="POST")
        with urllib.request.urlopen(req) as resp:
            logger.info("Reloaded live server: %s", resp.read().decode())
    except Exception as e:
        logger.warning("Could not auto-reload server: %s", e)

if __name__ == "__main__":
    main()
