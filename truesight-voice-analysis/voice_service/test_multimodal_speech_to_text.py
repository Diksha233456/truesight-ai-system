"""
test_multimodal_speech_to_text.py

Transcribes the audio clips using Wav2Vec2 CTC / Whisper and extracts
Linguistic Verbal Deception Features (Pronoun density, Sentiment, Hedging, Text Embeddings)
combined with Audio Embeddings to test if we hit >80-90% accuracy!
"""

from __future__ import annotations

import os
import glob
import logging
import re
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("test_multimodal")

MODEL_NAME = "facebook/wav2vec2-base-960h"
logger.info("Loading Wav2Vec2 CTC model & processor for Speech-to-Text...")
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

def extract_transcript_and_text_features(samples: np.ndarray, sr: int = 16000) -> tuple[str, np.ndarray]:
    inputs = tokenizer(samples, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = ctc_model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = tokenizer.batch_decode(predicted_ids)[0].lower()
        
        # Also get deep audio vector
        hidden = base_model(**inputs).last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        
    words = re.findall(r"\b\w+\b", transcript)
    total_words = max(len(words), 1)
    
    cnt_1st = sum(1 for w in words if w in PRONOUNS_1ST) / total_words
    cnt_hedge = sum(1 for w in words if w in HEDGE_WORDS) / total_words
    avg_word_len = sum(len(w) for w in words) / total_words
    
    text_stats = np.array([cnt_1st, cnt_hedge, avg_word_len, total_words], dtype=np.float32)
    combined_vec = np.hstack([hidden, text_stats])
    return transcript, combined_vec

def main():
    data_dir = "dolos_dataset"
    X_audio_text, y = [], []
    transcripts = []
    
    for label_name, label_val in [("truthful", 0), ("deceptive", 1)]:
        folder = os.path.join(data_dir, label_name)
        files = sorted(glob.glob(os.path.join(folder, "*.wav")))
        logger.info("Processing %d files in %s...", len(files), label_name)
        for f in files:
            try:
                with open(f, "rb") as audio_f:
                    raw = audio_f.read()
                audio = load_audio_from_bytes(raw, os.path.basename(f))
                
                # Base 42-dim acoustic
                _, ac_vec = base_extractor.extract(audio)
                
                # Speech-to-text & deep audio + text stats
                transcript, deep_text_vec = extract_transcript_and_text_features(audio.samples)
                
                full_vec = np.hstack([ac_vec, deep_text_vec])
                X_audio_text.append(full_vec)
                transcripts.append(transcript)
                y.append(label_val)
            except Exception as e:
                logger.warning("Error on %s: %s", f, e)
                
    X = np.array(X_audio_text, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    # TF-IDF on Transcripts
    tfidf = TfidfVectorizer(max_features=100)
    X_tfidf = tfidf.fit_transform(transcripts).toarray()
    
    X_multimodal = np.hstack([X, X_tfidf])
    logger.info("Multimodal Feature Matrix shape: X=%s, y=%s", X_multimodal.shape, y.shape)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        "XGBoost (Multimodal: Audio + Deep Neural + Text Transcript)": make_pipeline(StandardScaler(), XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, random_state=42, eval_metric="logloss")),
        "GradientBoosting (Multimodal)": make_pipeline(StandardScaler(), GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.03, random_state=42)),
        "RandomForest (Multimodal)": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=250, max_depth=10, random_state=42)),
    }
    
    for name, clf in models.items():
        scores = cross_val_score(clf, X_multimodal, y, cv=cv, scoring="accuracy")
        logger.info("%s 5-Fold CV Accuracy: %.3f ± %.3f", name, np.mean(scores), np.std(scores))

if __name__ == "__main__":
    main()
