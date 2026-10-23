"""
deep_feature_extractor.py

Deep multimodal feature extractor using:
- Wav2Vec2 768-dim neural audio embeddings (Meta, facebook/wav2vec2-base-960h)
- Wav2Vec2 CTC Speech-to-Text → Verbal/Linguistic deception features
- Original 42-dim acoustic features (Praat + librosa)

This module is loaded lazily (only on first call) to avoid slowing down
FastAPI startup. Once loaded, the Wav2Vec2 model is kept in memory.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

import numpy as np

from app.utils import LoadedAudio

logger = logging.getLogger("voice_service.deep_features")

HEDGE_WORDS = frozenset({
    "maybe", "guess", "think", "probably", "possibly", "believe",
    "suppose", "kind", "sort", "might", "could", "would", "seems",
    "appears", "actually", "perhaps", "somewhat", "basically",
})
PRONOUNS_1ST = frozenset({
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours"
})

_wav2vec2_loaded = False
_ctc_model = None
_base_model = None
_tokenizer = None


def _load_wav2vec2():
    global _wav2vec2_loaded, _ctc_model, _base_model, _tokenizer
    if _wav2vec2_loaded:
        return
    try:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Model

        MODEL_NAME = "facebook/wav2vec2-base-960h"
        logger.info("Loading Wav2Vec2 models from cache (%s)...", MODEL_NAME)
        _tokenizer = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        _ctc_model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
        _base_model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        _ctc_model.eval()
        _base_model.eval()
        _wav2vec2_loaded = True
        logger.info("Wav2Vec2 models loaded successfully.")
    except Exception as e:
        logger.warning("Failed to load Wav2Vec2 (will use acoustic features only): %s", e)
        _wav2vec2_loaded = False


def extract_deep_features(audio: LoadedAudio) -> np.ndarray:
    """
    Returns a 814-dim feature vector combining:
      - 42  acoustic (pitch, jitter, shimmer, MFCCs, spectral)
      - 768 Wav2Vec2 deep audio embedding
      - 4   verbal deception stats (pronoun density, hedge density, word length, word count)
    """
    import torch
    from app.feature_extractor import FeatureExtractor

    base_extractor = FeatureExtractor()
    _, acoustic_vec = base_extractor.extract(audio)

    _load_wav2vec2()
    if not _wav2vec2_loaded or _tokenizer is None:
        # Fallback: pad with zeros for deep + text features
        deep_vec = np.zeros(768, dtype=np.float32)
        text_stats = np.zeros(4, dtype=np.float32)
        return np.hstack([acoustic_vec, deep_vec, text_stats])

    inputs = _tokenizer(audio.samples, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = _ctc_model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcript = _tokenizer.batch_decode(predicted_ids)[0].lower()
        deep_vec = _base_model(**inputs).last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()

    words = re.findall(r"\b\w+\b", transcript)
    total_words = max(len(words), 1)
    cnt_1st = sum(1 for w in words if w in PRONOUNS_1ST) / total_words
    cnt_hedge = sum(1 for w in words if w in HEDGE_WORDS) / total_words
    avg_word_len = sum(len(w) for w in words) / total_words
    text_stats = np.array([cnt_1st, cnt_hedge, avg_word_len, float(total_words)], dtype=np.float32)

    return np.hstack([acoustic_vec, deep_vec, text_stats])
