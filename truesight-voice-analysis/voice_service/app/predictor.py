"""
predictor.py

Wraps the trained deception classifier: loads it once at service startup,
validates the input, and converts raw probability output into the API's
(verdict, score, confidence) contract.

Supports two model types:
1. Legacy: sklearn RandomForestClassifier on 42 acoustic features
2. Upgraded: sklearn Pipeline (Scaler → SelectKBest → Ensemble) on 814-dim
   multimodal features (Acoustic + Wav2Vec2 Deep Audio + Verbal Text features)
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import joblib
import numpy as np

from app.model_loader import resolve_model, ModelLoaderError

logger = logging.getLogger("voice_service.predictor")

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "deception_model.pkl")

DECEPTION_THRESHOLD = 0.5

# 814-dim multimodal: 42 acoustic + 768 Wav2Vec2 + 4 verbal
MULTIMODAL_FEATURE_DIM = 814
# 42-dim legacy acoustic-only
LEGACY_FEATURE_DIM = 42


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is requested but no model file could be loaded."""


class Predictor:
    """
    Thin wrapper around a scikit-learn model persisted with joblib.
    Loads the pretrained model once at service startup from local cache
    or downloads it automatically from the configured hosting location.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.explicit_model_path = model_path
        self.model_path = None
        self.metadata = None
        self.model = None
        self.is_multimodal = False
        self._load()

    def _load(self) -> None:
        try:
            if self.explicit_model_path:
                if not os.path.exists(self.explicit_model_path):
                    logger.error("Explicit model path does not exist: %s", self.explicit_model_path)
                    self.model = None
                    return
                resolved_path = self.explicit_model_path
                self.metadata = None
            else:
                path_obj, meta = resolve_model()
                resolved_path = str(path_obj)
                self.metadata = meta

            self.model_path = resolved_path
            self.model = joblib.load(resolved_path)
            # Detect if it's the multimodal pipeline or legacy RF
            self.is_multimodal = hasattr(self.model, "steps")
            mode = "Multimodal Pipeline (Wav2Vec2 + Ensemble)" if self.is_multimodal else "RandomForest (Acoustic Only)"
            version = self.metadata.get("model_version", "unknown") if self.metadata else "local"
            logger.info("Loaded pretrained deception model [version: %s, mode: %s] from %s", version, mode, resolved_path)
        except ModelLoaderError as e:
            logger.error("Pretrained model resolution failed: %s", e)
            self.model = None
        except Exception as e:
            logger.exception("Failed to load model from %s: %s", self.model_path or self.explicit_model_path, e)
            self.model = None

    def is_ready(self) -> bool:
        return self.model is not None

    def reload(self) -> None:
        """Re-read the model file from disk or cache."""
        self._load()

    def predict_raw(self, feature_vector: np.ndarray) -> Tuple[str, float, float]:
        """Run inference given a pre-computed feature vector (any dimension)."""
        if self.model is None:
            raise ModelNotLoadedError(
                "Deception model is not loaded. Pretrained model could not be loaded from cache. "
                "Ensure internet access is available for the initial download, run "
                "'python -m app.model_loader' to inspect/cache the model, or set "
                "TRUSIGHT_MODEL_PATH to point to a valid deception_model.pkl file."
            )
        x = feature_vector.reshape(1, -1)
        proba = self.model.predict_proba(x)[0]

        # Safety Guard: Ensure label index 1 maps to Deceptive (class=1) and index 0 maps to Truthful (class=0)
        classes = getattr(self.model, "classes_", [0, 1])
        if hasattr(self.model, "named_steps"):
            last_step = list(self.model.named_steps.values())[-1]
            classes = getattr(last_step, "classes_", [0, 1])

        if len(classes) >= 2 and classes[0] == 1 and classes[1] == 0:
            # Handle rare case where class order is inverted in model metadata
            deception_score = float(proba[0])
        else:
            deception_score = float(proba[1]) if len(proba) > 1 else float(proba[0])

        verdict = "Deceptive" if deception_score >= DECEPTION_THRESHOLD else "Truthful"
        confidence = float(max(proba))
        return verdict, deception_score, confidence

    def predict(self, feature_vector: np.ndarray) -> Tuple[str, float, float]:
        """
        Run inference on a single flattened feature vector.

        Returns:
            verdict: "Truthful" | "Deceptive"
            score: probability of deception, in [0, 1]
            confidence: model's confidence in the chosen verdict, in [0, 1]
        """
        return self.predict_raw(feature_vector)
