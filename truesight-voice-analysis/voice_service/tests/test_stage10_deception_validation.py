"""
tests/test_stage10_deception_validation.py

Unit and regression tests for Stage 10: Deception Model Validation:
1. Uncalibrated probability disclosure: verifies model scores are explicitly described as uncalibrated probabilities.
2. Non-inference from accent: verifies limitations disclaim accent/prosody as deception evidence.
3. Production model integrity: verifies production model loads without modifications, DECEPTION_THRESHOLD remains 0.5.
4. Response schema preservation: verifies AnalyzeResponse, StructuredExplanation, VoiceFeatures remain intact.
"""

import os
import unittest
import numpy as np

from app.predictor import Predictor, DECEPTION_THRESHOLD, LEGACY_FEATURE_DIM
from app.model import VoiceFeatures, StructuredExplanation, AnalyzeResponse
from app.explanation import generate_structured_explanation, generate_explanations


class TestStage10DeceptionValidation(unittest.TestCase):

    def setUp(self):
        self.predictor = Predictor()

    def test_production_model_loads_and_unmodified(self):
        """Production model must load successfully with default threshold 0.5."""
        self.assertTrue(self.predictor.is_ready(), "Model must be ready")
        self.assertEqual(DECEPTION_THRESHOLD, 0.5, "Decision threshold must remain 0.5")

        # Test dummy 42-dim vector prediction
        dummy_vec = np.zeros(LEGACY_FEATURE_DIM, dtype=np.float32)
        verdict, score, conf = self.predictor.predict(dummy_vec)
        self.assertIn(verdict, ["Truthful", "Deceptive"])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_uncalibrated_probability_disclosure(self):
        """Structured explanation must explicitly describe model output as uncalibrated."""
        feats = VoiceFeatures(
            pitch_mean=185.0, pitch_std=32.0, pitch_min=110.0, pitch_max=260.0,
            jitter_local=0.02, shimmer_local=0.08, speech_tempo=7.2, pause_ratio=0.15,
            pause_count=3, silence_duration=0.8, rms_energy_mean=0.08, rms_energy_std=0.02,
            zero_crossing_rate=0.08, spectral_centroid=1650.0, spectral_bandwidth=1550.0,
            spectral_rolloff=3200.0, mfcc_mean=[0.0]*13, mfcc_std=[0.0]*13
        )
        structured = generate_structured_explanation(features=feats, score=0.62, confidence=0.62)

        pred_context = structured.prediction_context.lower()
        self.assertIn("uncalibrated", pred_context)
        self.assertIn("probability", pred_context)
        self.assertIn("raw tree ensemble", pred_context)

    def test_no_deception_claims_from_accent_or_prosody(self):
        """Limitations must explicitly state that regional accents (including Indian English) do not indicate deception."""
        feats = VoiceFeatures(
            pitch_mean=185.0, pitch_std=32.0, pitch_min=110.0, pitch_max=260.0,
            jitter_local=0.02, shimmer_local=0.08, speech_tempo=7.2, pause_ratio=0.15,
            pause_count=3, silence_duration=0.8, rms_energy_mean=0.08, rms_energy_std=0.02,
            zero_crossing_rate=0.08, spectral_centroid=1650.0, spectral_bandwidth=1550.0,
            spectral_rolloff=3200.0, mfcc_mean=[0.0]*13, mfcc_std=[0.0]*13
        )
        structured = generate_structured_explanation(features=feats, score=0.62, confidence=0.62)
        legacy = generate_explanations(features=feats, score=0.62)

        lim_text = " ".join(structured.limitations).lower()
        self.assertIn("indian-accented english", lim_text)
        self.assertIn("do not prove deception", lim_text)
        self.assertIn("accent", lim_text)

        # Legacy notes caveat
        self.assertTrue(any("not definitive deception" in n.lower() for n in legacy))

    def test_analyze_response_schema_compatibility(self):
        """AnalyzeResponse contract must be fully preserved with legacy and structured fields."""
        feats = VoiceFeatures(
            pitch_mean=185.0, pitch_std=32.0, pitch_min=110.0, pitch_max=260.0,
            jitter_local=0.02, shimmer_local=0.08, speech_tempo=7.2, pause_ratio=0.15,
            pause_count=3, silence_duration=0.8, rms_energy_mean=0.08, rms_energy_std=0.02,
            zero_crossing_rate=0.08, spectral_centroid=1650.0, spectral_bandwidth=1550.0,
            spectral_rolloff=3200.0, mfcc_mean=[0.0]*13, mfcc_std=[0.0]*13
        )
        structured = generate_structured_explanation(feats, 0.48, 0.52)
        legacy = generate_explanations(feats, 0.48)

        resp = AnalyzeResponse(
            verdict="Truthful",
            score=0.48,
            confidence=0.52,
            explanation=legacy,
            features=feats,
            structured_explanation=structured,
            quality_warnings=None,
        )

        self.assertEqual(resp.verdict, "Truthful")
        self.assertEqual(resp.score, 0.48)
        self.assertIsInstance(resp.explanation, list)
        self.assertIsNotNone(resp.structured_explanation)


if __name__ == "__main__":
    unittest.main()
