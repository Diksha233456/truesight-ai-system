"""
tests/test_honest_explanations.py

Unit tests for Stage 4: Honest, Evidence-Based Explanations:
1. Separation into:
   - Measured Acoustic Observations (only when reliable)
   - Audio Quality Warnings (clipping, low SNR, short phonation)
   - Prediction Context (uncalibrated probability vs verified benchmark)
   - Scientific Limitations / Caveats (stress vs deception disclaimer)
2. Inconclusive handling for low voiced frames, glottal cycles, or unvoiced speech.
3. Verification that uncalibrated model probabilities are explicitly qualified.
4. Backward compatibility of generate_explanations and AnalyzeResponse.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.model import VoiceFeatures, StructuredExplanation, AnalyzeResponse
from app.explanation import (
    generate_structured_explanation,
    generate_explanations,
    PITCH_STD_HIGH,
    PAUSE_RATIO_HIGH,
    JITTER_HIGH,
    SHIMMER_HIGH,
)


def create_dummy_features(**overrides) -> VoiceFeatures:
    """Helper creating a baseline valid VoiceFeatures object."""
    default_kwargs = {
        "pitch_mean": 175.0,
        "pitch_std": 22.0,
        "pitch_min": 110.0,
        "pitch_max": 250.0,
        "jitter_local": 0.012,
        "shimmer_local": 0.045,
        "speech_tempo": 2.2,
        "pause_ratio": 0.18,
        "pause_count": 4,
        "silence_duration": 1.5,
        "rms_energy_mean": 0.08,
        "rms_energy_std": 0.02,
        "zero_crossing_rate": 0.07,
        "spectral_centroid": 1800.0,
        "spectral_bandwidth": 1600.0,
        "spectral_rolloff": 3400.0,
        "mfcc_mean": [0.0] * 13,
        "mfcc_std": [0.0] * 13,
        "is_pitch_reliable": True,
        "is_jitter_reliable": True,
        "is_shimmer_reliable": True,
        "voiced_ratio": 0.65,
    }
    default_kwargs.update(overrides)
    return VoiceFeatures(**default_kwargs)


class TestHonestExplanations(unittest.TestCase):

    def test_structured_explanation_categories(self):
        """StructuredExplanation must return 4 distinct categories."""
        feats = create_dummy_features()
        structured = generate_structured_explanation(
            features=feats,
            score=0.42,
            confidence=0.58,
            quality_warnings=["Test clipping warning."],
        )

        self.assertIsInstance(structured, StructuredExplanation)
        self.assertIsInstance(structured.acoustic_observations, list)
        self.assertIsInstance(structured.quality_warnings, list)
        self.assertIsInstance(structured.prediction_context, str)
        self.assertIsInstance(structured.limitations, list)

        # Observations present
        self.assertTrue(len(structured.acoustic_observations) > 0)
        # Quality warning captured
        self.assertIn("Test clipping warning.", structured.quality_warnings)
        # Prediction context explicitly notes uncalibrated model probability
        self.assertIn("uncalibrated", structured.prediction_context.lower())
        # Limitations explicitly note acoustic indicators do NOT prove deception
        self.assertTrue(
            any("do not prove deception" in lim.lower() for lim in structured.limitations)
        )

    def test_unreliable_jitter_shimmer_routes_to_warnings(self):
        """When jitter or shimmer are marked unreliable, they must be flagged in warnings, not asserted as instability."""
        feats = create_dummy_features(
            jitter_local=0.08,
            shimmer_local=0.15,
            is_jitter_reliable=False,
            is_shimmer_reliable=False,
        )

        structured = generate_structured_explanation(
            features=feats,
            score=0.65,
            confidence=0.65,
        )

        # Warnings should record the measurement limitations
        self.assertTrue(
            any("jitter measurement is inconclusive" in w.lower() for w in structured.quality_warnings),
            f"Expected inconclusive jitter in {structured.quality_warnings}",
        )
        self.assertTrue(
            any("shimmer measurement is inconclusive" in w.lower() for w in structured.quality_warnings),
            f"Expected inconclusive shimmer in {structured.quality_warnings}",
        )

        # Observations must NOT claim voice instability
        self.assertFalse(
            any("voice instability" in obs.lower() for obs in structured.acoustic_observations)
        )

    def test_uncalibrated_probability_qualifier(self):
        """Prediction context must clearly identify the score as an uncalibrated probability."""
        feats = create_dummy_features()
        structured = generate_structured_explanation(features=feats, score=0.88, confidence=0.88)
        self.assertIn("88.0%", structured.prediction_context)
        self.assertIn("uncalibrated", structured.prediction_context.lower())

    def test_legacy_explanation_format_backward_compatibility(self):
        """Legacy generate_explanations returns list of 2-5 strings ending with the scientific caveat."""
        feats = create_dummy_features(pitch_std=PITCH_STD_HIGH + 5.0)
        notes = generate_explanations(feats, score=0.7)

        self.assertIsInstance(notes, list)
        self.assertGreaterEqual(len(notes), 2)
        self.assertLessEqual(len(notes), 5)
        # Scientific disclaimer must be present
        self.assertTrue(
            any("not definitive deception" in n.lower() for n in notes),
            "Caveat must be included in legacy explanations",
        )

    def test_analyze_response_schema_compatibility(self):
        """AnalyzeResponse must accept both legacy and extended fields without validation errors."""
        feats = create_dummy_features()
        structured = generate_structured_explanation(feats, 0.45, 0.55)

        resp = AnalyzeResponse(
            verdict="Truthful",
            score=0.45,
            confidence=0.55,
            explanation=["Normal pitch.", "Note: Caveat."],
            features=feats,
            structured_explanation=structured,
            quality_warnings=["Mic clipping."],
        )

        self.assertEqual(resp.verdict, "Truthful")
        self.assertEqual(resp.score, 0.45)
        self.assertIsNotNone(resp.structured_explanation)
        self.assertEqual(resp.quality_warnings, ["Mic clipping."])


if __name__ == "__main__":
    unittest.main()
