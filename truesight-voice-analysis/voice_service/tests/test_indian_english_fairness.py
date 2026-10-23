"""
tests/test_indian_english_fairness.py

Regression test suite for Stage 8: Indian-English Fairness and Explanation Fixes.

Verifies:
1. High speech tempo alone (e.g. 7.5 onsets/s, typical of syllable-timed Indian English)
   does not generate stress, nervousness, urgency, or deception claims.
2. High pitch variability alone (e.g. 50+ Hz F0 std, typical of expressive prosody)
   does not generate emotional activation, stress, or deception claims.
3. High jitter/shimmer on connected speech does not generate vocal distress or deception claims.
4. Unreliable features (short phonation, insufficient glottal pulses) produce explicit
   inconclusive caveats rather than confident diagnostic interpretations.
5. Upload and live-analysis response schemas remain 100% backward compatible.
6. The explanation system explicitly disclaims that regional accents or dialects identify deception.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.model import VoiceFeatures, StructuredExplanation, AnalyzeResponse
from app.explanation import (
    generate_structured_explanation,
    generate_explanations,
    TEMPO_FAST,
    PITCH_STD_HIGH,
    JITTER_HIGH,
    SHIMMER_HIGH,
)


def create_baseline_features(**overrides) -> VoiceFeatures:
    """Helper creating a baseline valid VoiceFeatures object."""
    default_kwargs = {
        "pitch_mean": 125.0,
        "pitch_std": 22.0,
        "pitch_min": 85.0,
        "pitch_max": 210.0,
        "jitter_local": 0.012,
        "shimmer_local": 0.045,
        "speech_tempo": 2.5,
        "pause_ratio": 0.18,
        "pause_count": 4,
        "silence_duration": 1.2,
        "rms_energy_mean": 0.08,
        "rms_energy_std": 0.02,
        "zero_crossing_rate": 0.08,
        "spectral_centroid": 1500.0,
        "spectral_bandwidth": 1400.0,
        "spectral_rolloff": 3000.0,
        "mfcc_mean": [0.0] * 13,
        "mfcc_std": [0.0] * 13,
        "is_pitch_reliable": True,
        "is_jitter_reliable": True,
        "is_shimmer_reliable": True,
        "voiced_ratio": 0.65,
    }
    default_kwargs.update(overrides)
    return VoiceFeatures(**default_kwargs)


class TestIndianEnglishFairness(unittest.TestCase):

    def test_high_speech_tempo_alone_does_not_claim_stress_or_deception(self):
        """
        Syllable-timed Indian English often registers 6.5-8.0 onsets/sec.
        High tempo must NOT produce assertions of urgency, stress, nervousness, or deception.
        """
        feats = create_baseline_features(speech_tempo=7.5)
        structured = generate_structured_explanation(features=feats, score=0.45, confidence=0.55)
        legacy = generate_explanations(features=feats, score=0.45)

        # Check structured observations
        tempo_obs = [obs for obs in structured.acoustic_observations if "onset rate" in obs.lower()]
        self.assertTrue(len(tempo_obs) > 0, "Expected onset rate measurement in observations")
        tempo_text = tempo_obs[0].lower()

        # Must describe onset rate objectively
        self.assertIn("7.5 onsets/sec", tempo_text)
        # Must NOT assert psychological tension
        self.assertNotIn("urgency", tempo_text.replace("does not indicate urgency", ""))
        self.assertNotIn("heightened arousal", tempo_text)
        self.assertNotIn("nervous", tempo_text)
        self.assertNotIn("lie", tempo_text)
        self.assertNotIn("deceptive", tempo_text.replace("does not indicate urgency, stress, or deception", ""))

        # Check legacy notes
        legacy_tempo = [n for n in legacy if "onset rate" in n.lower() or "tempo" in n.lower()]
        self.assertTrue(len(legacy_tempo) > 0)
        for note in legacy_tempo:
            self.assertNotIn("faster than expected", note.lower())
            self.assertNotIn("stress", note.lower())
            self.assertNotIn("deception", note.lower())

    def test_high_pitch_variability_alone_does_not_claim_stress_or_deception(self):
        """
        Expressive sentence intonation in Indian English naturally exceeds 35 Hz F0 std.
        Must NOT assert emotional activation, stress, or deception.
        """
        feats = create_baseline_features(pitch_std=52.0, pitch_mean=130.0)
        structured = generate_structured_explanation(features=feats, score=0.40, confidence=0.60)
        legacy = generate_explanations(features=feats, score=0.40)

        # Check structured observations
        pitch_obs = [obs for obs in structured.acoustic_observations if "fundamental frequency" in obs.lower()]
        self.assertTrue(len(pitch_obs) > 0)
        pitch_text = pitch_obs[0].lower()

        self.assertIn("52.0 hz", pitch_text)
        self.assertNotIn("emotional activation", pitch_text)
        self.assertNotIn("stress", pitch_text.replace("does not indicate stress or deception", ""))
        self.assertNotIn("deception", pitch_text.replace("does not indicate stress or deception", ""))

        # Check legacy notes
        legacy_pitch = [n for n in legacy if "pitch" in n.lower()]
        self.assertTrue(len(legacy_pitch) > 0)
        for note in legacy_pitch:
            self.assertNotIn("emotional", note.lower())
            self.assertNotIn("stress", note.lower())

    def test_high_jitter_shimmer_alone_does_not_claim_stress_or_deception(self):
        """
        Connected speech perturbation exceeds clinical sustained-vowel norms without indicating pathology or deceit.
        """
        feats = create_baseline_features(
            jitter_local=0.038,
            shimmer_local=0.095,
            is_jitter_reliable=True,
            is_shimmer_reliable=True,
        )
        structured = generate_structured_explanation(features=feats, score=0.50, confidence=0.50)
        legacy = generate_explanations(features=feats, score=0.50)

        # Observations must note connected speech context and avoid asserting vocal distress
        obs_text = " ".join(structured.acoustic_observations).lower()
        self.assertIn("connected speech", obs_text)
        self.assertNotIn("vocal instability", obs_text)
        self.assertNotIn("physiological stress", obs_text)
        self.assertNotIn("micro-tremors", obs_text)

        # Legacy notes
        for note in legacy:
            self.assertNotIn("vocal frequency instability", note.lower())
            self.assertNotIn("vocal amplitude instability", note.lower())

    def test_unreliable_features_produce_inconclusive_caveats(self):
        """
        Short recordings or unvoiced speech must produce explicit inconclusive warnings.
        """
        feats = create_baseline_features(
            pitch_mean=0.0,
            pitch_std=0.0,
            is_pitch_reliable=False,
            is_jitter_reliable=False,
            is_shimmer_reliable=False,
        )
        structured = generate_structured_explanation(features=feats, score=0.50, confidence=0.50)
        legacy = generate_explanations(features=feats, score=0.50)

        warnings_text = " ".join(structured.quality_warnings).lower()
        self.assertIn("inconclusive", warnings_text)
        self.assertTrue(any("inconclusive" in n.lower() for n in legacy))

    def test_accent_limitations_explicitly_included(self):
        """
        Structured explanation limitations must explicitly state that regional accents,
        dialects (including Indian English), and prosody do NOT prove deception.
        """
        feats = create_baseline_features()
        structured = generate_structured_explanation(features=feats, score=0.45, confidence=0.55)

        limitations_text = " ".join(structured.limitations).lower()
        self.assertIn("indian-accented english", limitations_text)
        self.assertIn("do not prove deception", limitations_text)
        self.assertIn("accent", limitations_text)

    def test_api_compatibility_and_response_schema(self):
        """
        AnalyzeResponse must maintain 100% schema compatibility with legacy clients and frontend.
        """
        feats = create_baseline_features(speech_tempo=7.2, pitch_std=44.0)
        structured = generate_structured_explanation(feats, 0.35, 0.65)
        legacy = generate_explanations(feats, 0.35)

        resp = AnalyzeResponse(
            verdict="Truthful",
            score=0.35,
            confidence=0.65,
            explanation=legacy,
            features=feats,
            structured_explanation=structured,
            quality_warnings=None,
        )

        self.assertEqual(resp.verdict, "Truthful")
        self.assertIsInstance(resp.explanation, list)
        self.assertGreaterEqual(len(resp.explanation), 2)
        self.assertLessEqual(len(resp.explanation), 5)
        self.assertIsNotNone(resp.structured_explanation)
        self.assertIsInstance(resp.structured_explanation.acoustic_observations, list)
        self.assertIsInstance(resp.structured_explanation.limitations, list)


if __name__ == "__main__":
    unittest.main()
