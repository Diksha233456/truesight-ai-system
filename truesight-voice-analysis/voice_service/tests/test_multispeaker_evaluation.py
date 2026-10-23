"""
tests/test_multispeaker_evaluation.py

Unit and integration tests for Stage 9: Multi-Speaker Indian-English Evaluation:
1. Manifest integrity: verifies multiple unique speakers/sessions (45+) and Apache-2.0 license.
2. Duration guardrail: confirms clips under 0.5s are rejected with AudioValidationError.
3. Speaker pitch diversity: verifies real speech extracts across both male and female pitch bands.
4. Explanation safeguards: ensures Stage 8 safeguards prevent stress/deception claims across real multi-speaker recordings.
"""

import os
import csv
import unittest

from app.utils import load_audio_from_bytes, AudioValidationError
from app.feature_extractor import FeatureExtractor
from app.explanation import generate_structured_explanation, generate_explanations

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "external", "indian_english_multispeaker")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")
AUDIO_DIR = os.path.join(DATASET_DIR, "audio")


class TestMultiSpeakerIndianEnglish(unittest.TestCase):

    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_manifest_integrity_and_multiple_speakers(self):
        """Manifest must exist and contain at least 40 distinct speaker/session IDs."""
        self.assertTrue(os.path.exists(MANIFEST_PATH), f"Manifest missing at {MANIFEST_PATH}")
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.assertGreaterEqual(len(rows), 45, "Expected at least 45 audio recordings in manifest")
        unique_speakers = {r["speaker_id"] for r in rows}
        self.assertGreaterEqual(len(unique_speakers), 40, "Dataset must represent multiple distinct speakers")

        # License must be Apache-2.0
        for r in rows:
            self.assertEqual(r["license"], "Apache-2.0")

    def test_audio_guardrail_rejects_sub_half_second_clip(self):
        """Audio clips < 0.5s must be rejected by AudioValidationError."""
        short_file = os.path.join(AUDIO_DIR, "281474976883749_f2901_chunk_0.wav")
        if os.path.exists(short_file):
            with open(short_file, "rb") as f:
                content = f.read()
            with self.assertRaises(AudioValidationError) as ctx:
                load_audio_from_bytes(content, "281474976883749_f2901_chunk_0.wav")
            self.assertIn("too short", str(ctx.exception).lower())

    def test_pitch_diversity_across_speakers(self):
        """Dataset must contain both male-range (<160 Hz) and female-range (>=160 Hz) speakers."""
        # Test female-range clip: e.g. 281474976879627_f1171_chunk_0.wav
        f_clip = os.path.join(AUDIO_DIR, "281474976879627_f1171_chunk_0.wav")
        # Test male-range clip: e.g. 281474976879629_f1172_chunk_0.wav
        m_clip = os.path.join(AUDIO_DIR, "281474976879629_f1172_chunk_0.wav")

        self.assertTrue(os.path.exists(f_clip), f"Clip missing: {f_clip}")
        self.assertTrue(os.path.exists(m_clip), f"Clip missing: {m_clip}")

        with open(f_clip, "rb") as f:
            aud_f = load_audio_from_bytes(f.read(), "female_sample.wav")
        feats_f, vec_f = self.extractor.extract(aud_f)
        self.assertGreaterEqual(feats_f.pitch_mean, 160.0, "Expected female-range fundamental frequency")

        with open(m_clip, "rb") as f:
            aud_m = load_audio_from_bytes(f.read(), "male_sample.wav")
        feats_m, vec_m = self.extractor.extract(aud_m)
        self.assertLess(feats_m.pitch_mean, 160.0, "Expected male-range fundamental frequency")

    def test_explanation_safeguards_on_real_recordings(self):
        """Real Indian English recordings must not trigger claims of urgency, stress, or deception."""
        test_files = [
            "281474976879627_f1171_chunk_0.wav",
            "281474976879629_f1172_chunk_0.wav",
            "281474976883724_f2612_chunk_0.wav",
        ]
        for fn in test_files:
            fp = os.path.join(AUDIO_DIR, fn)
            if not os.path.exists(fp):
                continue
            with open(fp, "rb") as f:
                loaded = load_audio_from_bytes(f.read(), fn)
            features, _ = self.extractor.extract(loaded)

            structured = generate_structured_explanation(features, score=0.45, confidence=0.55)
            legacy = generate_explanations(features, score=0.45)

            all_text = " ".join(structured.acoustic_observations + legacy).lower()
            # Clean out negative disclaimers
            clean_text = all_text
            for dis in [
                "does not indicate urgency, stress, or deception",
                "does not indicate stress or deception",
                "do not indicate vocal distress or deception",
                "does not indicate vocal distress or deception",
                "does not indicate deception",
                "not definitive deception",
            ]:
                clean_text = clean_text.replace(dis, "")

            self.assertNotIn("urgency", clean_text)
            self.assertNotIn("stress", clean_text)
            self.assertNotIn("nervous", clean_text)
            self.assertNotIn("hesitation", clean_text)
            self.assertNotIn("deceptive", clean_text)
            self.assertNotIn("lying", clean_text)

            # Limitations must explicitly contain regional accent disclaimer
            lim_text = " ".join(structured.limitations).lower()
            self.assertIn("indian-accented english", lim_text)
            self.assertIn("accent", lim_text)


if __name__ == "__main__":
    unittest.main()
