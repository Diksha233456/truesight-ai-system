"""
tests/test_indic_evaluation_pipeline.py

Unit test suite for the Indian-Language Speech Acoustic Robustness Pipeline.
Runs entirely offline using small synthetic audio fixtures generated in-memory.
Does NOT require network access or downloading any external dataset.
"""

import io
import os
import shutil
import tempfile
import unittest

import numpy as np
import soundfile as sf

from evaluate_indic_acoustic_robustness import (
    IndicAcousticEvaluator,
    generate_indic_evaluation_report,
)
from app.utils import TARGET_SR
from app.feature_extractor import FEATURE_ORDER


def synthesize_synthetic_speech_wave(
    duration_s: float = 2.0,
    f0: float = 180.0,
    sr: int = TARGET_SR,
) -> bytes:
    """
    Synthesize an in-memory WAV byte stream with fundamental frequency and harmonics
    to simulate phonation for testing acoustic feature extraction offline.
    """
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    # Fundamental + 3 harmonics with slight modulation
    waveform = (
        0.5 * np.sin(2 * np.pi * f0 * t)
        + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.12 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    # Add an amplitude envelope
    envelope = np.ones_like(waveform)
    fade_len = int(sr * 0.05)
    if len(envelope) > 2 * fade_len:
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
    waveform = waveform * envelope

    buf = io.BytesIO()
    sf.write(buf, waveform.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TestIndicEvaluationPipeline(unittest.TestCase):

    def setUp(self):
        self.evaluator = IndicAcousticEvaluator()
        self.temp_dir = tempfile.mkdtemp(prefix="test_indic_pipe_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_process_single_audio_valid(self):
        """Verify robust 42-feature extraction on valid speech-like synthetic audio."""
        wav_bytes = synthesize_synthetic_speech_wave(duration_s=1.8, f0=210.0)
        res = self.evaluator.process_single_audio(
            wav_bytes, "hindi_sample.wav", language="Hindi", speaker_id="SPK_01"
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["language"], "Hindi")
        self.assertEqual(res["speaker_id"], "SPK_01")
        self.assertEqual(res["vector_dimension"], len(FEATURE_ORDER))
        self.assertTrue(res["has_correct_dimension"])
        self.assertTrue(res["is_vector_finite"])
        self.assertIn("key_acoustic_measurements", res)
        self.assertIn("reliability_flags", res)

        # Measurements should be finite numbers
        measurements = res["key_acoustic_measurements"]
        self.assertGreater(measurements["pitch_mean_hz"], 100.0)
        self.assertLess(measurements["pitch_mean_hz"], 350.0)

    def test_process_single_audio_silence_rejected(self):
        """Silent audio should be caught by quality validation with ValidationError."""
        silence = np.zeros(TARGET_SR * 2, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, silence, TARGET_SR, format="WAV", subtype="PCM_16")

        res = self.evaluator.process_single_audio(buf.getvalue(), "silent.wav", language="Kannada")
        self.assertFalse(res["success"])
        self.assertEqual(res["error_type"], "ValidationError")
        self.assertIn("silent", res["error_message"].lower())

    def test_process_single_audio_too_short_rejected(self):
        """Audio under minimum duration (0.5s) must be cleanly caught."""
        short_wave = np.random.uniform(-0.1, 0.1, int(TARGET_SR * 0.2)).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, short_wave, TARGET_SR, format="WAV", subtype="PCM_16")

        res = self.evaluator.process_single_audio(buf.getvalue(), "too_short.wav", language="Telugu")
        self.assertFalse(res["success"])
        self.assertEqual(res["error_type"], "ValidationError")
        self.assertIn("too short", res["error_message"].lower())

    def test_evaluate_manifest_offline(self):
        """Test multi-language manifest evaluation and aggregation across Indian languages."""
        # Create small test files in temp directory
        clips_info = [
            ("hindi_f1.wav", "Hindi", "SPK_H1", 200.0, 1.5),
            ("hindi_m1.wav", "Hindi", "SPK_H2", 120.0, 1.5),
            ("kannada_f1.wav", "Kannada", "SPK_K1", 220.0, 2.0),
            ("tamil_m1.wav", "Tamil", "SPK_T1", 130.0, 1.8),
        ]

        manifest_rows = []
        for fname, lang, spk, f0, dur in clips_info:
            path = os.path.join(self.temp_dir, fname)
            wav_bytes = synthesize_synthetic_speech_wave(duration_s=dur, f0=f0)
            with open(path, "wb") as f:
                f.write(wav_bytes)
            manifest_rows.append({
                "filename": fname,
                "audio_path": path,
                "language": lang,
                "speaker_id": spk,
            })

        summary = self.evaluator.evaluate_manifest(manifest_rows)

        self.assertEqual(summary["total_samples_attempted"], 4)
        self.assertEqual(summary["successfully_processed"], 4)
        self.assertEqual(summary["pipeline_extraction_failures"], 0)
        self.assertEqual(summary["robustness_success_rate"], 1.0)

        # Check language breakdown
        lang_groups = summary["language_group_summaries"]
        self.assertIn("Hindi", lang_groups)
        self.assertIn("Kannada", lang_groups)
        self.assertIn("Tamil", lang_groups)
        self.assertEqual(lang_groups["Hindi"]["evaluated_samples"], 2)
        self.assertEqual(lang_groups["Kannada"]["evaluated_samples"], 1)
        self.assertEqual(lang_groups["Tamil"]["evaluated_samples"], 1)

        # Report generation test
        json_path = os.path.join(self.temp_dir, "report.json")
        md_path = os.path.join(self.temp_dir, "report.md")
        generate_indic_evaluation_report(summary, json_path, md_path)

        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(md_path))
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        self.assertIn("Acoustic Feature Extraction Robustness", md_content)
        self.assertIn("Hindi", md_content)
        self.assertIn("Kannada", md_content)

    def test_no_deception_labels_assigned(self):
        """Ensure no deception verdicts or truth scores are assigned to speech samples."""
        wav_bytes = synthesize_synthetic_speech_wave(duration_s=1.5, f0=180.0)
        res = self.evaluator.process_single_audio(wav_bytes, "test.wav", language="Hindi")

        self.assertNotIn("verdict", res)
        self.assertNotIn("score", res)
        self.assertNotIn("deception_score", res)
        self.assertNotIn("is_truthful", res)
        self.assertNotIn("is_deceptive", res)


if __name__ == "__main__":
    unittest.main()
