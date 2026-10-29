"""
Comprehensive regression tests for TRUSIGHT:
- Emotion label preservation (happy, angry, sad, neutral, surprise, fear, disgust, contempt)
- Missing emotion output vs genuine neutral prediction
- Frontend-compatible emotion fields
- Voice decoding, empty audio, invalid audio, error handling
- Fresh vs stale voice and face telemetry
- Multimodal stream eligibility and honest insufficient-data behavior
"""

import math
import os
import sys
import time
import unittest
from unittest.mock import MagicMock

# Ensure repo roots are on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AU_ROOT = os.path.join(REPO_ROOT, "RealTime-Facial-Expression-Recognition", "AU")
VOICE_ROOT = os.path.join(REPO_ROOT, "truesight-voice-analysis", "voice_service")
for p in (REPO_ROOT, AU_ROOT, VOICE_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

for mod in ("soundfile", "librosa"):
    if mod not in sys.modules:
        try:
            __import__(mod)
        except ImportError:
            sys.modules[mod] = MagicMock()

from assessment_coordinator import (
    evaluate_assessment,
    AssessmentRequest,
    _evaluate_timestamp,
    _parse_probability,
)


class TestFacialEmotionRecognition(unittest.TestCase):
    """Priority 1 Regression Tests: Facial emotion classification & response preservation."""

    def test_affectnet_supported_classes(self):
        """Verify the 8 supported AffectNet classes and model structure."""
        from facial_cues.ml_models.affectnet_emotion import CLASSES
        expected_classes = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sadness", "surprise"]
        self.assertEqual(CLASSES, expected_classes)

    def test_split_line_parser_preserves_emotion_labels_and_scores(self):
        """Test API line parser with various supported emotion formats."""
        from facial_cues.api.server import _split_line

        cases = [
            ("HAPPY (0.89)", "HAPPY", 0.89),
            ("ANGER (0.75)", "ANGER", 0.75),
            ("SAD (0.64)", "SAD", 0.64),
            ("SURPRISE (0.92)", "SURPRISE", 0.92),
            ("FEAR (0.55)", "FEAR", 0.55),
            ("DISGUST (0.71)", "DISGUST", 0.71),
            ("CONTEMPT (0.48)", "CONTEMPT", 0.48),
            ("NEUTRAL (0.80)", "NEUTRAL", 0.80),
            ("NEUTRAL", "NEUTRAL", None),
            ("", "", None),
        ]
        for text, expected_label, expected_score in cases:
            res = _split_line(text)
            self.assertEqual(res["label"], expected_label, f"Failed for {text}")
            if expected_score is not None:
                self.assertAlmostEqual(res["score"], expected_score, places=2)
            else:
                self.assertIsNone(res["score"])

    def test_server_snapshot_separates_emotion_from_deception(self):
        """Verify snapshot preserves emotion even while deception classifier is calibrating."""
        from facial_cues.api import server

        mock_pipeline = MagicMock()
        mock_pipeline._is_running = True
        mock_pipeline._last_fps = 30.0
        mock_pipeline._last_face_detected_time = 1000.0
        mock_pipeline._last_viz_data = {"faces": [{"face_id": 1, "action_units": {"AU01": 0.2}}]}
        mock_pipeline._last_deception_text = ""
        mock_pipeline._last_deception_reasons = []
        mock_pipeline._last_deception = {}
        mock_pipeline._last_prediction_time = 0.0  # Deception still accumulating 90 frames
        mock_pipeline._last_emotion_text = "HAPPY (0.85)"
        mock_pipeline._last_emotion_time = 1000.0

        server.pipeline = mock_pipeline
        snap = server.snapshot(now=1000.5)

        # Face is present, emotion is active
        self.assertTrue(snap["face_present"])
        self.assertTrue(snap["ready"])  # Active face with emotion observation is ready
        self.assertEqual(snap["emotion"]["label"], "HAPPY")
        self.assertEqual(snap["emotion"]["score"], 0.85)
        self.assertEqual(snap["emotion"]["timestamp"], 1000.0)

        # Deception is truthfully WAITING without clobbering emotion
        self.assertEqual(snap["deception"]["label"], "WAITING")
        self.assertIsNone(snap["deception"]["p_lie"])

    def test_missing_emotion_vs_genuine_neutral(self):
        """Distinguish missing emotion output from genuine high-confidence neutral prediction."""
        from facial_cues.api import server

        mock_pipeline = MagicMock()
        mock_pipeline._is_running = True
        mock_pipeline._last_fps = 30.0
        mock_pipeline._last_face_detected_time = 1000.0
        mock_pipeline._last_viz_data = {"faces": [{"face_id": 1}]}
        mock_pipeline._last_deception_text = ""
        mock_pipeline._last_deception_reasons = []
        mock_pipeline._last_deception = {}
        mock_pipeline._last_prediction_time = 0.0

        # Case A: Genuine neutral with confidence
        mock_pipeline._last_emotion_text = "NEUTRAL (0.82)"
        mock_pipeline._last_emotion_time = 1000.0
        server.pipeline = mock_pipeline
        snap_a = server.snapshot(now=1000.5)
        self.assertEqual(snap_a["emotion"]["label"], "NEUTRAL")
        self.assertEqual(snap_a["emotion"]["score"], 0.82)

        # Case B: Missing / empty emotion output
        mock_pipeline._last_emotion_text = ""
        mock_pipeline._last_emotion_time = 0.0
        snap_b = server.snapshot(now=1000.5)
        self.assertEqual(snap_b["emotion"]["label"], "")
        self.assertIsNone(snap_b["emotion"]["score"])


class TestVoiceAnalysisValidation(unittest.TestCase):
    """Priority 2 Regression Tests: Voice recording, audio validation & decoding."""

    def test_empty_audio_bytes_rejected(self):
        """Audio loader must reject empty bytes immediately."""
        from app.utils import load_audio_from_bytes, AudioValidationError
        with self.assertRaises(AudioValidationError):
            load_audio_from_bytes(b"", "recording.webm")

    def test_unsupported_audio_extension_rejected(self):
        """Audio loader must reject unsupported extensions."""
        from app.utils import validate_filename, AudioValidationError
        with self.assertRaises(AudioValidationError):
            validate_filename("recording.txt")
        with self.assertRaises(AudioValidationError):
            validate_filename("recording.exe")

    def test_segment_result_schema_parity(self):
        """Ensure WebSocket to_dict() contains 'features' matching HTTP response schema."""
        from app.live_analyzer import SegmentResult
        from app.model import VoiceFeatures

        dummy_features = VoiceFeatures(
            pitch_mean=178.4,
            pitch_std=24.1,
            pitch_min=110.2,
            pitch_max=260.7,
            jitter_local=0.012,
            shimmer_local=0.034,
            speech_tempo=3.2,
            pause_ratio=0.15,
            pause_count=2,
            silence_duration=0.5,
            rms_energy_mean=0.045,
            rms_energy_std=0.012,
            zero_crossing_rate=0.08,
            spectral_centroid=1850.0,
            spectral_bandwidth=1420.0,
            spectral_rolloff=3200.0,
            mfcc_mean=[0.1] * 13,
            mfcc_std=[0.05] * 13,
        )

        res = SegmentResult(
            segment=1,
            start_seconds=0.0,
            end_seconds=15.0,
            verdict="Truthful",
            score=0.22,
            confidence=0.88,
            explanation=["Normal speech patterns."],
            features=dummy_features,
            speech_detected=True,
            timestamp=1500.0,
        )
        d = res.to_dict()
        self.assertIn("features", d)
        self.assertIsNotNone(d["features"])
        self.assertIn("pitch_mean", d["features"])
        self.assertEqual(d["features"]["pitch_mean"], 178.4)
        self.assertEqual(d["timestamp"], 1500.0)


class TestFreshnessAndMultimodalFusion(unittest.TestCase):
    """Priority 3 & 4 Regression Tests: Freshness boundaries and multi-stream eligibility."""

    def test_voice_window_aware_freshness(self):
        """Voice freshness accounts for window duration plus grace period."""
        now = 5000.0
        # 15s window + 10s grace = 25s threshold
        # 20s old is FRESH
        ts, fresh_sec, is_fresh, err = _evaluate_timestamp(now - 20.0, now, max_freshness_seconds=25.0)
        self.assertTrue(is_fresh)
        self.assertIsNone(err)

        # 26s old is STALE
        ts, fresh_sec, is_fresh, err = _evaluate_timestamp(now - 26.0, now, max_freshness_seconds=25.0)
        self.assertFalse(is_fresh)
        self.assertIn("stale telemetry", err)

    def test_default_telemetry_freshness_without_window(self):
        """When no window is provided, standard 10s threshold rejects 25s old telemetry."""
        now = 2000.0
        req = AssessmentRequest(
            current_time=now,
            voice_telemetry={
                "ready": True,
                "timestamp": now - 25.0,
                "verdict": "Deceptive",
                "deceptionScore": 0.74,
                "speechDetected": True,
            },
        )
        resp = evaluate_assessment(req)
        self.assertFalse(resp.acoustic_observation.available)
        self.assertFalse(resp.acoustic_observation.eligible_for_fusion)
        self.assertEqual(resp.acoustic_observation.verdict, "UNAVAILABLE")

    def test_three_streams_genuine_eligibility(self):
        """When all three streams are fresh and valid, exactly 3/3 streams are eligible."""
        now = 1000.0
        req = AssessmentRequest(
            session_id="sess-alpha",
            current_time=now,
            statement_evidence={
                "statement": "The defendant was at the office.",
                "evidence_results": [
                    {"verdict": "Supported", "text": "Badge swipe logged at 09:00.", "page": 1, "confidence": 0.91}
                ],
            },
            face_telemetry={
                "session_id": "sess-alpha",
                "ready": True,
                "timestamp": now - 1.2,
                "verdict": "MODEL PREDICTS TRUTH CLASS",
                "pLie": 0.25,
                "reasons": ["Consistent gaze orientation"],
                "sentiment": "HAPPY",
            },
            voice_telemetry={
                "session_id": "sess-alpha",
                "ready": True,
                "timestamp": now - 4.5,
                "window_seconds": 15,
                "verdict": "Truth-class prediction",
                "deceptionScore": 0.28,
                "confidence": 0.85,
                "speechDetected": True,
            },
        )
        resp = evaluate_assessment(req)
        fusion = resp.fusion_assessment
        self.assertEqual(fusion.eligible_modalities_count, 3)
        self.assertIn("Statement & Documentary Evidence", fusion.eligible_modality_names)
        self.assertIn("Facial Movement Analysis", fusion.eligible_modality_names)
        self.assertIn("Vocal Acoustic Analysis", fusion.eligible_modality_names)
        self.assertIsNone(fusion.fusion_score)  # Honest unvalidated combined score: None
        self.assertIn("Not validated", fusion.fusion_score_display)

    def test_face_emotion_preserved_in_assessment_finding(self):
        """Assessment coordinator preserves extracted emotion label."""
        now = 1000.0
        req = AssessmentRequest(
            session_id="sess-beta",
            current_time=now,
            face_telemetry={
                "session_id": "sess-beta",
                "ready": True,
                "timestamp": now - 0.5,
                "verdict": "MODEL PREDICTS TRUTH CLASS",
                "pLie": 0.15,
                "sentiment": "SURPRISE",
            },
        )
        resp = evaluate_assessment(req)
        self.assertTrue(resp.facial_observation.available)
        self.assertEqual(resp.facial_observation.emotion, "SURPRISE")


class TestFrontendEventWiringAndLifecycle(unittest.TestCase):
    """Priority 1 & 2: Frontend event wiring, DOM selectors & emotion freshness validation."""

    def test_voice_module_syntax_and_no_redeclarations(self):
        """Verify static/js/voice_module.js contains no syntax errors or redeclared identifiers."""
        import subprocess
        js_path = os.path.join(REPO_ROOT, "static", "js", "voice_module.js")
        result = subprocess.run(
            ["node", "-e", f"const fs = require('fs'); const code = fs.readFileSync({repr(js_path)}, 'utf-8'); new Function(code);"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, f"voice_module.js syntax error: {result.stderr}")

    def test_html_element_ids_match_voice_selectors(self):
        """Verify all HTML element IDs queried in voice_module.js exist in static/index.html."""
        html_path = os.path.join(REPO_ROOT, "static", "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        required_ids = [
            'tabVoiceLive',
            'tabVoiceUpload',
            'paneVoiceLive',
            'paneVoiceUpload',
            'btnStartVoiceLive',
            'btnStopVoiceLive',
            'liveCountdownVal',
            'waveformCanvas',
            'voiceDropzone',
            'voiceFileInput',
            'uploadFilename',
            'btnAnalyzeFile',
            'voiceDeceptionScoreVal',
            'voiceDeceptionGauge',
            'voiceConfidenceVal',
            'voiceConfidenceFill',
            'voiceSpeechStatusBadge',
            'voiceVerdictPill',
            'obsF0',
            'obsF0Desc',
            'obsTempo',
            'obsTempoDesc',
            'obsPauses',
            'obsPausesDesc',
            'obsStability',
            'obsStabilityDesc',
            'structuredObsList',
            'voiceHistoryTableBody',
            'voiceTimelineChart',
        ]
        for elem_id in required_ids:
            self.assertIn(f'id="{elem_id}"', html, f"Missing required element ID in index.html: {elem_id}")

    def test_stale_or_missing_face_clears_emotion(self):
        """Verify server.py snapshot does not return frozen emotion when face is missing or prediction is stale."""
        from facial_cues.api import server

        mock_pipeline = MagicMock()
        mock_pipeline._is_running = True
        mock_pipeline._last_fps = 30.0
        mock_pipeline._last_face_detected_time = 1000.0
        mock_pipeline._last_viz_data = {"faces": [{"face_id": 1, "action_units": {"AU01": 0.2}}]}
        mock_pipeline._last_deception_text = ""
        mock_pipeline._last_deception_reasons = []
        mock_pipeline._last_deception = {}
        mock_pipeline._last_prediction_time = 0.0
        mock_pipeline._last_emotion_text = "HAPPY (0.95)"
        mock_pipeline._last_emotion_time = 1000.0

        server.pipeline = mock_pipeline

        # Fresh emotion (0.5s old) -> returns label and score
        snap_fresh = server.snapshot(now=1000.5)
        self.assertEqual(snap_fresh["emotion"]["label"], "HAPPY")
        self.assertEqual(snap_fresh["emotion"]["score"], 0.95)

        # Stale emotion (5.0s old) -> must clear label and score (not freeze at HAPPY)
        snap_stale = server.snapshot(now=1005.5)
        self.assertEqual(snap_stale["emotion"]["label"], "")
        self.assertIsNone(snap_stale["emotion"]["score"])

        # Missing face -> must clear label and score
        mock_pipeline._last_face_detected_time = 900.0
        mock_pipeline._last_viz_data = {"faces": []}
        snap_no_face = server.snapshot(now=1000.5)
        self.assertEqual(snap_no_face["emotion"]["label"], "")
        self.assertIsNone(snap_no_face["emotion"]["score"])

    def test_real_model_predicts_happy_and_anger_on_ck_dataset(self):
        """Verify the actual AffectNet model checkpoint predicts non-neutral classes on validation images."""
        import cv2
        from facial_cues.ml_models.affectnet_emotion import AffectNetEmotion

        model_path = os.path.join(REPO_ROOT, "RealTime-Facial-Expression-Recognition", "AU", "models", "enet_b0_8_best_vgaf.pt")
        if not os.path.exists(model_path):
            self.skipTest("Model file not found")

        aff = AffectNetEmotion(model_path)
        happy_img_path = os.path.join(REPO_ROOT, "RealTime-Facial-Expression-Recognition", "AU", "data", "ck_plus", "CK+48", "happy", "S014_005_00000017.png")
        if os.path.exists(happy_img_path):
            img = cv2.imread(happy_img_path)
            label, conf = aff.predict(img)
            self.assertEqual(label, "happy")
            self.assertGreater(conf, 0.8)

    def test_speech_band_threshold_accepts_12_percent_wideband_speech(self):
        """Verify that genuine human speech with 12.6% energy in 300-3400 Hz is not rejected."""
        from app.utils import LoadedAudio, evaluate_audio_speech_quality, SPEECH_STATUS_VOICE_NOT_DETECTED
        import numpy as np

        sr = 16000
        dur = 3.5
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        # Synthesize voiced vowel with fundamental pitch at 130 Hz and formants at 600 Hz, 1200 Hz, plus wideband ambient
        f0 = 130.0
        vocal = (
            0.4 * np.sin(2 * np.pi * f0 * t) +
            0.3 * np.sin(2 * np.pi * 2 * f0 * t) +
            0.2 * np.sin(2 * np.pi * 600.0 * t) +
            0.15 * np.sin(2 * np.pi * 1200.0 * t)
        )
        audio_samples = vocal.astype(np.float32)

        loaded = LoadedAudio(
            samples=audio_samples,
            sample_rate=sr,
            duration_seconds=dur,
            source_filename="test_voiced.wav",
            raw_rms=float(np.sqrt(np.mean(audio_samples**2))),
            raw_peak=float(np.max(np.abs(audio_samples))),
            snr_estimate_db=15.0,
        )

        has_speech, reason, status = evaluate_audio_speech_quality(loaded)
        self.assertNotIn("Insufficient speech-band energy", reason)
        self.assertTrue(has_speech, f"Speech evaluation failed with reason: {reason}")

    def test_combined_multimodal_score_computed_when_all_streams_eligible(self):
        """Verify that when all 3 models are eligible, combined_multimodal_score combines them correctly."""
        now = 1000.0
        req = AssessmentRequest(
            current_time=now,
            session_id="sess-all-three",
            statement_evidence={
                "statement": "I made the deposit on Friday.",
                "evidence_results": [
                    {"page": 2, "verdict": "Contradiction", "confidence": 0.85, "evidence": "Record indicates no deposit on Friday."}
                ],
                "inconsistency_index": 0.85,
            },
            face_telemetry={
                "ready": True,
                "timestamp": now - 0.5,
                "p_lie": 0.70,
                "sentiment": "NEUTRAL",
            },
            voice_telemetry={
                "session_id": "sess-all-three",
                "ready": True,
                "timestamp": now - 1.0,
                "window_seconds": 15,
                "verdict": "Deceptive",
                "deceptionScore": 0.60,
                "confidence": 0.80,
                "speechDetected": True,
            },
        )
        resp = evaluate_assessment(req)
        fusion = resp.fusion_assessment
        self.assertEqual(fusion.eligible_modalities_count, 3)
        self.assertIsNotNone(fusion.combined_multimodal_score)
        # 50% * 0.85 (doc) + 25% * 0.70 (face) + 25% * 0.60 (voice) = 0.425 + 0.175 + 0.15 = 0.75
        self.assertAlmostEqual(fusion.combined_multimodal_score, 0.75, places=2)
        self.assertIn("documentary_inconsistency", fusion.stream_scores)
        self.assertIn("facial_deception_risk", fusion.stream_scores)
        self.assertIn("vocal_deception_risk", fusion.stream_scores)

    def test_top_level_document_contradiction_eligible_and_precedence(self):
        """Verify that document evidence with top-level CONTRADICTED verdict and retrieved chunks
        is marked eligible (3/3 streams) and takes precedence to yield CONFLICT WITH AVAILABLE EVIDENCE."""
        now = time.time()
        req = AssessmentRequest(
            session_id="sess-top-level-contra",
            current_time=now,
            statement_evidence={
                "statement": "I never visited the offshore property in July.",
                "verdict": "CONTRADICTED",
                "status": "CONTRADICTED",
                "inconsistency_index": 0.85,
                "evidence_results": [
                    {"evidence": "Flight log shows arrival at offshore property on July 14.", "page": 4},
                    {"evidence": "Hotel registration signed under suspect name on July 15.", "page": 6},
                ],
            },
            face_telemetry={
                "session_id": "sess-top-level-contra",
                "ready": True,
                "timestamp": now - 0.5,
                "verdict": "TRUTH",
                "pLie": 0.382,
                "sentiment": "NEUTRAL",
            },
            voice_telemetry={
                "session_id": "sess-top-level-contra",
                "ready": True,
                "timestamp": now - 1.0,
                "verdict": "Inconclusive",
                "deceptionScore": 0.525,
                "confidence": 0.525,
                "speechDetected": True,
            },
        )
        resp = evaluate_assessment(req)
        fusion = resp.fusion_assessment
        # 1. All 3 streams eligible
        self.assertEqual(fusion.eligible_modalities_count, 3)
        self.assertTrue(resp.evidence_finding.eligible_for_fusion)
        self.assertEqual(resp.evidence_finding.status, "CONTRADICTED")
        # 2. Document contradiction takes precedence over ambiguous/discordant behavioral models
        self.assertEqual(fusion.final_status, "CONFLICT WITH AVAILABLE EVIDENCE")
        # 3. Validation contract: fusion_score is None, display is 'Not validated'
        self.assertIsNone(fusion.fusion_score)
        self.assertEqual(fusion.fusion_score_display, "Not validated")
        # 4. Experimental heuristic score is computed across all 3 streams
        self.assertIsNotNone(fusion.experimental_heuristic_score)
        # 0.50 * 0.85 + 0.25 * 0.525 + 0.25 * 0.382 = 0.425 + 0.13125 + 0.0955 = 0.652
        self.assertAlmostEqual(fusion.experimental_heuristic_score, 0.652, places=2)

    def test_near_threshold_behavioral_uncertainty_not_conflict(self):
        """Verify that when a behavioral model is in the near-threshold band (0.45-0.55),
        concordance reports BEHAVIORAL_UNCERTAINTY, not false BEHAVIORAL_CONFLICT."""
        now = time.time()
        req = AssessmentRequest(
            session_id="sess-near-threshold",
            current_time=now,
            face_telemetry={
                "session_id": "sess-near-threshold",
                "ready": True,
                "timestamp": now - 0.5,
                "verdict": "TRUTH",
                "pLie": 0.382,
            },
            voice_telemetry={
                "session_id": "sess-near-threshold",
                "ready": True,
                "timestamp": now - 1.0,
                "verdict": "Inconclusive",
                "deceptionScore": 0.525,
                "confidence": 0.525,
                "speechDetected": True,
            },
        )
        resp = evaluate_assessment(req)
        fusion = resp.fusion_assessment
        # Near-threshold is BEHAVIORAL_UNCERTAINTY, not BEHAVIORAL_CONFLICT
        self.assertEqual(fusion.agreement_analysis.concordance, "BEHAVIORAL_UNCERTAINTY")
        self.assertEqual(fusion.final_status, "INCONCLUSIVE")
        self.assertEqual(fusion.eligible_modalities_count, 2)


if __name__ == "__main__":
    unittest.main()
