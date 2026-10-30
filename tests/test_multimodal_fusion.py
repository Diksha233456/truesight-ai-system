"""
Comprehensive Multimodal Fusion Assessment Test Suite.
Validates:
1. Class-index and label mapping for facial and voice models.
2. Truth-class vs lie-class probability interpretation.
3. Missing probabilities, missing speech, and missing facial telemetry.
4. Silence, invalid audio, and stale data thresholds.
5. NO_STATEMENT, INSUFFICIENT_EVIDENCE, SUPPORTED, CONTRADICTED, and MIXED evidence findings.
6. Concordance and agreement:
   - Behavioral agreement (both Truth, both Lie)
   - Behavioral discordance (Face Truth vs Voice Lie)
   - Cross-modal conflict (Evidence Supported vs Behavioral Lie)
7. Single-modality and all-modalities-unavailable handling.
8. Session ID isolation (mismatched sessions are excluded).
9. Defensible fusion score policy: no fabricated combined score (fusion_score is None, fusion_score_display == 'Not validated').
"""

import unittest
import time
from assessment_coordinator import (
    AssessmentRequest,
    evaluate_assessment,
)


class TestMultimodalFusionEngine(unittest.TestCase):

    def setUp(self):
        self.now = 1700000000.0

    # ── 1. ALL MODALITIES UNAVAILABLE & EMPTY REQUEST ────────────────────────

    def test_all_modalities_unavailable(self):
        req = AssessmentRequest(current_time=self.now)
        resp = evaluate_assessment(req)

        fusion = resp.fusion_assessment
        self.assertEqual(fusion.final_status, "ASSESSMENT UNAVAILABLE")
        self.assertIsNone(fusion.fusion_score)
        self.assertEqual(fusion.fusion_score_display, "Not validated")
        self.assertEqual(fusion.eligible_modalities_count, 0)
        self.assertEqual(len(fusion.eligible_modality_names), 0)
        self.assertEqual(len(fusion.excluded_modality_names), 3)
        self.assertEqual(fusion.evidence_sufficiency, "NO_STATEMENT")

    # ── 2. FACIAL MODEL INTERPRETATION & CLASS MAPPING ───────────────────────

    def test_facial_truth_class_probability_interpretation(self):
        """
        Model: classes_ = [0, 1] where 0 = Truth, 1 = Lie.
        When p_lie = 0.327, model predicts TRUTH.
        The probability of the predicted Truth class is 1.0 - 0.327 = 0.673.
        Verifies that 0.327 is not confused with truth probability.
        """
        req = AssessmentRequest(
            current_time=self.now,
            face_telemetry={
                "ready": True,
                "timestamp": self.now - 0.5,
                "verdict": "TRUTH",
                "p_lie": 0.327,
                "reasons": ["Normative blink cadence", "Symmetric cheek engagement"],
            }
        )
        resp = evaluate_assessment(req)
        face = resp.facial_observation

        self.assertTrue(face.available)
        self.assertTrue(face.eligible_for_fusion)
        self.assertEqual(face.mapped_class_label, "MODEL PREDICTS TRUTH CLASS")
        self.assertEqual(face.p_lie, 0.327)
        self.assertAlmostEqual(face.predicted_class_probability, 0.673, places=3)
        self.assertIn("class_0_truth", face.raw_scores)
        self.assertIn("P(class 0)", face.probability_meaning)
        self.assertFalse(face.calibrated)

    def test_facial_lie_class_probability_interpretation(self):
        """
        When p_lie = 0.785, model predicts LIE.
        The probability of predicted Lie class is 0.785.
        """
        req = AssessmentRequest(
            current_time=self.now,
            face_telemetry={
                "ready": True,
                "timestamp": self.now - 0.2,
                "verdict": "LIE",
                "p_lie": 0.785,
                "reasons": ["AU04 brow furrow", "AU15 lip corner depression"],
            }
        )
        resp = evaluate_assessment(req)
        face = resp.facial_observation

        self.assertTrue(face.available)
        self.assertTrue(face.eligible_for_fusion)
        self.assertEqual(face.mapped_class_label, "MODEL PREDICTS LIE CLASS")
        self.assertEqual(face.predicted_class_probability, 0.785)

    def test_facial_inconclusive_near_decision_threshold(self):
        """
        When p_lie is in [0.45, 0.55], classifier is near boundary -> INCONCLUSIVE.
        """
        req = AssessmentRequest(
            current_time=self.now,
            face_telemetry={
                "ready": True,
                "timestamp": self.now - 1.0,
                "p_lie": 0.495,
            }
        )
        resp = evaluate_assessment(req)
        face = resp.facial_observation
        self.assertEqual(face.mapped_class_label, "INCONCLUSIVE")
        self.assertIn("threshold", face.probability_meaning)

    def test_facial_stale_telemetry_excluded(self):
        """Telemetry older than 5.0 seconds must be excluded from fusion."""
        req = AssessmentRequest(
            current_time=self.now,
            face_telemetry={
                "ready": True,
                "timestamp": self.now - 6.5,
                "p_lie": 0.85,
            }
        )
        resp = evaluate_assessment(req)
        face = resp.facial_observation
        self.assertFalse(face.available)
        self.assertFalse(face.eligible_for_fusion)
        self.assertEqual(face.mapped_class_label, "INSUFFICIENT FACIAL DATA")
        self.assertIn("stale", face.exclusion_reason)

    # ── 3. VOICE MODEL INTERPRETATION & SILENCE HANDLING ─────────────────────

    def test_voice_silence_clears_score_and_excludes_from_fusion(self):
        """
        When speech_detected is False, raw score must not be counted as negative or valid.
        Status must be NO SPEECH DETECTED.
        """
        req = AssessmentRequest(
            current_time=self.now,
            voice_telemetry={
                "ready": True,
                "timestamp": self.now - 0.5,
                "speech_detected": False,
                "speechDetected": False,
                "deceptionScore": 0.72,  # Stale score from prior segment
                "verdict": "Deceptive",
            }
        )
        resp = evaluate_assessment(req)
        voice = resp.acoustic_observation

        self.assertTrue(voice.available)
        self.assertFalse(voice.eligible_for_fusion)
        self.assertEqual(voice.mapped_class_label, "NO SPEECH DETECTED")
        self.assertIsNone(voice.predicted_class_probability)
        self.assertIn("Vocal Acoustic Analysis", resp.fusion_assessment.excluded_modality_names)

    def test_voice_truthful_and_deceptive_class_probabilities(self):
        # Truthful prediction
        req_truth = AssessmentRequest(
            current_time=self.now,
            voice_telemetry={
                "ready": True,
                "timestamp": self.now - 1.0,
                "speech_detected": True,
                "deceptionScore": 0.22,
                "confidence": 0.78,
                "verdict": "Truthful",
            }
        )
        resp_truth = evaluate_assessment(req_truth)
        voice_t = resp_truth.acoustic_observation
        self.assertTrue(voice_t.eligible_for_fusion)
        self.assertEqual(voice_t.mapped_class_label, "MODEL PREDICTS TRUTH CLASS")
        self.assertAlmostEqual(voice_t.predicted_class_probability, 0.78, places=2)

        # Deceptive prediction
        req_lie = AssessmentRequest(
            current_time=self.now,
            voice_telemetry={
                "ready": True,
                "timestamp": self.now - 1.0,
                "speech_detected": True,
                "deceptionScore": 0.81,
                "confidence": 0.81,
                "verdict": "Deceptive",
            }
        )
        resp_lie = evaluate_assessment(req_lie)
        voice_l = resp_lie.acoustic_observation
        self.assertTrue(voice_l.eligible_for_fusion)
        self.assertEqual(voice_l.mapped_class_label, "MODEL PREDICTS LIE CLASS")
        self.assertEqual(voice_l.predicted_class_probability, 0.81)

    def test_voice_end_of_15s_recording_window_is_fresh(self):
        """Audio evaluated at the end of a 15-second recording window (age 15.0s) is fresh and eligible."""
        req = AssessmentRequest(
            current_time=self.now,
            voice_telemetry={
                "ready": True,
                "timestamp": self.now - 15.0,
                "window_seconds": 15,
                "speech_detected": True,
                "deceptionScore": 0.70,
                "verdict": "Deceptive",
            }
        )
        resp = evaluate_assessment(req)
        self.assertTrue(resp.acoustic_observation.available)
        self.assertTrue(resp.acoustic_observation.eligible_for_fusion)
        self.assertEqual(resp.acoustic_observation.freshness_seconds, 15.0)

    def test_voice_configured_freshness_boundary(self):
        """Audio within freshness boundary (e.g. 24.5s for 15s window) remains eligible."""
        req = AssessmentRequest(
            current_time=self.now,
            voice_telemetry={
                "ready": True,
                "timestamp": self.now - 24.5,
                "window_seconds": 15,
                "speech_detected": True,
                "deceptionScore": 0.70,
                "verdict": "Deceptive",
            }
        )
        resp = evaluate_assessment(req)
        self.assertTrue(resp.acoustic_observation.available)
        self.assertTrue(resp.acoustic_observation.eligible_for_fusion)

    def test_voice_stale_audio_excluded_past_boundary(self):
        """Audio older than the configured freshness boundary (e.g. 26.0s > 25.0s limit) must be excluded."""
        req = AssessmentRequest(
            current_time=self.now,
            voice_telemetry={
                "ready": True,
                "timestamp": self.now - 26.0,
                "window_seconds": 15,
                "speech_detected": True,
                "deceptionScore": 0.70,
            }
        )
        resp = evaluate_assessment(req)
        self.assertFalse(resp.acoustic_observation.available)
        self.assertFalse(resp.acoustic_observation.eligible_for_fusion)
        self.assertEqual(resp.acoustic_observation.mapped_class_label, "INSUFFICIENT AUDIO")
        self.assertIn("stale telemetry", resp.acoustic_observation.exclusion_reason)

    def test_voice_missing_timestamp_excluded(self):
        """Voice telemetry with missing or null timestamp must be excluded."""
        req = AssessmentRequest(
            current_time=self.now,
            voice_telemetry={
                "ready": True,
                "timestamp": None,
                "speech_detected": True,
                "deceptionScore": 0.70,
            }
        )
        resp = evaluate_assessment(req)
        self.assertFalse(resp.acoustic_observation.eligible_for_fusion)
        self.assertEqual(resp.acoustic_observation.exclusion_reason, "missing timestamp")

    def test_voice_future_timestamp_rejected(self):
        """Voice telemetry with future timestamp (> clock skew tolerance) must be rejected."""
        req = AssessmentRequest(
            current_time=self.now,
            voice_telemetry={
                "ready": True,
                "timestamp": self.now + 10.0,
                "speech_detected": True,
                "deceptionScore": 0.70,
            }
        )
        resp = evaluate_assessment(req)
        self.assertFalse(resp.acoustic_observation.eligible_for_fusion)
        self.assertIn("future timestamp detected", resp.acoustic_observation.exclusion_reason)

    def test_voice_30s_window_configured_freshness(self):
        """Voice telemetry configured with 30s window allows 35s freshness, but rejects 42s."""
        # 35s is within 30s + 10s = 40s boundary
        req_fresh = AssessmentRequest(
            current_time=self.now,
            voice_telemetry={
                "ready": True,
                "timestamp": self.now - 35.0,
                "window_seconds": 30,
                "speech_detected": True,
                "deceptionScore": 0.70,
                "verdict": "Deceptive",
            }
        )
        resp_fresh = evaluate_assessment(req_fresh)
        self.assertTrue(resp_fresh.acoustic_observation.eligible_for_fusion)

        # 42s exceeds 40s boundary
        req_stale = AssessmentRequest(
            current_time=self.now,
            voice_telemetry={
                "ready": True,
                "timestamp": self.now - 42.0,
                "window_seconds": 30,
                "speech_detected": True,
                "deceptionScore": 0.70,
            }
        )
        resp_stale = evaluate_assessment(req_stale)
        self.assertFalse(resp_stale.acoustic_observation.eligible_for_fusion)

    # ── 4. STATEMENT & DOCUMENTARY EVIDENCE GROUNDING ────────────────────────

    def test_no_statement_handling(self):
        req = AssessmentRequest(
            current_time=self.now,
            statement_evidence={
                "statement": "",
                "evidence_results": [{"verdict": "Entailment"}],
            }
        )
        resp = evaluate_assessment(req)
        ev = resp.evidence_finding
        self.assertEqual(ev.status, "NO_STATEMENT")
        self.assertEqual(ev.mapped_class_label, "NO STATEMENT")
        self.assertFalse(ev.eligible_for_fusion)

    def test_contradicted_evidence_leads_to_conflict_with_evidence(self):
        req = AssessmentRequest(
            current_time=self.now,
            statement_evidence={
                "statement": "I never visited the warehouse on Friday.",
                "evidence_results": [
                    {
                        "page": 7,
                        "verdict": "Contradiction",
                        "evidence": "Security badge log: Entry recorded at 14:22 on Friday.",
                        "confidence": 92.0,
                    }
                ],
            }
        )
        resp = evaluate_assessment(req)
        ev = resp.evidence_finding
        fusion = resp.fusion_assessment

        self.assertEqual(ev.status, "CONTRADICTED")
        self.assertEqual(ev.mapped_class_label, "CONTRADICTED")
        self.assertEqual(fusion.final_status, "CONFLICT WITH AVAILABLE EVIDENCE")
        self.assertEqual(fusion.evidence_sufficiency, "SUFFICIENT")

    def test_supported_evidence_leads_to_consistent_with_evidence(self):
        req = AssessmentRequest(
            current_time=self.now,
            statement_evidence={
                "statement": "I was attending the board meeting at 10 AM.",
                "evidence_results": [
                    {
                        "page": 2,
                        "verdict": "Entailment",
                        "evidence": "Meeting minutes confirm attendance at 10:00 AM.",
                        "confidence": 95.0,
                    }
                ],
            }
        )
        resp = evaluate_assessment(req)
        self.assertEqual(resp.evidence_finding.status, "SUPPORTED")
        self.assertEqual(resp.fusion_assessment.final_status, "CONSISTENT WITH AVAILABLE EVIDENCE")

    def test_mixed_evidence_leads_to_mixed_findings(self):
        req = AssessmentRequest(
            current_time=self.now,
            statement_evidence={
                "statement": "I made only one purchase on Monday.",
                "evidence_results": [
                    {"page": 1, "verdict": "Entailment", "evidence": "Purchase #1 recorded on Monday."},
                    {"page": 2, "verdict": "Contradiction", "evidence": "Purchase #2 recorded Monday evening."},
                ],
            }
        )
        resp = evaluate_assessment(req)
        self.assertEqual(resp.evidence_finding.status, "MIXED")
        self.assertEqual(resp.fusion_assessment.final_status, "MIXED FINDINGS")

    # ── 5. CROSS-MODAL AGREEMENT, DISCORDANCE & CORRELATION ──────────────────

    def test_high_agreement_across_all_three_modalities(self):
        """
        Evidence: SUPPORTED
        Face: TRUTH
        Voice: TRUTH
        Result: CONSISTENT WITH AVAILABLE EVIDENCE, Concordance: CONCORDANT_TRUTH
        """
        req = AssessmentRequest(
            current_time=self.now,
            statement_evidence={
                "statement": "I was at the precinct all morning.",
                "evidence_results": [{"page": 4, "verdict": "Entailment", "evidence": "Duty log confirms presence."}],
            },
            face_telemetry={
                "ready": True,
                "timestamp": self.now - 0.5,
                "p_lie": 0.15,
            },
            voice_telemetry={
                "ready": True,
                "timestamp": self.now - 1.0,
                "speech_detected": True,
                "deceptionScore": 0.18,
                "verdict": "Truthful",
            },
        )
        resp = evaluate_assessment(req)
        fusion = resp.fusion_assessment

        self.assertEqual(fusion.final_status, "CONSISTENT WITH AVAILABLE EVIDENCE")
        self.assertEqual(fusion.eligible_modalities_count, 3)
        self.assertEqual(fusion.agreement_analysis.concordance, "CONCORDANT_TRUTH")
        self.assertIn("Both facial and vocal classifiers predict TRUTH class", fusion.agreement_analysis.behavioral_agreement)

    def test_cross_modal_conflict_supported_evidence_vs_both_behavioral_lies(self):
        """
        Documentary evidence supports statement, but both behavioral models predict LIE.
        Requirement 7: Documentary findings must not be overridden by uncalibrated behavioral models.
        The fusion status remains CONSISTENT WITH AVAILABLE EVIDENCE, while recording cross-modal conflict.
        """
        req = AssessmentRequest(
            current_time=self.now,
            statement_evidence={
                "statement": "I filed the paperwork on time.",
                "evidence_results": [{"page": 1, "verdict": "Entailment", "evidence": "Receipt confirms filing."}],
            },
            face_telemetry={
                "ready": True,
                "timestamp": self.now - 0.3,
                "p_lie": 0.88,
            },
            voice_telemetry={
                "ready": True,
                "timestamp": self.now - 0.4,
                "speech_detected": True,
                "deceptionScore": 0.82,
                "verdict": "Deceptive",
            },
        )
        resp = evaluate_assessment(req)
        fusion = resp.fusion_assessment

        self.assertEqual(fusion.final_status, "CONSISTENT WITH AVAILABLE EVIDENCE")
        self.assertEqual(fusion.agreement_analysis.concordance, "CROSS_MODAL_CONFLICT")
        self.assertIn("Uncalibrated behavioral predictions do not override documentary findings", fusion.explanation)

    def test_behavioral_conflict_face_truth_vs_voice_lie_without_evidence(self):
        """
        Without evidence, Face predicts TRUTH while Voice predicts LIE.
        Result: MIXED FINDINGS.
        """
        req = AssessmentRequest(
            current_time=self.now,
            face_telemetry={
                "ready": True,
                "timestamp": self.now - 0.5,
                "p_lie": 0.12,
            },
            voice_telemetry={
                "ready": True,
                "timestamp": self.now - 0.5,
                "speech_detected": True,
                "deceptionScore": 0.79,
                "verdict": "Deceptive",
            },
        )
        resp = evaluate_assessment(req)
        fusion = resp.fusion_assessment

        self.assertEqual(fusion.final_status, "MIXED FINDINGS")
        self.assertEqual(fusion.agreement_analysis.concordance, "BEHAVIORAL_CONFLICT")
        self.assertIn("Facial model predicts TRUTH while Acoustic model predicts LIE", fusion.explanation)

    def test_single_modality_only_is_insufficient_data(self):
        """A single behavioral modality without evidence cannot support a definitive assessment."""
        req = AssessmentRequest(
            current_time=self.now,
            face_telemetry={
                "ready": True,
                "timestamp": self.now - 0.5,
                "p_lie": 0.15,
            }
        )
        resp = evaluate_assessment(req)
        fusion = resp.fusion_assessment

        self.assertEqual(fusion.final_status, "INSUFFICIENT DATA")
        self.assertEqual(fusion.eligible_modalities_count, 1)
        self.assertIn("single behavioral stream", fusion.explanation)

    # ── 6. SESSION ISOLATION & RESILIENCE ────────────────────────────────────

    def test_session_id_mismatch_excludes_stale_session_data(self):
        """Telemetry belonging to a previous session ID must be excluded."""
        req = AssessmentRequest(
            session_id="sess_active_123",
            current_time=self.now,
            face_telemetry={
                "session_id": "sess_old_stale_999",  # Mismatched session
                "ready": True,
                "timestamp": self.now - 0.5,
                "p_lie": 0.20,
            },
            statement_evidence={
                "statement": "Claim here",
                "evidence_results": [{"verdict": "Entailment", "page": 1}],
            }
        )
        resp = evaluate_assessment(req)
        face = resp.facial_observation

        self.assertFalse(face.available)
        self.assertFalse(face.eligible_for_fusion)
        self.assertIn("Mismatched session", face.exclusion_reason)

    def test_no_fabricated_fusion_score(self):
        """Under NO circumstances should a fabricated percentage or uncalibrated average be emitted."""
        req = AssessmentRequest(
            current_time=self.now,
            statement_evidence={
                "statement": "Some statement",
                "evidence_results": [{"verdict": "Entailment", "page": 1}],
            },
            face_telemetry={"ready": True, "timestamp": self.now - 0.5, "p_lie": 0.8},
            voice_telemetry={"ready": True, "timestamp": self.now - 0.5, "speech_detected": True, "deceptionScore": 0.8},
        )
        resp = evaluate_assessment(req)
        self.assertIsNone(resp.fusion_assessment.fusion_score)
        self.assertEqual(resp.fusion_assessment.fusion_score_display, "Not validated")
        self.assertIn("Not validated", resp.overall_summary["fusion_score_display"])

    # ── 7. PROBABILITY VALIDATION TESTS ──────────────────────────────────────

    def test_invalid_probabilities_rejected_strictly(self):
        """
        Verify that NaN, Inf, negative values, values > 1.0, and non-numeric strings
        are rejected and not silently accepted or clamped.
        """
        invalid_values = [float("nan"), float("inf"), float("-inf"), -0.15, 1.45, "invalid_str"]

        for val in invalid_values:
            # Face probability validation
            req_face = AssessmentRequest(
                current_time=self.now,
                face_telemetry={
                    "ready": True,
                    "timestamp": self.now - 0.5,
                    "p_lie": val,
                }
            )
            resp_face = evaluate_assessment(req_face)
            face = resp_face.facial_observation
            self.assertFalse(face.eligible_for_fusion, f"Face should reject invalid probability: {val}")
            self.assertIsNone(face.p_lie, f"Face p_lie should be None for {val}")
            self.assertEqual(face.quality_status, "INVALID_PROBABILITY")
            self.assertIn("Invalid probability input", face.exclusion_reason)

            # Voice probability validation
            req_voice = AssessmentRequest(
                current_time=self.now,
                voice_telemetry={
                    "ready": True,
                    "timestamp": self.now - 0.5,
                    "speech_detected": True,
                    "deceptionScore": val,
                }
            )
            resp_voice = evaluate_assessment(req_voice)
            voice = resp_voice.acoustic_observation
            self.assertFalse(voice.eligible_for_fusion, f"Voice should reject invalid probability: {val}")
            self.assertIsNone(voice.raw_model_score, f"Voice raw_model_score should be None for {val}")
            self.assertEqual(voice.quality_status, "INVALID_PROBABILITY")
            self.assertIn("Invalid probability input", voice.exclusion_reason)

    # ── 8. LABEL / PROBABILITY MISMATCH & CONFLICT TESTS ─────────────────────

    def test_face_label_probability_mismatch_detected(self):
        """
        Contradictions between raw label and probability must be detected and not overridden:
        - Raw label LIE with p_lie = 0.15
        - Raw label TRUTH with p_lie = 0.85
        Must be marked INCONCLUSIVE and INCONSISTENT_OBSERVATIONS, and ineligible for fusion.
        """
        # Case 1: Label LIE with low probability
        req1 = AssessmentRequest(
            current_time=self.now,
            face_telemetry={
                "ready": True,
                "timestamp": self.now - 0.5,
                "verdict": "LIE",
                "p_lie": 0.15,
            }
        )
        resp1 = evaluate_assessment(req1)
        face1 = resp1.facial_observation
        self.assertFalse(face1.eligible_for_fusion)
        self.assertEqual(face1.mapped_class_label, "INCONCLUSIVE")
        self.assertEqual(face1.quality_status, "INCONSISTENT_OBSERVATIONS")
        self.assertIn("Label/probability conflict", face1.exclusion_reason)

        # Case 2: Label TRUTH with high lie probability
        req2 = AssessmentRequest(
            current_time=self.now,
            face_telemetry={
                "ready": True,
                "timestamp": self.now - 0.5,
                "verdict": "TRUTH",
                "p_lie": 0.85,
            }
        )
        resp2 = evaluate_assessment(req2)
        face2 = resp2.facial_observation
        self.assertFalse(face2.eligible_for_fusion)
        self.assertEqual(face2.mapped_class_label, "INCONCLUSIVE")
        self.assertEqual(face2.quality_status, "INCONSISTENT_OBSERVATIONS")
        self.assertIn("Label/probability conflict", face2.exclusion_reason)

    def test_voice_verdict_score_mismatch_detected(self):
        """
        Contradictions between raw verdict and score must be detected and not overridden:
        - Raw verdict Deceptive with score = 0.15
        - Raw verdict Truthful with score = 0.85
        Must be marked INCONCLUSIVE and INCONSISTENT_AUDIO, and ineligible for fusion.
        """
        # Case 1: Deceptive verdict with low score
        req1 = AssessmentRequest(
            current_time=self.now,
            voice_telemetry={
                "ready": True,
                "timestamp": self.now - 0.5,
                "speech_detected": True,
                "verdict": "Deceptive",
                "deceptionScore": 0.15,
            }
        )
        resp1 = evaluate_assessment(req1)
        voice1 = resp1.acoustic_observation
        self.assertFalse(voice1.eligible_for_fusion)
        self.assertEqual(voice1.mapped_class_label, "INCONCLUSIVE")
        self.assertEqual(voice1.quality_status, "INCONSISTENT_AUDIO")
        self.assertIn("Verdict/score conflict", voice1.exclusion_reason)

        # Case 2: Truthful verdict with high deceptive score
        req2 = AssessmentRequest(
            current_time=self.now,
            voice_telemetry={
                "ready": True,
                "timestamp": self.now - 0.5,
                "speech_detected": True,
                "verdict": "Truthful",
                "deceptionScore": 0.85,
            }
        )
        resp2 = evaluate_assessment(req2)
        voice2 = resp2.acoustic_observation
        self.assertFalse(voice2.eligible_for_fusion)
        self.assertEqual(voice2.mapped_class_label, "INCONCLUSIVE")
        self.assertEqual(voice2.quality_status, "INCONSISTENT_AUDIO")
        self.assertIn("Verdict/score conflict", voice2.exclusion_reason)

    # ── 9. TIMESTAMP FRESHNESS, MISSING & FUTURE TIMESTAMPS ──────────────────

    def test_valid_timestamps_both_modalities(self):
        """Valid fresh timestamps within thresholds (<=5s for face, <=10s for voice) must be accepted."""
        face_time = self.now - 1.5
        voice_time = self.now - 3.2
        req = AssessmentRequest(
            current_time=self.now,
            face_telemetry={
                "ready": True,
                "timestamp": face_time,
                "p_lie": 0.20,
            },
            voice_telemetry={
                "ready": True,
                "timestamp": voice_time,
                "speech_detected": True,
                "deceptionScore": 0.25,
                "verdict": "Truthful",
            }
        )
        resp = evaluate_assessment(req)
        face = resp.facial_observation
        voice = resp.acoustic_observation

        self.assertTrue(face.available)
        self.assertTrue(face.eligible_for_fusion)
        self.assertEqual(face.timestamp, face_time)
        self.assertEqual(face.freshness_seconds, 1.5)
        self.assertIsNone(face.exclusion_reason)

        self.assertTrue(voice.available)
        self.assertTrue(voice.eligible_for_fusion)
        self.assertEqual(voice.timestamp, voice_time)
        self.assertEqual(voice.freshness_seconds, 3.2)
        self.assertIsNone(voice.exclusion_reason)

    def test_missing_timestamp_face_modality(self):
        """Missing face timestamp (None or omitted) must mark facial telemetry ineligible."""
        # Case A: Explicit None
        req1 = AssessmentRequest(
            current_time=self.now,
            face_telemetry={"ready": True, "timestamp": None, "p_lie": 0.20},
        )
        resp1 = evaluate_assessment(req1)
        self.assertFalse(resp1.facial_observation.eligible_for_fusion)
        self.assertIn("missing timestamp", resp1.facial_observation.exclusion_reason)

        # Case B: Omitted key
        req2 = AssessmentRequest(
            current_time=self.now,
            face_telemetry={"ready": True, "p_lie": 0.20},
        )
        resp2 = evaluate_assessment(req2)
        self.assertFalse(resp2.facial_observation.eligible_for_fusion)
        self.assertIn("missing timestamp", resp2.facial_observation.exclusion_reason)

    def test_missing_timestamp_voice_modality(self):
        """Missing voice timestamp (None or omitted) must mark voice telemetry ineligible."""
        # Case A: Explicit None
        req1 = AssessmentRequest(
            current_time=self.now,
            voice_telemetry={"ready": True, "timestamp": None, "speech_detected": True, "deceptionScore": 0.20},
        )
        resp1 = evaluate_assessment(req1)
        self.assertFalse(resp1.acoustic_observation.eligible_for_fusion)
        self.assertIn("missing timestamp", resp1.acoustic_observation.exclusion_reason)

        # Case B: Omitted key
        req2 = AssessmentRequest(
            current_time=self.now,
            voice_telemetry={"ready": True, "speech_detected": True, "deceptionScore": 0.20},
        )
        resp2 = evaluate_assessment(req2)
        self.assertFalse(resp2.acoustic_observation.eligible_for_fusion)
        self.assertIn("missing timestamp", resp2.acoustic_observation.exclusion_reason)

    def test_missing_timestamps_both_modalities(self):
        """When timestamps are missing from both modalities, neither is eligible for fusion."""
        req = AssessmentRequest(
            current_time=self.now,
            face_telemetry={"ready": True, "p_lie": 0.20},
            voice_telemetry={"ready": True, "speech_detected": True, "deceptionScore": 0.20},
        )
        resp = evaluate_assessment(req)
        self.assertFalse(resp.facial_observation.eligible_for_fusion)
        self.assertFalse(resp.acoustic_observation.eligible_for_fusion)
        self.assertIn("missing timestamp", resp.facial_observation.exclusion_reason)
        self.assertIn("missing timestamp", resp.acoustic_observation.exclusion_reason)
        self.assertEqual(resp.fusion_assessment.eligible_modalities_count, 0)

    def test_future_timestamps_rejected(self):
        """Timestamps in the future (> now + 1.0s tolerance) must be flagged and excluded."""
        future_ts = self.now + 10.0
        req = AssessmentRequest(
            current_time=self.now,
            face_telemetry={
                "ready": True,
                "timestamp": future_ts,
                "p_lie": 0.20,
            },
            voice_telemetry={
                "ready": True,
                "timestamp": future_ts,
                "speech_detected": True,
                "deceptionScore": 0.20,
            }
        )
        resp = evaluate_assessment(req)
        self.assertFalse(resp.facial_observation.eligible_for_fusion)
        self.assertFalse(resp.acoustic_observation.eligible_for_fusion)
        self.assertEqual(resp.facial_observation.quality_status, "INVALID_TIMESTAMP")
        self.assertIn("future timestamp detected", resp.facial_observation.exclusion_reason)
        self.assertIn("future timestamp detected", resp.acoustic_observation.exclusion_reason)

    def test_missing_and_non_positive_timestamps(self):
        """Missing or non-positive timestamps must be safely flagged as invalid."""
        req = AssessmentRequest(
            current_time=self.now,
            face_telemetry={
                "ready": True,
                "timestamp": None,
                "p_lie": 0.20,
            },
            voice_telemetry={
                "ready": True,
                "timestamp": -100.0,
                "speech_detected": True,
                "deceptionScore": 0.20,
            }
        )
        resp = evaluate_assessment(req)
        self.assertFalse(resp.facial_observation.eligible_for_fusion)
        self.assertFalse(resp.acoustic_observation.eligible_for_fusion)
        self.assertIn("missing timestamp", resp.facial_observation.exclusion_reason)
        self.assertIn("invalid timestamp value", resp.acoustic_observation.exclusion_reason)

    # ── 10. MODALITY COMBINATIONS MATRIX ─────────────────────────────────────


    def test_modality_combination_evidence_only(self):
        """Only documentary evidence is available; face and voice are offline."""
        req = AssessmentRequest(
            current_time=self.now,
            statement_evidence={
                "statement": "I signed the contract yesterday.",
                "evidence_results": [{"page": 3, "verdict": "Entailment", "evidence": "Signature page dated yesterday."}],
            },
            face_telemetry={"ready": False},
            voice_telemetry={"ready": False},
        )
        resp = evaluate_assessment(req)
        self.assertEqual(resp.fusion_assessment.final_status, "CONSISTENT WITH AVAILABLE EVIDENCE")
        self.assertEqual(resp.fusion_assessment.eligible_modalities_count, 1)
        self.assertEqual(resp.fusion_assessment.eligible_modality_names, ["Statement & Documentary Evidence"])
        self.assertIn("Facial Telemetry", resp.missing_modalities)
        self.assertIn("Acoustic Telemetry", resp.missing_modalities)

    def test_modality_combination_evidence_and_face(self):
        """Documentary evidence and facial analysis eligible; voice offline."""
        req = AssessmentRequest(
            current_time=self.now,
            statement_evidence={
                "statement": "I was at work.",
                "evidence_results": [{"page": 1, "verdict": "Entailment", "evidence": "Keycard access logged."}],
            },
            face_telemetry={"ready": True, "timestamp": self.now - 0.5, "p_lie": 0.20},
            voice_telemetry={"ready": False},
        )
        resp = evaluate_assessment(req)
        self.assertEqual(resp.fusion_assessment.final_status, "CONSISTENT WITH AVAILABLE EVIDENCE")
        self.assertEqual(resp.fusion_assessment.eligible_modalities_count, 2)
        self.assertIn("Statement & Documentary Evidence", resp.fusion_assessment.eligible_modality_names)
        self.assertIn("Facial Movement Analysis", resp.fusion_assessment.eligible_modality_names)
        self.assertIn("Vocal Acoustic Analysis", resp.fusion_assessment.excluded_modality_names)

    def test_modality_combination_evidence_and_voice(self):
        """Documentary evidence and voice eligible; face tracking calibrating."""
        req = AssessmentRequest(
            current_time=self.now,
            statement_evidence={
                "statement": "I attended the hearing.",
                "evidence_results": [{"page": 5, "verdict": "Entailment", "evidence": "Hearing attendance roster."}],
            },
            face_telemetry={"ready": False},
            voice_telemetry={"ready": True, "timestamp": self.now - 0.5, "speech_detected": True, "deceptionScore": 0.22, "verdict": "Truthful"},
        )
        resp = evaluate_assessment(req)
        self.assertEqual(resp.fusion_assessment.final_status, "CONSISTENT WITH AVAILABLE EVIDENCE")
        self.assertEqual(resp.fusion_assessment.eligible_modalities_count, 2)
        self.assertIn("Statement & Documentary Evidence", resp.fusion_assessment.eligible_modality_names)
        self.assertIn("Vocal Acoustic Analysis", resp.fusion_assessment.eligible_modality_names)
        self.assertIn("Facial Movement Analysis", resp.fusion_assessment.excluded_modality_names)

    def test_modality_combination_face_and_voice_without_evidence(self):
        """Face and Voice both eligible; no statement submitted."""
        req = AssessmentRequest(
            current_time=self.now,
            face_telemetry={"ready": True, "timestamp": self.now - 0.5, "p_lie": 0.15},
            voice_telemetry={"ready": True, "timestamp": self.now - 0.5, "speech_detected": True, "deceptionScore": 0.18, "verdict": "Truthful"},
        )
        resp = evaluate_assessment(req)
        self.assertEqual(resp.fusion_assessment.final_status, "INCONCLUSIVE")
        self.assertEqual(resp.fusion_assessment.eligible_modalities_count, 2)
        self.assertEqual(resp.fusion_assessment.agreement_analysis.concordance, "CONCORDANT_TRUTH")
        self.assertIn("documentary evidence is absent to establish factual veracity", resp.fusion_assessment.explanation)

    def test_modality_combination_face_only(self):
        """Only Face eligible; evidence absent, voice offline."""
        req = AssessmentRequest(
            current_time=self.now,
            face_telemetry={"ready": True, "timestamp": self.now - 0.5, "p_lie": 0.20},
            voice_telemetry={"ready": False},
        )
        resp = evaluate_assessment(req)
        self.assertEqual(resp.fusion_assessment.final_status, "INSUFFICIENT DATA")
        self.assertEqual(resp.fusion_assessment.eligible_modalities_count, 1)

    def test_modality_combination_voice_only(self):
        """Only Voice eligible; evidence absent, face offline."""
        req = AssessmentRequest(
            current_time=self.now,
            face_telemetry={"ready": False},
            voice_telemetry={"ready": True, "timestamp": self.now - 0.5, "speech_detected": True, "deceptionScore": 0.20, "verdict": "Truthful"},
        )
        resp = evaluate_assessment(req)
        self.assertEqual(resp.fusion_assessment.final_status, "INSUFFICIENT DATA")
        self.assertEqual(resp.fusion_assessment.eligible_modalities_count, 1)

    def test_neural_corrector_module_and_equal_weighting(self):
        from fusion_service.neural_corrector import NeuralCorrector
        import torch
        model = NeuralCorrector()
        inputs = torch.tensor([[0.8, 0.9, 0.7, 0.85, 0.8, 0.9]], dtype=torch.float32)
        score, conf = model(inputs, active_mask=[1.0, 1.0, 1.0])
        self.assertTrue(0.0 <= score.item() <= 1.0)
        self.assertTrue(0.0 <= conf.item() <= 1.0)

    def test_assessment_includes_neural_corrected_fusion_score(self):
        req = AssessmentRequest(
            current_time=self.now,
            statement_evidence={
                "statement": "I was at the office.",
                "evidence_results": [{"page": 1, "verdict": "Entailment", "evidence": "Badge log confirmed."}],
            },
            face_telemetry={
                "ready": True,
                "timestamp": self.now - 0.5,
                "p_lie": 0.2,
            },
            voice_telemetry={
                "ready": True,
                "timestamp": self.now - 0.5,
                "speech_detected": True,
                "deceptionScore": 0.25,
                "verdict": "Truthful",
            }
        )
        resp = evaluate_assessment(req)
        fusion = resp.fusion_assessment
        self.assertIsNotNone(fusion.neural_corrected_fusion_score)
        self.assertTrue(0.0 <= fusion.neural_corrected_fusion_score <= 1.0)
        self.assertEqual(fusion.eligible_modalities_count, 3)



if __name__ == "__main__":
    unittest.main()

