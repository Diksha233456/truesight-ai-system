"""
Unit tests for TRUSIGHT Multi-Source Assessment Coordinator.
Verifies honest presentation of evidence and nonverbal observations:
- Evidence-supported versus contradicted versus insufficient findings.
- Empty or missing evidence.
- Missing or stale face/voice results.
- Different score scales and null values.
- Invalid timestamps and source references.
- Confirmation that no combined deception probability is computed.
"""

import unittest
from assessment_coordinator import (
    AssessmentRequest,
    evaluate_assessment,
)


class TestAssessmentCoordinator(unittest.TestCase):

    def test_empty_request_is_handled_safely(self):
        req = AssessmentRequest()
        resp = evaluate_assessment(req)
        self.assertEqual(resp.status, "READY")
        self.assertEqual(resp.evidence_finding.status, "NO_STATEMENT")
        self.assertFalse(resp.facial_observation.available)
        self.assertFalse(resp.acoustic_observation.available)
        self.assertEqual(len(resp.active_modalities), 0)
        self.assertEqual(len(resp.missing_modalities), 3)
        # Verify no combined deception score is created
        self.assertNotIn("cross_deception_score", resp.overall_summary)
        self.assertNotIn("combined_deception_score", resp.overall_summary)

    def test_contradicted_evidence_finding(self):
        req = AssessmentRequest(
            statement_evidence={
                "statement": "I never met with the contractor on July 4th.",
                "deception_score": 85,
                "inconsistency_index": 0.85,
                "evidence_results": [
                    {
                        "page": 4,
                        "evidence": "Calendar log: Met with contractor at 14:00 on July 4th.",
                        "verdict": "Contradiction",
                        "conflict": "Date/meeting time direct conflict",
                        "rationale": "Calendar documents meeting that speaker denies.",
                    }
                ],
            }
        )
        resp = evaluate_assessment(req)
        self.assertEqual(resp.evidence_finding.status, "CONTRADICTED")
        self.assertEqual(resp.evidence_finding.total_contradictions, 1)
        self.assertEqual(resp.evidence_finding.total_support, 0)
        self.assertEqual(resp.evidence_finding.excerpts[0]["page"], 4)
        self.assertIn("Factual Contradiction Detected", resp.overall_summary["narrative"])

    def test_supported_evidence_finding(self):
        req = AssessmentRequest(
            statement_evidence={
                "statement": "I was at the precinct all morning.",
                "inconsistency_index": 0.05,
                "evidence_results": [
                    {
                        "page": 12,
                        "evidence": "Duty log confirms badge scan at 08:30 at main gate.",
                        "verdict": "Entailment",
                        "rationale": "Badge scan corroborates statement.",
                    }
                ],
            }
        )
        resp = evaluate_assessment(req)
        self.assertEqual(resp.evidence_finding.status, "SUPPORTED")
        self.assertEqual(resp.evidence_finding.total_support, 1)
        self.assertIn("Factual Support", resp.overall_summary["narrative"])

    def test_mixed_evidence_finding(self):
        req = AssessmentRequest(
            statement_evidence={
                "statement": "I made only one transaction on Monday.",
                "evidence_results": [
                    {
                        "page": 2,
                        "evidence": "Bank statement shows transaction #1 on Monday morning.",
                        "verdict": "Entailment",
                    },
                    {
                        "page": 3,
                        "evidence": "Bank statement shows second transaction #2 on Monday evening.",
                        "verdict": "Contradiction",
                    },
                ],
            }
        )
        resp = evaluate_assessment(req)
        self.assertEqual(resp.evidence_finding.status, "MIXED")
        self.assertEqual(resp.evidence_finding.total_contradictions, 1)
        self.assertEqual(resp.evidence_finding.total_support, 1)

    def test_insufficient_evidence_when_chunks_empty(self):
        req = AssessmentRequest(
            statement_evidence={
                "statement": "I parked on 5th avenue.",
                "evidence_results": [],
            }
        )
        resp = evaluate_assessment(req)
        self.assertEqual(resp.evidence_finding.status, "INSUFFICIENT_EVIDENCE")
        self.assertIn("Insufficient relevant excerpts", resp.overall_summary["narrative"])

    def test_stale_facial_telemetry_is_marked_unavailable(self):
        now = 1000.0
        # Face timestamp is 15 seconds old (freshness > 5s threshold)
        req = AssessmentRequest(
            current_time=now,
            face_telemetry={
                "ready": True,
                "timestamp": now - 15.0,
                "verdict": "LIE",
                "pLie": 0.88,
                "reasons": ["AU04 brow furrow"],
            },
        )
        resp = evaluate_assessment(req)
        self.assertFalse(resp.facial_observation.available)
        self.assertEqual(resp.facial_observation.label, "UNAVAILABLE")
        self.assertIsNone(resp.facial_observation.p_lie)
        self.assertEqual(resp.facial_observation.freshness_seconds, 15.0)
        self.assertIn("Facial Telemetry", resp.missing_modalities)

    def test_fresh_facial_telemetry_keeps_model_label_and_plie(self):
        now = 1000.0
        req = AssessmentRequest(
            current_time=now,
            face_telemetry={
                "ready": True,
                "timestamp": now - 0.5,
                "verdict": "TRUTH",
                "pLie": 0.12,
                "reasons": ["Normative facial symmetry"],
                "sentiment": "NEUTRAL",
            },
        )
        resp = evaluate_assessment(req)
        self.assertTrue(resp.facial_observation.available)
        self.assertEqual(resp.facial_observation.label, "TRUTH")
        self.assertEqual(resp.facial_observation.p_lie, 0.12)
        self.assertIn("Facial Telemetry", resp.active_modalities)

    def test_stale_voice_telemetry_marked_unavailable(self):
        now = 2000.0
        # Voice timestamp is 25 seconds old (threshold is 10s)
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
        self.assertEqual(resp.acoustic_observation.verdict, "UNAVAILABLE")
        self.assertIsNone(resp.acoustic_observation.raw_model_score)

    def test_score_scale_and_null_values_are_resilient(self):
        # Scale test: 0-100 legacy score converted safely
        req = AssessmentRequest(
            statement_evidence={
                "statement": "Some statement",
                "deception_score": 75,  # legacy 0-100
                "evidence_results": [{"verdict": "Neutral"}],
            },
            voice_telemetry={
                "ready": True,
                "timestamp": "invalid_timestamp_string",  # invalid format handled safely
                "deceptionScore": "0.62",  # string float handled safely
            },
        )
        resp = evaluate_assessment(req)
        self.assertEqual(resp.evidence_finding.legacy_score, 75)
        self.assertEqual(resp.evidence_finding.inconsistency_index, 0.75)
        self.assertIsNone(resp.acoustic_observation.timestamp)
        self.assertEqual(resp.acoustic_observation.raw_model_score, 0.62)

    def test_never_treats_missing_data_as_negative_evidence(self):
        # Only statement evidence present; face & voice absent
        req = AssessmentRequest(
            statement_evidence={
                "statement": "I was at the scene.",
                "evidence_results": [{"verdict": "Entailment", "page": 1}],
            }
        )
        resp = evaluate_assessment(req)
        # Missing face/voice should NOT diminish or alter the fact that evidence supported the claim
        self.assertEqual(resp.evidence_finding.status, "SUPPORTED")
        self.assertIn("Facial Telemetry", resp.missing_modalities)
        self.assertIn("Acoustic Telemetry", resp.missing_modalities)
        self.assertEqual(resp.overall_summary["evidence_status"], "SUPPORTED")


if __name__ == "__main__":
    unittest.main()
