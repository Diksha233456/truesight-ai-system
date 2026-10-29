"""
Regression tests for Phase 4:
- Context & negation preservation in PDF chunking.
- Date and numerical conflict handling in evidence assessment.
- Distinguishing SUPPORTED, CONTRADICTED, MIXED, INSUFFICIENT_EVIDENCE, and NO_STATEMENT.
- Ensuring page references and source excerpts are preserved.
- Verifying empty/failed retrieval never implies truth or deception.
- Verifying UI handling of missing telemetry without false negative inference.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "TruSight_Gemini_API" / "TruSight_Gemini_API"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from rag import chunk_text
from assessment_coordinator import (
    AssessmentRequest,
    evaluate_assessment,
)


class TestPhase4EvidenceAndHonestEvaluation(unittest.TestCase):

    def test_chunking_preserves_negation_and_sentence_integrity(self):
        """
        Verify that a negation clause ('The witness did NOT see the vehicle')
        is not severed across arbitrary word splits and page numbers remain exact.
        """
        mock_pages = [
            {
                "page": 3,
                "text": "The suspect was questioned at length. The witness did not see any blue sedan parked near the warehouse on July 14. Camera 4 recorded standard traffic."
            }
        ]
        # Chunk with small size to test boundary handling
        chunks = chunk_text(mock_pages, chunk_size=15, overlap=5)
        self.assertGreaterEqual(len(chunks), 1)
        # Verify page 3 is attached to every chunk
        for c in chunks:
            self.assertEqual(c["page"], 3)
        # Verify the sentence containing 'did not see' is intact within a chunk
        negation_found = any("did not see any blue sedan" in c["text"] for c in chunks)
        self.assertTrue(negation_found, "Sentence-aware chunking must keep negation clauses intact.")

    def test_evidence_contradicted_date_and_location(self):
        """Verify that explicit date/location contradictions return CONTRADICTED with exact excerpt."""
        req = AssessmentRequest(
            statement_evidence={
                "statement": "I was at the stadium on August 12.",
                "evidence_results": [
                    {
                        "page": 7,
                        "evidence": "Toll booth records show vehicle entered expressway 200 miles away on August 12.",
                        "verdict": "Contradiction",
                        "conflict": "Date/location mismatch: Expressway vs Stadium",
                        "rationale": "Automated toll record contradicts physical presence at stadium."
                    }
                ]
            }
        )
        resp = evaluate_assessment(req)
        self.assertEqual(resp.evidence_finding.status, "CONTRADICTED")
        self.assertEqual(resp.evidence_finding.total_contradictions, 1)
        self.assertEqual(resp.evidence_finding.excerpts[0]["page"], 7)
        self.assertIn("Toll booth records", resp.evidence_finding.excerpts[0]["text"])

    def test_evidence_supported_when_documents_corroborate(self):
        """Verify that documents corroborating claims return SUPPORTED."""
        req = AssessmentRequest(
            statement_evidence={
                "statement": "I signed the lease at the downtown branch.",
                "evidence_results": [
                    {
                        "page": 2,
                        "evidence": "Lease agreement signed in presence of notary at downtown office.",
                        "verdict": "Entailment",
                        "rationale": "Directly corroborates downtown signing."
                    }
                ]
            }
        )
        resp = evaluate_assessment(req)
        self.assertEqual(resp.evidence_finding.status, "SUPPORTED")
        self.assertEqual(resp.evidence_finding.total_support, 1)
        self.assertEqual(resp.evidence_finding.total_contradictions, 0)

    def test_evidence_mixed_when_contradictions_and_support_coexist(self):
        """Verify that coexisting conflicting and supporting excerpts produce MIXED status."""
        req = AssessmentRequest(
            statement_evidence={
                "statement": "I only made one bank transfer.",
                "evidence_results": [
                    {"page": 1, "evidence": "Wire transfer #1 confirmed.", "verdict": "Entailment"},
                    {"page": 2, "evidence": "Wire transfer #2 logged 10 minutes later.", "verdict": "Contradiction"},
                ]
            }
        )
        resp = evaluate_assessment(req)
        self.assertEqual(resp.evidence_finding.status, "MIXED")
        self.assertEqual(resp.evidence_finding.total_support, 1)
        self.assertEqual(resp.evidence_finding.total_contradictions, 1)

    def test_retrieval_failure_never_implies_truth_or_deception(self):
        """
        Verify that when zero relevant chunks are found or search returns empty,
        the system returns INSUFFICIENT_EVIDENCE and never infers truth or falsehood.
        """
        req = AssessmentRequest(
            statement_evidence={
                "statement": "I had coffee with my cousin yesterday.",
                "evidence_results": []  # Empty search results
            }
        )
        resp = evaluate_assessment(req)
        self.assertEqual(resp.evidence_finding.status, "INSUFFICIENT_EVIDENCE")
        # Inconsistency score must be None or 0.0, never fabricated confidence
        self.assertIn(resp.evidence_finding.inconsistency_index, (None, 0.0))
        self.assertIn("Insufficient relevant excerpts", resp.overall_summary["narrative"])

    def test_missing_statement_handled_as_no_statement(self):
        """Verify that absence of statement returns NO_STATEMENT."""
        req = AssessmentRequest(
            statement_evidence={
                "statement": "",
                "evidence_results": []
            }
        )
        resp = evaluate_assessment(req)
        self.assertEqual(resp.evidence_finding.status, "NO_STATEMENT")

    def test_ui_missing_telemetry_never_treated_as_deception(self):
        """
        Verify that when face and voice telemetry are unavailable,
        missing streams are listed in missing_modalities and never affect the evidence status.
        """
        req = AssessmentRequest(
            statement_evidence={
                "statement": "I arrived at 9 AM.",
                "evidence_results": [{"page": 4, "evidence": "Badge reader clocked in at 09:00.", "verdict": "Entailment"}]
            },
            face_telemetry={"ready": False},
            voice_telemetry={"ready": False}
        )
        resp = evaluate_assessment(req)
        self.assertEqual(resp.evidence_finding.status, "SUPPORTED")
        self.assertIn("Facial Telemetry", resp.missing_modalities)
        self.assertIn("Acoustic Telemetry", resp.missing_modalities)
        self.assertEqual(resp.facial_observation.label, "UNAVAILABLE")
        self.assertEqual(resp.acoustic_observation.verdict, "UNAVAILABLE")
        self.assertNotIn("cross_deception_score", resp.overall_summary)


if __name__ == "__main__":
    unittest.main()
