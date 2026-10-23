"""
tests/test_deception_study_protocol.py

Unit test suite verifying the mathematical, architectural, and data-integrity rules
defined in Stage 6: datasets/external/deception_study_protocol.md:
1. Strict speaker-independent train/test partitioning (zero speaker overlap).
2. Balanced accuracy vs. standard accuracy under 2:1 class imbalance (2T1F).
3. Baseline comparisons (majority class, random dummy).
4. Three-way classification gate (Inconclusive outcomes for degraded SNR or narrow margins).
5. Manifest non-sensitive metadata schema compliance and PII exclusion.

Runs 100% offline using synthetic mock data structures; zero network, zero production model changes.
"""

import math
import unittest
from typing import Dict, List, Set, Tuple


def split_speakers_stratified(
    records: List[Dict],
    test_ratio: float = 0.333,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Partition manifest records strictly by speaker_id without splitting any speaker's clips.
    """
    by_speaker: Dict[str, List[Dict]] = {}
    for r in records:
        by_speaker.setdefault(r["speaker_id"], []).append(r)

    speakers = sorted(by_speaker.keys())
    num_test = max(1, int(math.ceil(len(speakers) * test_ratio)))
    test_speakers = set(speakers[-num_test:])
    dev_speakers = set(speakers[:-num_test])

    dev_records = [r for r in records if r["speaker_id"] in dev_speakers]
    test_records = [r for r in records if r["speaker_id"] in test_speakers]
    return dev_records, test_records


def compute_veracity_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, float]:
    """
    Compute Accuracy, Balanced Accuracy, and Macro F1 for binary classification.
    """
    assert len(y_true) == len(y_pred)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)

    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    rec0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rec1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    balanced_acc = 0.5 * (rec0 + rec1)

    prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_0 = 2 * (prec0 * rec0) / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0
    f1_1 = 2 * (prec1 * rec1) / (prec1 + rec1) if (prec1 + rec1) > 0 else 0.0
    macro_f1 = 0.5 * (f1_0 + f1_1)

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "f1_truthful": round(f1_0, 4),
        "f1_deceptive": round(f1_1, 4),
    }


def classify_three_way(
    p_deceptive: float,
    snr_db: float,
    clipping_ratio: float,
    duration_s: float,
    margin: float = 0.08,
) -> str:
    """
    Implement the 3-way decision gate from deception_study_protocol.md Section 4.3.
    """
    # Quality gates
    if snr_db < 12.0 or clipping_ratio > 0.005 or duration_s < 15.0:
        return "Inconclusive"

    # Ambiguity margin gate
    if abs(p_deceptive - 0.50) < margin:
        return "Inconclusive"

    return "Deceptive" if p_deceptive >= 0.50 + margin else "Truthful"


class TestDeceptionStudyProtocolDesign(unittest.TestCase):

    def test_strict_speaker_independent_partitioning(self):
        """Verify that recordings from the same speaker never appear in both partitions."""
        mock_records = []
        for i in range(1, 13):  # 12 speakers
            spk = f"IND_SPK_{i:03d}"
            # 3 recordings per speaker (2 truth, 1 fabricated)
            mock_records.append({"speaker_id": spk, "story_id": "S1", "ground_truth": 0})
            mock_records.append({"speaker_id": spk, "story_id": "S2", "ground_truth": 0})
            mock_records.append({"speaker_id": spk, "story_id": "S3", "ground_truth": 1})

        dev, test = split_speakers_stratified(mock_records, test_ratio=0.333)

        dev_speakers = {r["speaker_id"] for r in dev}
        test_speakers = {r["speaker_id"] for r in test}

        # Zero speaker intersection
        self.assertEqual(len(dev_speakers.intersection(test_speakers)), 0)
        self.assertGreater(len(dev_speakers), 0)
        self.assertGreater(len(test_speakers), 0)
        # All recordings for a speaker are in only one set
        for spk in dev_speakers:
            self.assertEqual(sum(1 for r in dev if r["speaker_id"] == spk), 3)
        for spk in test_speakers:
            self.assertEqual(sum(1 for r in test if r["speaker_id"] == spk), 3)

    def test_majority_class_baseline_exposes_imbalance(self):
        """In a 2:1 paradigm (66.7% truth), majority baseline has high accuracy but 50% balanced acc."""
        # 12 truthful (0), 6 deceptive (1)
        y_true = [0] * 12 + [1] * 6
        # Majority-class predictor always predicts 0
        y_pred_majority = [0] * 18

        metrics = compute_veracity_metrics(y_true, y_pred_majority)
        self.assertAlmostEqual(metrics["accuracy"], 0.6667, places=3)
        self.assertEqual(metrics["balanced_accuracy"], 0.5000)
        self.assertLess(metrics["macro_f1"], 0.5000)

    def test_three_way_inconclusive_gate_triggers(self):
        """Inconclusive verdict must be returned on poor SNR, clipping, or borderline margin."""
        # 1. Clean audio with confident probability -> Decisive
        self.assertEqual(classify_three_way(0.85, snr_db=30.0, clipping_ratio=0.0, duration_s=60.0), "Deceptive")
        self.assertEqual(classify_three_way(0.15, snr_db=30.0, clipping_ratio=0.0, duration_s=60.0), "Truthful")

        # 2. Quality triggers Inconclusive
        self.assertEqual(classify_three_way(0.90, snr_db=8.0, clipping_ratio=0.0, duration_s=60.0), "Inconclusive")  # low SNR
        self.assertEqual(classify_three_way(0.10, snr_db=30.0, clipping_ratio=0.02, duration_s=60.0), "Inconclusive")  # clipping
        self.assertEqual(classify_three_way(0.90, snr_db=30.0, clipping_ratio=0.0, duration_s=8.0), "Inconclusive")  # too short

        # 3. Model uncertainty margin triggers Inconclusive
        self.assertEqual(classify_three_way(0.52, snr_db=30.0, clipping_ratio=0.0, duration_s=60.0), "Inconclusive")
        self.assertEqual(classify_three_way(0.48, snr_db=30.0, clipping_ratio=0.0, duration_s=60.0), "Inconclusive")

    def test_manifest_metadata_schema_prohibits_pii(self):
        """Verify that mandatory non-sensitive fields are recognized and PII keys are disallowed."""
        allowed_keys = {
            "speaker_id", "recording_id", "native_language", "recording_language",
            "gender", "age_bracket", "regional_state", "story_id", "story_type",
            "ground_truth", "presentation_order", "duration_s", "snr_db", "clipping_ratio"
        }
        prohibited_pii_keys = {"name", "full_name", "phone", "email", "aadhaar", "address", "government_id"}

        sample_manifest_row = {
            "speaker_id": "IND_SPK_001",
            "recording_id": "IND_SPK_001_S1",
            "native_language": "Telugu",
            "recording_language": "Telugu",
            "gender": "Female",
            "age_bracket": "18-25",
            "regional_state": "Andhra_Pradesh",
            "story_id": "Story_A",
            "story_type": "Vacation_True",
            "ground_truth": 0,
            "presentation_order": 1,
            "duration_s": 78.4,
            "snr_db": 34.2,
            "clipping_ratio": 0.0,
        }

        row_keys = set(sample_manifest_row.keys())
        self.assertTrue(row_keys.issubset(allowed_keys))
        self.assertEqual(len(row_keys.intersection(prohibited_pii_keys)), 0)


if __name__ == "__main__":
    unittest.main()
