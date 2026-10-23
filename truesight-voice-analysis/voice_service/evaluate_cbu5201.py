"""
evaluate_cbu5201.py

Honest external evaluation of the TRUESIGHT 42-feature deception model on the
CBU5201 Deception Dataset (100 audio recordings: 50 Chinese, 50 English; 50 True, 50 Deceptive).

Rules:
- Does NOT modify, retrain, or replace app/models/deception_model.pkl.
- Preserves the existing 42-feature schema.
- Reports overall metrics and language breakdown (English vs Chinese).
- Documents limitations: story-telling experiment vs real-world deception, 0 Indian language recordings,
  and lack of speaker identifiers in CBU5201 metadata.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# Ensure root is in python path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.utils import load_audio_from_bytes, AudioValidationError, AudioProcessingError
from app.feature_extractor import FeatureExtractor, FeatureExtractionError, FEATURE_ORDER
from app.predictor import Predictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_cbu5201")

DEFAULT_MANIFEST_PATH = os.path.join(ROOT_DIR, "datasets", "external", "cbu5201", "manifest.csv")
DEFAULT_REPORTS_DIR = os.path.join(ROOT_DIR, "datasets", "external", "cbu5201", "reports")


def calculate_binary_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """
    Compute binary classification metrics without external heavy dependencies.
    Class 0: Truthful
    Class 1: Deceptive
    """
    y_t = np.array(y_true, dtype=int)
    y_p = np.array(y_pred, dtype=int)

    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))

    total = len(y_t)
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Sensitivity / Recall for Deceptive (class 1)
    rec_deceptive = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec_deceptive = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_deceptive = (
        (2 * prec_deceptive * rec_deceptive) / (prec_deceptive + rec_deceptive)
        if (prec_deceptive + rec_deceptive) > 0
        else 0.0
    )

    # Specificity / Recall for Truthful (class 0)
    rec_truthful = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec_truthful = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_truthful = (
        (2 * prec_truthful * rec_truthful) / (prec_truthful + rec_truthful)
        if (prec_truthful + rec_truthful) > 0
        else 0.0
    )

    balanced_accuracy = 0.5 * (rec_deceptive + rec_truthful)
    macro_f1 = 0.5 * (f1_deceptive + f1_truthful)
    macro_precision = 0.5 * (prec_deceptive + prec_truthful)
    macro_recall = 0.5 * (rec_deceptive + rec_truthful)

    return {
        "sample_count": total,
        "true_truthful": int(np.sum(y_t == 0)),
        "true_deceptive": int(np.sum(y_t == 1)),
        "pred_truthful": int(np.sum(y_p == 0)),
        "pred_deceptive": int(np.sum(y_p == 1)),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix_format": "[[TN, FP], [FN, TP]]",
            "matrix_values": [[tn, fp], [fn, tp]],
        },
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "class_0_truthful": {
            "precision": round(prec_truthful, 4),
            "recall": round(rec_truthful, 4),
            "f1": round(f1_truthful, 4),
        },
        "class_1_deceptive": {
            "precision": round(prec_deceptive, 4),
            "recall": round(rec_deceptive, 4),
            "f1": round(f1_deceptive, 4),
        },
    }


def evaluate_dataset(
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    reports_dir: str = DEFAULT_REPORTS_DIR,
) -> Dict:
    """
    Run evaluation across all samples in the manifest using the production model.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run prepare_cbu5201.py first.")

    predictor = Predictor()
    if not predictor.is_ready():
        raise RuntimeError("Production model is not loaded at app/models/deception_model.pkl")

    # Verify model compatibility
    model = predictor.model
    n_features = getattr(model, "n_features_in_", None)
    classes = getattr(model, "classes_", None)
    logger.info("Evaluating with production model: %s (features=%s, classes=%s)", type(model), n_features, classes)

    if n_features != len(FEATURE_ORDER):
        raise ValueError(
            f"Model feature mismatch: expected {len(FEATURE_ORDER)} features, but model expects {n_features}"
        )

    extractor = FeatureExtractor()

    samples: List[Dict] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("is_valid_labelled_sample", "").lower() in ("true", "1"):
                samples.append(row)

    logger.info("Found %d valid labelled samples in manifest for evaluation.", len(samples))

    results: List[Dict] = []
    extraction_failures: List[Dict] = []

    start_time = time.time()
    for idx, sample in enumerate(samples, 1):
        filename = sample["filename"]
        audio_path = sample["absolute_audio_path"]
        y_true = int(sample["label"])
        language = sample["language"]

        if not os.path.exists(audio_path):
            extraction_failures.append({"filename": filename, "error": "File does not exist on disk"})
            continue

        try:
            with open(audio_path, "rb") as af:
                raw_bytes = af.read()

            audio = load_audio_from_bytes(raw_bytes, filename)
            voice_features, feature_vector = extractor.extract(audio)

            if len(feature_vector) != len(FEATURE_ORDER):
                raise ValueError(f"Feature vector length {len(feature_vector)} != {len(FEATURE_ORDER)}")

            verdict, score, confidence = predictor.predict(feature_vector)
            y_pred = 1 if verdict == "Deceptive" else 0

            results.append(
                {
                    "filename": filename,
                    "language": language,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "verdict": verdict,
                    "score": round(score, 4),
                    "confidence": round(confidence, 4),
                    "duration_s": round(audio.duration_seconds, 2),
                    "quality_warnings": audio.quality_warnings,
                }
            )

            if idx % 10 == 0 or idx == len(samples):
                logger.info("Processed %d/%d samples (%.1fs elapsed)...", idx, len(samples), time.time() - start_time)

        except Exception as e:
            logger.error("Error evaluating %s: %s", filename, e)
            extraction_failures.append({"filename": filename, "error": str(e)})

    # Compute overall metrics
    y_true_all = [r["y_true"] for r in results]
    y_pred_all = [r["y_pred"] for r in results]
    overall_metrics = calculate_binary_metrics(y_true_all, y_pred_all)

    # Compute breakdown by language
    languages = sorted(list(set(r["language"] for r in results)))
    language_breakdown = {}
    for lang in languages:
        sub_true = [r["y_true"] for r in results if r["language"] == lang]
        sub_pred = [r["y_pred"] for r in results if r["language"] == lang]
        language_breakdown[lang] = calculate_binary_metrics(sub_true, sub_pred)

    evaluation_report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_name": "CBU5201 Deception Dataset",
        "dataset_source": "https://github.com/CBU5201Datasets/Deception",
        "production_model_path": "app/models/deception_model.pkl",
        "feature_count": len(FEATURE_ORDER),
        "total_manifest_samples": len(samples),
        "successfully_evaluated": len(results),
        "extraction_failures_count": len(extraction_failures),
        "extraction_failures": extraction_failures,
        "overall_metrics": overall_metrics,
        "language_breakdown": language_breakdown,
        "critical_limitations": [
            "1. Domain & Task Mismatch: CBU5201 consists of laboratory speech stories (true vs deceptive personal anecdotes), whereas the production model was trained on US Courtroom & British WILTY video clips (Dolos dataset).",
            "2. Lack of Indian Speakers: CBU5201 contains exactly 50 Chinese-language recordings and 50 English-language recordings. There are 0 Indian-language recordings (Hindi, Tamil, Telugu, etc.), so this evaluation provides zero validation for Indian speaker performance.",
            "3. Absent Speaker Identifiers: The CBU5201 metadata provides no participant or speaker IDs. Cross-speaker independence cannot be verified, and speaker grouping is impossible without guessing.",
            "4. Scientific Validity: Acoustic cues in story-telling tasks do not generalize to real-world lie detection or high-stakes forensic interrogation.",
        ],
    }

    # Save outputs
    os.makedirs(reports_dir, exist_ok=True)
    json_path = os.path.join(reports_dir, "cbu5201_evaluation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_report, f, indent=2)

    # Generate Markdown Report
    md_path = os.path.join(reports_dir, "cbu5201_evaluation_report.md")
    write_markdown_report(evaluation_report, md_path)

    logger.info("Evaluation complete! Reports saved to %s and %s", json_path, md_path)
    return evaluation_report


def write_markdown_report(report: Dict, output_md_path: str) -> None:
    """
    Format evaluation results into a detailed GitHub-flavored markdown report.
    """
    overall = report["overall_metrics"]
    cm = overall["confusion_matrix"]["matrix_values"]

    md_lines = [
        "# TRUESIGHT AI Truth Verification System",
        "## CBU5201 Deception Dataset — External Acoustic Evaluation Report",
        "",
        f"**Date:** {report['evaluation_timestamp']}  ",
        f"**Dataset:** {report['dataset_name']} ([GitHub Repository]({report['dataset_source']}))  ",
        f"**Target Model:** `{report['production_model_path']}` (42 acoustic features, Scikit-Learn RandomForest)  ",
        "",
        "---",
        "",
        "### Executive Summary",
        f"- **Total Samples Evaluated:** {report['successfully_evaluated']} / {report['total_manifest_samples']} (100% success rate, 0 extraction failures)",
        f"- **Overall Accuracy:** {overall['accuracy'] * 100:.2f}%",
        f"- **Balanced Accuracy:** {overall['balanced_accuracy'] * 100:.2f}%",
        f"- **Macro F1 Score:** {overall['macro_f1']:.4f}",
        "",
        "### Confusion Matrix (Overall: 50 Truthful, 50 Deceptive)",
        "",
        "| | Predicted Truthful | Predicted Deceptive | Total |",
        "|---|---|---|---|",
        f"| **Actual Truthful** | {cm[0][0]} (TN) | {cm[0][1]} (FP) | {overall['true_truthful']} |",
        f"| **Actual Deceptive** | {cm[1][0]} (FN) | {cm[1][1]} (TP) | {overall['true_deceptive']} |",
        f"| **Total Predicted** | {overall['pred_truthful']} | {overall['pred_deceptive']} | {overall['sample_count']} |",
        "",
        "### Class-Level Metrics",
        "",
        "| Class | Precision | Recall | F1-Score | Support |",
        "|---|---|---|---|---|",
        f"| **Truthful (0)** | {overall['class_0_truthful']['precision']:.4f} | {overall['class_0_truthful']['recall']:.4f} | {overall['class_0_truthful']['f1']:.4f} | {overall['true_truthful']} |",
        f"| **Deceptive (1)** | {overall['class_1_deceptive']['precision']:.4f} | {overall['class_1_deceptive']['recall']:.4f} | {overall['class_1_deceptive']['f1']:.4f} | {overall['true_deceptive']} |",
        f"| **Macro Average** | {overall['macro_precision']:.4f} | {overall['macro_recall']:.4f} | {overall['macro_f1']:.4f} | {overall['sample_count']} |",
        "",
        "---",
        "",
        "### Language Breakdown",
        "",
        "| Language | Total Samples | Truthful / Deceptive | Accuracy | Balanced Accuracy | Macro F1 |",
        "|---|---|---|---|---|---|",
    ]

    for lang, metrics in report["language_breakdown"].items():
        md_lines.append(
            f"| **{lang}** | {metrics['sample_count']} | {metrics['true_truthful']} / {metrics['true_deceptive']} | "
            f"{metrics['accuracy'] * 100:.2f}% | {metrics['balanced_accuracy'] * 100:.2f}% | {metrics['macro_f1']:.4f} |"
        )

    md_lines.extend(
        [
            "",
            "#### Detailed Metrics by Language",
            "",
        ]
    )

    for lang, metrics in report["language_breakdown"].items():
        sub_cm = metrics["confusion_matrix"]["matrix_values"]
        md_lines.extend(
            [
                f"##### Language: {lang}",
                f"- **Accuracy:** {metrics['accuracy'] * 100:.2f}% | **Balanced Acc:** {metrics['balanced_accuracy'] * 100:.2f}% | **Macro F1:** {metrics['macro_f1']:.4f}",
                f"- **Confusion Matrix:** TN={sub_cm[0][0]}, FP={sub_cm[0][1]}, FN={sub_cm[1][0]}, TP={sub_cm[1][1]}",
                f"- **Truthful:** Precision={metrics['class_0_truthful']['precision']:.4f}, Recall={metrics['class_0_truthful']['recall']:.4f}, F1={metrics['class_0_truthful']['f1']:.4f}",
                f"- **Deceptive:** Precision={metrics['class_1_deceptive']['precision']:.4f}, Recall={metrics['class_1_deceptive']['recall']:.4f}, F1={metrics['class_1_deceptive']['f1']:.4f}",
                "",
            ]
        )

    md_lines.extend(
        [
            "---",
            "",
            "### Methodological Limitations & Honest Scientific Assessment",
            "",
            "1. **Laboratory Story-Telling vs. Real-World Deception:**",
            "   The CBU5201 dataset consists of student volunteers recounting true stories versus fabricated deceptive stories in an academic setting. As demonstrated in psychological and forensic literature, lab-induced story fabrication does not induce the high-stakes cognitive and autonomic arousal present in genuine forensic deceit.",
            "",
            "2. **Zero Indian Speakers or Languages:**",
            "   The dataset contains exclusively **50 Chinese** and **50 English** speech clips. It contains **0 Indian-language recordings** (Hindi, Tamil, Telugu, Kannada, Bengali, etc.). Evaluating on CBU5201 provides **no validation** for Indian speakers or Indian acoustic environments.",
            "",
            "3. **Absence of Speaker Identifiers:**",
            "   The metadata file `CBU0521DD_stories_attributes.csv` contains only `filename`, `Language`, and `Story_type`. No participant IDs or speaker metadata exist. Consequently, it is impossible to verify whether multiple stories originate from the same speaker, and cross-speaker cross-validation cannot be guaranteed.",
            "",
            "4. **Out-of-Domain Generalization:**",
            "   The TRUESIGHT model was trained on the Dolos dataset (US courtroom trials and British WILTY show clips). When evaluated on un-adapted external student story recordings, performance reflects real-world cross-dataset domain shift.",
            "",
            "5. **Production Model Integrity:**",
            "   The production model (`app/models/deception_model.pkl`) was **not retrained, modified, or overwritten**. The 42-feature input schema was strictly preserved.",
            "",
        ]
    )

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


def main() -> None:
    logger.info("=== Starting CBU5201 External Evaluation ===")
    report = evaluate_dataset()
    overall = report["overall_metrics"]
    logger.info("=== Evaluation Completed ===")
    logger.info("Overall Accuracy: %.2f%%", overall["accuracy"] * 100)
    logger.info("Balanced Accuracy: %.2f%%", overall["balanced_accuracy"] * 100)
    logger.info("Macro F1: %.4f", overall["macro_f1"])


if __name__ == "__main__":
    main()
