"""
evaluate_indic_acoustic_robustness.py

Acoustic Robustness & Feature Extraction Evaluation Pipeline for Indian-Language Speech.

PURPOSE & ETHICAL BOUNDS:
1. This pipeline measures ACOUSTIC ROBUSTNESS, signal quality, and normative feature
   distributions across Indian-language speech samples.
2. It DOES NOT classify truthful vs deceptive behavior, as ordinary speech recordings
   lack veracity ground-truth labels.
3. It preserves the strict 42-feature schema and tests whether Praat-Parselmouth and
   Librosa feature extractors execute reliably on regional Indian speech without crashing,
   NaN propagation, or numerical instability.
4. It computes language-group and speaker-group quality and acoustic statistics.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Ensure root is in sys.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.utils import (
    load_audio_from_bytes,
    AudioValidationError,
    AudioProcessingError,
    LoadedAudio,
    TARGET_SR,
)
from app.feature_extractor import FeatureExtractor, FEATURE_ORDER, VoiceFeatures

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_indic_robustness")


class IndicAcousticEvaluator:
    """
    Evaluator for analyzing acoustic feature extraction robustness, signal quality,
    and normative prosodic distributions on Indian-language speech recordings.
    """

    def __init__(self):
        self.extractor = FeatureExtractor()

    def process_single_audio(
        self,
        audio_bytes: bytes,
        filename: str,
        language: str = "UNKNOWN",
        speaker_id: str = "UNKNOWN",
    ) -> Dict:
        """
        Process a single audio file through the TRUESIGHT ingestion and 42-feature pipeline.
        Records acoustic measurements and quality flags without making veracity claims.
        """
        try:
            audio = load_audio_from_bytes(audio_bytes, filename)
        except AudioValidationError as ave:
            return {
                "filename": filename,
                "language": language,
                "speaker_id": speaker_id,
                "success": False,
                "error_type": "ValidationError",
                "error_message": str(ave),
            }
        except AudioProcessingError as ape:
            return {
                "filename": filename,
                "language": language,
                "speaker_id": speaker_id,
                "success": False,
                "error_type": "ProcessingError",
                "error_message": str(ape),
            }
        except Exception as e:
            return {
                "filename": filename,
                "language": language,
                "speaker_id": speaker_id,
                "success": False,
                "error_type": "UnexpectedDecodeError",
                "error_message": str(e),
            }

        try:
            features, vector = self.extractor.extract(audio)
        except Exception as fe_err:
            return {
                "filename": filename,
                "language": language,
                "speaker_id": speaker_id,
                "success": False,
                "error_type": "FeatureExtractionError",
                "error_message": str(fe_err),
            }

        # Validate vector integrity
        is_finite = bool(np.all(np.isfinite(vector)))
        has_correct_dim = len(vector) == len(FEATURE_ORDER)

        return {
            "filename": filename,
            "language": language,
            "speaker_id": speaker_id,
            "success": True,
            "duration_seconds": round(audio.duration_seconds, 3),
            "sample_rate": audio.sample_rate,
            "raw_rms": round(audio.raw_rms, 5),
            "raw_peak": round(audio.raw_peak, 5),
            "clipping_ratio": round(audio.clipping_ratio, 5),
            "snr_estimate_db": round(audio.snr_estimate_db, 2),
            "quality_warnings": audio.quality_warnings,
            "is_vector_finite": is_finite,
            "vector_dimension": len(vector),
            "has_correct_dimension": has_correct_dim,
            "reliability_flags": {
                "is_pitch_reliable": getattr(features, "is_pitch_reliable", True),
                "is_jitter_reliable": getattr(features, "is_jitter_reliable", True),
                "is_shimmer_reliable": getattr(features, "is_shimmer_reliable", True),
                "voiced_ratio": round(getattr(features, "voiced_ratio", 0.0), 4),
            },
            "key_acoustic_measurements": {
                "pitch_mean_hz": round(features.pitch_mean, 2),
                "pitch_std_hz": round(features.pitch_std, 2),
                "jitter_local": round(features.jitter_local, 5),
                "shimmer_local": round(features.shimmer_local, 5),
                "speech_tempo": round(features.speech_tempo, 2),
                "pause_ratio": round(features.pause_ratio, 4),
                "pause_count": features.pause_count,
                "silence_duration_s": round(features.silence_duration, 3),
                "spectral_centroid_hz": round(features.spectral_centroid, 2),
                "spectral_rolloff_hz": round(features.spectral_rolloff, 2),
                "zero_crossing_rate": round(features.zero_crossing_rate, 4),
            },
        }

    def evaluate_manifest(self, manifest_rows: List[Dict]) -> Dict:
        """
        Evaluate a collection of audio entries (from a manifest dictionary or CSV).
        """
        results: List[Dict] = []
        failures: List[Dict] = []

        start_time = time.time()
        for idx, row in enumerate(manifest_rows, 1):
            filename = row.get("filename", f"sample_{idx}.wav")
            audio_path = row.get("absolute_audio_path") or row.get("audio_path")
            language = row.get("language", "UNKNOWN")
            speaker_id = row.get("speaker_id", "UNKNOWN")

            if not audio_path or not os.path.exists(audio_path):
                failures.append({
                    "filename": filename,
                    "language": language,
                    "speaker_id": speaker_id,
                    "error_type": "FileNotFound",
                    "error_message": f"Audio path does not exist on disk: {audio_path}",
                })
                continue

            try:
                with open(audio_path, "rb") as af:
                    raw_bytes = af.read()
                res = self.process_single_audio(raw_bytes, filename, language, speaker_id)
                if res.get("success"):
                    results.append(res)
                else:
                    failures.append(res)
            except Exception as read_err:
                failures.append({
                    "filename": filename,
                    "language": language,
                    "speaker_id": speaker_id,
                    "error_type": "FileReadError",
                    "error_message": str(read_err),
                })

        duration = time.time() - start_time
        summary = self.aggregate_robustness_metrics(results, failures)
        summary["processing_time_seconds"] = round(duration, 2)
        return summary

    def aggregate_robustness_metrics(
        self,
        results: List[Dict],
        failures: List[Dict],
    ) -> Dict:
        """
        Aggregate acoustic statistics and signal quality metrics across languages.
        """
        total_attempted = len(results) + len(failures)
        success_count = len(results)
        failure_count = len(failures)

        # Language breakdown
        by_language: Dict[str, List[Dict]] = {}
        for r in results:
            lang = r.get("language", "UNKNOWN")
            by_language.setdefault(lang, []).append(r)

        lang_summaries = {}
        for lang, lang_results in by_language.items():
            durations = [r["duration_seconds"] for r in lang_results]
            snrs = [r["snr_estimate_db"] for r in lang_results]
            clippings = [r["clipping_ratio"] for r in lang_results]
            pitches = [
                r["key_acoustic_measurements"]["pitch_mean_hz"]
                for r in lang_results
                if r["reliability_flags"]["is_pitch_reliable"] and r["key_acoustic_measurements"]["pitch_mean_hz"] > 0
            ]
            jitters = [
                r["key_acoustic_measurements"]["jitter_local"]
                for r in lang_results
                if r["reliability_flags"]["is_jitter_reliable"]
            ]
            shimmers = [
                r["key_acoustic_measurements"]["shimmer_local"]
                for r in lang_results
                if r["reliability_flags"]["is_shimmer_reliable"]
            ]
            tempos = [r["key_acoustic_measurements"]["speech_tempo"] for r in lang_results]
            pause_ratios = [r["key_acoustic_measurements"]["pause_ratio"] for r in lang_results]
            warning_counts = sum(len(r.get("quality_warnings", [])) for r in lang_results)

            lang_summaries[lang] = {
                "evaluated_samples": len(lang_results),
                "total_quality_warnings": warning_counts,
                "mean_duration_s": round(float(np.mean(durations)), 2) if durations else 0.0,
                "mean_snr_db": round(float(np.mean(snrs)), 2) if snrs else 0.0,
                "max_clipping_ratio": round(float(np.max(clippings)), 4) if clippings else 0.0,
                "pitch_mean_hz": round(float(np.mean(pitches)), 1) if pitches else 0.0,
                "pitch_std_hz": round(float(np.std(pitches)), 1) if pitches else 0.0,
                "jitter_local_mean": round(float(np.mean(jitters)), 4) if jitters else 0.0,
                "shimmer_local_mean": round(float(np.mean(shimmers)), 4) if shimmers else 0.0,
                "tempo_mean": round(float(np.mean(tempos)), 2) if tempos else 0.0,
                "pause_ratio_mean": round(float(np.mean(pause_ratios)), 3) if pause_ratios else 0.0,
            }

        return {
            "total_samples_attempted": total_attempted,
            "successfully_processed": success_count,
            "pipeline_extraction_failures": failure_count,
            "robustness_success_rate": round(success_count / total_attempted, 4) if total_attempted > 0 else 0.0,
            "failures": failures,
            "language_group_summaries": lang_summaries,
            "scientific_disclaimer": (
                "ACOUSTIC ROBUSTNESS NOTICE: This pipeline assesses acoustic feature stability, "
                "sampling rate conversion, and signal quality across speech samples. These measurements "
                "represent objective physical vocal properties (F0, perturbation, energy, tempo) and "
                "DO NOT establish whether a speaker is lying or telling the truth. Deception detection "
                "requires verified ground-truth veracity labels, which ordinary speech corpora do not provide."
            ),
        }


def generate_indic_evaluation_report(
    summary: Dict,
    output_json_path: str,
    output_md_path: str,
    dataset_metadata: Optional[Dict] = None,
) -> None:
    """
    Save both JSON and Markdown summaries of the acoustic robustness evaluation.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    meta = dataset_metadata or {}
    dataset_name = meta.get("name", "Indian Speech Corpora")
    dataset_source = meta.get("source", "Open Speech / Indic Resources")

    md_lines = [
        f"# TRUESIGHT Indian-Language Speech Acoustic Robustness Report",
        f"**Dataset / Corpus:** {dataset_name}  ",
        f"**Source:** {dataset_source}  ",
        f"**Evaluation Type:** Acoustic Feature Extraction Robustness & Normative Analysis (Non-Veracity)  ",
        "",
        "---",
        "",
        "### 1. Robustness Summary",
        f"- **Total Samples Processed:** {summary['total_samples_attempted']}",
        f"- **Successful Feature Extractions (42-dim):** {summary['successfully_processed']}",
        f"- **Extraction Failures:** {summary['pipeline_extraction_failures']}",
        f"- **Pipeline Robustness Rate:** {summary['robustness_success_rate'] * 100:.2f}%",
        "",
        "### 2. Language Group Acoustic Norms",
        "",
        "| Language | Evaluated | Mean Duration (s) | Mean SNR (dB) | F0 Pitch Mean (Hz) | Local Jitter | Local Shimmer | Tempo (onsets/s) | Pause Ratio |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for lang, s in summary.get("language_group_summaries", {}).items():
        md_lines.append(
            f"| **{lang}** | {s['evaluated_samples']} | {s['mean_duration_s']}s | {s['mean_snr_db']} dB | "
            f"{s['pitch_mean_hz']} Hz | {s['jitter_local_mean']} | {s['shimmer_local_mean']} | "
            f"{s['tempo_mean']} | {s['pause_ratio_mean']} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "### 3. Essential Scientific & Methodological Boundaries",
        "1. **Robustness vs. Deception:**",
        "   This pipeline validates that TRUESIGHT's 42 acoustic feature extractors (Praat-Parselmouth pitch/jitter/shimmer "
        "   and Librosa spectral/MFCC algorithms) operate stably on Indian phonetics and mobile audio without crashing. "
        "   It **does not** and **cannot** perform lie detection on unlabelled speech.",
        "",
        "2. **Acoustic Diversity Across Indic Languages:**",
        "   Syllable-timed cadence in languages like Tamil, Telugu, Kannada, and Hindi differs markedly from Western "
        "   stress-timed English. Calibrating normal tempo and pause boundaries prevents false-positive classifications "
        "   in multi-lingual deployment.",
        "",
        "3. **Zero Veracity Labels:**",
        "   No truthful or deceptive labels were assigned to any ordinary speech recording.",
    ])

    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    logger.info("Saved reports to %s and %s", output_json_path, output_md_path)
