#!/usr/bin/env python3
"""
evaluate_indian_english.py - Acoustic robustness evaluation of TRUESIGHT on Indian-accented English.

Evaluates:
1. Audio quality and extraction success rate on Indian-accented English speech.
2. Descriptive acoustic distributions (F0, formants, perturbations, rhythm, spectral).
3. Direct methodological comparison with CBU5201 English recordings, detailing confounding factors.
4. Heuristic explanation audit: empirical check on whether app/explanation.py thresholds
   flag normal Indian-accent prosody (syllable timing, pause patterns) as deceptive cues.
"""

import os
import glob
import json
import logging
import numpy as np
from typing import Dict, List, Any

from app.utils import load_audio_from_bytes
from app.feature_extractor import FeatureExtractor, FEATURE_ORDER

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_indian_english")


def calculate_descriptive_stats(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "iqr": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    q75, q25 = np.percentile(arr, [75, 25])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "iqr": float(q75 - q25),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": int(len(arr))
    }


def analyze_cohort(audio_files: List[str], cohort_name: str) -> Dict[str, Any]:
    extractor = FeatureExtractor()
    results = []
    failed = []

    # Explanation heuristic thresholds from app/explanation.py
    TEMPO_FAST = 3.5
    TEMPO_SLOW = 1.5
    PAUSE_RATIO_HIGH = 0.30
    PITCH_STD_HIGH = 35.0
    JITTER_HIGH = 0.02
    SHIMMER_HIGH = 0.05
    HNR_LOW = 10.0

    trigger_counts = {
        "tempo_fast": 0,
        "tempo_slow": 0,
        "pause_ratio_high": 0,
        "pitch_std_high": 0,
        "jitter_high": 0,
        "shimmer_high": 0,
        "hnr_low": 0
    }

    for filepath in audio_files:
        filename = os.path.basename(filepath)
        try:
            with open(filepath, "rb") as f:
                content = f.read()

            audio = load_audio_from_bytes(content, filename)
            voice_feats, vector = extractor.extract(audio)

            if len(vector) != len(FEATURE_ORDER):
                failed.append({"file": filename, "reason": f"invalid_vector_dim_{len(vector)}"})
                continue

            feats_dict = voice_feats.model_dump() if hasattr(voice_feats, "model_dump") else voice_feats.dict()

            # Check explanation triggers
            tempo = float(feats_dict.get("speech_tempo", 0.0))
            pause_ratio = float(feats_dict.get("pause_ratio", 0.0))
            pitch_std = float(feats_dict.get("pitch_std", 0.0))
            jitter = float(feats_dict.get("jitter_local", 0.0))
            shimmer = float(feats_dict.get("shimmer_local", 0.0))
            hnr = float(feats_dict.get("hnr_mean", 0.0))

            if tempo > TEMPO_FAST:
                trigger_counts["tempo_fast"] += 1
            if 0 < tempo < TEMPO_SLOW:
                trigger_counts["tempo_slow"] += 1
            if pause_ratio > PAUSE_RATIO_HIGH:
                trigger_counts["pause_ratio_high"] += 1
            if pitch_std > PITCH_STD_HIGH:
                trigger_counts["pitch_std_high"] += 1
            if jitter > JITTER_HIGH:
                trigger_counts["jitter_high"] += 1
            if shimmer > SHIMMER_HIGH:
                trigger_counts["shimmer_high"] += 1
            if 0 < hnr < HNR_LOW:
                trigger_counts["hnr_low"] += 1

            record = {
                "file": filename,
                "duration": audio.duration_seconds,
                "sr": audio.sample_rate,
                "snr": audio.snr_estimate_db,
                "clipping": audio.clipping_ratio,
                "features": feats_dict
            }
            results.append(record)
        except Exception as e:
            failed.append({"file": filename, "reason": str(e)})

    # Aggregate statistics
    total = len(audio_files)
    success_count = len(results)
    success_rate = (success_count / total * 100.0) if total > 0 else 0.0

    durations = [r["duration"] for r in results]
    snrs = [r["snr"] for r in results]
    clippings = [r["clipping"] for r in results]

    feature_keys = [
        "pitch_mean", "pitch_std", "pitch_min", "pitch_max",
        "jitter_local", "shimmer_local", "hnr_mean",
        "speech_tempo", "pause_ratio", "pause_count", "silence_duration",
        "spectral_centroid", "spectral_rolloff", "zero_crossing_rate", "rms_mean"
    ]

    feature_stats = {}
    for k in feature_keys:
        vals = [r["features"].get(k, np.nan) for r in results]
        feature_stats[k] = calculate_descriptive_stats(vals)

    return {
        "cohort": cohort_name,
        "total_files": total,
        "successful_extractions": success_count,
        "failed_extractions": len(failed),
        "failure_rate_pct": float(100.0 - success_rate),
        "success_rate_pct": float(success_rate),
        "quality_metrics": {
            "duration": calculate_descriptive_stats(durations),
            "snr_db": calculate_descriptive_stats(snrs),
            "clipping_ratio": calculate_descriptive_stats(clippings)
        },
        "acoustic_features": feature_stats,
        "explanation_heuristic_triggers": {
            k: {
                "count": v,
                "percentage": float((v / success_count * 100.0) if success_count > 0 else 0.0)
            }
            for k, v in trigger_counts.items()
        },
        "failures": failed
    }


def main():
    ie_files = sorted(glob.glob("datasets/external/svarah/audio/*.wav"))
    cbu_files = sorted(glob.glob("datasets/external/cbu5201/extracted/Deception-main/CBU0521DD_stories/*.wav"))

    logger.info(f"Evaluating Indian English cohort ({len(ie_files)} files)...")
    ie_results = analyze_cohort(ie_files, "Indian_English_Sample")

    logger.info(f"Evaluating CBU5201 English cohort ({len(cbu_files)} files)...")
    cbu_results = analyze_cohort(cbu_files, "CBU5201_English_Stories")

    output_data = {
        "evaluation_title": "TRUESIGHT Stage 7: Indian-English Acoustic Domain Evaluation",
        "methodology": {
            "feature_schema": "42 acoustic features (Parselmouth F0/jitter/shimmer + Librosa MFCC/spectral/rhythm)",
            "indian_english_cohort": {
                "source": "Indian English Speech Corpus (svarah directory)",
                "sample_size": len(ie_files),
                "accent": "Indian-accented English (L1 Indic substrate: Indo-Aryan / Dravidian)",
                "task": "Read narrative speech",
                "recording_setup": "Studio/close-talk microphone, 48 kHz mono"
            },
            "cbu5201_cohort": {
                "source": "CBU0521DD Deception Dataset Stories",
                "sample_size": len(cbu_files),
                "accent": "UK/British-accented English (predominant CBU university cohort)",
                "task": "Autobiographical storytelling (truthful & deceptive narrative recall)",
                "recording_setup": "Room microphone / video camera audio, 48 kHz stereo"
            },
            "critical_distinction": (
                "Neither cohort in this domain evaluation provides valid Indian-English deception ground truth. "
                "The evaluation measures acoustic pipeline reliability, feature distributions, and explanation "
                "bias risks. Zero deception accuracy improvement is claimed."
            )
        },
        "cohort_evaluations": {
            "indian_english": ie_results,
            "cbu5201_english": cbu_results
        }
    }

    out_path = "datasets/external/svarah/indian_english_robustness_metrics.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved evaluation metrics to {out_path}")
    print("\n================== EVALUATION REPORT DATA ==================")
    print(f"Indian English Cohort: {ie_results['total_files']} files | Success: {ie_results['success_rate_pct']:.1f}%")
    print(f"  Duration: {ie_results['quality_metrics']['duration']['mean']:.2f}s (std: {ie_results['quality_metrics']['duration']['std']:.2f})")
    print(f"  SNR: {ie_results['quality_metrics']['snr_db']['mean']:.2f} dB (std: {ie_results['quality_metrics']['snr_db']['std']:.2f})")
    print(f"  Pitch Mean: {ie_results['acoustic_features']['pitch_mean']['mean']:.2f} Hz (std: {ie_results['acoustic_features']['pitch_mean']['std']:.2f})")
    print(f"  Pitch Std: {ie_results['acoustic_features']['pitch_std']['mean']:.2f} Hz (std: {ie_results['acoustic_features']['pitch_std']['std']:.2f})")
    print(f"  Jitter Local: {ie_results['acoustic_features']['jitter_local']['mean']:.5f}")
    print(f"  Shimmer Local: {ie_results['acoustic_features']['shimmer_local']['mean']:.5f}")
    print(f"  Speech Tempo: {ie_results['acoustic_features']['speech_tempo']['mean']:.2f} onsets/s")
    print(f"  Pause Ratio: {ie_results['acoustic_features']['pause_ratio']['mean']:.4f}")
    print(f"  Spectral Centroid: {ie_results['acoustic_features']['spectral_centroid']['mean']:.2f} Hz")
    print(f"  Heuristic 'Tempo Fast (>3.5)' triggers: {ie_results['explanation_heuristic_triggers']['tempo_fast']['percentage']:.1f}% ({ie_results['explanation_heuristic_triggers']['tempo_fast']['count']}/{ie_results['successful_extractions']})")
    print(f"  Heuristic 'Pause Ratio High (>0.30)' triggers: {ie_results['explanation_heuristic_triggers']['pause_ratio_high']['percentage']:.1f}% ({ie_results['explanation_heuristic_triggers']['pause_ratio_high']['count']}/{ie_results['successful_extractions']})")
    print(f"  Heuristic 'Pitch Std High (>35.0)' triggers: {ie_results['explanation_heuristic_triggers']['pitch_std_high']['percentage']:.1f}% ({ie_results['explanation_heuristic_triggers']['pitch_std_high']['count']}/{ie_results['successful_extractions']})")

    print(f"\nCBU5201 English Cohort: {cbu_results['total_files']} files | Success: {cbu_results['success_rate_pct']:.1f}%")
    print(f"  Duration: {cbu_results['quality_metrics']['duration']['mean']:.2f}s (std: {cbu_results['quality_metrics']['duration']['std']:.2f})")
    print(f"  SNR: {cbu_results['quality_metrics']['snr_db']['mean']:.2f} dB (std: {cbu_results['quality_metrics']['snr_db']['std']:.2f})")
    print(f"  Pitch Mean: {cbu_results['acoustic_features']['pitch_mean']['mean']:.2f} Hz (std: {cbu_results['acoustic_features']['pitch_mean']['std']:.2f})")
    print(f"  Pitch Std: {cbu_results['acoustic_features']['pitch_std']['mean']:.2f} Hz (std: {cbu_results['acoustic_features']['pitch_std']['std']:.2f})")
    print(f"  Jitter Local: {cbu_results['acoustic_features']['jitter_local']['mean']:.5f}")
    print(f"  Shimmer Local: {cbu_results['acoustic_features']['shimmer_local']['mean']:.5f}")
    print(f"  Speech Tempo: {cbu_results['acoustic_features']['speech_tempo']['mean']:.2f} onsets/s")
    print(f"  Pause Ratio: {cbu_results['acoustic_features']['pause_ratio']['mean']:.4f}")
    print(f"  Spectral Centroid: {cbu_results['acoustic_features']['spectral_centroid']['mean']:.2f} Hz")
    print(f"  Heuristic 'Tempo Fast (>3.5)' triggers: {cbu_results['explanation_heuristic_triggers']['tempo_fast']['percentage']:.1f}% ({cbu_results['explanation_heuristic_triggers']['tempo_fast']['count']}/{cbu_results['successful_extractions']})")
    print(f"  Heuristic 'Pause Ratio High (>0.30)' triggers: {cbu_results['explanation_heuristic_triggers']['pause_ratio_high']['percentage']:.1f}% ({cbu_results['explanation_heuristic_triggers']['pause_ratio_high']['count']}/{cbu_results['successful_extractions']})")
    print(f"  Heuristic 'Pitch Std High (>35.0)' triggers: {cbu_results['explanation_heuristic_triggers']['pitch_std_high']['percentage']:.1f}% ({cbu_results['explanation_heuristic_triggers']['pitch_std_high']['count']}/{cbu_results['successful_extractions']})")


if __name__ == "__main__":
    main()
