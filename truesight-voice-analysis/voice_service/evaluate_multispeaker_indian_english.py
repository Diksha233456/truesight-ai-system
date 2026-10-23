#!/usr/bin/env python3
"""
evaluate_multispeaker_indian_english.py

Stage 9 Multi-Speaker Indian-English Acoustic Evaluation:
- Ingests 50 real-world Indian English recordings across 46 unique speakers/sessions.
- Runs the TRUESIGHT 42-feature acoustic pipeline (Parselmouth + Librosa).
- Reports:
  1. Pipeline success/failure rates and dimension validity.
  2. Audio quality statistics (duration, SNR, digital clipping).
  3. Feature reliability gating (pitch reliability, jitter/shimmer glottal period checks).
  4. Speaker-level and cohort-level normative acoustic distributions.
  5. Audit of Stage 8 explanation safeguards across diverse accents and recording conditions.
  6. Strict ethical adherence: NO deception scores, NO veracity labels, NO model modifications.
"""

import os
import csv
import json
import logging
import numpy as np
from typing import Dict, List, Any

from app.utils import load_audio_from_bytes, AudioValidationError, AudioProcessingError
from app.feature_extractor import FeatureExtractor, FEATURE_ORDER
from app.explanation import generate_structured_explanation, generate_explanations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_multispeaker")

DATASET_DIR = os.path.join("datasets", "external", "indian_english_multispeaker")
MANIFEST_PATH = os.path.join(DATASET_DIR, "manifest.csv")
OUTPUT_METRICS_PATH = os.path.join(DATASET_DIR, "multispeaker_evaluation_metrics.json")


def calculate_stats(values: List[float]) -> Dict[str, float]:
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


def main():
    if not os.path.exists(MANIFEST_PATH):
        logger.error(f"Manifest not found at {MANIFEST_PATH}")
        return

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        manifest_rows = list(reader)

    logger.info(f"Loaded {len(manifest_rows)} manifest entries.")

    extractor = FeatureExtractor()
    results = []
    failures = []

    # Safeguard check counts: ensure Stage 8 fixes prevent biased claims
    safeguard_checks = {
        "urgency_claims": 0,
        "stress_claims": 0,
        "nervousness_claims": 0,
        "deception_claims": 0,
        "hesitation_claims": 0,
        "inconclusive_explanations_for_unreliable": 0,
        "accent_limitation_present_count": 0,
    }

    for idx, row in enumerate(manifest_rows, 1):
        fn = row["filename"]
        audio_rel_path = row["audio_path"]
        session_id = row.get("session_id", "UNKNOWN")
        speaker_id = row.get("speaker_id", session_id)
        transcript = row.get("transcript", "")

        audio_full_path = os.path.abspath(audio_rel_path)
        if not os.path.exists(audio_full_path):
            failures.append({
                "filename": fn,
                "speaker_id": speaker_id,
                "error_type": "FileNotFound",
                "message": f"File does not exist: {audio_full_path}"
            })
            continue

        try:
            with open(audio_full_path, "rb") as f_aud:
                audio_bytes = f_aud.read()

            loaded = load_audio_from_bytes(audio_bytes, fn)
            features, vector = extractor.extract(loaded)

            if len(vector) != len(FEATURE_ORDER):
                failures.append({
                    "filename": fn,
                    "speaker_id": speaker_id,
                    "error_type": "InvalidVectorDimension",
                    "message": f"Vector length {len(vector)} != {len(FEATURE_ORDER)}"
                })
                continue

            feats_dict = features.model_dump() if hasattr(features, "model_dump") else features.dict()

            # Test Stage 8 explanation behavior
            dummy_score = 0.50
            dummy_confidence = 0.50
            structured_expl = generate_structured_explanation(
                features=features,
                score=dummy_score,
                confidence=dummy_confidence,
                quality_warnings=loaded.quality_warnings
            )
            legacy_notes = generate_explanations(
                features=features,
                score=dummy_score,
                quality_warnings=loaded.quality_warnings
            )

            # Audit safeguard violations (ensure no positive claims of stress, nervousness, urgency, hesitation, or deceit)
            all_expl_text = " ".join(
                structured_expl.acoustic_observations +
                structured_expl.quality_warnings +
                legacy_notes
            ).lower()

            # Filter out explicit negative disclaimers
            eval_text = all_expl_text
            for neg in [
                "does not indicate urgency, stress, or deception",
                "does not indicate stress or deception",
                "does not indicate vocal distress or deception",
                "do not indicate vocal distress or deception",
                "does not indicate deception",
                "do not prove deception, stress, or nervousness",
                "do not prove deception",
                "not definitive deception",
                "vocal distress"
            ]:
                eval_text = eval_text.replace(neg, "")

            import re
            if re.search(r"\burgency\b", eval_text):
                safeguard_checks["urgency_claims"] += 1
            if re.search(r"\bstress\b", eval_text):
                safeguard_checks["stress_claims"] += 1
            if re.search(r"\bnervous(ness)?\b", eval_text):
                safeguard_checks["nervousness_claims"] += 1
            if re.search(r"\bhesitation\b", eval_text):
                safeguard_checks["hesitation_claims"] += 1
            if re.search(r"\b(deceptive|lying|deceit)\b", eval_text):
                safeguard_checks["deception_claims"] += 1

            if any("inconclusive" in w.lower() for w in structured_expl.quality_warnings):
                safeguard_checks["inconclusive_explanations_for_unreliable"] += 1

            if any("indian-accented english" in lim.lower() for lim in structured_expl.limitations):
                safeguard_checks["accent_limitation_present_count"] += 1

            rec = {
                "filename": fn,
                "session_id": session_id,
                "speaker_id": speaker_id,
                "prompt_token": row.get("prompt_token", ""),
                "transcript": transcript,
                "duration_seconds": loaded.duration_seconds,
                "sample_rate": loaded.sample_rate,
                "snr_db": loaded.snr_estimate_db,
                "clipping_ratio": loaded.clipping_ratio,
                "quality_warnings": loaded.quality_warnings,
                "reliability_flags": {
                    "is_pitch_reliable": getattr(features, "is_pitch_reliable", True),
                    "is_jitter_reliable": getattr(features, "is_jitter_reliable", True),
                    "is_shimmer_reliable": getattr(features, "is_shimmer_reliable", True),
                    "voiced_ratio": getattr(features, "voiced_ratio", 0.0),
                },
                "features": feats_dict,
                "structured_explanation": structured_expl.model_dump() if hasattr(structured_expl, "model_dump") else structured_expl.dict(),
                "legacy_explanation": legacy_notes,
            }
            results.append(rec)

        except (AudioValidationError, AudioProcessingError) as ape:
            failures.append({
                "filename": fn,
                "speaker_id": speaker_id,
                "error_type": type(ape).__name__,
                "message": str(ape)
            })
        except Exception as e:
            failures.append({
                "filename": fn,
                "speaker_id": speaker_id,
                "error_type": "UnexpectedExtractionError",
                "message": str(e)
            })

    total_files = len(manifest_rows)
    success_files = len(results)
    success_rate = (success_files / total_files * 100.0) if total_files > 0 else 0.0

    # Aggregate cohort acoustic metrics
    durations = [r["duration_seconds"] for r in results]
    snrs = [r["snr_db"] for r in results]
    clippings = [r["clipping_ratio"] for r in results]

    feature_keys = [
        "pitch_mean", "pitch_std", "pitch_min", "pitch_max",
        "jitter_local", "shimmer_local",
        "speech_tempo", "pause_ratio", "pause_count", "silence_duration",
        "spectral_centroid", "spectral_rolloff", "zero_crossing_rate", "rms_energy_mean"
    ]

    cohort_feature_stats = {}
    for k in feature_keys:
        vals = [r["features"].get(k, np.nan) for r in results]
        cohort_feature_stats[k] = calculate_stats(vals)

    # Reliability breakdown
    pitch_reliable_count = sum(1 for r in results if r["reliability_flags"]["is_pitch_reliable"])
    jitter_reliable_count = sum(1 for r in results if r["reliability_flags"]["is_jitter_reliable"])
    shimmer_reliable_count = sum(1 for r in results if r["reliability_flags"]["is_shimmer_reliable"])

    # Speaker / Session aggregation (group clips by speaker_id)
    speaker_groups: Dict[str, List[Dict]] = {}
    for r in results:
        spk = r["speaker_id"]
        speaker_groups.setdefault(spk, []).append(r)

    speaker_summaries = {}
    for spk, clips in speaker_groups.items():
        spk_pitches = [c["features"]["pitch_mean"] for c in clips if c["reliability_flags"]["is_pitch_reliable"] and c["features"]["pitch_mean"] > 0]
        spk_tempos = [c["features"]["speech_tempo"] for c in clips]
        spk_snrs = [c["snr_db"] for c in clips]
        spk_durations = [c["duration_seconds"] for c in clips]

        # Inferred gender band from pitch: male typically < 160 Hz, female typically >= 160 Hz
        mean_p = float(np.mean(spk_pitches)) if spk_pitches else 0.0
        gender_band = "Female-range (>=160 Hz)" if mean_p >= 160.0 else ("Male-range (<160 Hz)" if mean_p > 0 else "Unvoiced/Inconclusive")

        speaker_summaries[spk] = {
            "clip_count": len(clips),
            "total_duration_s": round(sum(spk_durations), 2),
            "inferred_pitch_band": gender_band,
            "mean_f0_hz": round(mean_p, 2),
            "mean_tempo_onsets_sec": round(float(np.mean(spk_tempos)), 2) if spk_tempos else 0.0,
            "mean_snr_db": round(float(np.mean(spk_snrs)), 2) if spk_snrs else 0.0,
        }

    # Gender band distribution across unique speakers
    male_range_spks = sum(1 for s in speaker_summaries.values() if "Male-range" in s["inferred_pitch_band"])
    female_range_spks = sum(1 for s in speaker_summaries.values() if "Female-range" in s["inferred_pitch_band"])
    inconclusive_spks = sum(1 for s in speaker_summaries.values() if "Inconclusive" in s["inferred_pitch_band"])

    # Recording condition breakdown (Acoustic environment: High SNR >= 20 dB vs Real-world SNR < 20 dB)
    clean_clips = [r for r in results if r["snr_db"] >= 20.0]
    noisy_clips = [r for r in results if r["snr_db"] < 20.0]

    # Utterance type breakdown: Short command (< 2.5s) vs Narrative utterance (>= 2.5s)
    short_clips = [r for r in results if r["duration_seconds"] < 2.5]
    long_clips = [r for r in results if r["duration_seconds"] >= 2.5]

    output_payload = {
        "evaluation_title": "TRUESIGHT Stage 9: Multi-Speaker Indian-English Domain Evaluation",
        "dataset_metadata": {
            "source": "Santhosh-kumar/Indian-Accent-Dataset (Hugging Face)",
            "license": "Apache-2.0",
            "access_type": "Open (Non-gated, direct access)",
            "total_clips_cataloged": total_files,
            "unique_speakers_sessions": len(speaker_groups),
            "speech_genre": "Crowdsourced Indian English conversational queries, commands, e-commerce, and general knowledge",
            "geographic_context": "India (regional brands, banking, geography, and local context)",
        },
        "critical_boundary": (
            "This multi-speaker evaluation measures acoustic compatibility, signal quality, and explanation "
            "fairness across diverse speakers. Because the dataset contains NO truth/deception ground truth, "
            "NO deception scores or accuracy metrics were calculated, and the production model was NOT retrained."
        ),
        "execution_summary": {
            "total_files": total_files,
            "successful_extractions": success_files,
            "failed_extractions": len(failures),
            "success_rate_pct": round(success_rate, 2),
            "unique_speakers_evaluated": len(speaker_groups),
            "speaker_pitch_diversity": {
                "male_frequency_band_speakers": male_range_spks,
                "female_frequency_band_speakers": female_range_spks,
                "inconclusive_pitch_speakers": inconclusive_spks,
            },
            "reliability_gating_summary": {
                "pitch_reliable_clips": pitch_reliable_count,
                "pitch_unreliable_clips": success_files - pitch_reliable_count,
                "jitter_reliable_clips": jitter_reliable_count,
                "jitter_unreliable_clips": success_files - jitter_reliable_count,
                "shimmer_reliable_clips": shimmer_reliable_count,
                "shimmer_unreliable_clips": success_files - shimmer_reliable_count,
            },
            "explanation_safeguards_audit": safeguard_checks,
            "quality_metrics": {
                "duration": calculate_stats(durations),
                "snr_db": calculate_stats(snrs),
                "clipping_ratio": calculate_stats(clippings),
            },
            "acoustic_features": cohort_feature_stats,
            "subgroup_comparisons": {
                "acoustic_environment": {
                    "high_snr_count": len(clean_clips),
                    "high_snr_f0_mean": float(np.mean([c["features"]["pitch_mean"] for c in clean_clips])) if clean_clips else 0.0,
                    "low_snr_count": len(noisy_clips),
                    "low_snr_f0_mean": float(np.mean([c["features"]["pitch_mean"] for c in noisy_clips])) if noisy_clips else 0.0,
                },
                "utterance_length": {
                    "short_commands_count": len(short_clips),
                    "short_commands_pitch_unreliable_rate": float((sum(1 for c in short_clips if not c["reliability_flags"]["is_pitch_reliable"]) / len(short_clips) * 100.0) if short_clips else 0.0),
                    "narrative_count": len(long_clips),
                    "narrative_pitch_unreliable_rate": float((sum(1 for c in long_clips if not c["reliability_flags"]["is_pitch_reliable"]) / len(long_clips) * 100.0) if long_clips else 0.0),
                }
            },
            "speaker_summaries": speaker_summaries,
            "failures": failures,
        }
    }

    with open(OUTPUT_METRICS_PATH, "w", encoding="utf-8") as f_out:
        json.dump(output_payload, f_out, indent=2)

    logger.info(f"Evaluation results successfully saved to {OUTPUT_METRICS_PATH}")

    print("\n================ STAGE 9 MULTI-SPEAKER EVALUATION SUMMARY ================")
    print(f"Total files: {total_files} | Successful extractions: {success_files} ({success_rate:.1f}%)")
    print(f"Unique speakers/sessions evaluated: {len(speaker_groups)}")
    print(f"  Male-range F0 speakers: {male_range_spks}")
    print(f"  Female-range F0 speakers: {female_range_spks}")
    print(f"  Inconclusive F0 speakers: {inconclusive_spks}")
    print(f"\nCohort Acoustic Distributions:")
    print(f"  Mean Duration: {output_payload['execution_summary']['quality_metrics']['duration']['mean']:.2f}s (range: {output_payload['execution_summary']['quality_metrics']['duration']['min']:.2f}s - {output_payload['execution_summary']['quality_metrics']['duration']['max']:.2f}s)")
    print(f"  Mean SNR: {output_payload['execution_summary']['quality_metrics']['snr_db']['mean']:.2f} dB (range: {output_payload['execution_summary']['quality_metrics']['snr_db']['min']:.2f} dB - {output_payload['execution_summary']['quality_metrics']['snr_db']['max']:.2f} dB)")
    print(f"  Mean F0: {cohort_feature_stats['pitch_mean']['mean']:.2f} Hz (std: {cohort_feature_stats['pitch_mean']['std']:.2f})")
    print(f"  Mean Speech Tempo: {cohort_feature_stats['speech_tempo']['mean']:.2f} onsets/s (median: {cohort_feature_stats['speech_tempo']['median']:.2f})")
    print(f"  Mean Pause Ratio: {cohort_feature_stats['pause_ratio']['mean']:.4f}")
    print(f"\nExplanation Safeguard Audit:")
    print(f"  Urgency claims generated: {safeguard_checks['urgency_claims']}")
    print(f"  Stress claims generated: {safeguard_checks['stress_claims']}")
    print(f"  Nervousness claims generated: {safeguard_checks['nervousness_claims']}")
    print(f"  Deception claims generated: {safeguard_checks['deception_claims']}")
    print(f"  Hesitation claims generated: {safeguard_checks['hesitation_claims']}")
    print(f"  Inconclusive warnings for short/unvoiced clips: {safeguard_checks['inconclusive_explanations_for_unreliable']}")
    print(f"  Accent limitation explicitly included: {safeguard_checks['accent_limitation_present_count']}/{success_files}")


if __name__ == "__main__":
    main()
