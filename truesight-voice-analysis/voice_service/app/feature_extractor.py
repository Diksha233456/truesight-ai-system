"""
feature_extractor.py

Extracts acoustic features from a preprocessed waveform using:
- Librosa: MFCCs, tempo, RMS energy, zero-crossing rate, spectral shape
- Praat-Parselmouth: pitch (F0), jitter, shimmer, and pause/silence timing

All extraction happens in `FeatureExtractor.extract`, which returns a
VoiceFeatures pydantic object plus the raw feature vector (numpy array)
that the predictor consumes.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call as praat_call

from app.model import VoiceFeatures
from app.utils import LoadedAudio

logger = logging.getLogger("voice_service.features")

# Praat pitch tracking bounds tuned for typical adult speech.
F0_MIN = 75.0
F0_MAX = 500.0

# Silence detection threshold, in dB below the signal's max intensity,
# used by Praat's "To TextGrid (silences)" pause detector.
SILENCE_THRESHOLD_DB = -25.0
MIN_SILENT_INTERVAL_S = 0.15
MIN_SOUNDING_INTERVAL_S = 0.10

N_MFCC = 13


class FeatureExtractionError(RuntimeError):
    """Raised when acoustic feature extraction fails on an otherwise-valid file."""


class FeatureExtractor:
    """Stateless extractor: one call per audio sample."""

    def extract(self, audio: LoadedAudio) -> Tuple[VoiceFeatures, np.ndarray]:
        """
        Run the full extraction pipeline.

        Returns:
            (VoiceFeatures, flat_vector) where flat_vector is the numeric
            array fed to the Random Forest predictor.
        """
        try:
            pitch_feats = self._extract_pitch_and_voice_quality(audio)
            timing_feats = self._extract_timing(audio)
            spectral_feats = self._extract_spectral(audio)
            mfcc_mean, mfcc_std = self._extract_mfcc(audio)
        except Exception as e:
            logger.exception("Feature extraction failed for %s", audio.source_filename)
            raise FeatureExtractionError(f"Failed to extract acoustic features: {e}") from e

        features = VoiceFeatures(
            pitch_mean=pitch_feats["pitch_mean"],
            pitch_std=pitch_feats["pitch_std"],
            pitch_min=pitch_feats["pitch_min"],
            pitch_max=pitch_feats["pitch_max"],
            jitter_local=pitch_feats["jitter_local"],
            shimmer_local=pitch_feats["shimmer_local"],
            speech_tempo=timing_feats["speech_tempo"],
            pause_ratio=timing_feats["pause_ratio"],
            pause_count=timing_feats["pause_count"],
            silence_duration=timing_feats["silence_duration"],
            rms_energy_mean=spectral_feats["rms_energy_mean"],
            rms_energy_std=spectral_feats["rms_energy_std"],
            zero_crossing_rate=spectral_feats["zero_crossing_rate"],
            spectral_centroid=spectral_feats["spectral_centroid"],
            spectral_bandwidth=spectral_feats["spectral_bandwidth"],
            spectral_rolloff=spectral_feats["spectral_rolloff"],
            mfcc_mean=mfcc_mean.tolist(),
            mfcc_std=mfcc_std.tolist(),
            is_pitch_reliable=pitch_feats.get("is_pitch_reliable", True),
            is_jitter_reliable=pitch_feats.get("is_jitter_reliable", True),
            is_shimmer_reliable=pitch_feats.get("is_shimmer_reliable", True),
            voiced_ratio=pitch_feats.get("voiced_ratio", 0.0),
        )

        vector = self.to_vector(features)
        return features, vector

    # ------------------------------------------------------------------
    # Pitch, jitter, shimmer (Parselmouth)
    # ------------------------------------------------------------------
    def _extract_pitch_and_voice_quality(self, audio: LoadedAudio) -> dict:
        sound = parselmouth.Sound(audio.samples, sampling_frequency=audio.sample_rate)

        # Support adult vocal diversity (accommodates lower baritone to higher female/youth fundamental ranges)
        pitch = sound.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
        f0_values = pitch.selected_array["frequency"]
        voiced = f0_values[f0_values > 0]

        total_frames = len(f0_values)
        voiced_ratio = float(len(voiced) / total_frames) if total_frames > 0 else 0.0

        # Pitch reliability criteria: at least 15 voiced frames (~150ms of phonation)
        is_pitch_reliable = len(voiced) >= 15

        if not is_pitch_reliable:
            if voiced.size > 0:
                pitch_mean = float(np.mean(voiced))
                pitch_std = float(np.std(voiced))
                pitch_min = float(np.min(voiced))
                pitch_max = float(np.max(voiced))
            else:
                pitch_mean = pitch_std = pitch_min = pitch_max = 0.0
        else:
            pitch_mean = float(np.mean(voiced))
            pitch_std = float(np.std(voiced))
            pitch_min = float(np.min(voiced))
            pitch_max = float(np.max(voiced))

        # Jitter and shimmer require sufficient glottal pulses from PointProcess
        # In speech phonetics, fewer than 20 glottal cycles produces meaningless jitter/shimmer.
        is_jitter_reliable = False
        is_shimmer_reliable = False
        jitter_local = 0.0
        shimmer_local = 0.0

        try:
            point_process = praat_call(sound, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)
            num_points = int(praat_call(point_process, "Get number of points"))

            # Require at least 25 glottal periods for statistically meaningful perturbation
            if num_points >= 25 and is_pitch_reliable:
                raw_jitter = praat_call(
                    point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3
                )
                raw_shimmer = praat_call(
                    [sound, point_process],
                    "Get shimmer (local)",
                    0.0, 0.0, 0.0001, 0.02, 1.3, 1.6,
                )

                if raw_jitter is not None and not np.isnan(raw_jitter) and 0.0 <= raw_jitter < 0.035:
                    jitter_local = float(raw_jitter)
                    is_jitter_reliable = True
                else:
                    jitter_local = float(raw_jitter) if (raw_jitter is not None and not np.isnan(raw_jitter) and raw_jitter >= 0.0) else 0.0
                    is_jitter_reliable = False

                if raw_shimmer is not None and not np.isnan(raw_shimmer) and 0.0 <= raw_shimmer < 0.35:
                    shimmer_local = float(raw_shimmer)
                    is_shimmer_reliable = True
                else:
                    shimmer_local = float(raw_shimmer) if (raw_shimmer is not None and not np.isnan(raw_shimmer) and raw_shimmer >= 0.0) else 0.0
                    is_shimmer_reliable = False
            else:
                # Under insufficient glottal pulses, set to 0.0 in the feature vector to preserve
                # 42-dim schema, but mark reliability flag as False.
                jitter_local = 0.0
                shimmer_local = 0.0
                is_jitter_reliable = False
                is_shimmer_reliable = False

        except Exception as e:
            logger.warning("Jitter/shimmer extraction failed gracefully: %s", e)
            jitter_local = 0.0
            shimmer_local = 0.0
            is_jitter_reliable = False
            is_shimmer_reliable = False

        return {
            "pitch_mean": pitch_mean,
            "pitch_std": pitch_std,
            "pitch_min": pitch_min,
            "pitch_max": pitch_max,
            "jitter_local": jitter_local,
            "shimmer_local": shimmer_local,
            "is_pitch_reliable": is_pitch_reliable,
            "is_jitter_reliable": is_jitter_reliable,
            "is_shimmer_reliable": is_shimmer_reliable,
            "voiced_ratio": voiced_ratio,
        }

    # ------------------------------------------------------------------
    # Tempo, pauses, silence (Parselmouth silence detection + librosa onsets)
    # ------------------------------------------------------------------
    def _extract_timing(self, audio: LoadedAudio) -> dict:
        duration = audio.duration_seconds

        # Speech tempo approximated via onset strength / detected onsets per second.
        onset_env = librosa.onset.onset_strength(y=audio.samples, sr=audio.sample_rate)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=audio.sample_rate, units="time"
        )
        speech_tempo = float(len(onsets) / duration) if duration > 0 else 0.0

        # Pause/silence detection via Praat intensity-based silence finder.
        sound = parselmouth.Sound(audio.samples, sampling_frequency=audio.sample_rate)
        silence_duration = 0.0
        pause_count = 0
        pause_ratio = 0.0

        try:
            intensity = sound.to_intensity()
            textgrid = praat_call(
                intensity,
                "To TextGrid (silences)",
                SILENCE_THRESHOLD_DB,
                MIN_SILENT_INTERVAL_S,
                MIN_SOUNDING_INTERVAL_S,
                "silent",
                "sounding",
            )
            n_intervals = int(praat_call(textgrid, "Get number of intervals", 1))
            for i in range(1, n_intervals + 1):
                label = praat_call(textgrid, "Get label of interval", 1, i)
                if label == "silent":
                    # Use "Get start time of interval" — the correct Praat command name
                    start = praat_call(textgrid, "Get start time of interval", 1, i)
                    end = praat_call(textgrid, "Get end time of interval", 1, i)
                    seg_len = float(end) - float(start)
                    silence_duration += seg_len
                    pause_count += 1
            pause_ratio = float(silence_duration / duration) if duration > 0 else 0.0
        except Exception as e:
            logger.warning("Silence/pause detection failed, defaulting to 0: %s", e)
            silence_duration = 0.0
            pause_count = 0
            pause_ratio = 0.0

        return {
            "speech_tempo": speech_tempo,
            "pause_ratio": pause_ratio,
            "pause_count": pause_count,
            "silence_duration": float(silence_duration),
        }

    # ------------------------------------------------------------------
    # Energy + spectral shape (Librosa)
    # ------------------------------------------------------------------
    def _extract_spectral(self, audio: LoadedAudio) -> dict:
        y, sr = audio.samples, audio.sample_rate

        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

        return {
            "rms_energy_mean": float(np.mean(rms)),
            "rms_energy_std": float(np.std(rms)),
            "zero_crossing_rate": float(np.mean(zcr)),
            "spectral_centroid": float(np.mean(centroid)),
            "spectral_bandwidth": float(np.mean(bandwidth)),
            "spectral_rolloff": float(np.mean(rolloff)),
        }

    # ------------------------------------------------------------------
    # MFCCs (Librosa)
    # ------------------------------------------------------------------
    def _extract_mfcc(self, audio: LoadedAudio) -> Tuple[np.ndarray, np.ndarray]:
        mfcc = librosa.feature.mfcc(y=audio.samples, sr=audio.sample_rate, n_mfcc=N_MFCC)
        return np.mean(mfcc, axis=1), np.std(mfcc, axis=1)

    # ------------------------------------------------------------------
    # Vectorization for the ML model
    # ------------------------------------------------------------------
    def to_vector(self, features: VoiceFeatures) -> np.ndarray:
        """
        Flatten VoiceFeatures into a fixed-order numeric vector.
        Order MUST match train_model.py's FEATURE_ORDER exactly.
        """
        base = [
            features.pitch_mean,
            features.pitch_std,
            features.pitch_min,
            features.pitch_max,
            features.jitter_local,
            features.shimmer_local,
            features.speech_tempo,
            features.pause_ratio,
            float(features.pause_count),
            features.silence_duration,
            features.rms_energy_mean,
            features.rms_energy_std,
            features.zero_crossing_rate,
            features.spectral_centroid,
            features.spectral_bandwidth,
            features.spectral_rolloff,
        ]
        return np.array(base + features.mfcc_mean + features.mfcc_std, dtype=np.float32)


# Canonical feature name order — used by both this module (implicitly, via
# to_vector's fixed ordering) and train_model.py so the two never drift apart.
FEATURE_ORDER = (
    [
        "pitch_mean", "pitch_std", "pitch_min", "pitch_max",
        "jitter_local", "shimmer_local",
        "speech_tempo", "pause_ratio", "pause_count", "silence_duration",
        "rms_energy_mean", "rms_energy_std",
        "zero_crossing_rate", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
    ]
    + [f"mfcc_mean_{i}" for i in range(N_MFCC)]
    + [f"mfcc_std_{i}" for i in range(N_MFCC)]
)
