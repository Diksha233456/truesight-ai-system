"""
utils.py

Shared audio I/O and preprocessing utilities for the Voice Analysis module.

Responsibilities:
- Safely load uploaded audio bytes into a normalized, mono, float32 waveform.
- Provide basic validation helpers (format, duration, silence checks).
- Centralize sampling-rate handling so every other module works with a
  consistent (audio, sample_rate) contract.
"""

from __future__ import annotations

import io
import os
import tempfile
import uuid
import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import soundfile as sf
import librosa

logger = logging.getLogger("voice_service.utils")

# Automatically ensure ffmpeg is available on PATH for librosa / audioread webm decoding
try:
    import shutil
    import imageio_ffmpeg
    _ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    _ffmpeg_dir = os.path.dirname(_ffmpeg_exe)
    _canonical_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    _canonical_path = os.path.join(_ffmpeg_dir, _canonical_name)
    if not os.path.exists(_canonical_path):
        try:
            shutil.copy2(_ffmpeg_exe, _canonical_path)
        except Exception:
            pass
    if _ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
except Exception as _ffmpeg_init_err:
    logger.debug("imageio_ffmpeg setup note: %s", _ffmpeg_init_err)

import sys
_scripts_dir = os.path.join(sys.prefix, "Scripts" if os.name == "nt" else "bin")
if os.path.isdir(_scripts_dir) and _scripts_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _scripts_dir + os.pathsep + os.environ.get("PATH", "")

# Target sample rate used throughout the pipeline. Librosa and Parselmouth
# both work fine at 16kHz for speech, and it keeps feature extraction fast.
TARGET_SR = 16000

# Formats we accept via soundfile/librosa. Anything else is rejected early
# with a clear error rather than failing deep inside feature extraction.
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".mpeg", ".mpg", ".webm"}

# Reasonable bounds so a corrupt or absurd file doesn't hang the service.
MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 300.0


class AudioValidationError(ValueError):
    """Raised when an uploaded file fails validation before feature extraction."""


class AudioProcessingError(RuntimeError):
    """Raised when a file passes validation but decoding/processing fails."""


@dataclass
class LoadedAudio:
    """Container for a fully preprocessed audio signal."""
    samples: np.ndarray  # float32, mono, normalized, shape (n_samples,)
    sample_rate: int
    duration_seconds: float
    source_filename: str
    raw_rms: float = 0.0
    raw_peak: float = 0.0
    clipping_ratio: float = 0.0
    snr_estimate_db: float = 0.0
    quality_warnings: list = None
    vad_stats: dict = None

    def __post_init__(self):
        if self.quality_warnings is None:
            self.quality_warnings = []
        if self.vad_stats is None:
            self.vad_stats = {}


def validate_filename(filename: str) -> None:
    """Raise AudioValidationError if the filename/extension is unsupported."""
    if not filename or "." not in filename:
        raise AudioValidationError(
            "Uploaded file has no discernible extension. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise AudioValidationError(
            f"Unsupported audio format '{ext}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def _to_mono(samples: np.ndarray) -> np.ndarray:
    """Collapse a (n_samples, n_channels) or (n_channels, n_samples) array to mono."""
    if samples.ndim == 1:
        return samples
    # soundfile returns shape (n_samples, n_channels)
    return np.mean(samples, axis=1)


def _normalize(samples: np.ndarray) -> np.ndarray:
    """Peak-normalize to [-1, 1] while avoiding division by zero on silence."""
    peak = np.max(np.abs(samples)) if samples.size else 0.0
    if peak < 1e-8:
        return samples.astype(np.float32)
    return (samples / peak).astype(np.float32)


# Extensions that require librosa/ffmpeg rather than soundfile for decoding.
_LIBROSA_ONLY_EXTENSIONS = {".webm", ".mp3", ".m4a", ".aac", ".mpeg", ".mpg"}


# Speech presence categorization constants
SPEECH_STATUS_VOICE_DETECTED = "VOICE DETECTED"
SPEECH_STATUS_VOICE_NOT_DETECTED = "VOICE NOT DETECTED"
SPEECH_STATUS_AUDIO_TOO_SHORT = "AUDIO TOO SHORT"
SPEECH_STATUS_AUDIO_QUALITY_TOO_LOW = "AUDIO QUALITY TOO LOW"


def load_audio_from_bytes(
    file_bytes: bytes,
    filename: str,
    allow_silence: bool = False,
    min_duration: float = MIN_DURATION_SECONDS,
) -> LoadedAudio:
    """
    Decode raw uploaded bytes into a validated, preprocessed mono waveform
    at TARGET_SR.

    Supports all SUPPORTED_EXTENSIONS including .webm (MediaRecorder/Opus output
    from browser getUserMedia), which requires ffmpeg to be on the system PATH.

    Raises:
        AudioValidationError: for bad filenames, empty files, or out-of-range duration.
        AudioProcessingError: for files that fail to decode despite a valid extension.
    """
    validate_filename(filename)

    if not file_bytes:
        raise AudioValidationError("Uploaded audio file is empty.")

    ext = os.path.splitext(filename)[1].lower()

    samples: np.ndarray
    sr: int

    if ext in _LIBROSA_ONLY_EXTENSIONS:
        # These formats require ffmpeg (via librosa/audioread). soundfile cannot
        # decode Opus/WebM natively, and audioread requires a file path on disk.
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(file_bytes)
            samples, sr = librosa.load(tmp_path, sr=None, mono=True)
        except Exception as librosa_error:
            logger.error("Failed to decode audio '%s' via librosa: %s", filename, librosa_error)
            raise AudioProcessingError(
                f"Could not decode '{ext}' audio. "
                "Ensure ffmpeg is installed and on the system PATH."
            ) from librosa_error
        finally:
            safe_remove(tmp_path)
    else:
        # First attempt: soundfile (fast in-memory, handles wav/flac/ogg natively).
        try:
            with io.BytesIO(file_bytes) as buf:
                samples, sr = sf.read(buf, always_2d=False, dtype="float32")
        except Exception as sf_error:
            # Fallback: write to temp file and use librosa/audioread/ffmpeg
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext)
            try:
                with os.fdopen(tmp_fd, "wb") as f:
                    f.write(file_bytes)
                samples, sr = librosa.load(tmp_path, sr=None, mono=False)
                samples = samples.T if samples.ndim > 1 else samples
            except Exception as librosa_error:
                logger.error(
                    "Failed to decode audio '%s': %s / %s",
                    filename, sf_error, librosa_error,
                )
                raise AudioProcessingError(
                    "Could not decode the uploaded audio file. It may be corrupted "
                    "or use an unsupported codec."
                ) from librosa_error
            finally:
                safe_remove(tmp_path)

    samples = _to_mono(np.asarray(samples))

    # Capture raw unnormalized energy levels before peak-normalizing.
    # Essential for distinguishing actual human speech from amplified room noise/hiss.
    raw_rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
    raw_peak = float(np.max(np.abs(samples))) if samples.size else 0.0

    # Digital clipping detection on the raw unnormalized signal
    # Clipping occurs when raw samples hit or exceed digital full scale (ceiling >= 0.95 and samples >= 0.99)
    clipping_ratio = 0.0
    if raw_peak >= 0.95:
        clipped_samples = np.sum(np.abs(samples) >= 0.99)
        clipping_ratio = float(clipped_samples / len(samples)) if len(samples) > 0 else 0.0

    quality_warnings = []
    if clipping_ratio > 0.01:
        quality_warnings.append(
            f"Audio signal clipping detected ({clipping_ratio * 100:.1f}% saturated). "
            "Microphone gain may be set too high; spectral features may be distorted."
        )

    # Estimate Signal-to-Noise Ratio (SNR) via 10th percentile energy floor vs peak frame energy
    snr_estimate_db = 0.0
    try:
        frame_len = int(TARGET_SR * 0.03)  # 30ms frames
        hop_len = int(TARGET_SR * 0.015)   # 15ms hop
        if len(samples) > frame_len:
            num_frames = 1 + (len(samples) - frame_len) // hop_len
            frame_energies = np.array([
                np.sqrt(np.mean(samples[i * hop_len : i * hop_len + frame_len] ** 2))
                for i in range(num_frames)
            ])
            noise_floor = float(np.percentile(frame_energies, 10))
            signal_peak = float(np.percentile(frame_energies, 95))
            effective_floor = max(noise_floor, 1e-5)
            if signal_peak > effective_floor:
                snr_estimate_db = float(20.0 * np.log10(signal_peak / effective_floor))
            else:
                snr_estimate_db = 0.0
    except Exception as snr_exc:
        logger.debug("SNR estimation skipped: %s", snr_exc)
        snr_estimate_db = 0.0

    if snr_estimate_db < 6.0 and raw_rms >= 0.003:
        quality_warnings.append(
            f"Low estimated signal-to-noise ratio ({snr_estimate_db:.1f} dB). "
            "High background noise (e.g. fan hum or mic hiss) may degrade acoustic precision."
        )

    if sr != TARGET_SR:
        samples = librosa.resample(samples.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    samples = _normalize(samples)

    duration = len(samples) / float(sr) if sr else 0.0
    if duration < min_duration:
        raise AudioValidationError(
            f"Audio too short ({duration:.2f}s). Minimum required: {min_duration:.2f}s."
        )
    if duration > MAX_DURATION_SECONDS:
        raise AudioValidationError(
            f"Audio too long ({duration:.2f}s). Maximum allowed: {MAX_DURATION_SECONDS}s."
        )

    # Reject effectively-silent recordings unless caller explicitly allows silence
    if not allow_silence and np.max(np.abs(samples)) < 1e-5:
        raise AudioValidationError("Audio appears to be silent; no signal detected.")

    return LoadedAudio(
        samples=samples,
        sample_rate=sr,
        duration_seconds=duration,
        source_filename=filename,
        raw_rms=raw_rms,
        raw_peak=raw_peak,
        clipping_ratio=clipping_ratio,
        snr_estimate_db=snr_estimate_db,
        quality_warnings=quality_warnings,
    )


def evaluate_audio_speech_quality(
    audio: LoadedAudio,
    features=None,
    min_duration: float = MIN_DURATION_SECONDS,
) -> Tuple[bool, str, str]:
    """
    Thoroughly evaluate audio for human speech presence and acoustic suitability.

    Distinguishes:
    1. VOICE NOT DETECTED: digital silence, ambient room noise, typing, clicks, fan hum.
    2. AUDIO TOO SHORT: insufficient audio or phonation duration.
    3. AUDIO QUALITY TOO LOW: extreme clipping (>15%) or severe SNR degradation.
    4. VOICE DETECTED: sufficient usable speech present.

    Returns:
        (has_speech: bool, reason: str, speech_status: str)
    """
    # 1. Duration check
    if audio.duration_seconds < min_duration:
        return (
            False,
            f"Audio duration ({audio.duration_seconds:.2f}s) is below the minimum required ({min_duration:.2f}s).",
            SPEECH_STATUS_AUDIO_TOO_SHORT,
        )

    # 2. Severe clipping check (only when raw signal saturates ADC ceiling)
    if audio.raw_peak >= 0.95 and audio.clipping_ratio > 0.15:
        return (
            False,
            f"Severe signal clipping detected ({audio.clipping_ratio * 100:.1f}% saturated). Microphone gain is too high.",
            SPEECH_STATUS_AUDIO_QUALITY_TOO_LOW,
        )

    # 3. Silence / near-silence / ambient room noise floor checks
    if audio.raw_peak < 1e-4:
        return (
            False,
            "Digital silence (near zero peak amplitude).",
            SPEECH_STATUS_VOICE_NOT_DETECTED,
        )

    # Ambient silence floor:
    # Conversational speech into a mic produces raw_peak >= 0.008.
    # Quiet rooms, laptop fans, breathing, and background mic hiss stay below these thresholds.
    if audio.raw_peak < 0.008 or (audio.raw_rms < 0.001 and audio.raw_peak < 0.015):
        return (
            False,
            f"Signal level is within ambient silence floor (raw RMS: {audio.raw_rms:.5f}, peak: {audio.raw_peak:.4f}).",
            SPEECH_STATUS_VOICE_NOT_DETECTED,
        )

    # Steady background noise / lack of speech contrast check:
    # Conversational speech has distinct syllable contrast against pauses (dynamic range >= 4.0 dB).
    # Ambient room hiss, fan noise, or AGC-amplified silence has nearly flat energy across frames.
    snr_db = getattr(audio, "snr_estimate_db", None)
    if snr_db is None or (snr_db == 0.0 and len(audio.samples) > int(audio.sample_rate * 0.03)):
        try:
            frame_len = int(audio.sample_rate * 0.03)
            hop_len = int(audio.sample_rate * 0.015)
            num_frames = 1 + (len(audio.samples) - frame_len) // hop_len
            frame_energies = np.array([
                np.sqrt(np.mean(audio.samples[i * hop_len : i * hop_len + frame_len] ** 2))
                for i in range(num_frames)
            ])
            noise_floor = float(np.percentile(frame_energies, 10))
            signal_peak = float(np.percentile(frame_energies, 95))
            effective_floor = max(noise_floor, 1e-5)
            if signal_peak > effective_floor:
                snr_db = float(20.0 * np.log10(signal_peak / effective_floor))
            else:
                snr_db = 0.0
            audio.snr_estimate_db = snr_db
        except Exception:
            snr_db = 0.0

    if audio.duration_seconds >= 3.0 and snr_db < 3.5:
        return (
            False,
            f"No speech dynamic contrast detected (estimated dynamic range: {snr_db:.1f} dB). Signal is flat ambient room noise or fan hum.",
            SPEECH_STATUS_VOICE_NOT_DETECTED,
        )

    # Speech-band energy ratio check:
    # Human speech formants (F1 ~300-900 Hz, F2 ~900-2500 Hz, F3 ~2500-3400 Hz) concentrate
    # most energy in the 300-3400 Hz telephony band. Fan hum (50/60/100/120 Hz), AC rumble,
    # and electrical noise concentrate energy below 300 Hz.
    # Require at least 15% of total spectral energy to be in the speech band for audio >= 3s.
    if audio.duration_seconds >= 3.0:
        try:
            fft_vals = np.abs(np.fft.rfft(audio.samples))
            freqs = np.fft.rfftfreq(len(audio.samples), d=1.0 / audio.sample_rate)
            total_energy = float(np.sum(fft_vals ** 2))
            if total_energy > 0:
                speech_band_energy = float(np.sum(fft_vals[(freqs >= 300.0) & (freqs <= 3400.0)] ** 2))
                speech_band_ratio = speech_band_energy / total_energy
                low_energy = float(np.sum(fft_vals[freqs < 250.0] ** 2))
                low_ratio = low_energy / total_energy

                # Reject if >85% energy below 250 Hz (fan motor / electrical hum)
                if low_ratio > 0.85:
                    return (
                        False,
                        f"Acoustic energy concentrated in sub-speech band ({low_ratio * 100:.1f}% < 250 Hz); consistent with fan motor or electrical hum.",
                        SPEECH_STATUS_VOICE_NOT_DETECTED,
                    )

                # Reject if <15% energy in speech band (300-3400 Hz)
                if speech_band_ratio < 0.15:
                    return (
                        False,
                        f"Insufficient speech-band energy ({speech_band_ratio * 100:.1f}% in 300–3400 Hz). Signal lacks human vocal formant energy.",
                        SPEECH_STATUS_VOICE_NOT_DETECTED,
                    )
        except Exception as fft_exc:
            logger.debug("FFT band check skipped: %s", fft_exc)

    # 4. Praat pitch tracking & harmonicity analysis
    try:
        import parselmouth
        from parselmouth.praat import call as praat_call

        sound = parselmouth.Sound(audio.samples, sampling_frequency=audio.sample_rate)

        # Pitch & Voicing Frames
        pitch = sound.to_pitch(pitch_floor=75.0, pitch_ceiling=500.0)
        f0 = pitch.selected_array["frequency"]
        voiced = f0[f0 > 0]
        voiced_frames = len(voiced)
        voiced_ratio = voiced_frames / len(f0) if len(f0) > 0 else 0.0

        # Mean F0 of voiced frames
        f0_mean = float(np.mean(voiced)) if voiced.size > 0 else 0.0

        # Harmonics-to-Noise Ratio (HNR)
        hnr_mean = float("nan")
        try:
            harmonicity = sound.to_harmonicity()
            hnr_mean = float(praat_call(harmonicity, "Get mean", 0, 0))
        except Exception:
            pass

        audio.vad_stats = {
            "voiced_frames": voiced_frames,
            "voiced_ratio": round(voiced_ratio, 4),
            "hnr_mean": round(hnr_mean, 2) if not np.isnan(hnr_mean) else None,
            "f0_mean": round(f0_mean, 2),
        }

        # F0 range check: Adult human speech F0 is 85-300 Hz (typical).
        # Fan hum at 100-120 Hz, electrical buzz at 50/60 Hz, and AC compressor drone
        # produce spurious pitch tracks below 135 Hz with very narrow pitch variance.
        # Reject if mean F0 < 75 Hz (sub-vocal) OR if low-frequency F0 is suspiciously steady
        # (std < 3 Hz and f0_mean < 135 Hz, indicating electrical mains/motor hum, not human voice).
        if voiced.size > 0 and audio.duration_seconds >= 3.0:
            f0_std = float(np.std(voiced))
            # Constant-frequency mains/motor hum (< 135 Hz and nearly zero variance)
            if f0_mean < 135.0 and f0_std < 3.0 and audio.duration_seconds >= 5.0:
                return (
                    False,
                    f"Constant-pitch hum detected (F0 mean={f0_mean:.0f} Hz, std={f0_std:.1f} Hz). "
                    "Consistent with motor hum or electrical noise, not human speech.",
                    SPEECH_STATUS_VOICE_NOT_DETECTED,
                )
            # Sub-vocal pitch: below typical human vocal range
            if f0_mean < 75.0:
                return (
                    False,
                    f"Detected pitch ({f0_mean:.0f} Hz) is below the human vocal range. "
                    "Consistent with mechanical vibration, fan hum, or electrical noise.",
                    SPEECH_STATUS_VOICE_NOT_DETECTED,
                )

        # Adaptive minimum phonation requirement based on recording length:
        # Conversational human speech has 25% - 60% voiced phonation.
        # Ambient noise, breathing, chair creaks, or fan harmonics produce either
        # sparse voicing (<6%) or low harmonicity.
        if audio.duration_seconds < 1.0:
            min_voiced = 12
            min_ratio = 0.0
        elif audio.duration_seconds < 3.0:
            min_voiced = 25
            min_ratio = 0.05
        elif audio.duration_seconds < 7.0:
            min_voiced = 45
            min_ratio = 0.06
        else:
            # Full live analysis windows (10s, 15s, 20s, 30s):
            # Human speech in a 15-30s window must contain at least 80 voiced frames (~800ms)
            # AND at least 6.0% voiced ratio. If a user did not talk, ambient noise produces
            # low voicing ratio (<6%) or erratic pitch.
            min_voiced = 80
            min_ratio = 0.06

        if voiced_frames < min_voiced:
            return (
                False,
                f"Insufficient vocal phonation ({voiced_frames} voiced frames in {audio.duration_seconds:.1f}s, minimum {min_voiced} required).",
                SPEECH_STATUS_VOICE_NOT_DETECTED,
            )

        if min_ratio > 0.0 and voiced_ratio < min_ratio:
            return (
                False,
                f"Voicing ratio too low ({voiced_ratio * 100:.1f}% < {min_ratio * 100:.1f}%), consistent with ambient room noise, mic hiss, or breathing.",
                SPEECH_STATUS_VOICE_NOT_DETECTED,
            )

        # Harmonics-to-Noise Ratio (HNR): Periodic human vocal cord phonation produces
        # clear harmonic structure (HNR >= 3.0 dB). Unvoiced noise, fan buzz, typing,
        # and room reverberation have negative or negligible HNR (< 3.0 dB).
        if np.isnan(hnr_mean) or hnr_mean < 3.0:
            return (
                False,
                f"Harmonics-to-noise ratio too low ({hnr_mean:.1f} dB < 3.0 dB). Signal is acoustic noise without periodic vocal cord phonation.",
                SPEECH_STATUS_VOICE_NOT_DETECTED,
            )

        # Combined check: low SNR + moderate HNR signals fan hum (periodic but no dynamic speech contrast).
        # Fan hum at 120 Hz has HNR 4-7 dB (periodic signal) but SNR < 2 dB (flat energy = no syllable contrast).
        # Real speech with HNR 4-7 dB would have SNR > 6 dB from syllable/pause alternation.
        if audio.duration_seconds >= 3.0 and audio.snr_estimate_db < 4.0 and not np.isnan(hnr_mean) and hnr_mean < 10.0:
            return (
                False,
                f"Periodic noise without speech dynamics (HNR={hnr_mean:.1f} dB, SNR={audio.snr_estimate_db:.1f} dB). "
                "Consistent with fan hum or electrical buzz, not conversational speech.",
                SPEECH_STATUS_VOICE_NOT_DETECTED,
            )

        # If voicing is sparse (<15% of the window), require high vocal harmonicity (>=7.5 dB)
        # to ensure the few detected frames represent actual vowels rather than broadband clicks/flutter.
        if audio.duration_seconds >= 3.0 and voiced_ratio < 0.15 and hnr_mean < 7.5:
            return (
                False,
                f"Marginal voicing ({voiced_ratio * 100:.1f}%) with low harmonicity ({hnr_mean:.1f} dB < 7.5 dB). Consistent with non-vocal ambient noise.",
                SPEECH_STATUS_VOICE_NOT_DETECTED,
            )

    except Exception as e:
        logger.warning("VAD pitch/harmonicity check failed, falling back to basic checks: %s", e)
        if audio.raw_rms < 0.008 and getattr(features, "pause_ratio", 0.0) >= 0.85:
            return (
                False,
                f"Excessive pause ({getattr(features, 'pause_ratio', 0.0)*100:.0f}%) with low energy.",
                SPEECH_STATUS_VOICE_NOT_DETECTED,
            )

    # 5. Low SNR with high background noise check
    hnr_mean_val = audio.vad_stats.get("hnr_mean") if getattr(audio, "vad_stats", None) else None
    if audio.snr_estimate_db < 2.0 and audio.raw_rms >= 0.02 and (hnr_mean_val is None or hnr_mean_val < 5.0):
        return (
            False,
            f"Excessive acoustic noise floor (estimated SNR: {audio.snr_estimate_db:.1f} dB). Noise overwhelms speech signal.",
            SPEECH_STATUS_AUDIO_QUALITY_TOO_LOW,
        )

    # 6. Feature-level speech activity validation (when extracted features are provided)
    if features is not None:
        pause_ratio = float(getattr(features, "pause_ratio", 0.0) or 0.0)
        feat_voiced_ratio = float(getattr(features, "voiced_ratio", 0.0) or 0.0)
        jitter = float(getattr(features, "jitter_local", 0.0) or 0.0)
        pitch_mean = float(getattr(features, "pitch_mean", 0.0) or 0.0)
        is_pitch_reliable = bool(getattr(features, "is_pitch_reliable", True))

        # Audio dominated by pauses (>90% silence without sustained voiced duration)
        voiced_duration = feat_voiced_ratio * audio.duration_seconds
        if (pause_ratio >= 0.88 and voiced_duration < 0.4) or (pause_ratio >= 0.95 and voiced_duration < 0.8):
            return (
                False,
                f"Audio is predominantly pause/silence ({pause_ratio * 100:.1f}% silent duration). No sustained speech detected.",
                SPEECH_STATUS_VOICE_NOT_DETECTED,
            )

        # Unreliable pitch or zero fundamental frequency
        if not is_pitch_reliable or pitch_mean <= 0.0:
            return (
                False,
                "Pitch tracking inconclusive; signal lacks stable vocal fundamental frequency.",
                SPEECH_STATUS_VOICE_NOT_DETECTED,
            )

        # Pathological / noise-induced jitter (>3.5%) with sparse voicing indicates noise tracking error
        if jitter >= 0.035 and feat_voiced_ratio < 0.20:
            return (
                False,
                f"Acoustic perturbation (jitter {jitter * 100:.1f}%) indicates non-vocal background noise rather than human speech.",
                SPEECH_STATUS_VOICE_NOT_DETECTED,
            )

    return (
        True,
        "Human voice activity detected.",
        SPEECH_STATUS_VOICE_DETECTED,
    )


def check_voice_presence(audio: LoadedAudio, features=None) -> Tuple[bool, str]:
    """
    Evaluate whether the audio contains genuine human vocal activity.
    Preserves 100% backward compatibility with existing callers.

    Returns:
        (has_speech: bool, reason: str)
    """
    has_speech, reason, _ = evaluate_audio_speech_quality(audio, features)
    return has_speech, reason


def save_upload_temp(file_bytes: bytes, filename: str, upload_dir: str) -> str:
    """
    Persist an uploaded file to disk under a random name, preserving the
    original extension. Returns the full path. Used because Parselmouth's
    Sound object is most reliably constructed from a file path.
    """
    os.makedirs(upload_dir, exist_ok=True)
    ext = os.path.splitext(filename)[1].lower() or ".wav"
    temp_name = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(upload_dir, temp_name)
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
    return temp_path


def write_wav(samples: np.ndarray, sample_rate: int, path: str) -> None:
    """Write a mono float32 signal to disk as 16-bit PCM WAV."""
    sf.write(path, samples, sample_rate, subtype="PCM_16")


def safe_remove(path: str) -> None:
    """Best-effort temp file cleanup that never raises."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError as e:
        logger.warning("Could not remove temp file %s: %s", path, e)
