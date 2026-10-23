"""
live_audio.py

Microphone capture layer for the TRUSIGHT Live Analysis feature.

Responsibilities
----------------
- Open the system default input device via sounddevice.
- Record a fixed-duration window of audio (15 / 20 / 30 seconds).
- Return a LoadedAudio object that is immediately consumable by the
  existing FeatureExtractor — same dataclass contract, same 16 kHz mono
  float32 representation.

This module is intentionally decoupled from all ML logic. It has no
knowledge of feature extraction, prediction, or the FastAPI application.

Dependencies
------------
sounddevice  – PortAudio wrapper (audio capture)
numpy        – signal arithmetic (already in requirements)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from app.utils import LoadedAudio, _normalize

logger = logging.getLogger("voice_service.live_audio")

# The whole pipeline is tuned for 16 kHz mono speech.
SAMPLE_RATE: int = 16_000

# Allowed recording window lengths (seconds).
VALID_WINDOWS: tuple[int, ...] = (15, 20, 30)
DEFAULT_WINDOW: int = 30


class MicrophoneError(RuntimeError):
    """Raised when the microphone cannot be opened or read."""


@dataclass
class DeviceInfo:
    """Minimal metadata about an audio input device."""

    index: int
    name: str
    max_input_channels: int
    default_sample_rate: float


def list_input_devices() -> List[DeviceInfo]:
    """
    Return a list of available audio input devices.

    Raises MicrophoneError if sounddevice / PortAudio is unavailable.
    """
    try:
        import sounddevice as sd  # local import so FastAPI starts even without PortAudio
    except OSError as exc:
        raise MicrophoneError(
            "PortAudio library not found. Cannot enumerate audio devices."
        ) from exc

    devices: List[DeviceInfo] = []
    for idx, info in enumerate(sd.query_devices()):
        if info["max_input_channels"] > 0:  # type: ignore[index]
            devices.append(
                DeviceInfo(
                    index=idx,
                    name=info["name"],  # type: ignore[index]
                    max_input_channels=info["max_input_channels"],  # type: ignore[index]
                    default_sample_rate=info["default_samplerate"],  # type: ignore[index]
                )
            )
    return devices


def record_window(
    duration_seconds: int = DEFAULT_WINDOW,
    device: Optional[int] = None,
    stop_event=None,
) -> LoadedAudio:
    """
    Synchronously record ``duration_seconds`` of audio from the microphone.

    Parameters
    ----------
    duration_seconds:
        Number of seconds to record. Must be one of VALID_WINDOWS.
    device:
        sounddevice device index, or None to use the system default.
    stop_event:
        Optional threading.Event. If set() before recording finishes the
        function returns whatever was captured so far (may be < duration).

    Returns
    -------
    LoadedAudio
        A fully preprocessed (16 kHz mono float32 peak-normalised) audio
        container ready for FeatureExtractor.extract().

    Raises
    ------
    MicrophoneError
        If PortAudio is missing, no input device is available, or
        permission is denied.
    ValueError
        If duration_seconds is not in VALID_WINDOWS.
    """
    if duration_seconds not in VALID_WINDOWS:
        raise ValueError(
            f"Invalid recording window: {duration_seconds}s. "
            f"Must be one of {VALID_WINDOWS}."
        )

    try:
        import sounddevice as sd
    except OSError as exc:
        raise MicrophoneError(
            "PortAudio library not found. Please install PortAudio or run on "
            "a machine with a working audio subsystem."
        ) from exc

    n_samples = int(duration_seconds * SAMPLE_RATE)

    logger.info(
        "Starting microphone recording: %ds (%d samples @ %d Hz)",
        duration_seconds,
        n_samples,
        SAMPLE_RATE,
    )

    try:
        # sd.rec() blocks until the full buffer is captured (or interrupted).
        # Channels=1 forces mono; dtype=float32 matches our pipeline.
        recorded: np.ndarray = sd.rec(
            frames=n_samples,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=device,
            blocking=False,  # we manage the wait ourselves for stop_event support
        )

        # Manually wait so we can honour stop_event between chunks.
        chunk = 0.25  # check every 250 ms
        elapsed = 0.0
        while elapsed < duration_seconds:
            if stop_event is not None and stop_event.is_set():
                logger.info("Recording interrupted by stop event at %.1fs", elapsed)
                sd.stop()
                break
            sd.sleep(int(chunk * 1000))
            elapsed += chunk
        else:
            sd.stop()

    except sd.PortAudioError as exc:
        raise MicrophoneError(
            f"Microphone error: {exc}. "
            "Check that a microphone is connected and that the application "
            "has permission to access it."
        ) from exc

    # recorded shape: (n_samples, 1) → flatten to (n_samples,)
    samples: np.ndarray = recorded[:, 0] if recorded.ndim == 2 else recorded

    # Trim any trailing zeros if recording was stopped early.
    # A zero-padded tail does not harm feature extraction but trimming is cleaner.
    nonzero = np.nonzero(samples)[0]
    if nonzero.size > 0:
        samples = samples[: nonzero[-1] + 1]

    samples = _normalize(samples.astype(np.float32))
    duration = float(len(samples)) / SAMPLE_RATE

    if duration < 0.5:
        raise MicrophoneError(
            "Captured audio is too short (< 0.5 s). "
            "The microphone may not be capturing any signal."
        )

    logger.info("Recording complete: %.2f seconds captured.", duration)

    return LoadedAudio(
        samples=samples,
        sample_rate=SAMPLE_RATE,
        duration_seconds=duration,
        source_filename=f"live_mic_{duration_seconds}s.wav",
    )
